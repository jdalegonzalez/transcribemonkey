# Load the CS-Dialogue dataset from huggingface datasets
import os
import typing
from datasets import load_dataset, DatasetDict, Audio, IterableDatasetDict
import torch
import evaluate
from jiwer import wer, cer
from peft import LoraConfig, get_peft_model

from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer
)

from transformers.pytorch_utils import Conv1D
from transformers.feature_extraction_utils import BatchFeature
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.trainer_seq2seq import Seq2SeqTrainer

from dataclasses import dataclass
from typing import List, Union
from utils import split_sentence

# Just defined to make type checkers happy
class MyWhisperProcessor(WhisperProcessor):
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizer
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: MyWhisperProcessor

    def __call__(self, features: List[BatchFeature]) -> BatchFeature:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        tn = typing.cast(torch.Tensor, labels_batch["input_ids"])
        labels = tn.masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item(): # type: ignore
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

model_name = "openai/whisper-large-v3"
processor = typing.cast(MyWhisperProcessor, WhisperProcessor.from_pretrained(model_name, task='transcribe'))

assert isinstance(processor, WhisperProcessor), "Expected a WhisperProcessor instance"
model = WhisperForConditionalGeneration.from_pretrained(model_name)
print(model)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False # gradient checkpointing is used, so we cannot use the cache
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
)

def get_specific_layer_names(model):
    # Create a list to store the layer names
    layer_names = []
    
    # Recursively visit all modules and submodules
    for name, module in model.named_modules():
        # Check if the module is an instance of the specified layers
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
            # model name parsing 

            layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
    
    return layer_names


model.generation_config.forced_decoder_ids = None
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

chinese_metric = evaluate.load("cer")
english_metric = evaluate.load("wer")

def load_cs_dialogue_dataset() -> Union[DatasetDict,IterableDatasetDict]:
    # If the dataset is cached, we'll load it from there.  Otherwise, we'll
    # load it from the base dir, perform the necessary transformations,
    # and cache it for future use.

    # Temporarily disable caching for debugging purposes
    if os.path.exists("/mnt/d/datasets/hf-cs-dialogue/cache"):
        print("Loading cached CS-Dialogue dataset...")
        t = load_dataset("/mnt/d/datasets/hf-cs-dialogue/cache", streaming=True)
        assert isinstance(t, DatasetDict) or isinstance(t, IterableDatasetDict), "Expected a DatasetDict or iterable"
        return t
    
    t = load_dataset("audiofolder", data_dir="/mnt/d/datasets/hf-cs-dialogue/data")
    assert isinstance(t, DatasetDict), "Expected a DatasetDict"
    t = t.cast_column("audio", Audio(sampling_rate=16000))
    t = t.map(prepare_dataset)
    print("Dataset prepared successfully!")
    # Cache the dataset for future use
    t.save_to_disk("/mnt/d/datasets/hf-cs-dialogue/cache")
    print("CS-Dialogue dataset cached successfully!")
    return t

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor( # type: ignore
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0] 

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer( # type: ignore
        batch["language"] + batch["sentence"]
    ).input_ids

    return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id 

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute mixed error rate (MER)
    mer = mixed_error_rate(predictions=pred_str, references=label_str)

    return {"mer": mer}

def mixed_error_rate(*, predictions, references) -> float:
    """
    Compute the mixed error rate (MER) for a list of predictions and references.
    """
    # Compute WER and CER
    preds = [predictions] if isinstance(predictions, str) else predictions
    refs = [references] if isinstance(references, str) else references

    mers = []
    for i, pred in enumerate(preds):
        # We don't care about errors in punctuation.
        pred_eng, peng_count, pred_chi, pchi_count = split_sentence(pred, drop_punc=True)
        ref_eng, reng_count, ref_chi, rchi_count = split_sentence(refs[i], drop_punc=True)
        word_error      = wer(ref_eng, pred_eng) if reng_count else peng_count + 0.0
        character_error = cer(ref_chi, pred_chi) if rchi_count else pchi_count + 0.0
        assert type(character_error) is float, f"Expected character_error to be a float, got {type(character_error)}"
        wer_count = peng_count + reng_count
        weighted_wer = word_error * wer_count
        cer_count = pchi_count + rchi_count
        weighted_cer = character_error * cer_count
        mixed_error = (weighted_wer + weighted_cer) / (wer_count + cer_count) if (wer_count + cer_count) > 0 else 0.0
        mers.append(mixed_error)

    # I haven't figured out how to do weighted average across multiple
    # prediction/reference pairs.  This average will do "something" but
    # I have no idea if that's really what we want.  BUT, I only every
    # use one prediction/reference pair at a time, so this is fine for now.
    return sum(mers)/len(mers) 

if __name__ == "__main__":

    print("Trying to load the CS-Dialogue dataset...")
    dataset = load_cs_dialogue_dataset()
    print("Dataset loaded successfully!")
    print(dataset["test"])  # Print the first example in the train set
    print(dataset["train"])  # Print the first example in the train set

    if hasattr(model, "enable_input_require_grads"):
        print("Enabling input gradients for the model...")
        model.enable_input_require_grads()
    else:
        print("Enabling input gradients for the model using forward hook...")
        def make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # Set the random seed for the dataset
    dataset = dataset.shuffle(seed=42)  # Shuffle the dataset with the seed
    print("Dataset shuffled successfully!")

    training_args = Seq2SeqTrainingArguments(
        output_dir="~/projects/transcribemonkey/whisper-cs-dialogue",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        save_strategy="best",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=None,
        load_best_model_at_end=True,
        metric_for_best_model="mer",
        greater_is_better=False,
        push_to_hub=False,
        #deepspeed="./ds_config.json",  # path to your deepspeed config file
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=peft_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"], # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.tokenizer,
    )
    trainer.train()

    peft_model.save_pretrained("whisper-cs-dialogue")