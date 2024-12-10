import os
from shutil import rmtree
import glob

from speakerbox.speakerbox import preprocess, train, eval_model

os.environ["WANDB_DISABLED"] = "true"

# Need to clean up anything left-over from the last run...
rmtree('./chunked-audio')
dataset = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
    labeled_diarized_audio_dir=glob.glob("audio_samples/*")
)

dataset_dict, value_counts = preprocess.prepare_dataset(
    dataset,
    # good if you have large variation in number of data points for each label
    equalize_data_within_splits=True,
    # set seed to get a reproducible data split
    seed=60,
)

# You can print the value_counts dataframe to see how many audio clips of each label
# (speaker) are present in each data subset.

train(dataset_dict)

eval_model(dataset_dict["valid"])