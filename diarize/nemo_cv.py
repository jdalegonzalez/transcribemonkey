#! /usr/bin/env python3
# -*- coding: utf-8 -*-
""" This file is an experiment using the nemo diarization model to split an audio file
into two separate audio files, one for each speaker.  It then uses the ClearVoice library
to clean up the audio files and finally uses the Whisper model to transcribe the audio files.
"""
import argparse
import os
import logging
from typing import Any, Union

import soundfile as sf
import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel
import torch
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.auto.modeling_auto import AutoModelForSpeechSeq2Seq
from transformers.pipelines import pipeline

from clearvoice import ClearVoice

from utils import DEVICE, AudioType, Bcolors

# Expect the diarization postprocessing config file to be located in the same directory as this script.
postprocessing_config = os.path.join(os.path.dirname(__file__), "postprocess_config.yaml")

# load model from Hugging Face model card directly (You need a Hugging Face token)
diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
assert type(diar_model) is SortformerEncLabelModel, "Diarization model is not of type SortformerEncLabelModel"
diar_model = diar_model.eval()

device = DEVICE
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

voice_separator = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])

def longest_segment_per_speaker(segments):
    """Given a list of segments, return the longest segment for each speaker.
    
    Args:
        segments: A list of dictionaries, each dictionary has the keys 'start', 'end', and 'speaker_id'.
    Returns:
        A dictionary with keys 'speaker_0' and 'speaker_1', each value is the longest segment for that speaker.
    """
    longest_segments = {}
    for segment in segments:
        speaker_id = segment['speaker_id']
        if speaker_id not in longest_segments:
            longest_segments[speaker_id] = None
        if longest_segments[speaker_id] is None or (segment['end'] - segment['start']) > (longest_segments[speaker_id]['end'] - longest_segments[speaker_id]['start']):
            longest_segments[speaker_id] = segment

    return longest_segments

def process_segments(segments):
    """Converts the SpeakerXSegment matrix of strings into a list of dictionaries.  
    The format produced by nemo is a bit unhelpful since it's a matrix of single 
    strings, "start end speaker_id".  We're going to convert this into a list
    of dictionaries, where each dictionary has the keys 'start', 'end', and 'speaker_id'.
    We'll also sort the list by start time.

    Args:
        segments: A list of strings, each string is in the format "start end speaker_id".
    Returns:
        A list of dictionaries, each dictionary has the keys 'start', 'end', and 'speaker_id'.
        The list is sorted by start time.
    """
    processed = []
    for speaker_segments in segments:
        for seg in speaker_segments:
            start, end, speaker_id = seg.split()
            processed.append({
                'start': float(start),
                'end': float(end),
                'speaker_id': speaker_id
            })
    # Sort the list by start time
    return sorted(processed, key=lambda x: x['start'])

def find_overlapping_segments(segments) ->list[tuple[dict, dict]]:
    """Given a list of segment dicts, sorted by start time, find overlapping segments.
    
    Args:
        segments: A list of dictionaries, each dictionary has the keys 'start', 'end', and 'speaker_id'.
    Returns:
        A list of tuples, each tuple contains two dictionaries representing overlapping segments.
    """
    
    overlapping_segments = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            seg1 = segments[i]
            seg2 = segments[j]
            if (seg1['start'] < seg2['end'] and seg1['end'] > seg2['start']):
                overlapping_segments.append((seg1, seg2))
    return overlapping_segments

def print_timeline(segments):
    """Prints an ascii graphic timeline of speaker start/stop times."""

    e = f'{Bcolors.ENDC}'
    b = Bcolors.BOLD
    ft = f'{b}{Bcolors.WHITE}'
    dots_per_second = 4  # Each second has 4 dots, one for each quarter of a second

    # The max of the last two segments' end times is the end of the timeline
    max_end = max(segments[-1]['end'], segments[-2]['end']) if len(segments) > 1 else segments[-1]['end']
    
    # We're going use a pipe to mark a segment and a dot for each portion of a second
    header = '0'
    h1 = ' '
    q = 0
    for i in range(int(max_end)):
        t = (i + 1) % 10
        if t == 0: q += 1
        h1     += (' ' * (dots_per_second - 1) + f'{q if q else ' '}')
        header += ('.' * (dots_per_second - 1) + f'{t}')

    # Add the remaining '...' for the last fraction of a second
    if max_end % 1 != 0:
        header += ('.' * int((max_end % 1) * dots_per_second))
    h1     = "    " + h1
    header = "    " + header
    last_speaker_1_end = 0
    last_speaker_2_end = 0
    s1 = ''
    s2 = ''

    def string_it(start_delta, speaker_duration, last_end) -> str:
        offset = -1 if last_end > 0 else 0
        pad = ' ' * (round(start_delta * dots_per_second) + offset)
        dur_str = '─' * (round(speaker_duration * dots_per_second) + 1) # Each quarter of a second is one character
        if len(dur_str) > 1:
            dur_str = '├' + dur_str[1:-1] + '┤'
        else:
            dur_str = '║'
        return pad + dur_str
    
    for segment in segments:
        start = segment['start']
        end = segment['end']
        speaker_id = segment['speaker_id']

        # Print spaces for each quarter of a second from the last end to the start of this segment
        if speaker_id == 'speaker_0':
            s1 += string_it(start - last_speaker_1_end, end - start, last_speaker_1_end)
            last_speaker_1_end = end
        else:
            s2 += string_it(start - last_speaker_2_end, end - start, last_speaker_2_end)
            last_speaker_2_end = end

    print()
    print(h1)
    print(f"{ft}{header}{e}")
    print(f"{Bcolors.BOLD + Bcolors.CYAN}S1: {s1}{e}")
    print(f"{Bcolors.BOLD + Bcolors.MAGENTA}S2: {s2}{e}")
    for segment in segments:
        start = segment['start']
        end = segment['end']
        speaker_id = segment['speaker_id']
        print(f"{Bcolors.BOLD + Bcolors.OKGREEN}{speaker_id} {start:.2f} - {end:.2f}{e}")

def split_audio(audio_input: Union[str, AudioType], segments: list[dict], dtype = "float32") -> tuple[dict[str, np.ndarray], int]:
    """ Creates two audio files from the original files, one for each speaker.
    Given the original audio file and the segment dict, create two audio files of
    equal length.  One with just the first speaker's audio and space for the second speaker,
    and one with just the second speaker's audio and space for the first speaker.
    Args:
        audio_input: The path to the original audio file.
        segments: A list of dictionaries, each with keys 'start', 'end', and 'speaker_id'.
    Returns:
        Two audio files, one for each speaker, and sample_rate.
    """

    if isinstance(audio_input, tuple):
        # If audio_input is already an AudioType, extract the audio and sample_rate
        audio, sample_rate = audio_input
    else:
        # Load the original audio file
        audio, sample_rate = sf.read(audio_input, dtype=dtype)

    # Create empty arrays for the two speakers
    result = {}

    # Calculate the start and end indices for the segment
    for segment in segments:
        if 'speaker_id' not in segment or 'start' not in segment or 'end' not in segment:
            logging.warning("Segment is missing required keys: 'speaker_id', 'start', or 'end'. Skipping segment.")
            continue

        id = segment['speaker_id']
        if id not in result:
            result[id] = np.zeros_like(audio)

        start_index = int(segment['start'] * sample_rate)
        end_index = int(segment['end'] * sample_rate)

        result[id][start_index:end_index] = audio[start_index:end_index]
    
    # Return the audio files and sample_rate as a tuple
    # Note: The audio files are numpy arrays, not file paths.
    return (result, sample_rate)

def clean_audio(input_path: str, write_file: bool = False, as_type:Any = np.float32, speaker_ndx: int = 0) -> np.ndarray:
    """Uses ClearVoice to clean the audio file.
    
    Args:
        input_path: The path to the audio file to clean.
    Returns:
        The cleaned audio as a numpy array.
    """
    output_wav = voice_separator(input_path=input_path, online_write=False)
    if write_file: voice_separator.write(output_wav, output_path=input_path)
    assert output_wav is not None, "ClearVoice did not return any output."

    return output_wav[speaker_ndx][0].astype(as_type) if output_wav[speaker_ndx][0].dtype != as_type else output_wav[speaker_ndx][0]

def do_transcription(audio1: np.ndarray, audio2: np.ndarray, sample_rate: int):
    """Takes the two audio files and produces a transcription for each speaker."""

    # Prompt will get used once I move to stable-whisper.
    prompt = 'You are a linguist listenging a bilingual podcast in English and Mandarin Chinese.  The hosts are Tom and Ula and they both speak both Mandarin and Chinese in the same sentance.'

    # TODO: Move all this code to stable-whisper so that I can use initial_prompt, 
    # my better model, etc...
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    gen__kwargs = {
        "task": "transcribe",
    }

    # These are going to have to be merged back together using the diarization segments.
    result1 = pipe(audio1, return_timestamps=True, generate_kwargs=gen__kwargs)
    result2 = pipe(audio2, return_timestamps=True, generate_kwargs=gen__kwargs)

    return result1['chunks'], result2['chunks']  # type: ignore - type is wrong

def merge_audio_and_diarization(diary: list[dict], transcription1: list[dict], transcription2: list[dict]):
    """Merges the diarization segments with the transcriptions for each speaker.
    
    Args:
        diary: A list of dictionaries, each with keys 'start', 'end', and 'speaker_id'.
        transcription1: A list of dictionaries, each with keys 'start', 'end', and 'text' for speaker 1.
        transcription2: A list of dictionaries, each with keys 'start', 'end', and 'text' for speaker 2.
    Returns:
        A merged list of dictionaries with keys 'start', 'end', 'speaker_id', and 'text'.
    """
    # TODO: I haven't run/tested this copilot generated code yet.
    merged = []
    for segment in diary:
        if segment['speaker_id'] == 'speaker_0':
            text = next((t['text'] for t in transcription1 if t['start'] == segment['start'] and t['end'] == segment['end']), '')
        else:
            text = next((t['text'] for t in transcription2 if t['start'] == segment['start'] and t['end'] == segment['end']), '')
        
        merged.append({
            'start': segment['start'],
            'end': segment['end'],
            'speaker_id': segment['speaker_id'],
            'text': text
        })
    
    return merged

def diarize_audio(audio_input: str):
    """Runs the diarization model on the audio file and returns the segments.
    
    Args:
        audio_input: The path to the audio file to process.
    Returns:
        A list of segments, each segment is a list of strings in the format "start end speaker_id".
    """
    # The bummer here is that the diarize function can only take a string or 
    # list of strings.  The string can either be a path to an audio file or the
    # path to a jsonl manifest file.  This means we'll have to create copies
    # of the audio file on disk.
    assert isinstance(diar_model, SortformerEncLabelModel), "Diarization model is not of type SortformerEncLabelModel"
    predicted_segments =  diar_model.diarize(
        audio=audio_input,
        batch_size=1,
        postprocessing_yaml=postprocessing_config
    )
    # Process the segments
    segments = process_segments(predicted_segments)
    return segments

def main(audio_input):
    """Main function to run the diarization model and process the audio file."""

    # The bummer here is that the diarize function can only take a string or 
    # list of strings.  The string can either be a path to an audio file or the
    # path to a jsonl manifest file.  This means we'll have to create copies
    # of the audio file on disk.
    predicted_segments = diar_model.diarize(audio=audio_input, batch_size=1)  # type: ignore - diar_model type is wrong

    # Process the segments
    segments = process_segments(predicted_segments)

    # Split the audio into separate files for each speaker
    speakers_dict, sample_rate = split_audio(audio_input, segments)
    path_part = os.path.dirname(audio_input)
    file_part = os.path.basename(audio_input)
    for speaker_id, audio in speakers_dict.items():
        sf_file = os.path.join(path_part, f"{speaker_id}_{file_part}")
        sf.write(sf_file, audio, sample_rate)


    # For now, the first speaker will be speaker_0, the second will be speaker_1 and
    # all others will be ignored.
    if len(speakers_dict) < 2:
        logging.error("Not enough speakers found in the audio file. At least two speakers are required.")
        return
    keys = list(speakers_dict.keys())

    s1 = keys[0]  # First speaker
    s2 = keys[1]  # Second speaker
    s1_file = os.path.join(path_part, f"{s1}_{file_part}")
    s2_file = os.path.join(path_part, f"{s2}_{file_part}")

    # Now, take those files and run them through ClearVoice to clean up any overlapping bits.
    cv = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
    #cv = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
    #cv = ClearVoice(task='speech_super_resolution', model_names=['MossFormer2_SR_48K'])

    # xx_output_wav is the wav data for 2 speakers, but since we know that
    # we only gave it one, one of the outputs will be mostly silence and the
    # other will be the cleaned up audio.  I think we can always go with 
    # the "0" element.

    s1_output_wav = cv(input_path=s1_file, online_write=False)
    assert s1_output_wav is not None, "ClearVoice did not return any output for speaker 1."
    s1_clean_audio = s1_output_wav[0][0]  # Get the cleaned audio for speaker 1
    s1_cleaned_file = os.path.join(path_part, f"cleaned_{file_part}_s1.wav")
    #cv.write(s1_output_wav, output_path=s1_cleaned_file)
    sf.write(s1_cleaned_file, s1_clean_audio, sample_rate) # This can come out once I'm happy with the transcription.

    s2_output_wav = cv(input_path=s2_file, online_write=False)
    assert s2_output_wav is not None, "ClearVoice did not return any output for speaker 1."
    s2_clean_audio = s2_output_wav[0][0]  # Get the cleaned audio for speaker 2
    s2_cleaned_file = os.path.join(path_part, f"cleaned_{file_part}_s2.wav")
    #cv.write(s2_output_wav, output_path=s2_cleaned_file)
    sf.write(s2_cleaned_file, s2_clean_audio, sample_rate)# This can come out once I'm happy with the transcription.


if __name__ == "__main__":
    """ No need for tons of args.  We'll just take the path to the audio file. """
    parser = argparse.ArgumentParser(description="Run diarization and transcription on an audio file.")
    parser.add_argument(
        "audio_input",
        type=str,
        help="Path to the audio file to process."
    )
    args = parser.parse_args() 
    # Replace 'path_to_your_audio_file.wav' with the path to your audio file
    main(args.audio_input)