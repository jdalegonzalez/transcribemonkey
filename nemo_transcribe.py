#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""This file is an experiment using the nemo diarization model to split an audio file
into two separate audio files, one for each speaker.  It then uses the ClearVoice library
to clean up the audio files and finally uses the Whisper model to transcribe the audio files.
"""
import argparse
import logging
import os

import numpy as np

from diarize.nemo_cv import diarize_audio, longest_segment_per_speaker, split_audio

import stable_whisper
from stable_whisper import WhisperResult
from hanziconv import HanziConv

from utils import (
    AudioType,
    audio_from_file,
    space_english_and_chinese,
    Bcolors
)

from subsegment import SubSegment
from speakers import guess_speaker, SpeakerGuess

from nemo.collections.audio.models import FlowMatchingAudioToAudioModel

# NOTE: To get this code to work, I had to manually "fix"/bodge the whisper/triton_ops.py file
# that got installed along with stable_whisper.
# See: https://github.com/openai/whisper/discussions/2597
# It's possible that I could try the other recommended fix:
#    pip install -U openai-whisper
#    pip install triton==3.2.0

# These are all the model config values that we'll use across all the experiments.
prompt = (
    'You are a transciber listening a podcast in English and Mandarin Chinese called "Mandarin Monkey". '
    'The speakers are Tom and Ula.  They both speak Mandarin and English and code-switch between both languages. '
    'Accurately transcribe the English parts as English and the Mandarin parts as Mandarin. '
    'Use simplified Chinese characters for the Mandarin parts. 好不好? '
    '**NEVER Translate**. Only transcribe. '
    'Include laughs, ums, ahs, and other non-verbal sounds in the transcription. '
)

DEFAULT_TRANSCRIBE_KWARGS = {
    'language': 'en',
    'task': 'transcribe',
    'beam_size': 5,
    'best_of': 5,
    'initial_prompt': prompt,
    'suppress_blank': False,
    'condition_on_previous_text': False,
    'suppress_tokens': [-1],
    'without_timestamps': True,
    'word_timestamps': True
}

vad = True  # Voice Activity Detection, set to True to enable VAD
language='en'
whisper_model = "large-v3"  # The Whisper model to use for transcription

def assign_speaker_names(audio: AudioType, segments: list[dict]) -> list[dict]:
    """Assign speaker names to the segments by using the audio classification model on the longest segment for each speaker."""
    longest_segments = longest_segment_per_speaker(segments)
    speaker_names = {}
    for speaker_id, segment in longest_segments.items():
        audio_start:float = segment['start']
        audio_end:float = segment['end']
        if audio_start is None or audio_end is None:
            logging.warning(f"Skipping segment {segment} due to missing start or end time.")
            continue
        guess = guess_speaker(SubSegment.slice_audio(audio, audio_start, audio_end))
        if guess != SpeakerGuess.NO_GUESS:
            speaker_names[speaker_id] = guess['label']
        else:
            speaker_names[speaker_id] = speaker_id  # Fallback to speaker_id if no guess is made

    # Map the speaker names to the segments
    new_segments = [
        {
            **segment,
            "speaker_id": speaker_names.get(segment["speaker_id"], segment["speaker_id"])
        } for segment in segments
    ]

    return new_segments


def dump_result(result: WhisperResult):
    """Dump the result of the transcription to the console."""
    for segment in result.segments:
        print(segment)

def transcribe3(audio: AudioType, segments: list[dict]) -> list[dict]:
    """ For this experiment, we're going to call transcribe over and over, once for each audio segment. """
    model = stable_whisper.load_model(whisper_model)
    for segment in segments:
        # TODO: We may want to look at ignoring really short segments, like less than 1 second.
        audio_start = segment.get('start', 0)
        audio_end = segment.get('end', 0)
        if audio_start is None or audio_end is None:
            logging.warning(f"Skipping segment {segment} due to missing start or end time.")
            continue
        sub_audio, _ = SubSegment.slice_audio(audio, audio_start, audio_end)
        result = model.transcribe(
            sub_audio,
            vad=vad,
            **DEFAULT_TRANSCRIBE_KWARGS
        )
        assert isinstance(result, WhisperResult), "Transcription result is not of type WhisperResult"
        segment['text'] = space_english_and_chinese(result.text)

    return segments

def transcribe4(audio: AudioType, segments: list[dict], temp_dir: str) -> list[dict]:
    """ For this experiment, we're going to use separated audio files (potentially cleaned up by ClearVoice) for each speaker."""
    """ AND we're going to pad then end of each segment, which will hopefully reduce hallucinations."""
    model = stable_whisper.load_model(whisper_model)
    
    # Grab the audio and split it into multiple files, one for each speaker.
    speakers_dict, sample_rate = split_audio(audio, segments)

    # Right now, ClearVoice doesn't make anything better AND it's very
    # slow.  So, we're going to skip it for now.
    # ClearVoice has to have a file, it won't take a numpy array. So, 
    # we need to write the audio data to files.
    for speaker_id, audio_data in speakers_dict.items():
        if audio_data is None:
            logging.warning(f"Skipping speaker {speaker_id} due to missing audio data.")
            continue

        # Clean the audio data using ClearVoice.
        import soundfile as sf
        speaker_file_path = os.path.join(temp_dir, f"speaker_{speaker_id}.wav")
        #sf.write(speaker_file_path, audio_data, sample_rate)
        #speakers_dict[speaker_id] = clean_audio(speaker_file_path, write_file=True)

    for segment in segments:
        # TODO: We may want to look at ignoring really short segments, like less than 1 second.
        audio_start = segment.get('start', 0)
        audio_end = segment.get('end', 0)
        if audio_start is None or audio_end is None:
            logging.warning(f"Skipping segment {segment} due to missing start or end time.")
            continue
        audio_data = speakers_dict.get(segment['speaker_id'])
        if audio_data is None:
            logging.warning(f"Skipping segment {segment} due to missing audio file for speaker {segment['speaker_id']}.")
            continue
        pad = 0.5  # seconds
        audio_end = min(audio_end + pad, audio_data.shape[0] / sample_rate)
        sub_audio, _ = SubSegment.slice_audio((audio_data, sample_rate), audio_start, audio_end)
        
        # For now, write this sub_audio to a file for debugging purposes.
        sub_audio_file = os.path.join(temp_dir, f"sub_audio_{segment['speaker_id']}_{audio_start:.2f}_{audio_end:.2f}.wav")
        import soundfile as sf
        sf.write(sub_audio_file, sub_audio, sample_rate)

        kwargs = dict(DEFAULT_TRANSCRIBE_KWARGS)
        kwargs['language'] = None
        result = model.transcribe(
            sub_audio,
            vad=vad,
            **kwargs
        )
        assert isinstance(result, WhisperResult), "Transcription result is not of type WhisperResult"
        segment['text'] = space_english_and_chinese(HanziConv.toSimplified(result.text.strip()))

    # Kill segments with no text.
    segments = [seg for seg in segments if seg.get('text', '').strip()]

    return segments

def experiment1(audio_input: str, temp_dir: str):
    """We're going to see if clearvoice does a better job separating the speakers if it's been primed with full overlap.
    
    I've commented out the priming with the overlap code.  I've now got a fine-tuned version of the 
    MossFormer2_SE_16K model that has been trained on segments that have both overlapping and non-overlapping
    speech.

    - Without the priming, it didn't separate speech that WASN'T overlapping.

    - Next, we'll use the diarization to build wav file, one for each speaker.  Then, 
      we'll drop the priming audio into the front of the file and we'll see if what remains
      is clean, single speaker throughout.

    """
    import soundfile as sf
    from diarize.nemo_cv import clean_audio

    # Grab the overlapping sample, put it on the front of the sample to split and save it in a file
    overlap_sample = os.path.join("diarize/samples", "overlap_sample.wav")
    if not os.path.exists(overlap_sample):
        raise FileNotFoundError(f"Overlap sample not found: {overlap_sample}")
    # Create a temporary wav file that has the audio input and the overapping sample concatenated.

    segments = diarize_audio(audio_input)
    from diarize.nemo_cv import print_timeline
    print_timeline(segments)
    audio: AudioType = audio_from_file(audio_input)
    speakers_dict, sample_rate = split_audio(audio, segments)
    overlap_waveform, _ = audio_from_file(overlap_sample)
    for speaker_id, audio_data in speakers_dict.items():
        temp_audio_file = os.path.join(temp_dir, f"speaker_{speaker_id}.wav")            
        combined_waveform = np.concatenate((overlap_waveform, audio_data), axis=0)
        sf.write(temp_audio_file, combined_waveform, sample_rate)
        clean_audio(temp_audio_file, write_file=True)

def experiment2(audio_input: str, temp_dir: str):
    """ FlowMatchingAudioToAudioModel in Nemo is supposed to clean a degraded audio signal.
    
    While this probably is working, we're getting sufficiently clean signal already that
    adding this step doesn't really add much value.
    """
    from nemo.collections.audio.models import AudioToAudioModel
    model = AudioToAudioModel.from_pretrained('nvidia/sr_ssl_flowmatching_16k_430m')
    assert type(model) is  FlowMatchingAudioToAudioModel, "Diarization model is not of type AudioToAudioModel"
    model = model.eval()
    t = model.process(
        paths2audio_files=[audio_input],
        output_dir=temp_dir,
        batch_size=1,
        num_workers=1
    )

def clean_overlap(audio:AudioType, primer: AudioType, temp_dir: str, overlapping_segments: list[tuple[dict, dict]], speaker_id:str, ndx: int) -> AudioType:
    """ Clean the overlapping segments using ClearVoice.
    
    We're going to find just the portion that overlaps, add the primer to the front of it,
    then clean it up using ClearVoice, then splice the cleaned audio back into the
    original audio file.

    Args:
        audio (AudioType): The audio file to process.
        primer (AudioType): The primer audio file to use for cleaning.
        temp_dir (str): Temporary directory to store intermediate files.
        overlapping_segments (list[tuple[dict, dict]]): List of overlapping segments.
        speaker_id (str): The speaker whose audio we're cleaning.
    Returns:
        AudioType: The cleaned audio file. 
    """

    import soundfile as sf
    from diarize.nemo_cv import clean_audio

    audio_waveform, sample_rate = audio
    audio_seconds = len(audio_waveform) / sample_rate
    primer_waveform, _ = primer

    # Put 1 second of silence between the primer and the overlap audio.
    silence_duration = 1  # seconds
    primer_waveform = np.pad(primer_waveform, (0, int(silence_duration * sample_rate)), mode='constant')

    buffer_end = 2
    buffer_start = 1
    for i, overlap_pairs in enumerate(overlapping_segments):
        segment1, segment2 = overlap_pairs
        start1 = segment1.get('start', 0)
        end1 = segment1.get('end', 0)
        start2 = segment2.get('start', 0)
        end2 = segment2.get('end', 0)

        overlap_start = min(0,max(start1, start2) - buffer_start) # Start 1 second before the overlap starts
        overlap_end = min(end1, end2) + 1  # End 1 second after the overlap ends
        # Get the overlapping portion of the audio.
        overlap_start = max(start1, start2)
        overlap_end = min(audio_seconds, min(end1, end2) + buffer_end)

        # Create a temp file that is primer + section of audio that overlaps.
        overlap_audio, _ = SubSegment.slice_audio(audio, overlap_start, overlap_end)
        overlap_filename = os.path.join(temp_dir, f"extracted_{speaker_id}_{i}.wav")
        sf.write(overlap_filename, overlap_audio, sample_rate)
        print(f"\nExtract: {overlap_filename} {overlap_start:.2f}:{overlap_end:.2f} {len(overlap_audio)} samples")

        # Combine the primer, silence, and overlap audio.
        combined_waveform = np.concatenate((primer_waveform, overlap_audio), axis=0)
        print(f"p: {primer_waveform.shape}, o: {overlap_audio.shape}, c: {combined_waveform.shape}")
        print(f"p: {primer_waveform.dtype}, o: {overlap_audio.dtype}, c: {combined_waveform.dtype}")
        temp_audio_file = os.path.join(temp_dir, f"overlap_dirty_{speaker_id}_{i}.wav") 
        sf.write(temp_audio_file, combined_waveform, sample_rate)        

        # Clean this audio file using ClearVoice.
        # HUGE assumption here.  It seems like Ula is always the second speaker.  Maybe that's true, 
        # maybe it's not.  Maybe it's a function of primer audio.  I will probably need to experiment and
        # determine how to "pick" the right speaker.  Maybe get both 1 and 2 and then ID match?
        cleaned_audio = clean_audio(temp_audio_file, write_file=True, as_type=audio_waveform.dtype, speaker_ndx=ndx)
        print(f"ca: {cleaned_audio.shape}, {cleaned_audio.dtype}")
        temp_audio_file = os.path.join(temp_dir, f"overlap_clean_{speaker_id}_{i}.wav") 
        sf.write(temp_audio_file, cleaned_audio, sample_rate)

        # Grab just the part of the audio that was in the original audio and splice it back in.
        # The start should be length of the primer + silence
        splice_start = len(primer_waveform)
        splice_end = len(combined_waveform)

        piece = cleaned_audio[splice_start:splice_end]
        temp_audio_file = os.path.join(temp_dir, f"splice_{speaker_id}_{i}.wav")
        sf.write(temp_audio_file, piece, sample_rate)
        print(f"Splice: {temp_audio_file} {splice_start:.2f}:{splice_end:.2f} {len(piece)} samples")

        # Splice this back into the original audio.
        print("start, end, conv", overlap_start, overlap_end,(int(overlap_end * sample_rate) - int(overlap_start * sample_rate)), "splice len", len(piece))
        audio = SubSegment.splice_audio(audio, overlap_start, (piece, sample_rate))

    # Now, we need to write the cleaned audio back to a file.
    cleaned_audio_file = os.path.join(temp_dir, f"cleaned_overlap_{speaker_id}.wav")
    # I don't _think_ this is necessary because the copies all happened in place but
    # just in case...
    audio_waveform, sample_rate = audio
    sf.write(cleaned_audio_file, audio_waveform, sample_rate)

    return audio

def experiment3(audio_input: str, temp_dir: str):
    """ We're going to try to JUST pull the segments that appear to have overlapping speech.
    The idea is that we can grab the overlapping speech segments, prime the clearvoice model
    with a known overlapping sample, split the audio into two sources, then remerge the 
    supposedly clean audio segments back into each "single speaker" audio file.

    The audio input is expected to be the speech of just one of the speakers.

    Args:
        audio_input (str): Path to the audio file to process.
        temp_dir (str): Temporary directory to store intermediate files.
    """

    # TODO: Verify that the cleaned chunk is the piece I want and that it's going in the right place.
    #       On-going.  It's closer.  There were bugs.
    # TODO: Try using sepformer from speechbrain instead of ClearVoice.
    # TODO: Try the longer overlap sample. (this is experiment 1 but on the mixed audio, not the split audio.)
    # TODO: REMEMBER you reverted to finetuned MossFormer2_SE_16K_430m model so maybe re-check default model


    from diarize.nemo_cv import find_overlapping_segments
    import soundfile as sf
    dtype = 'float64'
    segments = diarize_audio(audio_input)
    overlapping_segments = find_overlapping_segments(segments)
    speakers_dict, sample_rate = split_audio(audio_input, segments, dtype=dtype)
    primer_file = os.path.join("diarize/samples", "overlap_sample.wav")
    primer = sf.read(primer_file, dtype=dtype)

    if overlapping_segments:
        for speaker_id, audio_data in speakers_dict.items():
            single_speaker_filepath = os.path.join(temp_dir, f"single_speaker_{speaker_id}.wav")
            sf.write(single_speaker_filepath, audio_data, sample_rate)
            # This is a weak way to get the index (and needing an index is weak in and of itself)
            # but it's good enough for an experiment.
            speaker_ndx = int(speaker_id.replace('speaker_', ''))
            clean_overlap((audio_data, sample_rate), primer, temp_dir, overlapping_segments, speaker_id, speaker_ndx)

def print_segment(segment: dict):

    e = f'{Bcolors.ENDC}'
    b = Bcolors.BOLD
    ft = f'{b}{Bcolors.WHITE}'
    fs = f'{Bcolors.OKGREEN}'
    fe = f'{Bcolors.WARNING}{Bcolors.ITALIC}'

    start = segment.get('start', 0)
    end = segment.get('end', 0)
    speaker = segment.get('speaker_id', 'Unknown').capitalize()
    text = segment.get('text', '')

    fp = Bcolors.BOLD + (Bcolors.CYAN if speaker == 'Tom' else Bcolors.MAGENTA)
    print(f'{fs}{start} - {end}{e} {fp}{speaker}{e}: {text}')


def main(audio_input: str, temp_dir: str):
    # Step 1: Diarization
    segments = diarize_audio(audio_input)

    # Step 2: Assign speaker names to the "speaker_x" fields using our
    # speaker recognition model.
    # We're looking for a dict that maps "speaker_x" to a name.
    audio: AudioType = audio_from_file(audio_input)
    segments = assign_speaker_names(audio, segments)

    # Step 3: Transcribe the audio segments
    segments = transcribe4(audio, segments, temp_dir=temp_dir)
    for segment in segments:
        print_segment(segment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diarization and transcription on an audio file.")
    parser.add_argument(
        "audio_input",
        type=str,
        help="Path to the audio file to process."
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="temp",
        help="Temporary directory to store intermediate files (default: temp).",
        dest="temp_dir"
    )

    args = parser.parse_args()
    os.makedirs(args.temp_dir, exist_ok=True)  # Ensure the temp directory exists
    #main(args.audio_input, args.temp_dir)
    #experiment1(args.audio_input, args.temp_dir)
    #experiment2(args.audio_input, args.temp_dir)
    experiment3(args.audio_input, args.temp_dir)
