#! python
import argparse
import os
import sys
import math
import logging

from typing import Optional

import numpy as np
from transformers import pipeline

import stable_whisper
import faster_whisper
from faster_whisper.transcribe import Segment

import torch

import librosa
import soundfile as sf

from pytubefix import YouTube
from pytubefix.cli import on_progress

##### Constants #####
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = "large-v3"
SAMPLING_RATE = 16000 # 22050 #
#####################


class SubSegment():
    """
    When faster-whisper segments audio, in many cases, the segments span
    multiple speakers - which isn't ideal for transcription.  So, we're
    going to split the segments on punctuation - assuming that for the
    most part, we'll get complete sentences from each speaker.  These
    SubSegment classes capture those sentences.
    """
    def __init__(self,
        yt_id: str,
        id: str,
        start: float,
        end: float,
        text: str,
        audio: tuple[np.ndarray, float],
        speaker: Optional[str] = None
    ):
        self.yt_id: str = yt_id
        self.id: str = id
        self._start: float = float(start)
        self._end: float = float(end)
        self.text: str = text
        self.audio: tuple[np.ndarray, float] = audio
        self.speaker: Optional[str] = speaker

    @property
    def start(self):
        return self._start
    @start.setter
    def start(self, value):
        self._start = float(value)

    @property
    def end(self):
        return self._end
    @end.setter
    def end(self, value):
        self._end = float(value)

    def slice(self) -> tuple[np.ndarray, float]:
        """
        Returns the section of audio that this SubSegment represents

        :return: the audio snippet as tuple(ndarray, sample_rate)
        """
        audio_data, sr = self.audio
        result = audio_data[int(self.start * sr):int(self.end * sr)]
        return result, sr
    
    def file_dest(self, path:str=""):
        file_name = f'{self.yt_id}.{self.id}.wav'
        return os.path.join(path, file_name)
    
    def save_audio(self, path:str=""):
        dest = self.file_dest(path)
        audio_slice, sr = self.slice()
        sf.write(dest, audio_slice, sr)

    def __repr__(self):
        return f'[id: {self.id}, start: {self.start}, end: {self.end}, text: "{self.text}", speaker: "{self.speaker}"]'

    def __str__(self):
        return f'[{self.id}] {self.speaker if self.speaker else "Speaker"}: [{self.start}] {self.text} [{self.end}]'
    
def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video",
        usage="transcribe -v <video id> -t <huggingface auth token> [-o name for audio files]"
    )
    parser.add_argument(
        "-v", "--video", 
        help="The Youtube ID for the video to be transcribed",
        dest='video_id',
        required=True
    )
    parser.add_argument(
        "-d", "--directory",
        help="The name of the folder to put all of the downloaded and generated audio files.",
        dest="audio_folder",
        default=""
    )
    parser.add_argument(
        "-o", "--out",
        help="The name of the base name of the downloaded audio file.  Default is to use the video ID.",
        dest="filename",
        default=""
    )
    parser.add_argument(
        "-s", "--short",
        help="Whether the ID is a video short or a full length video.",
        dest="is_a_short",
        type=bool,
        default=False
    )
    parser.add_argument(
        "-a", "--save",
        help="Whether to save the audio segments as wav files.",
        dest="should_save",
        type=bool,
        default=False
    )
    return parser.parse_args()

def extract_features(audio_file: tuple[np.ndarray, float] | str):
    # Load audio file
    if type(audio_file) == str:
        y, sr = librosa.load(audio_file)
    else:
        y, sr = audio_file

    # Extract mel
    return np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T,axis=0)

def get_youtube_audio(
        video_id: str,
        path:Optional[str]="",
        yt_object:Optional[YouTube]=None,
        is_short:bool=False,
        progress_callback = None,
        filename:Optional[str]=""
    ):
    """
    Given a YouTube object, extract the audio component of it.

    Parameters:
        vt (class YouTube): The pytubefix core developer interface

    Returns:
        string: The name of the file where the audio was saved.
        Stream: The video stream for the selected YouTube audio.
        YouTube: The YouTube object that was created (or the one passed in)
    """

    yt_link = (
        f'https://www.youtube.com/shorts/{video_id}' 
        if is_short else
        f'https://www.youtube.com/watch?v={video_id}'
    )


    yt = yt_object if yt_object else YouTube(yt_link, on_progress_callback=progress_callback)

    filepath = os.path.join(
        path, 
        f'{filename.strip() if filename and filename.strip() else video_id}_audio.mp4'
    )
    audio = yt.streams.get_audio_only()
    if audio:
        filename = audio.download(filename=filepath)
    else:
        print(f'No audio in YouTube video', file=sys.stderr)
        exit(2)

    if not progress_callback:
        logging.info(f'Audio saved as filename {filename}')

    return filename, audio, yt

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = [
        -1,
    ]
    for token, token_id in tokenizer.get_vocab().items():
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(token_id)
    return numeral_symbol_tokens

def transcribe(
        audio_file, 
        model_name=MODEL_NAME,
        device=DEVICE, 
        compute_type=COMPUTE_TYPE, 
        suppress_numerals=True,
        language="en",
        print_info=False):

    """
    Passes the audio to faster_whisper for transcription and then
    deletes the whisper model that was created.  This is probably
    not necessary since we're using the CPU and not the GPU but
    whatever.  Maybe it will help with big files.
    """
    whisper_model = stable_whisper.load_faster_whisper(
        model_name, device=device, compute_type=compute_type
    )

    sr = whisper_model.feature_extractor.sampling_rate
    audio_waveform = faster_whisper.decode_audio(audio_file, sampling_rate=sr)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )
    transcription = whisper_model.transcribe(
        audio_waveform,
        language=language,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
        word_timestamps=True
    )

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    segs, info = transcription
    
    if print_info:
        logging.info(f'Transcribing audio with duration: {info.duration}')

    return segs, [audio_waveform, sr]

def add_subsegment(segments:list[SubSegment], new_segment:SubSegment, collapse_speaker:bool=True):
    """
    Adds a segment to a segements array.  If the speaker
    of the new segment is the same as the speaker of the last
    segment in the array, the new_segment is appended to the last
    segment.  Otherwise, a new segment is added to the segments
    array
    param: segments: Array of SubSegment objects
    param: new_segment: A single SubSegment to add to the array or append to the last
    return: The resulting array
    """

    # Get the last segment
    check_seg = segments[-1] if segments else None
    if check_seg and check_seg.speaker.strip() == new_segment.speaker.strip() and collapse_speaker:
        check_seg.end = new_segment.end
        check_seg.text += new_segment.text
    else:
        segments.append(new_segment)


# punctuation:str="\"'.。,，!！?？:：”)]}、"
def split_segment(
        yt_id: str,
        audio: tuple[np.ndarray, float], 
        segment: Segment, 
        last_sub: Optional[SubSegment],
        punctuation:str="\"'.。!！?？:：”)]}、") -> list[SubSegment]:

    id_base = segment.id
    sub_id = 0
    result = []
    subseg = None
    punc_tuples = tuple(list(punctuation))

    def do_append(seg_to_append):
        # There are times when whisper's word timings aren't accurate.
        # There is a package called stable-ts, which I am importing,
        # that is supposed to make it better - and I guess it does -
        # but at the expense of REALLY slowing things down.  So,
        # although it feels janky, I'm adding a bunch of special case
        # stuff to do a little cleanup.
        # - The first segment should start at 0 if it's almost 0
        # - If there is a gap between the last sentence and this one,
        #   we'll assign part of the gap to the end of the previous sentence
        #   and part to this one.
        # - If the text is identical and the duration is less than a second,
        #   just bump the last guy and drop the segment.

        trunky = 100
        prev_subseg = result[-1] if result else last_sub

        if (prev_subseg and
            seg_to_append.text == prev_subseg.text and
            prev_subseg.end  - seg_to_append.end < 1):
            prev_subseg.end = seg_to_append.end
            return

        # If the very first segment doesn't start at 0, we'll add a buffer
        # segment to account for all the space.
        if not prev_subseg and seg_to_append.start > 0:
            result.append(SubSegment(
                yt_id=yt_id,
                id='0.0', 
                start=0,
                end=seg_to_append.start,
                text="",
                audio=audio
            ))
    
        if not prev_subseg and seg_to_append.start > 0 and seg_to_append.start < 10:
            seg_to_append.start = 0

        if prev_subseg and prev_subseg.end != subseg.start:
            # Truncate the left_over to two decimal places.
            diff = seg_to_append.start - prev_subseg.end
            end_pad = diff * .35
            beg_pad = diff - end_pad
            prev_subseg.end = math.floor((prev_subseg.end + end_pad) * trunky) / trunky
            seg_to_append.start = math.floor((seg_to_append.start - beg_pad) * trunky) / trunky

        result.append(seg_to_append)

    for word in segment.words:
        if subseg is None:
            sub_id += 1
            subseg = SubSegment(
                yt_id=yt_id,
                id=f'{id_base}.{sub_id}', 
                start=word.start,
                end=word.end,
                text=word.word,
                audio=audio
            )
        else:
            subseg.text += word.word
            subseg.end = word.end

        if word.word.endswith(punc_tuples):
            do_append(subseg)
            subseg = None
    
    # If we've got a leftover subseg, we'll add it now.
    if subseg:
        do_append(subseg)
        subseg = None

    return result


def get_segments(video_id, segments, audio, on_seg=None) -> list[SubSegment]:

    model = "./trained-speakerbox"
    classifier = pipeline("audio-classification", model=model, device=DEVICE)

    flat_subs = []
    quit_looping = False
    for segment in segments:
        if quit_looping:
            break
        subs = split_segment(video_id, audio, segment, flat_subs[-1] if flat_subs else None)
        cnt = 0
        sub_len = len(subs)
        for subseg in subs:
            raw, sampling_rate = subseg.slice()
            ### If there is too little audio in the sample, we're just going to ignore attempting
            ### set the speaker
            if len(raw) > 1000:
                subseg.speaker = classifier({"sampling_rate": sampling_rate, "raw": raw}, top_k=1)[0]['label'].capitalize()
            else:
                subseg.speaker = ""
            logging.debug(f'Adding subsegment {subseg.id}, speaker: {subseg.speaker}')
            add_subsegment(flat_subs, subseg, collapse_speaker=False)
            if on_seg:
                quit_looping = on_seg(f'Added segment {subseg.id}', cnt, sub_len)
                if quit_looping:
                    break

    return flat_subs

if __name__ == '__main__':
    args = get_arguments()

    audio_file, _, yt = get_youtube_audio(args.video_id, path=args.audio_folder, is_short=args.is_a_short, filename=args.filename)

    print(f'Transcribing: "{yt.title}"')
    transcript_segments, audio = transcribe(audio_file, print_info=True)

    flat_subs = get_segments(args.video_id, transcript_segments, audio)

    for seg in flat_subs:
        print(seg)
        if args.should_save:
            seg.save_audio(args.audio_folder)

