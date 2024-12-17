#! python
import argparse
import os
import sys
import json
import math
import logging

from typing import Optional, TypedDict, Union

from hanziconv import HanziConv

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

class SpeakerGuess(TypedDict):
    score: float
    label: str

class SegmentDict(TypedDict):
    row_id: str
    original_start: float
    original_end: float
    modified_start: float
    modified_end: float
    text: str
    speaker: str
    speaker_confidence: float
    selectable: bool
    selected: bool
    export: bool

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
        speaker: Optional[str] = None,
        speaker_confidence: Optional[float] = None,
        selectable: Optional[bool] = True,
        selected: Optional[bool] = False,
        export: Optional[bool] = True
    ):
        self.yt_id: str = yt_id
        self.id: str = id
        self._start: float = float(start)
        self._end: float = float(end)
        self.text: str = text
        self.audio: tuple[np.ndarray, float] = audio
        self.speaker: Optional[str] = speaker
        self.speaker_confidence: Optional[float] = speaker_confidence
        self.selectable = selectable
        self.selected = selected
        self.export = export

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

    def to_dict(self) -> SegmentDict:
        result = {}
        result['row_id'] = self.id
        result['speaker'] = self.speaker
        result['speaker_confidence'] = self.speaker_confidence
        result['original_start'] = self.start
        result['original_end'] = self.end
        result['modified_start'] = self.start
        result['modified_end'] = self.end
        result['text'] = HanziConv.toSimplified(self.text)

        result['selectable'] = self.selectable
        result['selected'] = self.selected
        result['export'] = self.export

        return result

    def set_speaker(self, guess:SpeakerGuess):
        self.speaker = guess['label'].capitalize()
        self.speaker_confidence = guess['score']

    def slice_audio(audio:tuple[np.ndarray, float], start:float, end:float) -> tuple[np.ndarray, float]:
        """
        Class function for slicing a piece of audio given a start and stop.
        There is an instance version that operates on an instance's start, stop
        and audio but this static version lets a caller get a chunk of audio
        without creating an instance of the class.
        """
        audio_data, sr = audio
        result = audio_data[int(start * sr):int(end * sr)]
        return result, sr

    def slice(self) -> tuple[np.ndarray, float]:
        """
        Returns the section of audio that this SubSegment represents

        :return: the audio snippet as tuple(ndarray, sample_rate)
        """
        return SubSegment.slice_audio(self.audio, self.start, self.end)
    
    def static_file_dest(yt_id: str, seg_id: str, path:str="") -> str:
        file_name = f'{yt_id}.{seg_id}.wav'
        return os.path.join(path, file_name)

    def file_dest(self, path:str=""):
        return SubSegment.static_file_dest(self.yt_id, self.id, path=path)
    
    def save_audio_slice(dest: str, audio:tuple[np.ndarray, float], start: float, end: float) -> None:
        audio_slice, sr = SubSegment.slice_audio(audio, start, end)
        sf.write(dest, audio_slice, sr)

    def save_audio(self, path:str=""):
        dest = self.file_dest(path)
        SubSegment.save_audio_slice(dest, self.audio, self.start, self.end)

    def speaker_string(self):
        speak_str = ""
        if self.speaker is None:
            speak_str = "Speaker: "
        elif self.speaker.strip():
            speak_str = f'{self.speaker.strip()} ({round(self.speaker_confidence, 4)}): '
        return speak_str
    
    def __repr__(self):
        return f'[id: {self.id}, start: {self.start}, end: {self.end}, text: "{self.text}", speaker: "{self.speaker_string()}"]'

    def __str__(self):
        return f'[{self.id}] {self.speaker_string()}[{self.start}] {self.text} [{self.end}]'
    
def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video"
    )
    parser.add_argument(
        "-v", "--video", 
        help="The Youtube ID for the video to be transcribed",
        dest='video_id',
        required=True
    )
    parser.add_argument(
        "-g", "--segment-folder",
        help="The root folder to put the individual audio segments.  If missing, audio segments will not be generated.",
        dest="audio_folder",
        default=""
    )
    parser.add_argument(
        "-o", "--out",
        help="The path output audio file.  The default is ./<youtube_id>.mp4",
        dest="filename",
        default=""
    )
    parser.add_argument(
        "-t", "--transcript",
        help="If present, not the literal 'false' and not ending in .json, we'll save the json at <transcript_json>/<youtube_id>.transcription.json.  If present and pointing to something ending in .json, we'll save to <transcript_json>.",
        dest="transcript_json",
        default=""
    )
    parser.add_argument(
        "-s", "--short",
        help="Whether the ID is a video short or a full length video.",
        dest="is_a_short",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '-l', '--log_level',
        help="The python logging level for output",
        dest='log_level',
        default="WARNING"
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

    if filename.endswith('.mp4'):
        filepath = filename
    else:
        filepath = os.path.join(filename, f'{video_id}_audio.mp4')

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
    audio_waveform, _ = audio_from_file(audio_file, sampling_rate=sr)
    suppress_tokens = (
        find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
        if suppress_numerals
        else [-1]
    )

    ## Things I could maybe tune...
    ## beam_size
    ## best_of
    ## hotwords
    ## initial_prompt
    ## vad_filter
    ## suppress_blank
    transcription = whisper_model.transcribe(
        audio_waveform,
        language=language,
        suppress_tokens=suppress_tokens,
        without_timestamps=True,
        word_timestamps=True,
        initial_prompt="This is a dialog between Tom and Ula. Tom speaks English.  Ula speaks Mandarin - 你好妈 - Uh... em... Yeah. I'm good. 你呢.",
        beam_size=1,
        best_of=10,
        suppress_blank=False
    )

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    segs, info = transcription
    
    if print_info:
        logging.info(f'Transcribing audio with duration: {info.duration}')

    return transcription, [audio_waveform, sr]

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
    if check_seg and check_seg.speaker == new_segment.speaker and collapse_speaker:
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

    result = []
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
            new_start_end = round(seg_to_append.start, 2)
            seg_to_append.start = new_start_end
            result.append(SubSegment(
                yt_id=yt_id,
                id='0.0', 
                start=0,
                end=new_start_end,
                text="",
                speaker="",
                audio=audio
            ))
    
        if prev_subseg and prev_subseg.end != subseg.start:
            # Truncate the left_over to two decimal places.
            diff = seg_to_append.start - prev_subseg.end
            end_pad = diff * .35
            beg_pad = diff - end_pad
            prev_subseg.end = math.floor((prev_subseg.end + end_pad) * trunky) / trunky
            seg_to_append.start = math.floor((seg_to_append.start - beg_pad) * trunky) / trunky

        result.append(seg_to_append)

    id_base = segment.id
    sub_id = 0
    subseg = None
    punc_tuples = tuple(list(punctuation))

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
            subseg.text += word.word.replace("Eula", "Ula")
            subseg.end = word.end

        if word.word.endswith(punc_tuples):
            do_append(subseg)
            subseg = None
    
    # If we've got a leftover subseg, we'll add it now.
    if subseg:
        do_append(subseg)
        subseg = None

    return result


def get_segments(video_id, transcript, audio, on_seg=None) -> list[SubSegment]:

    model = "./trained-speakerbox"
    classifier = pipeline("audio-classification", model=model, device=DEVICE)

    segments, info = transcript
    duration = info.duration

    flat_subs = []
    quit_looping = False
    speaker_certainty_cutoff = .59

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
            if len(subseg.text.strip()) and len(raw) > 1000:
                subseg.set_speaker(classifier({"sampling_rate": sampling_rate, "raw": raw}, top_k=1)[0])
            else:
                subseg.speaker = ""
                subseg.speaker_confidence = 0
                subseg.text = "<Intro Music>" if not subseg.text.strip() and subseg.start == 0.0 else subseg.text

            if flat_subs and flat_subs[-1].speaker != subseg.speaker:
                flat_subs[-1].text = flat_subs[-1].text.strip()
                logging.info(flat_subs[-1])

            # We're only going to collapse the speaker segments when
            # we're pretty sure that the speaker assignment was accurate.
            # We're deciding that "pretty sure" is the speaker_certainty_cutoff
            add_subsegment(flat_subs, subseg, collapse_speaker=subseg.speaker_confidence >= speaker_certainty_cutoff)

            if on_seg:
                quit_looping = on_seg(f'Added segment {subseg.id}', cnt, sub_len, subseg.end / duration)
                if quit_looping:
                    break

        if flat_subs:
            flat_subs[-1].text = flat_subs[-1].text.strip()

    return flat_subs

def audio_from_file(source, sampling_rate = SAMPLING_RATE):
    aud = faster_whisper.decode_audio(source, sampling_rate=sampling_rate)
    return (aud, sampling_rate)

def save(
        title:str, 
        vid:str, 
        audio_filename:str, 
        audio_folder:str, 
        json_destination: str,
        save_json:bool, 
        save_audio:bool, 
        audio: tuple[np.ndarray, float], 
        segments:Union[SegmentDict, SubSegment]):

        if not save_audio and not save_json:
            return

        base = audio_folder if audio_folder.endswith(os.path.sep) \
            else os.path.join(audio_folder, 'audio_samples')
        
        def dest(speaker: str) -> str:
            sp = "_unknown_" if speaker is None or not speaker.strip() else speaker.strip().lower()
            p = os.path.join(base, vid, sp)
            if not os.path.exists(p):
                os.makedirs(p)
            return p

        transcript = {}
        transcript["title"] = title
        transcript["YouTubeID"] = vid
        transcript["AudioFile"] = audio_filename
        transcript["transcription"] = []

        for seg in segments:
            values:SegmentDict = seg if type(seg) == dict else seg.to_dict()
            logging.info(seg)
            transcript["transcription"].append(values)
            if save_audio and values['export']:
                segment_file = SubSegment.static_file_dest(vid, values['row_id'], path=dest(values['speaker']))
                SubSegment.save_audio_slice(segment_file, audio, values['modified_start'], values['modified_end'])

        with open(json_destination, 'w', encoding='utf8') as stream:
            json.dump(transcript, stream, indent = 2, ensure_ascii=False)
            stream.write("\n")

if __name__ == '__main__':
    args = get_arguments()
    logging.basicConfig(level=args.log_level.upper())

    audio_file, _, yt = get_youtube_audio(args.video_id, is_short=args.is_a_short, filename=args.filename)

    logging.info(f'Transcribing: "{yt.title}"')
    transcript, audio = transcribe(audio_file, print_info=True)

    flat_subs = get_segments(args.video_id, transcript, audio)

    json_path = args.transcript_json
    save_json = bool(json_path and json_path.lower() != 'false')
    json_dest = json_path if json_path.endswith('.json') else os.path.join(json_path, f'{args.video_id}.transcript.json')
    should_save_segments = bool(args.audio_folder)
    save(
        yt.title,
        args.video_id,
        audio_file,
        args.audio_folder, 
        json_dest, 
        save_json, should_save_segments, audio, flat_subs
    )
    
    if not should_save_segments and not save_json:
        for seg in flat_subs:
            print(seg)