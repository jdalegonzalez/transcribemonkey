#! python
import argparse
import os
import re
import json
import math
import logging
import html
import datetime
import string
import sys
from enum import Enum
from collections import UserDict

from word2number import w2n

# EXCEPTIONALLY janky but translator uses a very old httpx.
# If I need to fix it, the fix is to monkey patch httpx to
# be backward compatible with googletrans.  What I've
# currently done is just live with an old httpcore.
#import httpcore
#setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')

from googletrans import Translator

from typing import Optional, TypedDict, Union, Iterable

from xpinyin import Pinyin
from hanziconv import HanziConv
import jieba

import numpy as np
from transformers import pipeline, Pipeline

import stable_whisper
import faster_whisper
from faster_whisper.transcribe import Segment, Word, TranscriptionInfo
import whisper

import torch

import soundfile as sf

from pytubefix import YouTube
from pytubefix.cli import on_progress

from docxtpl import DocxTemplate

class PropertyDict(UserDict):
    def __init__(self, dict=None, /, **kwargs):
        super().__init__(dict, kwargs=kwargs)

    def __getattr__(self, key):
        if key in self.data: return self.data[key]
        else: raise AttributeError(key)

    def __setattr__(self, key, value):
        if key == 'data':
            super().__setattr__(key, value)
        else:
            self.data[key] = value

class WhisperSegment(PropertyDict):
    def __init__(self, dict=None, /, **kwargs):
        super().__init__(dict, kwargs=kwargs)
        self._words = None

    def __getattr__(self, key):
        if key == 'words':
            if not self._words: self._words = [WhisperSegment(w) for w in self.data['words']]
            return self._words
        return super().__getattr__(key)



class TranscribeMethods(Enum):
    FASTER_WHISPER = 'faster-whisper'
    STABLE_WHISPER = 'stable-whisper'
    HUGGING_FACE = 'hugging_face-whisper'

##### Constants #####
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = "large-v3"
WHISPER_MODEL_DIR = "./models"
MAX_SEGMENT_LENGTH = 60 # in seconds
SPEAKER_COLLAPSE_CERTAINTY_CUTOFF = .60 # 0 - 1
SWITCH_SPEAKER_CUTOFF = .70 # 0 - 1
TOO_SHORT_TO_GUESS_SECONDS = .2
TOO_SHORT_TO_BE_REAL_SECONDS = .1
MIN_PROBABILITY_TO_INCLUDE = .4

# punctuation:str="\"'.。,，!！?？:：”)]}、"
DEFAULT_PUNCTUATION = "\"'.。!！?？:：”)]}、" # no comma

NO_GUESS = {'label': "", 'score': 0}
HF_MODELS = {
    "tiny.en": "openai/whisper-tiny.en",
    "tiny": "openai/whisper-tiny",
    "base.en": "openai/whisper-base.en",
    "base": "openai/whisper-base",
    "small.en": "openai/whisper-small.en",
    "small": "openai/whisper-small",
    "medium.en": "openai/whisper-medium.en",
    "medium": "openai/whisper-medium",
    "large-v1": "openai/whisper-large-v1",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large": "openai/whisper-large-v3"
}

# These are intentionally repetitive because I'm
# trying to make sure I don't kill something by
# accident so I want long, specific strings.
# Maybe later I'll look at patters for some of
# them.
hallucinations = [
    "Subtitles by the Amara.org community you.",
    "Subtitles by the Amara.org community",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
    "请点击订阅点赞。",
    "曲 李宗盛嗯。 ",
    "优优独播剧场——YoYo Television Series Exclusive",
    "优独播剧场——YoYo Television Series Exclusive",
    "场——YoYo Television Series Exclusive",
    "ING PAO CANADAING PAO TORONTO",
]

translator = Translator()
pinyin_obj = Pinyin()

# There is no real reason to put this 'priming' request here.
# I'm doing it because jieba logs some stuff on the first 
# request you make to it and I want it to be before all the
# stuff I log.
jieba.lcut("好大家好")

prompt_str = \
"This is a dialog between Tom and Ula. " \
"Tom speaks English. " \
"Ula speaks Mandarin. " \
"Mandarin Monkey. " \
"- 你好妈 - Uh... em... Yeah. I'm good. 你呢. " \
"Produce a verbatim transcription."

## Things I could maybe tune...
## beam_size: 1,
## best_of: 10
## initial_prompt: prompt_str
## vad_filter: 
## suppress_blank: False

DEFAULT_TRANSCRIBE_KWARGS = {
    'language': 'en',
    'task': 'transcribe',
    'beam_size': 1,
    'best_of': 10,
    'initial_prompt': prompt_str,
    'suppress_blank': False,
    'condition_on_previous_text': False,
    'suppress_tokens': [-1],
    'without_timestamps': True,
    'word_timestamps': True
}


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

AudioType = tuple[np.ndarray, float]
TranscriptionType = tuple[Iterable[Segment], TranscriptionInfo]

def audio_filename(file_or_path: str, video_id: str):
    if file_or_path.endswith('.mp4'):
        result = file_or_path
    else:
        result = os.path.join(file_or_path, f'{video_id}_audio.mp4')
    return result

def transcript_json_name(file_or_path: str, video_id: str):
    if file_or_path.endswith('.json'):
        result = file_or_path
    else:
        result = os.path.join(file_or_path, f'{video_id}.transcript.json')
    return result

def to_minutes_seconds(seconds:float) -> str:
    secs = round(seconds)
    minute_part = secs // 60
    second_part = secs % 60
    return f'{minute_part:02}:{second_part:02}'

chinese_characters_range = '\u3002-\u9fff'
chinese_character_re = re.compile(f'[{chinese_characters_range}]')
def contains_chinese(text:str) -> bool:
    return chinese_character_re.search(text) is not None

def combine_words(words:list[str]):
    was_chinese = False
    text = ""
    pinyin = ""
    pinyin_pfx = ""
    for word in words:
        is_chinese = contains_chinese(word)
        prefix = " " if text and word != ' ' and was_chinese != is_chinese else ""
        text += (prefix + word)
        was_chinese = is_chinese
        pinyin_pfx = " " if pinyin and word not in DEFAULT_PUNCTUATION else ""
        pinyin += (pinyin_pfx + pinyin_obj.get_pinyin(word, '', tone_marks="marks")) if is_chinese \
            else (prefix + word)
        pinyin_pfx = " "

    return (text,pinyin)

def separate_english(text):
    if contains_chinese(text):
        return combine_words(jieba.lcut(text))
    return (text, "")

# anomalous words are very long/short/improbable
def word_anomaly_score(word: Word) -> float:
    probability = word.probability
    duration = word.end - word.start
    score = 0.0
    if probability < 0.15:
        score += 1.0
    if duration < 0.133:
        score += (0.133 - duration) * 15
    if duration > 2.0:
        score += duration - 2.0
    return score

def is_anomaly(score, num_words, text:Union[str, list[Word]]):
    # This was the old score
    #return score >= 3 or score + 0.01 >= len(words)
    if score > 6 and score <= 7:
        txt = text if type(text) == str else " ".join([w.word.strip() for w in text])
        logging.info(f"Almost Anomaly: ({score}) '{txt}'")
    return score > 7 or score + 0.01 >= num_words

def is_segment_anomaly(segment: Segment) -> bool:
    if segment is None or not segment.words: return False
    words = [w for w in segment.words if w.word.strip() and w.word not in DEFAULT_PUNCTUATION + ","]
    words = words[:8]
    score = sum(word_anomaly_score(w) for w in words)
    return is_anomaly(score, len(words), words)

class SubSegment():
    """
    When faster-whisper segments audio, in many cases, the segments span
    multiple speakers - which isn't ideal for transcription.  So, we're
    going to split the segments on punctuation - assuming that for the
    most part, we'll get complete sentences from each speaker.  These
    SubSegment classes capture those sentences.
    """
    def __init__(self,
        id: str,
        start: float,
        audio: AudioType,
        end: Optional[float] = None,
        text: Optional[str] = "",
        speaker: Optional[str] = "",
        speaker_confidence: Optional[float] = None,
        selectable: Optional[bool] = True,
        selected: Optional[bool] = False,
        export: Optional[bool] = True
    ):
        self.id: str = id
        self._start: float = float(start)
        self._end: float = float(end) if end is not None else self._start
        self.text: str = text
        self.pinyin: str = ''
        self.translation: str = ''
        self.audio: AudioType = audio
        self.original_speaker: Optional[str] = speaker
        self.speaker: Optional[str] = speaker
        self.speaker_confidence: Optional[float] = speaker_confidence
        self.selectable = selectable
        self.selected = selected
        self.export = export
        self._anomaly_score = 0

    def from_whisper_segment(audio: AudioType, segment:WhisperSegment):
        return SubSegment(
            id=str(segment.id),
            audio=audio,
            start=segment.start,
            end=segment.end,
            text=segment.text
        )
    
    def start_segment(audio: AudioType):
        return SubSegment(
            id='0.0', 
            start=0,
            audio=audio
        )
    
    @property
    def speaker_guess(self):
        return SpeakerGuess(score=self.speaker_confidence, label=(self.speaker or "").lower())

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

    @property
    def duration(self):
        return self._end - self._start

    def add_word(self, word:Word) -> None:
        score = word_anomaly_score(word)
        text=word.word.replace("Yula", "Ula").replace("Eula", "Ula").replace("Yulo", "Ula")
        self.text += text
        self.end = word.end
        self._anomaly_score += score    

    def is_anomaly(self, punc = DEFAULT_PUNCTUATION + ",") -> bool:
        num_words = word_count(self.text, punc=punc)
        return is_anomaly(self._anomaly_score, num_words, self.text)
    
    def anomaly_score(self):
        return self._anomaly_score
    
    def from_segment(audio: AudioType, seg:Segment, is_clips:bool=False):
        ret = SubSegment(
            audio=audio,
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text=seg.text
        )
        ret.finalize()
        return ret
    
    def to_dict(self) -> SegmentDict:
        result = {}
        result['row_id'] = self.id
        result['original_speaker'] = self.original_speaker
        result['speaker'] = self.speaker
        result['speaker_confidence'] = self.speaker_confidence
        result['original_start'] = self.start
        result['original_end'] = self.end
        result['modified_start'] = self.start
        result['modified_end'] = self.end
        result['text'] = HanziConv.toSimplified(self.text.strip())
        result['pinyin'] = self.pinyin
        result['translation'] = self.translation
        result['selectable'] = self.selectable
        result['selected'] = self.selected
        result['export'] = self.export

        return result

    def finalize(self, translate: bool=False):

        self.text = HanziConv.toSimplified(self.text.strip())

        for hal in hallucinations:
            if self.text.endswith(hal):
                self.text = self.text[:-len(hal)]
                break

        text, pinyin = separate_english(self.text)
        self.text = text.strip()

        if translate:
            self.pinyin = pinyin.strip()
            self.translation = translate(self.text, src='zh-cn', dest='en')

    def set_speaker(self, guess:SpeakerGuess):
        self.original_speaker = guess['label'].capitalize()
        self.speaker = self.original_speaker
        self.speaker_confidence = guess['score']


    def slice_spectrogram(audio:AudioType, start: float, end:float, video_id: str = "") -> None:
        import librosa
        import librosa.display
        import matplotlib
        import matplotlib.pyplot as plt
        
        add_chroma = False
        color_bars = False
        nrows = 3 if add_chroma else 2

        fig, ax = plt.subplots(nrows=nrows, ncols=1, sharex=True)

        raw, sr = SubSegment.slice_audio(audio, start, end)
        #raw, sr = audio

        # Standard spectrogram
        D = librosa.stft(raw)  # STFT of y
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax[0])
        ax[0].set(title='STFT (log scale)')

        # Compute the mel spectrogram
        S = librosa.feature.melspectrogram(y=raw, sr=sr)
        M_db = librosa.power_to_db(S, ref=np.max)
        img2 = librosa.display.specshow(M_db, x_axis='time', y_axis='mel', ax=ax[1])
        ax[1].set(title='Mel')

        # Compute the chroma
        if add_chroma:
            chroma = librosa.feature.chroma_cqt(y=raw, sr=sr)
            img3 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', ax=ax[2])
            ax[2].set(title='Chroma')

        # To eliminate redundant axis labels, we'll use "label_outer" on all subplots:
        for ax_i in ax: ax_i.label_outer()

        # And we can share colorbars:
        if color_bars: fig.colorbar(img1, ax=[ax[0], ax[1]], format='%+2.0f dB')

        # Or have individual colorbars:
        if add_chroma and color_bars: fig.colorbar(img3, ax=[ax[2]])

        # We can then even do fancy things like zoom into a particular time and frequency
        # region.  Since the axes are shared, this will apply to all three subplots at once.
        #ax[0].set(xlim=[start, end])

        # Display the spectrogram
        if video_id: fig.canvas.manager.set_window_title(video_id)

        # This little trick allows us to have the labels start from
        # the "start" value rather than 0
        class Fmt(matplotlib.ticker.StrMethodFormatter):
            def __call__(self, x, pos=None):
                return self.fmt.format(x=(x+start), pos=pos)

        ax[0].margins(x=0)
        ax[0].minorticks_on()
        ax[0].xaxis.set_major_formatter(Fmt('{x:.2f}'))

        plt.show()
        plt.close()

    def slice_audio(audio:AudioType, start:float, end:float) -> AudioType:
        """
        Class function for slicing a piece of audio given a start and stop.
        There is an instance version that operates on an instance's start, stop
        and audio but this static version lets a caller get a chunk of audio
        without creating an instance of the class.
        """
        audio_data, sr = audio
        result = audio_data[int(start * sr):int(end * sr)]
        return result, sr

    def spectogram(self, title="Spectrogram") -> None:
        return SubSegment.slice_spectrogram(self.audio, self.start, self.end, title=title)
    
    def slice(self) -> AudioType:
        """
        Returns the section of audio that this SubSegment represents

        :return: the audio snippet as tuple(ndarray, sample_rate)
        """
        return SubSegment.slice_audio(self.audio, self.start, self.end)
    
    def static_file_dest(yt_id: str, seg_id: str, path:str="") -> str:
        file_name = f'{yt_id}.{seg_id}.wav'
        return os.path.join(path, file_name)

    def file_dest(self, yt_id: str, path:str=""):
        return SubSegment.static_file_dest(yt_id, self.id, path=path)
    
    def save_audio_slice(dest: str, audio:AudioType, start: float, end: float) -> None:
        audio_slice, sr = SubSegment.slice_audio(audio, start, end)
        sf.write(dest, audio_slice, sr)

    def save_audio(self, yt_id: str, path:str=""):
        dest = self.file_dest(yt_id, path)
        SubSegment.save_audio_slice(dest, self.audio, self.start, self.end)

    def speaker_string(self):
        speak_str = ""
        if self.speaker is None:
            speak_str = "Speaker: "
        elif self.speaker.strip():
            speak_str = f'{self.speaker.strip()} ({round(self.speaker_confidence, 4):.4f}): '
        return speak_str

    def __repr__(self):
        return f'[id: {self.id}, start: {self.start}, end: {self.end}, text: "{self.text.strip()}", speaker: "{self.speaker_string()}"]'

    def __str__(self):
        return f'[{to_minutes_seconds(self.start)}] {self.speaker_string()}{self.text.strip()}'

def translate(txt:str, src:str=None, dest:str=None) -> str:
    if src is None: src = 'zh-cn' if contains_chinese(txt) else 'en'
    if dest is None: dest = 'en' if src == 'zh-cn' else 'zh-cn'
    result = translator.translate(txt, src=src, dest=dest)
    return result.text

def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video"
    )
    parser.add_argument(
        "-a", "--annote",
        help="Use PyAnnote for diarization",
        dest="use_pyannote",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-c", "--clips",
        help="A set of clips to request in the form of start1,stop1,start2,stop2",
        dest="clips",
        default=None
    )
    parser.add_argument(
        "-d", "--doc", 
        help="Create a word document from a json file.  The -t argument must also be passed.",
        dest='word_doc',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-e", "--episode",
        help="The episode number, if there is one (and you know it).",
        dest="episode",
        default=""
    )
    parser.add_argument(
        "-g", "--segment-folder",
        help="The root folder to put the individual audio segments.",
        dest="audio_folder",
        default=""
    )
    parser.add_argument('-k', '--kill_anomalies',
        help="Whether or not to drop dialog that looks like a hallucination.",
        dest='kill_anomalies',
        default=False,
        action='store_true'
    )
    parser.add_argument('--lang',
        help="The language the audio is in, if known.",
        dest='lang',
        default="en"
    )
    parser.add_argument(
        '-l', '--log_level',
        help="The python logging level for output",
        dest='log_level',
        default="WARNING"
    )
    parser.add_argument(
        "-o", "--out",
        help="The path output audio file.  The default is ./<youtube_id>.mp4",
        dest="filename",
        default=""
    )
    parser.add_argument(
        "-p", "--plot",
        help="Shows the melspectorgram for a section of audio.  The clip argument must also be passed in.",
        dest="do_plot",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "-s", "--short",
        help="Whether the ID is a video short or a full length video.",
        dest="is_a_short",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-t", "--transcript",
        help="If we've been asked to transcribe, this is either the path create the json with the default name or, if the argument ends in .json, the full path and name of the file.  The default name is <transcript_json>/<youtube_id>.transcription.json. If we're creating the word doc, it's the full path to the json file or the json folder.",
        dest="transcript_json",
        default=""
    )
    parser.add_argument(
        "-w", "--whisper",
        help="Use the whisper model instead of the faster-whisper one.",
        dest="use_whisper",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "-v", "--video", 
        help="The Youtube ID for the video to be transcribed",
        dest='video_id',
        default=None
    )

    args = parser.parse_args()
    if args.video_id is None and not args.word_doc:
        parser.error("Youtube ID (-v, --video) is required.")

    if args.word_doc:
        exists = args.transcript_json and os.path.exists(args.transcript_json)
        if not exists:
            parser.error("A path to the json transcript (-t, --transcript) is required when generating a word doc")
        is_file = os.path.isfile(args.transcript_json)
        if not is_file and args.video_id is None:
            parser.error("A full path to a json file is required when generating a doc if you don't also supply a video ID")
        if not is_file:
            new_file = os.path.join(args.transcript_json, f'{args.video_id}.transcript.json')
            if not os.path.exists(new_file) or not os.path.isfile(new_file):
                parser.error(f'The transcript {new_file} does not exist')
            args.transcript_json = new_file

    if args.do_plot and not args.clips:
        parser.error("Clips must be specified to plot the spectogram.  Use -c '0' if you really want to do the entire file.")

    if args.clips:
        clips = [float(clip) for clip in args.clips.split(",")]
        args.clips = clips

    return args

class FakeYouTube():
    def __init__(self, title:str = 'Fake Title'):
        self.title = title

def cached_youtube(video_id: str, filename: str, transcript_path: str) -> Union[None, tuple[str, None, FakeYouTube]]:
    transcript_json = transcript_json_name(transcript_path, video_id)
    filepath = audio_filename(filename, video_id)
    result = None
    if os.path.isfile(transcript_json):
        # We've already transcribed and this is a re-transcribe.
        # No reason to do the download again (and we don't want
        # to be flagged as a bot because we got the same file
        # over and over).
        json_data = load_transcript_from_file(transcript_json)
        audiofile = json_data.get('AudioFile', None) if json_data else None
        tx_video_id = json_data.get('YouTubeID', None) if json_data else None
        title = json_data.get('title', "") if json_data else ""
        absp = os.path.abspath
        if absp(audiofile) == absp(filepath) and tx_video_id == video_id:
            logging.info("Returning cached audio.")
            # The audio file and youtube id match, so we've done this before.
            yt = FakeYouTube(title)
            audio = None # I could load it here but it isn't used and at what hertz?
            result = (filepath, audio, yt)

    return result

def get_youtube_audio(
        video_id: str,
        yt_object:Optional[YouTube]=None,
        is_short:bool=False,
        progress_callback = None,
        filename:Optional[str]="",
        transcript_path:Optional[str]=""
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

    # If we've already produced a transcript and are just doing
    # it again, we'll use the cached audio and title.
    result = cached_youtube(video_id, filename, transcript_path)
    if result is not None: return result
    
    yt_link = (
        f'https://www.youtube.com/shorts/{video_id}' 
        if is_short else
        f'https://www.youtube.com/watch?v={video_id}'
    )
    filepath = audio_filename(filename, video_id)

    yt = yt_object if yt_object else YouTube(yt_link, on_progress_callback=progress_callback,use_oauth=True)

    audio = yt.streams.get_audio_only()
    if audio:
        filename = audio.download(filename=filepath)
    else:
        print(f'No audio in YouTube video', file=sys.stderr)
        exit(2)

    if not progress_callback:
        logging.info(f'Audio saved as filename {filename}')

    return (filename, audio, yt)

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = [
        -1,
    ]
    for token, token_id in tokenizer.get_vocab().items():
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(token_id)

    return numeral_symbol_tokens

def load_transcript_from_file(filename: str) -> dict:
    with open(filename, encoding='utf8') as f:
        transcript = json.load(f)
    
    return transcript

def strip_json_suffix(filename):
    full_suffix = '.transcript.json'
    json_suffix = '.json'
    if filename.endswith(full_suffix):
        return filename[:-len(full_suffix)]
    if filename.endswith(json_suffix):
        return filename[:-len(json_suffix)]
    return filename

def plot_spectogram_from_file(filename: str, video_id: str, transcript: str, clip:list[float]):
    # We've got a couple of ways to try and find the audio file
    # we're supposed to be plotting.  First, maybe, the filename is a 
    # file, not a folder.  If it's a file, we found it.  If it's a folder, 
    # we'll keep going.  The block is structured as successive "if" instead
    # of if/elif so that any part of the process and fail and we'll try
    # something else.

    path_to_file = None
    filepath = audio_filename(filename, video_id)
    if filepath and os.path.isfile(filepath):
        # found it.
        path_to_file = filepath

    if not path_to_file and transcript:
        # OK, let's grab the filename from the transcript id
        filepath = transcript_json_name(transcript, video_id)
        if os.path.isfile(filepath):
            # Parse the json and grab the audio file from it
            json_data = load_transcript_from_file(filepath)
            filename = json_data.get('AudioFile', None) if json_data else None
            video_id = json_data.get('YouTubeID', video_id) if json_data else video_id
            path_to_file = filename if filename and os.path.isfile(filename) else None
    
    if not path_to_file:
        logging.error(f"Couldn't find audio file for path '{filename}' and video '{video_id}'")
        return
    
    audio = audio_from_file(path_to_file, whisper.audio.SAMPLE_RATE)
    if len(clip) == 0:
        logging.error("No clip passed.")
    elif len(clip) > 2:
        logging.warning(f"Only generating for the first clip")
    elif len(clip) < 2:
        raw, sr = audio
        clip.append(len(raw)/sr)

    SubSegment.slice_spectrogram(audio, clip[0], clip[1], title=video_id)

    return

def create_word_doc(filename):
    json_data = load_transcript_from_file(filename)
    
    def esc(dta, ele):
        dta[ele] = html.escape(dta[ele])

    if json_data is not None:
        esc(json_data, 'title')
        episode = json_data.get('episode')
        json_data['footer_title'] = f'Episode {episode}' if episode else json_data['title']
        for line in json_data['transcription']:
            line['minutes_seconds'] = to_minutes_seconds(line['modified_start'])
            esc(line, 'text')

        doc = DocxTemplate("./interview-transcription-template.docx")
        doc.render(json_data)
        # We'll just stick the ouput in the same location
        # as the json
        save_name = f'{strip_json_suffix(filename)}.docx'
        doc.save(save_name)

        return save_name
    
    return None

def default_on_seg(text: str, num: int, count: int, percent_complete: float) -> bool:
    return False

def transcribe(
        audio_file:str, 
        model_name:str=MODEL_NAME,
        device:str=DEVICE, 
        compute_type:str=COMPUTE_TYPE, 
        suppress_numerals:bool=False,
        language:str="en",
        on_seg:Optional[callable] = default_on_seg,
        clips: list[float] = None) -> Union[None, tuple[TranscriptionType, AudioType]]:

    """
    Passes the audio to faster_whisper for transcription and then
    deletes the whisper model that was created.  This is probably
    not necessary since we're using the CPU and not the GPU but
    whatever.  Maybe it will help with big files.
    """

    # There is a package called stable-ts, which adds transcribe_stable
    # that is supposed to make it timestamps better. And, maybe it does??
    # but maybe it slows things WAY down.  Change below from transcribe
    # to transcribe_stable and back to see.
    method = TranscribeMethods.STABLE_WHISPER
    #method = TranscribeMethods.STABLE_WHISPER if not clips else TranscribeMethods.FASTER_WHISPER

    
    whisper_model = stable_whisper.load_faster_whisper(
        model_name, device=device, compute_type=compute_type
    )
    sr = whisper_model.feature_extractor.sampling_rate
    tokenizer = whisper_model.hf_tokenizer

    if on_seg(f'{model_name} loaded. Using {method}', 0, 0, 0): return None

    kwargs = dict(DEFAULT_TRANSCRIBE_KWARGS)
    kwargs['language'] = language
    kwargs['hotwords'] = "Ula Mandarin Monkey"

    if clips is not None: kwargs['clip_timestamps'] = clips
    if suppress_numerals: kwargs['suppress_tokens'] = find_numeral_symbol_tokens(tokenizer)
    
    audio_waveform, _ = audio_from_file(audio_file, sampling_rate=sr)
    dur = to_minutes_seconds(len(audio_waveform) / sr)
    logging.info(f'Transcribing audio with duration: {dur}')
    
    if on_seg(f'{audio_file} loaded. Duration: {dur}, sample rate: {sr}', 0, 0, 0): return None

    if method == TranscribeMethods.STABLE_WHISPER:
        # O5OjKjno9Pw
        # duration: 41:08
        # "large-v3" INFO:root:Transcription finished in 00:27:51
        _, info = whisper_model.transcribe(audio_waveform, **kwargs)
        result:stable_whisper.WhisperResult = whisper_model.transcribe_stable(audio_waveform, **kwargs)
        transcription = (result, info)
    else:
        # O5OjKjno9Pw
        # duration: 41:08
        # "large-v3" INFO:root:Transcription finished in 00:31:15
        transcription = whisper_model.transcribe(audio_waveform, **kwargs)

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    
    if on_seg(f'Segments loaded. ', 0, 0, 0): return None

    return (transcription, (audio_waveform, sr))

def matching_speaker(seg: SubSegment, seg2: SubSegment):
    if not seg or not seg2:
        return None

    # Never merge with a "non-speaker"
    if seg.speaker_confidence == 0:
        return None

    if seg.speaker == seg2.speaker:
        return seg.speaker
    if not seg.speaker and seg2.speaker:
        return seg2.speaker
    if seg.speaker and not seg2.speaker:
        return seg.speaker
    
    return None

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

    check_seg = segments[-1] if segments else None
    match = matching_speaker(check_seg, new_segment)
    new_dur = new_segment.end - (check_seg.start if check_seg else new_segment.start)
    if match is not None and collapse_speaker and new_dur < MAX_SEGMENT_LENGTH:
        check_seg.speaker = match
        check_seg.end = new_segment.end
        check_seg.text += new_segment.text
        return False
    else:
        segments.append(new_segment)
        return True

def speaker_for_clip(
    audio: AudioType,
    classifier: Pipeline,
    piece:Union[Word, SubSegment],
    previous: Union[SpeakerGuess, None]=None) -> SpeakerGuess:

    # If there isn't text, then there can't be a speaker.
    duration = piece.end - piece.start
    is_subseg = type(piece) == SubSegment
    txt = (piece.text if is_subseg else piece.word).strip()
    if not txt: return previous if previous is not None else NO_GUESS

    # These systems have a tendency to chop the end of a word
    # more than they chop the front.  So, we're going to 
    # mostly pad the end.
    pad = min(.5, duration / 2)
    start_pad = 0
    end_pad = pad - start_pad
    #audio_start = max(0, piece.start - pad)
    audio_start = piece.start + .05
    audio_end = piece.end + end_pad

    guess = guess_speaker(SubSegment.slice_audio(audio, audio_start, audio_end), classifier)
    if guess == NO_GUESS and previous is not None: guess = previous

    return (guess, audio_start, audio_end)

def _is_episode(episode: str, word: str) -> bool:
    if episode: return False
    w = (word or "").strip().lower().translate(str.maketrans('','',string.punctuation))
    return (w == "episode")

def _word_audio_too_short(word: Word) -> bool:
    return word.end - word.start < TOO_SHORT_TO_BE_REAL_SECONDS
    
def _expand_last_seg(subseg:Union[SubSegment,None], word: Word):
    if subseg is not None: subseg.end = word.end

def _add_word_to_subseg(id_base: str, audio: AudioType, subseg:Union[SubSegment, None], sub_id: int, word: Word):
    if subseg is None:
        sub_id += 1
        subseg = SubSegment(
            id=f'{id_base}.{sub_id}', 
            start=word.start,
            audio=audio
    )
    subseg.add_word(word)

    return (subseg, sub_id)
    

def _append_segment(start_seg:SubSegment, segments:list[SubSegment], seg_to_append:SubSegment, last_segment:SubSegment, is_clips: bool, kill_anomalies: bool = False):
        # There are times when whisper's word timings aren't accurate.
        # There is a package called stable-ts, which I am importing,
        # that is supposed to make it better but it's still a little
        # crappy...  So, although it feels janky, I'm adding some special case
        # stuff to do a little cleanup.
        # - If there is a gap between the last sentence and this one,
        #   we'll assign part of the gap to the end of the previous sentence
        #   and part to this one.
        # - If the text is identical and the duration is less than a second,
        #   just bump the last guy and drop the segment.

        if kill_anomalies and seg_to_append and seg_to_append.is_anomaly():
            logging.info(f"IGNORING ANOMALY: '{str(seg_to_append)}' - anom: '{seg_to_append.anomaly_score()}'")
            return

        prev_subseg = segments[-1] if segments else last_segment
        trunky = 100
        if (prev_subseg and
            seg_to_append.text == prev_subseg.text and
            prev_subseg.end  - seg_to_append.end < 1):
            # If the text is the same and the audio is very short,
            # just bump the end and move on.
            prev_subseg.end = seg_to_append.end
            return

        # If the very first segment doesn't start at 0, we'll add a buffer
        # segment to account for all the space.
        if not prev_subseg and seg_to_append.start > 0 and not is_clips:
            new_start_end = round(seg_to_append.start, 2)
            start_seg.end = new_start_end
            seg_to_append.start = new_start_end
            segments.append(start_seg)
    
        if prev_subseg and prev_subseg.end != seg_to_append.start:
            # Truncate the left_over to two decimal places.
            diff = seg_to_append.start - prev_subseg.end
            end_pad = diff * .35
            beg_pad = diff - end_pad
            prev_subseg.end = math.floor((prev_subseg.end + end_pad) * trunky) / trunky
            seg_to_append.start = math.floor((seg_to_append.start - beg_pad) * trunky) / trunky

        segments.append(seg_to_append)

def _clean_up_episode(episode:str) -> str:

    episode = episode.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).strip()
    if episode:
        try:
            episode = str(w2n.word_to_num(episode))
        except ValueError:
            episode = ""

        logging.info(f'Detected episode: {episode}')

    return episode

def _should_skip_word(subseg:Union[SubSegment, None], word: Word) -> bool:
    if subseg is None: return False
    last_text = subseg.text.strip()
    duration = word.end - word.start
    if (_word_audio_too_short(word) and word.word.strip() == last_text) or \
        (duration < TOO_SHORT_TO_GUESS_SECONDS and word.probability < MIN_PROBABILITY_TO_INCLUDE):
        _expand_last_seg(subseg, word)
        return True
    
    return False

def _speakers_switched(audio:AudioType, classifier:Pipeline, word:Word, guess:SpeakerGuess, prev_speaker:str) -> tuple[bool, str, SpeakerGuess]:
    if not classifier: return (False, "", NO_GUESS)
    prev = guess
    guess, start, end = speaker_for_clip(audio, classifier, word, previous=prev)
    speaker = guess['label'] if guess['score'] > SWITCH_SPEAKER_CUTOFF else prev_speaker
    if not prev_speaker: prev_speaker = speaker            
    switched = speaker and speaker != prev_speaker
    return (switched, speaker, guess)

def _split_segment(
        audio: AudioType, 
        segment: Segment, 
        last_sub: Optional[SubSegment],
        punctuation:str=DEFAULT_PUNCTUATION,
        classifier:Optional[Pipeline] = None,
        is_clips: bool = False,
        capture_episode: bool = True,
        kill_anomalies: bool = False
    ) -> tuple[list[SubSegment], str, str]:

    start_segment = SubSegment.start_segment(audio)
    punc_tuples = tuple(list(punctuation))

    id_base = segment.id

    result = []
    sub_id = 0
    subseg = None
    episode = ""
    add_to_episode = capture_episode and last_sub and _is_episode(episode, last_sub.text.split(' ')[-1])
    guess = last_sub.speaker_guess if last_sub else None
    speaker = guess['label'] if guess else ""

    for word in segment.words:

        if _should_skip_word(subseg or last_sub, word): continue

        # If we've switched speakers, we're going to break the subsegment
        switched, speaker, guess = _speakers_switched(audio, classifier, word, guess, speaker)
        if subseg and switched:
            if subseg: _append_segment(start_segment, result, subseg, last_sub, is_clips, kill_anomalies=kill_anomalies)
            subseg = None
            add_to_episode = False

        if add_to_episode: episode += word.word

        subseg, sub_id = _add_word_to_subseg(id_base, audio, subseg, sub_id, word)
        add_to_episode = add_to_episode or _is_episode(episode, word.word)

        # If this is a series of clips, we're going to assume each one is a single utterance 
        # and we shouldn't try any fancy breaking.

        # But, occasionally, we'll get in a spot where for some reason, we're not given any
        # punctuation for a LOOONG time.  This makes for very difficult to manage
        # chucks of text.  So, if adding the word makes the segment longer than a
        # max length, we're just going to stop.
        if word.word.strip().endswith(punc_tuples) or subseg.duration >= MAX_SEGMENT_LENGTH:
            _append_segment(start_segment, result, subseg, last_sub, is_clips, kill_anomalies=kill_anomalies)
            subseg = None
            add_to_episode = False

    # [end for word in segment.words]
        
    # If we've got a leftover subseg, we'll add it now.
    if subseg:
        _append_segment(start_segment, result, subseg, last_sub, is_clips, kill_anomalies=kill_anomalies)

    episode = _clean_up_episode(episode)

    return (result, episode)

# [end def split_segment]


def guess_speaker(audio: AudioType, classifier: Pipeline) -> SpeakerGuess:
    # If we have too short a sample, we'll just return NO_GUESS
    raw, sr = audio
    if len(raw) < (sr * TOO_SHORT_TO_GUESS_SECONDS): return NO_GUESS
    return classifier({"sampling_rate": sr, "raw": raw}, top_k=1)[0]

def word_count(txt:str, punc=DEFAULT_PUNCTUATION + ",") -> int:
    trans = str.maketrans(str.maketrans('','',punc))
    words = [ w for w in jieba.lcut(txt.replace("'","")) if len(w.strip().translate(trans)) ]
    return len(words)

def get_segments(
        transcript: TranscriptionType,
        audio:AudioType,
        episode:str="",
        on_seg:callable=default_on_seg,
        is_clips:bool = False,
        ignore_shorter_than:float = 0,
        kill_anomalies: bool = True) -> tuple[str, list[SubSegment]]:

    model = "./transcribe-monkey"
    classifier: Pipeline = pipeline("audio-classification", model=model, device=DEVICE)

    segments, info = transcript
    duration = info.duration

    flat_subs = []
    quit_looping = False

    for segment in segments:

        if quit_looping: break

        duration = segment.end - segment.start
        if flat_subs and (duration < ignore_shorter_than or segment.text.strip() in hallucinations):
            logging.debug(f"IGNORING TOO SHORT: {segment.text} ({duration})")
            flat_subs[-1].end = segment.end
            continue

        subs, potential_episode = _split_segment(
            audio, segment, flat_subs[-1] if flat_subs else None,
            classifier=classifier, is_clips=is_clips, capture_episode=not episode,
            kill_anomalies=kill_anomalies
        )
        if not episode and potential_episode: episode = potential_episode

        cnt = 0
        sub_len = len(subs)
        for subseg in subs:

            if quit_looping: break

            cnt += 1

            ### If there is too little audio in the sample, we're just going to ignore attempting
            ### set the speaker
            if len(subseg.text.strip()) and subseg.duration >= TOO_SHORT_TO_GUESS_SECONDS:
                guess, _, _ = speaker_for_clip(audio, classifier, subseg)
                subseg.set_speaker(guess)
            else:
                subseg.speaker = ""
                subseg.speaker_confidence = 0
                subseg.text = "<Intro Music>" if not subseg.text.strip() and subseg.start == 0.0 else subseg.text

            # We're only going to collapse the speaker segments when
            # we're pretty sure that the speaker assignment was accurate.
            # We're deciding that "pretty sure" is the SPEAKER_COLLAPSE_CERTAINTY_CUTOFF

            if subseg.speaker_confidence < SPEAKER_COLLAPSE_CERTAINTY_CUTOFF:
                subseg.speaker = ""

            # don't collapse speakers if the spoken text is an anomaly
            if add_subsegment(flat_subs, subseg, collapse_speaker=not subseg.is_anomaly()):
                # a new segment got added so the previous segment can be finalized and
                # logged (if there is a previous one.  len - 2 is the index)
                if len(flat_subs) > 1:
                    sub = flat_subs[-2]
                    sub.finalize()
                    logging.info(sub)

            quit_looping = on_seg(f'Added segment {subseg.id}', cnt, sub_len, subseg.end / (duration if duration else 1))
        
        # [end for subseg in subs]

    # [ end for segment in segments]

    if flat_subs:
        # catch the last segment which won't have been printed or finalized
        flat_subs[-1].finalize()
        logging.info(flat_subs[-1])

    return (episode, flat_subs)

def audio_from_file(source:str, sampling_rate:int = None):
    # If we don't get a sampling rate, use whatever faster whisper
    # defaults to.
    kwargs = {'sampling_rate': sampling_rate} if sampling_rate is not None else {}
    # This sucks a little bit, but we just "know" that the default sampling rate
    # is 16000 because we looked at the decode_audio function.  So, if we
    # get "None" as a rate, we just return 16000
    aud = faster_whisper.decode_audio(source, **kwargs)
    sr = sampling_rate if sampling_rate else 16000
    return (aud, sr)

def save(
        title:str, 
        episode: str,
        vid:str, 
        audio_filename:str, 
        audio_folder:str, 
        json_destination: str,
        save_json:bool, 
        save_audio:bool, 
        audio: AudioType, 
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
        transcript["episode"] = episode
        transcript["YouTubeID"] = vid
        transcript["AudioFile"] = audio_filename
        transcript["transcription"] = []

        # There can be only one selected.  If for some reason
        # we see more than one, we'll fix it and warn.
        seen_selected = False
        for seg in segments:
            values:SegmentDict = dict(seg) if type(seg) == dict else seg.to_dict()
            if values['selected'] and seen_selected:
                logging.warning(f'More than one selected row at {len(transcript["transcription"])}')
                values['selected'] = False
            values.pop('size', None)
            seen_selected = bool(seen_selected or values['selected'])
            transcript["transcription"].append(values)
            if save_audio and values['export'] and values['speaker'] is not None and values['speaker'].strip() :
                segment_file = SubSegment.static_file_dest(vid, values['row_id'], path=dest(values['speaker']))
                SubSegment.save_audio_slice(segment_file, audio, values['modified_start'], values['modified_end'])

        with open(json_destination, 'w', encoding='utf8') as stream:
            json.dump(transcript, stream, indent = 2, ensure_ascii=False)
            stream.write("\n")

def duration_to_hours_minutes_seconds(dur):
    seconds_in_day = 24 * 60 * 60
    dur_pieces = divmod(dur.days * seconds_in_day + dur.seconds, 60)
    seconds = dur_pieces[1]
    dur_pieces = divmod(dur_pieces[0], 60)
    hours = dur_pieces[0]
    minutes = dur_pieces[1]
    return (hours, minutes, seconds)

def get_hf_token() -> str:
    token = os.getenv('HF_TOKEN', None)
    if token is None:
        path = os.path.expanduser('~/.cache/huggingface/token')
        if os.path.isfile(path):
            with open(path) as f:
                token = f.read()
    return token


def whisper_transcribe(
        audio_file:str, 
        model_name:str=MODEL_NAME,
        device:str=DEVICE, 
        compute_type:str=COMPUTE_TYPE, 
        language:str="en",
        on_seg:Optional[callable] = default_on_seg,
        clips: list[float] = None) -> Union[None, tuple[TranscriptionType, AudioType]]:

    sr = whisper.audio.SAMPLE_RATE

    model = whisper.load_model(model_name, device=device, download_root=WHISPER_MODEL_DIR)
    audio_waveform, _ = audio_from_file(audio_file, sampling_rate=sr)

    if on_seg(f'{model_name} loaded. Using whisper', 0, 0, 0): return None

    fp16 = False if device == "cpu" else True
    kwargs = dict(DEFAULT_TRANSCRIBE_KWARGS)
    kwargs['language'] = language
    kwargs['fp16'] = fp16
    if clips is not None: kwargs['clip_timestamps'] = clips

    results = whisper.transcribe(
        model=model, 
        audio=audio_waveform, 
        **kwargs
    )

    info = TranscriptionInfo(
        language=language, 
        language_probability=1,
        duration=len(audio_waveform) / sr, 
        duration_after_vad=len(audio_waveform) / sr, 
        all_language_probs=[], 
        transcription_options=kwargs,
        vad_options=None        
    )

    res = [WhisperSegment(seg) for seg in results['segments']]
    return ((res, info), (audio_waveform, sr))


def pyannote_transcribe(audio_file:str, use_whisper:bool = False, 
    model_name:str=MODEL_NAME,
    device:str=DEVICE, 
    language:str = "en"):

    from pyannote.audio import Pipeline as PyAnnotePipeline
    pipe = PyAnnotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=get_hf_token()
    )

    pipe.to(torch.device(device))

    # apply pretrained pipeline
    pyannote_sample_rate = 44100

    audio, _ = audio_from_file(audio_file, pyannote_sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    mapping = {"waveform": waveform, "sample_rate": pyannote_sample_rate}
    diarization = pipe(mapping)
   
    model = "./transcribe-monkey"
    classifier: Pipeline = pipeline("audio-classification", model=model, device=DEVICE)
    whisper_audio = audio_from_file(audio_file, whisper.audio.SAMPLE_RATE)
    audio_data, sr = whisper_audio

    kwargs = dict(DEFAULT_TRANSCRIBE_KWARGS)
    kwargs['language'] = language
    info = TranscriptionInfo(
        language=language, 
        language_probability=1,
        duration=len(audio_data) / sr, 
        duration_after_vad=len(audio_data) / sr, 
        all_language_probs=[], 
        transcription_options=kwargs,
        vad_options=None        
    )

    if use_whisper:
        model = whisper.load_model(model_name, device=device, download_root=WHISPER_MODEL_DIR)
        fp16 = False if device == "cpu" else True
        kwargs['fp16'] = fp16
        def t(start, end):
            if start > -1 and end > -1: kwargs['clip_timestamps'] = [start, end]
            results = whisper.transcribe(model=model, audio=audio_data, **kwargs)
            segs = [WhisperSegment(seg) for seg in results['segments']]
            return (segs, info)
    else:
        model = stable_whisper.load_faster_whisper(
            model_name, device=device, compute_type=COMPUTE_TYPE
        )
        kwargs['hotwords'] = "Ula"
        def t(start, end):
            if start > -1 and end > -1: kwargs['clip_timestamps'] = [start, end]
            kwargs['clip_timestamps'] = [start, end]
            return (model.transcribe_stable(audio_data, **kwargs), info)


    # At the end of every transciption, there is the chance of a hallucination. 
    # If we try to do the transcription one segment at a time, we'll end up with 
    # TONS of hallucinations.  So, we'll try transcribing the entire thing in one
    # pass and then assigning to speakers/segments based on the segmentation that
    # pyannote does.  Note, this weird "conditionally create a transcript function/closure
    # then call it" pattern is left over from when I was trying to call it segment by
    # segment.
    transcript, _ = t(-1, -1)

    # print the result
    g = None
    def squish(transcript):
        new_txt = ""
        for row in transcript: new_txt += (" " + HanziConv.toSimplified(row.text.strip()))
        return new_txt.strip()

    episode = ""
    for turn, track, speaker in diarization.itertracks(yield_label=True):
        last_end = turn.end
        w = Word(turn.start, turn.end, "X", probability=0.0)
        g, _, _ = speaker_for_clip(whisper_audio, classifier, w, g)
        label = g['label'].capitalize()
        transcript = t(turn.start, turn.end)
        print(f"track: {track} {turn.start} - {turn.end} speaker_{speaker} ({label})")
        episode, flat_subs = get_segments(transcript, whisper_audio, episode=episode, is_clips=True)
        print(squish(flat_subs))
        print('')

    # clear gpu vram
    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':

    args = get_arguments()

    logging.basicConfig(level=args.log_level.upper())

    if args.use_pyannote:
        audio_file = audio_filename(args.filename, args.video_id)
        pyannote_transcribe(audio_file, args.use_whisper, language=args.lang)
        quit()

    if args.do_plot:
        plot_spectogram_from_file(args.filename, args.video_id, args.transcript_json, args.clips)
        quit()

    if args.word_doc:
        saved_doc = create_word_doc(args.transcript_json)
        if saved_doc:
            print(f'Transcript saved as {saved_doc}')
        else:
            logging.error('No document created.')
        quit()

    audio_file, _, yt = get_youtube_audio(args.video_id, is_short=args.is_a_short, filename=args.filename)

    logging.info(f'Transcribing: "{yt.title}"')

    start_trans = datetime.datetime.now()

    if args.use_whisper:
        transcript, audio = whisper_transcribe(audio_file, language=args.lang, clips=args.clips)
    else:
        transcript, audio = transcribe(audio_file, language=args.lang, clips=args.clips)
    
    episode, flat_subs = get_segments(
        transcript,
        audio,
        episode=args.episode,
        is_clips=(args.clips is not None and args.clips),
        kill_anomalies=args.kill_anomalies
    )

    json_path = args.transcript_json
    save_json = bool(json_path and json_path.lower() != 'false')

    json_dest = transcript_json_name(json_path, args.video_id)
    should_save_segments = bool(args.audio_folder)
    save(
        yt.title,
        episode,
        args.video_id,
        audio_file,
        args.audio_folder, 
        json_dest, 
        save_json, should_save_segments, audio, flat_subs
    )
    
    if not should_save_segments and not save_json:
        for seg in flat_subs:
            print(seg)
    
    end_trans = datetime.datetime.now()
    diff = end_trans - start_trans
    hours, minutes, seconds = duration_to_hours_minutes_seconds(diff)
    logging.info(f"Transcription finished in {hours:02}:{minutes:02}:{seconds:02}")