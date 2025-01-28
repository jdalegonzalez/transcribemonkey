#! python
import os
import json
import math
import logging
import html
import datetime
import string
import sys
from enum import Enum

from word2number import w2n

# EXCEPTIONALLY janky but translator uses a very old httpx.
# If I need to fix it, the fix is to monkey patch httpx to
# be backward compatible with googletrans.  What I've
# currently done is just live with an old httpcore.
#import httpcore
#setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')

from googletrans import Translator

from typing import Optional, Union

from transformers import pipeline, Pipeline

import stable_whisper
import faster_whisper
from faster_whisper.transcribe import Segment, Word, TranscriptionInfo
import whisper

import torch

from pytubefix import YouTube

from docxtpl import DocxTemplate

from subsegment import SubSegment
from transcribe_args import get_arguments
from utils import (
    AudioType,
    DEFAULT_PUNCTUATION,
    HALLUCINATIONS,
    is_a_number,
    PropertyDict,
    SegmentDict,
    SpeakerGuess,
    to_minutes_seconds,
    TranscriptionType,
    WhisperSegment
)

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

translator = Translator()

prompt_str = \
"好大家好. 我是 Ula. 我说中文。 "\
"[Laughter] 好啊！And I'm Tom and I speak both 英文 and 中文。 " \
"Our podcast is called Mandarin Monkey. You can find us at mandarinmonkey.com. " \
"Uh, well, [ha ha ha] yeah.  This is episode 381!  第三百八十一集！" \
"I'm 很开心啊。 I'm very happy too.  Yeah.  We went to Pier Thirty-Nine, just to eat something." \
"Ula is a wonderful host and my name is Tom.  I'm a co-host."

## Things I could maybe tune...
## beam_size: 1,
## best_of: 10
## initial_prompt: prompt_str
## vad_filter: 
## suppress_blank: False

DEFAULT_TRANSCRIBE_KWARGS = {
    'language': 'en',
    'task': 'transcribe',
    'beam_size': 5,
    'best_of': 5,
    'initial_prompt': prompt_str,
    'suppress_blank': False,
    'condition_on_previous_text': False,
    'suppress_tokens': [-1],
    'without_timestamps': True,
    'word_timestamps': True
}


#####################

class TranscribeMethods(Enum):
    FASTER_WHISPER = 'faster-whisper'
    STABLE_WHISPER = 'stable-whisper'
    HUGGING_FACE = 'hugging_face-whisper'

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

def translate(txt:str, src:str=None, dest:str=None) -> str:
    if src is None: src = 'zh-cn' if contains_chinese(txt) else 'en'
    if dest is None: dest = 'en' if src == 'zh-cn' else 'zh-cn'
    result = translator.translate(txt, src=src, dest=dest)
    return result.text

class FakeYouTube():
    def __init__(self, title:str = 'Fake Title'):
        self.title = title

def cached_youtube(video_id: str, filename: str, transcript_path: str) -> Union[None, tuple[str, None, FakeYouTube]]:
    if not transcript_path: transcript_path = "./transcripts/"
    transcript_json = transcript_json_name(transcript_path, video_id)
    if not filename: filename = "./audio_out/"
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

def print_json_data(json_data):
    class Bcolors:
        MAGENTA = '\033[95m'
        WHITE = '\033[97m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        CYAN = '\033[96m'
        WARNING = '\033[93m'
        ENDC = '\033[0m'
        FAIL = '\033[91m'
        BOLD = '\033[1m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'

    e = f'{Bcolors.ENDC}'
    b = Bcolors.BOLD
    ft = f'{b}{Bcolors.WHITE}'
    fs = f'{Bcolors.OKGREEN}'
    fe = f'{Bcolors.WARNING}{Bcolors.ITALIC}'

    def get_start_end(data, field):
        # Check the old format...
        modified = data.get('modified', None)
        source = modified if modified else data
        key = field if modified else f'modified_{field}'
        return source[key]
    
    if json_data is not None:
        episode = json_data.get('episode', "")
        title = json_data.get('title')
        print(f'{ft}{title}{e}')
        if episode: print(f'{fe}Episode: {episode}{e}')
        for line in json_data['transcription']:
            start = to_minutes_seconds(get_start_end(line, 'start'))
            end = to_minutes_seconds(get_start_end(line, 'end'))
            fp = Bcolors.BOLD + (Bcolors.CYAN if line['speaker'] == 'Tom' else Bcolors.MAGENTA)
            print(f'{fs}{start} - {end}{e} {fp}{line['speaker']}{e}: {line['text']}')

def print_text_transcript(filename):
    json_data = load_transcript_from_file(filename)
    print_json_data(json_data)    

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
        if flat_subs and (duration < ignore_shorter_than or segment.text.strip() in HALLUCINATIONS):
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

def json_representation(
    title: str, 
    episode: str, 
    vid: str,
    audio_filename:str, 
    segments:Union[SegmentDict,SubSegment],
    save_audio_func: Optional[callable] = None,
) -> dict:

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
        if save_audio_func: save_audio_func(values)

    return transcript

def write_json(dest, obj):
    with open(dest, 'w', encoding='utf8') as stream:
        json.dump(obj, stream, indent = 2, ensure_ascii=False, default=SubSegment.default)
        stream.write("\n")

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
    segments:Union[SegmentDict, SubSegment]
) -> None:

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

    def save_func(values):
        if save_audio and values['export'] and values['speaker'] is not None and values['speaker'].strip() :
            segment_file = SubSegment.static_file_dest(vid, values['row_id'], path=dest(values['speaker']))
            SubSegment.save_audio_slice(segment_file, audio, values['modified_start'], values['modified_end'])

    sf = save_func if save_audio else None

    transcript = json_representation(title, episode, vid, audio_filename, segments, save_audio_func=sf)
    write_json(json_destination, transcript)

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


def words_for_pyannote_segments(
    model,
    audio, 
    segments,
    model_name: str = MODEL_NAME, 
    device: str = DEVICE, 
    use_whisper:bool = False, 
    language: str = "en"):

    audio_data, sr = audio
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

    timestamps = [segments[0].start]
    # go through the segments building 30 sec (ish) clips
    if len(segments) > 1:
        for seg in segments:
            last_stamp = timestamps[-1]
            if seg.end - last_stamp > 30: 
                timestamps.append(seg.start)
                timestamps.append(seg.start)    

    timestamps.append(segments[-1].end)

    kwargs['clip_timestamps'] = timestamps
    if use_whisper:
        model = whisper.load_model(
            model_name, device=device, download_root=WHISPER_MODEL_DIR
        )
        kwargs['fp16'] = False if device == "cpu" else True
        transcription = whisper.transcribe(model=model, audio=audio_data, **kwargs)
        segs = [WhisperSegment(seg) for seg in transcription['segments']]
        result = (segs, info)
    else:
        model = stable_whisper.load_faster_whisper(
            model_name, device=device, compute_type=COMPUTE_TYPE
        )
        transcription = model.transcribe_stable(audio_data, **kwargs)
        result = (transcription, info)

    segments, _ = result
    bag_of_words = []

    def to_tdict(word):
        return PropertyDict({
            'word': word.word,
            'start': word.start,
            'end': word.end,
            'probability': word.probability
        })
    
    def likely_fake(word):
        return word.end - word.start < .22 or word.probability < .6

    def words_to_add(words):
        # We're going to do two things.  First, 
        # we're going to convert our list of WhisperWords into
        # our common format - which is just a property dict
        # Second, we're going to trim any likely hallucinations off 
        # the back.
        word_list = [ to_tdict(w) for w in words ]
        while word_list and likely_fake(word_list[-1]): word_list.pop()
        return word_list

    for seg in segments: bag_of_words += words_to_add(seg.words)

    return bag_of_words

def pyannote_collapse_convert_segments(audio, segments:list[WhisperSegment]) -> list[SubSegment]:

    results = []
    for ndx, seg in enumerate(segments):
        # Empty segments at this point need to be collapsed.  We'll just add their time 
        # to the previous next segment (or dump if it's the last)
        seg.id = ndx
        last_seg = results[-1] if results else None
        if not last_seg:
            results.append(SubSegment.from_whisper_segment(audio, seg))
            continue
        if not last_seg.text:
            # We added an empty segment, so collapse this one, into the 
            # previous one.
            last_seg.speaker = seg.speaker
            last_seg.text = seg.text
            last_seg.end = seg.end
        elif last_seg.text and not seg.text:
            # This segment doesn't have words, so collapse it into
            # the previous.
            last_seg.end = seg.end
        elif last_seg.speaker == seg.speaker:
            # Both segments have text but they're also the same speaker
            # so collapse them
            last_seg.merge(seg)
        else:
            results.append(SubSegment.from_whisper_segment(audio, seg))

    return results

def pyannote_transcribe(audio_file:str, use_whisper:bool = False, 
    model_name:str=MODEL_NAME,
    device:str=DEVICE, 
    language:str = "en",
    episode: str = ""):

    checkpoints = True # should we save as json the data we've got at various points.

    # apply pretrained pipeline
    pyannote_sample_rate = 44100
    from pyannote.audio import Pipeline as PyAnnotePipeline
    pipe = PyAnnotePipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=get_hf_token()
    )

    pipe.to(torch.device(device))

    audio, _ = audio_from_file(audio_file, pyannote_sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    mapping = {"waveform": waveform, "sample_rate": pyannote_sample_rate}
    diarization = pipe(mapping)
   
    model = "./transcribe-monkey"
    classifier: Pipeline = pipeline("audio-classification", model=model, device=DEVICE)

    g = None
    segments = []
    def skip_segment(previous, current) -> bool:
        # if this segment is completely contained in the previous one
        # and the speakers are the same, we're going to dump it.
        if not previous: return False
        if previous.speaker != current.speaker: return False
        return current.start >= previous.start and current.end <= previous.end
    
    def append_segment(current, segments):
        if segments and segments[-1].speaker == current.speaker:
            segments[-1].end = current.end
        else:
            segments.append(current)

        return segments[-1]

    whisper_audio = audio_from_file(audio_file, whisper.audio.SAMPLE_RATE)

    for turn, _, _ in diarization.itertracks(yield_label=True):
        w = Word(turn.start, turn.end, "X", probability=0.0)
        g, _, _ = speaker_for_clip(whisper_audio, classifier, w, g)
        label = g['label'].capitalize()
        segment = WhisperSegment({'start': turn.start, 'end': turn.end, 'speaker': label, 'speaker_confidence': g['score'], 'words': []})
        if segments and skip_segment(segments[-1],segment): continue
        seg = append_segment(segment, segments)

    # If we got here and somehow don't have segments, bail.
    if not segments: return
    if checkpoints: write_json('./initial_segments.json', segments)

    if use_whisper:
        model = whisper.load_model(
            model_name, device=device, download_root=WHISPER_MODEL_DIR
        )
    else:
        model = stable_whisper.load_faster_whisper(
            model_name, device=device, compute_type=COMPUTE_TYPE
        )

    bag_of_words = words_for_pyannote_segments(model, whisper_audio, segments, model_name, device, use_whisper, language)
    if checkpoints: write_json('./bag_of_words.json',bag_of_words)

    # For each segment, grap the words that go in the segment.  If
    # we find a word that stradles the end of the segment, we'll
    # assign it to the next one.
    def should_add_to_seg(word, seg):
        return word.start <= seg.end and word.end <= seg.end

    def add_with_boundary_fixup(ndx, segments, segment, word, fix_boundary):
        # If the first word starts with a continuation character, and
        # we've got a previous segment, we're going to assume that we
        # misapplied the previous word - which sometimes happens on a segment
        # boundary.
        if fix_boundary and ndx and not segment.words and word.word.startswith(','):
            last_seg = segments[ndx - 1]
            last_word = last_seg.words.pop()
            segment.words.append(last_word)

        segment.words.append(word)

    def put_words_in_segments(words, segs, fix_boundary:bool = True):
        episode = ""
        add_to_episode = False
        for ndx, seg in enumerate(segs):
            seg.words = []
            while words and should_add_to_seg(words[0], seg):
                word = words.pop(0)
                add_to_episode = add_to_episode and is_a_number(word)
                if add_to_episode: episode += word
                add_to_episode = add_to_episode or word.word.strip().lower() == "episode"
                add_with_boundary_fixup(ndx, segments, seg, word, fix_boundary)
        return episode
    
    possible_episode = put_words_in_segments(bag_of_words, segments)
    if not episode: episode = possible_episode

    # Theoretically, the bag of words should be empty now.  If it isn't, just
    # add them all to the end.
    for word in bag_of_words: segments[-1].words.append(word)
    if checkpoints: write_json('./segs_with_words.json', segments)
    for seg in segments:
        if seg.speaker == "Ula" and not seg.contains_chinese():
            chinese_bag = words_for_pyannote_segments(model, whisper_audio, [seg], model_name, device, use_whisper, language="zh")
            put_words_in_segments(chinese_bag, [seg], fix_boundary=False)
    if checkpoints: write_json('./segs_with_words_2pass.json', segments)

    # clear gpu vram
    del model
    torch.cuda.empty_cache()

    results = pyannote_collapse_convert_segments(whisper_audio, segments)
    if checkpoints: write_json('./segs_collapsed.json', results)

    return (episode, results, whisper_audio)

def unsegmented_transcribe(audio_file, use_whisper: bool = False, language: str = "en", episode: str = "", kill_anomalies: bool = False, clips: list[float] = None):
    tfunc = whisper_transcribe if use_whisper else transcribe
    transcript, audio = tfunc(audio_file, language=language, clips=clips)
    episode, flat_subs = get_segments(
        transcript,
        audio,
        episode=episode,
        is_clips=(clips is not None and clips),
        kill_anomalies=kill_anomalies
    )

    return (episode, flat_subs, audio)

if __name__ == '__main__':

    args = get_arguments()

    logging.basicConfig(level=args.log_level.upper())

    if args.do_plot:
        plot_spectogram_from_file(args.filename, args.video_id, args.transcript_json, args.clips)
        quit()

    if args.print_text:
        print_text_transcript(args.transcript_json)
        quit()

    if args.word_doc:
        saved_doc = create_word_doc(args.transcript_json)
        if saved_doc:
            print(f'Transcript saved as {saved_doc}')
        else:
            logging.error('No document created.')
        quit()

    json_path = args.transcript_json
    save_json = bool(json_path and json_path.lower() != 'false')

    audio_file, _, yt = get_youtube_audio(
        args.video_id,
        is_short=args.is_a_short,
        filename=args.filename,
        transcript_path=json_path
    )

    logging.info(f'Transcribing: "{yt.title}"')

    start_trans = datetime.datetime.now()

    if args.use_pyannote:
        episode, flat_subs, audio = pyannote_transcribe(
            audio_file, args.use_whisper, language=args.lang, episode=args.episode)
    else:
        episode, flat_subs, audio = unsegmented_transcribe(
            audio_file, args.use_whisper, language=args.lang, episode=args.episode,
            kill_anomalies=args.kill_anomalies, clips=args.clips)

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
        data = json_representation(yt.title, episode, args.video_id, audio_file, flat_subs)
        print_json_data(data)
    
    end_trans = datetime.datetime.now()
    diff = end_trans - start_trans
    hours, minutes, seconds = duration_to_hours_minutes_seconds(diff)
    logging.info(f"Transcription finished in {hours:02}:{minutes:02}:{seconds:02}")