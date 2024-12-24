#! python
import argparse
import os
import sys
import json
import math
import logging
import html
import datetime
from enum import Enum

from word2number import w2n

from typing import Optional, TypedDict, Union, Iterable

from hanziconv import HanziConv
import jieba

import numpy as np
from transformers import pipeline

import stable_whisper
import faster_whisper
from faster_whisper.transcribe import Segment, TranscriptionInfo

import torch

import soundfile as sf

from pytubefix import YouTube
from pytubefix.cli import on_progress

from docxtpl import DocxTemplate

class TranscribeMethods(Enum):
    FASTER_WHISPER = 'faster-whisper'
    STABLE_WHISPER = 'stable-whisper'
    HUGGING_FACE = 'hugging_face-whisper'

##### Constants #####
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MODEL_NAME = "large-v3-turbo"
SAMPLING_RATE = 16000 # 22050 #

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

def to_minutes_seconds(seconds:float) -> str:
    secs = round(seconds)
    minute_part = secs // 60
    second_part = secs % 60
    return f'{minute_part:02}:{second_part:02}'

AudioType = tuple[np.ndarray, float]
TranscriptionType = tuple[Iterable[Segment], TranscriptionInfo]

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
        audio: AudioType,
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
        self.audio: AudioType = audio
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
        result['text'] = HanziConv.toSimplified(self.text.strip())

        result['selectable'] = self.selectable
        result['selected'] = self.selected
        result['export'] = self.export

        return result

    def set_speaker(self, guess:SpeakerGuess):
        self.speaker = guess['label'].capitalize()
        self.speaker_confidence = guess['score']

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

    def slice(self) -> AudioType:
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
    
    def save_audio_slice(dest: str, audio:AudioType, start: float, end: float) -> None:
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
            speak_str = f'{self.speaker.strip()} ({round(self.speaker_confidence, 4):.4f}): '
        return speak_str
    
    def __repr__(self):
        return f'[id: {self.id}, start: {self.start}, end: {self.end}, text: "{self.text}", speaker: "{self.speaker_string()}"]'

    def __str__(self):
        return f'[{to_minutes_seconds(self.start)}] {self.speaker_string()}{self.text.strip()}'
    
def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video"
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
        help="The root folder to put the individual audio segments.  If missing, audio segments will not be generated.",
        dest="audio_folder",
        default=""
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

    return args

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

def transcribe(
        audio_file:str, 
        model_name:str=MODEL_NAME,
        device:str=DEVICE, 
        compute_type:str=COMPUTE_TYPE, 
        suppress_numerals:bool=True,
        language:str="en") -> tuple[TranscriptionType, AudioType]:

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

    prompt_str = \
    "This is a dialog between Tom and Ula. " \
    "Tom speaks English. " \
    "Ula speaks Mandarin. " \
    "- 你好妈 - Uh... em... Yeah. I'm good. 你呢. " \
    "Produce a verbatim transcription."

    ## Things I could maybe tune...
    ## beam_size
    ## best_of
    ## hotwords
    ## initial_prompt
    ## vad_filter
    ## suppress_blank

    kwargs = {
        'language':           language,
        'without_timestamps': True,
        'word_timestamps':    True,
        'initial_prompt':     prompt_str,
        'beam_size': 1,
        'best_of': 10,
        'suppress_blank': False
    }

    # Hugging face version gets killed on my machine, so...
    # I wouldn't recommend using it.
    if method == TranscribeMethods.HUGGING_FACE:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(HF_MODELS[model_name])
        tokenizer = processor.tokenizer
        whisper_model = stable_whisper.load_hf_whisper(model_name, device)
        sr = whisper_model.sampling_rate
        kwargs.pop('without_timestamps')
        kwargs.pop('initial_prompt')
        kwargs.pop('beam_size')
        kwargs.pop('best_of')
        kwargs.pop('suppress_blank')
    else:    
        whisper_model = stable_whisper.load_faster_whisper(
            model_name, device=device, compute_type=compute_type
        )
        sr = whisper_model.feature_extractor.sampling_rate
        tokenizer = whisper_model.hf_tokenizer

    kwargs['suppress_tokens'] = (
        find_numeral_symbol_tokens(tokenizer)
        if suppress_numerals
        else [-1]
    )
    
    audio_waveform, _ = audio_from_file(audio_file, sampling_rate=sr)

    noun = ''

    transcription = None
    if method == TranscribeMethods.STABLE_WHISPER:
        # O5OjKjno9Pw
        # duration: 41:08
        # "large-v3" INFO:root:Transcription finished in 00:27:51
        _, info = whisper_model.transcribe(audio_waveform, **kwargs)
        results = whisper_model.transcribe_stable(audio_waveform, **kwargs)
        transcription = [results, info]
        noun = 'Transcribed'
    else:
        # O5OjKjno9Pw
        # duration: 41:08
        # "large-v3" INFO:root:Transcription finished in 00:31:15
        transcription = whisper_model.transcribe(audio_waveform, **kwargs)
        noun = 'Transcribing'

    # clear gpu vram
    del whisper_model
    torch.cuda.empty_cache()
    _, info = transcription
    logging.info(f'{noun} audio with duration: {to_minutes_seconds(info.duration)}')

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

    episode = ""
    add_to_episode = False
    for word in segment.words:
        if add_to_episode:
            episode += word.word

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

        add_to_episode = add_to_episode or (not episode and word.word.strip().lower() == "episode")
        if word.word.endswith(punc_tuples):
            do_append(subseg)
            subseg = None
            add_to_episode = False
    
    # If we've got a leftover subseg, we'll add it now.
    if subseg:
        do_append(subseg)
        subseg = None

    for c in punc_tuples:
        episode = episode.replace(c, ' ')

    episode = episode.strip()
    if episode:
        try:
            episode = str(w2n.word_to_num(episode))
        except ValueError:
            episode = ""

        logging.info(f'Detected episode: {episode}')

    return (result, episode)


def get_segments(
        video_id:str,
        transcript: TranscriptionType,
        audio:AudioType,
        on_seg:callable=None,
        episode:str="") -> tuple[str, list[SubSegment]]:

    model = "./trained-speakerbox"
    classifier = pipeline("audio-classification", model=model, device=DEVICE)

    segments, info = transcript
    duration = info.duration

    flat_subs = []
    quit_looping = False
    speaker_certainty_cutoff = .65

    for segment in segments:
        if quit_looping:
            break
        subs, potential_episode = split_segment(video_id, audio, segment, flat_subs[-1] if flat_subs else None)
        if not episode and potential_episode:
            episode = potential_episode
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
            # Also, there is a special case that happens when a sentence is VERY short.
            # Usually the speaker detector doesn't do a good job.
            # We also know that this sub ends in a stop character and our splitter sees
            # a stop character as a word.  So, "one word" is a length < 3 (1, or 2)
            is_one_word = len(jieba.lcut(subseg.text)) < 3
            add_subsegment(flat_subs, subseg, collapse_speaker=subseg.speaker_confidence >= speaker_certainty_cutoff and not is_one_word)

            if on_seg:
                quit_looping = on_seg(f'Added segment {subseg.id}', cnt, sub_len, subseg.end / duration)
                if quit_looping:
                    break

        if flat_subs:
            flat_subs[-1].text = flat_subs[-1].text.strip()

    return episode, flat_subs

def audio_from_file(source, sampling_rate = SAMPLING_RATE):
    aud = faster_whisper.decode_audio(source, sampling_rate=sampling_rate)
    return (aud, sampling_rate)

def save(
        title:str, 
        episode: str,
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
            values.pop('size')
            seen_selected = bool(seen_selected or values['selected'])
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

    transcript, audio = transcribe(audio_file)
    episode, flat_subs = get_segments(args.video_id, transcript, audio, args.episode)

    json_path = args.transcript_json
    save_json = bool(json_path and json_path.lower() != 'false')

    json_dest = json_path if json_path.endswith('.json') else os.path.join(json_path, f'{args.video_id}.transcript.json')
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
    seconds_in_day = 24 * 60 * 60
    diff_pieces = divmod(diff.days * seconds_in_day + diff.seconds, 60)
    seconds = diff_pieces[1]
    diff_pieces = divmod(diff_pieces[0], 60)
    hours = diff_pieces[0]
    minutes = diff_pieces[1]
    logging.info(f"Transcription finished in {hours:02}:{minutes:02}:{seconds:02}")