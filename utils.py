import re
import numpy as np
import logging
from typing import Optional, TypedDict, Union, Iterable
from collections import UserDict

from faster_whisper.transcribe import Segment, Word, TranscriptionInfo

import jieba
from xpinyin import Pinyin


# There is no real reason to put this 'priming' request here.
# I'm doing it because jieba logs some stuff on the first 
# request you make to it and I want it to be before all the
# stuff I log.
jieba.lcut("好大家好")
pinyin_obj = Pinyin()

# punctuation:str="\"'.。,，!！?？:：”)]}、"
DEFAULT_PUNCTUATION = "\"'.。!！?？:：”)]}、" # no comma

# These are intentionally repetitive because I'm
# trying to make sure I don't kill something by
# accident so I want long, specific strings.
# Maybe later I'll look at patters for some of
# them.
HALLUCINATIONS = [
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

AudioType = tuple[np.ndarray, float]
TranscriptionType = tuple[Iterable[Segment], TranscriptionInfo]

chinese_characters_range = '\u3002-\u9fff'
chinese_character_re = re.compile(f'[{chinese_characters_range}]')
def contains_chinese(text:str) -> bool:
    return chinese_character_re.search(text) is not None

def combine_words(words:list[str]):
    was_chinese = False
    text = ""
    pinyin = ""
    pinyin_pfx = ""
    punc = DEFAULT_PUNCTUATION + ","
    ended_with_punc = False
    ended_with_space = False
    for word in words:
        if not text:
            text += word
            continue
        is_chinese = contains_chinese(word)
        starts_with_punc = word[0] in punc
        prefix = " " if \
            word != ' ' \
            and not (starts_with_punc or ended_with_space) \
            and (ended_with_punc or was_chinese != is_chinese) \
            else ""
        text += (prefix + word)
        was_chinese = is_chinese
        ended_with_punc = text[-1] in punc
        ended_with_space = text[-1] == " "

        pinyin_pfx = " " if pinyin and word not in punc else ""
        pinyin += (pinyin_pfx + pinyin_obj.get_pinyin(word, '', tone_marks="marks")) if is_chinese \
            else (prefix + word)

    return (text,pinyin)

def separate_english(text:str, left_trim=True) -> tuple[str, str]:
    sep = text
    pinyin = ""
    sep, pinyin = combine_words(jieba.lcut(text)) if contains_chinese(text) else (text, "")
    if left_trim: sep = sep.lstrip()
    return (sep, pinyin)

def to_minutes_seconds(seconds:float) -> str:
    secs = round(seconds)
    minute_part = secs // 60
    second_part = secs % 60
    return f'{minute_part:02}:{second_part:02}'

def word_count(txt:str, punc=DEFAULT_PUNCTUATION + ",") -> int:
    trans = str.maketrans(str.maketrans('','',punc))
    words = [ w for w in jieba.lcut(txt.replace("'","")) if len(w.strip().translate(trans)) ]
    return len(words)

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

    def __getstate__(self):
        return self.data

    def __setstate__(self, state):
        self.data = state

    def default(o):
        if issubclass(type(o),PropertyDict): return o.data
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

class WhisperSegment(PropertyDict):

    def __init__(self, dict=None, /, **kwargs):
        super().__init__(dict, kwargs=kwargs)
        self._words = None
        self._text = None

    def __getattr__(self, key):
        if key == 'words':
            if not self._words: self._words = [WhisperSegment(w) for w in self.data['words']]
            return self._words
        elif key == 'text':
            if not self._text: self._text, _ = separate_english("".join([w.word for w in self.words]))
            return self._text
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if key == 'words':
            self._text = None
            self._words = [WhisperSegment(w) for w in value]
        else:
            super().__setattr__(key, value)

    def contains_chinese(self):
        if not self._text: return contains_chinese("".join([w.word for w in self.words]))
        return contains_chinese(self._text)

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

