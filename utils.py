import re
import numpy as np
import logging
from typing import Iterable, Self, TypedDict, Union
from collections import UserDict

from faster_whisper.transcribe import Segment, Word, TranscriptionInfo

import jieba
from xpinyin import Pinyin

from word2number import w2n

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

class PropertyDict(UserDict):    
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
    def default(self,o):
        if issubclass(type(o),PropertyDict): return o.data
        raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

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
        ended_with_punc = text[-1] in punc and text[-1] != "'"
        ended_with_space = text[-1] == " "

        pinyin_pfx = " " if pinyin and word not in punc else ""
        pinyin += (pinyin_pfx + pinyin_obj.get_pinyin(word, '', tone_marks="marks")) if is_chinese \
            else (prefix + word)

    return (text,pinyin)

def is_a_number(val:str) -> bool:
    if not val: return False

    t = None

    try: t = float(val)
    except ValueError: t = None
    if t: return True

    try: t = w2n.word_to_num(val)
    except ValueError: t = None
    return t is not None

def split_sentence(text:str, drop_punc = False) -> tuple[str, int, str, int]:
    """
    Splits a sentence that contains english and chinese into the english and chinese parts.

    Args:
        text (str): The sentence to split.
        drop_punc (bool): If True, punctuation will be dropped from the english text.
    Returns:
        tuple[str, int, str, int]: A tuple containing the english text, word count,
                                   chinese text, and number of chinese characters.
    """
    words = jieba.lcut(text)
    english_text = ""
    chinese_text = ""
    english_count = 0
    chinese_count = 0
    punc = DEFAULT_PUNCTUATION + "，,!！."
    trans = str.maketrans(str.maketrans('','',punc))
    for word in words:
        word = word.strip().translate(trans) if drop_punc else word.strip()
        if not word: continue  # skip empty words
        if contains_chinese(word):
            chinese_text += word
            chinese_count += len(word)
        else:
            space = " " if english_text and not english_text.endswith(" ") and not word.startswith(tuple(" " + punc)) else ""
            english_text += (space + word)
            english_count += (1 if len(word.strip().translate(trans)) else 0)

    return (english_text, english_count, chinese_text, chinese_count)

def separate_english(text:str) -> tuple[str, str]:
    sep = text
    pinyin = ""
    sep, pinyin = combine_words(jieba.lcut(text)) if contains_chinese(text) else (text, "")
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
def word_anomaly_score(word: Union[Word, PropertyDict]) -> float:
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
        if type(text) == str:
            txt = text.strip()
        else:
            assert type(text) == list, f"Expected list of words, got {type(text)}"
            txt = " ".join([w.word.strip() for w in text])

        logging.info(f"Almost Anomaly: ({score}) '{txt}'")

    return score > 7 or score + 0.01 >= num_words

def is_segment_anomaly(segment: Segment) -> bool:
    if segment is None or not segment.words: return False
    words = [w for w in segment.words if w.word.strip() and w.word not in DEFAULT_PUNCTUATION + ","]
    words = words[:8]
    score = sum(word_anomaly_score(w) for w in words)
    return is_anomaly(score, len(words), words)


class WhisperSegment(PropertyDict):

    def __init__(self, dict=None, /, **kwargs):
        
        arg_words = None
        if dict:
            arg_words = dict.pop('words', None)
            dict.pop('text',None)

        if kwargs:        
            arg_words = kwargs.pop('words', arg_words)
            kwargs.pop('text',None)

        super().__init__(dict, **kwargs)

        self._words = None
        self._text = None
        if arg_words: self.words = arg_words
    
    def __getattr__(self, key):
        if key == 'words':
            if not self._words: self._words = [PropertyDict(w) for w in self.data.get('words', [])]
            return self._words
        elif key == 'text':
            if not self._text: 
                self._text, _ = separate_english("".join([w.word for w in self.words]))
            return self._text
        return super().__getattr__(key)

    def _to_word(self, w):
        if type(w) == PropertyDict: return w
        if type(w) == str: return PropertyDict(word=w)
        if type(w) == dict: return PropertyDict(w)
        raise AttributeError(f"Can't covert {type(w)} to PropertyDict")

    def __setattr__(self, key, value):
        if key == 'words':
            self._text = None        
            self._words = [self._to_word(w) for w in value]
        elif key == 'text':
            pass
        else:
            super().__setattr__(key, value)

    def contains_chinese(self):
        if not self._text: return contains_chinese("".join([w.word for w in self.words]))
        return contains_chinese(self._text)
    
    def anomaly_score(self) -> float:
        if not self._words: return 0
        score = 0
        for word in self._words: score += word_anomaly_score(word)
        return score

    def merge(self, other:Self):
        add_to_end = other.end > self.end
        self.end = max(self.end, other.end)
        self.start = min(self.start, other.start)
        self._text = None
        my_words = [] if self._words is None else self._words
        other_words = [] if other._words is None else other._words
        if add_to_end: 
            my_words += other_words
            result = my_words
        else:
            other_words += my_words
            result = other_words
        self._words = result

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

