import os
from typing import Optional, Self, Union
import numpy as np

from faster_whisper.transcribe import Segment, Word

from hanziconv import HanziConv

import soundfile as sf

from utils import (
    AudioType,
    DEFAULT_PUNCTUATION,
    is_anomaly,
    HALLUCINATIONS,
    SegmentDict,
    separate_english,
    SpeakerGuess,
    to_minutes_seconds,
    WhisperSegment, 
    word_anomaly_score,
    word_count,
)

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
            speaker=segment.speaker,
            speaker_confidence=segment.speaker_confidence,
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
    
    def merge(self, other:Union[Self, WhisperSegment]):
        add_to_end = other.end > self.end
        self.end = max(self.end, other.end)
        self.start = min(self.start, other.start)
        new_text = self.text + other.text if add_to_end else other.text + self.text
        self.text =  new_text
        self._anomaly_score += other.anomaly_score()
        
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

        for hal in HALLUCINATIONS:
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

    def default(o):
        if issubclass(type(o), SubSegment): return o.to_dict()
        return WhisperSegment.default(o)