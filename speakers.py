# -*- coding: utf-8 -*-
"""
This file collects all the functions associated with identifying the 
speakers in audio segments.
"""
from transformers import pipeline, Pipeline
from typing import TypedDict
from utils import AudioType, DEVICE

NO_GUESS = {'label': "", 'score': 0}
TOO_SHORT_TO_GUESS_SECONDS = .2

classifier_model = "./transcribe-monkey"
classifier: Pipeline = pipeline("audio-classification", model=classifier_model, device=DEVICE)

class SpeakerGuess(TypedDict):
    score: float
    label: str

    @staticmethod
    def NO_GUESS() -> 'SpeakerGuess':
        return NO_GUESS

def guess_speaker(audio: AudioType) -> SpeakerGuess:
    # If we have too short a sample, we'll just return NO_GUESS
    raw, sr = audio
    if len(raw) < (sr * TOO_SHORT_TO_GUESS_SECONDS): return NO_GUESS
    return classifier({"sampling_rate": sr, "raw": raw}, top_k=1)[0]

