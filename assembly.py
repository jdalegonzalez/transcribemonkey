# `pip3 install assemblyai` (macOS)
# `pip install assemblyai` (Windows)

import assemblyai as aai

aai.settings.api_key = "9e4d986e31344fda8c75a66fdeb45544"
config = aai.TranscriptionConfig(speaker_labels=True)
transcriber = aai.Transcriber()

# transcript = transcriber.transcribe("https://assembly.ai/news.mp4")
transcript = transcriber.transcribe("./output_audio.mp4", config)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")
