#! python
import sys

from transformers import pipeline
import faster_whisper

if __name__ == '__main__':

    model = "./trained-speakerbox"
    classifier = pipeline("audio-classification", model=model, device="cpu")
    
    command = (sys.argv[1] or "").strip().lower()
    if command == "test":
        source = sys.argv[2]
        sr = 16000

        raw = faster_whisper.decode_audio(source, sampling_rate=sr)

        data = {}
        data['raw'] = raw
        data['sampling_rate'] = sr
        pred = classifier(data, top_k=1)

        print(pred[0]['label'].capitalize())
    elif command == "save":
        classifier.model.push_to_hub("transcribe-monkey")
    else:
        print(f'Unknown command: "{command}"')