import sys

from transformers import pipeline
import faster_whisper

if __name__ == '__main__':

    model = "./trained-speakerbox"
    classifier = pipeline("audio-classification", model=model, device="cpu")
    source = sys.argv[1]
    sr = 16000

    #classifier(str(tmp_audio_chunk_save_path), top_k=1)[0]
    # sr, raw = wavfile.read(source)
    raw = faster_whisper.decode_audio(source, sampling_rate=sr)
    # with open(source, "rb") as f:
    #     raw = f.read()
    data = {}
    data['raw'] = raw
    data['sampling_rate'] = sr
    pred = classifier(data, top_k=1)
    #pred = classifier(sys.argv[1], top_k=1)
    print(pred[0]['label'].capitalize())