import os
from typing import Union
from pyannote.audio import Pipeline

def get_hf_token() -> Union[str,None]:
    token = os.getenv('HF_TOKEN', None)
    if token is None:
        path = os.path.expanduser('~/.cache/huggingface/token')
        if os.path.isfile(path):
            with open(path) as f:
                token = f.read()
    return token

token = get_hf_token()
print(f"Using Hugging Face token: {token}")
pipeline = Pipeline.from_pretrained("pyannote/separation-ami-1.0", use_auth_token=token)
#diarization, sources = pipeline("audio.wav")
