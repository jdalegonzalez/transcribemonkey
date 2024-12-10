
import faster_whisper

import numpy as np
from audiostream.sources.thread import ThreadSource


class MpgAudioSource(ThreadSource):
    """A data source for float32 mono binary data, as loaded by libROSA/soundfile."""
    def __init__(self, stream, source, *args, **kwargs):
        super().__init__(stream, *args, **kwargs)
        self.chunksize = kwargs.get('chunksize', 1024)
        self.sampling_rate = kwargs.get('sampling_rate', 16000)
        self.source = source
        self.data = faster_whisper.decode_audio(self.source, sampling_rate=self.sampling_rate)
        self.cursor = 0

    def get_bytes(self):
        chunk = self.data[self.cursor:self.cursor+self.chunksize]
        self.cursor += self.chunksize

        if not isinstance(chunk, np.ndarray):
            chunk = np.array(chunk)

        assert len(chunk.shape) == 1 and chunk.dtype == np.dtype('float32')

        # Convert to 16 bit format.
        return (chunk * 2**15).astype('int16').tobytes()

