import sounddevice as sd

from kivy.clock import Clock

class AudioSegment():
    def __init__(self, audio_data, start:float, end:float, sample_rate:int, **kwargs):
        super().__init__()
        self._audio_data = audio_data
        self.sample_rate = sample_rate
        self.start = start
        self.end = end
        self.callback = kwargs.get('callback')
        self.stream = None
        self.stream = None
        self.device = None
        self.all_frames = None

    @property
    def segment(self):
        return self.start, self.end

    @segment.setter
    def segment(self, start_end):
        self.start, self.end = start_end

    def stop(self):
        sd.stop()

    def play(self, offset=0):
        sd.stop()

        ctx = sd._CallbackContext(loop=False)
        start = int((self.start + offset) * self.sample_rate)
        end = int(self.end*self.sample_rate)
        
        ctx.frames = ctx.check_data(
            self._audio_data[start:end],
            None, self.device
        )

        if self.all_frames is None:
            if offset == 0:
                self.all_frames = ctx.frames
            else: 
                # This block is taken from the check_data function in the
                # context.  The check_data function does a ton of stuff 
                # though and I don't want to do all of it just to get
                # the number of frames for the entire sample.
                import numpy as np
                dta = np.asarray(self._audio_data[int(self.start * self.sample_rate):end])
                if dta.ndim < 2:
                    dta = dta.reshape(-1, 1)
                self.all_frames, _ = dta.shape

        offset_frame = self.all_frames - ctx.frames

        def clock_func(dt):
            div = self.all_frames if self.all_frames else 1
            percent = round((offset_frame + ctx.frame) / div, 2)
            ended = percent > .99
            stat = "playing" if sd.get_stream().active else \
                "ended" if ended else "stopped"
            self.callback(percent, stat)
            return sd.get_stream().active            
        
        def _callback(outdata, frames, _, status):
            assert len(outdata) == frames
            ctx.callback_enter(status, outdata)
            ctx.write_outdata(outdata)
            ctx.callback_exit()

        Clock.schedule_interval(clock_func, .02)
        ctx.start_stream(sd.OutputStream, self.sample_rate, ctx.output_channels,
                        ctx.output_dtype, _callback, False,
                        prime_output_buffers_using_stream_callback=False)

    def __repr__(self):
        return f'[id: {self.id}, start: {self.start}, end: {self.end}, text: "{self.text}", speaker: "{self.speaker}"]'

    def __str__(self):
        return f'[{self.id}] {self.speaker if self.speaker else "Speaker"}: [{self.start}] {self.text} [{self.end}]'

