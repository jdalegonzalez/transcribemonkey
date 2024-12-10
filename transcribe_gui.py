import argparse
import json
import re
import os
import threading
import math
import logging as lg

import numpy as np

from hanziconv import HanziConv

import sounddevice as sd
import soundfile as sf

import xml.etree.ElementTree as ET

from pytubefix import Stream

from transcribe import get_youtube_audio, transcribe, get_segments

import kivy
kivy.require('2.3.0')

from kivy.app import App
from kivy.metrics import dpi2px, NUMERIC_FORMATS
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.graphics import Color, RoundedRectangle, Ellipse, Scale, Translate, PushMatrix, PopMatrix
from kivy.clock import mainthread
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.logger import Logger, LOG_LEVELS

from kivy.graphics.svg import Svg
from kivy.properties import (
        AliasProperty,
        BooleanProperty,
        ColorProperty,
        ListProperty,
        NumericProperty,
        ObjectProperty,
        StringProperty,
    )

SAMPLING_RATE = 44100

def app_path():
    import pathlib
    return str(pathlib.Path(__file__).parent.resolve())

def download_dir():
    return os.path.join(app_path(), 'audio_out')

class OnScreenLogger(lg.Handler):
    def __init__(self, on_record, level: int | str = 0) -> None:
        super().__init__(level)
        self.on_record = on_record

    def emit(self, record):
        if record.levelname == "DEBUG" or record.levelname == "INFO":
            self.on_record(record.getMessage())

class RoundSvgButtonWidget(Button):
    RE_LIST = re.compile(r'([A-Za-z]|-?[0-9]+\.?[0-9]*(?:e-?[0-9]*)?)')

    def parse_float(txt: str) -> float:
        if not txt:
            return 0.

        if txt[-2:] in NUMERIC_FORMATS:
            return dpi2px(txt[:-2], txt[-2:])
        
        return float(txt)

    def parse_hw(txt: str, vbox_value: float) -> float:
        if txt.endswith('%'):
            return float(vbox_value * txt[:-1] / 100.)        
        return RoundSvgButtonWidget.parse_float(txt)

    def parse_list(str) -> list:
        return re.findall(RoundSvgButtonWidget.RE_LIST, str)

    def get_activefile(self):
        return self.filelist[self.img_index] if self.img_index < len(self.filelist) else None
    def get_img_index(self) -> int:
        return self._img_index
    def set_img_index(self, value: int):
        v = 0
        try:
            v = int(value)
        except(ValueError):
            v = 0 # if it's not an int, just go with 0
        
        if self.img_index != v:
            self._img_index = v
            return True
        
        return False

    background_shape = StringProperty()
    filename = StringProperty()
    color = ColorProperty([0, 0, 0, 1])
    bg_color = ColorProperty([0, 0, 0, 0])
    background_color=(0, 0, 0, 0)
    down_color = ColorProperty((0, 0, 0, .5))
    offset = ListProperty((0, 0))
    button_down = BooleanProperty(False)
    pressed = ListProperty(None, allownone=True)
    img_index = AliasProperty(get_img_index, set_img_index, bind=('filename',))
    activefile = AliasProperty(get_activefile, None, bind=('filename', 'img_index',))
    scale = ListProperty((1, 1))

    def bg_class(self):
        return Ellipse #  Right now, we're defaulting to Ellipse
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._img_index = 0
        self.max = 0
        self.filelist = []
        self.vbox_x = 0.
        self.vbox_y = 0.
        self.vbox_width = 0.
        self.vbox_height = 0.
        self.down_color_ins = None
        self.background_normal = ""
        self.scale_obj = None
        self.translate = None
        self.register_event_type('on_click')
        bgclazz = self.bg_class()

        with self.canvas.before:
            self.bgColor = Color(rgba=self.bg_color)
            sz = min(self.width, self.height)
            self.background = bgclazz(pos=self.pos, size=(sz, sz))

        with self.canvas:
            self.down_color_ins = Color(rgba=(0, 0, 0, 0))
            PushMatrix()
            self.translate = Translate(1, 1)
            self.scale_obj = Scale(1, 1, 1)
            self.svg = Svg(self.filename)
            if self.filename:
                self.svg.color = self.color if self.color else (0, 0, 0, 0)
            PopMatrix()
            self.mask = bgclazz(pos=self.pos, size=(sz,sz))

    def set_filelist(self):
        self.filelist = (self.filename or "").split(",")
        self.max = max(0,len(self.filelist) - 1)
        self.img_index = 0

    def on_img_index(self, _, value):
        value = self.get_activefile()
        if value != self.svg.source:
            self.svg.clear()
            self.svg.source = value
            self.svg.source = value # Not sure why this has to happen twice 

    def next_image(self):
        # If somehow our image index got out of bounds, 
        # we'll just reset to 0
        if self.img_index + 1 < len(self.filelist):
            self.img_index += 1
        else:
            self.img_index = 0

        return self.get_activefile()
    
    def on_bg_color(self, _, value):
        self.bgColor.rgba = value

    def on_size(self, _, value):
        self.mask.size = value
        self.background.size = value

    def on_scale(self, _, scale):
        x, y = scale
        self.scale_obj.x = x
        self.scale_obj.y = y

    def on_pos(self, _, pos):
        px, py = pos
        # No idea why an offset is necessary but 0,0 
        # doesn't line the image with the background
        ox, oy = self.offset
        self.mask.pos = pos
        self.background.pos = pos
        self.translate.x = px - ox
        self.translate.y = py - oy

    def on_click(self, *_):
        return True
    
    def on_state(self, _, state):
        self.down_color_ins.rgba = self.down_color if state == "down" else (0, 0, 0, 0)
        return True
    
    def on_touch_up(self, touch):
        # For some reason, the default button behavior generates
        # two touch up events.  So, we're going to hijhack it and
        # only change state to "normal" if we're not already normal.
        if self.state == "down":
            self.state = "normal"
            if self.collide_point(*touch.pos):
                self.next_image()
                self.dispatch("on_click", touch.pos)
        return True
    
    def on_backround_color(self, _, value):
        self.background.rgba = value

    def on_color(self, _, value):
        if self.filename:
            self.svg.color = value

    def on_filename(self, *_):
        self.set_filelist()
        v = self.get_activefile()

        # Sucks that the Svg object doesn't expose the view_box.
        # So, we grabbed the viewbox processing code from the source
        # and we're using it here.
        tree = ET.parse(v)
        root = tree.getroot()
        view_box = RoundSvgButtonWidget.parse_list(root.get('viewBox', '0 0 100% 100%'))
        self.vbox_x = RoundSvgButtonWidget.parse_float(view_box[0])
        self.vbox_y = RoundSvgButtonWidget.parse_float(view_box[1])
        self.vbox_width = RoundSvgButtonWidget.parse_hw(view_box[2], Window.width)
        self.vbox_height = RoundSvgButtonWidget.parse_hw(view_box[3], Window.height)
        self.svg.set_tree(tree)  
        self.svg.color = self.color if self.color else (0, 0, 0, 0)

class RoundRectSvgButtonWidget(RoundSvgButtonWidget):
    def bg_class(self):
        return RoundedRectangle

class TranscribeEvents(EventDispatcher):
    def __init__(self, **kwargs):
        self.register_event_type('on_rowselect')
        self.register_event_type('on_update_request')
        super(TranscribeEvents, self).__init__(**kwargs)
    def row_selected(self, row):
        self.dispatch('on_rowselect', row)
    def update_request(self, editrow):
        self.dispatch('on_update_request', editrow)
    def on_update_request(self, _):
        pass
    def on_rowselect(self, _):
        pass

events = TranscribeEvents()

class TimeEdit(BoxLayout):
    time_label = ObjectProperty()
    adjust_slider = ObjectProperty()
    time_value = NumericProperty(None, allownone=True)
    base_time = NumericProperty(None, allownone=True)
    step = NumericProperty(.01)

    def slider_changed(self, _, touch):
        
        if not self.adjust_slider.collide_point(*touch.pos):
            return False

        if self.base_time is None:
            self.base_time = self.time_value

        if self.base_time and self.time_value:
            val = self.validate
            time_value = round(self.base_time + self.adjust_slider.value, 2)
            if not callable(val) or val(self, time_value):
                self.time_value = time_value

        return True
    
    def on_time_value(self, _, val):

        if self.base_time is None:
            self.base_time = val
            self.adjust_slider.value = 0
    
        if self.adjust_slider is not None and \
           val is not None and \
           self.base_time is not None:
            offset = max(self.adjust_slider.min, min(val - self.base_time, self.adjust_slider.max))
            self.adjust_slider.value = offset

    def decrease_time(self, _):
        time_value = round(self.time_value - self.step, 2) if \
            self.time_value is not None else 0
        
        # If we have a parent that has defined a validation function,
        # we'll call that function to see if the decrease is valid.  
        # If not, we'll assume the decrease is valid.
        val = self.validate
        if not callable(val) or val(self, time_value):
            self.time_value = time_value

    def increase_time(self, _):
        time_value = round(self.time_value + self.step, 2) if \
            self.time_value is not None else 0

        # If we have a parent that has defined a validation function,
        # we'll call that function to see if the increase is valid.  
        # If not, we'll assume the increase is valid.
        val = self.validate
        if not callable(val) or val(self, time_value):
            self.time_value = time_value

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

    def from_file(source: str, sampling_rate: int = SAMPLING_RATE) -> np.array:
        """
        Helper function to read an audio file through ffmpeg.
        """
        import faster_whisper
        aud = faster_whisper.decode_audio(source, sampling_rate=sampling_rate)
        return (aud, sampling_rate)

    @property
    def segment(self):
        return self.start, self.end

    @segment.setter
    def segment(self, start_end):
        self.start, self.end = start_end

    def stop(self):
        sd.stop()
        if callable(self.callback):
            self.callback(1.0, "stopped")

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
            if not sd.get_stream().active:
                self.callback(1.0, "stopped")
                return False
            else:
                percent = round((offset_frame + ctx.frame) / self.all_frames, 2)
                self.callback(percent, "playing")
                return True
        
        def _callback(outdata, frames, _, status):
            assert len(outdata) == frames
            ctx.callback_enter(status, outdata)
            ctx.write_outdata(outdata)
            ctx.callback_exit()

        Clock.schedule_interval(clock_func, .03)
        ctx.start_stream(sd.OutputStream, self.sample_rate, ctx.output_channels,
                        ctx.output_dtype, _callback, False,
                        prime_output_buffers_using_stream_callback=False)

    def save(self, dest):
        slice = self._audio_data[int(self.start * self.sample_rate):int(self.end * self.sample_rate)]
        sf.write(dest, slice, self.sample_rate)

    def __repr__(self):
        return f'[id: {self.id}, start: {self.start}, end: {self.end}, text: "{self.text}", speaker: "{self.speaker}"]'

    def __str__(self):
        return f'[{self.id}] {self.speaker if self.speaker else "Speaker"}: [{self.start}] {self.text} [{self.end}]'

class ProgressDialog(FloatLayout):
    status = ObjectProperty(None)
    progress = ObjectProperty(None)
    cancel = ObjectProperty(None)

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = StringProperty("")

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)
    export = ObjectProperty(None)
    path = StringProperty("")
    file = StringProperty("")

class EditRow(GridLayout):

    start_time = ObjectProperty()
    end_time = ObjectProperty()
    play_button = ObjectProperty()
    speaker = ObjectProperty()
    slider = ObjectProperty()
    audio_file = StringProperty()
    sentence = ObjectProperty()
    active_row = ObjectProperty(None, allownone=True)
    time_label = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear_data()

    @property
    def audio_length(self):
        a, sr = self.audio
        return len(a) / sr
    
    def play_stop_click(self, widget, _):

        start_playing = widget.img_index == 1

        if not start_playing:
            self.segment.stop()
            return
    
        if self.audio and self.start_time.time_value is not None and self.end_time.time_value is not None:
            self.restore_slider = self.slider.value
            audio, sr = self.audio            
            seg_length = self.end_time.time_value - self.start_time.time_value            
            offset = seg_length * ((self.slider.value / self.slider.max) if self.slider else 0)
            self.segment = AudioSegment(
                audio, 
                self.start_time.time_value, 
                self.end_time.time_value, 
                sr,
                callback=self.on_playback_event
            )
            self.segment.play(offset=offset)
        else:
            self.play_button.img_index = 0 # not playing

    def audio_slider_change(self, _, pos):
        pass

    def on_playback_event(self, percent_complete, status):
        # Any updates here, for some reason, cause
        # massive clicking/horrible sounds.  It only seems
        # to matter for big files.  SO... I might be able
        # to solve the problem by cutting up the audio, but
        # for now, I'll just not do anything
        if self.slider is not None:
            self.slider.value = int(self.slider.max * percent_complete)
            pass

        if status == "stopped":
            self.play_button.img_index = 0
            self.slider.value = self.restore_slider

    def on_audio_file(self, _, value):
        self.play_button.img_index = 0
        self.audio_file = value.strip() if value else ""
        self.audio = AudioSegment.from_file(self.audio_file) if self.audio_file else None

    def clear_data(self):
        self.audio_file = ""
        self.audio = None
        self.active_row = None
        self.segment = None
        self.restore_slider = 0

        if self.slider:
            self.slider.value = 0
        if self.start_time:
            self.start_time.time_value = None
        if self.end_time:
            self.end_time.time_value = None
        if self.speaker:
            self.speaker.disabled = True
            self.speaker.text = ""
        if self.sentence:
            self.sentence.text = ""

    def on_triple_tap(self, inp):
        Clock.schedule_once(lambda _: inp.select_all())

    def set_data(self, row):
        self.restore_slider = 0
        self.slider.value = 0
        self.active_row = row
        self.start_time.base_time = None
        self.start_time.time_value = row.modified_start
        self.end_time.base_time = None
        self.end_time.time_value = row.modified_end
        self.speaker.disabled = False
        self.speaker.text = row.speaker_name
        self.sentence.text = row.transcription

    def validate_time(self, instance, time_value):
        is_start_time = instance == self.start_time
        if is_start_time:
            # Start time can't be less than 0 or more than the end time.
            result = time_value >= 0 and time_value < self.end_time.time_value
        else:
            # End time can't be more than the total length of the audio or less the start_value
            result = time_value > self.start_time.time_value and time_value < self.audio_length
        return result
    
    def author_selected(self, *_):
        pass
    
    def update_clicked(self, *_):
        # If we don't have an active row,
        # there is nothing to update.  Likewise
        # if our data matches, there is nothing
        # to update.
        if self.active_row:
            events.update_request(self)

class TranscriptRow(ToggleButtonBehavior, GridLayout):
    row_id = StringProperty()
    speaker_name = StringProperty()
    transcription = StringProperty()
    original_start = NumericProperty()
    original_end = NumericProperty()
    modified_start = NumericProperty()
    modified_end = NumericProperty()
    export_checkbox = ObjectProperty()

    background_color = ListProperty([1, 1, 1, 1])
    border_color = ListProperty([0, 1, .25, .25])

    def export(self):
        return self.export_checkbox.active
    
    def on_state(self, _, value):
        if (value == "down"):
            events.row_selected(self)

class TranscriptScreen(Widget):

    title_label = ObjectProperty()
    transcript_grid = ObjectProperty()
    edit_row = ObjectProperty()
    video_edit = ObjectProperty()
    transcribe_btn = ObjectProperty()
    short_checkbox = ObjectProperty()
    lines = ListProperty()

    loaded = BooleanProperty(False)

    stop = threading.Event()

    def __init__(self, **kwargs):
        self.filename = ""
        self.top_line = 0
        events.bind(on_rowselect=self.on_rowselect)
        events.bind(on_update_request=self.on_update_request)
        self.transcribe_canceled = False
        self._popup = None
        self.handler = OnScreenLogger(self.on_logging)
        super(TranscriptScreen, self).__init__(**kwargs)

    def do_remove(self, widget, ndx):
        kid_height = widget.children[0].height if widget.children else 100
        contained_kids = len(widget.children)
        desired_kids = math.ceil(widget.height / kid_height)
        if contained_kids > desired_kids:
            # This print can go if we're ever happy enough with virtual scroll
            # print("Will remove", contained_kids, desired_kids)                      
            widget.remove_widget(widget.children[ndx])
            return True
        return False
    
    def add_to_bottom(self, widget):
        # We remove a row from the top of the list and add
        # a row to the bottom - if we've got one we an add.

        # Remember that the bottom on screen row is at index 0
        # and the top on screen row is at index len(children) - 1
        if self.can_add_bottom():
            first_kid = 0
            last_kid = len(widget.children) - 1
            row = self.create_transcript_row(
                self.lines[last_kid + self.top_line + 1], last_kid + self.top_line + 1
            )
            widget.add_widget(row, index = first_kid) # add a widget to the bottom
            # remove the first widget
            if self.do_remove(widget, -1):
                self.top_line += 1

    def add_to_top(self, widget):
        # We remove a row from the top of the list and add
        # a row to the bottom - if we've got one we an add.

        # Remember that the bottom on screen row is at index 0
        # and the top on screen row is at index len(children) - 1
        if self.can_add_top():
            first_kid = 0
            last_kid = len(widget.children) - 1
            row = self.create_transcript_row(
                self.lines[first_kid + self.top_line - 1], first_kid + self.top_line - 1
            )
            # add a widget to the top
            widget.add_widget(row, index = last_kid + 1)
             # remove the last widget
            if self.do_remove(widget, 0):
                self.top_line -= 1

    def can_add_top(self):
        return self.top_line > 0
    
    def can_add_bottom(self):
        return self.top_line + len(self.transcript_grid.children) + 1 < len(self.lines)
    
    def split_row(self, *args):
        kids = self.transcript_grid.children 
        row = self.edit_row.active_row
        ndx = kids.index(row)

        # Allocate half the duration to the new row and half to the old
        # If we're trying to allocate less than half a second, we're going
        # to refuse to do it.
        dur = (row.modified_end - row.modified_start) / 2
        if dur > .05:
            new_row = self.clone_transcript_row(self.edit_row.active_row)

            # Figure out what the ID of the new row should be...
            next_row =  self.next_row(row)
            id_pieces = row.row_id.split(".")
            id_pieces[-1] = str(int(id_pieces[-1])+1)
            new_id = ".".join(id_pieces)
            if not next_row is None and next_row.row_id == new_id:
                new_id = row.row_id + ".0"
            new_row.row_id = new_id
            
            row.modified_end = row.modified_start + dur
            self.edit_row.end_time.base_time = None
            self.edit_row.end_time.time_value = row.modified_end
            new_row.modified_start = row.modified_end
            self.transcript_grid.add_widget(new_row, ndx)
            self.do_remove(self.transcript_grid, -1)

    def collapse_row(self, *args):
        print("COLLAPSE", *args)

    def on_grid_size(self, widget, size):
        existing_kids = len(widget.children)
        kid_height = widget.children[0].height if existing_kids else 100
        _, h = size
        needed_kids = math.ceil(h / kid_height)
        while self.can_add_bottom() and needed_kids > existing_kids:
            # This print can go if we're ever happy enough with virtual scroll
            # print("adding kid", needed_kids, existing_kids)
            self.add_to_bottom(widget)
            existing_kids = len(widget.children)

    def on_scroll(self, widget, *_args):
        kids = len(widget.children[0].children)
        row_height = widget.children[0].children[0].height
        onscreen = math.floor(widget.height / row_height)
        buffer = math.floor((kids - onscreen) / 2)
        delta = widget.children[0].height - widget.height
        clamped_scroll = max(0, min(1, widget.scroll_y))
        offscreen_top = math.floor((delta * (1 - clamped_scroll)) / row_height)
        offscreen_bot = math.floor((delta * clamped_scroll) / row_height)
        scrolling_down = _args[0].button == "scrolldown"
        scrolling_up = _args[0].button == "scrollup"
        # This print can go if we're ever happy enough with virtual scroll
        # print("top", offscreen_top, "bot", offscreen_bot, "buffer", buffer, "onscreen", onscreen, _args[0].button, scrolling_down, scrolling_up)
        if scrolling_up and offscreen_bot <= buffer and self.can_add_bottom():
            Clock.schedule_once(lambda _: self.add_to_bottom(widget.children[0]))
        elif scrolling_down and offscreen_top <= buffer and self.can_add_top():
            Clock.schedule_once(lambda _: self.add_to_top(widget.children[0]))

    @mainthread
    def dismiss_popup(self):
        if self._popup:
            self._popup.dismiss()
            self._popup = None

    @mainthread
    def cancel_transcribe(self):
        self.stop.set()
        self.dismiss_popup()

    @mainthread
    def on_logging(self, message):
        if not self.stop.is_set() and self._popup and self._popup.content and self._popup.content.status:
            self._popup.content.status.text = message

    @mainthread
    def update_progress(self, dialog=None, text=None, prog=None):
        if dialog is None:
            return
        if text is not None:
            dialog.status.text = text
        if prog is not None:
            dialog.progress.value = prog

    def clone_transcript_row(self, row):
        return TranscriptRow(
            background_color = row.background_color,
            row_id = row.row_id,
            speaker_name = row.speaker_name,
            transcription = row.transcription,
            original_start = row.original_start,
            original_end = row.original_end,
            modified_start = row.modified_start,
            modified_end = row.modified_end
        )
    
    def create_transcript_row(self, line, ndx):
        background_color = [1, 1, 1, 1] if ndx % 2 == 0 else [.95, .95, .95, 1]
        return TranscriptRow(
            background_color = background_color,
            row_id = str(line['id']),
            speaker_name = line['speaker'],
            transcription = HanziConv.toSimplified(line['text'].strip()),
            original_start = round(line['original']['start'],2),
            original_end = round(line['original']['end'],2),
            modified_start = round(line['modified']['start'],2),
            modified_end = round(line['modified']['end'],2)
        )

    @mainthread
    def load_transcript_from_dict(self, transcript):

        self.edit_row.clear_data()

        self.title_label.text = transcript["title"] if transcript["title"] else ""
        self.video_edit.text = transcript["YouTubeID"] if transcript["YouTubeID"] else ""
        self.edit_row.audio_file = transcript["AudioFile"] if transcript["AudioFile"] else ""
        self.lines = transcript["transcription"] if transcript["transcription"] else []
        self.top_line = 0

        # For each line of transcription, add another line to
        # the scroll area
        self.transcript_grid.clear_widgets()
        accume_height = 0
        ndx = 0
        for line in self.lines:
            row = self.create_transcript_row(line, ndx)
            ndx += 1
            accume_height += row.height
            self.transcript_grid.add_widget(row)
            if accume_height >= self.transcript_grid.height:
                break

        self.loaded = True


    def transcribe_thread(self, dialog):

        def on_progress(stream: Stream, _: bytes, bytes_remaining: int):
            filesize = stream.filesize
            bytes_received = filesize - bytes_remaining
            self.update_progress(dialog, text=None, prog=(bytes_received/filesize) * 1000)

        def on_segment(text, num, length) -> bool:
            if not length:
                num = 0
                length = 1

            self.update_progress(dialog, text=text, prog = 700 + ((num/length)*100))

            return self.stop.is_set() 

        audio_file = None
        yt = None
        if not self.stop.is_set():
            self.update_progress(dialog, text = f'Downloading {self.video_edit.text.strip()}', prog = 0)
            vid = self.video_edit.text.strip()
            is_short = self.short_checkbox.active
            audio_file, _, yt = get_youtube_audio(vid, path=download_dir(), is_short=is_short, progress_callback=on_progress)

        transcript_segments = None
        audio = None
        Logger.parent.addHandler(self.handler)

        if audio_file and not self.stop.is_set():
            self.update_progress(dialog, text = f'Transcribing {yt.title}', prog=0)
            transcript_segments, audio = transcribe(audio_file)
            self.update_progress(dialog, prog=600)

        flat_subs = None
        if audio and transcript_segments and not self.stop.is_set():
            flat_subs = get_segments(vid, transcript_segments, audio, on_seg=on_segment)

        Logger.parent.removeHandler(self.handler)

        transcript = None
        if flat_subs and not self.stop.is_set():
            self.update_progress(dialog, text=f'Loading data', prog=800)
            transcript = {}
            transcript["title"] = yt.title
            transcript["YouTubeID"] = vid
            transcript["AudioFile"] = audio_file
            transcript["transcription"] = []
            length = len(flat_subs)
            for segment in flat_subs:
                new_row = {}
                new_row['id'] = segment.id
                new_row['speaker'] = segment.speaker
                new_row['text'] = segment.text
                new_row['original'] = {}
                new_row['original']['start'] = segment.start
                new_row['original']['end'] = segment.end
                new_row['modified'] = {}
                new_row['modified']['start'] = segment.start
                new_row['modified']['end'] = segment.end
                transcript["transcription"].append(new_row)
                self.update_progress(dialog, prog = 800 + ((len(transcript["transcription"])/length)*100))

        if transcript and not self.stop.is_set():
            self.filename = ""
            self.load_transcript_from_dict(transcript)

        self.cancel_transcribe()

    @mainthread
    def do_transcribe(self):
        # if we've got a running transcribe, kill it.
        if not self.stop.is_set():
            self.stop.set()

        self.stop.clear()
        content = ProgressDialog(cancel=self.cancel_transcribe)
        self._popup = Popup(
            title=f'Transcribing {self.video_edit.text.strip()}', 
            content=content, size_hint=(0.5, 0.5))
        self._popup.open()

        threading.Thread(target=self.transcribe_thread, args=(content,)).start()

    def show_load(self):
        p, _ = os.path.split(self.filename)
        p = p or app_path()
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup, path=p)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.8, 0.8))
        self._popup.open()

    def show_save(self):
        p, f = os.path.split(self.filename)
        p = p or app_path()
        f = f or f'{self.video_edit.text.strip()}.transcript.json'
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup, path=p, file=f)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.8, 0.8))
        self._popup.open()

    def on_rowselect(self, _, row):
        self.edit_row.set_data(row)

    def previous_row(self, row):
        kids = self.transcript_grid.children
        ndx = kids.index(row)
        return kids[ndx+1] if ndx+1 < len(kids) else None
    
    def next_row(self, row):
        # OK, the weird thing is that the children are
        # in reverse order.  First added, last in
        # list.  SO, next has to work in the opposite
        # way as you would expect.
        kids = self.transcript_grid.children 
        ndx = kids.index(row)
        return kids[ndx-1] if ndx > 0 else None

    def on_update_request(self, _, editrow):
        # The user has requested that the row be updated.
        if editrow.active_row.speaker_name != editrow.speaker.text:
            editrow.active_row.speaker_name = editrow.speaker.text

        if editrow.active_row.transcription.strip() != editrow.sentence.text.strip():
            editrow.active_row.transcription = editrow.sentence.text

        # There is one little trickyness in that the start
        # has been updated, the end of the previous row has
        # to be updated to match and if the end has been 
        # updated, the start of the next row has to be 
        # updated to match.  We don't currently support
        # gaps or overlaps.
        new_start = editrow.start_time.time_value
        new_end = editrow.end_time.time_value

        keep_sync = editrow.sync_time.active

        if editrow.active_row.modified_start != new_start:
            editrow.active_row.modified_start = new_start
            prev_row = self.previous_row(editrow.active_row)
            if keep_sync and prev_row is not None:
                prev_row.modified_end = new_start
        
        if editrow.active_row.modified_end != new_end:
            editrow.active_row.modified_end = new_end
            next_row = self.next_row(editrow.active_row)
            if keep_sync and next_row is not None:
                next_row.modified_start = new_end
                # There is a weird case where the last element is sometimes
                # effectively a 0 length repeat of earlier audio.  This will kill
                # that row.
                if new_end >= next_row.modified_end:
                    self.transcript_grid.remove_widget(next_row)
        
        nr = self.next_row(editrow.active_row)
        if nr:
            editrow.active_row.state = "normal"
            nr.state = "down"

    def on_text(self, _, text):
        self.transcribe_btn.disabled = not text.strip()

    def get_path(self):
        return app_path()
    
    def load(self, path, filename):
        file = os.path.join(path, filename[0])
        self.load_transcript_from_file(file)
        self.dismiss_popup()


    def save(self, path, filename, export):
        # Save grab all the various bits and pieces and build a
        # json ball that we can save.
        path = path or app_path()
        if not filename:
            return
        
        audio, sr = self.edit_row.audio if export else [None, 0]

        transcript = {}
        transcript["title"] = self.title_label.text.strip()
        transcript["YouTubeID"] = self.video_edit.text.strip()
        transcript["AudioFile"] = self.edit_row.audio_file
        transcript["transcription"] = []
        kids = self.transcript_grid.children

        def dest(id, speaker):
            p = os.path.join(path, "audio_samples", self.video_edit.text.strip(), speaker.strip().lower())
            if not os.path.exists(p):
                os.makedirs(p)
            return str(os.path.join(p, f'{self.video_edit.text.strip()}.{id}.wav'))

        for kid in kids:
            line = {}
            line['id'] = kid.row_id
            line['speaker'] = kid.speaker_name
            line['original'] = {}
            line['original']['start'] = kid.original_start
            line['original']['end'] = kid.original_end
            line['modified'] = {}
            line['modified']['start'] = kid.modified_start
            line['modified']['end'] = kid.modified_end
            line['text'] = kid.transcription
            transcript["transcription"].insert(0, line)
            if export and kid.export():
                seg = AudioSegment(audio, kid.modified_start, kid.modified_end, sr)
                seg.save(dest(kid.row_id, kid.speaker_name))

        with open(os.path.join(path, filename), 'w', encoding='utf8') as stream:
            json.dump(transcript, stream, indent = 2, ensure_ascii=False)

        self.dismiss_popup()

    def load_transcript_from_file(self, filename: str):
        self.filename = filename
        with open(filename, encoding='utf8') as f:
            transcript = json.load(f)

        self.load_transcript_from_dict(transcript)

class TranscriptApp(App):

    def build(self):
        args = get_arguments()
        screen = TranscriptScreen()
        # If we're given a json file as an argument, we'll
        # load it into the UI.
        if args.transcript_json:
            screen.load_transcript_from_file(args.transcript_json)

        return screen
    
def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video",
        usage="transcribe -v <video id> -t <huggingface auth token> [-o name for audio files]"
    )
    parser.add_argument(
        "-t", "--transcript", 
        help="The path to a saved transcript to load when the app loads",
        dest='transcript_json',
    )
    return parser.parse_args()

if __name__ == '__main__':
    TranscriptApp().run()