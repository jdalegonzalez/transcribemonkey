import argparse
import json
import re
import os
import threading
import sys
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
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, RoundedRectangle, Ellipse, Scale, Translate, PushMatrix, PopMatrix
from kivy.clock import mainthread
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.logger import Logger
from kivy.config import Config

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
_is_osx = sys.platform == 'darwin'
app = None

class SentenceInput(TextInput):

    meta_key_map = {
        'cursor_left': 'cursor_home',
        'cursor_right': 'cursor_end',
        'cursor_up': 'cursor_up',
        'cursor_down': 'cursor_down',
    }

    def on_triple_tap(self, *args):
        # We're doing this twice.  Once to get it set and
        # once to get it displayed.
        Clock.schedule_once(lambda x: self.select_all())
    
    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        modifiers = set(modifiers)
        key, _ = keycode
        is_shortcut = (
            modifiers == {'ctrl'}
            or _is_osx and modifiers == {'meta'}
        )
        if is_shortcut:
            k = SentenceInput.meta_key_map.get(self.interesting_keys.get(key))
            if k:
                self.do_cursor_movement(k, is_shortcut=True)
                return
            
        return super().keyboard_on_key_down(window, keycode, text, modifiers)

    def do_cursor_movement(self, action, control=False, alt=False, is_shortcut=False):

        if not control and not alt:
            col, row = self.cursor
            handle = False
            if action == 'cursor_up' and row == 0 and col > 0:
                col = 0
                handle = True
            elif action == 'cursor_down' and row == len(self._lines) - 1 and col < len(self._lines[row]):
                col = len(self._lines[row])
                handle = True
            elif action == 'cursor_beginning':
                col = 0
                row = 0
                handle = True
            elif action == 'cursor_end':
                col = len(self._lines[row])
                row = max(0,len(self._lines) - 1)
                handle = True
            if handle:
                self.cursor = col, row
                return
            
        return super().do_cursor_movement(action, control, alt)


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
        self.register_event_type('on_export_checked')
        self.register_event_type('on_rowselect')
        self.register_event_type('on_update_request')
        super(TranscribeEvents, self).__init__(**kwargs)

    def export_checked(self, ndx, active):
        self.dispatch('on_export_checked', ndx, active)

    def row_selected(self, ndx):
        self.dispatch('on_rowselect', ndx)

    def update_request(self, editrow):
        self.dispatch('on_update_request', editrow)

    def on_export_checked(self, _ndx, _active):
        pass
    def on_rowselect(self, _):
        pass
    def on_update_request(self, _):
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
    
    def on_time_value(self, widget, val):

        val_f = self.validate
        if callable(val_f) and not val_f(self, round(val, 2)):
            widget.text = '{:07.2f}'.format(self.time_value)

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
    active_ndx = NumericProperty(None, allownone=True)
    time_label = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clear_data()

    @property
    def audio_length(self):
        a, sr = self.audio
        rat = sr if sr else 1
        return len(a) / rat
    
    def play_stop_click(self, widget, _):

        start_playing = widget.img_index == 1

        if not start_playing:
            self.segment.stop()
            return
    
        if self.audio and self.start_time.time_value is not None and self.end_time.time_value is not None:
            self.restore_slider = self.slider.value
            audio, sr = self.audio            
            seg_length = self.end_time.time_value - self.start_time.time_value
            mx = self.slider.max if self.slider and self.slider.max else 1
            offset = seg_length * ((self.slider.value / mx) if self.slider else 0)
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
        if self.slider is not None:
            self.slider.value = int(self.slider.max * percent_complete)

        if status == "stopped":
            self.play_button.img_index = 0
        elif status == "ended":
            self.play_button.img_index = 0
            self.slider.value = self.restore_slider

    def on_audio_file(self, _, value):
        self.play_button.img_index = 0
        self.audio_file = value.strip() if value else ""
        self.audio = AudioSegment.from_file(self.audio_file) if self.audio_file else None

    def clear_data(self):
        self.audio_file = ""
        self.audio = None
        self.active_ndx = None
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

    def set_data(self, ndx, line):
        self.restore_slider = 0
        self.slider.value = 0
        self.active_ndx = ndx
        self.start_time.base_time = None
        self.start_time.time_value = line['modified_start']
        self.end_time.base_time = None
        self.end_time.time_value = line['modified_end']
        self.speaker.disabled = False
        self.speaker.text = line['speaker']
        self.sentence.text = line['text']

    def validate_time(self, instance, time_value):
        if time_value is None:
            return True
        
        is_start_time = instance == self.start_time
        if is_start_time:
            # Start time can't be less than 0 or more than the end time.
            result = time_value >= 0 and (self.end_time.time_value is None or time_value < self.end_time.time_value)
            # If you're messing with the start time, lets reset the slider to 0
            if result:
                self.slider.value = 0
        else:
            # End time can't be more than the total length of the audio or less the start_value
            result = (self.start_time.time_value is None or time_value > self.start_time.time_value) and time_value < self.audio_length

        return result
    
    def author_selected(self, *_):
        pass
    
    def update_clicked(self, *_):
        # If we don't have an active row,
        # there is nothing to update.  Likewise
        # if our data matches, there is nothing
        # to update.
        if self.active_ndx is not None:
            events.update_request(self)

class TranscriptRow(RecycleDataViewBehavior, BoxLayout):
    row_id = StringProperty()
    speaker = StringProperty()
    text = StringProperty()
    original_start = NumericProperty()
    original_end = NumericProperty()
    modified_start = NumericProperty()
    modified_end = NumericProperty()
    export = BooleanProperty()
    selected = BooleanProperty()

    background_color = ListProperty([1, 1, 1, 1])
    border_color = ListProperty([0, 1, .25, .25])

    def export_checked(self, _, active):
        events.export_checked(self.index, active)

    def on_touch_down(self, touch):
        ''' Add selection on touch down '''
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)
            
    def apply_selection(self, rv, index, is_selected):
        ''' Respond to the selection of items in the view '''
        self.selected = is_selected
        rv.data[index]['selected'] = is_selected
        if is_selected:
            events.row_selected(self.index)

    def refresh_view_attrs(self, rv, index, data):
        ''' Catch and handle the view changes '''
        self.index = index
        return super().refresh_view_attrs(rv, index, data)
    
    def set_line(self, line):
        self.row_id = line['row_id']
        self.speaker = line['speaker']
        self.original_start = line['original_start']
        self.original_end = line['original_end']
        self.modified_start = line['modified_start']
        self.modified_end = line['modified_end']
        self.text = line['text']

        self.selectable = line['selectable']
        self.selected = line['selected']
        self.export = line['export']

class TranscriptScreen(Widget):

    title_label = ObjectProperty()
    transcript_view = ObjectProperty()
    edit_row = ObjectProperty()
    video_edit = ObjectProperty()
    transcribe_btn = ObjectProperty()
    short_checkbox = ObjectProperty()
    lines = ListProperty([])

    loaded = BooleanProperty(False)
    dirty = BooleanProperty(False)

    stop = threading.Event()

    def __init__(self, **kwargs):
        self.filename = ""
        self.top_line = 0
        self.row_for_height = TranscriptRow()

        events.bind(on_rowselect=self.on_rowselect)
        events.bind(on_update_request=self.on_update_request)
        events.bind(on_export_checked=self.on_export_checked)
        self.transcribe_canceled = False
        self._popup = None
        self.handler = OnScreenLogger(self.on_logging)
        Window.bind(on_request_close=self.on_request_close)
        Window.bind(on_resize=self.on_window_resize)
        super(TranscriptScreen, self).__init__(**kwargs)

        self.window_resize_trigger = Clock.create_trigger(self.window_resize_triggered)

    def window_resize_triggered(self, *args):
        for line in self.lines:
            line['size'] = self.calculate_size(line)
        Clock.schedule_once(lambda _x: self.transcript_view.refresh_from_data())

    def on_window_resize(self,*args):
        self.window_resize_trigger()

    def on_request_close(self,*args):
        if self.dirty:
            self.dirty = False
            def close_cancel(*_):
                self.dismiss_popup()
                app.stop()
            self.show_save(cancel=close_cancel)
            return True
        
    def list_keyboard_key_down(self, _kb, keycode, text, modifiers):
        modifiers = set(modifiers)
        _, keyname = keycode
        is_shortcut = (
            modifiers == {'ctrl'}
            or _is_osx and modifiers == {'meta'}
        )

        # if there's text, it's not a command
        if text:
            return
        
        nr = None
        new_ndx = None
        current_ndx = self.edit_row.active_ndx

        keyname = "home" if keyname == "up" and is_shortcut else keyname
        keyname = "end" if keyname == "down" and is_shortcut else keyname

        if keyname == 'up':
            if current_ndx is None:
                nr = self.lines[0] if self.lines else None
                new_ndx = 0 if self.lines else None
            else:
                nr, new_ndx = self.previous_row(self.edit_row.active_ndx)
        elif keyname == "home":
            nr, new_ndx = (self.lines[0], 0)        
        elif keyname == 'down':
            if current_ndx is None:
                nr = self.lines[len(self.lines) - 1] if self.lines else None
                new_ndx = len(self.lines) - 1 if self.lines else None
            else:
                nr, new_ndx = self.next_row(self.edit_row.active_ndx)
        elif keyname == "end":
            nr, new_ndx = (self.lines[len(self.lines) - 1], len(self.lines) - 1)            

        if nr:
            self.transcript_view.layout_manager.select_node(new_ndx)
            self.scroll_into_view(new_ndx, keyname)

    def scroll_into_view(self, ndx, key):

        def flarm(*_):
            rv = self.transcript_view
            va = self.transcript_view.view_adapter
            lm = self.transcript_view.layout_manager
            vp = self.transcript_view.get_viewport()
            widget = va.get_visible_view(ndx)
            if not widget:
                # If we were scrolling down, we're going to assume 
                # that ndx should be last
                if key == "home":
                    new_scroll_y = 1
                elif key == "end":
                    new_scroll_y = 0
                else:
                    widgets = list(lm.view_indices)
                    if key == "down":
                        widget = widgets[0]
                        dir = -1
                    else: 
                        widget = widgets[-1]
                        dir = 1
                    widget_h = widget.height + widget.padding[1] + widget.padding[3]
                    new_scroll_y = 1 if ndx == 0 else 1 if ndx == len(self.lines) - 1 else rv.scroll_y + (dir * (widget_h / (lm.height - rv.height)))
                rv.scroll_y = new_scroll_y
            else:
                # Make sure the view is completely visible.
                widget_h = widget.height + widget.padding[1] + widget.padding[3]
                _vp_right, vp_bottom, _vp_width, vp_height = vp
                bottom = widget.top - vp_bottom

                if key == "down":
                    dir = -1
                    showing = bottom
                else:
                    dir = 1
                    top = (vp_height - bottom)
                    showing = widget_h + top

                if showing < (widget_h * .95):
                    new_scroll_y = 1 if ndx == 0 else 0 if ndx == len(self.lines) - 1 else rv.scroll_y + (dir * (widget_h / (lm.height - rv.height)))
                    rv.scroll_y = new_scroll_y
                elif ndx == 0 and rv.scroll_y < 1:
                    rv.scroll_y = 1
                elif ndx == len(self.lines) - 1 and rv.scroll_y > 0:
                    rv.scroll_y = 0

        Clock.schedule_once(flarm)

    def split_row(self, *args):
        ndx = self.edit_row.active_ndx
        row, _ = self.current_row(ndx)

        # Allocate half the duration to the new row and half to the old
        # If we're trying to allocate less than half a second, we're going
        # to refuse to do it.
        dur = round((row['modified_end'] - row['modified_start']) / 2,2)
        if dur <= .05:
            return

        # Figure out what the ID of the new row should be...
        next_row, _ =  self.next_row(ndx)
        id_pieces = row['row_id'].split(".")
        id_pieces[-1] = str(int(id_pieces[-1])+1)
        new_id = ".".join(id_pieces)        
        if not next_row is None and next_row['row_id'] == new_id:
            new_id = row['row_id'] + ".0"

        # If the slider is currently positioned at the beginning of the 
        # clip, or the cursor is at the beginning of the string, 
        # we'll use a naive "split the thing in half and duplicate" the 
        # text.  Otherwise, we'll use the cursor position to determine
        # what stays in this row and the slider to determine the end time.
        cur = self.edit_row.sentence.cursor_index()
        val = self.edit_row.slider.value
        new_row = dict(row)
        new_row['row_id'] = new_id

        if val != 0 and cur != 0:
            dur = (val/self.edit_row.slider.max)*(row['modified_end'] - row['modified_start'])
            end = round(row['modified_start'] + dur, 2)
            row['text'] = self.edit_row.sentence.text[:cur]
            new_row['text'] = self.edit_row.sentence.text[cur:]
            row['modified_end'] = end
            new_row['modified_start'] = end
        else:            
            row['modified_end'] = round(row['modified_start'] + dur,2)
            new_row['modified_start'] = row['modified_end']

        self.insert_row(ndx + 1, new_row)
        self.edit_row.end_time.base_time = None
        self.edit_row.end_time.time_value = row['modified_end']


    def collapse_row(self, *args):
        ndx = self.edit_row.active_ndx
        row, _ = self.current_row(ndx)

        next_row, next_ndx = self.next_row(ndx)
        # If the speaker's are different, we're going to 
        # refuse to combine.  Otherwise, merge the text
        # and times.
        s1 = row['speaker'].strip() if row else None
        s2 = next_row['speaker'].strip() if next_row else None
        if next_row and row and s1 == s2 or (len(s1) == 0 or len(s2) == 2):
            ### If you're collapsing back something that you just split,
            ### you probably don't want the text duplicated, so... if
            ### both lines of text are identical, we'll just keep one.
            row['speaker'] = next_row['speaker'] if len(next_row['speaker'].strip()) > 0 else \
                row['speaker']
            if row['text'].strip() != next_row['text'].strip():
                row['text'] += next_row['text']
            row['modified_end'] = next_row['modified_end']
            row['original_end'] = next_row['original_end']
            self.remove_row(next_ndx)

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

    def calculate_size(self, line):
        self.row_for_height.set_line(line)
        self.row_for_height.width = self.width
        self.row_for_height.text_widget.texture_update()
        return (self.row_for_height.size[0], self.row_for_height.text_widget.texture_size[1])
    
    @mainthread
    def load_transcript_from_dict(self, transcript):

        def conv(itm):
            # If we see the "id" in the itm, 
            # it's the old format.  So, we'll 
            # convert from it.
            if itm.get('id', None):
                itm['row_id'] = itm['id']
                itm.pop('id', None)

            # If we see "original", it's the old
            # time format, so we'll convert from it
            if itm.get('original', None):
                orig = itm.pop('original')
                itm['original_start'] = orig['start']
                itm['original_end'] = orig['end']

            # If we see "modified", it's the old
            # time format, so we'll convert from it
            if itm.get('modified', None):
                mod = itm.pop('modified')
                itm['modified_start'] = mod['start']
                itm['modified_end'] = mod['end']

            itm['selectable'] = itm.get('selectable', True)
            itm['selected'] = itm.get('selected', False)
            itm['export'] = itm.get('export', True)

            itm['size'] = self.calculate_size(itm)

            return itm

        self.edit_row.clear_data()

        self.title_label.text = transcript["title"] if transcript["title"] else ""
        self.video_edit.text = transcript["YouTubeID"] if transcript["YouTubeID"] else ""
        self.edit_row.audio_file = transcript["AudioFile"] if transcript["AudioFile"] else ""
        self.lines = [ conv(itm) for itm in transcript["transcription"] ] if transcript["transcription"] else []

        selected_index = None
        scroll_height = 0
        ndx = 0
        for line in self.lines:
            if line['selected']:
                selected_index = ndx
                break
            ndx += 1
            scroll_height += line['size'][1]
        
        # If we've got a selected row, we're going to make sure it's scrolled into view.
        if selected_index is not None:
            tv  = self.transcript_view
            tvh = tv.height
            def update(_tm):
                tv.scroll_y = min(1, max(0, 1 - scroll_height/(tv.children[0].height-tvh)))
                tv.layout_manager.select_node(selected_index)
            if scroll_height > tvh:
                Clock.schedule_once(update)

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
                new_row['row_id'] = segment.id
                new_row['speaker'] = segment.speaker
                new_row['original_start'] = segment.start
                new_row['original_end'] = segment.end
                new_row['modified_start'] = segment.start
                new_row['modified_end'] = segment.end
                new_row['text'] = HanziConv.toSimplified(segment.text)
                transcript["transcription"].append(new_row)
                self.update_progress(dialog, prog = 800 + ((len(transcript["transcription"])/length)*100))

        if transcript and not self.stop.is_set():
            self.filename = ""
            self.dirty = True
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

    def show_save(self, cancel=None):
        if cancel is None:
            cancel = self.dismiss_popup

        p, f = os.path.split(self.filename)
        p = p or app_path()
        f = f or f'{self.video_edit.text.strip()}.transcript.json'
        content = SaveDialog(save=self.save, cancel=cancel, path=p, file=f)
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.8, 0.8))
        self._popup.open()

    def on_rowselect(self, _, ndx:int):
        self.edit_row.set_data(ndx, self.lines[ndx])

    ### These wrapper functions let me change my mind
    ### about where I keep the individual transcript rows.
    def previous_row(self, row_ndx: int) -> tuple[dict, int]:
        return self.row_for_index(row_ndx - 1)
    
    def row_for_index(self, ndx:int) -> tuple[dict, int]:
        return (self.lines[ndx], ndx) if ndx >= 0 and ndx < len(self.lines) else (None, None)
    
    def current_row(self, row_ndx:int) -> tuple[dict, int]:
        return self.row_for_index(row_ndx)
    
    def next_row(self, row_ndx:int) -> tuple[dict, int]:
        return self.row_for_index(row_ndx + 1)

    def insert_row(self, row_ndx:int, new_row:dict) -> tuple[dict, int]:
        self.lines.insert(row_ndx, new_row)

    def remove_row(self, row_ndx: int):
        del self.lines[row_ndx]
    ### End of wrappers

    def on_export_checked(self, evt, index:int, checked:bool):
        self.lines[index]['export'] = checked

    def on_update_request(self, _, editrow):
        # The user has requested that the row be updated.
        row = self.lines[editrow.active_ndx]
        if row['speaker'] != editrow.speaker.text:
            self.dirty = True
            row['speaker'] = editrow.speaker.text

        if row['text'] != editrow.sentence.text:
            self.dirty = True
            row['text'] = editrow.sentence.text
            row['size'] = self.calculate_size(row)
            self.transcript_view.refresh_from_data()

        # There is one little trickyness in that the start
        # has been updated, the end of the previous row has
        # to be updated to match and if the end has been 
        # updated, the start of the next row has to be 
        # updated to match.  We don't currently support
        # gaps or overlaps.
        new_start = editrow.start_time.time_value
        new_end = editrow.end_time.time_value

        keep_sync = editrow.sync_time.active

        if row['modified_start'] != new_start:
            self.dirty = True
            row['modified_start'] = new_start
            prev_row, _ = self.previous_row(editrow.active_ndx)
            if keep_sync and prev_row is not None:
                prev_row['modified_end'] = new_start
        
        if row['modified_end'] != new_end:
            self.dirty = True
            row['modified_end'] = new_end
            next_row, next_ndx = self.next_row(editrow.active_ndx)
            if keep_sync and next_row is not None:
                next_row['modified_start'] = new_end
                # There is a weird case where the last element is sometimes
                # effectively a 0 length repeat of earlier audio.  This will kill
                # that row.
                if new_end >= next_row['modified_end']:
                    self.remove_row(next_ndx)
        
        nr, new_ndx = self.next_row(editrow.active_ndx)
        self.transcript_view.refresh_from_data()
        if nr:
            self.transcript_view.layout_manager.select_node(new_ndx)            
            self.scroll_into_view(new_ndx, "down")



    def on_text(self, _, text):
        self.transcribe_btn.disabled = not text.strip()

    def get_path(self):
        return app_path()
    
    def load(self, path, filename):
        file = os.path.join(path, filename[0])
        self.load_transcript_from_file(file)
        self.dismiss_popup()


    def save(self, path, filename, export, save_finished):
        # Save grab all the various bits and pieces and build a
        # json ball that we can save.
        path = path or app_path()
        if not filename:
            return
        self.dirty = False
        audio, sr = self.edit_row.audio if export else [ None, 0 ]

        transcript = {}
        transcript["title"] = self.title_label.text.strip()
        transcript["YouTubeID"] = self.video_edit.text.strip()
        transcript["AudioFile"] = self.edit_row.audio_file
        transcript["transcription"] = self.lines

        def dest(id: str, speaker: str) -> str:
            p = os.path.join(path, "audio_samples", self.video_edit.text.strip(), speaker.strip().lower())
            if not os.path.exists(p):
                os.makedirs(p)
            return str(os.path.join(p, f'{self.video_edit.text.strip()}.{id}.wav'))

        for kid in self.lines:
            if export and kid['export']:
                seg = AudioSegment(audio, kid['modified_start'], kid['modified_end'], sr)
                seg.save(dest(kid['row_id'], kid['speaker_name']))

        with open(os.path.join(path, filename), 'w', encoding='utf8') as stream:
            json.dump(transcript, stream, indent = 2, ensure_ascii=False)
            stream.write("\n")

        save_finished()

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
    Config.set('kivy', 'exit_on_escape', 0)
    app = TranscriptApp()
    app.run()
