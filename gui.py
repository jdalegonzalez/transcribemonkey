#! python
import argparse
import json
import os
import threading
import sys
import logging as lg

from typing import Optional

from pytubefix import Stream

from transcribe import (
    audio_from_file,
    get_segments,
    get_youtube_audio,
    save as save_results,
    transcribe,
)

from utils import (
    contains_chinese,
    to_minutes_seconds,
)

from transcribe import translate as transcribe_translate
from popups import LoadDialog, SaveDialog, ProgressDialog, TranslateDialog, DefaultButtonsPopup
from transcribe_events import TranscribeEvents
from audio_segment import AudioSegment

import kivy
kivy.require('2.3.0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.behaviors.focus import FocusBehavior
from kivy.uix.textinput import TextInput
from kivy.clock import mainthread
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.config import Config

from kivy.properties import (
        AliasProperty,
        BooleanProperty,
        ListProperty,
        NumericProperty,
        ObjectProperty,
        ObservableList,
        StringProperty
)


SAMPLING_RATE = 44100
_is_osx = sys.platform == 'darwin'
app = None

events = TranscribeEvents()

class SentenceInput(TextInput):

    meta_key_map = {
        'cursor_home': 'home',
        'cursor_end': 'end',
        'cursor_left': 'cursor_home',
        'cursor_right': 'cursor_end',
        'cursor_up': 'home',
        'cursor_down': 'end'
    }

    def __init__(self, **kwargs):
        self.burn_space = False
        super().__init__(**kwargs)
    
    def on_triple_tap(self, *_) -> None:
        Clock.schedule_once(lambda x: self.select_all())
    
    def keyboard_on_key_down(self, window, keycode:tuple[int, str], text:str, modifiers: ObservableList):
        modifiers = set(modifiers)
        key, name = keycode
        is_shortcut = (
            'ctrl' in modifiers
            or _is_osx and 'meta' in modifiers
        )
        handled = False
        control='ctrl' in modifiers
        alt='alt' in modifiers

        if modifiers == {'meta'} and name == '=':
            handled = events.split_join_request(self, 'split')
        elif modifiers == {'meta'} and name == '-':
            handled = events.split_join_request(self, 'join')
        elif events.common_keyboard_events(self, name, is_shortcut, modifiers):
            handled = True
        elif control and alt and name == 'left':
            events.slider_pos_request(self, -1)
            handled = True
        elif control and alt and name == 'right':
            events.slider_pos_request(self, 2)
            handled = True
        elif control and alt and name == 'up':
            events.time_change(self, 0, "start")
            handled = True
        elif control and alt and name == 'down':
            events.time_change(self, 0, "end")
            handled = True
        elif control and alt and name == 'home':
            events.slider_pos_request(self, 0)
            handled = True
        elif control and alt and name == 'end':
            events.slider_pos_request(self, .8)
            handled = True
        elif is_shortcut:
            k = SentenceInput.meta_key_map.get(self.interesting_keys.get(key))
            if k:
                self.do_cursor_movement(k, control=control, alt=alt, meta='meta' in modifiers)
                handled = True
            elif name == "=":
                start_end = "end" if 'shift' in modifiers else "start"
                handled = events.time_change(self, 1, start_end)
            elif name == "-":
                start_end = "end" if 'shift' in modifiers else "start"
                handled = events.time_change(self, -1, start_end)
            elif name == "spacebar":
                handled = events.play_stop_request(self)
                self.burn_space = handled
            elif name == "enter":
                handled = events.update_request(self, advance=False)
        elif modifiers == {'shift'} and name == 'enter':
            handled = events.update_request(self)
        elif modifiers == {'shift'} and name == 'spacebar':
            handled = events.play_stop_request(self)
            self.burn_space = self.burn_space or handled
        elif name == 'escape':
            handled = events.update_escape(self)
        
        return handled or super().keyboard_on_key_down(window, keycode, text, modifiers)

    def keyboard_on_textinput(self, window, text):
        burn_space = self.burn_space
        self.burn_space = False
        if text == ' ' and burn_space:
            return
        super().keyboard_on_textinput(window, text)


    def _handle_selection(self):
        if self._selection and not self._selection_finished:
            self._selection_to = self.cursor_index()
            self._update_selection()
        else:
            self.cancel_selection()

    def do_cursor_movement(self, action, control=False, alt=False, meta=False):

        if not control:
            col, row = self.cursor
            handle = False

            if alt and action == 'cursor_up':
                events.previous_row_request(self)
                return True
            if alt and action == 'cursor_down':
                events.next_row_request(self)
                return True
            if action == 'cursor_up' and row == 0 and col > 0:
                col = 0
                handle = True
            elif action == 'cursor_down' and row == len(self._lines) - 1 and col < len(self._lines[row]):
                col = len(self._lines[row])
                handle = True
            elif alt and action == 'cursor_right':
                if alt: col, row = self._move_cursor_word_right()
                else: col = len(self._lines[row])
                handle = True
            elif alt and action == 'cursor_left':
                if alt: col, row = self._move_cursor_word_left()
                else: col = 0
                handle = True
            elif action == 'cursor_home':
                col = 0
                handle = True
            elif action == 'cursor_end':
                col = len(self._lines[row])
                handle = True
            elif action == 'home':
                col = 0
                row = 0
                handle = True
            elif action == 'end':
                row = max(0,len(self._lines) - 1)
                col = len(self._lines[row])
                handle = True
            
            if handle:
                self.cursor = col, row
                self._handle_selection()                
                return True
            
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

class TimeEdit(FocusBehavior, BoxLayout):
    time_label = ObjectProperty()
    adjust_slider = ObjectProperty()
    time_value = NumericProperty(None, allownone=True)
    base_time = NumericProperty(None, allownone=True)
    step = NumericProperty(.01)

    def request_update(self):
        events.update_request(self)

    def keyboard_on_key_down(self, _kb, keycode:tuple[int, str], text: str, modifiers:ObservableList) -> None:

        if not super().keyboard_on_key_down(_kb, keycode, text, modifiers):
            modifiers = set(modifiers)
            _, key = keycode
            is_shortcut = (
                modifiers == {'ctrl'}
                or _is_osx and modifiers == {'meta'}
            )
            control = 'ctrl' in modifiers
            alt = 'alt' in modifiers
            if key == 'left' or key == '-' or key == '_':
                self.decrease_time()
            elif key == 'right' or key == '=' or key == '+':
                self.increase_time()
            elif key == 'spacebar':
                events.play_stop_request(self)
            elif key == 'enter':
                self.request_update()
            elif control and alt and key == 'up':
                events.time_change(self, 0, "start")
            elif control and alt and key == 'down':
                events.time_change(self, 0, "end")
            elif alt and key == 'up':
                events.focus_request(self, 'previous_row')
            elif alt and  key == 'down':
                events.focus_request(self, 'next_row')
            else:
                events.common_keyboard_events(self, key, is_shortcut, modifiers)

    def slider_changed(self, widget, _):
        if widget != self.adjust_slider:
            return False

        if self.base_time is None:
            self.base_time = self.time_value

        if self.base_time and self.time_value:
            val = self.validate
            time_value = round(self.base_time + self.adjust_slider.value, 2)
            if not callable(val) or val(self, time_value):
                self.time_value = time_value
            else:
                def f(*_): self.adjust_slider.value = round(self.time_value - self.base_time,2)
                Clock.schedule_once(f)
                

        return True
    
    def on_time_value(self, widget, val):
        if self.base_time is None:
            self.base_time = val
            if self.adjust_slider: self.adjust_slider.value = 0
    
        if val is None:
            return
        
        val_f = self.validate
        if callable(val_f) and not val_f(self, round(val, 2)):
            widget.text = '{:07.2f}'.format(self.time_value)

        if self.adjust_slider is not None:
            offset = max(self.adjust_slider.min, min(val - self.base_time, self.adjust_slider.max))
            self.adjust_slider.value = offset

    def set_time(self, val):
        if not callable(self.validate) or self.validate(self, val):
            self.time_value = val

    def decrease_time(self, *_):
        time_value = round(self.time_value - self.step, 2) if \
            self.time_value is not None else 0
        
        # If we have a parent that has defined a validation function,
        # we'll call that function to see if the decrease is valid.  
        # If not, we'll assume the decrease is valid.
        val = self.validate
        if not callable(val) or val(self, time_value):
            self.time_value = time_value

    def increase_time(self, *_):
        time_value = round(self.time_value + self.step, 2) if \
            self.time_value is not None else 0

        # If we have a parent that has defined a validation function,
        # we'll call that function to see if the increase is valid.  
        # If not, we'll assume the increase is valid.
        val = self.validate
        if not callable(val) or val(self, time_value):
            self.time_value = time_value

class EditRow(GridLayout):

    def is_dirty(self) -> bool:
        if self.line is None or self.active_ndx is None or \
            self.start_time.time_value is None or self.end_time.time_value is None or \
            self.sentence.text is None or self.speaker.text is None:
            return False
        
        return (
            self.start_time.time_value != self.line['modified_start'] or
            self.end_time.time_value != self.line['modified_end'] or
            self.sentence.text != self.line['text'] or
            self.speaker.text != self.line['speaker']
        )

    start_time = ObjectProperty()
    end_time = ObjectProperty()
    play_button = ObjectProperty()
    speaker = ObjectProperty()
    slider = ObjectProperty()
    audio_file = StringProperty()
    sentence = ObjectProperty()
    active_ndx = NumericProperty(None, allownone=True)
    time_label = ObjectProperty()
    dirty = BooleanProperty(False)

    def __init__(self, **kwargs):
        self.line = None
        events.bind(on_play_stop_request=self.on_play_stop_request)
        super().__init__(**kwargs)
        self.clear_data()

    @property
    def audio_length(self):
        a, sr = self.audio
        rat = sr if sr else 1
        return len(a) / rat
    
    def on_play_stop_request(self, *_):
        self.play_stop()

    def play_stop(self, *_):
        self.play_button.click()

    @mainthread
    def play_stop_click(self, widget, _):

        start_playing = self.segment is None or not self.segment.playing

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

    def audio_slider_down(self, widget, touch):
        if widget.collide_point(*touch.pos):
            self.save_focus()

    def audio_slider_up(self, widget, touch):
        if touch.grab_current == widget:
            self.restore_focus()
        else:
            self._focus_widget = None

    @mainthread
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
        self.audio = audio_from_file(self.audio_file, sampling_rate=SAMPLING_RATE) if self.audio_file else None

    def clear_data(self):
        self._focus_widget = None
        self.audio_file = ""
        self.audio = None
        self.active_ndx = None
        self.line = None
        self.segment = None
        self.restore_slider = 0
        self.dirty = False

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

        # If we've got changes and a reselect of the same line
        # occurs, don't throw those changes away
        if self.dirty and self.active_ndx == ndx:
            return

        self.speaker.disabled = False
        self.dirty = False if self.active_ndx != ndx else self.dirty

        slider_restore = self.restore_slider if ndx == self.active_ndx else 0
        slider_value = self.slider.value if ndx == self.active_ndx else 0


        if ndx != self.active_ndx:
            self.active_ndx = ndx
            self.line = line

        if self.start_time.time_value != line['modified_start']:            
            self.start_time.base_time = None
            self.start_time.time_value = line['modified_start']

        if self.end_time.time_value != line['modified_end']:
            self.end_time.base_time = None
            self.end_time.time_value = line['modified_end']

        if self.speaker.text != line['speaker']:
            self.speaker.text = line['speaker']

        if self.sentence.text != line['text']:
            self.sentence.text = line['text']

        def f(*_):
            self.restore_slider = slider_restore
            self.slider.value = slider_value

        Clock.schedule_once(f)

    def save_focus(self):
        if self.start_time.focus:
            self._focus_widget = self.start_time
        elif self.end_time.focus:
            self._focus_widget = self.end_time
        elif self.sentence.focus:
            self._focus_widget = self.sentence
        else:
            self._focus_widget = self
    
    def restore_focus(self):
        if self._focus_widget is not None:
            widget = self._focus_widget
            self._focus_widget = None
            if widget is self:
                events.focus_request(self, 'current_row')
            else:
                widget.focus = True
        
    def validate_time(self, instance, time_value):
        if time_value is None:
            return True

        is_start_time = instance == self.start_time
        if is_start_time:
            # Start time can't be less than 0 or more than the end time.
            result = time_value >= 0 and (self.end_time.time_value is None or time_value < self.end_time.time_value)
            # If you're messing with the start time, lets reset the slider to 0
            if result and (time_value < self.start_time.time_value) or self.start_time.focus:
                self.slider.value = 0
        else:
            # End time can't be more than the total length of the audio or less the start_value
            result = (self.start_time.time_value is None or time_value > self.start_time.time_value) and time_value < self.audio_length

        self.dirty = self.is_dirty()

        return result
    
    def text_changed(self, *_):
        self.dirty = self.is_dirty()

    def speaker_pressed(self, *_):
        self.save_focus()

    def speaker_selected(self, *_):
        self.restore_focus()
        self.dirty = self.is_dirty()

    def update_clicked(self, *_):
        # If we don't have an active row,
        # there is nothing to update.  Likewise
        # if our data matches, there is nothing
        # to update.
        self.dirty = False
        if self.active_ndx is not None:
            return_focus = self.sentence
            if self.start_time.focus:
                return_focus = self.start_time
            elif self.end_time.focus:
                return_focus = self.end_time
            elif self.sentence.focus:
                return_focus = self.sentence
            events.update_request(return_focus, advance=False)

class TranscriptRow(RecycleDataViewBehavior, BoxLayout):
    def get_minutes_seconds(self) -> str:
        return to_minutes_seconds(self.modified_start)
    
    row_id = StringProperty()
    speaker = StringProperty()
    speaker_confidence = NumericProperty()
    text = StringProperty()
    original_start = NumericProperty()
    original_end = NumericProperty()
    modified_start = NumericProperty()
    modified_end = NumericProperty()
    start_tag = AliasProperty(get_minutes_seconds, None, bind=('modified_start',))
    export = BooleanProperty()
    selected = BooleanProperty()

    background_color = ListProperty([1, 1, 1, 1])
    border_color = ListProperty([0, 1, .25, .25])

    def on_export(self, widget, active):
        pass

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
    
    def from_dict(self, line):
        self.row_id = str(line.get('id', line.get('row_id', '')))
        self.speaker = line['speaker']
        self.speaker_confidence = line['speaker_confidence']
        self.original_start = line['original_start']
        self.original_end = line['original_end']
        self.modified_start = line['modified_start']
        self.modified_end = line['modified_end']
        self.text = line['text']

        self.selectable = line['selectable']
        self.selected = line['selected']
        self.export = line['export']
    
    def to_dict(self):
        line = {}
        line['row_id'] = self.row_id
        line['speaker'] = self.speaker
        line['speaker_confidence'] = self.speaker_confidence
        line['original_start'] = self.original_start
        line['original_end'] = self.original_end
        line['modified_start'] = self.modified_start
        line['modified_end'] = self.modified_end
        line['text'] = self.text

        line['selectable'] = self.selectable
        line['selected'] = self.selected
        line['export'] = self.export 


class TranscriptScreen(Widget):
    title_label = ObjectProperty()
    transcript_view = ObjectProperty()
    edit_row = ObjectProperty()
    video_edit = ObjectProperty()
    episode_edit = ObjectProperty()
    transcribe_btn = ObjectProperty()
    short_checkbox = ObjectProperty()
    lines = ListProperty([])

    loaded = BooleanProperty(False)
    dirty = BooleanProperty(False)

    stop = threading.Event()

    def __init__(self, **kwargs):
        self.top_line = 0
        self.row_for_height = TranscriptRow()

        events.bind(on_rowselect=self.on_rowselect)
        events.bind(on_update_request=self.on_update_request)
        events.bind(on_update_escape=self.on_update_escape)
        events.bind(on_export_checked=self.on_export_checked)
        events.bind(on_focus_request=self.on_focus_request)
        events.bind(on_save_request=self.on_save_request)
        events.bind(on_time_change=self.on_time_change)
        events.bind(on_slider_pos_request=self.on_slider_pos_request)
        events.bind(on_split_join_request=self.on_split_join_request)
        events.bind(on_next_row_request=self.on_next_row_request)
        events.bind(on_previous_row_request=self.on_previous_row_request)
        events.bind(on_set_speaker_request=self.on_set_speaker_request)

        self.transcribe_canceled = False
        self._popup = None
        self._focus_widget = None
        self.handler = OnScreenLogger(self.on_logging)
        Window.bind(on_request_close=self.on_close_request)
        Window.bind(on_resize=self.on_window_resize)
        super(TranscriptScreen, self).__init__(**kwargs)

        self.window_resize_trigger = Clock.create_trigger(self.window_resize_triggered)

    def window_resize_triggered(self, *args):
        for line in self.lines:
            line['size'] = self.calculate_size(line)
        if self.transcript_view:
            Clock.schedule_once(lambda _x: self.transcript_view.refresh_from_data())

    def on_window_resize(self,*args):
        self.window_resize_trigger()

    def on_close_request(self,*args):
        if self.dirty:
            self.dirty = False
            def close_cancel(*_):
                self.dismiss_popup()
                app.stop()
            self.show_save(cancel=close_cancel)
            return True
        
    def list_keyboard_key_down(self, _kb, keycode:tuple[int, str], text:str, modifiers: ObservableList) -> None:

        modifiers = set(modifiers)
        _, keyname = keycode
        is_shortcut = (
            modifiers == {'ctrl'}
            or _is_osx and modifiers == {'meta'}
        )

        nr = None
        new_ndx = None
        current_ndx = self.edit_row.active_ndx

        keyname = "home" if keyname == "up" and is_shortcut else keyname
        keyname = "end" if keyname == "down" and is_shortcut else keyname

        rv = self.transcript_view
        va = rv.view_adapter
        lm = rv.layout_manager
        rv_y = rv.pos[1]

        if keyname == "enter":
            self.edit_row.sentence.focus = True
        elif keyname == 'spacebar':
            events.play_stop_request(self)
            self.edit_row.start_time.focus = True
        elif keyname == "pageup" or keyname == "pagedown":
            indexes = list(sorted(lm.view_indices.values()))
            last_index = True
            if keyname == "pageup":

                if indexes[0] == 0:
                    last_index = False
                    new_ndx = 0
                    new_scroll_y = 1
                else:
                    wdg_top = va.get_visible_view(indexes[0])
                    wdg_top_y = rv.to_parent(0, wdg_top.pos[1])[1] - rv_y
                    wdg_top_showing = rv.height - wdg_top_y
                    top_percent = wdg_top_showing / wdg_top.height

                    # If we're paging down, and the top guy is more than 50% off the screen
                    # we'll subtract the part that is off the screen, leaving the top row
                    # as the bottom row.
                    amount = wdg_top_y + wdg_top.height if top_percent > .5 else wdg_top_y
                    new_scroll_y = min(1, rv.scroll_y + (amount / (lm.height - rv.height)))
            else:

                if indexes[-1] == len(self.lines) - 1:
                    new_ndx = len(self.lines) - 1
                    new_scroll_y = 0
                else:
                    wdg_bot = va.get_visible_view(indexes[-1])
                    wdg_bot_y = rv.to_parent(0, wdg_bot.pos[1])[1] - rv_y
                    wdg_bot_showing = wdg_bot.height + wdg_bot_y
                    bot_percent = wdg_bot_showing / wdg_bot.height
                    amount = rv.height - wdg_bot_y if bot_percent < .5 else rv.height + wdg_bot.height
                    new_scroll_y = max(0, rv.scroll_y - (amount / (lm.height - rv.height)))
                
            rv.scroll_y = new_scroll_y

            def f(*_):
                indexes = list(sorted(lm.view_indices.values()))
                self.select_row(indexes[-1] if last_index else indexes[0])

            Clock.schedule_once(f)
            return
            
        elif events.common_keyboard_events(lm, keyname, is_shortcut, modifiers):
            return
        elif keyname == 'up':
            if current_ndx is None:
                nr = self.lines[0] if self.lines else None
                new_ndx = 0 if self.lines else None
            else:
                nr, new_ndx = self.previous_row(self.edit_row.active_ndx)
        elif keyname == 'down':
            if current_ndx is None:
                nr = self.lines[len(self.lines) - 1] if self.lines else None
                new_ndx = len(self.lines) - 1 if self.lines else None
            else:
                nr, new_ndx = self.next_row(self.edit_row.active_ndx)
        elif keyname == "home":
            nr, new_ndx = (self.lines[0], 0)        
        elif keyname == "end":
            nr, new_ndx = (self.lines[len(self.lines) - 1], len(self.lines) - 1)            

        if nr:
            self.scroll_into_view(new_ndx, keyname, select_row=new_ndx)

    def scroll_into_view(self, ndx, key, select_row:int=None):

        def f(*_):
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
                    indexes = list(sorted(lm.view_indices.values()))
                    if key == "down":
                        widget = va.get_visible_view(indexes[0])
                        dir = -1
                    else: 
                        widget = va.get_visible_view(indexes[-1])
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
                if select_row is not None:
                    self.select_row(select_row)

        Clock.schedule_once(f)

    @mainthread
    def update_clip(self, transcript):
        new_txt = ""
        # For some reason, there seems to be trash segement after the
        # bit I'm actually trying to get.  So, I'm going to ignore
        # the final segment.
        for row in transcript['transcription']: new_txt += (" " + row['text'])
        self.edit_row.sentence.text = new_txt.strip()

    def translate(self, *_):
        editrow: EditRow = self.edit_row
        ndx = editrow.active_ndx
        row, _ = self.current_row(ndx)
        if not row: return
        txt = editrow.sentence.text
        trans = transcribe_translate(txt)
        self.show_translate(txt, trans)

    def re_transcribe(self, *_):
        editrow: EditRow = self.edit_row
        ndx = editrow.active_ndx
        row, _ = self.current_row(ndx)
        if not row: return
        clips=[editrow.start_time.time_value, editrow.end_time.time_value]
        # admitted hack.  If we've got chinese characters in the current
        # text, we're going to set the language to chinese.  Otherwise,
        # we're going with english.
        lang = "zh" if contains_chinese(editrow.sentence.text) else "en"
        self.do_transcribe(audio_file=editrow.audio_file, clips=clips, lang=lang, return_focus=editrow.sentence)

    def _next_id(self, row, ndx):
        next_row, _ =  self.next_row(ndx)
        id_pieces = row['row_id'].split(".")
        id_pieces[-1] = str(int(id_pieces[-1])+1)
        new_id = ".".join(id_pieces)        
        if not next_row is None and next_row['row_id'] == new_id:
            new_id = row['row_id'] + ".0"
    
        return new_id
    
    def split_row(self, *_):
        ndx = self.edit_row.active_ndx
        row, _ = self.current_row(ndx)

        # Allocate half the duration to the new row and half to the old
        # If we're trying to allocate less than half a second, we're going
        # to refuse to do it.
        dur = round((row['modified_end'] - row['modified_start']) / 2,2)
        if dur <= .05:
            return

        # Figure out what the ID of the new row should be...

        # If the slider is currently positioned at the beginning of the 
        # clip, or the cursor is at the beginning of the string, 
        # we'll use a naive "split the thing in half and duplicate" the 
        # text.  Otherwise, we'll use the cursor position to determine
        # what stays in this row and the slider to determine the end time.
        new_row = dict(row)
        new_row['row_id'] = self._next_id(row, ndx)

        cur = self.edit_row.sentence.cursor_index()
        val = self.edit_row.slider.value

        if val != 0 and cur != 0:
            row['text'] = self.edit_row.sentence.text[:cur]
            new_row['text'] = self.edit_row.sentence.text[cur:]
            self.edit_row.sentence.text = row['text']
            dur = (val/self.edit_row.slider.max)*(row['modified_end'] - row['modified_start'])

        row['modified_end'] = round(row['modified_start'] + dur, 2)
        new_row['modified_start'] = row['modified_end']
        spin = self.edit_row.speaker
        speaker = spin.text
        cur_index = spin.values.index(speaker)
        new_index = 1 if cur_index == 0 else 0
        new_row['speaker'] = spin.values[new_index]

        self.insert_row(ndx + 1, new_row)
        self.edit_row.end_time.base_time = None
        self.edit_row.end_time.time_value = row['modified_end']

        self.transcript_view.refresh_from_data()

        def f(_):
            self.transcript_view.layout_manager.focus = True
            #self.scroll_into_view(ndx+1, "down", select_row=ndx+1)

        Clock.schedule_once(f)

    def join_row(self, *_):
        ndx = self.edit_row.active_ndx
        row, _ = self.current_row(ndx)

        next_row, next_ndx = self.next_row(ndx)
        # If the speaker's are different, we're going to 
        # refuse to combine.  Otherwise, merge the text
        # and times.
        s1 = row['speaker'].strip() if row else None
        s2 = next_row['speaker'].strip() if next_row else None
        if next_row and row and (s1 == s2 or not s1 or not s2):
            ### If you're collapsing back something that you just split,
            ### you probably don't want the text duplicated, so... if
            ### both lines of text are identical, we'll just keep one.
            row['speaker'] = next_row['speaker'] if len(next_row['speaker'].strip()) > 0 else \
                row['speaker']
            spc = " " if len(next_row['text']) and next_row['text'][0] != ' ' else ''
            row['text'] += (spc + next_row['text'])
            row['modified_end'] = next_row['modified_end']
            row['original_end'] = next_row['original_end']
            self.remove_row(next_ndx)

            self.transcript_view.refresh_from_data()
    
            def f(*_): self.transcript_view.layout_manager.focus = True
            Clock.schedule_once(f)

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
        self.row_for_height.from_dict(line)
        self.row_for_height.width = self.width
        self.row_for_height.text_widget.texture_update()
        return (self.row_for_height.size[0], self.row_for_height.text_widget.texture_size[1])
    
    @mainthread
    def load_transcript_from_dict(self, transcript):

        def conv(itm):
            # If we see the "id" in the itm, 
            # it's the old format.  So, we'll 
            # convert from it.
            if itm.get('id', None) is not None:
                itm['row_id'] = ("" + str(itm['id']))
                itm.pop('id', None)
            elif itm.get('row_id', None) is not None:
                itm['row_id'] = ("" + str(itm['row_id']))
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

        if self.edit_row.active_ndx is not None:
            self.deselect_row(self.edit_row.active_ndx)

        self.edit_row.clear_data()
        self.title_label.text = transcript.get("title", '')
        self.video_edit.text = transcript.get("YouTubeID", '')
        self.episode_edit.text = str(transcript.get('episode',''))
        self.edit_row.audio_file = transcript.get("AudioFile", '')
        self.lines = [ conv(itm) for itm in transcript["transcription"] ] if transcript["transcription"] else []
        print
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
        if selected_index is None and self.lines:
            selected_index = 0
            scroll_height = 0
            self.lines[0]['selected'] = True

        if selected_index is not None:
            tv = self.transcript_view
            lm = tv.layout_manager
            lm.focus = True

            tvh = tv.height
            def update(_tm):
                tv.scroll_y = min(1, max(0, 1 - scroll_height/(lm.height-tvh)))
                self.select_row(selected_index)
                tv.refresh_from_data()

            if scroll_height > tvh:
                Clock.schedule_once(update)
            else:
                self.select_row(selected_index)


        self.loaded = True


    def transcribe_thread(self, dialog, audio_file, clips, lang, return_focus):

        def on_progress(stream: Stream, _: bytes, bytes_remaining: int):
            filesize = stream.filesize
            bytes_received = filesize - bytes_remaining
            self.update_progress(dialog, text=None, prog=(bytes_received/filesize) * 1000)

        def on_segment(text, _num, _length, percent_complete) -> bool:
            self.update_progress(dialog, text=text, prog = percent_complete * 1000)
            return self.stop.is_set() 

        yt = None
        vid = self.video_edit.text.strip()
        is_short = self.short_checkbox.active
        if not audio_file and not self.stop.is_set():
            self.update_progress(dialog, text = f'Downloading {self.video_edit.text.strip()}', prog = 0)
            audio_file, _, yt = get_youtube_audio(vid, filename=download_dir(), is_short=is_short, progress_callback=on_progress)

        tscript = None
        audio = None
        Logger.parent.addHandler(self.handler)
        if audio_file and not self.stop.is_set():
            message = yt.title if yt is not None else f'{clips[0]}:{clips[1]}'
            self.update_progress(dialog, text = f'Transcribing {message}', prog=0)
            kwargs = {'clips': clips} if clips else {}
            if lang is not None: kwargs['language'] = lang
            tscript, audio = transcribe(audio_file, **kwargs)
            self.update_progress(dialog, prog=0)

        flat_subs = None

        if audio and tscript and not self.stop.is_set():

            episode = self.episode_edit.text
            is_clips = clips is not None
            ignore = .1 if is_clips else 0
            # If we're doing clips, for some reason we get a tiny extra
            # segment that is trash but gets translated into a hallucinated
            # string.  We're going to toss them.
            episode, flat_subs = get_segments(
                tscript, 
                audio, 
                episode=episode, 
                on_seg=on_segment, 
                is_clips=is_clips,
                ignore_shorter_than=ignore
            )

        Logger.parent.removeHandler(self.handler)

        transcript = None
        if flat_subs and not self.stop.is_set():
            self.update_progress(dialog, text=f'Loading data', prog=0)
            transcript = {
                'title': yt.title if yt else '',
                'episode': episode,
                'YouTubeID': vid,
                'AudioFile': audio_file,
                'transcription': []
            }
            length = len(flat_subs)
            for segment in flat_subs:
                new_row = segment.to_dict()
                transcript["transcription"].append(new_row)
                self.update_progress(dialog, prog = ((len(transcript["transcription"])/length)*1000))

        if transcript and not self.stop.is_set():
            if clips is None:
                self.edit_row.json_file = ""
                self.dirty = True
                self.load_transcript_from_dict(transcript)
            else:
                self.update_clip(transcript)

        self.cancel_transcribe()

        if return_focus is not None:
            def f(*_): return_focus.focus = True
            Clock.schedule_once(f)
    
    @mainthread
    def do_transcribe(self, 
        audio_file: Optional[str] = None,
        clips:Optional[list[float]]=None,
        lang:Optional[str]=None, 
        return_focus:Optional[Widget]=None):
        
        # if we've got a running transcribe, kill it.
        if not self.stop.is_set():
            self.stop.set()

        self.stop.clear()
        content = ProgressDialog(cancel=self.cancel_transcribe)
        self._popup = DefaultButtonsPopup(
            title=f'Transcribing {self.video_edit.text.strip()}', 
            content=content, size_hint=(0.5, 0.5))
        self._popup.open()

        threading.Thread(target=self.transcribe_thread, args=(content,audio_file,clips,lang,return_focus)).start()

    def show_translate(self, chinese, english):
        content = TranslateDialog(chinese=chinese,english=english, cancel=self.dismiss_popup)
        self._popup = DefaultButtonsPopup(title='Translation', content=content,size_hint=(1,0.8))
        self._popup.open()

    def show_load(self):
        p, _ = os.path.split(self.edit_row.json_file)
        p = p or app_path()
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup, path=p)
        self._popup = DefaultButtonsPopup(title="Load file", content=content,
                            size_hint=(0.8, 0.8))
        self._popup.open()

    def show_save(self, return_focus:Optional[Widget]=None,cancel:Optional[callable]=None):
        focus_widget = return_focus \
            if return_focus is not None else \
            self.transcript_view.layout_manager

        if cancel is None: cancel = self.dismiss_popup

        p, f = os.path.split(self.edit_row.json_file)
        p = p or app_path()
        f = f or f'{self.video_edit.text.strip()}.transcript.json'

        if self._popup is not None:
            self.dismiss_popup()
        
        def restore_focus(*_):
            self._popup = None
            def q(*_): focus_widget.focus = True
            Clock.schedule_once(q, .5)

        content = SaveDialog(save=self.save, cancel=cancel, path=p, file=f)
        self._popup = DefaultButtonsPopup(
            title="Save file",
            content=content, size_hint=(0.8, 0.8)
        )
        self._popup.bind(on_dismiss=restore_focus)
        self._popup.open()

    def on_rowselect(self, _, ndx:int):
        for row in self.selected_rows(): row['selected'] = False
        self.lines[ndx]['selected'] = True
        self.edit_row.set_data(ndx, self.lines[ndx])

    def deselect_row(self, ndx:int = None) -> None:
        if ndx is None: ndx = self.edit_row.active_ndx
        if ndx is not None and ndx >= 0 and ndx < len(self.lines):
            self.lines[ndx]['selected'] = False

        self.transcript_view.layout_manager.deselect_node(ndx)

    def selected_rows(self):
        for el in self.lines:
            if el['selected']: yield el

    def select_row(self, ndx:int) -> None:
        for row in self.selected_rows(): row['selected'] = False
        if ndx is not None and ndx >= 0 and ndx < len(self.lines):
            self.lines[ndx]['selected'] = True
        self.transcript_view.layout_manager.select_node(ndx)

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
        if self.lines is not None and index is not None:
            self.lines[index]['export'] = checked

    def on_save_request(self, _, requester:Widget):
        self.show_save(return_focus=requester)

    def on_time_change(self, _, requester:Widget, direction: int, start_end: str) -> None:

        widget_name = start_end.lower()
        if widget_name == 'start':
            widget = self.edit_row.start_time
        elif widget_name == 'end':
            widget = self.edit_row.end_time
        if not widget: return

        if direction > 0:
            widget.increase_time()
        elif direction < 0:
            widget.decrease_time()
        else:
            # 0 means use the time of the slider.
            new_time = float(self.edit_row.time_label.text)
            widget.set_time(new_time)
            self.edit_row.slider.value = .5 if widget_name == "end" else 0
            self.edit_row.restore_slider = self.edit_row.slider.value

    def on_slider_pos_request(self, _, requester:Widget, percent: float) -> None:

        slider = self.edit_row.slider
        # percent less than zero means just bump the slider a negative step
        # percent greater than 1 means just bump the slider a positive step.
        if percent < 0:
            slider.value = max(slider.min, slider.value - slider.step)
        elif percent > 1:
            slider.value = min(slider.max, slider.value + slider.step)
        else:
            slider.value = int(slider.max * percent)

    def on_split_join_request(self, _, requester:Widget, split_join: str) -> None:
        if split_join == 'split':
            self.split_row()
            self.edit_row.slider.value = 0
        elif split_join == 'join':
            self.edit_row.slider.value = .8
            self.join_row()

    def on_next_row_request(self, _, requester:Widget) -> None:
        next_row, next_ndx = self.next_row(self.edit_row.active_ndx)
        if next_row is not None:
            self.scroll_into_view(next_ndx, "down")
            self.select_row(next_ndx)

    def on_previous_row_request(self, _, requester:Widget) -> None:
        prev_row, prev_ndx = self.previous_row(self.edit_row.active_ndx)
        if prev_row is not None:
            self.scroll_into_view(prev_ndx, "up")
            self.select_row(prev_ndx)

    def on_set_speaker_request(self, _, requester:Widget, speaker_num: int) -> None:
        speaker_spinner = self.edit_row.speaker
        values = speaker_spinner.values
        if speaker_num >= 1 and speaker_num <= len(values):
            speaker_spinner.text = values[speaker_num - 1]


    def on_focus_request(self, _, requester:Widget, location: str):
        if location == 'start_time':
            self.edit_row.start_time.focus = True
        elif location == 'end_time':
            self.edit_row.end_time.focus = True
        elif location == 'previous_row':
            self.transcript_view.layout_manager.focus = True
            self.list_keyboard_key_down(None, (273, 'up'), '', [])
        elif location == 'next_row':
            self.transcript_view.layout_manager.focus = True
            self.list_keyboard_key_down(None, (274, 'down'), '', [])
        elif location == 'current_row':
            self.transcript_view.layout_manager.focus = True

    def on_update_escape(self, *_):
        def f(_):
            self.transcript_view.layout_manager.focus = True
        Clock.schedule_once(f, .5)

    def on_update_request(self, _, requester, advance = True):


        # The user has requested that the row be updated.
        editrow = self.edit_row
        row = editrow.line
        
        # The editrow won't update itself if it gets a request
        # to change it's index and it believes it's dirty - because
        # that throws away changes.  SO... since this is a request
        # to capture those changes, we're going to tell it it's 
        # clean now.
        editrow.dirty = False

        if row['speaker'] != editrow.speaker.text:
            self.dirty = True
            row['speaker'] = editrow.speaker.text

        if row['text'] != editrow.sentence.text:
            self.dirty = True
            row['text'] = editrow.sentence.text
            row['size'] = self.calculate_size(row)


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
        
        self.transcript_view.refresh_from_data()

        if advance:
            def f(_):
                nr, new_ndx = self.next_row(editrow.active_ndx)
                if nr:
                    self.scroll_into_view(new_ndx, "down", select_row=new_ndx)
                requester.focus = True
            func = f
        else:
            def f(*_): requester.focus = True
            func = f

        Clock.schedule_once(func)

    def on_text(self, _, text):
        self.transcribe_btn.disabled = not text.strip()

    def get_path(self):
        return app_path()
    
    def load(self, path, filename):
        file = os.path.join(path, filename)
        self.load_transcript_from_file(file)
        self.dismiss_popup()


    def save(self, path, filename, export, save_finished):
        # Save grab all the various bits and pieces and build a
        # json ball that we can save.
        path = path or app_path()
        if not filename:
            return
        self.dirty = False
        json_file = str(os.path.join(path, filename))
        self.edit_row.json_file = json_file
        save_results(
            self.title_label.text.strip(),
            self.episode_edit.text.strip(),
            self.video_edit.text.strip(),
            self.edit_row.audio_file,
            str(path),
            json_file,
            True,
            export,
            self.edit_row.audio,
            self.lines
        )

        save_finished()

    def load_transcript_from_file(self, filename: str):
        self.edit_row.json_file = filename
        with open(filename, encoding='utf8') as f:
            transcript = json.load(f)

        self.load_transcript_from_dict(transcript)

class TranscriptApp(App):
    def build(self):
        args = get_arguments()
        screen = TranscriptScreen()
        # If we're given a json file as an argument, we'll
        # load it into the UI.
        if args.video_id:
            screen.video_edit.text = args.video_id

        if args.transcript_json:
            screen.load_transcript_from_file(args.transcript_json)

        return screen
    
def get_arguments():
    """
    Sets up the argument parser and grabs the passed in arguments.

    :return: The parsed arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description="Creates a transcription of a Youtube video"
    )
    parser.add_argument(
        '-l', '--log_level',
        help="The python logging level for output.  Currently unused.  Here for compatibility with ./transcribe.py",
        dest='log_level',
        default="WARNING"
    )
    parser.add_argument(
        "-o", "--out",
        help="The path output audio file.  The default is ./<youtube_id>.mp4.  Currently unused.  Here for compatibility with ./transcribe.py",
        dest="filename",
        default=""
    )
    parser.add_argument(
        "-t", "--transcript", 
        help="The path to a saved transcript to load when the app loads",
        dest='transcript_json',
    )
    parser.add_argument(
        "-v", "--video", 
        help="The Youtube ID for the video to be transcribed",
        dest='video_id',
        default=None
    )
    args = parser.parse_args()
    if args.transcript_json and os.path.isdir(args.transcript_json) and args.video_id is not None:
        filename = os.path.join(args.transcript_json, f'{args.video_id}.transcript.json')
        if os.path.isfile(filename):
            args.transcript_json = filename
    return args

if __name__ == '__main__':
    Config.set('kivy', 'exit_on_escape', 0)
    app = TranscriptApp()
    app.run()
