from kivy.event import EventDispatcher
from kivy.uix.widget import Widget

class TranscribeEvents(EventDispatcher):
    def __init__(self, **kwargs):
        self.register_event_type('on_export_checked')
        self.register_event_type('on_rowselect')
        self.register_event_type('on_update_request')
        self.register_event_type('on_update_escape')
        self.register_event_type('on_play_stop_request')
        self.register_event_type('on_focus_request')
        self.register_event_type('on_save_request')
        self.register_event_type('on_time_change')
        self.register_event_type('on_slider_pos_request')
        self.register_event_type('on_split_join_request')
        self.register_event_type('on_next_row_request')
        self.register_event_type('on_previous_row_request')
        self.register_event_type('on_set_speaker_request')
        super(TranscribeEvents, self).__init__(**kwargs)

    def export_checked(self, ndx: int, active: bool) -> bool:
        self.dispatch('on_export_checked', ndx, active)
        return True

    def row_selected(self, ndx: int) -> bool:
        self.dispatch('on_rowselect', ndx)
        return True

    def next_row_request(self, requester: Widget) -> bool:
        self.dispatch('on_next_row_request', requester)
        return True

    def previous_row_request(self, requester: Widget) -> bool:
        self.dispatch('on_previous_row_request', requester)
        return True

    def update_request(self, requester: Widget, advance=True) -> bool:
        self.dispatch('on_update_request', requester, advance = advance)
        return True
    
    def update_escape(self, requester: Widget) -> bool:
        self.dispatch('on_update_escape', requester)
        return True

    def play_stop_request(self, requester: Widget) -> bool:
        self.dispatch('on_play_stop_request', requester)
        return True

    def focus_request(self, requester: Widget, location: str) -> bool:
        self.dispatch('on_focus_request', requester, location)
        return True

    def save_request(self, requester: Widget) -> bool:
        self.dispatch('on_save_request', requester)
        return True
    
    def time_change(self, requester: Widget, direction: int, start_end: str) -> bool:
        self.dispatch('on_time_change', requester, direction, start_end)
        return True
    
    def slider_pos_request(self, requester: Widget, percent: float) -> bool:
        self.dispatch('on_slider_pos_request', requester, percent)
        return True

    def split_join_request(self, requester: Widget, split_join: str) -> bool:
        self.dispatch('on_split_join_request', requester, split_join)
        return True

    def set_speaker_request(self, requester: Widget, speaker_num: int) -> bool:
        self.dispatch('on_set_speaker_request', requester, speaker_num)
        return True

    def on_export_checked(self, *_) -> None:
        pass
    def on_rowselect(self, *_) -> None:
        pass
    def on_update_request(self, *_args, **_kwargs) -> None:
        pass
    def on_update_escape(self, *_) -> None:
        pass
    def on_play_stop_request(self, *_) -> None:
        pass
    def on_focus_request(self, *_) -> None:
        pass
    def on_save_request(self, *_) -> None:
        pass
    def on_time_change(self, *_) -> None:
        pass
    def on_slider_pos_request(self, *_) -> None:
        pass
    def on_split_join_request(self, *_) -> None:
        pass
    def on_next_row_request(self, *_) -> None:
        pass
    def on_previous_row_request(self, *_) -> None:
        pass
    def on_set_speaker_request(self, *_) -> None:
        pass

    def common_keyboard_events(self, requester:Widget, key, is_shortcut, modifiers) -> bool:
        if not is_shortcut:
            return False
        meta = {'meta'} == modifiers

        if key.isdigit() and meta:
            self.set_speaker_request(requester, int(key))
            return True
        if key == 'j' and meta:
            self.split_join_request(requester, 'join')
            return True
        if key == 'b':
            self.focus_request(requester, 'start_time')
            return True
        if key == 'e':
            self.focus_request(requester, 'end_time')
            return True
        if key == 's':
            self.save_request(requester)
            return True
        if key == 'home':
            # Set the slider to 0
            self.slider_pos_request(requester, 0)
            self.focus_request(requester, 'start_time')
            return True
        if key == 'end':
            # Set the slider to 90% of max
            self.slider_pos_request(requester, .9)
            self.focus_request(requester, 'end_time')
            return True
        
        return False
