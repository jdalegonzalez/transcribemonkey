from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.behaviors.focus import FocusBehavior

from kivy.properties import (
        BooleanProperty,
        ObjectProperty,
        StringProperty
    )

class ProgressDialog(FloatLayout):
    status = ObjectProperty(None)
    progress = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def on_escape(self, *_):
        self.cancel()

class TranslateDialog(FloatLayout):
    chinese = StringProperty(None)
    english = StringProperty(None)
    cancel = ObjectProperty(None)

    def on_escape(self, *_):
        self.cancel()

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    path = StringProperty("")
    file = StringProperty("")

    def on_enter(self, *_):
        self.load(self.path, self.file)
    def on_escape(self, *_):
        self.cancel()

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)
    export = BooleanProperty(False)
    path = StringProperty("")
    file = StringProperty("")

    def on_enter(self, *_):
        self.save(self.path, self.file, self.export, self.cancel)
    def on_escape(self, *_):
        self.cancel()

ENTER_KEY = 13
ESCAPE_KEY = 27
class DefaultButtonsPopup(FocusBehavior, Popup):

    def __init__(self, **kwargs):
        on_enter = getattr(kwargs['content'], "on_enter", None)
        on_escape = getattr(kwargs['content'], "on_escape", None)
        self._on_enter = on_enter if callable(on_enter) else None
        self._on_escape = on_escape if callable(on_escape) else None
        self.focus = True
        super().__init__(**kwargs)

    def _handle_keyboard(self, _window, key, *_args):
        if key == ENTER_KEY and self._on_enter:
            self._on_enter(self, _window, *_args)
        elif key == ESCAPE_KEY and self._on_escape:
            self._on_escape(self, _window, *_args)

        return super()._handle_keyboard(_window, key, _args)
