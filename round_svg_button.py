import re

import xml.etree.ElementTree as ET

from kivy.uix.button import Button
from kivy.metrics import dpi2px, NUMERIC_FORMATS
from kivy.graphics import Color, RoundedRectangle, Ellipse, Scale, Translate, PushMatrix, PopMatrix
from kivy.core.window import Window
from kivy.graphics.svg import Svg

from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ColorProperty,
    ListProperty,
    StringProperty
)

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
    
    def click(self, touch=None):
        self.dispatch("on_click", touch.pos if touch else None)
        self.next_image()

    def on_touch_up(self, touch):
        # For some reason, the default button behavior generates
        # two touch up events.  So, we're going to hijhack it and
        # only change state to "normal" if we're not already normal.
        if self.state == "down":
            self.state = "normal"
            if self.collide_point(*touch.pos):
                self.click()
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

