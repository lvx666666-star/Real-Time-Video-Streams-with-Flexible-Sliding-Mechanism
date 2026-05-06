from enum import Enum


class WindowLabel(str, Enum):
    ACTION = "action"
    BACKGROUND = "background"
    VAGUE = "vague"


class FrameLabel(str, Enum):
    SUSPICIOUS = "suspicious"
    BACKGROUND = "background"
    VAGUE = "vague"

