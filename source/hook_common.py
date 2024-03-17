import dataclasses
import typing

ShouldSuppressEventFlag = bool
KeyId = typing.Literal[
    "a",
    "s",
    "d",
    "f",
    "h",
    "g",
    "z",
    "x",
    "c",
    "v",
    "b",
    "q",
    "w",
    "e",
    "r",
    "y",
    "t",
    "number_1",
    "number_2",
    "number_3",
    "number_4",
    "number_6",
    "number_5",
    "equal",
    "number_9",
    "number_7",
    "dash",
    "number_8",
    "number_0",
    "right_square",
    "o",
    "u",
    "left_square",
    "i",
    "p",
    "l",
    "j",
    "apostrophe",
    "k",
    "semicolon",
    "backslash",
    "comma",
    "slash",
    "n",
    "m",
    "dot",
    "grave",
    "keypad_dot",
    "keypad_asterisk",
    "keypad_plus",
    "keypad_clear",
    "keypad_slash",
    "keypad_enter",
    "keypad_dash",
    "keypad_equal",
    "keypad_0",
    "keypad_1",
    "keypad_2",
    "keypad_3",
    "keypad_4",
    "keypad_5",
    "keypad_6",
    "keypad_7",
    "keypad_8",
    "keypad_9",
    "enter",
    "tab",
    "space",
    "backspace",
    "escape",
    "right_command",
    "command",
    "shift",
    "caps_lock",
    "option",
    "control",
    "right_shift",
    "right_option",
    "right_control",
    "function",
    "f17",
    "volume_up",
    "volume_down",
    "mute",
    "f18",
    "f19",
    "f20",
    "f5",
    "f6",
    "f7",
    "f3",
    "f8",
    "f9",
    "f11",
    "f13",
    "f16",
    "f14",
    "f10",
    "f12",
    "f15",
    "help",
    "home",
    "page_up",
    "delete",
    "f4",
    "end",
    "f2",
    "page_down",
    "f1",
    "left_arrow",
    "right_arrow",
    "down_arrow",
    "up_arrow",
    "mission_control",
    "dictation",
    "spotlight_search",
    "focus",
    "globe",
    "brightness_up",
    "brightness_down",
    "play",
    "fast",
    "rewind",
]
ButtonId = typing.Literal["left_button", "right_button"]
CursorPositionPixels = complex
WheelScrollDisplacementUnits = complex
FingerId = int
FingerPositionInches = complex


class KeyPressingCallback(typing.Protocol):
    def __call__(
        self, key: KeyId, is_native_repeat: bool, /
    ) -> ShouldSuppressEventFlag: ...


class KeyReleasingCallback(typing.Protocol):
    def __call__(self, key: KeyId, /) -> ShouldSuppressEventFlag: ...


class MouseButtonEventCallback(typing.Protocol):
    def __call__(
        self, button: ButtonId, click_count: int, /
    ) -> ShouldSuppressEventFlag: ...


class MouseCursorMovingCallback(typing.Protocol):
    def __call__(self, new_position: CursorPositionPixels, /) -> None: ...


class MouseWheelScrollingCallback(typing.Protocol):
    def __call__(
        self,
        scroll_displacement: WheelScrollDisplacementUnits,
        is_continuous: bool,
        is_done_by_momentum: bool,
        /,
    ) -> ShouldSuppressEventFlag: ...


class TrackpadFingerPositionsUpdatingCallback(typing.Protocol):
    def __call__(
        self, finger_positions: dict[FingerId, FingerPositionInches], /
    ) -> None: ...


@dataclasses.dataclass
class Handler:
    handle_key_pressing: KeyPressingCallback
    handle_key_releasing: KeyReleasingCallback
    handle_mouse_button_pressing: MouseButtonEventCallback
    handle_mouse_button_releasing: MouseButtonEventCallback
    handle_mouse_cursor_moving: MouseCursorMovingCallback
    handle_mouse_wheel_scrolling: MouseWheelScrollingCallback
    handle_trackpad_finger_positions_updating: TrackpadFingerPositionsUpdatingCallback
