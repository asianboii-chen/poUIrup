import util

import Quartz

import dataclasses
import functools
import math
import time

from hook_common import *


NativeKeyCode = int

_NATIVE_KEY_CODES: dict[KeyId, NativeKeyCode] = {
    "a": 0,
    "s": 1,
    "d": 2,
    "f": 3,
    "h": 4,
    "g": 5,
    "z": 6,
    "x": 7,
    "c": 8,
    "v": 9,
    "b": 11,
    "q": 12,
    "w": 13,
    "e": 14,
    "r": 15,
    "y": 16,
    "t": 17,
    "number_1": 18,
    "number_2": 19,
    "number_3": 20,
    "number_4": 21,
    "number_6": 22,
    "number_5": 23,
    "equal": 24,
    "number_9": 25,
    "number_7": 26,
    "dash": 27,
    "number_8": 28,
    "number_0": 29,
    "right_square": 30,
    "o": 31,
    "u": 32,
    "left_square": 33,
    "i": 34,
    "p": 35,
    "l": 37,
    "j": 38,
    "apostrophe": 39,
    "k": 40,
    "semicolon": 41,
    "backslash": 42,
    "comma": 43,
    "slash": 44,
    "n": 45,
    "m": 46,
    "dot": 47,
    "grave": 50,
    "keypad_dot": 65,
    "keypad_asterisk": 67,
    "keypad_plus": 69,
    "keypad_clear": 71,
    "keypad_slash": 75,
    "keypad_enter": 76,
    "keypad_dash": 78,
    "keypad_equal": 81,
    "keypad_0": 82,
    "keypad_1": 83,
    "keypad_2": 84,
    "keypad_3": 85,
    "keypad_4": 86,
    "keypad_5": 87,
    "keypad_6": 88,
    "keypad_7": 89,
    "keypad_8": 91,
    "keypad_9": 92,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "backspace": 51,
    "escape": 53,
    "right_command": 54,
    "command": 55,
    "shift": 56,
    "caps_lock": 57,
    "option": 58,
    "control": 59,
    "right_shift": 60,
    "right_option": 61,
    "right_control": 62,
    "function": 63,
    "f17": 64,
    "volume_up": 72,
    "volume_down": 73,
    "mute": 74,
    "f18": 79,
    "f19": 80,
    "f20": 90,
    "f5": 96,
    "f6": 97,
    "f7": 98,
    "f3": 99,
    "f8": 100,
    "f9": 101,
    "f11": 103,
    "f13": 105,
    "f16": 106,
    "f14": 107,
    "f10": 109,
    "f12": 111,
    "f15": 113,
    "help": 114,
    "home": 115,
    "page_up": 116,
    "delete": 117,
    "f4": 118,
    "end": 119,
    "f2": 120,
    "page_down": 121,
    "f1": 122,
    "left_arrow": 123,
    "right_arrow": 124,
    "down_arrow": 125,
    "up_arrow": 126,
    "mission_control": 160,
    "dictation": 176,
    "spotlight_search": 177,
    "focus": 178,
    "globe": 179,
}

SUPPORTED_KEYS = set(_NATIVE_KEY_CODES)

_KEY_BY_NATIVE_KEY_CODE: dict[NativeKeyCode, KeyId] = {
    code: key for key, code in _NATIVE_KEY_CODES.items()
}

_MODIFIER_NATIVE_FLAG_MASKS: dict[KeyId, int] = {
    "caps_lock": Quartz.kCGEventFlagMaskAlphaShift,
    "shift": Quartz.kCGEventFlagMaskShift,
    "control": Quartz.kCGEventFlagMaskControl,
    "option": Quartz.kCGEventFlagMaskAlternate,
    "command": Quartz.kCGEventFlagMaskCommand,
    "help": Quartz.kCGEventFlagMaskHelp,
    "function": Quartz.kCGEventFlagMaskSecondaryFn,
}


SUPPORTED_MODIFIERS = set(_MODIFIER_NATIVE_FLAG_MASKS)

MODIFIER_BY_OPTIONALLY_PAIRED_KEY: dict[KeyId, KeyId] = (
    {modifier: modifier for modifier in SUPPORTED_MODIFIERS}
) | {
    "right_control": "control",
    "right_option": "option",
    "right_command": "command",
    "right_shift": "shift",
}  # type: ignore

_MEDIA_KEY_NX_KEY_TYPES: dict[KeyId, int] = {
    "volume_up": 0,
    "volume_down": 1,
    "brightness_up": 2,
    "brightness_down": 3,
    "caps_lock": 4,
    "mute": 7,
    "play": 16,
    "fast": 19,
    "rewind": 20,
}

_MEDIA_KEY_BY_NX_KEY_TYPE: dict[int, KeyId] = {
    key_type: key for key, key_type in _MEDIA_KEY_NX_KEY_TYPES.items()
}

SUPPORTED_BUTTONS: set[ButtonId] = {"left_button", "right_button"}

_NS_EVENT_TYPE_CONSTANT_NAMES = {
    1: "NSEventTypeLeftMouseDown",
    2: "NSEventTypeLeftMouseUp",
    3: "NSEventTypeRightMouseDown",
    4: "NSEventTypeRightMouseUp",
    5: "NSEventTypeMouseMoved",
    6: "NSEventTypeLeftMouseDragged",
    7: "NSEventTypeRightMouseDragged",
    8: "NSEventTypeMouseEntered",
    9: "NSEventTypeMouseExited",
    10: "NSEventTypeKeyDown",
    11: "NSEventTypeKeyUp",
    12: "NSEventTypeFlagsChanged",
    13: "NSEventTypeAppKitDefined",
    14: "NSEventTypeSystemDefined",
    15: "NSEventTypeApplicationDefined",
    16: "NSEventTypePeriodic",
    17: "NSEventTypeCursorUpdate",
    18: "NSEventTypeRotate",
    19: "NSEventTypeBeginGesture",
    20: "NSEventTypeEndGesture",
    22: "NSEventTypeScrollWheel",
    23: "NSEventTypeTabletPoint",
    24: "NSEventTypeTabletProximity",
    25: "NSEventTypeOtherMouseDown",
    26: "NSEventTypeOtherMouseUp",
    27: "NSEventTypeOtherMouseDragged",
    29: "NSEventTypeGesture",
    30: "NSEventTypeMagnify",
    31: "NSEventTypeSwipe",
    32: "NSEventTypeSmartMagnify",
    33: "NSEventTypeQuickLook",
    34: "NSEventTypePressure",
    37: "NSEventTypeDirectTouch",
    38: "NSEventTypeChangeMode",
}


@dataclasses.dataclass
class AutoRepeatState:
    auto_repeating_key: KeyId | None
    prev_key_press_event_time: float
    has_started_repeating: bool


@dataclasses.dataclass
class ClickLevelState:
    prev_mouse_button_pressed: ButtonId | None
    prev_mouse_button_press_event_time: float
    prev_click_level: int


@dataclasses.dataclass
class State:
    is_active: bool
    event_tap: Quartz.NSMachPort | None

    # macOS natively remembers its modifier state, regardless of whether the modifiers' key press events (sent by the system with type NSEventTypeFlagsChanged) were suppressed. This field remembers which flags are actually unsuppressed, meaning they're active from the user's perspective after knowing which key events they suppressed or emulated.
    unsuppressed_flags: int

    is_caps_lock_natively_pressed: bool
    was_caps_lock_native_press_suppressed: bool

    auto_repeat_state: AutoRepeatState
    click_level_state: ClickLevelState


def _get_key_from_event(
    key_press_or_release_event: Quartz.CGEventRef,
) -> tuple[NativeKeyCode, KeyId | None]:
    native_key_code = Quartz.CGEventGetIntegerValueField(
        key_press_or_release_event, Quartz.kCGKeyboardEventKeycode
    )
    return native_key_code, _KEY_BY_NATIVE_KEY_CODE.get(native_key_code, None)


def _terminate_auto_repeat_by_new_keyboard_event_if_needed(
    state: State, event_target_key: KeyId, is_press_event: bool
) -> None:
    if (
        event_target_key == state.auto_repeat_state.auto_repeating_key
    ) != is_press_event:
        state.auto_repeat_state.auto_repeating_key = None


def _get_cursor_position_from_event(
    mouse_event: Quartz.CGEventRef,
) -> CursorPositionPixels:
    p = Quartz.CGEventGetLocation(mouse_event)
    return p.x + 1j * p.y


def _update_click_level_state_with_new_mouse_button_event(
    click_level_state: ClickLevelState,
    event_target_button: ButtonId,
    is_press_event: bool,
) -> int:
    curr_time = time.time()
    has_prev_click_level_expired = (
        event_target_button != click_level_state.prev_mouse_button_pressed
        or (
            is_press_event
            and curr_time - click_level_state.prev_mouse_button_press_event_time
            >= Quartz.NSEvent.doubleClickInterval()
        )
    )
    if has_prev_click_level_expired:
        click_level_state.prev_mouse_button_pressed = (
            event_target_button if is_press_event else None
        )
        click_level_state.prev_click_level = 0
    if is_press_event:
        click_level_state.prev_mouse_button_press_event_time = curr_time
        click_level_state.prev_click_level += 1
    return max(click_level_state.prev_click_level, 1)


def _handle_mouse_button_event(
    state: State,
    handle: MouseButtonEventCallback,
    button: ButtonId,
    mouse_event: Quartz.CGEventRef,
    is_press: bool,
) -> Quartz.CGEventRef:
    click_level = _update_click_level_state_with_new_mouse_button_event(
        state.click_level_state, button, is_press
    )
    Quartz.CGEventSetIntegerValueField(
        mouse_event, Quartz.kCGMouseEventClickState, click_level
    )
    should_suppress = handle(button, click_level)
    return None if should_suppress else mouse_event


def _handle_native_event(
    state: State,
    handler: Handler,
    proxy: Quartz.CGEventTapProxy,
    event_type: int,
    event: Quartz.CGEventRef,
    user_data: None,
) -> Quartz.CGEventRef:
    if event_type == Quartz.kCGEventTapDisabledByUserInput:
        return event

    if event_type == Quartz.kCGEventTapDisabledByTimeout:
        Quartz.CGEventTapEnable(state.event_tap, True)
        return event

    source_pid = Quartz.CGEventGetIntegerValueField(
        event, Quartz.kCGEventSourceUnixProcessID
    )
    is_emulated = source_pid != 0

    if event_type != Quartz.NSEventTypeFlagsChanged and not is_emulated:
        Quartz.CGEventSetFlags(event, state.unsuppressed_flags)

    if event_type == Quartz.NSEventTypeSystemDefined:
        ns_event = Quartz.NSEvent.eventWithCGEvent_(event)
        event_subtype: int = ns_event.subtype()
        data1: int = ns_event.data1()
        data2: int = ns_event.data2()

        if event_subtype == 7:
            # An extraneous event coming after a mouse button press or release.
            return event

        if event_subtype == 211:
            # A press or release of Caps Lock.

            is_press: bool = data1 != 0
            if is_press:
                # Activating Caps Lock may send two press events. The first event is normal and sent as soon as the key is pressed. If the key is held for just a bit, a second event is sent a bit after (the infamously over-engineered macOS Caps Lock activation delay).
                if state.is_caps_lock_natively_pressed:
                    if state.was_caps_lock_native_press_suppressed:
                        return None
                    else:
                        return event

                state.is_caps_lock_natively_pressed = True
                should_suppress = handler.handle_key_pressing("caps_lock", False)
                state.was_caps_lock_native_press_suppressed = should_suppress

            else:
                state.is_caps_lock_natively_pressed = False
                should_suppress = handler.handle_key_releasing("caps_lock")

            if should_suppress:
                return None

            _terminate_auto_repeat_by_new_keyboard_event_if_needed(
                state, "caps_lock", is_press
            )

            return event

        if event_subtype == 8:
            # A press or release of a media key.

            nx_key_type = data1 >> 16
            is_press = (data1 >> 8 & 1) == 0
            is_native_repeat = (data1 & 1) != 0

            if nx_key_type not in _MEDIA_KEY_BY_NX_KEY_TYPE:
                util.log_error(
                    f"{'Pressing' if is_press else 'Releasing'} unsupported media key with NX key type {nx_key_type}."
                )
                return event
            key = _MEDIA_KEY_BY_NX_KEY_TYPE[nx_key_type]

            if key == "caps_lock":
                # Caps Lock activation is managed by flag-change.
                return event

            if is_press:
                should_suppress = handler.handle_key_pressing(key, is_native_repeat)
            else:
                should_suppress = handler.handle_key_releasing(key)
            if should_suppress:
                return None

            _terminate_auto_repeat_by_new_keyboard_event_if_needed(
                state, "caps_lock", is_press
            )

            return event

        util.log_error(
            f"Receiving unsupported system event with subtype {event_subtype}, data1 {data1}, and data2 {data2}."
        )
        return event

    if event_type == Quartz.NSEventTypeKeyDown:
        code, key = _get_key_from_event(event)
        if key is None:
            util.log_error(
                f"Receiving key press event with unsupported key code {code}."
            )
            return event

        is_native_repeat = bool(
            Quartz.CGEventGetIntegerValueField(
                event,
                Quartz.kCGKeyboardEventAutorepeat,  # spell-checker: disable-line
            )
        )
        should_suppress = handler.handle_key_pressing(key, is_native_repeat)
        if should_suppress:
            return None

        _terminate_auto_repeat_by_new_keyboard_event_if_needed(
            state, key, is_press_event=True
        )

        return event

    if event_type == Quartz.NSEventTypeKeyUp:
        code, key = _get_key_from_event(event)
        if key is None:
            util.log_error(
                f"Receiving key release event with unsupported key code {code}."
            )
            return event

        should_suppress = handler.handle_key_releasing(key)
        if should_suppress:
            return None

        _terminate_auto_repeat_by_new_keyboard_event_if_needed(
            state, key, is_press_event=False
        )

        return event

    if event_type == Quartz.NSEventTypeFlagsChanged:
        curr_flags: int = Quartz.CGEventGetFlags(event)
        prev_flags = state.unsuppressed_flags
        state.unsuppressed_flags = curr_flags

        code, key = _get_key_from_event(event)
        if key is None or key not in MODIFIER_BY_OPTIONALLY_PAIRED_KEY:
            util.log_error(
                f"Received flag-change event associated with key code {code} with new flags {bin(curr_flags)}, handling which is not yet supported."
            )
            return event

        if key == "caps_lock":
            # The Caps Lock events are handled by certain NSEventTypeSystemDefined-events, where the equivalent press event happens more immediately.

            if not state.was_caps_lock_native_press_suppressed:
                return event

            state.unsuppressed_flags = prev_flags
            # BUG Returning None does not suppress the LED on the physical Caps Lock key from changing its state, meaning that the LED will get out-of-sync with Caps Lock's activation status.
            return None

        modifier = MODIFIER_BY_OPTIONALLY_PAIRED_KEY[key]
        flag_mask = _MODIFIER_NATIVE_FLAG_MASKS[modifier]
        is_press = (curr_flags & flag_mask) != 0

        if is_press:
            should_suppress = handler.handle_key_pressing(key, False)
        else:
            should_suppress = handler.handle_key_releasing(key)

        if should_suppress:
            state.unsuppressed_flags = prev_flags
            return None

        _terminate_auto_repeat_by_new_keyboard_event_if_needed(state, key, is_press)

        return event

    if event_type == Quartz.NSEventTypeLeftMouseDown:
        return _handle_mouse_button_event(
            state,
            handler.handle_mouse_button_pressing,
            "left_button",
            event,
            is_press=True,
        )

    if event_type == Quartz.NSEventTypeLeftMouseUp:
        return _handle_mouse_button_event(
            state,
            handler.handle_mouse_button_releasing,
            "left_button",
            event,
            is_press=False,
        )

    if event_type == Quartz.NSEventTypeRightMouseDown:
        return _handle_mouse_button_event(
            state,
            handler.handle_mouse_button_pressing,
            "right_button",
            event,
            is_press=True,
        )

    if event_type == Quartz.NSEventTypeRightMouseUp:
        return _handle_mouse_button_event(
            state,
            handler.handle_mouse_button_releasing,
            "right_button",
            event,
            is_press=False,
        )

    if event_type in {
        Quartz.NSEventTypeMouseMoved,
        Quartz.NSEventTypeLeftMouseDragged,
        Quartz.NSEventTypeRightMouseDragged,
    }:
        pos = _get_cursor_position_from_event(event)
        handler.handle_mouse_cursor_moving(pos)
        return event

    if event_type == Quartz.NSEventTypeScrollWheel:
        momentum_phase = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGScrollWheelEventMomentumPhase
        )
        if momentum_phase == 3:
            # The momentum ending phase; nothing interesting to report.
            return event

        is_done_by_momentum = momentum_phase == 2

        dx = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGScrollWheelEventDeltaAxis2
        )
        dy = Quartz.CGEventGetIntegerValueField(
            event, Quartz.kCGScrollWheelEventDeltaAxis1
        )
        if dx == 0 and dy == 0:
            return event

        ns_event = Quartz.NSEvent.eventWithCGEvent_(event)
        if not ns_event.isDirectionInvertedFromDevice():
            dx *= -1
            dy *= -1

        is_continuous = (
            Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGScrollWheelEventIsContinuous
            )
            != 0
        )

        should_suppress = handler.handle_mouse_wheel_scrolling(
            dx + 1j * dy, is_continuous, is_done_by_momentum
        )
        return None if should_suppress else event

    if event_type == Quartz.NSEventTypeGesture:
        ns_event = Quartz.NSEvent.eventWithCGEvent_(event)

        TRACKPAD_PHYSICAL_WIDTH_INCHES = 8.5 * 0.75
        TRACKPAD_PHYSICAL_DEPTH_INCHES = 5 * 0.75

        finger_pos: dict[int, FingerPositionInches] = {}
        for touch in ns_event.allTouches():
            if touch.phase() == Quartz.NSTouchPhaseEnded:
                continue

            i = touch.identity()._value()
            p = touch.normalizedPosition()
            finger_pos[i] = (
                p[0] * TRACKPAD_PHYSICAL_WIDTH_INCHES
                + 1j * (1 - p[1]) * TRACKPAD_PHYSICAL_DEPTH_INCHES
            )

        handler.handle_trackpad_finger_positions_updating(finger_pos)

        return event

    if event_type == Quartz.NSEventTypeMagnify:
        # spell-checker: ignore Exposé
        # HACK Apparently this is what Mission Control and App Exposé internally listen to. Suppress to disable those default behaviors as we'll probably implement mouse gestures to handle them anyway. I haven't found any side effects for doing this yet--luckily pinching and double-tapping zoom features still work just fine with this forced suppression.
        # return None
        # FIXME (hack on top of hack) To allow tracking gestures made with trackpad, let's let pass this event for now, at least until we actually assign and implement the gesture actions. (We could even keep this the official behavior that native gestures won't be suppressed quietly, so that the user must disable Mission Control and App Exposé in System Settings to use the custom gestures, which is arguably also reasonable.)
        return event

    if event_type in _NS_EVENT_TYPE_CONSTANT_NAMES:
        util.log_error(
            f"Received event with type {event_type} ({_NS_EVENT_TYPE_CONSTANT_NAMES[event_type]}), handling which is not yet supported."
        )
    else:
        util.log_error(f"Received unsupported event with type {event_type}.")

    return event


def create() -> State:
    return State(
        is_active=False,
        event_tap=None,
        unsuppressed_flags=0,
        is_caps_lock_natively_pressed=False,
        was_caps_lock_native_press_suppressed=False,
        auto_repeat_state=AutoRepeatState(
            auto_repeating_key=None,
            prev_key_press_event_time=-math.inf,
            has_started_repeating=False,
        ),
        click_level_state=ClickLevelState(
            prev_mouse_button_pressed=None,
            prev_mouse_button_press_event_time=-math.inf,
            prev_click_level=0,
        ),
    )


def register_handler(state: State, handler: Handler) -> None:
    state.event_tap = Quartz.CGEventTapCreate(
        Quartz.kCGAnnotatedSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionDefault,
        Quartz.kCGEventMaskForAllEvents,
        functools.partial(_handle_native_event, state, handler),
        None,
    )
    loop_source = Quartz.CFMachPortCreateRunLoopSource(None, state.event_tap, 0)
    event_loop = Quartz.CFRunLoopGetCurrent()
    Quartz.CFRunLoopAddSource(event_loop, loop_source, Quartz.kCFRunLoopDefaultMode)


def activate(state: State) -> None:
    if state.is_active:
        return
    state.is_active = True

    state.unsuppressed_flags = Quartz.CGEventSourceFlagsState(
        Quartz.kCGEventSourceStateHIDSystemState
    )
    Quartz.CGEventTapEnable(state.event_tap, True)


def deactivate(state: State) -> None:
    if not state.is_active:
        return
    state.is_active = False

    Quartz.CGEventTapEnable(state.event_tap, False)


def process(state: State) -> None:
    RUN_LOOP_MAX_DURATION_SECONDS = 1 / 60
    run_loop_duration_seconds = RUN_LOOP_MAX_DURATION_SECONDS

    auto_repeat_state = state.auto_repeat_state
    if auto_repeat_state.auto_repeating_key is not None:
        if auto_repeat_state.has_started_repeating:
            target_interval = Quartz.NSEvent.keyRepeatInterval()
        else:
            target_interval = Quartz.NSEvent.keyRepeatDelay()
        curr_time = time.time()
        elapsed = curr_time - auto_repeat_state.prev_key_press_event_time
        if elapsed < target_interval:
            run_loop_duration_seconds = min(
                target_interval - elapsed, RUN_LOOP_MAX_DURATION_SECONDS
            )
        else:
            auto_repeat_state.has_started_repeating = True
            auto_repeat_state.prev_key_press_event_time = curr_time
            emulate_key_press(state, auto_repeat_state.auto_repeating_key)

    return_after_handling = True
    Quartz.CFRunLoopRunInMode(
        Quartz.kCFRunLoopDefaultMode, run_loop_duration_seconds, return_after_handling
    )


def check_is_modifier_active(
    state: State, modifier_or_optionally_paired_key: KeyId
) -> bool:
    modifier = MODIFIER_BY_OPTIONALLY_PAIRED_KEY[modifier_or_optionally_paired_key]
    return (state.unsuppressed_flags & _MODIFIER_NATIVE_FLAG_MASKS[modifier]) != 0


def _send_event(event: Quartz.CGEventRef) -> None:
    Quartz.CGEventPost(Quartz.kCGAnnotatedSessionEventTap, event)


def _send_flags_changed_event(state: State, associated_key: KeyId) -> None:
    source = None
    event = Quartz.CGEventCreate(source)
    Quartz.CGEventSetType(event, Quartz.NSEventTypeFlagsChanged)
    Quartz.CGEventSetIntegerValueField(
        event, Quartz.kCGKeyboardEventKeycode, _NATIVE_KEY_CODES[associated_key]
    )
    Quartz.CGEventSetFlags(event, state.unsuppressed_flags)
    _send_event(event)


def _send_media_key_event(
    state: State, key: KeyId, is_press: bool, is_repeat: bool = False
) -> None:
    press_or_release_code = 0xA | int(not is_press)
    flags = state.unsuppressed_flags | press_or_release_code << 2
    event_subtype = 8
    nx_key_type = _MEDIA_KEY_NX_KEY_TYPES[key]
    data1 = nx_key_type << 16 | press_or_release_code << 8 | int(is_repeat)
    ns_event = Quartz.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
        Quartz.NSSystemDefined,
        (0, 0),  # location
        flags,
        0,  # timestamp
        0,  # windowNumber
        0,  # context
        event_subtype,
        data1,
        -1,  # data2
    )
    _send_event(ns_event.CGEvent())


def _send_regular_key_event(
    state: State, key: KeyId, is_press: bool, is_repeat: bool = False
) -> None:
    source = None
    event = Quartz.CGEventCreateKeyboardEvent(source, _NATIVE_KEY_CODES[key], is_press)
    Quartz.CGEventSetFlags(event, state.unsuppressed_flags)
    if is_press:
        Quartz.CGEventSetIntegerValueField(
            event,
            Quartz.kCGKeyboardEventAutorepeat,  # spell-checker: disable-line
            int(is_repeat),
        )
    _send_event(event)


def emulate_key_press(
    state: State, key: KeyId, should_emulate_native_repeat: bool = False
) -> None:
    _terminate_auto_repeat_by_new_keyboard_event_if_needed(
        state, key, is_press_event=True
    )
    if (
        state.auto_repeat_state is not None
        and state.auto_repeat_state.has_started_repeating
    ):
        should_emulate_native_repeat = True

    if key == "caps_lock":
        state.unsuppressed_flags ^= _MODIFIER_NATIVE_FLAG_MASKS[key]
        # BUG Sending this event does not change the physical LED on the key, similar to how suppressing the incoming Caps Lock flag change event does not prevent the LED from changing.
        _send_flags_changed_event(state, key)
        return

    if key in SUPPORTED_MODIFIERS:
        state.unsuppressed_flags |= _MODIFIER_NATIVE_FLAG_MASKS[key]
        _send_flags_changed_event(state, key)
        return

    if key in _MEDIA_KEY_NX_KEY_TYPES:
        _send_media_key_event(
            state, key, is_press=True, is_repeat=should_emulate_native_repeat
        )
        return

    _send_regular_key_event(
        state, key, is_press=True, is_repeat=should_emulate_native_repeat
    )


def emulate_key_press_with_auto_repeat(state: State, key: KeyId) -> None:
    state.auto_repeat_state = AutoRepeatState(
        auto_repeating_key=key,
        prev_key_press_event_time=time.time(),
        has_started_repeating=False,
    )

    emulate_key_press(state, key)


def emulate_key_release(state: State, key: KeyId) -> None:
    _terminate_auto_repeat_by_new_keyboard_event_if_needed(
        state, key, is_press_event=False
    )

    if key == "caps_lock":
        # Caps Lock only induces one flag change event natively anyway.
        return

    if key in SUPPORTED_MODIFIERS:
        state.unsuppressed_flags &= ~_MODIFIER_NATIVE_FLAG_MASKS[key]
        _send_flags_changed_event(state, key)
        return

    if key in _MEDIA_KEY_NX_KEY_TYPES:
        _send_media_key_event(state, key, is_press=False)
        return

    _send_regular_key_event(state, key, is_press=False)


def _send_mouse_button_event(state: State, button: ButtonId, is_press: bool) -> None:
    if button == "left_button":
        button_num = Quartz.kCGMouseButtonLeft
        if is_press:
            event_type = Quartz.NSEventTypeLeftMouseDown
        else:
            event_type = Quartz.NSEventTypeLeftMouseUp
    elif button == "right_button":
        button_num = Quartz.kCGMouseButtonRight
        if is_press:
            event_type = Quartz.NSEventTypeRightMouseDown
        else:
            event_type = Quartz.NSEventTypeRightMouseUp
    else:
        raise Exception(f'Mouse button "{button}" is not yet supported on macOS.')

    click_level = _update_click_level_state_with_new_mouse_button_event(
        state.click_level_state, button, is_press
    )

    source = None
    pos = get_cursor_position()
    pos_point = Quartz.CGPointMake(pos.real, pos.imag)
    event = Quartz.CGEventCreateMouseEvent(source, event_type, pos_point, button_num)
    Quartz.CGEventSetFlags(event, state.unsuppressed_flags)
    Quartz.CGEventSetIntegerValueField(
        event, Quartz.kCGMouseEventClickState, click_level
    )
    _send_event(event)


def emulate_mouse_button_press(state: State, button: ButtonId) -> None:
    _send_mouse_button_event(state, button, is_press=True)


def emulate_mouse_button_release(state: State, button: ButtonId) -> None:
    _send_mouse_button_event(state, button, is_press=False)


def get_cursor_position() -> CursorPositionPixels:
    source = None
    return _get_cursor_position_from_event(Quartz.CGEventCreate(source))


@dataclasses.dataclass
class WindowInfo:
    title: str | None
    owner_app_name: str


def get_active_window_info() -> WindowInfo:
    app = (
        Quartz.NSWorkspace.sharedWorkspace().frontmostApplication()  # spell-checker: disable-line
    )
    pid = app.processIdentifier()
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
    )
    title = None
    for window in windows:
        if window[Quartz.kCGWindowOwnerPID] == pid:
            # TODO Getting the window's title requires the Screen Recording permissions or otherwise the "kCGWindowName" never exists. Find a way to verify that this permission is enabled in System Settings and, if not, prompt the user to enable it so that they don't unknowingly get hit with `None` titles all the time.
            title = window.get(Quartz.kCGWindowName, None)
            break

    return WindowInfo(title, app.localizedName())
