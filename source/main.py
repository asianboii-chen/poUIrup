import gui
import hook
import util

import cmath
import collections
import dataclasses
import datetime
import functools
import json
import math
import time
import typing


@dataclasses.dataclass
class KeyboardState:
    pass


@dataclasses.dataclass
class MouseState:
    pressed_buttons: set[hook.ButtonId]
    is_making_gesture: bool
    recorded_position: hook.CursorPositionPixels
    recorded_time: float


@dataclasses.dataclass
class TrackpadState:
    is_making_gesture: bool
    fingers_making_gesture: set[hook.FingerId]
    prev_recorded_finger_positions: dict[hook.FingerId, hook.FingerPositionInches]
    prev_recorded_time: float
    curr_recorded_finger_positions: dict[hook.FingerId, hook.FingerPositionInches]


GestureMovement = typing.Literal["E", "S", "W", "N"]
Gesture = list[GestureMovement]


@dataclasses.dataclass
class GestureState:
    gesture_source: typing.Literal["mouse", "trackpad"] | None
    gesture_recognized: Gesture
    pending_movement: GestureMovement | None
    pending_movement_distance: float
    is_gesture_paused: bool


@dataclasses.dataclass
class TrackerState:
    pressed_keys_and_buttons: set[hook.KeyId | hook.ButtonId]
    prev_mouse_wheel_scroll_time: float
    prev_active_window_info: hook.WindowInfo | None


@dataclasses.dataclass
class AppState:
    is_running: bool
    running_start_time: float


EventTraceKind = typing.Literal[
    "key_press",
    "mouse_button_press",
    "mouse_wheel_scroll",
    "trackpad_gesture",
    "window_change",
]


class EventTrace(typing.TypedDict):
    protocol: str
    kind: EventTraceKind
    timestamp: str


class ModifiableEventTrace(EventTrace):
    modifiers: list[hook.KeyId]


class KeyboardEventTrace(ModifiableEventTrace):
    key: hook.KeyId


class MouseButtonEventTrace(ModifiableEventTrace):
    button: hook.ButtonId
    click_level: int


class MouseWheelEventTrace(ModifiableEventTrace):
    dy: int
    dx: int


class TrackpadGestureEventTrace(EventTrace):
    gesture: str


class WindowChangeEventTrace(EventTrace):
    window: str
    owner_app: str


def _get_active_modifiers(hook_state: hook.State) -> set[hook.KeyId]:
    return {
        modifier
        for modifier in hook.SUPPORTED_MODIFIERS
        if hook.check_is_modifier_active(hook_state, modifier)
    }


def _log_event_trace(trace: EventTrace) -> None:
    print(json.dumps(trace), end=",\n")


def _create_event_trace_metadata(event_kind: EventTraceKind) -> EventTrace:
    trace = EventTrace(
        # FIXME Change this version number whenever protocol changes.
        protocol="v3",
        kind=event_kind,
        timestamp=datetime.datetime.now().astimezone().isoformat(),
    )
    return trace


def _detect_and_log_window_change_if_needed(tracker_state: TrackerState) -> None:
    active_window_info = hook.get_active_window_info()
    if active_window_info != tracker_state.prev_active_window_info:
        tracker_state.prev_active_window_info = active_window_info
        _log_event_trace(
            WindowChangeEventTrace(
                **_create_event_trace_metadata("window_change"),
                window=active_window_info.title or "",
                owner_app=active_window_info.owner_app_name,
            )
        )


def _handle_hook_key_pressing(
    tracker_state: TrackerState,
    hook_state: hook.State,
    key: hook.KeyId,
    is_native_repeat: bool,
) -> hook.ShouldSuppressEventFlag:
    if (
        key not in tracker_state.pressed_keys_and_buttons
        and key not in hook.MODIFIER_BY_OPTIONALLY_PAIRED_KEY
    ):
        _detect_and_log_window_change_if_needed(tracker_state)
        _log_event_trace(
            KeyboardEventTrace(
                **_create_event_trace_metadata("key_press"),
                modifiers=sorted(_get_active_modifiers(hook_state)),
                key=key,
            )
        )
        tracker_state.pressed_keys_and_buttons.add(key)

    return False


def _handle_hook_key_releasing(
    tracker_state: TrackerState,
    key: hook.KeyId,
) -> hook.ShouldSuppressEventFlag:
    tracker_state.pressed_keys_and_buttons.discard(key)
    return False


_MIN_GESTURE_UPDATE_INTERVAL_SECONDS = 1 / 24


_ALLOW_GESTURES_BY_RIGHT_BUTTON = False


def _handle_hook_mouse_button_pressing(
    mouse_state: MouseState,
    tracker_state: TrackerState,
    hook_state: hook.State,
    gesture_state: GestureState,
    button: hook.ButtonId,
    click_level: int,
) -> hook.ShouldSuppressEventFlag:
    if button == "right_button" and click_level == 1:
        if _ALLOW_GESTURES_BY_RIGHT_BUTTON and gesture_state.gesture_source is None:
            mouse_state.is_making_gesture = True
            mouse_state.recorded_position = hook.get_cursor_position()
            mouse_state.recorded_time = time.time()
            gesture_state.gesture_source = "mouse"
            return True

    if button not in tracker_state.pressed_keys_and_buttons:
        _detect_and_log_window_change_if_needed(tracker_state)
        _log_event_trace(
            MouseButtonEventTrace(
                **_create_event_trace_metadata("mouse_button_press"),
                modifiers=sorted(_get_active_modifiers(hook_state)),
                button=button,
                click_level=click_level,
            )
        )
        tracker_state.pressed_keys_and_buttons.add(button)

    return False


def _handle_hook_mouse_button_releasing(
    mouse_state: MouseState,
    tracker_state: TrackerState,
    hook_state: hook.State,
    gesture_state: GestureState,
    button: hook.ButtonId,
    click_level: int,
) -> hook.ShouldSuppressEventFlag:
    if (
        _ALLOW_GESTURES_BY_RIGHT_BUTTON
        and button == "right_button"
        and click_level == 1
    ):
        did_make_gesture = False
        if mouse_state.is_making_gesture:
            if len(gesture_state.gesture_recognized) > 0:
                did_make_gesture = True
                _perform_gestured_action(gesture_state.gesture_recognized)

            mouse_state.is_making_gesture = False
            gesture_state.gesture_source = None
            gesture_state.gesture_recognized.clear()
            gesture_state.pending_movement = None

        if not did_make_gesture:
            hook.emulate_mouse_button_press(hook_state, "right_button")
            hook.emulate_mouse_button_release(hook_state, "right_button")

        return True

    tracker_state.pressed_keys_and_buttons.discard(button)

    return False


def _update_gesture_from_mouse(
    gesture_state: GestureState,
    mouse_state: MouseState,
    new_cursor_position: hook.CursorPositionPixels,
) -> None:
    curr_time = time.time()
    interval = curr_time - mouse_state.recorded_time
    if interval < _MIN_GESTURE_UPDATE_INTERVAL_SECONDS:
        return

    prev_pos = mouse_state.recorded_position
    mouse_state.recorded_position = new_cursor_position
    mouse_state.recorded_time = curr_time

    PIXELS_PER_INCH_MOVED = 216

    displacement_pixels = new_cursor_position - prev_pos
    displacement_inches = displacement_pixels / PIXELS_PER_INCH_MOVED

    _update_gesture_from_new_displacement(gesture_state, displacement_inches, interval)


def _handle_hook_mouse_cursor_moving(
    mouse_state: MouseState,
    gesture_state: GestureState,
    new_pos: hook.CursorPositionPixels,
) -> None:
    if mouse_state.is_making_gesture:
        _update_gesture_from_mouse(gesture_state, mouse_state, new_pos)


_MIN_MOUSE_WHEEL_SCROLL_TRACK_INTERVAL_SECONDS = 0.125


def _handle_hook_mouse_wheel_scrolling(
    tracker_state: TrackerState,
    hook_state: hook.State,
    scroll_displacement: hook.WheelScrollDisplacementUnits,
    is_continuous: bool,
    is_done_by_momentum: bool,
) -> hook.ShouldSuppressEventFlag:
    if not is_done_by_momentum:
        curr_time = time.time()
        interval = curr_time - tracker_state.prev_mouse_wheel_scroll_time
        tracker_state.prev_mouse_wheel_scroll_time = curr_time
        if interval < _MIN_MOUSE_WHEEL_SCROLL_TRACK_INTERVAL_SECONDS:
            return False

        _detect_and_log_window_change_if_needed(tracker_state)

        dy = scroll_displacement.imag
        dx = scroll_displacement.real
        _log_event_trace(
            MouseWheelEventTrace(
                **_create_event_trace_metadata("mouse_wheel_scroll"),
                modifiers=sorted(_get_active_modifiers(hook_state)),
                dy=int(dy > 0) - int(dy < 0),
                dx=int(dx > 0) - int(dx < 0),
            )
        )

    return False


def _update_gesture_from_new_displacement(
    gesture_state: GestureState,
    new_displacement_inches: complex,
    time_since_prev_update: float,
) -> None:
    MIN_SPEED_TO_UPDATE_GESTURE_INCHES_PER_SECOND = 3

    speed = abs(new_displacement_inches) / time_since_prev_update
    if speed < MIN_SPEED_TO_UPDATE_GESTURE_INCHES_PER_SECOND:
        if len(gesture_state.gesture_recognized) > 0:
            gesture_state.is_gesture_paused = True
        return

    direction = (cmath.phase(new_displacement_inches) + math.tau / 8) % math.tau
    movement = ("E", "S", "W", "N")[int(direction / (math.tau / 4))]

    if gesture_state.is_gesture_paused or movement != gesture_state.pending_movement:
        gesture_state.pending_movement = movement
        gesture_state.pending_movement_distance = 0

    gesture_state.pending_movement_distance += abs(new_displacement_inches)

    prev_movement = (
        gesture_state.gesture_recognized[-1]
        if len(gesture_state.gesture_recognized) > 0
        else None
    )
    if not gesture_state.is_gesture_paused and movement == prev_movement:
        return

    MIN_MOVEMENT_DISTANCE_TO_RECOGNIZE_INCHES = 0.25

    if (
        gesture_state.pending_movement_distance
        >= MIN_MOVEMENT_DISTANCE_TO_RECOGNIZE_INCHES
    ):
        gesture_state.gesture_recognized.append(movement)
        gesture_state.is_gesture_paused = False


def _perform_gestured_action(gesture: Gesture) -> None:
    return


_MIN_FINGERS_TO_START_GESTURE_FROM_TRACKPAD = 4


def _update_gesture_from_trackpad(
    gesture_state: GestureState,
    trackpad_state: TrackpadState,
) -> None:
    curr_time = time.time()
    interval = curr_time - trackpad_state.prev_recorded_time
    if interval < _MIN_GESTURE_UPDATE_INTERVAL_SECONDS:
        return

    prev_pos = trackpad_state.prev_recorded_finger_positions
    curr_pos = trackpad_state.curr_recorded_finger_positions

    trackpad_state.prev_recorded_time = curr_time
    trackpad_state.prev_recorded_finger_positions = curr_pos

    MIN_FINGER_SPEED_TO_START_GESTURE_INCHES_PER_SECOND = 3

    if not trackpad_state.is_making_gesture:
        fingers = set()
        for i in curr_pos:
            if i not in prev_pos:
                continue

            speed = abs(curr_pos[i] - prev_pos[i]) / interval
            if speed >= MIN_FINGER_SPEED_TO_START_GESTURE_INCHES_PER_SECOND:
                fingers.add(i)

        if len(fingers) < _MIN_FINGERS_TO_START_GESTURE_FROM_TRACKPAD:
            return

        trackpad_state.is_making_gesture = True
        trackpad_state.fingers_making_gesture = fingers

    sum_displacement = 0
    fingers_moved = 0

    for i in trackpad_state.fingers_making_gesture:
        if i not in curr_pos or i not in prev_pos:
            continue

        sum_displacement += curr_pos[i] - prev_pos[i]
        fingers_moved += 1

    if fingers_moved == 0:
        return

    mean_displacement = sum_displacement / fingers_moved
    _update_gesture_from_new_displacement(gesture_state, mean_displacement, interval)


def _handle_hook_trackpad_finger_positions_updating(
    tracker_state: TrackerState,
    trackpad_state: TrackpadState,
    gesture_state: GestureState,
    new_finger_positions: dict[hook.FingerId, hook.FingerPositionInches],
) -> None:
    trackpad_state.curr_recorded_finger_positions = new_finger_positions

    if trackpad_state.is_making_gesture:
        should_end_gesture = True
        for i in trackpad_state.fingers_making_gesture:
            if i in new_finger_positions:
                should_end_gesture = False
                break

        if should_end_gesture:
            if len(gesture_state.gesture_recognized) > 0:
                _perform_gestured_action(gesture_state.gesture_recognized)

                _detect_and_log_window_change_if_needed(tracker_state)
                _log_event_trace(
                    TrackpadGestureEventTrace(
                        **_create_event_trace_metadata("trackpad_gesture"),
                        gesture="".join(gesture_state.gesture_recognized),
                    )
                )

            trackpad_state.is_making_gesture = False
            gesture_state.gesture_source = None
            gesture_state.gesture_recognized.clear()
            gesture_state.pending_movement = None
            gesture_state.is_gesture_paused = False

            return

        _update_gesture_from_trackpad(gesture_state, trackpad_state)

    if gesture_state.gesture_source is not None:
        return

    if len(new_finger_positions) >= _MIN_FINGERS_TO_START_GESTURE_FROM_TRACKPAD:
        _update_gesture_from_trackpad(gesture_state, trackpad_state)
        if trackpad_state.is_making_gesture:
            gesture_state.gesture_source = "trackpad"


def _handle_gui_icon_menu_showing(hook_state: hook.State) -> None:
    hook.deactivate(hook_state)


def _handle_gui_icon_menu_hid(hook_state: hook.State) -> None:
    hook.activate(hook_state)


def _handle_gui_icon_menu_item_exit_clicked(app_state: AppState) -> None:
    _stop_running_app(app_state)


def _handle_new_app_instance_started(app_state: AppState) -> None:
    _stop_running_app(app_state)


def _stop_running_app(app_state: AppState) -> None:
    app_state.is_running = False


def main() -> None:
    keyboard_state = KeyboardState()
    mouse_state = MouseState(
        pressed_buttons=set(),
        is_making_gesture=False,
        recorded_position=0,
        recorded_time=-math.inf,
    )
    trackpad_state = TrackpadState(
        is_making_gesture=False,
        fingers_making_gesture=set(),
        prev_recorded_finger_positions={},
        prev_recorded_time=-math.inf,
        curr_recorded_finger_positions={},
    )
    gesture_state = GestureState(
        gesture_source=None,
        gesture_recognized=[],
        pending_movement=None,
        pending_movement_distance=0,
        is_gesture_paused=False,
    )

    tracker_state = TrackerState(
        pressed_keys_and_buttons=set(),
        prev_mouse_wheel_scroll_time=-math.inf,
        prev_active_window_info=None,
    )
    _detect_and_log_window_change_if_needed(tracker_state)

    app_state = AppState(is_running=True, running_start_time=time.time())

    hook_state = hook.create()
    hook.register_handler(
        hook_state,
        handler=hook.Handler(
            handle_key_pressing=functools.partial(
                _handle_hook_key_pressing, tracker_state, hook_state
            ),
            handle_key_releasing=functools.partial(
                _handle_hook_key_releasing, tracker_state
            ),
            handle_mouse_button_pressing=functools.partial(
                _handle_hook_mouse_button_pressing,
                mouse_state,
                tracker_state,
                hook_state,
                gesture_state,
            ),
            handle_mouse_button_releasing=functools.partial(
                _handle_hook_mouse_button_releasing,
                mouse_state,
                tracker_state,
                hook_state,
                gesture_state,
            ),
            handle_mouse_cursor_moving=functools.partial(
                _handle_hook_mouse_cursor_moving, mouse_state, gesture_state
            ),
            handle_mouse_wheel_scrolling=functools.partial(
                _handle_hook_mouse_wheel_scrolling, tracker_state, hook_state
            ),
            handle_trackpad_finger_positions_updating=functools.partial(
                _handle_hook_trackpad_finger_positions_updating,
                tracker_state,
                trackpad_state,
                gesture_state,
            ),
        ),
    )
    hook.activate(hook_state)

    gui_state = gui.create()
    gui.register_handler(
        gui_state,
        gui.Handler(
            handle_icon_menu_showing=functools.partial(
                _handle_gui_icon_menu_showing, hook_state
            ),
            handle_icon_menu_hid=functools.partial(
                _handle_gui_icon_menu_hid, hook_state
            ),
            handle_icon_menu_item_exit_clicked=functools.partial(
                _handle_gui_icon_menu_item_exit_clicked, app_state
            ),
        ),
    )

    util.ensure_single_instance(
        handle_new_instance_started=functools.partial(
            _handle_new_app_instance_started, app_state
        )
    )

    while app_state.is_running:
        hook.process(hook_state)
        gui.process(gui_state)

        MAX_RUN_DURATION_SECONDS = 60 << 30
        if time.time() - app_state.running_start_time >= MAX_RUN_DURATION_SECONDS:
            _stop_running_app(app_state)


if __name__ == "__main__":
    main()
