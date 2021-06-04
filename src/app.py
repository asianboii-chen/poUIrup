'''
poUIrup v4.2.0b
Qianlang Chen
F 05/28/21
'''
from collections import defaultdict
import math
import sys
from threading import Timer
import time
from typing import Callable, DefaultDict, Dict, Set, Tuple

if sys.platform == 'win32':
    from pynput.keyboard import _win32 as keyboard
    from pynput.keyboard._win32 import Key, KeyCode
    from pynput.mouse import _win32 as mouse
    from pynput.mouse._win32 import Button
elif sys.platform == 'darwin':
    from pynput.keyboard import _darwin as keyboard
    from pynput.keyboard._darwin import Key, KeyCode
    from pynput.mouse import _darwin as mouse
    from pynput.mouse._darwin import Button
else:
    from pynput.keyboard import _xorg as keyboard
    from pynput.keyboard._xorg import Key, KeyCode
    from pynput.mouse import _xorg as mouse
    from pynput.mouse._xorg import Button

class App:
    IS_WINDOWS = sys.platform == 'win32'
    IS_MAC_OS = sys.platform == 'darwin'
    IS_LINUX = not IS_WINDOWS and not IS_MAC_OS
    
    def start():
        Ui.configure()
        App.configure()
        Ui.start()
        while Ui._keyboard_listener.is_alive() and Ui._mouse_listener.is_alive():
            time.sleep(.125)
    
    def configure():
        exec(open('config/mac_os_default.py').read())

class Ui:
    # Config begin
    GRAVE = ONE = TWO = THREE = FOUR = FIVE = SIX = SEVEN = EIGHT = NINE = ZERO = DASH = int(-1)
    EQUAL = Q = W = E = R = T = Y = U = I = O = P = LEFT_SQUARE = RIGHT_SQUARE = int(-1)
    BACK_SLASH = A = S = D = F = G = H = J = K = L = SEMICOLON = APOSTROPHE = Z = X = int(-1)
    C = V = B = N = M = COMMA = DOT = SLASH = ALT = ALT_R = BACKSPACE = CAPS_LOCK = int(-1)
    CMD = CMD_R = CTRL = CTRL_R = DELETE = DOWN = END = ENTER = ESC = F1 = F2 = F3 = int(-1)
    F4 = F5 = F6 = F7 = F8 = F9 = F10 = F11 = F12 = HOME = LEFT = PAGE_DOWN = PAGE_UP = int(-1)
    RIGHT = SHIFT = SHIFT_R = SPACE = TAB = UP = int(-1)
    _IGNORED_KEYS: Set[int] = None
    # mods -> (in_key -> (out_key, should_press_shift))
    _CHARACTER_LAYOUT: Tuple[Dict[int, Tuple[int, bool]]] = None
    _SHIFT_LOCKED_KEYS: Set[int] = None
    # in_key -> func(is_repetition)
    _EXECUTION_LAYOUT: Dict[int, Callable[[bool], None]] = None
    _KEY_MASK = -1
    _MIN_STICKY_TRIGGER_DURATION = math.nan
    _MAX_STICKY_TRIGGER_DURATION = math.nan
    _STICKY_DURATION = math.nan
    # in_key -> func(is_repetition)
    _FUNCTION_LAYOUT: Dict[int, Callable[[bool], None]] = None
    _MAX_DOUBLE_CLICK_INTERVAL = math.nan
    # Config end
    FN = 0x100
    SPEC = 0x101
    TOG = 0x102
    _MODS: Tuple[int] = (Key.shift.value.vk, Key.ctrl.value.vk, Key.alt.value.vk,
                         Key.cmd.value.vk, FN, SPEC, TOG)
    
    pressed_mods: Set[int] = set()
    pressed_stickies: Set[int] = set()
    shift_lock = False
    _keyboard: keyboard.Controller = None
    _keyboard_listener: keyboard.Listener = None
    _mouse: mouse.Controller = None
    _mouse_listener: mouse.Listener = None
    # is_press -> (time_pressed, out_key, tap_key, is_sticky)
    _pressed_keys: Dict[int, Tuple[float, int, int, bool]] = {}
    # (time_pressed, in_key, out_key)
    _last_key_press: Tuple[float, int, int] = (0., -1, -1)
    _last_mod = -1
    # mod -> num_currently_pressed
    # Modifiers like Fn can have num_currently_pressed up to 2 since there are two Fn keys (left
    # and right)
    _pressed_count_by_mod: DefaultDict[int, int] = defaultdict(int)
    _pressed_continuous_mods: Set[int] = set()
    # (time_pressed, button, level (1 = single click, 2 = double click, etc.))
    _last_button_press: Tuple[float, Button, int] = (0., None, 1)
    
    def configure():
        for key in Key:
            const_name = str(key).split('.')[1].upper()
            if hasattr(Ui, const_name): setattr(Ui, const_name, key.value.vk)
    
    def start():
        Ui._keyboard = keyboard.Controller()
        Ui._keyboard_listener = keyboard.Listener(on_press=Ui._handle_keyboard_press,
                                                  on_release=Ui._handle_keyboard_release,
                                                  suppress=True)
        Ui._keyboard_listener.start()
        Ui._keyboard_listener.wait()
        Ui._mouse = mouse.Controller()
        Ui._mouse_listener = mouse.Listener(on_click=Ui._handle_mouse_click,
                                            on_move=Ui._handle_mouse_move,
                                            on_scroll=Ui._handle_mouse_scroll,
                                            suppress=True)
        Ui._mouse_listener.start()
        
        Ui._mouse_listener.wait()
        # Sending some events here solves a weird Pynput lazy-import bug with the mouse
        Ui.touch_key(True, Ui._KEY_MASK)
        Ui.touch_key(False, Ui._KEY_MASK)
        print('Ready (press F11 to quit)')
    
    def press_char(in_key: int):
        in_shift = Ui.SHIFT in Ui.pressed_mods
        if Ui.shift_lock and in_key in Ui._SHIFT_LOCKED_KEYS:
            in_shift = not in_shift
        out_key, out_shift = Ui._CHARACTER_LAYOUT[in_shift][in_key]
        Ui.press_combo(out_key, {Ui.SHIFT} if out_shift else set())
        Ui._record_pressed_key(in_key, out_key, -1, False)
        Ui.release_stickies()
    
    def press_dual(in_key: int, out_press_mod: int, out_tap_key: int, is_sticky: bool):
        Ui.pressed_mods.add(out_press_mod)
        Ui._pressed_count_by_mod[out_press_mod] += 1
        Ui.press_key(in_key, out_press_mod, out_tap_key, is_sticky, False)
    
    def press_sequence(*sequence: Tuple[int, Set[int]]):
        Ui.release_stickies()
        for out_key, out_mods in sequence:
            Ui.press_combo(out_key, out_mods)
            Ui.touch_key(False, out_key)
    
    def press_combo(out_key: int, out_mods: Set[int]):
        Ui.touch_mods(True, out_mods)
        Ui.touch_key(True, out_key)
        Ui.touch_mods(False, out_mods)
    
    def press_key(in_key: int,
                  out_press_key: int,
                  out_tap_key: int = -1,
                  is_sticky: bool = False,
                  should_release_stickies: bool = True):
        Ui.touch_key(True, out_press_key)
        Ui._record_pressed_key(in_key, out_press_key, out_tap_key, is_sticky)
        if should_release_stickies: Ui.release_stickies()
    
    def touch_mods(should_press: bool, out_mods: Set[int]):
        for mod in Ui._MODS:
            if (mod in out_mods) != (mod in Ui.pressed_mods):
                Ui.touch_key(should_press == (mod in out_mods), mod)
    
    def touch_key(should_press: bool, out_key: int):
        if not Ui._is_virtual(out_key):
            Ui._keyboard.touch(KeyCode.from_vk(out_key), should_press)
    
    def release_stickies():
        if not Ui.pressed_stickies: return
        print('Releasing and masking stickies')
        Ui.touch_key(True, Ui._KEY_MASK)
        Ui.touch_key(False, Ui._KEY_MASK)
        for mod in Ui.pressed_stickies: Ui.touch_key(False, mod)
        Ui.pressed_stickies.clear()
        Ui.pressed_mods.clear()
        Ui._pressed_count_by_mod.clear()
        Ui._pressed_continuous_mods.clear()
        Ui._last_key_press = (0., -1, -1)
    
    def touch_button(should_press: bool, button: Button):
        if should_press:
            last_time, last_button, last_level = Ui._last_button_press
            curr_time = time.time()
            if (button == last_button and
                    time.time() - last_time < Ui._MAX_DOUBLE_CLICK_INTERVAL):
                level = last_level + 1
            else:
                level = 1
            if level == 1: Ui._mouse.press(button)
            else: Ui._mouse.click(button, level)
            Ui._last_button_press = (curr_time, button, level)
        else:
            Ui._mouse.release(button)
        Ui._last_key_press = (0., -1, -1)
        Ui.release_stickies()
    
    def stop():
        print('Quitting...')
        Ui._keyboard_listener.stop()
        Ui._mouse_listener.stop()
    
    def _is_virtual(key): return key >= 0x100
    
    def _get_key(key_obj):
        if key_obj in Ui._IGNORED_KEYS:
            Ui._last_key_press = (0., -1, -1)
            return -1
        return key_obj.value.vk if isinstance(key_obj, Key) else key_obj.vk
    
    def _record_pressed_key(in_key, out_press_key, out_tap_key, is_sticky):
        time_pressed = time.time()
        Ui._pressed_keys[in_key] = (time_pressed, out_press_key, out_tap_key, is_sticky)
        Ui._last_key_press = (time_pressed, in_key, out_press_key)
    
    def _handle_keyboard_press(key_obj):
        in_key = Ui._get_key(key_obj)
        if in_key == -1: return # Ignored key
        if in_key == Ui.F11:
            Ui.stop()
            return
        is_repetition = in_key in Ui._pressed_keys or in_key == Ui._last_key_press[1]
        if not is_repetition: Ui._last_key_press = (time.time(), in_key, -1)
        
        if (in_key in Ui._EXECUTION_LAYOUT or
                in_key in Ui._FUNCTION_LAYOUT and Ui.FN in Ui.pressed_mods):
            # Modifier, function, or other special key
            active_layout = (Ui._EXECUTION_LAYOUT
                             if in_key in Ui._EXECUTION_LAYOUT else Ui._FUNCTION_LAYOUT)
            active_layout[in_key](is_repetition)
        elif in_key in Ui._CHARACTER_LAYOUT[0] and not (Ui.pressed_mods - {Ui.SHIFT}):
            # Character
            Ui.press_char(in_key)
        else:
            # Normal press
            Ui.press_key(in_key, in_key)
    
    def _handle_keyboard_release(key_obj):
        in_key = Ui._get_key(key_obj)
        if in_key not in Ui._pressed_keys: return
        time_pressed, out_press_key, out_tap_key, is_sticky = Ui._pressed_keys.pop(in_key)
        time_released = time.time()
        # Whether this key was pressed and released without any other keyboard or mouse events
        # in between
        is_continuous = Ui._last_key_press[1] == in_key
        if out_press_key in Ui._MODS: is_continuous |= Ui._last_key_press[2] in Ui._MODS
        should_reset_last_press = is_continuous
        
        if Ui.pressed_stickies:
            max_duration = math.inf
        elif len(Ui.pressed_mods) == 1:
            max_duration = Ui._MAX_STICKY_TRIGGER_DURATION
        else:
            max_duration = 0 # there are other mods pressed: sticky disallowed
        if (is_sticky and is_continuous and
                Ui._MIN_STICKY_TRIGGER_DURATION <= time_released - time_pressed < max_duration):
            # Sticky
            print('Pressing stickies:', Ui._pressed_continuous_mods | {out_press_key})
            Ui.pressed_stickies.add(out_press_key)
            for mod in Ui._pressed_continuous_mods:
                Ui.pressed_mods.add(mod)
                Ui.pressed_stickies.add(mod)
                Ui.touch_key(True, mod)
            last = Ui._last_key_press[1]
            should_reset_last_press = False
            Timer(Ui._STICKY_DURATION,
                  lambda: Ui._last_key_press[1] == last and Ui.release_stickies()).start()
        else:
            # Normal release
            if out_press_key in Ui._MODS and Ui._pressed_count_by_mod[out_press_key] > 0:
                Ui._pressed_count_by_mod[out_press_key] -= 1
                if Ui._pressed_count_by_mod[out_press_key] == 0:
                    Ui.pressed_mods.remove(out_press_key)
                    Ui.touch_key(False, out_press_key)
                if not Ui.pressed_mods:
                    Ui._pressed_continuous_mods.clear()
                elif is_continuous:
                    Ui._pressed_continuous_mods.add(out_press_key)
                    should_reset_last_press = False
            else:
                Ui.touch_key(False, out_press_key)
            
            if (out_tap_key != -1 and is_continuous and
                    time_released - time_pressed < Ui._MIN_STICKY_TRIGGER_DURATION):
                # Tap
                Ui.touch_key(True, out_tap_key)
                Ui.touch_key(False, out_tap_key)
                Ui.release_stickies()
                should_reset_last_press = True
        
        if should_reset_last_press: Ui._last_key_press = (0., -1, -1)
    
    def _handle_mouse_click(x, y, button, is_press):
        Ui.touch_button(is_press, button)
    
    def _handle_mouse_move(x, y):
        # print('Move x', x, 'y', y)
        pass
    
    def _handle_mouse_scroll(x, y, dx, dy):
        # print('Scroll x', x, 'y', y, 'dx', dx, 'dy', dy)
        pass

if __name__ == '__main__': App.start()
