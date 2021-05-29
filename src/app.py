'''
poUIrup v4.2.0b
Qianlang Chen
F 05/28/21
'''
from collections import defaultdict
import math
import sys
from threading import Thread, Timer
import time
from typing import Callable, DefaultDict, Dict, Set, Tuple

if sys.platform == 'win32':
    from pynput.keyboard import _win32 as keyboard
    from pynput.keyboard._win32 import Key, KeyCode
    from pynput.mouse import _win32 as mouse
elif sys.platform == 'darwin':
    from pynput.keyboard import _darwin as keyboard
    from pynput.keyboard._darwin import Key, KeyCode
    from pynput.mouse import _darwin as mouse
else:
    from pynput.keyboard import _xorg as keyboard
    from pynput.keyboard._xorg import Key, KeyCode
    from pynput.mouse import _xorg as mouse

class App:
    IS_WINDOWS = sys.platform == 'win32'
    IS_MAC_OS = sys.platform == 'darwin'
    IS_LINUX = not IS_WINDOWS and not IS_MAC_OS
    
    def start():
        Keyboard.configure()
        Thread(target=Keyboard.start).start()
        # Thread(target=Mouse.start).start()

class Keyboard:
    # Config begin
    GRAVE = ONE = TWO = THREE = FOUR = FIVE = SIX = SEVEN = EIGHT = NINE = ZERO = DASH = int(-1)
    EQUAL = Q = W = E = R = T = Y = U = I = O = P = LEFT_SQUARE = RIGHT_SQUARE = int(-1)
    BACK_SLASH = A = S = D = F = G = H = J = K = L = SEMICOLON = APOSTROPHE = Z = X = int(-1)
    C = V = B = N = M = COMMA = DOT = FORWARD_SLASH = ALT = ALT_R = BACKSPACE = int(-1)
    CAPS_LOCK = CMD = CMD_R = CTRL = CTRL_R = DELETE = DOWN = END = ENTER = ESC = F1 = int(-1)
    F2 = F3 = F4 = F5 = F6 = F7 = F8 = F9 = F10 = F11 = F12 = HOME = LEFT = PAGE_DOWN = int(-1)
    PAGE_UP = RIGHT = SHIFT = SHIFT_R = SPACE = TAB = UP = int(-1)
    # mods -> (in_key -> (out_key, should_press_shift))
    _NORMAL_LAYOUT: Tuple[Dict[int, Tuple[int, bool]]] = None
    _SHIFT_LOCKED_KEYS: Set[int] = None
    # in_key -> (can_repeat, should_release_stickies, func)
    _EXECUTION_LAYOUT: Dict[int, Tuple[bool, bool, Callable[[Set[int]], None]]] = None
    _KEY_MASK = -1
    _MIN_STICKY_TRIGGER_DURATION = math.nan
    _MAX_STICKY_TRIGGER_DURATION = math.nan
    _STICKY_DURATION = math.nan
    # in_key -> (can_repeat, should_release_stickies, func)
    _FUNCTION_LAYOUT: Dict[int, Tuple[bool, bool, Callable[[Set[int]], None]]] = None
    # Config end
    FN = 0x100
    SPEC = 0x101
    TOG = 0x102
    _MODS: Tuple[int] = (Key.shift.value.vk, Key.ctrl.value.vk, Key.alt.value.vk,
                         Key.cmd.value.vk, FN, SPEC, TOG)
    
    pressed_mods: Set[int] = set()
    shift_lock = False
    _controller: keyboard.Controller = None
    _listener: keyboard.Listener = None
    # is_press -> (time_pressed, out_key, tap_key, is_sticky)
    _pressed_keys: Dict[int, Tuple[float, int, int, bool]] = {}
    # (time, in_key)
    _last_press: Tuple[float, int] = (0., -1)
    _last_mod = -1
    # mod -> num_currently_pressed
    # Modifiers like Fn can have num_currently_pressed up to 2 since there are two Fn keys (left
    # and right)
    _pressed_count_by_mod: DefaultDict[int, int] = defaultdict(int)
    _has_stickies = False
    
    def configure():
        print(Key.media_volume_down, Key.media_volume_down.value,
              Key.media_volume_down.value.vk)
        for key in Key:
            const_name = str(key).split('.')[1].upper()
            if hasattr(Keyboard, const_name): setattr(Keyboard, const_name, key.value.vk)
        
        exec(open('config/mac_os_default.py').read())
    
    def start():
        Keyboard._controller = keyboard.Controller()
        Keyboard._listener = keyboard.Listener(on_press=Keyboard._handle_listener_press,
                                               on_release=Keyboard._handle_listener_release,
                                               suppress=True)
        Keyboard._listener.start()
        Keyboard._listener.wait()
        print('Ready (press F11 to quit)')
        Keyboard._listener.join()
    
    def press_char(in_key: int):
        in_shift = Keyboard.SHIFT in Keyboard.pressed_mods
        if Keyboard.shift_lock and in_key in Keyboard._SHIFT_LOCKED_KEYS:
            in_shift = not in_shift
        out_key, out_shift = Keyboard._NORMAL_LAYOUT[in_shift][in_key]
        Keyboard.press_combo(out_key, {Keyboard.SHIFT} if out_shift else set())
        Keyboard._pressed_keys[in_key] = (time.time(), out_key, -1, False)
    
    def press_dual(in_key: int, out_press_mod: int, out_tap_key: int, is_sticky: bool):
        Keyboard.pressed_mods.add(out_press_mod)
        Keyboard._pressed_count_by_mod[out_press_mod] += 1
        Keyboard.press_key(in_key, out_press_mod, out_tap_key, is_sticky)
        Keyboard._last_mod = in_key
    
    def press_sequence(*args: Tuple[int, Set[int]]):
        for out_key, out_mods in args:
            Keyboard.press_combo(out_key, out_mods)
            Keyboard.touch_key(False, out_key)
    
    def press_combo(out_key: int, out_mods: Set[int]):
        Keyboard.touch_mods(True, out_mods)
        Keyboard.touch_key(True, out_key)
        Keyboard.touch_mods(False, out_mods)
    
    def press_key(in_key: int,
                  out_press_key: int,
                  out_tap_key: int = -1,
                  is_sticky: bool = False):
        Keyboard.touch_key(True, out_press_key)
        Keyboard._pressed_keys[in_key] = (time.time(), out_press_key, out_tap_key, is_sticky)
    
    def touch_mods(should_press: bool, out_mods: Set[int]):
        for mod in Keyboard._MODS:
            if (mod in out_mods) != (mod in Keyboard.pressed_mods):
                Keyboard.touch_key(should_press == (mod in out_mods), mod)
    
    def touch_key(should_press: bool, out_key: int):
        if Keyboard._is_virtual(out_key): return
        Keyboard._controller.touch(KeyCode.from_vk(out_key), should_press)
    
    def stop():
        print('Quitting...')
        Keyboard._listener.stop()
    
    def _is_virtual(key): return key >= 0x100
    
    def _get_key(key_obj): return key_obj.value.vk if isinstance(key_obj, Key) else key_obj.vk
    
    def _release_pressed_mods():
        if not Keyboard.pressed_mods: return
        if Keyboard._has_stickies:
            print('Masking')
            Keyboard.touch_key(True, Keyboard._KEY_MASK)
            Keyboard.touch_key(False, Keyboard._KEY_MASK)
        for mod in Keyboard.pressed_mods: Keyboard.touch_key(False, mod)
        Keyboard.pressed_mods.clear()
        Keyboard._pressed_count_by_mod.clear()
        Keyboard._has_stickies = False
    
    def _handle_listener_press(key_obj):
        in_key = Keyboard._get_key(key_obj)
        if in_key == Keyboard.F11:
            Keyboard.stop()
            return
        is_repetition = in_key in Keyboard._pressed_keys or in_key == Keyboard._last_press[1]
        if not is_repetition: Keyboard._last_press = (time.time(), in_key)
        should_release_stickies = True
        
        if in_key in Keyboard._EXECUTION_LAYOUT or (in_key in Keyboard._FUNCTION_LAYOUT and
                                                    Keyboard.FN in Keyboard.pressed_mods):
            # Modifier, function, or other special key
            active_layout = (Keyboard._EXECUTION_LAYOUT if in_key in Keyboard._EXECUTION_LAYOUT
                             else Keyboard._FUNCTION_LAYOUT)
            can_repeat, should_release_stickies, func = active_layout[in_key]
            if not can_repeat and is_repetition: return
            func(Keyboard.pressed_mods)
        elif in_key in Keyboard._NORMAL_LAYOUT[0] and not (Keyboard.pressed_mods -
                                                           {Keyboard.SHIFT}):
            # Character
            Keyboard.press_char(in_key)
        else:
            # Normal press
            Keyboard.touch_key(True, in_key)
            Keyboard._pressed_keys[in_key] = (time.time(), in_key, -1, False)
        
        if Keyboard._has_stickies and should_release_stickies:
            Keyboard._release_pressed_mods()
    
    def _handle_listener_release(key_obj):
        in_key = Keyboard._get_key(key_obj)
        if in_key not in Keyboard._pressed_keys: return # Pynput bug? Why is this possible
        time_pressed, out_press_key, out_tap_key, is_sticky = Keyboard._pressed_keys.pop(in_key)
        time_released = time.time()
        # Whether this key was pressed and released without any other keyboard or mouse events
        # in between
        is_continuous = Keyboard._last_press[1] == in_key
        
        if (is_sticky and is_continuous and
            (Keyboard._has_stickies or Keyboard._MIN_STICKY_TRIGGER_DURATION <=
             time_released - time_pressed < Keyboard._MAX_STICKY_TRIGGER_DURATION)):
            # Sticky
            print(f'Applying sticky 0x{out_press_key:02x}')
            Keyboard._has_stickies = True
            Timer(Keyboard._STICKY_DURATION, lambda:
                  (Keyboard._last_mod == in_key and Keyboard._release_pressed_mods())).start()
        else:
            # Normal release
            if out_press_key in Keyboard._MODS:
                Keyboard._pressed_count_by_mod[out_press_key] -= 1
                if sum(Keyboard._pressed_count_by_mod.values()) == 0:
                    Keyboard._release_pressed_mods()
            else:
                Keyboard.touch_key(False, out_press_key)
            if (out_tap_key != -1 and is_continuous and
                    time_released - time_pressed < Keyboard._MIN_STICKY_TRIGGER_DURATION):
                # Tap
                Keyboard.touch_key(True, out_tap_key)
                Keyboard.touch_key(False, out_tap_key)
        
        if is_continuous: Keyboard._last_press = (0., -1)

class Mouse:
    _controller: mouse.Controller
    _listener: mouse.Listener
    
    def start():
        Mouse._controller = mouse.Controller()
        Mouse._listener = mouse.Listener(on_click=Mouse._handle_controller_click,
                                         on_move=Mouse._handle_controller_move,
                                         on_scroll=Mouse._handle_controller_scroll)
        Mouse._listener.start()
        Mouse._listener.join()
    
    def _handle_controller_click(x, y, button, press):
        print('Click x', x, 'y', y, 'button', button, 'press', press)
    
    def _handle_controller_move(x, y):
        print('Move x', x, 'y', y)
    
    def _handle_controller_scroll(x, y, dx, dy):
        print('Scroll x', x, 'y', y, 'dx', dx, 'dy', dy)

if __name__ == '__main__': App.start()
