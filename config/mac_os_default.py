if bool(False): from app import Key, Set, Tuple, Ui

## VIRTUAL KEY CODES (not defined by pynput) ###################################################

Ui.A = 0x00
Ui.S = 0x01
Ui.D = 0x02
Ui.F = 0x03
Ui.H = 0x04
Ui.G = 0x05
Ui.Z = 0x06
Ui.X = 0x07
Ui.C = 0x08
Ui.V = 0x09
Ui.B = 0x0b
Ui.Q = 0x0c
Ui.W = 0x0d
Ui.E = 0x0e
Ui.R = 0x0f
Ui.Y = 0x10
Ui.T = 0x11
Ui.ONE = 0x12
Ui.TWO = 0x13
Ui.THREE = 0x14
Ui.FOUR = 0x15
Ui.SIX = 0x16
Ui.FIVE = 0x17
Ui.EQUAL = 0x18
Ui.NINE = 0x19
Ui.SEVEN = 0x1a
Ui.DASH = 0x1b
Ui.EIGHT = 0x1c
Ui.ZERO = 0x1d
Ui.RIGHT_SQUARE = 0x1e
Ui.O = 0x1f
Ui.U = 0x20
Ui.LEFT_SQUARE = 0x21
Ui.I = 0x22
Ui.P = 0x23
Ui.L = 0x25
Ui.J = 0x26
Ui.APOSTROPHE = 0x27
Ui.K = 0x28
Ui.SEMICOLON = 0x29
Ui.BACK_SLASH = 0x2a
Ui.COMMA = 0x2b
Ui.SLASH = 0x2c
Ui.N = 0x2d
Ui.M = 0x2e
Ui.DOT = 0x2f
Ui.GRAVE = 0x32

## NORMAL LAYOUT ###############################################################################

# in_key -> (out_key, should_press_shift)
Ui._CHARACTER_LAYOUT = (
    # Not shifted
    {
        Ui.GRAVE: (Ui.THREE, True),
        Ui.ONE: (Ui.FOUR, True),
        Ui.TWO: (Ui.SEVEN, True),
        Ui.THREE: (Ui.EIGHT, True),
        Ui.FOUR: (Ui.EQUAL, False),
        Ui.FIVE: (Ui.LEFT_SQUARE, True),
        Ui.SIX: (Ui.RIGHT_SQUARE, True),
        Ui.SEVEN: (Ui.COMMA, True),
        Ui.EIGHT: (Ui.DOT, True),
        Ui.NINE: (Ui.NINE, True),
        Ui.ZERO: (Ui.ZERO, True),
        Ui.DASH: (Ui.LEFT_SQUARE, False),
        Ui.EQUAL: (Ui.RIGHT_SQUARE, False),
        Ui.Q: (Ui.APOSTROPHE, False),
        Ui.W: (Ui.COMMA, False),
        Ui.E: (Ui.DOT, False),
        Ui.R: (Ui.P, False),
        Ui.T: (Ui.Y, False),
        Ui.Y: (Ui.F, False),
        Ui.U: (Ui.G, False),
        Ui.I: (Ui.C, False),
        Ui.O: (Ui.R, False),
        Ui.P: (Ui.L, False),
        Ui.LEFT_SQUARE: (Ui.SLASH, False),
        Ui.RIGHT_SQUARE: (Ui.BACK_SLASH, False),
        Ui.CAPS_LOCK: (Ui.DASH, False),
        Ui.A: (Ui.A, False),
        Ui.S: (Ui.O, False),
        Ui.D: (Ui.E, False),
        Ui.F: (Ui.I, False),
        Ui.G: (Ui.U, False),
        Ui.H: (Ui.D, False),
        Ui.J: (Ui.H, False),
        Ui.K: (Ui.T, False),
        Ui.L: (Ui.N, False),
        Ui.SEMICOLON: (Ui.S, False),
        Ui.APOSTROPHE: (Ui.DASH, True),
        Ui.Z: (Ui.SEMICOLON, False),
        Ui.X: (Ui.Q, False),
        Ui.C: (Ui.J, False),
        Ui.V: (Ui.K, False),
        Ui.B: (Ui.X, False),
        Ui.N: (Ui.B, False),
        Ui.M: (Ui.M, False),
        Ui.COMMA: (Ui.W, False),
        Ui.DOT: (Ui.V, False),
        Ui.SLASH: (Ui.Z, False),
    },
    # Shifted
    {
        Ui.GRAVE: (Ui.GRAVE, False),
        Ui.ONE: (Ui.ONE, False),
        Ui.TWO: (Ui.TWO, False),
        Ui.THREE: (Ui.THREE, False),
        Ui.FOUR: (Ui.FOUR, False),
        Ui.FIVE: (Ui.FIVE, False),
        Ui.SIX: (Ui.SIX, False),
        Ui.SEVEN: (Ui.SEVEN, False),
        Ui.EIGHT: (Ui.EIGHT, False),
        Ui.NINE: (Ui.NINE, False),
        Ui.ZERO: (Ui.ZERO, False),
        Ui.DASH: (Ui.SIX, True),
        Ui.EQUAL: (Ui.GRAVE, True),
        Ui.Q: (Ui.APOSTROPHE, True),
        Ui.W: (Ui.SLASH, True),
        Ui.E: (Ui.ONE, True),
        Ui.R: (Ui.P, True),
        Ui.T: (Ui.Y, True),
        Ui.Y: (Ui.F, True),
        Ui.U: (Ui.G, True),
        Ui.I: (Ui.C, True),
        Ui.O: (Ui.R, True),
        Ui.P: (Ui.L, True),
        Ui.LEFT_SQUARE: (Ui.FIVE, True),
        Ui.RIGHT_SQUARE: (Ui.TWO, True),
        Ui.CAPS_LOCK: (Ui.EQUAL, True),
        Ui.A: (Ui.A, True),
        Ui.S: (Ui.O, True),
        Ui.D: (Ui.E, True),
        Ui.F: (Ui.I, True),
        Ui.G: (Ui.U, True),
        Ui.H: (Ui.D, True),
        Ui.J: (Ui.H, True),
        Ui.K: (Ui.T, True),
        Ui.L: (Ui.N, True),
        Ui.SEMICOLON: (Ui.S, True),
        Ui.APOSTROPHE: (Ui.BACK_SLASH, True),
        Ui.Z: (Ui.SEMICOLON, True),
        Ui.X: (Ui.Q, True),
        Ui.C: (Ui.J, True),
        Ui.V: (Ui.K, True),
        Ui.B: (Ui.X, True),
        Ui.N: (Ui.B, True),
        Ui.M: (Ui.M, True),
        Ui.COMMA: (Ui.W, True),
        Ui.DOT: (Ui.V, True),
        Ui.SLASH: (Ui.Z, True),
    },
)

Ui._SHIFT_LOCKED_KEYS = {
    Ui.ONE,
    Ui.TWO,
    Ui.THREE,
    Ui.FOUR,
    Ui.FIVE,
    Ui.SIX,
    Ui.SEVEN,
    Ui.EIGHT,
    Ui.NINE,
    Ui.ZERO,
    Ui.R,
    Ui.T,
    Ui.Y,
    Ui.U,
    Ui.I,
    Ui.O,
    Ui.P,
    Ui.A,
    Ui.S,
    Ui.D,
    Ui.F,
    Ui.G,
    Ui.H,
    Ui.J,
    Ui.K,
    Ui.L,
    Ui.SEMICOLON,
    Ui.X,
    Ui.C,
    Ui.V,
    Ui.B,
    Ui.N,
    Ui.M,
    Ui.COMMA,
    Ui.DOT,
    Ui.SLASH,
}

Ui._IGNORED_KEYS = {x for x in Key if str(x).startswith('Key.media_')}

## MODIFIER AND SPECIAL KEY LAYOUT #############################################################

_target_layout = Ui._EXECUTION_LAYOUT

def _press_key(in_key: int, out_key: int):
    global _target_layout
    _target_layout[in_key] = lambda _is_repetition: Ui.press_key(in_key, out_key)

def _press_sequence(in_key: int, should_override_mods: bool, *sequence: Tuple[int, Set[int]]):
    def f(_is_repetition):
        if should_override_mods: Ui.press_sequence(sequence)
        else: Ui.press_sequence(*((key, mods | Ui.pressed_mods) for key, mods in sequence))
    
    global _target_layout
    _target_layout[in_key] = f

def _press_dual(in_key: int, out_press_mod: int, out_tap_key: int, is_sticky: bool):
    def f(is_repetition):
        if not is_repetition:
            Ui.press_dual(in_key, out_press_mod, out_tap_key, is_sticky)
        elif out_press_mod in Ui.pressed_stickies:
            Ui.press_sequence((out_tap_key, Ui.pressed_mods))
    
    global _target_layout
    _target_layout[in_key] = f

def _press_f_key(in_key: int, *sequence: Tuple[int, Set[int]]):
    def f(_is_repetition):
        if Ui.pressed_mods: Ui.press_key(in_key, in_key)
        else: Ui.press_sequence(*sequence)
    
    global _target_layout
    _target_layout[in_key] = f

def _press_backspace(in_key: int):
    def f(_is_repetition):
        if Ui.CMD in Ui.pressed_mods:
            Ui.press_sequence((Ui.BACKSPACE, {Ui.ALT}))
        elif Ui.FN in Ui.pressed_mods:
            Ui.press_sequence((Ui.LEFT, {Ui.SHIFT, Ui.CMD}), (Ui.BACKSPACE, set()))
        else:
            Ui.press_key(in_key, Ui.BACKSPACE)
    
    global _target_layout
    _target_layout[in_key] = f

_press_dual(Ui.SPACE, Ui.SHIFT, Ui.SPACE, True)
_press_dual(Ui.SHIFT, Ui.CMD, Ui.ESC, True)
_press_dual(Ui.CTRL, Ui.FN, -1, True)
_press_dual(Ui.ALT, Ui.CTRL, -1, True)
_press_dual(Ui.CMD, Ui.ALT, -1, True)
_press_dual(Ui.CTRL_R, Ui.FN, Ui.TAB, False)
_press_dual(Ui.DOWN, Ui.FN, Ui.ENTER, True)
_press_key(Ui.ALT_R, Ui.LEFT)
Ui._AUTO_REPEAT_KEYS.add(Ui.ALT_R)
_press_key(Ui.LEFT, Ui.DOWN)
_press_backspace(Ui.CMD_R)
Ui._AUTO_REPEAT_KEYS.add(Ui.CMD_R)
_press_backspace(Ui.BACKSPACE)

_press_f_key(Ui.F1, (Ui.LEFT_SQUARE, {Ui.SHIFT, Ui.CMD}))
_press_f_key(Ui.F2, (Ui.RIGHT_SQUARE, {Ui.SHIFT, Ui.CMD}))
_press_f_key(Ui.F3, (Ui.F, {Ui.CMD}), (Ui.G, {Ui.CMD}))
_press_f_key(Ui.F5, (Ui.R, {Ui.CMD}), (Ui.GRAVE, {Ui.CTRL}), (Ui.F5, {}))
_press_f_key(Ui.F9, (Ui.GRAVE, {Ui.CTRL}), (Ui.F9, {}))

# AKA modifier cancellation key, the key pressed to pretend modifiers aren't used for a sticky
Ui._KEY_MASK = Ui.CMD_R

## FUNCTION LAYOUT #############################################################################

_target_layout = Ui._FUNCTION_LAYOUT

_press_sequence(Ui.A, False, (Ui.LEFT, {Ui.CMD}))
_press_sequence(Ui.G, False, (Ui.RIGHT, {Ui.CMD}))
_press_key(Ui.E, Ui.UP)
_press_key(Ui.S, Ui.LEFT)
_press_key(Ui.D, Ui.DOWN)
_press_key(Ui.F, Ui.RIGHT)
_press_sequence(Ui.W, False, (Ui.LEFT, {Ui.ALT}))
_press_sequence(Ui.R, False, (Ui.RIGHT, {Ui.ALT}))

_press_sequence(Ui.UP, False)

## NUMERICAL ###################################################################################

Ui._MIN_STICKY_TRIGGER_DURATION = .125
Ui._MAX_STICKY_TRIGGER_DURATION = .625
Ui._STICKY_DURATION = 1.
Ui._AUTO_REPEAT_DELAY = .5
Ui._AUTO_REPEAT_INTERVAL = .03125
Ui._MAX_DOUBLE_CLICK_INTERVAL = .5
