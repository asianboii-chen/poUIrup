if bool(False): from app import Key, Keyboard, Set, Tuple

## VIRTUAL KEY CODES (not defined by pynput) ###################################################

Keyboard.A = 0x00
Keyboard.S = 0x01
Keyboard.D = 0x02
Keyboard.F = 0x03
Keyboard.H = 0x04
Keyboard.G = 0x05
Keyboard.Z = 0x06
Keyboard.X = 0x07
Keyboard.C = 0x08
Keyboard.V = 0x09
Keyboard.B = 0x0b
Keyboard.Q = 0x0c
Keyboard.W = 0x0d
Keyboard.E = 0x0e
Keyboard.R = 0x0f
Keyboard.Y = 0x10
Keyboard.T = 0x11
Keyboard.ONE = 0x12
Keyboard.TWO = 0x13
Keyboard.THREE = 0x14
Keyboard.FOUR = 0x15
Keyboard.SIX = 0x16
Keyboard.FIVE = 0x17
Keyboard.EQUAL = 0x18
Keyboard.NINE = 0x19
Keyboard.SEVEN = 0x1a
Keyboard.DASH = 0x1b
Keyboard.EIGHT = 0x1c
Keyboard.ZERO = 0x1d
Keyboard.RIGHT_SQUARE = 0x1e
Keyboard.O = 0x1f
Keyboard.U = 0x20
Keyboard.LEFT_SQUARE = 0x21
Keyboard.I = 0x22
Keyboard.P = 0x23
Keyboard.L = 0x25
Keyboard.J = 0x26
Keyboard.APOSTROPHE = 0x27
Keyboard.K = 0x28
Keyboard.SEMICOLON = 0x29
Keyboard.BACK_SLASH = 0x2a
Keyboard.COMMA = 0x2b
Keyboard.FORWARD_SLASH = 0x2c
Keyboard.N = 0x2d
Keyboard.M = 0x2e
Keyboard.DOT = 0x2f
Keyboard.GRAVE = 0x32

## NORMAL LAYOUT ###############################################################################

# in_key -> (out_key, should_press_shift)
Keyboard._NORMAL_LAYOUT = (
    # Not shifted
    {
        Keyboard.GRAVE: (Keyboard.THREE, True),
        Keyboard.ONE: (Keyboard.FOUR, True),
        Keyboard.TWO: (Keyboard.SEVEN, True),
        Keyboard.THREE: (Keyboard.EIGHT, True),
        Keyboard.FOUR: (Keyboard.EQUAL, False),
        Keyboard.FIVE: (Keyboard.LEFT_SQUARE, True),
        Keyboard.SIX: (Keyboard.RIGHT_SQUARE, True),
        Keyboard.SEVEN: (Keyboard.COMMA, True),
        Keyboard.EIGHT: (Keyboard.DOT, True),
        Keyboard.NINE: (Keyboard.NINE, True),
        Keyboard.ZERO: (Keyboard.ZERO, True),
        Keyboard.DASH: (Keyboard.LEFT_SQUARE, False),
        Keyboard.EQUAL: (Keyboard.RIGHT_SQUARE, False),
        Keyboard.Q: (Keyboard.APOSTROPHE, False),
        Keyboard.W: (Keyboard.COMMA, False),
        Keyboard.E: (Keyboard.DOT, False),
        Keyboard.R: (Keyboard.P, False),
        Keyboard.T: (Keyboard.Y, False),
        Keyboard.Y: (Keyboard.F, False),
        Keyboard.U: (Keyboard.G, False),
        Keyboard.I: (Keyboard.C, False),
        Keyboard.O: (Keyboard.R, False),
        Keyboard.P: (Keyboard.L, False),
        Keyboard.LEFT_SQUARE: (Keyboard.FORWARD_SLASH, False),
        Keyboard.RIGHT_SQUARE: (Keyboard.BACK_SLASH, False),
        Keyboard.CAPS_LOCK: (Keyboard.DASH, False),
        Keyboard.A: (Keyboard.A, False),
        Keyboard.S: (Keyboard.O, False),
        Keyboard.D: (Keyboard.E, False),
        Keyboard.F: (Keyboard.I, False),
        Keyboard.G: (Keyboard.U, False),
        Keyboard.H: (Keyboard.D, False),
        Keyboard.J: (Keyboard.H, False),
        Keyboard.K: (Keyboard.T, False),
        Keyboard.L: (Keyboard.N, False),
        Keyboard.SEMICOLON: (Keyboard.S, False),
        Keyboard.APOSTROPHE: (Keyboard.DASH, True),
        Keyboard.Z: (Keyboard.SEMICOLON, False),
        Keyboard.X: (Keyboard.Q, False),
        Keyboard.C: (Keyboard.J, False),
        Keyboard.V: (Keyboard.K, False),
        Keyboard.B: (Keyboard.X, False),
        Keyboard.N: (Keyboard.B, False),
        Keyboard.M: (Keyboard.M, False),
        Keyboard.COMMA: (Keyboard.W, False),
        Keyboard.DOT: (Keyboard.V, False),
        Keyboard.FORWARD_SLASH: (Keyboard.Z, False),
    },
    # Shifted
    {
        Keyboard.GRAVE: (Keyboard.GRAVE, False),
        Keyboard.ONE: (Keyboard.ONE, False),
        Keyboard.TWO: (Keyboard.TWO, False),
        Keyboard.THREE: (Keyboard.THREE, False),
        Keyboard.FOUR: (Keyboard.FOUR, False),
        Keyboard.FIVE: (Keyboard.FIVE, False),
        Keyboard.SIX: (Keyboard.SIX, False),
        Keyboard.SEVEN: (Keyboard.SEVEN, False),
        Keyboard.EIGHT: (Keyboard.EIGHT, False),
        Keyboard.NINE: (Keyboard.NINE, False),
        Keyboard.ZERO: (Keyboard.ZERO, False),
        Keyboard.DASH: (Keyboard.SIX, True),
        Keyboard.EQUAL: (Keyboard.GRAVE, True),
        Keyboard.Q: (Keyboard.APOSTROPHE, True),
        Keyboard.W: (Keyboard.FORWARD_SLASH, True),
        Keyboard.E: (Keyboard.ONE, True),
        Keyboard.R: (Keyboard.P, True),
        Keyboard.T: (Keyboard.Y, True),
        Keyboard.Y: (Keyboard.F, True),
        Keyboard.U: (Keyboard.G, True),
        Keyboard.I: (Keyboard.C, True),
        Keyboard.O: (Keyboard.R, True),
        Keyboard.P: (Keyboard.L, True),
        Keyboard.LEFT_SQUARE: (Keyboard.FIVE, True),
        Keyboard.RIGHT_SQUARE: (Keyboard.TWO, True),
        Keyboard.CAPS_LOCK: (Keyboard.EQUAL, True),
        Keyboard.A: (Keyboard.A, True),
        Keyboard.S: (Keyboard.O, True),
        Keyboard.D: (Keyboard.E, True),
        Keyboard.F: (Keyboard.I, True),
        Keyboard.G: (Keyboard.U, True),
        Keyboard.H: (Keyboard.D, True),
        Keyboard.J: (Keyboard.H, True),
        Keyboard.K: (Keyboard.T, True),
        Keyboard.L: (Keyboard.N, True),
        Keyboard.SEMICOLON: (Keyboard.S, True),
        Keyboard.APOSTROPHE: (Keyboard.BACK_SLASH, True),
        Keyboard.Z: (Keyboard.SEMICOLON, True),
        Keyboard.X: (Keyboard.Q, True),
        Keyboard.C: (Keyboard.J, True),
        Keyboard.V: (Keyboard.K, True),
        Keyboard.B: (Keyboard.X, True),
        Keyboard.N: (Keyboard.B, True),
        Keyboard.M: (Keyboard.M, True),
        Keyboard.COMMA: (Keyboard.W, True),
        Keyboard.DOT: (Keyboard.V, True),
        Keyboard.FORWARD_SLASH: (Keyboard.Z, True),
    },
)

Keyboard._SHIFT_LOCKED_KEYS = {
    Keyboard.ONE,
    Keyboard.TWO,
    Keyboard.THREE,
    Keyboard.FOUR,
    Keyboard.FIVE,
    Keyboard.SIX,
    Keyboard.SEVEN,
    Keyboard.EIGHT,
    Keyboard.NINE,
    Keyboard.ZERO,
    Keyboard.R,
    Keyboard.T,
    Keyboard.Y,
    Keyboard.U,
    Keyboard.I,
    Keyboard.O,
    Keyboard.P,
    Keyboard.A,
    Keyboard.S,
    Keyboard.D,
    Keyboard.F,
    Keyboard.G,
    Keyboard.H,
    Keyboard.J,
    Keyboard.K,
    Keyboard.L,
    Keyboard.SEMICOLON,
    Keyboard.X,
    Keyboard.C,
    Keyboard.V,
    Keyboard.B,
    Keyboard.N,
    Keyboard.M,
    Keyboard.COMMA,
    Keyboard.DOT,
    Keyboard.FORWARD_SLASH,
}

Keyboard._IGNORED_KEYS = {x for x in Key if str(x).startswith('Key.media_')}

## MODIFIER AND SPECIAL KEY LAYOUT #############################################################

Keyboard._EXECUTION_LAYOUT = _target_layout = {}

def _press_key(in_key: int, out_key: int):
    global _target_layout
    _target_layout[in_key] = lambda _, _1: Keyboard.press_key(in_key, out_key)

def _press_sequence(in_key: int, should_override_mods: bool, *args: Tuple[int, Set[int]]):
    def f(pressed_mods, _):
        if should_override_mods:
            Keyboard.press_sequence(args)
        else:
            Keyboard.press_sequence(*((key, mods | pressed_mods) for key, mods in args))
    
    global _target_layout
    _target_layout[in_key] = f

def _press_dual(in_key: int, out_press_mod: int, out_tap_key: int, is_sticky: bool):
    def f(pressed_mods, is_repetition):
        if not is_repetition:
            Keyboard.press_dual(in_key, out_press_mod, out_tap_key, is_sticky)
        elif out_press_mod in Keyboard._pressed_stickies:
            Keyboard.press_sequence((out_tap_key, pressed_mods))
            Keyboard._release_stickies()
    
    global _target_layout
    _target_layout[in_key] = f

def _press_backspace(in_key: int):
    def f(pressed_mods, _):
        if Keyboard.CMD in pressed_mods:
            Keyboard.press_sequence((Keyboard.BACKSPACE, {Keyboard.ALT}))
        elif Keyboard.FN in pressed_mods:
            Keyboard.press_sequence((Keyboard.LEFT, {Keyboard.SHIFT, Keyboard.CMD}),
                                    (Keyboard.BACKSPACE, set()))
        else:
            Keyboard.press_key(in_key, Keyboard.BACKSPACE)
    
    global _target_layout
    _target_layout[in_key] = f

_press_dual(Keyboard.SPACE, Keyboard.SHIFT, Keyboard.SPACE, True)
_press_dual(Keyboard.SHIFT, Keyboard.CMD, Keyboard.ESC, True)
_press_dual(Keyboard.CTRL, Keyboard.FN, -1, True)
_press_dual(Keyboard.ALT, Keyboard.CTRL, -1, True)
_press_dual(Keyboard.CMD, Keyboard.ALT, -1, True)
_press_dual(Keyboard.CTRL_R, Keyboard.FN, Keyboard.TAB, True)
_press_dual(Keyboard.DOWN, Keyboard.FN, Keyboard.ENTER, True)
_press_key(Keyboard.ALT_R, Keyboard.LEFT)
_press_key(Keyboard.LEFT, Keyboard.DOWN)
_press_backspace(Keyboard.CMD_R)
_press_backspace(Keyboard.BACKSPACE)

# AKA modifier cancellation key, the key pressed to pretend modifiers aren't used for a sticky
Keyboard._KEY_MASK = Keyboard.CMD_R

## FUNCTION LAYOUT #############################################################################

Keyboard._FUNCTION_LAYOUT = _target_layout = {}

_press_sequence(Keyboard.A, False, (Keyboard.LEFT, {Keyboard.CMD}))
_press_sequence(Keyboard.G, False, (Keyboard.RIGHT, {Keyboard.CMD}))
_press_key(Keyboard.E, Keyboard.UP)
_press_key(Keyboard.S, Keyboard.LEFT)
_press_key(Keyboard.D, Keyboard.DOWN)
_press_key(Keyboard.F, Keyboard.RIGHT)
_press_sequence(Keyboard.W, False, (Keyboard.LEFT, {Keyboard.ALT}))
_press_sequence(Keyboard.R, False, (Keyboard.RIGHT, {Keyboard.ALT}))

## TIMING ######################################################################################

Keyboard._MIN_STICKY_TRIGGER_DURATION = .125
Keyboard._MAX_STICKY_TRIGGER_DURATION = .625
Keyboard._STICKY_DURATION = 1.
