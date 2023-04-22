import importlib
import typing

Quartz = importlib.import_module('Quartz')


class Context(typing.NamedTuple):
    pass


def create() -> Context:
    def handle_native_event(proxy, event_type, event, user_data):
        EVENT_TYPE_CONSTANT_NAMES = {
            1: 'NSEventTypeLeftMouseDown',
            2: 'NSEventTypeLeftMouseUp',
            3: 'NSEventTypeRightMouseDown',
            4: 'NSEventTypeRightMouseUp',
            5: 'NSEventTypeMouseMoved',
            6: 'NSEventTypeLeftMouseDragged',
            7: 'NSEventTypeRightMouseDragged',
            8: 'NSEventTypeMouseEntered',
            9: 'NSEventTypeMouseExited',
            10: 'NSEventTypeKeyDown',
            11: 'NSEventTypeKeyUp',
            12: 'NSEventTypeFlagsChanged',
            13: 'NSEventTypeAppKitDefined',
            14: 'NSEventTypeSystemDefined',
            15: 'NSEventTypeApplicationDefined',
            16: 'NSEventTypePeriodic',
            17: 'NSEventTypeCursorUpdate',
            18: 'NSEventTypeRotate',
            19: 'NSEventTypeBeginGesture',
            20: 'NSEventTypeEndGesture',
            22: 'NSEventTypeScrollWheel',
            23: 'NSEventTypeTabletPoint',
            24: 'NSEventTypeTabletProximity',
            25: 'NSEventTypeOtherMouseDown',
            26: 'NSEventTypeOtherMouseUp',
            27: 'NSEventTypeOtherMouseDragged',
            29: 'NSEventTypeGesture',
            30: 'NSEventTypeMagnify',
            31: 'NSEventTypeSwipe',
            32: 'NSEventTypeSmartMagnify',
            33: 'NSEventTypeQuickLook',
            34: 'NSEventTypePressure',
            37: 'NSEventTypeDirectTouch',
            38: 'NSEventTypeChangeMode',
        }
        print(EVENT_TYPE_CONSTANT_NAMES.get(event_type, '(unknown)'),
              event_type)
        return event

    event_tap = Quartz.CGEventTapCreate(Quartz.kCGAnnotatedSessionEventTap,
                                        Quartz.kCGHeadInsertEventTap,
                                        Quartz.kCGEventTapOptionListenOnly,
                                        Quartz.kCGEventMaskForAllEvents,
                                        handle_native_event, None)
    loop_source = Quartz.CFMachPortCreateRunLoopSource(None, event_tap, 0)
    event_loop = Quartz.CFRunLoopGetCurrent()
    Quartz.CFRunLoopAddSource(event_loop, loop_source,
                              Quartz.kCFRunLoopDefaultMode)
    Quartz.CGEventTapEnable(event_tap, True)

    return Context()


def process(context: Context) -> None:
    Quartz.CFRunLoopRunInMode(Quartz.kCFRunLoopDefaultMode, 1 / 60, True)
