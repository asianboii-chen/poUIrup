import importlib
import typing

Quartz = importlib.import_module('Quartz')


class Context(typing.NamedTuple):
    pass


def create() -> Context:
    def handle_native_event(proxy, event_type, event, user_data):
        print(
            f'{proxy=} {type(proxy)=} {event_type=} {type(event_type)=} {event=} {type(event)=} {user_data=} {type(user_data)=}'
        )
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
