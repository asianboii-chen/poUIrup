import wx
import wx.adv

import dataclasses
import functools
import typing


@dataclasses.dataclass
class State:
    app: wx.App
    indicator: wx.Frame
    icon: wx.adv.TaskBarIcon


@dataclasses.dataclass
class Handler:
    handle_menu_exit_clicked: typing.Callable[[], None]


_MENU_ITEM_TOGGLE = wx.NewId()
_MENU_ITEM_EXIT = wx.NewId()


def _handle_icon_event(state: State, handler: Handler, event: wx.CommandEvent) -> None:
    event_id = event.GetId()
    if event_id == _MENU_ITEM_TOGGLE:
        if state.indicator.IsShown():
            state.indicator.Hide()
        else:
            state.indicator.Show()
    elif event_id == _MENU_ITEM_EXIT:
        handler.handle_menu_exit_clicked()


def _create_icon_popup_menu(state: State, handler: Handler) -> wx.Menu:
    menu = wx.Menu()
    menu.Bind(wx.EVT_MENU, functools.partial(_handle_icon_event, state, handler))

    menu.Append(_MENU_ITEM_TOGGLE, "Toggle")
    menu.AppendSeparator()
    menu.Append(_MENU_ITEM_EXIT, "Exit")
    return menu


def create(handler: Handler) -> State:
    app = wx.App()
    indicator = wx.Frame(parent=None, style=wx.STAY_ON_TOP)
    icon = wx.adv.TaskBarIcon(wx.adv.TBI_CUSTOM_STATUSITEM)

    state = State(app, indicator, icon)

    INDICATOR_BITMAP_PATH = "/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/ToggleOff.png"
    indicator_bitmap = wx.StaticBitmap(
        parent=indicator, bitmap=wx.Bitmap(INDICATOR_BITMAP_PATH)
    )
    indicator.Size = indicator_bitmap.Size
    indicator.SetTransparent(127)

    # BUG At least on macOS, the icon has this default behavior that, when it's clicked and activated (it doesn't matter if the menu actually shows up), it blocks the event loop until it's deactivated specifically by clicking on the icon again (not anywhere outside). Blocking the event loop here prevents the hook from working, so no UI events can be processed when the icon is activated at all, just like when a secure text field is focused.
    #
    # This bug is mysteriously amplified by the presence of the hook: the icon can't be deactivated within about one second of activation, regardless of how long the hook yields or at all. If the hook is not created at all, then this problem disappears. (The blocking still persists.)
    ICON_IMAGE_PATH = (
        "/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/Default.png"
    )
    icon.SetIcon(wx.Icon(ICON_IMAGE_PATH))
    icon.CreatePopupMenu = functools.partial(_create_icon_popup_menu, state, handler)

    return state


def process(state: State) -> None:
    state.app.Yield(onlyIfNeeded=True)


def request_show_indicator(state: State) -> None:
    wx.CallAfter(state.indicator.Show)


def request_hide_indicator(state: State) -> None:
    wx.CallAfter(state.indicator.Hide)
