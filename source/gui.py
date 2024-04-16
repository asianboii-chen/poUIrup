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
    handle_icon_menu_showing: typing.Callable[[], None]
    handle_icon_menu_hid: typing.Callable[[], None]
    handle_icon_menu_item_exit_clicked: typing.Callable[[], None]


def _handle_icon_menu_hid(handler: Handler, event: wx.MenuEvent) -> None:
    # BUG At least on macOS, even after the menu closed, sometimes the icon stays in the active state (like waiting for the menu to close), which therefore keeps the thread blocked and prevents the hook from resuming working. When that happens, the user must click on the icon again to force it to deactivate.
    wx.CallAfter(handler.handle_icon_menu_hid)


_ICON_MENU_ITEM_TOGGLE = wx.NewId()
_ICON_MENU_ITEM_EXIT = wx.NewId()


def _handle_icon_menu_item_clicked(
    state: State, handler: Handler, event: wx.CommandEvent
) -> None:
    event_id = event.GetId()
    if event_id == _ICON_MENU_ITEM_TOGGLE:
        if state.indicator.IsShown():
            state.indicator.Hide()
        else:
            state.indicator.Show()
    elif event_id == _ICON_MENU_ITEM_EXIT:
        handler.handle_icon_menu_item_exit_clicked()


def _create_icon_menu(state: State, handler: Handler) -> wx.Menu:
    menu = wx.Menu()
    menu.Bind(wx.EVT_MENU_OPEN, lambda *args: handler.handle_icon_menu_showing())
    menu.Bind(wx.EVT_MENU_CLOSE, functools.partial(_handle_icon_menu_hid, handler))
    menu.Bind(
        wx.EVT_MENU, functools.partial(_handle_icon_menu_item_clicked, state, handler)
    )

    menu.Append(_ICON_MENU_ITEM_TOGGLE, "Toggle")
    menu.AppendSeparator()
    menu.Append(_ICON_MENU_ITEM_EXIT, "Exit")
    return menu


def create() -> State:
    app = wx.App()

    indicator = wx.Frame(parent=None, style=wx.STAY_ON_TOP)
    indicator.SetTransparent(127)
    INDICATOR_BITMAP_PATH = "/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/ToggleOff.png"
    indicator_bitmap = wx.StaticBitmap(
        parent=indicator, bitmap=wx.Bitmap(INDICATOR_BITMAP_PATH)
    )
    indicator.Size = indicator_bitmap.Size

    icon = wx.adv.TaskBarIcon(wx.adv.TBI_CUSTOM_STATUSITEM)
    ICON_IMAGE_PATH = (
        "/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/Default.png"
    )
    icon.SetIcon(wx.Icon(ICON_IMAGE_PATH))

    return State(app, indicator, icon)


def register_handler(state: State, handler: Handler) -> None:
    # BUG At least on macOS, the icon has this default behavior that, when it's clicked and activated (it doesn't matter if the menu actually shows up), it blocks the event loop until it's deactivated specifically by clicking on the icon again (not anywhere outside). Blocking the event loop here prevents the hook from working, so no UI events can be processed when the icon is activated at all, just like when a secure text field is focused.
    #
    # This bug is mysteriously amplified by the presence of the hook: the icon can't be deactivated within about one second of activation, regardless of how long the hook yields or at all. If the hook is not created at all, then this problem disappears. (The blocking still persists.) So, the app currently alleviates this bug by temporarily deactivating the hook while the icon's menu shows up.
    state.icon.CreatePopupMenu = functools.partial(_create_icon_menu, state, handler)


def process(state: State) -> None:
    state.app.Yield(onlyIfNeeded=True)
