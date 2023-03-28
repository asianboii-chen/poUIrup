import wx
import wx.adv

import typing


class Context(typing.NamedTuple):
    app: wx.App
    indicator: wx.Frame
    indicator_bitmap: wx.StaticBitmap
    icon: wx.adv.TaskBarIcon


class Handler(typing.NamedTuple):
    on_menu_click_exit: typing.Callable[[], None]


def create(handler: Handler) -> Context:
    app = wx.App()

    # Indicator
    indicator = wx.Frame(parent=None, style=wx.STAY_ON_TOP)
    INDICATOR_BITMAP_PATH = '/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/ToggleOff.png'
    indicator_bitmap = wx.StaticBitmap(parent=indicator,
                                       bitmap=wx.Bitmap(INDICATOR_BITMAP_PATH))
    indicator.Size = indicator_bitmap.Size

    # Task bar icon
    icon = wx.adv.TaskBarIcon(wx.adv.TBI_CUSTOM_STATUSITEM)
    ICON_IMAGE_PATH = '/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/Default.png'
    icon.SetIcon(wx.Icon(ICON_IMAGE_PATH))

    def create_icon_popup_menu() -> wx.Menu:
        nonlocal handler

        MENU_ITEM_TOGGLE = wx.NewId()
        MENU_ITEM_EXIT = wx.NewId()

        menu = wx.Menu()
        menu.Append(MENU_ITEM_TOGGLE, 'Toggle')
        menu.AppendSeparator()
        menu.Append(MENU_ITEM_EXIT, 'Exit')

        def handle_event(event: wx.CommandEvent) -> None:
            nonlocal handler
            event_id = event.GetId()
            if event_id == MENU_ITEM_TOGGLE:
                print('Clicked on "toggle" (not yet implemented)')
            elif event_id == MENU_ITEM_EXIT:
                handler.on_menu_click_exit()

        menu.Bind(wx.EVT_MENU, handle_event)
        return menu

    icon.CreatePopupMenu = create_icon_popup_menu

    return Context(app, indicator, indicator_bitmap, icon)


def process(context: Context) -> None:
    context.app.Yield(onlyIfNeeded=True)


def request_show_indicator(context: Context) -> None:
    wx.CallAfter(context.indicator.Show)
