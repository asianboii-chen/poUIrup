import wx, wx.adv
import wx


def main():
    app = wx.App()

    frame = wx.Frame(parent=None, style=wx.BORDER_NONE | wx.STAY_ON_TOP)
    frame.SetBackgroundColour(wx.Colour(0x20, 0x20, 0x20, alpha=0x80))
    IMAGE_PATH = '/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/ToggleOff.png'
    bitmap = wx.StaticBitmap(parent=frame, bitmap=wx.Bitmap(IMAGE_PATH))
    frame.SetSize(bitmap.Size)
    frame.Show()

    icon = wx.adv.TaskBarIcon()
    ICON_PATH = '/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/Default.png'
    icon.SetIcon(wx.Icon(ICON_PATH))

    app.MainLoop()


if __name__ == '__main__':
    main()
