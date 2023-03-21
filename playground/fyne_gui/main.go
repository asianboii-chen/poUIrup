/*
Limitations:
	1. The window grabs the input focus while showing up.
	2. There's no way to move the window or have up show up in a secondary monitor.
	3. The Fyne library requires the window to run in the precious main thread.
*/

package main

import (
	"time"

	// "fyne.io/fyne/v2/app"
	// "fyne.io/fyne/v2/canvas"
	// "fyne.io/fyne/v2/container"
	// "fyne.io/fyne/v2/driver/desktop"
	// "fyne.io/fyne/v2/layout"
)

func main() {
	image := canvas.NewImageFromFile(
		"/Users/the-chenergy/repos/jungle/asianboiisui/OldVersions/4.0/Data/ToggleOff.png")
	image.FillMode = canvas.ImageFillOriginal

	demoApp := app.New()
	driver, ok := demoApp.Driver().(desktop.Driver)
	if !ok {
		panic("not a desktop driver")
	}
	window := driver.CreateSplashWindow()
	window.SetContent(container.New(layout.NewMaxLayout(), image))
	go func() {
		time.Sleep(3141592654)
		window.Close()
	}()
	window.ShowAndRun()
}
