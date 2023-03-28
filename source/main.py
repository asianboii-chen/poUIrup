import gui

import threading
import time


def main() -> None:
    running = True

    def handle_menu_click_exit() -> None:
        nonlocal running
        running = False

    gui_context = gui.create(gui.Handler(handle_menu_click_exit))

    def show_indicator_after_delay() -> None:
        time.sleep(2)
        gui.request_show_indicator(gui_context)

    threading.Thread(target=show_indicator_after_delay).start()

    while running:
        gui.process(gui_context)
        time.sleep(1 / 60)  # Handle UI events here


if __name__ == '__main__':
    main()
