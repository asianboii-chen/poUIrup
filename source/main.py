import gui
import util

import threading
import time


def main() -> None:
    running = True

    def exit_app():
        nonlocal running
        running = False

    util.ensure_single_instance(on_new_instance_start=exit_app)

    gui_context = gui.create(gui.Handler(on_menu_click_exit=exit_app))

    def show_indicator_after_delay() -> None:
        time.sleep(2)
        gui.request_show_indicator(gui_context)

    threading.Thread(target=show_indicator_after_delay).start()

    while running:
        gui.process(gui_context)
        time.sleep(1 / 60)  # Handle UI events here


if __name__ == '__main__':
    main()
