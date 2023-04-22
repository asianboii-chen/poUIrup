import gui
import mapper
import util


def main() -> None:
    running = True

    def exit_app():
        nonlocal running
        running = False

    util.ensure_single_instance(on_new_instance_start=exit_app)

    mapper_context = mapper.create()
    gui_context = gui.create(gui.Handler(on_menu_click_exit=exit_app))

    while running:
        gui.process(gui_context)
        mapper.process(mapper_context)


if __name__ == '__main__':
    main()
