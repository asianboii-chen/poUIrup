#include "gui.h"

#include <FL/Fl.H>
#include <FL/Fl_Window.H>

#include <chrono>
#include <functional>
#include <utility>

/**
  A helper wrapper to run a function in the main thread. FLTK requires all GUI
  modifications to be done in the main thread. So if you are going to call any
  GUI-related functions, call it with this wrapper instead.

  Example Usage:
    run_on_main_thread(
        [](Fl_Window& window) { window.show(); },
        std::ref(window));
*/
template <typename Function, typename... Args>
static void run_on_main_thread(Function&& function, Args&&... args) {
  auto bound_function = std::bind(function, std::forward<Args>(args)...);
  Fl::add_idle(
      [](void* user_data) {
        auto f = static_cast<decltype(bound_function)*>(user_data);
        (*f)();
        delete f;
      },
      new decltype(bound_function)(std::move(bound_function)));
}

void pouirup::gui::start() {
  //
}

void pouirup::gui::stop() {
  //
}
