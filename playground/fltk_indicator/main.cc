#include <FL/Fl.H>
#include <FL/Fl_Window.H>

#include <chrono>
#include <iostream>
#include <thread>

int main() {
  // Helper function to run a function in the main thread.
  // (FLTK requires all GUI modifications to be done in main thread.)
  const auto run_on_main_thread = [](auto&& function, auto&&... args) {
    auto bound_function =
        std::bind(function, std::forward<decltype(args)>(args)...);
    Fl::add_idle(
        [](void* user_data) {
          auto f = static_cast<decltype(bound_function)*>(user_data);
          (*f)();
          delete f;
        },
        new decltype(bound_function)(std::move(bound_function)));
  };

  auto window = std::make_unique<Fl_Window>(420, 420, "The great test app");
  bool finished = false;

  std::thread show_window([&window, &finished, &run_on_main_thread] {
    std::cout << "Running; window will show in a moment...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    run_on_main_thread(
        [](Fl_Window& window, bool& finished) {
          std::cout << "Showing window...\n";
          window.show();
          Fl::run();
          finished = true;
        },
        std::ref(*window), std::ref(finished));
  });

  while (!finished) Fl::wait(.03125);
  show_window.join();
}
