#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

int main() {
  // Ensure single instance.
  [] {
    const auto lock_path =
        std::filesystem::path(std::getenv("TMPDIR")) / "pouirup.lock";
    if (std::filesystem::exists(lock_path)) {
      // Terminate the already-running instance.
      [&lock_path] {
        std::ifstream lock_file(lock_path);
        int running_pid;
        lock_file >> running_pid;
        if (lock_file.fail()) return;
        if (kill(running_pid, SIGTERM) != 0) return;
        for (int wait_ms = 125; wait_ms <= 2000; wait_ms *= 2) {
          std::this_thread::sleep_for(std::chrono::milliseconds(wait_ms));
          if (kill(running_pid, 0) != 0) return;
        }
        throw std::runtime_error("The existing process (PID " +
                                 std::to_string(running_pid) +
                                 ") took too long to exit.");
      }();
    }

    std::ofstream lock_file(lock_path);
    lock_file << getpid() << std::endl;
    if (lock_file.fail()) {
      throw std::runtime_error("Failed to create a lock_file file at " +
                               lock_path.string() + ".");
    }
  }();

  // Handle termination due to starting a second instance.
  [] {
    std::signal(SIGTERM, [](int /* sig */) {
      std::cout << "(PID " << getpid()
                << ") Exiting due to a second process starting...\n";
      exit(0);
    });
  }();

  // Program's main logic.
  [] {
    for (int t = 24; t > 0; t--) {
      if (t % 4 == 0)
        std::cout << "(PID " << getpid() << ") Running for " << t
                  << " more seconds...\n";
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "(PID " << getpid() << ") Exiting...\n";
  }();
}
