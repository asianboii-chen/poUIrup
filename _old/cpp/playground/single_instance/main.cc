/**
  This program demonstrates how to create a simple single-instance enforcer in
  C++. Its implementation is tested on macOS and is intended to work on Linux
  and Windows as well.

  The program is designed to have at most one instance running at any time.
  Launching a second instance while one is running terminates the old instance.
  This is done by creating a fake lock_file file when an instance starts,
  writing the instance's process ID into that lock_file file. When a new
  instance starts, if the lock_file file exists, it sends a termination signal
  (SIGTERM) to the recorded PID. The old instance can capture that termination
  signal, do its clean-up work, and exit gracefully.

  The lock_file file is fake and not actually (f)locked, which therefore is not
  protected from malicious modifications that can make the single-instance
  control fail. However this risk should be low as the lock_file file is stored
  in a relative hidden place not meant to be easily-accessible to the user.

  Usage:
    1. Compile and launch this program in background.
    2. Launch another instance of this program.
    3. Observe that the old instance terminated gracefully, leaving the new
    instance running normally.
*/

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
    if (lock_file.fail())
      throw std::runtime_error("Failed to create a lock_file file at " +
                               lock_path.string() + ".");

    std::cerr << "Lock file for current process (PID " << getpid()
              << ") was created at " << lock_path.string() << ".\n";
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
