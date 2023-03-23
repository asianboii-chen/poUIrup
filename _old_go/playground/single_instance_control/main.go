package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path"
	"runtime"
	"syscall"
	"time"
)

func main() {
	{ // Ensure single-instance
		switch runtime.GOOS {
		case "darwin":
			lockPath := path.Join(os.TempDir(), "the_great_test_app.lock")
			lock, err := os.OpenFile(lockPath, os.O_RDWR|os.O_CREATE, 0666)
			if err != nil {
				panic(err)
			}
			defer lock.Close()
			tryLock := func() error {
				return syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
			}
			if err := tryLock(); err != nil { // Locking failed; another instance exists
				var runningPID int
				if _, err := fmt.Fscan(lock, &runningPID); err != nil {
					panic(err)
				}
				runningProcess, err := os.FindProcess(runningPID)
				if err != nil {
					panic(err)
				}
				if err := runningProcess.Signal(syscall.SIGTERM); err != nil {
					panic(err)
				}
				func() {
					delay := time.Duration(42 * time.Millisecond)
					const maxDelay = time.Duration(2023 * time.Millisecond)
					for delay <= maxDelay {
						time.Sleep(delay)
						if err := runningProcess.Signal(syscall.Signal(0)); err != nil {
							return
						}
						delay *= 2
					}
					log.Panicf(
						"the already-running process (%v) took too long to exit after being interrupted\n",
						runningPID)
				}()
				if err := tryLock(); err != nil {
					panic(err)
				}
			}
			defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
			if _, err := lock.Seek(0, io.SeekStart); err != nil {
				panic(err)
			}
			fmt.Fprintln(lock, os.Getpid())
			defer os.Remove(lockPath)

		default:
			panic(runtime.GOOS)
		}
	}

	shouldExit := make(chan bool)
	go func() { // Main program logic
		for i := 24; i > 0; i-- {
			if i%4 == 0 {
				fmt.Println(os.Getpid(), "running for", i, "more seconds")
			}
			time.Sleep(1 * time.Second)
		}
		shouldExit <- true
	}()

	exit := func() { // Clean-up work
		fmt.Println(os.Getpid(), "exiting")
	}

	{ // Handle normal exit or sigterm
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, syscall.SIGTERM)
		select {
		case <-sig:
			exit()
			os.Exit(0)
		case <-shouldExit:
			exit()
		}
	}
}
