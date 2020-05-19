try:
    from torch.multiprocessing import *
    from torch.multiprocessing import start_processes
except ImportError as e:
    # XXX In case of torch unavailable
    from multiprocessing import *
    import sys, signal
    import ctypes, ctypes.util
    from ml import logging
    if platform.system() == "Windows":
        path_libc = ctypes.util.find_library("msvcrt")
    else:
        path_libc = ctypes.util.find_library("c")

    try:
        libc = ctypes.CDLL(path_libc)
    except OSError as e:
        logging.error("Unable to load the system C library: {e}")
        sys.exit(1)

    _supports_context = sys.version_info >= (3, 4)
    def _python_version_check():
        if not _supports_context:
            raise RuntimeError("Requires python 3.4 or higher to use "
                               "ml.multiprocessing.spawn and "
                               "ml.multiprocessing.ProcessContext helper "
                               "to launch multiple processes.")

    def _wrap(fn, i, args, error_queue):
        # prctl(2) is a Linux specific system call.
        # On other systems the following function call has no effect.
        # This is set to ensure that non-daemonic child processes can
        # terminate if their parent terminates before they do.
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGINT.value)
        try:
            fn(i, *args)
        except KeyboardInterrupt:
            pass  # SIGINT; Killed by parent, do nothing
        except Exception:
            # Propagate exception to parent process, keeping original traceback
            import traceback
            error_queue.put(traceback.format_exc())
            sys.exit(1)

    class ProcessContext:
        def __init__(self, processes, error_queues):
            _python_version_check()
            self.error_queues = error_queues
            self.processes = processes
            self.sentinels = {
                process.sentinel: index
                for index, process in enumerate(processes)
            }

        def pids(self):
            return [int(process.pid) for process in self.processes]

        def join(self, timeout=None):
            r"""
            Tries to join one or more processes in this spawn context.
            If one of them exited with a non-zero exit status, this function
            kills the remaining processes and raises an exception with the cause
            of the first process exiting.
            Returns ``True`` if all processes have been joined successfully,
            ``False`` if there are more processes that need to be joined.
            Arguments:
                timeout (float): Wait this long before giving up on waiting.
            """
            # Ensure this function can be called even when we're done.
            if len(self.sentinels) == 0:
                return True

            # Wait for any process to fail or all of them to succeed.
            ready = connection.wait(
                self.sentinels.keys(),
                timeout=timeout,
            )

            error_index = None
            for sentinel in ready:
                index = self.sentinels.pop(sentinel)
                process = self.processes[index]
                process.join()
                if process.exitcode != 0:
                    error_index = index
                    break

            # Return if there was no error.
            if error_index is None:
                # Return whether or not all processes have been joined.
                return len(self.sentinels) == 0

            # Assume failure. Terminate processes that are still alive.
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

            # There won't be an error on the queue if the process crashed.
            if self.error_queues[error_index].empty():
                exitcode = self.processes[error_index].exitcode
                if exitcode < 0:
                    name = signal.Signals(-exitcode).name
                    raise Exception(
                        "process %d terminated with signal %s" %
                        (error_index, name)
                    )
                else:
                    raise Exception(
                        "process %d terminated with exit code %d" %
                        (error_index, exitcode)
                    )

            original_trace = self.error_queues[error_index].get()
            msg = f"\n\n-- Process {error_index} terminated with the following error:\n"
            msg += original_trace
            raise Exception(msg)

    def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
        _python_version_check()
        mp = get_context(start_method)
        error_queues = []
        processes = []
        for i in range(nprocs):
            error_queue = SimpleQueue()
            process = Process(
                target=_wrap,
                args=(fn, i, args, error_queue),
                daemon=daemon,
            )
            process.start()
            error_queues.append(error_queue)
            processes.append(process)

        context = ProcessContext(processes, error_queues)
        if not join:
            return context

        # Loop on join until it returns True or raises an exception.
        while not context.join():
            pass
