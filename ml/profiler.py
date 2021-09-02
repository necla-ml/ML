from line_profiler import LineProfiler as Profiler

import contextlib
import time

import torch
from torch.profiler import (
    profile,
    schedule,
    tensorboard_trace_handler,
    record_function,
    ProfilerActivity
)

@contextlib.contextmanager
def profile_time(trace_name,
                name,
                enabled=True,
                stream=None,
                end_stream=None):
    """Print time spent by CPU and GPU.
    Useful as a temporary context manager to find sweet spots of code
    suitable for async implementation.
    From: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/utils/profiling.py
    Usage:
        >> with profile_time('test_trace', 'test_infer') as pt:
        >>    model(inputs)
    """
    if (not enabled) or not torch.cuda.is_available():
        yield
        return
    stream = stream if stream else torch.cuda.current_stream()
    end_stream = end_stream if end_stream else stream
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream.record_event(start)
    try:
        cpu_start = time.monotonic()
        yield
    finally:
        cpu_end = time.monotonic()
        end_stream.record_event(end)
        end.synchronize()
        cpu_time = (cpu_end - cpu_start) * 1000
        gpu_time = start.elapsed_time(end)
        msg = f'{trace_name} {name} cpu_time {cpu_time:.2f} ms '
        msg += f'gpu_time {gpu_time:.2f} ms stream {stream}'
        print(msg, end_stream)

# FIXME: workaround to prevent inference getting stuck inside the profiler context
with torch.profiler.profile() as p:
        pass

class TorchProfiler:
    def __init__(self, 
                record_func_name='inference',
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                record_shapes=False, 
                profile_memory=True,
                scheduler=schedule(wait=1, warmup=1, active=2),
                trace_handler=tensorboard_trace_handler('./log')
                ):
        self.activities = activities
        self.profile = profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_flops=True,
            schedule=scheduler,
            on_trace_ready=trace_handler
        )
        self.record_function = record_function(record_func_name)

    def run(self, func, *args, num_iter=5, **kwargs):
        with self.profile as p:
            for i in range(num_iter):
                func(*args, **kwargs)
                p.step()