"""Opt-in collection wall timings and CUDA stream spans without RNG draws."""

from contextlib import contextmanager, nullcontext
import time

import torch


class CollectionTimer:
    def __init__(self, device, enabled=False):
        self.enabled = enabled
        self.device = torch.device(device)
        self.seconds = {}
        self.native_seconds = {}
        self.counts = {"reset_calls": 0, "reset_worlds": 0, "spawn_attempts": 0, "spawn_fallbacks": 0}
        self.events = []
        self.started = time.perf_counter() if enabled else 0.0

    def phase(self, name, cuda=False):
        return self._measure(name, cuda) if self.enabled else nullcontext()

    @contextmanager
    def _measure(self, name, cuda):
        events = None
        started = time.perf_counter()
        if cuda and self.device.type == "cuda":
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            events[0].record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            if events is not None:
                events[1].record(torch.cuda.current_stream(self.device))
                self.events.append((name, events))
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started

    def count(self, name, amount=1):
        if self.enabled:
            self.counts[name] = self.counts.get(name, 0) + int(amount)

    def finish(self):
        stream_seconds = {}
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        synchronization = time.perf_counter() - started
        for name, (start, end) in self.events:
            stream_seconds[name] = stream_seconds.get(name, 0.0) + start.elapsed_time(end) / 1000
        total = time.perf_counter() - self.started
        accounted = sum(value for name, value in self.seconds.items() if "." not in name)
        return {"version": 1, "seconds": self.seconds, "counts": self.counts,
            "native_seconds": self.native_seconds,
                "cuda_stream_seconds": stream_seconds, "synchronize_seconds": synchronization,
                "total_seconds": total, "unaccounted_seconds": max(0.0, total - accounted - synchronization)}