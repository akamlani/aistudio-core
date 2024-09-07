import platform
import multiprocessing
import psutil
import logging
import datetime
import gc
from   typing import Any, List

logger = logging.getLogger(__name__)

# compute memory needed to run the model: file_size (MB) * 1.2

# 7B model still requires 280 GB just to fit the model on the hardware
# Nvidia T4: 16 GB RAM

def free_memory() -> None:
    """Frees memory by running the garbage collector."""
    gc.collect()

def get_file_size_mb(filename: str) -> float:
    return os.stat(filename).st_size / 1024 / 1024

def get_data_sz_mb(data: List[str]):
    return sum(len(s.encode("utf-8")) for s in data) / 1024 / 1024

def profile_device() -> dict:
    return dict(num_cores=multiprocessing.cpu_count())

def profile_memory(gb_unit: bool = True) -> dict:
    # convert to either GB or MB
    scale = (1024.0**3) if gb_unit else (1024.0**2)
    memory = psutil.virtual_memory()
    return dict(
        cpu_processor=platform.processor(),
        memory_units="GB" if gb_unit else "MB",
        total_memory=round(memory.total / scale, 3),
        available_memory=round(memory.available / scale, 3),
        used_memory=round(memory.used / scale, 3),
    )


class Timer:
    def __init__(self) -> None:
        """Initializes the timer."""
        self.start = datetime.datetime.utcnow()
        self.stop = self.start

    def __enter__(self) -> "Timer":
        """Starts the timer."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Stops the timer."""
        self.stop = datetime.datetime.utcnow()

    @property
    def elapsed(self) -> float:
        """Calculates the elapsed time in seconds."""
        return (self.stop - self.start).total_seconds()
