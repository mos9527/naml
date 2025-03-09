from naml.modules import tqdm
from naml.plot import simple_animated, Generator
from functools import wraps
from typing import Callable


def run_epochs(title: str = ""):
    """Wraps a function that trains a model and returns the loss over epochs."""

    def _wrapper(fn) -> Callable:
        @wraps(fn)
        def _inner(n_epochs: int, *args, **kwargs):
            def _generator():
                for epoch in tqdm(range(n_epochs)):
                    val = fn(*args, **kwargs)
                    yield (val,)

            simple_animated(_generator(), title=title)

        return _inner

    return _wrapper
