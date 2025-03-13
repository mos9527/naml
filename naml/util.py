from naml.modules import tqdm, Tuple
from naml.plot import simple_animated, Generator
from functools import wraps
from typing import Callable
from dataclasses import dataclass
import math

class ret_loss:
    def update(self, *args, **kwds): raise NotImplementedError
    def reset(self): raise NotImplementedError       
    @property 
    def value(self) -> Tuple[float]: raise NotImplementedError

@dataclass
class ret_accumlated_loss(ret_loss):
    loss: float = 0
    n: int = 0
    exp : bool = False

    def update(self, loss: float, n: int):
        self.loss += loss
        self.n += n
    
    def reset(self):
        self.loss = 0
        self.n = 0

    @property
    def value(self):
        if self.n:
            if self.exp:
                return (math.exp(self.loss / self.n),)
            else:
                return (self.loss / self.n, )            
        else:
            return (0, )

@dataclass
class ret_single_loss(ret_loss):
    val : float = 0

    def update(self, val : float):
        self.val = val

    def reset(self):
        pass

    @property
    def value(self):
        return (self.val, )



def run_epochs(title: str = ""):
    """Wraps a function that trains a model and returns the loss over epochs.
    
    Args:

        ret_fn: callable that returns a tuple of values to be plotted over epochs.
    """

    def _wrapper(fn) -> Callable:
        @wraps(fn)
        def _inner(n_epochs: int, *args, **kwargs):
            def _generator():
                for _ in tqdm(range(n_epochs)):
                    acc = fn(*args, **kwargs)
                    assert isinstance(acc, ret_loss), "Return value must be a `ret_loss` instance."                    
                    yield acc.value

            simple_animated(_generator(), title=title)

        return _inner

    return _wrapper
