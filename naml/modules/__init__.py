from typing import List, Tuple, Dict, Any, Union, Generator, TypeVar
from tqdm.notebook import tqdm
import torch

T = TypeVar("T")

from matplotlib import pyplot as plt
import numpy as np

import torch.functional as F
import torch.nn as nn
import torch.optim as optim
