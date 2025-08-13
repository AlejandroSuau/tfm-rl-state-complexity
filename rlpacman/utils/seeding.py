from __future__ import annotations
import numpy as np
import torch

def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import random
        random.seed(seed)
    except Exception:
        pass