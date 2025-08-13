from __future__ import annotations
import numpy as np

def success_rate(dones):
    dones = np.asarray(dones, dtype=bool)
    return dones.mean() if len(dones) else 0.0