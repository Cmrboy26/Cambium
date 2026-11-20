"""Task-neutral encoding of an actor's own item-use and mining progress."""

import numpy as np


def item_progress_observations(eating_remaining, eating_duration, mining_progress):
    remaining = np.asarray(eating_remaining)
    duration = np.asarray(eating_duration)
    mining = np.asarray(mining_progress)
    if (remaining.dtype.kind not in "iu" or duration.dtype.kind not in "iu"
            or mining.dtype.kind not in "iuf"):
        raise ValueError("Item progress needs integer tick counts and numeric mining progress")
    try:
        remaining, duration, mining = np.broadcast_arrays(remaining, duration, mining)
    except ValueError:
        raise ValueError("Item progress shapes must broadcast") from None
    if (np.any(remaining < 0) or np.any(duration < 0) or np.any(remaining > duration)
            or not np.isfinite(mining).all() or np.any(mining < 0) or np.any(mining > 1)):
        raise ValueError("Invalid item progress state")
    eating = np.zeros(remaining.shape, dtype=np.float64)
    active = remaining > 0
    np.divide(remaining, duration, out=eating, where=active)
    eating[active] = 1.0 - eating[active]
    return np.stack((eating, mining), axis=-1).astype(np.float32)