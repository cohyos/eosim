"""Numpy compatibility utilities for EOSIM.

Handles version differences in numpy API (e.g., trapz vs trapezoid).
"""

import numpy as np


def integrate_trapz(y, x=None, axis=-1):
    """Trapezoidal integration compatible with numpy 1.x and 2.x.

    Args:
        y: Array to integrate
        x: Sample points (optional)
        axis: Axis along which to integrate

    Returns:
        Integrated value(s)
    """
    # Try new numpy 2.0 function first
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x, axis=axis)
    else:
        return np.trapz(y, x, axis=axis)
