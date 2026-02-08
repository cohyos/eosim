"""
EOSIM Automatic Gain Control (AGC) Module.

Provides AGC algorithms for optimizing sensor image display contrast
and dynamic range compression, plus polarity/palette mapping for
thermal imagery.

Example 1: Apply histogram equalization AGC
    >>> from eosim.agc import AGCProcessor, AGCParameters, AGCMode
    >>> import numpy as np
    >>> raw = np.random.uniform(8000, 12000, (480, 640))  # 14-bit sensor data
    >>> params = AGCParameters(mode=AGCMode.HISTOGRAM_EQ, output_bits=8)
    >>> agc = AGCProcessor(params)
    >>> display = agc.process(raw)

Example 2: Apply CLAHE for local contrast enhancement
    >>> params = AGCParameters(mode=AGCMode.CLAHE, clahe_clip_limit=3.0)
    >>> agc = AGCProcessor(params)
    >>> enhanced = agc.process(raw)

Example 3: Apply ironbow palette to AGC output
    >>> from eosim.agc import PolarityMapper, Polarity
    >>> mapper = PolarityMapper(Polarity.IRONBOW)
    >>> rgb = mapper.apply(display)

Example 4: Quick one-shot AGC
    >>> from eosim.agc import apply_agc, AGCMode
    >>> display = apply_agc(raw, mode=AGCMode.LINEAR, linear_percent=2.0)
"""

from eosim.agc.agc import (
    AGCMode,
    Polarity,
    AGCParameters,
    AGCState,
    AGCProcessor,
    PolarityMapper,
    apply_agc,
    apply_polarity,
    create_agc_processor,
)

__all__ = [
    "AGCMode",
    "Polarity",
    "AGCParameters",
    "AGCState",
    "AGCProcessor",
    "PolarityMapper",
    "apply_agc",
    "apply_polarity",
    "create_agc_processor",
]
