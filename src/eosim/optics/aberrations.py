"""
Zernike polynomial aberrations for optical wavefront modeling.

Provides functions for computing Zernike polynomials and wavefront errors
used in PSF modeling with optical aberrations.
"""

from dataclasses import dataclass
from math import factorial
from typing import Optional
import numpy as np
from numpy.typing import NDArray


# Noll index to (n, m) conversion table for first 37 Zernike terms
# Based on Noll (1976) ordering
NOLL_TO_NM = {
    1: (0, 0),    # Piston
    2: (1, 1),    # Tilt X
    3: (1, -1),   # Tilt Y
    4: (2, 0),    # Defocus
    5: (2, -2),   # Astigmatism 45°
    6: (2, 2),    # Astigmatism 0°
    7: (3, -1),   # Coma Y
    8: (3, 1),    # Coma X
    9: (3, -3),   # Trefoil Y
    10: (3, 3),   # Trefoil X
    11: (4, 0),   # Primary Spherical
    12: (4, 2),   # Secondary Astigmatism 0°
    13: (4, -2),  # Secondary Astigmatism 45°
    14: (4, 4),   # Tetrafoil 0°
    15: (4, -4),  # Tetrafoil 22.5°
    16: (5, 1),   # Secondary Coma X
    17: (5, -1),  # Secondary Coma Y
    18: (5, 3),   # Secondary Trefoil X
    19: (5, -3),  # Secondary Trefoil Y
    20: (5, 5),   # Pentafoil X
    21: (5, -5),  # Pentafoil Y
    22: (6, 0),   # Secondary Spherical
}

# Human-readable names for common Zernike terms
ZERNIKE_NAMES = {
    1: "piston",
    2: "tilt_x",
    3: "tilt_y",
    4: "defocus",
    5: "astigmatism_45",
    6: "astigmatism_0",
    7: "coma_y",
    8: "coma_x",
    9: "trefoil_y",
    10: "trefoil_x",
    11: "spherical",
    12: "secondary_astigmatism_0",
    13: "secondary_astigmatism_45",
    22: "secondary_spherical",
}


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert Noll index j to (n, m) radial and azimuthal indices.

    The Noll ordering is a standard way to index Zernike polynomials
    sequentially starting from j=1.

    Args:
        j: Noll index (1-indexed)

    Returns:
        Tuple of (n, m) where n is radial order and m is azimuthal frequency
    """
    if j in NOLL_TO_NM:
        return NOLL_TO_NM[j]

    # Compute for arbitrary j using Noll's formula
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))
    m_sum = (n + 1) * (n + 2) // 2
    remainder = j - m_sum

    if n % 2 == 0:
        m = 2 * ((remainder + 1) // 2)
    else:
        m = 2 * (remainder // 2) + 1

    if j % 2 == 0:
        m = -m

    return (n, abs(m) if m >= 0 else -abs(m))


def nm_to_noll(n: int, m: int) -> int:
    """Convert (n, m) indices to Noll index.

    Args:
        n: Radial order (0, 1, 2, ...)
        m: Azimuthal frequency (-n to n, same parity as n)

    Returns:
        Noll index j (1-indexed)
    """
    # Base index for this radial order
    j_base = n * (n + 1) // 2 + 1

    # Offset within this radial order
    if m >= 0:
        offset = 2 * abs(m) - (1 if m > 0 else 0)
    else:
        offset = 2 * abs(m)

    return j_base + offset


def radial_polynomial(n: int, m: int, rho: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the radial component R_n^m(ρ) of a Zernike polynomial.

    R_n^m(ρ) = Σ_k (-1)^k (n-k)! / [k! ((n+m)/2-k)! ((n-m)/2-k)!] × ρ^(n-2k)

    Args:
        n: Radial order
        m: Azimuthal frequency (absolute value used)
        rho: Radial coordinate array (0 to 1)

    Returns:
        Radial polynomial values at each rho
    """
    m = abs(m)
    result = np.zeros_like(rho)

    for k in range((n - m) // 2 + 1):
        num = (-1) ** k * factorial(n - k)
        den = (
            factorial(k)
            * factorial((n + m) // 2 - k)
            * factorial((n - m) // 2 - k)
        )
        result = result + (num / den) * rho ** (n - 2 * k)

    return result


def zernike_polynomial(
    n: int,
    m: int,
    rho: NDArray[np.floating],
    theta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute Zernike polynomial Z_n^m on normalized pupil coordinates.

    For m >= 0: Z_n^m(ρ,θ) = R_n^m(ρ) × cos(mθ)
    For m < 0:  Z_n^m(ρ,θ) = R_n^|m|(ρ) × sin(|m|θ)

    Args:
        n: Radial order (0, 1, 2, ...)
        m: Azimuthal frequency (-n to n)
        rho: Radial coordinate (0 to 1)
        theta: Azimuthal angle in radians

    Returns:
        Zernike polynomial values
    """
    R = radial_polynomial(n, m, rho)

    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(abs(m) * theta)


def zernike_noll(
    j: int,
    rho: NDArray[np.floating],
    theta: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute Zernike polynomial by Noll index.

    Args:
        j: Noll index (1-indexed)
        rho: Radial coordinate (0 to 1)
        theta: Azimuthal angle in radians

    Returns:
        Zernike polynomial values
    """
    n, m = noll_to_nm(j)
    return zernike_polynomial(n, m, rho, theta)


def create_pupil_grid(
    size: int,
    normalized: bool = True,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Create a grid of pupil coordinates.

    Args:
        size: Grid size in pixels
        normalized: If True, coordinates are normalized to [-1, 1]

    Returns:
        Tuple of (rho, theta, mask) where mask is True inside unit circle
    """
    if normalized:
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
    else:
        x = np.linspace(-size // 2, size // 2, size) / (size // 2)
        y = np.linspace(-size // 2, size // 2, size) / (size // 2)

    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    # Mask for valid pupil region
    mask = rho <= 1.0

    return rho, theta, mask


def compute_wavefront(
    coefficients: dict[int, float],
    size: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute total wavefront error from Zernike coefficients.

    W(ρ,θ) = Σ_j c_j × Z_j(ρ,θ)

    Args:
        coefficients: Dict mapping Noll index to coefficient in waves
        size: Grid size in pixels

    Returns:
        Tuple of (wavefront, mask) where wavefront is in waves
    """
    rho, theta, mask = create_pupil_grid(size)
    wavefront = np.zeros((size, size), dtype=np.float64)

    for j, coeff in coefficients.items():
        if coeff != 0:
            Z = zernike_noll(j, rho, theta)
            wavefront = wavefront + coeff * Z

    # Set values outside pupil to zero
    wavefront = np.where(mask, wavefront, 0.0)

    return wavefront, mask


def rms_wavefront(
    wavefront: NDArray[np.floating],
    mask: NDArray[np.floating],
) -> float:
    """Compute RMS wavefront error within pupil.

    Args:
        wavefront: Wavefront array in waves
        mask: Boolean mask for valid pupil region

    Returns:
        RMS wavefront error in waves
    """
    valid = wavefront[mask]
    return float(np.sqrt(np.mean(valid**2)))


def peak_to_valley(
    wavefront: NDArray[np.floating],
    mask: NDArray[np.floating],
) -> float:
    """Compute peak-to-valley wavefront error.

    Args:
        wavefront: Wavefront array in waves
        mask: Boolean mask for valid pupil region

    Returns:
        Peak-to-valley error in waves
    """
    valid = wavefront[mask]
    return float(np.max(valid) - np.min(valid))


def strehl_ratio(rms_wavefront_waves: float) -> float:
    """Compute Strehl ratio from RMS wavefront error.

    Uses Maréchal approximation: S ≈ exp(-(2π×σ)²)
    Valid for σ < 0.1 waves (S > 0.67)

    Args:
        rms_wavefront_waves: RMS wavefront error in waves

    Returns:
        Strehl ratio (0 to 1)
    """
    return float(np.exp(-(2 * np.pi * rms_wavefront_waves) ** 2))


def strehl_from_coefficients(coefficients: dict[int, float]) -> float:
    """Compute Strehl ratio from Zernike coefficients.

    For orthonormal Zernike polynomials, RMS² = Σ c_j²
    (excluding piston, tip, and tilt which don't affect image quality)

    Args:
        coefficients: Dict mapping Noll index to coefficient in waves

    Returns:
        Strehl ratio (0 to 1)
    """
    # Exclude piston (j=1), tip (j=2), tilt (j=3)
    rms_squared = sum(c**2 for j, c in coefficients.items() if j > 3)
    rms = np.sqrt(rms_squared)
    return strehl_ratio(rms)


@dataclass
class AberrationSet:
    """Collection of optical aberration coefficients.

    All coefficients are in waves RMS at the reference wavelength.
    """

    defocus: float = 0.0
    astigmatism_0: float = 0.0
    astigmatism_45: float = 0.0
    coma_x: float = 0.0
    coma_y: float = 0.0
    spherical: float = 0.0
    trefoil_x: float = 0.0
    trefoil_y: float = 0.0

    def to_coefficients(self) -> dict[int, float]:
        """Convert to Noll index coefficient dictionary."""
        return {
            4: self.defocus,
            5: self.astigmatism_45,
            6: self.astigmatism_0,
            7: self.coma_y,
            8: self.coma_x,
            9: self.trefoil_y,
            10: self.trefoil_x,
            11: self.spherical,
        }

    @property
    def rms_total(self) -> float:
        """Total RMS wavefront error in waves."""
        coeffs = self.to_coefficients()
        return float(np.sqrt(sum(c**2 for c in coeffs.values())))

    @property
    def strehl(self) -> float:
        """Strehl ratio for this aberration set."""
        return strehl_ratio(self.rms_total)

    @classmethod
    def diffraction_limited(cls) -> "AberrationSet":
        """Create a diffraction-limited (zero aberration) set."""
        return cls()

    @classmethod
    def from_coefficients(cls, coefficients: dict[int, float]) -> "AberrationSet":
        """Create from Noll coefficient dictionary."""
        return cls(
            defocus=coefficients.get(4, 0.0),
            astigmatism_45=coefficients.get(5, 0.0),
            astigmatism_0=coefficients.get(6, 0.0),
            coma_y=coefficients.get(7, 0.0),
            coma_x=coefficients.get(8, 0.0),
            trefoil_y=coefficients.get(9, 0.0),
            trefoil_x=coefficients.get(10, 0.0),
            spherical=coefficients.get(11, 0.0),
        )


def defocus_from_distance(
    focal_length_mm: float,
    object_distance_m: float,
    focus_distance_m: float,
) -> float:
    """Compute defocus aberration from focus error.

    Args:
        focal_length_mm: Lens focal length in mm
        object_distance_m: Actual object distance in meters
        focus_distance_m: Distance lens is focused at in meters

    Returns:
        Defocus in waves (at 10 μm reference wavelength)
    """
    f = focal_length_mm / 1000  # Convert to meters

    # Thin lens formula: 1/f = 1/do + 1/di
    # Focus error causes defocus wavefront
    if object_distance_m == float("inf"):
        di_actual = f
    else:
        di_actual = 1 / (1 / f - 1 / object_distance_m)

    if focus_distance_m == float("inf"):
        di_focus = f
    else:
        di_focus = 1 / (1 / f - 1 / focus_distance_m)

    # Defocus in waves ≈ Δz / (8 × F#² × λ)
    # This is a simplified approximation
    delta_z = di_actual - di_focus
    reference_wavelength_m = 10e-6  # 10 μm
    f_number = 2.0  # Assume F/2 for estimation

    defocus_waves = delta_z / (8 * f_number**2 * reference_wavelength_m)
    return defocus_waves
