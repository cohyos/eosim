"""
Quantum Efficiency (QE) models for various detector types.

Provides spectral quantum efficiency curves for converting incident
photons to signal electrons.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

from eosim.sensor.fpa import DetectorType


@dataclass
class QEModel:
    """Quantum efficiency model from wavelength-QE data.

    Attributes:
        wavelengths_um: Wavelength sample points in micrometers
        qe_values: QE values at each wavelength (0-1)
        detector_type: Optional detector type identifier
    """

    wavelengths_um: NDArray[np.floating]
    qe_values: NDArray[np.floating]
    detector_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and create interpolator."""
        self.wavelengths_um = np.asarray(self.wavelengths_um, dtype=np.float64)
        self.qe_values = np.asarray(self.qe_values, dtype=np.float64)

        if len(self.wavelengths_um) != len(self.qe_values):
            raise ValueError("Wavelength and QE arrays must have same length")

        if np.any(self.qe_values < 0) or np.any(self.qe_values > 1):
            raise ValueError("QE values must be between 0 and 1")

        # Sort by wavelength
        sort_idx = np.argsort(self.wavelengths_um)
        self.wavelengths_um = self.wavelengths_um[sort_idx]
        self.qe_values = self.qe_values[sort_idx]

        # Create interpolator
        self._interp = interp1d(
            self.wavelengths_um,
            self.qe_values,
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, wavelength_um: Union[float, NDArray]) -> Union[float, NDArray]:
        """Get QE at specified wavelength(s).

        Args:
            wavelength_um: Wavelength(s) in micrometers

        Returns:
            QE value(s) between 0 and 1
        """
        result = self._interp(wavelength_um)
        if np.isscalar(wavelength_um):
            return float(result)
        return result

    @property
    def peak_qe(self) -> float:
        """Maximum QE value."""
        return float(np.max(self.qe_values))

    @property
    def peak_wavelength_um(self) -> float:
        """Wavelength of peak QE."""
        idx = np.argmax(self.qe_values)
        return float(self.wavelengths_um[idx])

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """Wavelength range where QE > 0."""
        nonzero = self.qe_values > 0.01
        if not np.any(nonzero):
            return (0.0, 0.0)
        valid_wl = self.wavelengths_um[nonzero]
        return (float(valid_wl[0]), float(valid_wl[-1]))

    def integrate(
        self,
        wavelength_min_um: float,
        wavelength_max_um: float,
        n_samples: int = 100,
    ) -> float:
        """Integrate QE over wavelength range.

        Args:
            wavelength_min_um: Start wavelength
            wavelength_max_um: End wavelength
            n_samples: Number of integration points

        Returns:
            Integrated QE (dimensionless, for averaging)
        """
        wavelengths = np.linspace(wavelength_min_um, wavelength_max_um, n_samples)
        qe_values = self(wavelengths)
        return float(np.trapezoid(qe_values, wavelengths) / (wavelength_max_um - wavelength_min_um))

    def effective_qe(
        self,
        spectral_radiance: NDArray[np.floating],
        wavelengths_um: NDArray[np.floating],
    ) -> float:
        """Compute spectrally-weighted effective QE.

        Args:
            spectral_radiance: Spectral radiance at each wavelength
            wavelengths_um: Wavelength array

        Returns:
            Effective QE weighted by spectral radiance
        """
        qe = self(wavelengths_um)
        weighted = spectral_radiance * qe
        return float(np.trapezoid(weighted, wavelengths_um) / np.trapezoid(spectral_radiance, wavelengths_um))


# Pre-defined QE curves for common detector types
_QE_DATA = {
    DetectorType.SI_CCD: {
        'wavelengths': [0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10],
        'qe': [0.10, 0.30, 0.50, 0.70, 0.80, 0.75, 0.60, 0.40, 0.20, 0.05, 0.0],
    },
    DetectorType.SI_CMOS: {
        'wavelengths': [0.30, 0.35, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10],
        'qe': [0.15, 0.35, 0.55, 0.70, 0.75, 0.70, 0.55, 0.35, 0.15, 0.03, 0.0],
    },
    DetectorType.INGAAS: {
        'wavelengths': [0.80, 0.90, 1.00, 1.10, 1.30, 1.50, 1.70, 1.80, 1.90],
        'qe': [0.10, 0.40, 0.60, 0.70, 0.75, 0.70, 0.50, 0.20, 0.0],
    },
    DetectorType.INSB: {
        'wavelengths': [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.3, 5.5, 6.0],
        'qe': [0.30, 0.60, 0.70, 0.75, 0.78, 0.75, 0.70, 0.60, 0.40, 0.20, 0.0],
    },
    DetectorType.HGCDTE_MWIR: {
        'wavelengths': [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.2, 5.5],
        'qe': [0.40, 0.60, 0.70, 0.75, 0.78, 0.76, 0.72, 0.65, 0.50, 0.30],
    },
    DetectorType.HGCDTE_LWIR: {
        'wavelengths': [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        'qe': [0.30, 0.50, 0.60, 0.65, 0.60, 0.55, 0.45, 0.25, 0.10],
    },
    DetectorType.QWIP: {
        'wavelengths': [7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0],
        'qe': [0.05, 0.15, 0.25, 0.30, 0.28, 0.20, 0.10, 0.02],
    },
    DetectorType.MICROBOLOMETER: {
        # For microbolometers, this represents absorption (not true QE)
        'wavelengths': [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        'qe': [0.70, 0.80, 0.85, 0.85, 0.82, 0.78, 0.72, 0.65, 0.55],
    },
}


def create_qe_model(
    detector_type: Union[DetectorType, str],
) -> QEModel:
    """Create QE model for a standard detector type.

    Args:
        detector_type: Detector type enum or string

    Returns:
        QEModel with typical QE curve
    """
    if isinstance(detector_type, str):
        detector_type = DetectorType(detector_type)

    if detector_type not in _QE_DATA:
        raise ValueError(f"Unknown detector type: {detector_type}")

    data = _QE_DATA[detector_type]
    return QEModel(
        wavelengths_um=np.array(data['wavelengths']),
        qe_values=np.array(data['qe']),
        detector_type=detector_type.value,
    )


def create_flat_qe(
    qe_value: float,
    wavelength_min_um: float,
    wavelength_max_um: float,
) -> QEModel:
    """Create a flat (constant) QE model.

    Args:
        qe_value: Constant QE value (0-1)
        wavelength_min_um: Start of sensitivity range
        wavelength_max_um: End of sensitivity range

    Returns:
        QEModel with constant QE in range
    """
    wavelengths = np.array([
        wavelength_min_um - 0.1,
        wavelength_min_um,
        wavelength_max_um,
        wavelength_max_um + 0.1,
    ])
    qe = np.array([0.0, qe_value, qe_value, 0.0])
    return QEModel(wavelengths, qe, "flat")


def create_gaussian_qe(
    peak_wavelength_um: float,
    peak_qe: float,
    fwhm_um: float,
) -> QEModel:
    """Create a Gaussian-shaped QE curve.

    Args:
        peak_wavelength_um: Wavelength of peak QE
        peak_qe: Maximum QE value
        fwhm_um: Full width at half maximum

    Returns:
        QEModel with Gaussian shape
    """
    sigma = fwhm_um / (2 * np.sqrt(2 * np.log(2)))
    wavelengths = np.linspace(
        peak_wavelength_um - 3 * fwhm_um,
        peak_wavelength_um + 3 * fwhm_um,
        51,
    )
    qe = peak_qe * np.exp(-0.5 * ((wavelengths - peak_wavelength_um) / sigma) ** 2)
    qe = np.clip(qe, 0, 1)

    return QEModel(wavelengths, qe, "gaussian")


class SpectrallyWeightedQE:
    """QE model that accounts for spectral weighting.

    Used when the QE varies significantly across the band and
    accurate integration is needed.
    """

    def __init__(
        self,
        qe_model: QEModel,
        n_spectral_samples: int = 50,
    ) -> None:
        """Initialize with QE model.

        Args:
            qe_model: Underlying QE model
            n_spectral_samples: Number of samples for spectral integration
        """
        self.qe_model = qe_model
        self.n_samples = n_spectral_samples

    def compute_signal(
        self,
        spectral_irradiance: callable,
        wavelength_min_um: float,
        wavelength_max_um: float,
        pixel_area_m2: float,
        integration_time_s: float,
    ) -> float:
        """Compute signal electrons from spectral irradiance.

        Args:
            spectral_irradiance: Function E(wavelength) in W/m²/μm
            wavelength_min_um: Band minimum wavelength
            wavelength_max_um: Band maximum wavelength
            pixel_area_m2: Pixel area in square meters
            integration_time_s: Integration time in seconds

        Returns:
            Signal in electrons
        """
        from eosim.core.constants import PLANCK_H, SPEED_OF_LIGHT

        wavelengths = np.linspace(wavelength_min_um, wavelength_max_um, self.n_samples)
        dw = wavelengths[1] - wavelengths[0]

        # Get QE at each wavelength
        qe = self.qe_model(wavelengths)

        # Get irradiance at each wavelength
        E = np.array([spectral_irradiance(w) for w in wavelengths])

        # Photon energy at each wavelength
        photon_energy = PLANCK_H * SPEED_OF_LIGHT / (wavelengths * 1e-6)

        # Integrate: electrons = ∫ E(λ) × QE(λ) × A × t / E_photon dλ
        integrand = E * qe * pixel_area_m2 * integration_time_s / photon_energy
        electrons = np.trapezoid(integrand, wavelengths)

        return float(electrons)
