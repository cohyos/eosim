"""
Radiometric Calibration for EOSIM.

Provides tools for converting between DN, radiance, and temperature.

Example 1: Basic DN to temperature conversion
    >>> from eosim.calibration import RadiometricCalibrator
    >>> cal = RadiometricCalibrator()
    >>> cal.fit([1000, 5000, 10000], [280, 300, 320])
    >>> temp = cal.dn_to_temperature(raw_image)

Example 2: Calibration curve fitting
    >>> from eosim.calibration import CalibrationCurve
    >>> curve = CalibrationCurve.from_blackbody_data(
    ...     dn_values=[1000, 3000, 5000, 7000, 9000],
    ...     temperatures=[280, 290, 300, 310, 320],
    ... )
    >>> temp = curve.dn_to_temperature(dn_image)

Example 3: Radiance conversion with Planck function
    >>> from eosim.calibration import temperature_to_radiance
    >>> L = temperature_to_radiance(300, wavelength_um=10.0)
    >>> print(f"Radiance at 300K: {L:.2f} W/(m²·sr·μm)")
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import numpy as np
from numpy.typing import NDArray


# Physical constants
H = 6.62607015e-34  # Planck constant [J·s]
C = 299792458.0  # Speed of light [m/s]
K_B = 1.380649e-23  # Boltzmann constant [J/K]
SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]


def temperature_to_radiance(
    temperature_k: Union[float, NDArray],
    wavelength_um: float,
    emissivity: float = 1.0,
) -> Union[float, NDArray]:
    """Convert temperature to spectral radiance using Planck function.

    L = ε × (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)

    Args:
        temperature_k: Temperature in Kelvin
        wavelength_um: Wavelength in micrometers
        emissivity: Surface emissivity (0-1)

    Returns:
        Spectral radiance [W/(m²·sr·μm)]
    """
    T = np.asarray(temperature_k)
    wavelength_m = wavelength_um * 1e-6

    c1 = 2 * H * C**2
    c2 = H * C / K_B

    T = np.maximum(T, 1.0)  # Avoid division by zero
    exponent = c2 / (wavelength_m * T)
    exponent = np.minimum(exponent, 700)  # Prevent overflow

    radiance = c1 / (wavelength_m**5) / (np.exp(exponent) - 1)

    # Convert from per-meter to per-micrometer
    radiance = radiance * 1e-6 * emissivity

    return float(radiance) if radiance.ndim == 0 else radiance


def radiance_to_temperature(
    radiance: Union[float, NDArray],
    wavelength_um: float,
    emissivity: float = 1.0,
) -> Union[float, NDArray]:
    """Convert spectral radiance to apparent temperature (inverse Planck).

    Args:
        radiance: Spectral radiance [W/(m²·sr·μm)]
        wavelength_um: Wavelength in micrometers
        emissivity: Surface emissivity (0-1)

    Returns:
        Apparent temperature [K]
    """
    L = np.asarray(radiance)
    wavelength_m = wavelength_um * 1e-6

    # Correct for emissivity
    if emissivity != 1.0:
        L = L / emissivity

    # Convert from per-μm to per-m
    L_scaled = L * 1e6
    L_scaled = np.maximum(L_scaled, 1e-20)  # Avoid log(0)

    c1 = 2 * H * C**2
    c2 = H * C / K_B

    T = c2 / (wavelength_m * np.log(c1 / (wavelength_m**5 * L_scaled) + 1))

    return float(T) if T.ndim == 0 else T


def dn_to_radiance(
    dn: Union[int, NDArray],
    gain: float,
    offset: float,
) -> Union[float, NDArray]:
    """Convert digital number to radiance using linear model.

    L = gain × DN + offset

    Args:
        dn: Digital number value(s)
        gain: Responsivity [W/(m²·sr·μm)/DN]
        offset: Dark offset [W/(m²·sr·μm)]

    Returns:
        Radiance [W/(m²·sr·μm)]
    """
    dn = np.asarray(dn)
    return gain * dn + offset


@dataclass
class CalibrationCurve:
    """Calibration curve for DN to temperature conversion.

    Supports polynomial and lookup table interpolation.

    Attributes:
        coefficients: Polynomial coefficients (highest degree first)
        dn_range: Valid DN range (min, max)
        temp_range: Temperature range (min, max)
        method: 'polynomial' or 'interpolation'
    """

    coefficients: NDArray
    dn_range: Tuple[float, float]
    temp_range: Tuple[float, float]
    method: str = "polynomial"
    _dn_lut: Optional[NDArray] = None
    _temp_lut: Optional[NDArray] = None

    @classmethod
    def from_blackbody_data(
        cls,
        dn_values: List[float],
        temperatures: List[float],
        degree: int = 3,
        method: str = "polynomial",
    ) -> "CalibrationCurve":
        """Create calibration curve from blackbody measurements.

        Args:
            dn_values: List of DN values at each temperature
            temperatures: List of blackbody temperatures [K]
            degree: Polynomial degree for fitting
            method: 'polynomial' or 'interpolation'

        Returns:
            CalibrationCurve instance
        """
        dn_arr = np.array(dn_values)
        temp_arr = np.array(temperatures)

        if method == "polynomial":
            # Fit polynomial: T = f(DN)
            coefficients = np.polyfit(dn_arr, temp_arr, degree)
        else:
            coefficients = np.array([])  # Not used for interpolation

        curve = cls(
            coefficients=coefficients,
            dn_range=(float(dn_arr.min()), float(dn_arr.max())),
            temp_range=(float(temp_arr.min()), float(temp_arr.max())),
            method=method,
        )

        if method == "interpolation":
            # Sort by DN for interpolation
            sorted_indices = np.argsort(dn_arr)
            curve._dn_lut = dn_arr[sorted_indices]
            curve._temp_lut = temp_arr[sorted_indices]

        return curve

    def dn_to_temperature(
        self,
        dn: Union[int, float, NDArray],
        clip: bool = True,
    ) -> Union[float, NDArray]:
        """Convert DN to temperature.

        Args:
            dn: Digital number(s)
            clip: Clip to valid temperature range

        Returns:
            Temperature(s) in Kelvin
        """
        dn = np.asarray(dn, dtype=np.float64)

        if self.method == "polynomial":
            temp = np.polyval(self.coefficients, dn)
        else:
            temp = np.interp(dn, self._dn_lut, self._temp_lut)

        if clip:
            temp = np.clip(temp, self.temp_range[0], self.temp_range[1])

        return float(temp) if temp.ndim == 0 else temp

    def temperature_to_dn(
        self,
        temperature: Union[float, NDArray],
        clip: bool = True,
    ) -> Union[float, NDArray]:
        """Convert temperature to DN (inverse calibration).

        Args:
            temperature: Temperature(s) in Kelvin
            clip: Clip to valid DN range

        Returns:
            Digital number(s)
        """
        temp = np.asarray(temperature, dtype=np.float64)

        if self.method == "interpolation":
            # Inverse interpolation
            dn = np.interp(temp, self._temp_lut[::-1], self._dn_lut[::-1])
        else:
            # Numerical inversion of polynomial
            # Create lookup and interpolate
            dn_range = np.linspace(self.dn_range[0], self.dn_range[1], 1000)
            temp_range = np.polyval(self.coefficients, dn_range)
            dn = np.interp(temp, temp_range, dn_range)

        if clip:
            dn = np.clip(dn, self.dn_range[0], self.dn_range[1])

        return float(dn) if dn.ndim == 0 else dn

    def residual_error(
        self,
        dn_values: List[float],
        temperatures: List[float],
    ) -> float:
        """Compute RMS residual error of calibration.

        Args:
            dn_values: Test DN values
            temperatures: True temperatures

        Returns:
            RMS error in Kelvin
        """
        predicted = self.dn_to_temperature(np.array(dn_values), clip=False)
        actual = np.array(temperatures)
        return float(np.sqrt(np.mean((predicted - actual) ** 2)))


class RadiometricCalibrator:
    """Complete radiometric calibration for thermal sensors.

    Handles conversion between DN, radiance, and temperature
    with support for gain/offset calibration and Planck physics.
    """

    def __init__(
        self,
        wavelength_um: float = 10.0,
        emissivity: float = 1.0,
    ) -> None:
        """Initialize radiometric calibrator.

        Args:
            wavelength_um: Sensor wavelength [μm]
            emissivity: Target emissivity (0-1)
        """
        self._wavelength_um = wavelength_um
        self._emissivity = emissivity
        self._calibration_curve: Optional[CalibrationCurve] = None
        self._gain: Optional[float] = None
        self._offset: Optional[float] = None

    @property
    def wavelength_um(self) -> float:
        """Get sensor wavelength."""
        return self._wavelength_um

    @wavelength_um.setter
    def wavelength_um(self, value: float) -> None:
        """Set sensor wavelength."""
        self._wavelength_um = value

    @property
    def is_calibrated(self) -> bool:
        """Check if calibrator has been fitted."""
        return self._calibration_curve is not None

    def fit(
        self,
        dn_values: List[float],
        temperatures: List[float],
        degree: int = 3,
        method: str = "polynomial",
    ) -> float:
        """Fit calibration from blackbody measurements.

        Args:
            dn_values: DN values at each temperature
            temperatures: Blackbody temperatures [K]
            degree: Polynomial degree
            method: 'polynomial' or 'interpolation'

        Returns:
            Calibration RMS error [K]
        """
        self._calibration_curve = CalibrationCurve.from_blackbody_data(
            dn_values, temperatures, degree, method
        )

        # Compute gain/offset for linear approximation
        dn_arr = np.array(dn_values)
        rad_arr = np.array(
            [temperature_to_radiance(T, self._wavelength_um) for T in temperatures]
        )

        # Linear fit: L = gain * DN + offset
        A = np.vstack([dn_arr, np.ones(len(dn_arr))]).T
        result = np.linalg.lstsq(A, rad_arr, rcond=None)
        self._gain, self._offset = result[0]

        return self._calibration_curve.residual_error(dn_values, temperatures)

    def fit_two_point(
        self,
        dn_cold: float,
        dn_hot: float,
        temp_cold: float,
        temp_hot: float,
    ) -> None:
        """Fit simple two-point calibration.

        Args:
            dn_cold: DN at cold reference
            dn_hot: DN at hot reference
            temp_cold: Cold reference temperature [K]
            temp_hot: Hot reference temperature [K]
        """
        self.fit([dn_cold, dn_hot], [temp_cold, temp_hot], degree=1, method="polynomial")

    def dn_to_temperature(
        self,
        dn: Union[int, float, NDArray],
    ) -> Union[float, NDArray]:
        """Convert DN to temperature.

        Args:
            dn: Digital number(s)

        Returns:
            Temperature(s) [K]

        Raises:
            ValueError: If calibrator not fitted
        """
        if not self.is_calibrated:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self._calibration_curve.dn_to_temperature(dn)

    def temperature_to_dn(
        self,
        temperature: Union[float, NDArray],
    ) -> Union[float, NDArray]:
        """Convert temperature to DN.

        Args:
            temperature: Temperature(s) [K]

        Returns:
            Digital number(s)

        Raises:
            ValueError: If calibrator not fitted
        """
        if not self.is_calibrated:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        return self._calibration_curve.temperature_to_dn(temperature)

    def dn_to_radiance(
        self,
        dn: Union[int, float, NDArray],
        wavelength_um: Optional[float] = None,
    ) -> Union[float, NDArray]:
        """Convert DN to spectral radiance.

        Args:
            dn: Digital number(s)
            wavelength_um: Override wavelength [μm]

        Returns:
            Radiance [W/(m²·sr·μm)]
        """
        if self._gain is None or self._offset is None:
            # Use temperature as intermediate
            temp = self.dn_to_temperature(dn)
            wl = wavelength_um or self._wavelength_um
            return temperature_to_radiance(temp, wl, self._emissivity)

        return dn_to_radiance(dn, self._gain, self._offset)

    def radiance_to_temperature(
        self,
        radiance: Union[float, NDArray],
        wavelength_um: Optional[float] = None,
    ) -> Union[float, NDArray]:
        """Convert radiance to apparent temperature.

        Args:
            radiance: Spectral radiance [W/(m²·sr·μm)]
            wavelength_um: Override wavelength [μm]

        Returns:
            Apparent temperature [K]
        """
        wl = wavelength_um or self._wavelength_um
        return radiance_to_temperature(radiance, wl, self._emissivity)

    def temperature_to_radiance(
        self,
        temperature: Union[float, NDArray],
        wavelength_um: Optional[float] = None,
    ) -> Union[float, NDArray]:
        """Convert temperature to radiance.

        Args:
            temperature: Temperature(s) [K]
            wavelength_um: Override wavelength [μm]

        Returns:
            Radiance [W/(m²·sr·μm)]
        """
        wl = wavelength_um or self._wavelength_um
        return temperature_to_radiance(temperature, wl, self._emissivity)

    def get_calibration_info(self) -> dict:
        """Get calibration information.

        Returns:
            Dictionary with calibration parameters
        """
        if not self.is_calibrated:
            return {"calibrated": False}

        return {
            "calibrated": True,
            "wavelength_um": self._wavelength_um,
            "emissivity": self._emissivity,
            "dn_range": self._calibration_curve.dn_range,
            "temp_range": self._calibration_curve.temp_range,
            "method": self._calibration_curve.method,
            "gain": self._gain,
            "offset": self._offset,
        }
