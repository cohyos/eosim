"""Tests for core.spectral module."""

import pytest
import numpy as np

from eosim.core.spectral import (
    SpectralRegion,
    SpectralBand,
    WavelengthGrid,
    SpectralQuantity,
    planck_radiance,
    wien_displacement,
    stefan_boltzmann,
    band_integrated_radiance,
    BAND_VIS,
    BAND_MWIR,
    BAND_LWIR,
)


class TestSpectralRegion:
    """Tests for SpectralRegion enum."""

    def test_from_visible_wavelength(self) -> None:
        """Test visible wavelength classification."""
        assert SpectralRegion.from_wavelength(0.55) == SpectralRegion.VIS

    def test_from_mwir_wavelength(self) -> None:
        """Test MWIR wavelength classification."""
        assert SpectralRegion.from_wavelength(4.0) == SpectralRegion.MWIR

    def test_from_lwir_wavelength(self) -> None:
        """Test LWIR wavelength classification."""
        assert SpectralRegion.from_wavelength(10.0) == SpectralRegion.LWIR

    def test_invalid_wavelength_below_range(self) -> None:
        """Test error for wavelength below VIS range."""
        with pytest.raises(ValueError, match="below VIS"):
            SpectralRegion.from_wavelength(0.1)

    def test_invalid_wavelength_above_range(self) -> None:
        """Test error for wavelength above VLWIR range."""
        with pytest.raises(ValueError, match="above VLWIR"):
            SpectralRegion.from_wavelength(100.0)


class TestSpectralBand:
    """Tests for SpectralBand."""

    def test_create_band(self) -> None:
        """Test creating a spectral band."""
        band = SpectralBand(name="test", lambda_min_um=8.0, lambda_max_um=12.0)
        assert band.name == "test"
        assert band.lambda_min_um == 8.0
        assert band.lambda_max_um == 12.0

    def test_center_wavelength_computed(self) -> None:
        """Test center wavelength is computed as geometric mean."""
        band = SpectralBand(name="test", lambda_min_um=8.0, lambda_max_um=12.0)
        expected = np.sqrt(8.0 * 12.0)
        assert band.lambda_center_um == pytest.approx(expected)

    def test_bandwidth(self) -> None:
        """Test bandwidth calculation."""
        band = SpectralBand(name="test", lambda_min_um=8.0, lambda_max_um=12.0)
        assert band.bandwidth_um == 4.0

    def test_wavelength_conversion_to_meters(self) -> None:
        """Test wavelength conversion to meters."""
        band = SpectralBand(name="test", lambda_min_um=10.0, lambda_max_um=12.0)
        assert band.lambda_min_m == 10e-6
        assert band.lambda_max_m == 12e-6

    def test_contains_wavelength(self) -> None:
        """Test wavelength containment check."""
        band = SpectralBand(name="test", lambda_min_um=8.0, lambda_max_um=12.0)
        assert band.contains(10.0) is True
        assert band.contains(7.0) is False
        assert band.contains(13.0) is False

    def test_invalid_band_limits(self) -> None:
        """Test error for invalid band limits."""
        with pytest.raises(ValueError, match="must be less than"):
            SpectralBand(name="test", lambda_min_um=12.0, lambda_max_um=8.0)

    def test_from_region(self) -> None:
        """Test creating band from spectral region."""
        band = SpectralBand.from_region(SpectralRegion.LWIR)
        assert band.lambda_min_um == 8.0
        assert band.lambda_max_um == 14.0

    def test_predefined_bands(self) -> None:
        """Test predefined band constants."""
        assert BAND_VIS.lambda_min_um == 0.38
        assert BAND_MWIR.lambda_min_um == 3.0
        assert BAND_LWIR.lambda_max_um == 14.0


class TestWavelengthGrid:
    """Tests for WavelengthGrid."""

    def test_uniform_grid(self) -> None:
        """Test creating uniform wavelength grid."""
        grid = WavelengthGrid.uniform(8.0, 12.0, 5)
        assert len(grid.wavelengths_um) == 5
        assert grid.lambda_min_um == 8.0
        assert grid.lambda_max_um == 12.0
        assert grid.is_uniform is True

    def test_from_band(self) -> None:
        """Test creating grid from spectral band."""
        grid = WavelengthGrid.from_band(BAND_LWIR, n_points=50)
        assert grid.lambda_min_um == BAND_LWIR.lambda_min_um
        assert grid.lambda_max_um == BAND_LWIR.lambda_max_um
        assert grid.n_wavelengths == 50

    def test_logarithmic_grid(self) -> None:
        """Test creating logarithmic wavelength grid."""
        grid = WavelengthGrid.logarithmic(1.0, 10.0, 5)
        assert len(grid.wavelengths_um) == 5
        # Check logarithmic spacing
        log_wavelengths = np.log10(grid.wavelengths_um)
        diffs = np.diff(log_wavelengths)
        assert np.allclose(diffs, diffs[0])

    def test_wavelength_to_meters(self) -> None:
        """Test wavelength conversion to meters."""
        grid = WavelengthGrid.uniform(8.0, 12.0, 5)
        assert grid.wavelengths_m[0] == pytest.approx(8e-6)

    def test_non_increasing_wavelengths_error(self) -> None:
        """Test error for non-increasing wavelengths."""
        with pytest.raises(ValueError, match="strictly increasing"):
            WavelengthGrid(wavelengths_um=np.array([10.0, 9.0, 8.0]))


class TestSpectralQuantity:
    """Tests for SpectralQuantity."""

    def test_create_spectral_quantity(self) -> None:
        """Test creating spectral quantity."""
        wavelengths = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
        values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        sq = SpectralQuantity(wavelengths_um=wavelengths, values=values, units="W/m^2")
        assert len(sq.wavelengths_um) == 5
        assert sq.units == "W/m^2"

    def test_interpolate(self) -> None:
        """Test interpolation to new wavelength grid."""
        wavelengths = np.array([8.0, 10.0, 12.0])
        values = np.array([1.0, 3.0, 1.0])
        sq = SpectralQuantity(wavelengths_um=wavelengths, values=values)

        new_wavelengths = np.array([9.0, 11.0])
        interpolated = sq.interpolate(new_wavelengths)
        assert interpolated[0] == pytest.approx(2.0)  # Linear interp
        assert interpolated[1] == pytest.approx(2.0)

    def test_integrate(self) -> None:
        """Test spectral integration."""
        wavelengths = np.array([8.0, 10.0, 12.0])
        values = np.array([1.0, 1.0, 1.0])  # Constant
        sq = SpectralQuantity(wavelengths_um=wavelengths, values=values)

        integral = sq.integrate()
        assert integral == pytest.approx(4.0)  # 1 × (12-8) = 4

    def test_band_average(self) -> None:
        """Test band average calculation."""
        wavelengths = np.linspace(8.0, 12.0, 100)
        values = np.ones_like(wavelengths) * 5.0  # Constant value
        sq = SpectralQuantity(wavelengths_um=wavelengths, values=values)

        band = SpectralBand(name="test", lambda_min_um=8.0, lambda_max_um=12.0)
        avg = sq.band_average(band)
        assert avg == pytest.approx(5.0)


class TestPlanckRadiance:
    """Tests for Planck radiance calculations."""

    def test_planck_at_room_temperature(self) -> None:
        """Test Planck radiance at room temperature peaks in LWIR."""
        # Room temperature ~300K should peak around 9.7 μm
        wavelengths = np.linspace(1.0, 20.0, 100)
        radiance = planck_radiance(wavelengths, 300.0)

        peak_idx = np.argmax(radiance)
        peak_wavelength = wavelengths[peak_idx]
        # Wien's law: λ_peak = 2898/T = 2898/300 ≈ 9.66 μm
        assert 9.0 < peak_wavelength < 10.5

    def test_planck_higher_temperature(self) -> None:
        """Test higher temperature gives higher radiance."""
        radiance_300K = planck_radiance(10.0, 300.0)
        radiance_400K = planck_radiance(10.0, 400.0)
        assert radiance_400K > radiance_300K

    def test_planck_scalar_and_array(self) -> None:
        """Test Planck works with scalar and array inputs."""
        scalar_result = planck_radiance(10.0, 300.0)
        array_result = planck_radiance(np.array([10.0]), 300.0)
        assert scalar_result == pytest.approx(array_result[0])


class TestWienDisplacement:
    """Tests for Wien displacement law."""

    def test_wien_room_temperature(self) -> None:
        """Test Wien's law at room temperature."""
        peak = wien_displacement(300.0)
        assert peak == pytest.approx(9.66, rel=0.01)

    def test_wien_sun_temperature(self) -> None:
        """Test Wien's law at solar temperature."""
        peak = wien_displacement(5778.0)
        # Solar peak should be in visible range
        assert 0.4 < peak < 0.7


class TestStefanBoltzmann:
    """Tests for Stefan-Boltzmann law."""

    def test_stefan_boltzmann_room_temperature(self) -> None:
        """Test total radiant exitance at room temperature."""
        M = stefan_boltzmann(300.0)
        # Known value: σT⁴ = 5.67e-8 × 300⁴ ≈ 459 W/m²
        assert M == pytest.approx(459, rel=0.01)

    def test_stefan_boltzmann_scaling(self) -> None:
        """Test T⁴ scaling."""
        M1 = stefan_boltzmann(300.0)
        M2 = stefan_boltzmann(600.0)
        # Doubling temperature should increase by 2⁴ = 16
        assert M2 / M1 == pytest.approx(16.0)


class TestBandIntegratedRadiance:
    """Tests for band-integrated radiance."""

    def test_band_integrated_positive(self) -> None:
        """Test band-integrated radiance is positive."""
        radiance = band_integrated_radiance(BAND_LWIR, 300.0)
        assert radiance > 0

    def test_hotter_gives_more_radiance(self) -> None:
        """Test hotter temperature gives more band-integrated radiance."""
        rad_300 = band_integrated_radiance(BAND_LWIR, 300.0)
        rad_400 = band_integrated_radiance(BAND_LWIR, 400.0)
        assert rad_400 > rad_300
