"""Tests for EOSIM Atmosphere Enhancement (Stage B)."""

import pytest
import numpy as np

from eosim.atmosphere import (
    # Base classes
    PathGeometry,
    AtmosphereConditions,
    # Enhanced models
    LUTAtmosphere,
    AerosolModel,
    TurbulenceModel,
    AdjacencyModel,
    ContrastTransmission,
)


class TestLUTAtmosphere:
    """Tests for LUT-based atmosphere model."""

    def test_basic_transmission(self):
        """Test basic transmission computation."""
        model = LUTAtmosphere()
        path = PathGeometry(ground_range_m=5000, altitude_end_m=1000)
        conditions = AtmosphereConditions(visibility_km=23.0)

        result = model.compute(10.0, path, conditions)

        assert 0 < result.transmission <= 1.0
        assert result.path_radiance >= 0

    def test_visibility_effect(self):
        """Test that visibility affects transmission."""
        model = LUTAtmosphere()
        path = PathGeometry(ground_range_m=5000, altitude_end_m=1000)

        clear = AtmosphereConditions(visibility_km=50.0)
        hazy = AtmosphereConditions(visibility_km=5.0)

        result_clear = model.compute(10.0, path, clear)
        result_hazy = model.compute(10.0, path, hazy)

        # Clear should have higher transmission
        assert result_clear.transmission > result_hazy.transmission

    def test_range_effect(self):
        """Test that longer range reduces transmission."""
        model = LUTAtmosphere()
        conditions = AtmosphereConditions()

        short_path = PathGeometry(ground_range_m=1000, altitude_end_m=500)
        long_path = PathGeometry(ground_range_m=10000, altitude_end_m=500)

        result_short = model.compute(10.0, short_path, conditions)
        result_long = model.compute(10.0, long_path, conditions)

        assert result_short.transmission > result_long.transmission

    def test_spectral_transmission(self):
        """Test spectral transmission computation."""
        model = LUTAtmosphere()
        path = PathGeometry(ground_range_m=5000, altitude_end_m=1000)
        conditions = AtmosphereConditions()

        wavelengths = np.linspace(8, 14, 10)
        result = model.compute(wavelengths, path, conditions)

        assert len(result.transmission) == 10
        assert all(0 <= t <= 1 for t in result.transmission)

    def test_aerosol_types(self):
        """Test different aerosol models."""
        path = PathGeometry(ground_range_m=5000, altitude_end_m=500)
        conditions = AtmosphereConditions(visibility_km=15.0)

        rural = LUTAtmosphere(aerosol_type="rural")
        urban = LUTAtmosphere(aerosol_type="urban")
        maritime = LUTAtmosphere(aerosol_type="maritime")

        r1 = rural.compute(0.55, path, conditions)
        r2 = urban.compute(0.55, path, conditions)
        r3 = maritime.compute(0.55, path, conditions)

        # All should produce valid transmission
        assert 0 < r1.transmission < 1
        assert 0 < r2.transmission < 1
        assert 0 < r3.transmission < 1


class TestAerosolModel:
    """Tests for aerosol optical properties model."""

    def test_extinction(self):
        """Test aerosol extinction computation."""
        model = AerosolModel(aerosol_type="rural")
        ext = model.extinction(0.55, aod_550=0.1)

        assert ext > 0
        # AOD at reference wavelength should be close to input
        assert abs(ext - 0.1) < 0.01

    def test_wavelength_dependence(self):
        """Test Angstrom wavelength dependence."""
        model = AerosolModel(aerosol_type="rural")

        ext_blue = model.extinction(0.4, aod_550=0.1)
        ext_red = model.extinction(0.7, aod_550=0.1)

        # Blue should have higher extinction (rural has positive Angstrom)
        assert ext_blue > ext_red

    def test_phase_function(self):
        """Test Henyey-Greenstein phase function."""
        model = AerosolModel()

        # Forward scattering (cos_theta = 1)
        forward = model.phase_function(1.0)
        # Backward scattering (cos_theta = -1)
        backward = model.phase_function(-1.0)

        # Forward should be stronger than backward
        assert forward > backward

    def test_different_aerosol_types(self):
        """Test different aerosol type properties."""
        rural = AerosolModel("rural")
        maritime = AerosolModel("maritime")

        # Maritime has lower Angstrom exponent (larger particles)
        assert rural.angstrom_exponent > maritime.angstrom_exponent


class TestTurbulenceModel:
    """Tests for atmospheric turbulence model."""

    def test_cn2_profile(self):
        """Test Cn² profile computation."""
        model = TurbulenceModel(cn2_ground=1e-14)

        cn2_ground = model.cn2_profile(0)
        cn2_high = model.cn2_profile(5000)

        # Ground should be higher than altitude
        assert cn2_ground > cn2_high
        assert cn2_ground > 0
        assert cn2_high > 0

    def test_fried_parameter(self):
        """Test Fried parameter computation."""
        model = TurbulenceModel()

        r0 = model.fried_parameter(
            wavelength_um=0.55,
            path_length_m=5000,
        )

        # Typical r0 is centimeters to tens of centimeters
        assert 0.001 < r0 < 1.0  # meters

    def test_wavelength_effect_on_r0(self):
        """Test wavelength effect on Fried parameter."""
        model = TurbulenceModel()

        r0_vis = model.fried_parameter(0.55, 5000)
        r0_ir = model.fried_parameter(10.0, 5000)

        # Longer wavelength = larger r0
        assert r0_ir > r0_vis

    def test_seeing_blur(self):
        """Test seeing blur computation."""
        model = TurbulenceModel()

        fwhm = model.seeing_blur_fwhm(
            wavelength_um=0.55,
            path_length_m=5000,
        )

        # Typical seeing is a few arcseconds
        assert 0.1 < fwhm < 100  # arcseconds

    def test_scintillation_index(self):
        """Test scintillation index computation."""
        model = TurbulenceModel()

        sigma_i2 = model.scintillation_index(
            wavelength_um=0.55,
            path_length_m=5000,
            aperture_m=0.1,
        )

        # Scintillation index should be positive
        assert sigma_i2 > 0


class TestAdjacencyModel:
    """Tests for adjacency effects model."""

    def test_basic_adjacency(self):
        """Test basic adjacency computation."""
        model = AdjacencyModel()
        path = PathGeometry(ground_range_m=3000, altitude_end_m=500)
        conditions = AtmosphereConditions(visibility_km=10.0)

        result = model.compute_adjacency(
            wavelength_um=10.0,
            path=path,
            conditions=conditions,
            target_radiance=30.0,
            background_radiance=25.0,
        )

        assert result.adjacency_radiance >= 0
        assert result.spherical_albedo > 0
        assert result.effective_radius_m > 0

    def test_adjacency_increases_with_scattering(self):
        """Test that adjacency increases with more scattering (lower visibility)."""
        model = AdjacencyModel()
        path = PathGeometry(ground_range_m=3000, altitude_end_m=500)

        clear = AtmosphereConditions(visibility_km=50.0)
        hazy = AtmosphereConditions(visibility_km=5.0)

        result_clear = model.compute_adjacency(10.0, path, clear, 30.0, 25.0)
        result_hazy = model.compute_adjacency(10.0, path, hazy, 30.0, 25.0)

        # Hazy conditions should have more adjacency
        assert result_hazy.adjacency_radiance > result_clear.adjacency_radiance

    def test_adjacency_psf(self):
        """Test adjacency PSF computation."""
        model = AdjacencyModel()

        distances, psf = model.adjacency_psf(radius_m=500, n_points=50)

        assert len(distances) == 50
        assert len(psf) == 50
        assert psf[0] > psf[-1]  # Should decrease with distance
        assert np.abs(np.sum(psf) - 1.0) < 0.01  # Normalized

    def test_image_adjacency(self):
        """Test applying adjacency to an image."""
        model = AdjacencyModel()
        path = PathGeometry(ground_range_m=3000, altitude_end_m=500)
        conditions = AtmosphereConditions(visibility_km=10.0)

        # Create test image
        image = np.random.randn(50, 50) * 5 + 30

        result = model.apply_adjacency_to_image(
            image, 10.0, path, conditions, gsd_m=1.0
        )

        assert result.shape == image.shape
        # Result should be blurred (lower standard deviation)
        assert np.std(result) <= np.std(image)


class TestContrastTransmission:
    """Tests for contrast transmission model."""

    def test_basic_contrast(self):
        """Test basic contrast transmission."""
        model = ContrastTransmission()

        ct = model.contrast_transmission(
            range_km=5.0,
            visibility_km=23.0,
        )

        assert 0 < ct <= 1

    def test_range_effect(self):
        """Test that contrast decreases with range."""
        model = ContrastTransmission()

        ct_near = model.contrast_transmission(1.0, 23.0)
        ct_far = model.contrast_transmission(10.0, 23.0)

        assert ct_near > ct_far

    def test_visibility_effect(self):
        """Test visibility effect on contrast."""
        model = ContrastTransmission()

        ct_clear = model.contrast_transmission(5.0, 50.0)
        ct_hazy = model.contrast_transmission(5.0, 5.0)

        assert ct_clear > ct_hazy

    def test_max_detection_range(self):
        """Test maximum detection range computation."""
        model = ContrastTransmission()

        r_max = model.maximum_detection_range(
            inherent_contrast=0.5,
            threshold_contrast=0.02,
            visibility_km=23.0,
        )

        assert r_max > 0
        # Should be related to visibility
        assert r_max < 100  # km

    def test_apparent_contrast(self):
        """Test apparent contrast computation."""
        model = ContrastTransmission()

        inherent = 0.8
        apparent = model.apparent_contrast(inherent, 5.0, 23.0)

        assert 0 < apparent < inherent


class TestIntegration:
    """Integration tests combining atmosphere components."""

    def test_complete_atmosphere_chain(self):
        """Test complete atmospheric effect chain."""
        # Create models
        atm = LUTAtmosphere(aerosol_type="rural")
        adj = AdjacencyModel()
        turb = TurbulenceModel()

        # Define path
        path = PathGeometry(ground_range_m=5000, altitude_end_m=1000)
        conditions = AtmosphereConditions(visibility_km=15.0)

        # Compute transmission
        result = atm.compute(10.0, path, conditions)
        assert result.transmission > 0

        # Compute adjacency
        adj_result = adj.compute_adjacency(
            10.0, path, conditions, 30.0, 25.0
        )
        assert adj_result.adjacency_radiance >= 0

        # Compute turbulence blur
        seeing = turb.seeing_blur_fwhm(10.0, path.slant_range_m)
        assert seeing > 0

    def test_spectral_atmosphere_analysis(self):
        """Test spectral analysis across LWIR band."""
        model = LUTAtmosphere()
        path = PathGeometry(ground_range_m=3000, altitude_end_m=500)
        conditions = AtmosphereConditions()

        wavelengths = np.linspace(8, 14, 25)
        result = model.compute(wavelengths, path, conditions)

        # Check for ozone absorption dip around 9.6 um
        # Find minimum transmission
        min_idx = np.argmin(result.transmission)
        min_wavelength = wavelengths[min_idx]

        # Should be in the ozone band region
        assert 9.0 < min_wavelength < 10.5
