"""
GPU-Accelerated Operations for EOSIM.

Provides GPU-accelerated versions of common simulation operations
with transparent CPU fallback when GPU is not available.

Example 1: GPU-accelerated PSF convolution
    >>> from eosim.gpu import GPUContext, GPUConvolver
    >>> ctx = GPUContext()
    >>> convolver = GPUConvolver(ctx)
    >>> # Create a test image and PSF kernel
    >>> image = ctx.random((512, 512))
    >>> psf = ctx.normal(0, 1, (31, 31))
    >>> psf = psf / ctx.sum(psf)  # Normalize
    >>> result = convolver.convolve2d(image, psf)
    >>> print(f"Convolved on {ctx.device_name}")

Example 2: Batch Planck radiance computation
    >>> from eosim.gpu import GPUContext, GPURadiance
    >>> ctx = GPUContext()
    >>> radiance = GPURadiance(ctx)
    >>> temperatures = ctx.array([280, 290, 300, 310, 320])
    >>> wavelengths = ctx.linspace(8, 14, 100)
    >>> # Compute radiance for all T and λ combinations
    >>> L = radiance.planck_spectral(temperatures, wavelengths)
    >>> print(f"Radiance cube shape: {L.shape}")  # (5, 100)

Example 3: GPU noise generation
    >>> from eosim.gpu import GPUContext, GPUNoiseGenerator
    >>> ctx = GPUContext()
    >>> noise_gen = GPUNoiseGenerator(ctx, seed=42)
    >>> signal = ctx.ones((480, 640)) * 10000  # 10k electrons
    >>> noisy = noise_gen.apply_shot_noise(signal)
    >>> noisy = noise_gen.apply_read_noise(noisy, sigma=25.0)
    >>> print(f"SNR: {ctx.mean(signal) / ctx.sqrt(ctx.mean((noisy - signal)**2)):.1f}")
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

from eosim.core.compat import integrate_trapz
from .context import GPUContext, to_cpu


class GPUConvolver:
    """GPU-accelerated 2D convolution operations.

    Uses FFT-based convolution for efficient large-kernel operations.
    Automatically falls back to CPU if GPU is not available.
    """

    def __init__(self, ctx: Optional[GPUContext] = None) -> None:
        """Initialize GPU convolver.

        Args:
            ctx: GPU context. Creates new one if not provided.
        """
        self._ctx = ctx or GPUContext()
        self._xp = self._ctx.xp

    @property
    def device_name(self) -> str:
        """Get device name."""
        return self._ctx.device_name

    def convolve2d(
        self,
        image,
        kernel,
        mode: str = "same",
        boundary: str = "wrap",
    ):
        """Perform 2D convolution using FFT.

        Args:
            image: Input image array
            kernel: Convolution kernel
            mode: Output size mode ('same', 'full', 'valid')
            boundary: Boundary handling ('wrap', 'reflect', 'constant')

        Returns:
            Convolved image
        """
        xp = self._xp
        image = xp.asarray(image)
        kernel = xp.asarray(kernel)

        # Get dimensions
        ih, iw = image.shape[-2:]
        kh, kw = kernel.shape[-2:]

        # Compute padded size for FFT (power of 2 for efficiency)
        fh = _next_power_of_2(ih + kh - 1)
        fw = _next_power_of_2(iw + kw - 1)

        # Pad image based on boundary condition
        if boundary == "wrap":
            padded = xp.zeros((fh, fw), dtype=image.dtype)
            padded[:ih, :iw] = image
        elif boundary == "reflect":
            # Reflect padding
            pad_h = (fh - ih) // 2
            pad_w = (fw - iw) // 2
            padded = xp.pad(
                image,
                ((pad_h, fh - ih - pad_h), (pad_w, fw - iw - pad_w)),
                mode="reflect",
            )
        else:  # constant (zero)
            padded = xp.zeros((fh, fw), dtype=image.dtype)
            padded[:ih, :iw] = image

        # Pad kernel (center it)
        kernel_padded = xp.zeros((fh, fw), dtype=kernel.dtype)
        kh2, kw2 = kh // 2, kw // 2
        kernel_padded[:kh, :kw] = kernel

        # Roll kernel to center it at origin
        kernel_padded = xp.roll(kernel_padded, (-kh2, -kw2), axis=(0, 1))

        # FFT convolution
        image_fft = xp.fft.fft2(padded)
        kernel_fft = xp.fft.fft2(kernel_padded)
        result_fft = image_fft * kernel_fft
        result = xp.real(xp.fft.ifft2(result_fft))

        # Extract output based on mode
        if mode == "same":
            return result[:ih, :iw]
        elif mode == "full":
            return result[: ih + kh - 1, : iw + kw - 1]
        else:  # valid
            return result[kh - 1 : ih, kw - 1 : iw]

    def apply_psf(
        self,
        image,
        psf,
        normalize: bool = True,
    ):
        """Apply point spread function to image.

        Args:
            image: Input image
            psf: Point spread function kernel
            normalize: Normalize PSF to sum to 1

        Returns:
            PSF-convolved image
        """
        xp = self._xp
        psf = xp.asarray(psf)

        if normalize:
            psf = psf / xp.sum(psf)

        return self.convolve2d(image, psf, mode="same")

    def apply_mtf(
        self,
        image,
        mtf,
    ):
        """Apply modulation transfer function in frequency domain.

        Args:
            image: Input image
            mtf: 2D MTF array (same size as image or will be resized)

        Returns:
            MTF-filtered image
        """
        xp = self._xp
        image = xp.asarray(image)
        mtf = xp.asarray(mtf)

        # Resize MTF if needed
        if mtf.shape != image.shape:
            # Simple resize using interpolation
            from scipy import ndimage

            scale_h = image.shape[0] / mtf.shape[0]
            scale_w = image.shape[1] / mtf.shape[1]
            mtf_cpu = to_cpu(mtf)
            mtf_resized = ndimage.zoom(mtf_cpu, (scale_h, scale_w), order=1)
            mtf = xp.asarray(mtf_resized)

        # Apply MTF in frequency domain
        image_fft = xp.fft.fftshift(xp.fft.fft2(image))
        filtered_fft = image_fft * mtf
        result = xp.real(xp.fft.ifft2(xp.fft.ifftshift(filtered_fft)))

        return result


class GPURadiance:
    """GPU-accelerated radiance calculations.

    Provides efficient batch computation of Planck radiance
    and related radiometric quantities.
    """

    # Physical constants
    H = 6.62607015e-34  # Planck constant [J·s]
    C = 299792458.0  # Speed of light [m/s]
    K_B = 1.380649e-23  # Boltzmann constant [J/K]
    SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m²·K⁴)]

    def __init__(self, ctx: Optional[GPUContext] = None) -> None:
        """Initialize GPU radiance calculator.

        Args:
            ctx: GPU context. Creates new one if not provided.
        """
        self._ctx = ctx or GPUContext()
        self._xp = self._ctx.xp

    def planck_radiance(
        self,
        temperature_k,
        wavelength_um: float,
    ):
        """Compute Planck spectral radiance.

        L = (2hc²/λ⁵) × 1/(exp(hc/λkT) - 1)

        Args:
            temperature_k: Temperature array [K]
            wavelength_um: Wavelength [μm]

        Returns:
            Spectral radiance [W/(m²·sr·μm)]
        """
        return self._ctx.planck_radiance(temperature_k, wavelength_um)

    def planck_spectral(
        self,
        temperatures_k,
        wavelengths_um,
    ):
        """Compute Planck radiance for multiple temperatures and wavelengths.

        Args:
            temperatures_k: 1D array of temperatures [K]
            wavelengths_um: 1D array of wavelengths [μm]

        Returns:
            2D array of radiance [n_temps × n_wavelengths]
        """
        xp = self._xp
        T = xp.asarray(temperatures_k)
        wl = xp.asarray(wavelengths_um)

        # Reshape for broadcasting: T is (n_t, 1), wl is (1, n_w)
        T = T.reshape(-1, 1)
        wl = wl.reshape(1, -1)

        wavelength_m = wl * 1e-6

        c1 = 2 * self.H * self.C**2
        c2 = self.H * self.C / self.K_B

        T = xp.maximum(T, 1.0)
        exponent = c2 / (wavelength_m * T)
        exponent = xp.minimum(exponent, 700)

        radiance = c1 / (wavelength_m**5) / (xp.exp(exponent) - 1)
        return radiance * 1e-6

    def band_integrated_radiance(
        self,
        temperature_k,
        wavelength_min_um: float,
        wavelength_max_um: float,
        n_points: int = 100,
    ):
        """Compute band-integrated radiance.

        Args:
            temperature_k: Temperature array [K]
            wavelength_min_um: Minimum wavelength [μm]
            wavelength_max_um: Maximum wavelength [μm]
            n_points: Number of integration points

        Returns:
            Band-integrated radiance [W/(m²·sr)]
        """
        xp = self._xp
        wavelengths = xp.linspace(wavelength_min_um, wavelength_max_um, n_points)
        delta_wl = (wavelength_max_um - wavelength_min_um) / (n_points - 1)

        T = xp.asarray(temperature_k)
        T_flat = T.flatten()

        # Compute spectral radiance
        spectral = self.planck_spectral(T_flat, wavelengths)

        # Trapezoidal integration (use compat function for numpy 2.x)
        spectral_cpu = to_cpu(spectral)
        integrated = integrate_trapz(spectral_cpu, axis=1) * delta_wl

        return xp.asarray(integrated).reshape(T.shape)

    def stefan_boltzmann(
        self,
        temperature_k,
        emissivity: float = 1.0,
    ):
        """Compute total emitted radiance using Stefan-Boltzmann law.

        M = ε × σ × T⁴

        Args:
            temperature_k: Temperature array [K]
            emissivity: Surface emissivity (0-1)

        Returns:
            Total exitance [W/m²]
        """
        xp = self._xp
        T = xp.asarray(temperature_k)
        return emissivity * self.SIGMA * T**4

    def wien_peak(
        self,
        temperature_k,
    ):
        """Compute Wien peak wavelength.

        λ_max = b / T, where b = 2897.77 μm·K

        Args:
            temperature_k: Temperature array [K]

        Returns:
            Peak wavelength [μm]
        """
        xp = self._xp
        T = xp.asarray(temperature_k)
        T = xp.maximum(T, 1.0)
        b = 2897.77  # Wien's displacement constant [μm·K]
        return b / T

    def apparent_temperature(
        self,
        radiance,
        wavelength_um: float,
    ):
        """Compute apparent temperature from radiance (inverse Planck).

        Args:
            radiance: Spectral radiance [W/(m²·sr·μm)]
            wavelength_um: Wavelength [μm]

        Returns:
            Apparent temperature [K]
        """
        xp = self._xp
        L = xp.asarray(radiance)
        wavelength_m = wavelength_um * 1e-6

        c1 = 2 * self.H * self.C**2
        c2 = self.H * self.C / self.K_B

        # Solve Planck equation for T
        L_scaled = L * 1e6  # Convert from per-μm to per-m
        L_scaled = xp.maximum(L_scaled, 1e-20)  # Avoid log(0)

        T = c2 / (wavelength_m * xp.log(c1 / (wavelength_m**5 * L_scaled) + 1))

        return T


class GPUNoiseGenerator:
    """GPU-accelerated noise generation for sensor simulation.

    Implements various noise sources with proper statistics:
    - Shot noise (Poisson)
    - Read noise (Gaussian)
    - Dark current
    - Fixed pattern noise (PRNU, DSNU)
    """

    def __init__(
        self,
        ctx: Optional[GPUContext] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize GPU noise generator.

        Args:
            ctx: GPU context. Creates new one if not provided.
            seed: Random seed for reproducibility
        """
        self._ctx = ctx or GPUContext()
        self._xp = self._ctx.xp
        self._seed = seed

        if seed is not None:
            if self._ctx.is_gpu:
                self._xp.random.seed(seed)
            else:
                self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

    def apply_shot_noise(
        self,
        signal,
    ):
        """Apply shot noise (Poisson statistics).

        Args:
            signal: Signal in electrons

        Returns:
            Signal with shot noise
        """
        xp = self._xp
        signal = xp.asarray(signal)
        signal = xp.maximum(signal, 0)  # Ensure non-negative

        if self._ctx.is_gpu:
            return xp.random.poisson(signal).astype(signal.dtype)
        else:
            return self._rng.poisson(signal).astype(signal.dtype)

    def apply_read_noise(
        self,
        signal,
        sigma: float,
    ):
        """Apply read noise (Gaussian).

        Args:
            signal: Signal in electrons
            sigma: Read noise standard deviation [electrons]

        Returns:
            Signal with read noise
        """
        xp = self._xp
        signal = xp.asarray(signal)

        if self._ctx.is_gpu:
            noise = xp.random.normal(0, sigma, signal.shape)
        else:
            noise = self._rng.normal(0, sigma, signal.shape)

        return signal + noise.astype(signal.dtype)

    def apply_dark_current(
        self,
        signal,
        dark_current_e_per_s: float,
        integration_time_s: float,
    ):
        """Apply dark current noise.

        Args:
            signal: Signal in electrons
            dark_current_e_per_s: Dark current rate [e⁻/pixel/s]
            integration_time_s: Integration time [s]

        Returns:
            Signal with dark current
        """
        xp = self._xp
        signal = xp.asarray(signal)

        # Mean dark signal
        dark_mean = dark_current_e_per_s * integration_time_s

        # Dark current has Poisson statistics
        if self._ctx.is_gpu:
            dark = xp.random.poisson(dark_mean, signal.shape)
        else:
            dark = self._rng.poisson(dark_mean, signal.shape)

        return signal + dark.astype(signal.dtype)

    def apply_prnu(
        self,
        signal,
        prnu_sigma: float,
        prnu_map=None,
    ):
        """Apply Photo-Response Non-Uniformity (gain variation).

        Args:
            signal: Signal in electrons
            prnu_sigma: PRNU standard deviation (fractional, e.g., 0.01 = 1%)
            prnu_map: Optional fixed PRNU map. Generated if None.

        Returns:
            Signal with PRNU applied
        """
        xp = self._xp
        signal = xp.asarray(signal)

        if prnu_map is None:
            if self._ctx.is_gpu:
                prnu_map = 1.0 + xp.random.normal(0, prnu_sigma, signal.shape)
            else:
                prnu_map = 1.0 + self._rng.normal(0, prnu_sigma, signal.shape)

        return signal * prnu_map

    def apply_dsnu(
        self,
        signal,
        dsnu_sigma: float,
        dsnu_map=None,
    ):
        """Apply Dark Signal Non-Uniformity (offset variation).

        Args:
            signal: Signal in electrons
            dsnu_sigma: DSNU standard deviation [electrons]
            dsnu_map: Optional fixed DSNU map. Generated if None.

        Returns:
            Signal with DSNU applied
        """
        xp = self._xp
        signal = xp.asarray(signal)

        if dsnu_map is None:
            if self._ctx.is_gpu:
                dsnu_map = xp.random.normal(0, dsnu_sigma, signal.shape)
            else:
                dsnu_map = self._rng.normal(0, dsnu_sigma, signal.shape)

        return signal + dsnu_map

    def generate_fpn_maps(
        self,
        shape: Tuple[int, int],
        prnu_sigma: float = 0.01,
        dsnu_sigma: float = 10.0,
    ) -> Tuple:
        """Generate fixed pattern noise maps.

        These can be reused across frames for temporal consistency.

        Args:
            shape: Array shape (height, width)
            prnu_sigma: PRNU standard deviation (fractional)
            dsnu_sigma: DSNU standard deviation [electrons]

        Returns:
            Tuple of (prnu_map, dsnu_map)
        """
        xp = self._xp

        if self._ctx.is_gpu:
            prnu_map = 1.0 + xp.random.normal(0, prnu_sigma, shape)
            dsnu_map = xp.random.normal(0, dsnu_sigma, shape)
        else:
            prnu_map = 1.0 + self._rng.normal(0, prnu_sigma, shape)
            dsnu_map = self._rng.normal(0, dsnu_sigma, shape)

        return prnu_map, dsnu_map

    def apply_all_noise(
        self,
        signal,
        read_noise_e: float = 25.0,
        dark_current_e_per_s: float = 100.0,
        integration_time_s: float = 0.01,
        prnu_sigma: float = 0.01,
        dsnu_sigma: float = 10.0,
        prnu_map=None,
        dsnu_map=None,
    ):
        """Apply all noise sources in correct order.

        Order: PRNU → Shot → Dark → DSNU → Read

        Args:
            signal: Clean signal in electrons
            read_noise_e: Read noise [electrons]
            dark_current_e_per_s: Dark current [e⁻/pixel/s]
            integration_time_s: Integration time [s]
            prnu_sigma: PRNU standard deviation (fractional)
            dsnu_sigma: DSNU standard deviation [electrons]
            prnu_map: Optional fixed PRNU map
            dsnu_map: Optional fixed DSNU map

        Returns:
            Noisy signal
        """
        # 1. Apply PRNU (multiplicative)
        noisy = self.apply_prnu(signal, prnu_sigma, prnu_map)

        # 2. Apply shot noise
        noisy = self.apply_shot_noise(noisy)

        # 3. Apply dark current
        noisy = self.apply_dark_current(
            noisy, dark_current_e_per_s, integration_time_s
        )

        # 4. Apply DSNU (additive offset)
        noisy = self.apply_dsnu(noisy, dsnu_sigma, dsnu_map)

        # 5. Apply read noise
        noisy = self.apply_read_noise(noisy, read_noise_e)

        return noisy


def _next_power_of_2(n: int) -> int:
    """Find next power of 2 >= n."""
    return 1 << (n - 1).bit_length()
