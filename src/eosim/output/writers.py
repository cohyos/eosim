"""
Output writers for saving simulation results.

Supports multiple formats including:
- NumPy arrays (.npy, .npz)
- Standard image formats (PNG, TIFF, JPEG)
- ENVI format for hyperspectral
- HDF5 for complex datasets
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
import json
from datetime import datetime


@dataclass
class ImageMetadata:
    """Metadata for saved images.

    Attributes:
        timestamp: Simulation timestamp
        sensor_type: Sensor type used
        spectral_band: Spectral band (min, max) in μm
        integration_time: Integration time in seconds
        range_m: Range to target in meters
        custom: Additional custom metadata
    """
    timestamp: str = ""
    sensor_type: str = ""
    spectral_band: tuple[float, float] = (0.0, 0.0)
    integration_time: float = 0.0
    range_m: float = 0.0
    custom: dict = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.custom is None:
            self.custom = {}

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'sensor_type': self.sensor_type,
            'spectral_band': self.spectral_band,
            'integration_time': self.integration_time,
            'range_m': self.range_m,
            'custom': self.custom,
        }


def save_numpy(
    filepath: Union[str, Path],
    image: NDArray,
    metadata: Optional[ImageMetadata] = None,
    compressed: bool = True,
) -> None:
    """Save image as NumPy file.

    Args:
        filepath: Output file path
        image: Image array
        metadata: Optional metadata
        compressed: Use compressed format (.npz)
    """
    filepath = Path(filepath)

    if compressed or filepath.suffix == '.npz':
        data = {'image': image}
        if metadata is not None:
            data['metadata'] = np.array([json.dumps(metadata.to_dict())])
        np.savez_compressed(filepath.with_suffix('.npz'), **data)
    else:
        np.save(filepath.with_suffix('.npy'), image)


def load_numpy(
    filepath: Union[str, Path],
) -> tuple[NDArray, Optional[ImageMetadata]]:
    """Load image from NumPy file.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (image, metadata)
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npz':
        with np.load(filepath) as data:
            image = data['image']
            metadata = None
            if 'metadata' in data:
                meta_dict = json.loads(str(data['metadata'][0]))
                metadata = ImageMetadata(**meta_dict)
            return image, metadata
    else:
        return np.load(filepath), None


def save_png(
    filepath: Union[str, Path],
    image: NDArray,
    normalize: bool = True,
    bit_depth: int = 16,
) -> None:
    """Save image as PNG.

    Args:
        filepath: Output file path
        image: Image array
        normalize: Normalize to full range
        bit_depth: Output bit depth (8 or 16)
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required for PNG output: pip install Pillow")

    filepath = Path(filepath).with_suffix('.png')

    if normalize:
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            image = (image - img_min) / (img_max - img_min)
        else:
            image = np.zeros_like(image)

    if bit_depth == 16:
        image = (image * 65535).astype(np.uint16)
        mode = 'I;16'
    else:
        image = (image * 255).astype(np.uint8)
        mode = 'L'

    pil_image = Image.fromarray(image, mode=mode)
    pil_image.save(filepath)


def save_tiff(
    filepath: Union[str, Path],
    image: NDArray,
    metadata: Optional[ImageMetadata] = None,
) -> None:
    """Save image as TIFF (preserves full precision).

    Args:
        filepath: Output file path
        image: Image array
        metadata: Optional metadata
    """
    try:
        from PIL import Image
        from PIL.TiffTags import TAGS
    except ImportError:
        raise ImportError("PIL required for TIFF output: pip install Pillow")

    filepath = Path(filepath).with_suffix('.tiff')

    # Convert to appropriate dtype
    if image.dtype == np.float64 or image.dtype == np.float32:
        pil_image = Image.fromarray(image.astype(np.float32), mode='F')
    elif image.dtype == np.uint16:
        pil_image = Image.fromarray(image, mode='I;16')
    else:
        pil_image = Image.fromarray(image.astype(np.uint8), mode='L')

    # Add metadata as TIFF tags
    tiff_info = {}
    if metadata is not None:
        tiff_info[270] = json.dumps(metadata.to_dict())  # ImageDescription

    pil_image.save(filepath, tiffinfo=tiff_info)


def save_raw(
    filepath: Union[str, Path],
    image: NDArray,
    metadata: Optional[ImageMetadata] = None,
) -> None:
    """Save image as raw binary with header file.

    Args:
        filepath: Output file path
        image: Image array
        metadata: Optional metadata
    """
    filepath = Path(filepath)
    raw_path = filepath.with_suffix('.raw')
    hdr_path = filepath.with_suffix('.hdr')

    # Save raw binary
    image.tofile(raw_path)

    # Save header
    header = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'byte_order': 'little' if image.dtype.byteorder in ['<', '='] else 'big',
    }
    if metadata is not None:
        header['metadata'] = metadata.to_dict()

    with open(hdr_path, 'w') as f:
        json.dump(header, f, indent=2)


def load_raw(
    filepath: Union[str, Path],
) -> tuple[NDArray, Optional[ImageMetadata]]:
    """Load raw binary image.

    Args:
        filepath: Input file path

    Returns:
        Tuple of (image, metadata)
    """
    filepath = Path(filepath)
    raw_path = filepath.with_suffix('.raw')
    hdr_path = filepath.with_suffix('.hdr')

    with open(hdr_path, 'r') as f:
        header = json.load(f)

    dtype = np.dtype(header['dtype'])
    shape = tuple(header['shape'])

    image = np.fromfile(raw_path, dtype=dtype).reshape(shape)

    metadata = None
    if 'metadata' in header:
        metadata = ImageMetadata(**header['metadata'])

    return image, metadata


def save_envi(
    filepath: Union[str, Path],
    image: NDArray,
    wavelengths: Optional[list[float]] = None,
    metadata: Optional[ImageMetadata] = None,
) -> None:
    """Save as ENVI format (common for hyperspectral).

    Args:
        filepath: Output file path
        image: Image array (2D or 3D)
        wavelengths: Wavelengths for each band (if 3D)
        metadata: Optional metadata
    """
    filepath = Path(filepath)
    dat_path = filepath.with_suffix('.dat')
    hdr_path = filepath.with_suffix('.hdr')

    # Ensure BSQ interleave (band sequential)
    image.tofile(dat_path)

    # Build ENVI header
    lines = [
        "ENVI",
        f"samples = {image.shape[-1]}",
        f"lines = {image.shape[-2]}",
    ]

    if image.ndim == 3:
        lines.append(f"bands = {image.shape[0]}")
        lines.append("interleave = bsq")
    else:
        lines.append("bands = 1")

    dtype_map = {
        'uint8': 1, 'int16': 2, 'int32': 3,
        'float32': 4, 'float64': 5, 'uint16': 12,
        'uint32': 13, 'int64': 14, 'uint64': 15,
    }
    lines.append(f"data type = {dtype_map.get(str(image.dtype), 4)}")
    lines.append("byte order = 0")  # Little endian

    if wavelengths is not None:
        wl_str = ', '.join(f"{w:.4f}" for w in wavelengths)
        lines.append(f"wavelength = {{{wl_str}}}")
        lines.append("wavelength units = Micrometers")

    if metadata is not None:
        lines.append(f"description = {{{json.dumps(metadata.to_dict())}}}")

    with open(hdr_path, 'w') as f:
        f.write('\n'.join(lines))


class SimulationWriter:
    """Writer for complete simulation results."""

    def __init__(
        self,
        output_dir: Union[str, Path],
        prefix: str = "sim",
    ) -> None:
        """Initialize writer.

        Args:
            output_dir: Output directory
            prefix: File name prefix
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self._counter = 0

    def _get_path(self, suffix: str) -> Path:
        """Get output file path."""
        name = f"{self.prefix}_{self._counter:04d}{suffix}"
        return self.output_dir / name

    def save(
        self,
        result: Any,  # PipelineResult
        format: str = "npz",
        include_intermediates: bool = False,
    ) -> Path:
        """Save simulation result.

        Args:
            result: PipelineResult object
            format: Output format ("npz", "png", "tiff", "raw")
            include_intermediates: Save intermediate images

        Returns:
            Path to saved file
        """
        metadata = ImageMetadata(
            custom=result.metadata if hasattr(result, 'metadata') else {},
        )

        if format == "npz":
            path = self._get_path('.npz')
            data = {'digital_image': result.digital_image}
            if result.radiance_image is not None:
                data['radiance_image'] = result.radiance_image
            if include_intermediates and hasattr(result, 'intermediate_images'):
                for name, img in result.intermediate_images.items():
                    data[f'intermediate_{name}'] = img
            data['metadata'] = np.array([json.dumps(metadata.to_dict())])
            np.savez_compressed(path, **data)

        elif format == "png":
            path = self._get_path('.png')
            save_png(path, result.digital_image)

        elif format == "tiff":
            path = self._get_path('.tiff')
            save_tiff(path, result.digital_image, metadata)

        elif format == "raw":
            path = self._get_path('.raw')
            save_raw(path, result.digital_image, metadata)

        else:
            raise ValueError(f"Unknown format: {format}")

        self._counter += 1
        return path
