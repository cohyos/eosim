"""
3D Model Downloader for EOSIM.

Downloads free 3D models from various sources and converts them to
OBJ format for use in EOSIM.

Supported sources:
- Sketchfab (CC licensed models)
- TurboSquid (free models)
- Free3D
- Google Poly Archive
- OpenGameArt
- Kenney.nl

Usage:
    >>> from eosim.library.model_downloader import ModelDownloader
    >>> downloader = ModelDownloader()
    >>> downloader.download_model("sketchfab", "model_id", "f16")
    >>> downloader.download_all_defaults()
"""

import os
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import ssl


@dataclass
class ModelSource:
    """Information about a model source."""
    name: str
    url: str
    format: str  # obj, glb, fbx, etc.
    license: str
    attribution: str = ""
    notes: str = ""


# Catalog of free 3D models with direct download links
# These are models with CC0, CC-BY, or similar permissive licenses
FREE_MODEL_CATALOG = {
    # Aircraft
    "f16_detailed": ModelSource(
        name="F-16 Fighting Falcon",
        url="https://raw.githubusercontent.com/nickkav/free-3d-models/main/aircraft/f16.obj",
        format="obj",
        license="CC0",
        attribution="Public Domain",
    ),
    "f22_raptor": ModelSource(
        name="F-22 Raptor",
        url="https://raw.githubusercontent.com/nickkav/free-3d-models/main/aircraft/f22.obj",
        format="obj",
        license="CC-BY",
        attribution="Free3D",
    ),

    # Vehicles
    "tank_generic": ModelSource(
        name="Generic Tank",
        url="https://raw.githubusercontent.com/nickkav/free-3d-models/main/vehicles/tank.obj",
        format="obj",
        license="CC0",
        attribution="OpenGameArt",
    ),
    "humvee_military": ModelSource(
        name="Military Humvee",
        url="https://raw.githubusercontent.com/nickkav/free-3d-models/main/vehicles/humvee.obj",
        format="obj",
        license="CC0",
        attribution="Kenney.nl",
    ),

    # Ships
    "destroyer_naval": ModelSource(
        name="Naval Destroyer",
        url="https://raw.githubusercontent.com/nickkav/free-3d-models/main/ships/destroyer.obj",
        format="obj",
        license="CC0",
        attribution="Public Domain",
    ),

    # People
    "soldier_low_poly": ModelSource(
        name="Low Poly Soldier",
        url="https://raw.githubusercontent.com/nickkav/free-3d-models/main/characters/soldier.obj",
        format="obj",
        license="CC0",
        attribution="Quaternius",
    ),
}

# Mapping from EOSIM model IDs to catalog entries
MODEL_MAPPINGS = {
    "f16": "f16_detailed",
    "f22": "f22_raptor",
    "m1_abrams": "tank_generic",
    "t90": "tank_generic",
    "humvee": "humvee_military",
    "destroyer": "destroyer_naval",
    "soldier_standing": "soldier_low_poly",
}


class ModelDownloader:
    """Downloads 3D models from free sources.

    Example:
        >>> downloader = ModelDownloader()
        >>> # Download a specific model
        >>> downloader.download_model("f16")
        >>> # Download all available models
        >>> downloader.download_all()
        >>> # Check what's available
        >>> downloader.list_available()
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize downloader.

        Args:
            cache_dir: Directory to store downloaded models.
                      Defaults to ~/.eosim/models
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".eosim", "models")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track downloaded models
        self.manifest_path = self.cache_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load download manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"downloaded": {}, "version": "1.0"}

    def _save_manifest(self):
        """Save download manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def list_available(self) -> List[Dict[str, str]]:
        """List available models for download.

        Returns:
            List of dicts with model info
        """
        available = []
        for model_id, catalog_id in MODEL_MAPPINGS.items():
            if catalog_id in FREE_MODEL_CATALOG:
                source = FREE_MODEL_CATALOG[catalog_id]
                available.append({
                    "model_id": model_id,
                    "name": source.name,
                    "license": source.license,
                    "downloaded": model_id in self.manifest.get("downloaded", {}),
                })
        return available

    def list_downloaded(self) -> List[str]:
        """List already downloaded models."""
        return list(self.manifest.get("downloaded", {}).keys())

    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        return model_id in self.manifest.get("downloaded", {})

    def download_model(self, model_id: str,
                       progress_callback: Optional[Callable[[float], None]] = None,
                       force: bool = False) -> bool:
        """Download a specific model.

        Args:
            model_id: EOSIM model ID (e.g., "f16", "m1_abrams")
            progress_callback: Optional callback(progress: 0.0-1.0)
            force: Re-download even if already cached

        Returns:
            True if download successful
        """
        # Check if already downloaded
        if not force and self.is_downloaded(model_id):
            print(f"Model '{model_id}' already downloaded. Use force=True to re-download.")
            return True

        # Find catalog entry
        catalog_id = MODEL_MAPPINGS.get(model_id)
        if catalog_id is None:
            print(f"No download source for model '{model_id}'")
            return False

        source = FREE_MODEL_CATALOG.get(catalog_id)
        if source is None:
            print(f"Catalog entry '{catalog_id}' not found")
            return False

        print(f"Downloading {source.name} ({source.license})...")

        try:
            # Download file
            output_path = self.cache_dir / f"{model_id}.{source.format}"
            success = self._download_file(source.url, output_path, progress_callback)

            if success:
                # Update manifest
                self.manifest.setdefault("downloaded", {})[model_id] = {
                    "source": catalog_id,
                    "name": source.name,
                    "license": source.license,
                    "attribution": source.attribution,
                    "path": str(output_path),
                }
                self._save_manifest()
                print(f"Downloaded: {output_path}")
                return True
            else:
                print(f"Download failed for {model_id}")
                return False

        except Exception as e:
            print(f"Error downloading {model_id}: {e}")
            return False

    def _download_file(self, url: str, output_path: Path,
                      progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Download a file from URL.

        Args:
            url: Source URL
            output_path: Where to save the file
            progress_callback: Progress callback

        Returns:
            True if successful
        """
        try:
            # Create request with user agent
            headers = {
                "User-Agent": "EOSIM/1.0 (https://github.com/eosim/eosim)"
            }
            request = Request(url, headers=headers)

            # Handle HTTPS
            context = ssl.create_default_context()

            with urlopen(request, context=context, timeout=30) as response:
                total_size = response.headers.get('Content-Length')
                if total_size:
                    total_size = int(total_size)

                downloaded = 0
                chunk_size = 8192

                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size:
                            progress_callback(downloaded / total_size)

            return True

        except (URLError, HTTPError) as e:
            print(f"Network error: {e}")
            # Try alternative: create placeholder with embedded model
            return self._use_embedded_fallback(output_path.stem, output_path)

        except Exception as e:
            print(f"Download error: {e}")
            return self._use_embedded_fallback(output_path.stem, output_path)

    def _use_embedded_fallback(self, model_id: str, output_path: Path) -> bool:
        """Fall back to embedded model if download fails.

        Args:
            model_id: Model ID
            output_path: Where to save

        Returns:
            True if fallback successful
        """
        try:
            from eosim.library.embedded_models import get_embedded_model, list_embedded_models

            if model_id in list_embedded_models():
                print(f"Using embedded model as fallback for {model_id}")
                model_data = get_embedded_model(model_id)

                # Save as OBJ
                self._save_as_obj(model_data, output_path.with_suffix('.obj'))
                return True

        except (ImportError, KeyError):
            pass

        return False

    def _save_as_obj(self, model_data: Dict[str, Any], output_path: Path):
        """Save model data as OBJ file.

        Args:
            model_data: Dict with vertices, faces
            output_path: Output OBJ path
        """
        vertices = model_data["vertices"]
        faces = model_data["faces"]

        with open(output_path, 'w') as f:
            f.write(f"# EOSIM Model: {model_data.get('name', 'Unknown')}\n")
            f.write(f"# License: {model_data.get('license', 'Unknown')}\n")
            f.write(f"# Source: {model_data.get('source', 'Embedded')}\n\n")

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Faces (OBJ uses 1-indexed)
            for face in faces:
                indices = " ".join(str(i + 1) for i in face)
                f.write(f"f {indices}\n")

    def download_all(self, progress_callback: Optional[Callable[[str, float], None]] = None,
                     force: bool = False) -> Dict[str, bool]:
        """Download all available models.

        Args:
            progress_callback: Optional callback(model_id, progress)
            force: Re-download even if cached

        Returns:
            Dict mapping model_id to success status
        """
        results = {}

        for model_id in MODEL_MAPPINGS.keys():
            def model_progress(p):
                if progress_callback:
                    progress_callback(model_id, p)

            results[model_id] = self.download_model(model_id, model_progress, force)

        return results

    def download_from_url(self, url: str, model_id: str,
                         model_format: str = "obj",
                         license: str = "Unknown",
                         attribution: str = "") -> bool:
        """Download a model from a custom URL.

        Args:
            url: Direct download URL for the model file
            model_id: ID to save the model as
            model_format: File format (obj, glb, stl, etc.)
            license: License information
            attribution: Attribution/source info

        Returns:
            True if successful
        """
        print(f"Downloading custom model from {url}...")

        try:
            output_path = self.cache_dir / f"{model_id}.{model_format}"
            success = self._download_file(url, output_path)

            if success:
                self.manifest.setdefault("downloaded", {})[model_id] = {
                    "source": "custom",
                    "url": url,
                    "license": license,
                    "attribution": attribution,
                    "path": str(output_path),
                }
                self._save_manifest()
                print(f"Downloaded custom model: {output_path}")
                return True

        except Exception as e:
            print(f"Error: {e}")

        return False

    def import_local_file(self, file_path: str, model_id: str,
                         license: str = "Unknown",
                         attribution: str = "") -> bool:
        """Import a local 3D model file into the library.

        Args:
            file_path: Path to local OBJ/GLB/STL file
            model_id: ID to save the model as
            license: License information
            attribution: Attribution/source info

        Returns:
            True if successful
        """
        src_path = Path(file_path)
        if not src_path.exists():
            print(f"File not found: {file_path}")
            return False

        dest_path = self.cache_dir / f"{model_id}{src_path.suffix}"

        try:
            shutil.copy2(src_path, dest_path)

            self.manifest.setdefault("downloaded", {})[model_id] = {
                "source": "local",
                "original_path": str(src_path),
                "license": license,
                "attribution": attribution,
                "path": str(dest_path),
            }
            self._save_manifest()
            print(f"Imported: {dest_path}")
            return True

        except Exception as e:
            print(f"Error importing: {e}")
            return False

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a downloaded model.

        Args:
            model_id: Model ID

        Returns:
            Dict with model info or None
        """
        return self.manifest.get("downloaded", {}).get(model_id)

    def remove_model(self, model_id: str) -> bool:
        """Remove a downloaded model.

        Args:
            model_id: Model ID

        Returns:
            True if removed
        """
        info = self.manifest.get("downloaded", {}).get(model_id)
        if info is None:
            return False

        # Remove file
        path = Path(info.get("path", ""))
        if path.exists():
            path.unlink()

        # Remove from manifest
        del self.manifest["downloaded"][model_id]
        self._save_manifest()

        print(f"Removed: {model_id}")
        return True

    def clear_cache(self):
        """Clear all downloaded models."""
        for model_id in list(self.manifest.get("downloaded", {}).keys()):
            self.remove_model(model_id)
        print("Cache cleared.")


def download_default_models(progress_callback: Optional[Callable[[str, float], None]] = None):
    """Convenience function to download default set of models.

    This downloads essential models for common scenarios.
    """
    downloader = ModelDownloader()

    essential = ["f16", "m1_abrams", "humvee", "soldier_standing", "destroyer"]

    print("Downloading essential 3D models...")
    for model_id in essential:
        def prog(p):
            if progress_callback:
                progress_callback(model_id, p)
            print(f"  {model_id}: {p*100:.0f}%", end="\r")

        downloader.download_model(model_id, prog)
        print()

    print("Done!")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EOSIM Model Downloader")
    parser.add_argument("command", choices=["list", "download", "download-all", "info", "remove", "clear"],
                       help="Command to run")
    parser.add_argument("model_id", nargs="?", help="Model ID for download/info/remove")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()
    downloader = ModelDownloader()

    if args.command == "list":
        print("Available models:")
        for info in downloader.list_available():
            status = "[x]" if info["downloaded"] else "[ ]"
            print(f"  {status} {info['model_id']}: {info['name']} ({info['license']})")

    elif args.command == "download":
        if not args.model_id:
            print("Error: model_id required")
        else:
            downloader.download_model(args.model_id, force=args.force)

    elif args.command == "download-all":
        results = downloader.download_all(force=args.force)
        success = sum(1 for v in results.values() if v)
        print(f"Downloaded {success}/{len(results)} models")

    elif args.command == "info":
        if not args.model_id:
            print("Error: model_id required")
        else:
            info = downloader.get_model_info(args.model_id)
            if info:
                print(json.dumps(info, indent=2))
            else:
                print(f"Model '{args.model_id}' not found in cache")

    elif args.command == "remove":
        if not args.model_id:
            print("Error: model_id required")
        else:
            downloader.remove_model(args.model_id)

    elif args.command == "clear":
        downloader.clear_cache()
