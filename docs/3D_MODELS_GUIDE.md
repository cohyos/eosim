# 3D Object Library Developer Guide

## Overview
The EOSIM 3D Object Library (`eosim.library`) provides a standardized way to manage, visualize, and use 3D assets in simulations. It bridges the gap between procedural placeholders and high-fidelity external models.

## Key Features
- **Hybrid Source System**: Supports both procedural generation (for immediate use) and file-based assets (OBJ/GLB/STL).
- **Standards Compliance**: Each object tracks its **SISO-REF-010** Enumeration ID, ensuring interoperability with DIS/HLA systems.
- **Metadata Rich**: Tracks source URLs, license info, and physical dimensions.

## Adding New Models

### 1. Registering Metadata
To add a new object, edit `src/eosim/library/models3d.py` and add an entry to `MODEL_SOURCES`:

```python
"new_tank": {
    "name": "New Main Battle Tank",
    "category": "vehicle",
    "dimensions": {"length": 10.0, "width": 3.5, "height": 2.5},
    "source": "procedural",
    "siso_id": "1.1.225.1.1.99",  # Look up in SISO-REF-010
    "source_url": "https://example.com/model",
    "license": "CC0"
}
```

### 2. Physical Assets (The "Drop-in" Workflow)
The system checks for local files before falling back to procedural generation.
To "install" a high-quality model for an ID (e.g., `aim120`):
1.  Obtain the file (OBJ, GLB, STL, PLY).
2.  Rename it to match the ID: `aim120.glb`.
3.  Place it in the user's cache directory:
    - Windows: `C:\Users\<User>\.eosim\models\`
    - Linux/Mac: `~/.eosim/models/`

The `ObjectViewer3D` will automatically load this file next time it launches.

### 3. Procedural Fallbacks
If no file is found, `ModelLibrary._generate_procedural()` is called.
- **New Categories**: If adding a new category, update `_generate_procedural` to call a specific creator method (e.g., `_create_missile`).

## Object Viewer
Located at `src/eosim/library/object_viewer.py`.
- **Launch**: Run `run_browser.bat` or `python -m eosim.library.object_viewer`.
- **Copy Snippet**: The viewer generates Python code for `scenario.add_target(...)` based on the selected object.

## Metadata Standards
### SISO-REF-010 IDs
We use the standard 7-digit enumeration:
`Entity.Kind.Domain.Country.Category.Subcategory.Specific`
Example (F-16): `1.2.225.1.1.3`
- Kind: 1 (Platform)
- Domain: 2 (Air)
- Country: 225 (USA)
- Category: 1 (Fighter)

### Licensing
Ensure `license` field is accurate. Preferred licenses for defaults:
- **CC0 (Public Domain)**
- **MIT / Apache 2.0**
- **NASA Open Data**

Avoid "Non-Commercial" (NC) assets for core libraries if possible, or clearly mark them.
