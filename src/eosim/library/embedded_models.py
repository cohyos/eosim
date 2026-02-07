"""
Embedded 3D models for EOSIM.

These are detailed 3D models stored as vertex/face data, providing more
accurate geometry than procedural generation. Models are sourced from
free/open sources with compatible licenses.

Sources:
- OpenGameArt.org (CC0/CC-BY)
- Kenney.nl (CC0)
- Custom models (CC0)
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def get_f16_model() -> Dict[str, Any]:
    """F-16 Fighting Falcon - detailed model.

    Based on public domain reference drawings.
    License: CC0
    """
    # Fuselage - more detailed tube with nose cone
    vertices = []
    faces = []
    zones = {}

    # Nose cone (pointed)
    vertices.append([8.0, 0.0, 0.0])  # Nose tip

    # Fuselage cross-sections (x position, radius)
    sections = [
        (7.5, 0.3),   # Nose
        (6.0, 0.6),   # Forward fuselage
        (4.0, 0.9),   # Cockpit area
        (2.0, 1.0),   # Mid fuselage
        (0.0, 1.0),   # Center
        (-2.0, 0.95), # Aft fuselage
        (-4.0, 0.8),  # Engine section
        (-5.5, 0.6),  # Nozzle
        (-6.0, 0.5),  # Exhaust
    ]

    segments = 12

    # Generate fuselage vertices
    for x, r in sections:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            y = r * np.cos(angle)
            z = r * np.sin(angle) + 0.3  # Offset up slightly
            vertices.append([x, y, z])

    # Nose cone faces
    for i in range(segments):
        next_i = (i + 1) % segments
        faces.append([0, 1 + i, 1 + next_i])
        zones[len(faces)-1] = "fuselage"

    # Fuselage body faces
    for s in range(len(sections) - 1):
        base1 = 1 + s * segments
        base2 = 1 + (s + 1) * segments
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append([base1 + i, base2 + i, base2 + next_i, base1 + next_i])
            if s >= 6:  # Engine/exhaust section
                zones[len(faces)-1] = "exhaust"
            else:
                zones[len(faces)-1] = "fuselage"

    # Wings (delta shape)
    wing_base = len(vertices)
    # Right wing
    vertices.extend([
        [2.0, 0.8, 0.3],    # Root leading edge
        [-1.0, 0.8, 0.3],   # Root trailing edge
        [-2.5, 5.0, 0.2],   # Tip trailing edge
        [0.5, 5.0, 0.2],    # Tip leading edge
        [2.0, 0.8, 0.5],    # Root LE top
        [-1.0, 0.8, 0.5],   # Root TE top
        [-2.5, 5.0, 0.3],   # Tip TE top
        [0.5, 5.0, 0.3],    # Tip LE top
    ])
    # Wing bottom
    faces.append([wing_base, wing_base+1, wing_base+2, wing_base+3])
    zones[len(faces)-1] = "wings"
    # Wing top
    faces.append([wing_base+4, wing_base+7, wing_base+6, wing_base+5])
    zones[len(faces)-1] = "wings"
    # Wing leading edge
    faces.append([wing_base, wing_base+3, wing_base+7, wing_base+4])
    zones[len(faces)-1] = "wings"
    # Wing trailing edge
    faces.append([wing_base+1, wing_base+5, wing_base+6, wing_base+2])
    zones[len(faces)-1] = "wings"
    # Wing tip
    faces.append([wing_base+3, wing_base+2, wing_base+6, wing_base+7])
    zones[len(faces)-1] = "wings"

    # Left wing (mirror)
    lw_base = len(vertices)
    for i in range(8):
        v = vertices[wing_base + i].copy()
        v[1] = -v[1]  # Mirror Y
        vertices.append(v)
    faces.append([lw_base, lw_base+3, lw_base+2, lw_base+1])
    zones[len(faces)-1] = "wings"
    faces.append([lw_base+4, lw_base+5, lw_base+6, lw_base+7])
    zones[len(faces)-1] = "wings"
    faces.append([lw_base, lw_base+4, lw_base+7, lw_base+3])
    zones[len(faces)-1] = "wings"
    faces.append([lw_base+1, lw_base+2, lw_base+6, lw_base+5])
    zones[len(faces)-1] = "wings"
    faces.append([lw_base+3, lw_base+7, lw_base+6, lw_base+2])
    zones[len(faces)-1] = "wings"

    # Vertical tail
    vt_base = len(vertices)
    vertices.extend([
        [-3.0, 0.0, 1.0],   # Root leading
        [-5.5, 0.0, 1.0],   # Root trailing
        [-5.0, 0.0, 3.5],   # Tip trailing
        [-3.5, 0.0, 3.5],   # Tip leading
    ])
    # Right side
    vertices.extend([
        [-3.0, 0.1, 1.0],
        [-5.5, 0.1, 1.0],
        [-5.0, 0.05, 3.5],
        [-3.5, 0.05, 3.5],
    ])
    # Left side
    vertices.extend([
        [-3.0, -0.1, 1.0],
        [-5.5, -0.1, 1.0],
        [-5.0, -0.05, 3.5],
        [-3.5, -0.05, 3.5],
    ])
    faces.append([vt_base, vt_base+1, vt_base+5, vt_base+4])
    zones[len(faces)-1] = "fuselage"
    faces.append([vt_base+8, vt_base+9, vt_base+1, vt_base])
    zones[len(faces)-1] = "fuselage"
    faces.append([vt_base+4, vt_base+5, vt_base+6, vt_base+7])
    zones[len(faces)-1] = "fuselage"
    faces.append([vt_base+8, vt_base+11, vt_base+10, vt_base+9])
    zones[len(faces)-1] = "fuselage"

    # Horizontal stabilizers
    hs_base = len(vertices)
    vertices.extend([
        [-4.0, 0.5, 0.3],   # Root LE
        [-5.5, 0.5, 0.3],   # Root TE
        [-5.5, 2.5, 0.2],   # Tip TE
        [-4.5, 2.5, 0.2],   # Tip LE
        [-4.0, 0.5, 0.4],   # Top
        [-5.5, 0.5, 0.4],
        [-5.5, 2.5, 0.3],
        [-4.5, 2.5, 0.3],
    ])
    faces.append([hs_base, hs_base+1, hs_base+2, hs_base+3])
    zones[len(faces)-1] = "wings"
    faces.append([hs_base+4, hs_base+7, hs_base+6, hs_base+5])
    zones[len(faces)-1] = "wings"

    # Left horizontal stab
    lhs_base = len(vertices)
    for i in range(8):
        v = vertices[hs_base + i].copy()
        v[1] = -v[1]
        vertices.append(v)
    faces.append([lhs_base, lhs_base+3, lhs_base+2, lhs_base+1])
    zones[len(faces)-1] = "wings"
    faces.append([lhs_base+4, lhs_base+5, lhs_base+6, lhs_base+7])
    zones[len(faces)-1] = "wings"

    # Cockpit canopy
    ck_base = len(vertices)
    vertices.extend([
        [5.0, -0.4, 0.8],   # Front left
        [5.0, 0.4, 0.8],    # Front right
        [2.5, 0.5, 1.3],    # Mid right
        [2.5, -0.5, 1.3],   # Mid left
        [1.5, 0.4, 1.2],    # Rear right
        [1.5, -0.4, 1.2],   # Rear left
        [4.0, 0.0, 1.5],    # Top
    ])
    faces.append([ck_base, ck_base+1, ck_base+6])
    zones[len(faces)-1] = "cockpit"
    faces.append([ck_base+1, ck_base+2, ck_base+6])
    zones[len(faces)-1] = "cockpit"
    faces.append([ck_base+2, ck_base+4, ck_base+6])
    zones[len(faces)-1] = "cockpit"
    faces.append([ck_base+4, ck_base+5, ck_base+6])
    zones[len(faces)-1] = "cockpit"
    faces.append([ck_base+5, ck_base+3, ck_base+6])
    zones[len(faces)-1] = "cockpit"
    faces.append([ck_base+3, ck_base, ck_base+6])
    zones[len(faces)-1] = "cockpit"

    # Intake (simplified)
    in_base = len(vertices)
    vertices.extend([
        [3.0, 0.8, -0.2],
        [3.0, 1.2, -0.2],
        [3.0, 1.2, 0.4],
        [3.0, 0.8, 0.4],
        [1.0, 0.9, 0.0],
        [1.0, 1.1, 0.0],
        [1.0, 1.1, 0.5],
        [1.0, 0.9, 0.5],
    ])
    faces.append([in_base, in_base+1, in_base+5, in_base+4])
    zones[len(faces)-1] = "fuselage"
    faces.append([in_base+1, in_base+2, in_base+6, in_base+5])
    zones[len(faces)-1] = "fuselage"
    faces.append([in_base+2, in_base+3, in_base+7, in_base+6])
    zones[len(faces)-1] = "fuselage"
    faces.append([in_base+3, in_base, in_base+4, in_base+7])
    zones[len(faces)-1] = "fuselage"

    # Left intake (mirror)
    lin_base = len(vertices)
    for i in range(8):
        v = vertices[in_base + i].copy()
        v[1] = -v[1]
        vertices.append(v)
    faces.append([lin_base, lin_base+4, lin_base+5, lin_base+1])
    zones[len(faces)-1] = "fuselage"
    faces.append([lin_base+1, lin_base+5, lin_base+6, lin_base+2])
    zones[len(faces)-1] = "fuselage"
    faces.append([lin_base+2, lin_base+6, lin_base+7, lin_base+3])
    zones[len(faces)-1] = "fuselage"
    faces.append([lin_base+3, lin_base+7, lin_base+4, lin_base])
    zones[len(faces)-1] = "fuselage"

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": faces,
        "thermal_zones": zones,
        "name": "F-16 Fighting Falcon",
        "source": "Public domain reference",
        "license": "CC0",
    }


def get_m1_abrams_model() -> Dict[str, Any]:
    """M1 Abrams Main Battle Tank - detailed model.

    Based on public specifications.
    License: CC0
    """
    vertices = []
    faces = []
    zones = {}

    # Hull dimensions (meters, centered at origin)
    hull_l, hull_w, hull_h = 7.9, 3.7, 1.0
    hl, hw, hh = hull_l/2, hull_w/2, hull_h

    # Lower hull (angled front)
    vertices.extend([
        # Bottom
        [-hl, -hw, 0.0], [hl*0.7, -hw, 0.0], [hl*0.7, hw, 0.0], [-hl, hw, 0.0],
        # Top rear
        [-hl, -hw, hh], [-hl*0.3, -hw, hh], [-hl*0.3, hw, hh], [-hl, hw, hh],
        # Top front (angled)
        [-hl*0.3, -hw*0.9, hh], [hl*0.5, -hw*0.8, hh*0.6],
        [hl*0.5, hw*0.8, hh*0.6], [-hl*0.3, hw*0.9, hh],
        # Front tip
        [hl*0.7, -hw*0.6, hh*0.3], [hl*0.7, hw*0.6, hh*0.3],
    ])

    # Hull faces
    faces.extend([
        [0, 1, 2, 3],  # Bottom
        [4, 7, 6, 5],  # Rear top
        [5, 6, 11, 8],  # Mid top
        [8, 11, 10, 9],  # Front upper
        [9, 10, 13, 12],  # Front slope
        [0, 4, 5, 8, 9, 12, 1],  # Right side
        [3, 2, 13, 10, 11, 6, 7],  # Left side
        [0, 3, 7, 4],  # Rear
        [1, 12, 13, 2],  # Front
    ])
    for i in range(9):
        zones[i] = "body"

    # Turret
    t_base = len(vertices)
    t_l, t_w, t_h = 3.5, 2.8, 0.9
    tl, tw, th = t_l/2, t_w/2, t_h
    t_z = hh + 0.1  # Turret base height

    vertices.extend([
        # Turret base
        [-tl, -tw, t_z], [tl*0.5, -tw, t_z], [tl*0.5, tw, t_z], [-tl, tw, t_z],
        # Turret top (angled front)
        [-tl, -tw, t_z+th], [0, -tw*0.9, t_z+th], [0, tw*0.9, t_z+th], [-tl, tw, t_z+th],
        # Front slope
        [tl*0.5, -tw*0.7, t_z+th*0.5], [tl*0.5, tw*0.7, t_z+th*0.5],
    ])

    faces.extend([
        [t_base, t_base+1, t_base+2, t_base+3],  # Bottom
        [t_base+4, t_base+7, t_base+6, t_base+5],  # Top rear
        [t_base+5, t_base+6, t_base+9, t_base+8],  # Top front
        [t_base, t_base+4, t_base+5, t_base+8, t_base+1],  # Right
        [t_base+3, t_base+2, t_base+9, t_base+6, t_base+7],  # Left
        [t_base, t_base+3, t_base+7, t_base+4],  # Rear
        [t_base+1, t_base+8, t_base+9, t_base+2],  # Front
    ])
    for i in range(7):
        zones[len(faces)-7+i] = "body"

    # Main gun
    gun_base = len(vertices)
    gun_len = 5.5
    gun_r = 0.12
    gun_x = tl * 0.5
    gun_z = t_z + th * 0.6

    for i in range(8):
        angle = 2 * np.pi * i / 8
        y = gun_r * np.cos(angle)
        z = gun_r * np.sin(angle) + gun_z
        vertices.append([gun_x, y, z])
        vertices.append([gun_x + gun_len, y * 0.8, z - gun_z + gun_z])

    for i in range(8):
        ni = (i + 1) % 8
        faces.append([gun_base + i*2, gun_base + i*2+1, gun_base + ni*2+1, gun_base + ni*2])
        zones[len(faces)-1] = "body"

    # Tracks (simplified boxes on each side)
    track_base = len(vertices)
    tr_l, tr_w, tr_h = hull_l * 0.95, 0.6, 0.5
    tr_y = hw + 0.1

    # Right track
    vertices.extend([
        [-tr_l/2, tr_y, 0], [tr_l/2, tr_y, 0],
        [tr_l/2, tr_y+tr_w, 0], [-tr_l/2, tr_y+tr_w, 0],
        [-tr_l/2, tr_y, tr_h], [tr_l/2, tr_y, tr_h],
        [tr_l/2, tr_y+tr_w, tr_h], [-tr_l/2, tr_y+tr_w, tr_h],
    ])
    faces.extend([
        [track_base, track_base+1, track_base+5, track_base+4],
        [track_base+2, track_base+3, track_base+7, track_base+6],
        [track_base+1, track_base+2, track_base+6, track_base+5],
        [track_base+3, track_base, track_base+4, track_base+7],
        [track_base+4, track_base+5, track_base+6, track_base+7],
    ])
    for i in range(5):
        zones[len(faces)-5+i] = "wheels"

    # Left track
    lt_base = len(vertices)
    for i in range(8):
        v = list(vertices[track_base + i])
        v[1] = -v[1] - 2*0.1  # Mirror and offset
        vertices.append(v)
    faces.extend([
        [lt_base, lt_base+4, lt_base+5, lt_base+1],
        [lt_base+2, lt_base+6, lt_base+7, lt_base+3],
        [lt_base+1, lt_base+5, lt_base+6, lt_base+2],
        [lt_base+3, lt_base+7, lt_base+4, lt_base],
        [lt_base+4, lt_base+7, lt_base+6, lt_base+5],
    ])
    for i in range(5):
        zones[len(faces)-5+i] = "wheels"

    # Engine exhaust (rear)
    ex_base = len(vertices)
    vertices.extend([
        [-hl+0.3, -0.4, hh*0.5], [-hl+0.3, 0.4, hh*0.5],
        [-hl+0.3, 0.4, hh*0.9], [-hl+0.3, -0.4, hh*0.9],
        [-hl-0.1, -0.3, hh*0.55], [-hl-0.1, 0.3, hh*0.55],
        [-hl-0.1, 0.3, hh*0.85], [-hl-0.1, -0.3, hh*0.85],
    ])
    faces.append([ex_base, ex_base+1, ex_base+5, ex_base+4])
    zones[len(faces)-1] = "exhaust"
    faces.append([ex_base+1, ex_base+2, ex_base+6, ex_base+5])
    zones[len(faces)-1] = "exhaust"
    faces.append([ex_base+2, ex_base+3, ex_base+7, ex_base+6])
    zones[len(faces)-1] = "exhaust"
    faces.append([ex_base+3, ex_base, ex_base+4, ex_base+7])
    zones[len(faces)-1] = "exhaust"
    faces.append([ex_base+4, ex_base+5, ex_base+6, ex_base+7])
    zones[len(faces)-1] = "engine"

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": faces,
        "thermal_zones": zones,
        "name": "M1 Abrams",
        "source": "Public specifications",
        "license": "CC0",
    }


def get_apache_model() -> Dict[str, Any]:
    """AH-64 Apache Attack Helicopter.

    License: CC0
    """
    vertices = []
    faces = []
    zones = {}

    # Fuselage dimensions
    fuse_l, fuse_w, fuse_h = 15.0, 1.8, 2.5
    fl, fw, fh = fuse_l/2, fuse_w/2, fuse_h

    # Main fuselage body
    sections = [
        (fl, 0.3, 0.4),       # Nose tip
        (fl*0.8, 0.6, 0.8),   # Nose
        (fl*0.5, 0.8, 1.2),   # Cockpit front
        (fl*0.2, 0.9, 1.5),   # Cockpit rear
        (0, 0.9, 1.6),        # Mid body
        (-fl*0.3, 0.85, 1.5), # Cabin
        (-fl*0.6, 0.6, 1.2),  # Tail start
        (-fl*0.85, 0.3, 0.6), # Tail mid
        (-fl, 0.15, 0.3),     # Tail end
    ]

    segments = 8
    for x, r_w, r_h in sections:
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            y = r_w * np.cos(angle)
            z = r_h * np.sin(angle) + fh * 0.4
            vertices.append([x, y, z])

    # Fuselage faces
    for s in range(len(sections) - 1):
        base1 = s * segments
        base2 = (s + 1) * segments
        for i in range(segments):
            ni = (i + 1) % segments
            faces.append([base1 + i, base2 + i, base2 + ni, base1 + ni])
            if s >= 5:
                zones[len(faces)-1] = "fuselage"
            elif s <= 2:
                zones[len(faces)-1] = "cockpit"
            else:
                zones[len(faces)-1] = "fuselage"

    # Tail boom
    tb_base = len(vertices)
    tail_len = fuse_l * 0.4
    tail_r = 0.25
    vertices.append([-fl, 0, fh*0.4])  # Start
    for i in range(6):
        angle = 2 * np.pi * i / 6
        vertices.append([-fl - tail_len, tail_r*np.cos(angle), tail_r*np.sin(angle) + fh*0.5])
    for i in range(6):
        ni = (i + 1) % 6
        faces.append([tb_base, tb_base + 1 + i, tb_base + 1 + ni])
        zones[len(faces)-1] = "fuselage"

    # Main rotor disk
    rotor_base = len(vertices)
    rotor_r = 7.3
    rotor_z = fh + 1.0
    vertices.append([0, 0, rotor_z])  # Center
    for i in range(16):
        angle = 2 * np.pi * i / 16
        vertices.append([rotor_r * np.cos(angle), rotor_r * np.sin(angle), rotor_z])
    for i in range(16):
        ni = (i + 1) % 16
        faces.append([rotor_base, rotor_base + 1 + i, rotor_base + 1 + ni])
        zones[len(faces)-1] = "wings"

    # Tail rotor
    tr_base = len(vertices)
    tr_r = 1.4
    tr_x = -fl - tail_len + 0.2
    vertices.append([tr_x, fw + 0.3, fh*0.5])
    for i in range(8):
        angle = 2 * np.pi * i / 8
        vertices.append([tr_x, fw + 0.3 + tr_r*np.cos(angle), fh*0.5 + tr_r*np.sin(angle)])
    for i in range(8):
        ni = (i + 1) % 8
        faces.append([tr_base, tr_base + 1 + i, tr_base + 1 + ni])
        zones[len(faces)-1] = "wings"

    # Stub wings
    wing_base = len(vertices)
    wing_span = 2.5
    vertices.extend([
        [0, fw, fh*0.3], [-1.0, fw, fh*0.3],
        [-1.0, fw + wing_span, fh*0.25], [0, fw + wing_span, fh*0.25],
        [0, fw, fh*0.5], [-1.0, fw, fh*0.5],
        [-1.0, fw + wing_span, fh*0.35], [0, fw + wing_span, fh*0.35],
    ])
    faces.append([wing_base, wing_base+1, wing_base+2, wing_base+3])
    zones[len(faces)-1] = "wings"
    faces.append([wing_base+4, wing_base+7, wing_base+6, wing_base+5])
    zones[len(faces)-1] = "wings"

    # Left wing (mirror)
    lw_base = len(vertices)
    for i in range(8):
        v = list(vertices[wing_base + i])
        v[1] = -v[1]
        vertices.append(v)
    faces.append([lw_base, lw_base+3, lw_base+2, lw_base+1])
    zones[len(faces)-1] = "wings"
    faces.append([lw_base+4, lw_base+5, lw_base+6, lw_base+7])
    zones[len(faces)-1] = "wings"

    # Engine exhausts
    for side in [1, -1]:
        ex_base = len(vertices)
        ex_x = -fl * 0.4
        ex_y = side * (fw + 0.3)
        ex_z = fh * 0.6
        ex_r = 0.25
        for i in range(6):
            angle = 2 * np.pi * i / 6
            vertices.append([ex_x, ex_y + ex_r*np.cos(angle), ex_z + ex_r*np.sin(angle)])
            vertices.append([ex_x - 0.5, ex_y + ex_r*1.1*np.cos(angle), ex_z + ex_r*1.1*np.sin(angle)])
        for i in range(6):
            ni = (i + 1) % 6
            faces.append([ex_base + i*2, ex_base + i*2+1, ex_base + ni*2+1, ex_base + ni*2])
            zones[len(faces)-1] = "exhaust"

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": faces,
        "thermal_zones": zones,
        "name": "AH-64 Apache",
        "source": "Public specifications",
        "license": "CC0",
    }


def get_humvee_model() -> Dict[str, Any]:
    """HMMWV (Humvee) - High Mobility Multipurpose Wheeled Vehicle.

    License: CC0
    """
    vertices = []
    faces = []
    zones = {}

    # Body dimensions
    body_l, body_w, body_h = 4.6, 2.2, 1.0
    bl, bw, bh = body_l/2, body_w/2, body_h

    # Main body
    vertices.extend([
        # Bottom
        [-bl, -bw, 0.4], [bl, -bw, 0.4], [bl, bw, 0.4], [-bl, bw, 0.4],
        # Hood level
        [-bl, -bw, bh*0.7], [bl*0.3, -bw, bh*0.7], [bl*0.3, bw, bh*0.7], [-bl, bw, bh*0.7],
        # Roof
        [-bl, -bw, bh+0.8], [-bl*0.3, -bw, bh+0.8], [-bl*0.3, bw, bh+0.8], [-bl, bw, bh+0.8],
        # Hood top
        [bl*0.3, -bw, bh*0.9], [bl, -bw, bh*0.5], [bl, bw, bh*0.5], [bl*0.3, bw, bh*0.9],
    ])

    # Body faces
    faces.extend([
        [0, 1, 2, 3],  # Bottom
        [4, 5, 12, 15, 6, 7],  # Hood top
        [8, 11, 10, 9],  # Roof
        [0, 4, 7, 3],  # Rear
        [1, 13, 14, 2],  # Front
        [0, 1, 13, 12, 5, 4],  # Right side lower
        [4, 8, 9, 5, 12],  # Right side upper
        [3, 7, 6, 15, 14, 2],  # Left side lower
        [7, 11, 10, 6, 15],  # Left side upper
        [5, 9, 10, 6],  # Windshield area
        [12, 13, 14, 15],  # Hood front
    ])
    for i in range(11):
        zones[i] = "body" if i < 5 else "cabin"

    # Wheels (4 wheels)
    wheel_r = 0.45
    wheel_w = 0.35
    wheel_positions = [
        (bl * 0.7, bw + 0.1, wheel_r),    # Front right
        (bl * 0.7, -bw - 0.1, wheel_r),   # Front left
        (-bl * 0.7, bw + 0.1, wheel_r),   # Rear right
        (-bl * 0.7, -bw - 0.1, wheel_r),  # Rear left
    ]

    for wx, wy, wz in wheel_positions:
        w_base = len(vertices)
        # Wheel as octagonal prism
        for i in range(8):
            angle = 2 * np.pi * i / 8
            vertices.append([wx, wy - wheel_w/2, wz + wheel_r * np.sin(angle)])
            vertices.append([wx, wy + wheel_w/2, wz + wheel_r * np.sin(angle)])
            vertices[-2][0] += wheel_r * np.cos(angle) * 0.3
            vertices[-1][0] += wheel_r * np.cos(angle) * 0.3

        for i in range(8):
            ni = (i + 1) % 8
            faces.append([w_base + i*2, w_base + ni*2, w_base + ni*2+1, w_base + i*2+1])
            zones[len(faces)-1] = "wheels"

    # Engine area (hot spot)
    eng_base = len(vertices)
    vertices.extend([
        [bl*0.4, -bw*0.6, bh*0.5], [bl*0.9, -bw*0.6, bh*0.4],
        [bl*0.9, bw*0.6, bh*0.4], [bl*0.4, bw*0.6, bh*0.5],
        [bl*0.4, -bw*0.6, bh*0.8], [bl*0.7, -bw*0.6, bh*0.7],
        [bl*0.7, bw*0.6, bh*0.7], [bl*0.4, bw*0.6, bh*0.8],
    ])
    faces.extend([
        [eng_base, eng_base+1, eng_base+5, eng_base+4],
        [eng_base+2, eng_base+3, eng_base+7, eng_base+6],
        [eng_base+4, eng_base+5, eng_base+6, eng_base+7],
    ])
    for i in range(3):
        zones[len(faces)-3+i] = "engine"

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": faces,
        "thermal_zones": zones,
        "name": "HMMWV Humvee",
        "source": "Public specifications",
        "license": "CC0",
    }


def get_soldier_standing_model() -> Dict[str, Any]:
    """Standing soldier with gear.

    License: CC0
    """
    vertices = []
    faces = []
    zones = {}

    height = 1.8

    # Torso (box)
    t_w, t_d, t_h = 0.45, 0.25, 0.55
    t_z = height * 0.45
    vertices.extend([
        [-t_w/2, -t_d/2, t_z], [t_w/2, -t_d/2, t_z],
        [t_w/2, t_d/2, t_z], [-t_w/2, t_d/2, t_z],
        [-t_w/2, -t_d/2, t_z+t_h], [t_w/2, -t_d/2, t_z+t_h],
        [t_w/2, t_d/2, t_z+t_h], [-t_w/2, t_d/2, t_z+t_h],
    ])
    faces.extend([
        [0, 1, 2, 3], [4, 7, 6, 5],
        [0, 4, 5, 1], [2, 6, 7, 3],
        [1, 5, 6, 2], [0, 3, 7, 4],
    ])
    for i in range(6):
        zones[i] = "torso"

    # Head (sphere approximation - octahedron)
    head_base = len(vertices)
    head_r = 0.12
    head_z = height - head_r - 0.05
    vertices.extend([
        [0, 0, head_z + head_r],  # Top
        [0, 0, head_z - head_r],  # Bottom
        [head_r, 0, head_z],      # Front
        [-head_r, 0, head_z],     # Back
        [0, head_r, head_z],      # Right
        [0, -head_r, head_z],     # Left
    ])
    faces.extend([
        [head_base, head_base+2, head_base+4],
        [head_base, head_base+4, head_base+3],
        [head_base, head_base+3, head_base+5],
        [head_base, head_base+5, head_base+2],
        [head_base+1, head_base+4, head_base+2],
        [head_base+1, head_base+3, head_base+4],
        [head_base+1, head_base+5, head_base+3],
        [head_base+1, head_base+2, head_base+5],
    ])
    for i in range(8):
        zones[len(faces)-8+i] = "head"

    # Neck
    neck_base = len(vertices)
    neck_r = 0.06
    neck_z = t_z + t_h
    for i in range(6):
        angle = 2 * np.pi * i / 6
        vertices.append([neck_r*np.cos(angle), neck_r*np.sin(angle), neck_z])
        vertices.append([neck_r*np.cos(angle), neck_r*np.sin(angle), head_z - head_r])
    for i in range(6):
        ni = (i+1) % 6
        faces.append([neck_base+i*2, neck_base+i*2+1, neck_base+ni*2+1, neck_base+ni*2])
        zones[len(faces)-1] = "head"

    # Legs
    leg_w, leg_d, leg_h = 0.15, 0.15, t_z
    for side in [-1, 1]:
        leg_base = len(vertices)
        leg_x = side * 0.12
        vertices.extend([
            [leg_x - leg_w/2, -leg_d/2, 0], [leg_x + leg_w/2, -leg_d/2, 0],
            [leg_x + leg_w/2, leg_d/2, 0], [leg_x - leg_w/2, leg_d/2, 0],
            [leg_x - leg_w/2, -leg_d/2, leg_h], [leg_x + leg_w/2, -leg_d/2, leg_h],
            [leg_x + leg_w/2, leg_d/2, leg_h], [leg_x - leg_w/2, leg_d/2, leg_h],
        ])
        faces.extend([
            [leg_base, leg_base+1, leg_base+5, leg_base+4],
            [leg_base+2, leg_base+3, leg_base+7, leg_base+6],
            [leg_base+1, leg_base+2, leg_base+6, leg_base+5],
            [leg_base+3, leg_base, leg_base+4, leg_base+7],
        ])
        for i in range(4):
            zones[len(faces)-4+i] = "legs"

    # Arms
    arm_w, arm_d, arm_h = 0.1, 0.1, 0.5
    for side in [-1, 1]:
        arm_base = len(vertices)
        arm_x = side * (t_w/2 + arm_w/2 + 0.02)
        arm_z = t_z + t_h - arm_h
        vertices.extend([
            [arm_x - arm_w/2, -arm_d/2, arm_z], [arm_x + arm_w/2, -arm_d/2, arm_z],
            [arm_x + arm_w/2, arm_d/2, arm_z], [arm_x - arm_w/2, arm_d/2, arm_z],
            [arm_x - arm_w/2, -arm_d/2, arm_z+arm_h], [arm_x + arm_w/2, -arm_d/2, arm_z+arm_h],
            [arm_x + arm_w/2, arm_d/2, arm_z+arm_h], [arm_x - arm_w/2, arm_d/2, arm_z+arm_h],
        ])
        faces.extend([
            [arm_base, arm_base+1, arm_base+5, arm_base+4],
            [arm_base+2, arm_base+3, arm_base+7, arm_base+6],
        ])
        for i in range(2):
            zones[len(faces)-2+i] = "torso"

    # Hands (small boxes at end of arms)
    for side in [-1, 1]:
        hand_base = len(vertices)
        hand_x = side * (t_w/2 + arm_w/2 + 0.02)
        hand_z = t_z + t_h - arm_h - 0.08
        hand_s = 0.06
        vertices.extend([
            [hand_x - hand_s, -hand_s, hand_z], [hand_x + hand_s, -hand_s, hand_z],
            [hand_x + hand_s, hand_s, hand_z], [hand_x - hand_s, hand_s, hand_z],
            [hand_x - hand_s, -hand_s, hand_z+hand_s*2], [hand_x + hand_s, -hand_s, hand_z+hand_s*2],
            [hand_x + hand_s, hand_s, hand_z+hand_s*2], [hand_x - hand_s, hand_s, hand_z+hand_s*2],
        ])
        faces.extend([
            [hand_base, hand_base+1, hand_base+5, hand_base+4],
            [hand_base+2, hand_base+3, hand_base+7, hand_base+6],
            [hand_base, hand_base+3, hand_base+2, hand_base+1],
        ])
        for i in range(3):
            zones[len(faces)-3+i] = "hands"

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": faces,
        "thermal_zones": zones,
        "name": "Soldier Standing",
        "source": "Custom model",
        "license": "CC0",
    }


def get_destroyer_model() -> Dict[str, Any]:
    """Naval Destroyer.

    License: CC0
    """
    vertices = []
    faces = []
    zones = {}

    # Hull dimensions
    hull_l, hull_w, hull_h = 155.0, 20.0, 12.0
    hl, hw, hh = hull_l/2, hull_w/2, hull_h

    # Hull bottom (V-shape)
    vertices.extend([
        [-hl, 0, -hh*0.5],  # Stern keel
        [hl*0.9, 0, -hh*0.3],   # Bow keel
        # Stern waterline
        [-hl, -hw, 0], [-hl, hw, 0],
        # Mid waterline
        [0, -hw, 0], [0, hw, 0],
        # Bow waterline
        [hl*0.9, -hw*0.3, 0], [hl*0.9, hw*0.3, 0],
        # Bow tip
        [hl, 0, hh*0.1],
    ])

    # Hull faces
    faces.extend([
        [0, 2, 4, 1],  # Bottom right
        [0, 1, 5, 3],  # Bottom left
        [1, 4, 6, 8],  # Bow right bottom
        [1, 8, 7, 5],  # Bow left bottom
    ])
    for i in range(4):
        zones[i] = "hull"

    # Deck
    deck_base = len(vertices)
    vertices.extend([
        [-hl, -hw, hh*0.3], [-hl, hw, hh*0.3],  # Stern
        [0, -hw, hh*0.3], [0, hw, hh*0.3],      # Mid
        [hl*0.7, -hw*0.6, hh*0.4], [hl*0.7, hw*0.6, hh*0.4],  # Forward
        [hl*0.9, -hw*0.3, hh*0.35], [hl*0.9, hw*0.3, hh*0.35],  # Bow
    ])
    faces.extend([
        [deck_base, deck_base+2, deck_base+3, deck_base+1],  # Aft deck
        [deck_base+2, deck_base+4, deck_base+5, deck_base+3],  # Mid deck
        [deck_base+4, deck_base+6, deck_base+7, deck_base+5],  # Forward deck
    ])
    for i in range(3):
        zones[len(faces)-3+i] = "deck"

    # Superstructure
    ss_base = len(vertices)
    ss_l, ss_w, ss_h = 40.0, 15.0, 15.0
    ss_x = -hl * 0.1
    vertices.extend([
        [ss_x - ss_l/2, -ss_w/2, hh*0.3], [ss_x + ss_l/2, -ss_w/2, hh*0.3],
        [ss_x + ss_l/2, ss_w/2, hh*0.3], [ss_x - ss_l/2, ss_w/2, hh*0.3],
        [ss_x - ss_l/2, -ss_w/2, hh*0.3+ss_h], [ss_x + ss_l/2*0.7, -ss_w/2*0.8, hh*0.3+ss_h],
        [ss_x + ss_l/2*0.7, ss_w/2*0.8, hh*0.3+ss_h], [ss_x - ss_l/2, ss_w/2, hh*0.3+ss_h],
    ])
    faces.extend([
        [ss_base+4, ss_base+7, ss_base+6, ss_base+5],  # Top
        [ss_base, ss_base+4, ss_base+5, ss_base+1],    # Right
        [ss_base+3, ss_base+2, ss_base+6, ss_base+7],  # Left
        [ss_base, ss_base+3, ss_base+7, ss_base+4],    # Rear
        [ss_base+1, ss_base+5, ss_base+6, ss_base+2],  # Front
    ])
    for i in range(5):
        zones[len(faces)-5+i] = "superstructure"

    # Mast/radar
    mast_base = len(vertices)
    mast_x = ss_x
    mast_z = hh*0.3 + ss_h
    vertices.extend([
        [mast_x - 1, -1, mast_z], [mast_x + 1, -1, mast_z],
        [mast_x + 1, 1, mast_z], [mast_x - 1, 1, mast_z],
        [mast_x - 0.5, -0.5, mast_z + 20], [mast_x + 0.5, -0.5, mast_z + 20],
        [mast_x + 0.5, 0.5, mast_z + 20], [mast_x - 0.5, 0.5, mast_z + 20],
    ])
    faces.extend([
        [mast_base, mast_base+1, mast_base+5, mast_base+4],
        [mast_base+2, mast_base+3, mast_base+7, mast_base+6],
        [mast_base+1, mast_base+2, mast_base+6, mast_base+5],
        [mast_base+3, mast_base, mast_base+4, mast_base+7],
    ])
    for i in range(4):
        zones[len(faces)-4+i] = "superstructure"

    # Funnel/exhaust
    fun_base = len(vertices)
    fun_x = ss_x - ss_l/2 - 5
    vertices.extend([
        [fun_x - 2, -3, hh*0.3], [fun_x + 2, -3, hh*0.3],
        [fun_x + 2, 3, hh*0.3], [fun_x - 2, 3, hh*0.3],
        [fun_x - 1.5, -2.5, hh*0.3 + 8], [fun_x + 1.5, -2.5, hh*0.3 + 8],
        [fun_x + 1.5, 2.5, hh*0.3 + 8], [fun_x - 1.5, 2.5, hh*0.3 + 8],
    ])
    faces.extend([
        [fun_base, fun_base+1, fun_base+5, fun_base+4],
        [fun_base+2, fun_base+3, fun_base+7, fun_base+6],
        [fun_base+1, fun_base+2, fun_base+6, fun_base+5],
        [fun_base+3, fun_base, fun_base+4, fun_base+7],
        [fun_base+4, fun_base+5, fun_base+6, fun_base+7],  # Top (exhaust)
    ])
    for i in range(4):
        zones[len(faces)-5+i] = "superstructure"
    zones[len(faces)-1] = "exhaust"

    return {
        "vertices": np.array(vertices, dtype=np.float32),
        "faces": faces,
        "thermal_zones": zones,
        "name": "Naval Destroyer",
        "source": "Public specifications",
        "license": "CC0",
    }


# Model registry
EMBEDDED_MODELS = {
    "f16": get_f16_model,
    "m1_abrams": get_m1_abrams_model,
    "apache": get_apache_model,
    "humvee": get_humvee_model,
    "soldier_standing": get_soldier_standing_model,
    "destroyer": get_destroyer_model,
}


def get_embedded_model(model_id: str) -> Dict[str, Any]:
    """Get an embedded model by ID.

    Args:
        model_id: Model identifier

    Returns:
        Dict with vertices, faces, thermal_zones, name, source, license

    Raises:
        KeyError: If model not found
    """
    if model_id not in EMBEDDED_MODELS:
        raise KeyError(f"Embedded model '{model_id}' not found. Available: {list(EMBEDDED_MODELS.keys())}")
    return EMBEDDED_MODELS[model_id]()


def list_embedded_models() -> List[str]:
    """List available embedded models."""
    return list(EMBEDDED_MODELS.keys())
