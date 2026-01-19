"""
Lumina Studio - Calibration Generator Module
校准板生成模块
"""

import os
import tempfile
from typing import Optional

import numpy as np
import trimesh
from PIL import Image

from config import PrinterConfig, ColorSystem
from utils import Stats, safe_fix_3mf_names


def _generate_voxel_mesh(voxel_matrix: np.ndarray, material_index: int,
                          grid_h: int, grid_w: int) -> Optional[trimesh.Trimesh]:
    """Generate mesh for a specific material from voxel data."""
    scale_x = PrinterConfig.NOZZLE_WIDTH
    scale_y = PrinterConfig.NOZZLE_WIDTH
    scale_z = PrinterConfig.LAYER_HEIGHT
    shrink = PrinterConfig.SHRINK_OFFSET

    vertices, faces = [], []
    total_z_layers = voxel_matrix.shape[0]

    for z in range(total_z_layers):
        z_bottom, z_top = z * scale_z, (z + 1) * scale_z
        layer_mask = (voxel_matrix[z] == material_index)
        if not np.any(layer_mask):
            continue

        for y in range(grid_h):
            world_y = y * scale_y
            row = layer_mask[y]
            padded_row = np.pad(row, (1, 1), mode='constant')
            diff = np.diff(padded_row.astype(int))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                x0, x1 = start * scale_x + shrink, end * scale_x - shrink
                y0, y1 = world_y + shrink, world_y + scale_y - shrink

                base_idx = len(vertices)
                vertices.extend([
                    [x0, y0, z_bottom], [x1, y0, z_bottom], [x1, y1, z_bottom], [x0, y1, z_bottom],
                    [x0, y0, z_top], [x1, y0, z_top], [x1, y1, z_top], [x0, y1, z_top]
                ])
                cube_faces = [
                    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]
                ]
                faces.extend([[v + base_idx for v in f] for f in cube_faces])

    if not vertices:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    return mesh


def generate_calibration_board(color_mode: str, block_size_mm: float,
                                gap_mm: float, backing_color: str):
    """Generate a 1024-color calibration board as 3MF."""

    color_conf = ColorSystem.get(color_mode)
    slot_names = color_conf['slots']
    preview_colors = color_conf['preview']
    color_map = color_conf['map']

    backing_id = color_map.get(backing_color, 0)

    # Grid setup
    grid_dim, padding = 32, 1
    total_w = total_h = grid_dim + (padding * 2)

    pixels_per_block = max(1, int(block_size_mm / PrinterConfig.NOZZLE_WIDTH))
    pixels_gap = max(1, int(gap_mm / PrinterConfig.NOZZLE_WIDTH))

    voxel_w = total_w * (pixels_per_block + pixels_gap)
    voxel_h = total_h * (pixels_per_block + pixels_gap)

    backing_layers = int(PrinterConfig.BACKING_MM / PrinterConfig.LAYER_HEIGHT)
    total_layers = PrinterConfig.COLOR_LAYERS + backing_layers

    full_matrix = np.full((total_layers, voxel_h, voxel_w), backing_id, dtype=int)

    # Generate 1024 permutations
    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        stack = digits[::-1]

        row = (i // grid_dim) + padding
        col = (i % grid_dim) + padding
        px = col * (pixels_per_block + pixels_gap)
        py = row * (pixels_per_block + pixels_gap)

        for z in range(PrinterConfig.COLOR_LAYERS):
            full_matrix[z, py:py+pixels_per_block, px:px+pixels_per_block] = stack[z]

    # Corner markers - 根据模式设置不同的角点颜色
    # 角点位置: (row, col, mat_id)
    # row=0是顶部, row=total_h-1是底部
    # col=0是左边, col=total_w-1是右边
    if "RYBW" in color_mode:
        # RYBW: slots = [White(0), Red(1), Yellow(2), Blue(3)]
        # corner_labels: TL=White, TR=Red, BR=Blue, BL=Yellow
        corners = [
            (0, 0, 0),              # TL = White
            (0, total_w-1, 1),      # TR = Red
            (total_h-1, total_w-1, 3),  # BR = Blue
            (total_h-1, 0, 2)       # BL = Yellow
        ]
    else:
        # CMYW: slots = [White(0), Cyan(1), Magenta(2), Yellow(3)]
        # corner_labels: TL=White, TR=Cyan, BR=Magenta, BL=Yellow
        corners = [
            (0, 0, 0),              # TL = White
            (0, total_w-1, 1),      # TR = Cyan
            (total_h-1, total_w-1, 2),  # BR = Magenta
            (total_h-1, 0, 3)       # BL = Yellow
        ]

    for r, c, mat_id in corners:
        px = c * (pixels_per_block + pixels_gap)
        py = r * (pixels_per_block + pixels_gap)
        for z in range(PrinterConfig.COLOR_LAYERS):
            full_matrix[z, py:py+pixels_per_block, px:px+pixels_per_block] = mat_id

    # Build 3MF
    scene = trimesh.Scene()
    for mat_id in range(4):
        mesh = _generate_voxel_mesh(full_matrix, mat_id, voxel_h, voxel_w)
        if mesh:
            mesh.visual.face_colors = preview_colors[mat_id]
            name = slot_names[mat_id]
            # Set multiple name attributes to increase compatibility
            mesh.metadata['name'] = name
            scene.add_geometry(mesh, node_name=name, geom_name=name)

    # Export
    mode_tag = color_conf['name']
    output_path = os.path.join(tempfile.gettempdir(), f"Lumina_Calibration_{mode_tag}.3mf")
    scene.export(output_path)

    # Fix object names in 3MF for better slicer compatibility
    safe_fix_3mf_names(output_path, slot_names)

    # Preview
    bottom_layer = full_matrix[0].astype(np.uint8)
    preview_arr = np.zeros((voxel_h, voxel_w, 3), dtype=np.uint8)
    for mat_id, rgba in preview_colors.items():
        preview_arr[bottom_layer == mat_id] = rgba[:3]

    Stats.increment("calibrations")

    return output_path, Image.fromarray(preview_arr), f"✅ 校准板已生成！已组合为一个对象 | 颜色: {', '.join(slot_names)}"
