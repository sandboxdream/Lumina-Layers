"""
Lumina Studio - Image Converter Module (Vector Mode Upgrade + Woodblock Mode)
图像转换模块 - 矢量模式升级版 + 版画模式
"""

import os
import tempfile
import numpy as np
import trimesh
import cv2
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
from scipy.spatial import KDTree
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# ========== NEW: Woodblock Mode Imports ==========
try:
    from skimage import color, segmentation
    from skimage.measure import regionprops

    # --- 修复代码开始：兼容新旧版本的 scikit-image ---
    try:
        # 新版本 (0.19+) 的路径
        from skimage import graph
    except ImportError:
        # 旧版本的路径
        from skimage.future import graph
    # --- 修复代码结束 ---

    WOODBLOCK_AVAILABLE = True
except Exception as e:
    WOODBLOCK_AVAILABLE = False
    print(f"\n[DEBUG] scikit-image 导入失败: {e}")
    # import traceback
    # traceback.print_exc()
    print("[WARNING] scikit-image not available. Woodblock mode disabled.\n")
# ========== END NEW IMPORTS ==========

from config import (
    PrinterConfig,
    ColorSystem,
    PREVIEW_SCALE,
    PREVIEW_MARGIN
)
from utils import Stats, safe_fix_3mf_names


def create_keychain_loop(width_mm, length_mm, hole_dia_mm, thickness_mm, attach_x_mm, attach_y_mm):

    print(f"[DEBUG] create_keychain_loop called: width={width_mm}, length={length_mm}, hole={hole_dia_mm}, thick={thickness_mm}, x={attach_x_mm}, y={attach_y_mm}")

    half_w = width_mm / 2
    circle_radius = half_w
    hole_radius = min(hole_dia_mm / 2, circle_radius * 0.8)

    rect_height = max(0.2, length_mm - circle_radius)

    circle_center_y = rect_height

    n_arc = 32
    outer_pts = []


    outer_pts.append((-half_w, 0))

    outer_pts.append((half_w, 0))

    outer_pts.append((half_w, rect_height))


    for i in range(1, n_arc):
        angle = np.pi * i / n_arc
        x = circle_radius * np.cos(angle)
        y = circle_center_y + circle_radius * np.sin(angle)
        outer_pts.append((x, y))

    # 左边
    outer_pts.append((-half_w, rect_height))

    outer_pts = np.array(outer_pts)
    n_outer = len(outer_pts)


    n_hole = 32
    hole_pts = []
    for i in range(n_hole):
        angle = 2 * np.pi * i / n_hole
        x = hole_radius * np.cos(angle)
        y = circle_center_y + hole_radius * np.sin(angle)
        hole_pts.append((x, y))
    hole_pts = np.array(hole_pts)
    n_hole_pts = len(hole_pts)

    outer_center = outer_pts.mean(axis=0)
    hole_center = np.array([0, circle_center_y])

    vertices = []
    faces = []

    for pt in outer_pts:
        vertices.append([pt[0], pt[1], 0])

    for pt in hole_pts:
        vertices.append([pt[0], pt[1], 0])

    for pt in outer_pts:
        vertices.append([pt[0], pt[1], thickness_mm])

    for pt in hole_pts:
        vertices.append([pt[0], pt[1], thickness_mm])

    bottom_outer_start = 0
    bottom_hole_start = n_outer
    top_outer_start = n_outer + n_hole_pts
    top_hole_start = n_outer + n_hole_pts + n_outer

    for i in range(n_outer):
        i_next = (i + 1) % n_outer
        bi = bottom_outer_start + i
        bi_next = bottom_outer_start + i_next
        ti = top_outer_start + i
        ti_next = top_outer_start + i_next
        faces.append([bi, bi_next, ti_next])
        faces.append([bi, ti_next, ti])

    for i in range(n_hole_pts):
        i_next = (i + 1) % n_hole_pts
        bi = bottom_hole_start + i
        bi_next = bottom_hole_start + i_next
        ti = top_hole_start + i
        ti_next = top_hole_start + i_next
        faces.append([bi, ti, ti_next])
        faces.append([bi, ti_next, bi_next])

    def connect_rings(outer_indices, hole_indices, vertices_arr, is_top=True):

        ring_faces = []
        n_o = len(outer_indices)
        n_h = len(hole_indices)


        oi = 0
        hi = 0

        def get_2d(idx):
            return np.array([vertices_arr[idx][0], vertices_arr[idx][1]])

        total_steps = n_o + n_h
        for _ in range(total_steps):
            o_curr = outer_indices[oi % n_o]
            o_next = outer_indices[(oi + 1) % n_o]
            h_curr = hole_indices[hi % n_h]
            h_next = hole_indices[(hi + 1) % n_h]

            dist_o = np.linalg.norm(get_2d(o_next) - get_2d(h_curr))
            dist_h = np.linalg.norm(get_2d(o_curr) - get_2d(h_next))

            if oi >= n_o:

                if is_top:
                    ring_faces.append([o_curr, h_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, h_next])
                hi += 1
            elif hi >= n_h:

                if is_top:
                    ring_faces.append([o_curr, o_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, o_next])
                oi += 1
            elif dist_o < dist_h:

                if is_top:
                    ring_faces.append([o_curr, o_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, o_next])
                oi += 1
            else:

                if is_top:
                    ring_faces.append([o_curr, h_next, h_curr])
                else:
                    ring_faces.append([o_curr, h_curr, h_next])
                hi += 1

        return ring_faces

    vertices_arr = np.array(vertices)

    bottom_outer_idx = list(range(bottom_outer_start, bottom_outer_start + n_outer))
    bottom_hole_idx = list(range(bottom_hole_start, bottom_hole_start + n_hole_pts))
    bottom_faces = connect_rings(bottom_outer_idx, bottom_hole_idx, vertices_arr, is_top=False)
    faces.extend(bottom_faces)

    top_outer_idx = list(range(top_outer_start, top_outer_start + n_outer))
    top_hole_idx = list(range(top_hole_start, top_hole_start + n_hole_pts))
    top_faces = connect_rings(top_outer_idx, top_hole_idx, vertices_arr, is_top=True)
    faces.extend(top_faces)

    vertices_arr = np.array(vertices)
    vertices_arr[:, 0] += attach_x_mm
    vertices_arr[:, 1] += attach_y_mm

    mesh = trimesh.Trimesh(vertices=vertices_arr, faces=np.array(faces))
    mesh.fix_normals()

    print(f"[DEBUG] Mesh created: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")

    return mesh


def load_calibrated_lut(npy_path):
    """Load and validate LUT file."""
    try:
        lut_grid = np.load(npy_path)
        measured_colors = lut_grid.reshape(-1, 3)
    except:
        return None, None, "❌ LUT文件损坏"

    valid_rgb, valid_stacks = [], []
    base_blue = np.array([30, 100, 200])
    dropped = 0

    for i in range(1024):
        digits = []
        temp = i
        for _ in range(5):
            digits.append(temp % 4)
            temp //= 4
        stack = digits[::-1]

        real_rgb = measured_colors[i]
        dist = np.linalg.norm(real_rgb - base_blue)

        if dist < 60 and 3 not in stack:
            dropped += 1
            continue

        valid_rgb.append(real_rgb)
        valid_stacks.append(stack)

    return np.array(valid_rgb), np.array(valid_stacks), f"✅ LUT已加载 (过滤了{dropped}个异常点)"


def create_slab_mesh(voxel_matrix, mat_id, height):
    """Generate optimized mesh from voxel data (Legacy Pixel Mode)."""
    vertices, faces = [], []
    shrink = 0.05

    for z in range(voxel_matrix.shape[0]):
        z_bottom, z_top = z, z + 1
        mask = (voxel_matrix[z] == mat_id)
        if not np.any(mask):
            continue

        for y in range(height):
            world_y = (height - 1 - y)
            row = mask[y]
            padded = np.pad(row, (1, 1), mode='constant')
            diff = np.diff(padded.astype(int))
            starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

            for start, end in zip(starts, ends):
                x0, x1 = start + shrink, end - shrink
                y0, y1 = world_y + shrink, world_y + 1 - shrink

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


def create_vector_mesh(voxel_matrix, mat_id, height):

    layer_groups = []
    prev_mask = None
    start_z = 0

    for z in range(voxel_matrix.shape[0]):
        curr_mask = (voxel_matrix[z] == mat_id)

        if not np.any(curr_mask):
            if prev_mask is not None and np.any(prev_mask):
                layer_groups.append((start_z, z - 1, prev_mask))
                prev_mask = None
            continue

        if prev_mask is None:
            start_z = z
            prev_mask = curr_mask.copy()
        elif np.array_equal(curr_mask, prev_mask):
            pass
        else:
            layer_groups.append((start_z, z - 1, prev_mask))
            start_z = z
            prev_mask = curr_mask.copy()

    if prev_mask is not None and np.any(prev_mask):
        layer_groups.append((start_z, voxel_matrix.shape[0] - 1, prev_mask))

    print(f"[VECTOR] Mat ID {mat_id}: Merged {voxel_matrix.shape[0]} layers into {len(layer_groups)} groups")


    all_meshes = []

    for start_z, end_z, mask in layer_groups:
        num_layers = end_z - start_z + 1
        z_height = float(num_layers)
        print(f"[VECTOR] Processing group z={start_z}-{end_z} (height={z_height})")
        mask_uint8 = (mask.astype(np.uint8) * 255)
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

        print(f"[VECTOR] Applied minimal morphological cleanup (kernel={kernel_size}x{kernel_size}) for high-fidelity")
        contours, hierarchy = cv2.findContours(
            mask_uint8,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        print(f"[VECTOR] Found {len(contours)} contours")

        polygons = []

        for idx, contour in enumerate(contours):

            contour_area = cv2.contourArea(contour)
            if contour_area < 4.0:
                continue


            epsilon = 0.1
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            points_2d = []
            for point in approx[:, 0, :]:
                x, y = point
                world_y = (height - 1 - y)
                points_2d.append([float(x), float(world_y)])

            try:
                poly = Polygon(points_2d)

                if not poly.is_valid:
                    poly = poly.buffer(0)

                if poly.is_valid and poly.area > 0.01:
                    poly = poly.buffer(6.0)

                    is_hole = False
                    if hierarchy is not None and hierarchy[0][idx][3] != -1:
                        is_hole = True

                    polygons.append((poly, is_hole))

            except Exception as e:
                print(f"[VECTOR] Warning: Failed to create polygon: {e}")
                continue

        outer_polys = [p for p, is_hole in polygons if not is_hole]
        hole_polys = [p for p, is_hole in polygons if is_hole]

        if len(outer_polys) == 0:
            continue

        if len(outer_polys) > 1:
            merged = unary_union(outer_polys)
        else:
            merged = outer_polys[0]

        for hole in hole_polys:
            merged = merged.difference(hole)

        final_polygons = []
        if isinstance(merged, Polygon):
            final_polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            final_polygons = list(merged.geoms)

        for poly in final_polygons:
            if poly.area < 0.01:
                continue

            try:
                mesh = trimesh.creation.extrude_polygon(
                    poly,
                    height=z_height
                )

                mesh.apply_translation([0, 0, start_z])

                all_meshes.append(mesh)

            except Exception as e:
                print(f"[VECTOR] Warning: Failed to extrude polygon: {e}")
                continue

    if not all_meshes:
        return None

    combined = trimesh.util.concatenate(all_meshes)
    combined.process()

    print(f"[VECTOR] Mat ID {mat_id}: Final mesh has {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    return combined


def create_woodblock_mesh(voxel_matrix, mat_id, height, original_image_lab=None,
                         lut_kdtree=None, lut_rgb=None):

    if not WOODBLOCK_AVAILABLE:
        print("[WOODBLOCK] Fallback to vector mode (scikit-image not available)")
        return create_vector_mesh(voxel_matrix, mat_id, height)

    print(f"[WOODBLOCK] Processing material ID {mat_id}...")

    layer_groups = []
    prev_mask = None
    start_z = 0

    for z in range(voxel_matrix.shape[0]):
        curr_mask = (voxel_matrix[z] == mat_id)

        if not np.any(curr_mask):
            if prev_mask is not None and np.any(prev_mask):
                layer_groups.append((start_z, z - 1, prev_mask))
                prev_mask = None
            continue

        if prev_mask is None:
            start_z = z
            prev_mask = curr_mask.copy()
        elif np.array_equal(curr_mask, prev_mask):
            pass  # 继续当前组
        else:
            layer_groups.append((start_z, z - 1, prev_mask))
            start_z = z
            prev_mask = curr_mask.copy()

    if prev_mask is not None and np.any(prev_mask):
        layer_groups.append((start_z, voxel_matrix.shape[0] - 1, prev_mask))

    print(f"[WOODBLOCK] Mat {mat_id}: Merged {voxel_matrix.shape[0]} layers → {len(layer_groups)} groups")

    # 处理每个层组
    all_meshes = []

    for group_idx, (start_z, end_z, mask) in enumerate(layer_groups):
        z_height = float(end_z - start_z + 1)

        print(f"[WOODBLOCK] Group {group_idx+1}/{len(layer_groups)}: z={start_z}-{end_z}, height={z_height}")

        # 转换为 uint8
        mask_uint8 = (mask.astype(np.uint8) * 255)

        #STEP 3: 边缘保护去噪
        kernel_size = 3  # 最小核，保护细节
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

        #STEP 4: 轮廓提取
        contours, hierarchy = cv2.findContours(
            mask_uint8,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        print(f"[WOODBLOCK] Found {len(contours)} raw contours")

        #STEP 5: 智能轮廓处理与几何修复
        polygons = []
        nozzle_width = 0.4
        min_feature_px = 4.0

        for idx, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)

            if contour_area < min_feature_px:
                continue

            epsilon = 0.1
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            points_2d = []
            for point in approx[:, 0, :]:
                x, y = point
                world_y = (height - 1 - y)
                points_2d.append([float(x), float(world_y)])

            try:
                poly = Polygon(points_2d)

                if not poly.is_valid:
                    poly = poly.buffer(0)

                if not poly.is_valid or poly.area < 0.01:
                    continue

                test_shrink = poly.buffer(-min_feature_px / 2.0)

                if test_shrink.is_empty or test_shrink.area < 0.01:

                    rescue_distance = min_feature_px / 2.0 + 0.5  # +0.5px 附着冗余

                    poly = poly.buffer(
                        distance=rescue_distance,
                        join_style=2,  # JOIN_STYLE.mitre
                        mitre_limit=5.0  # 防止过长尖刺
                    )
                    print(f"[WOODBLOCK] Rescued thin feature: area={contour_area:.1f}px²")

                else:

                    poly = poly.buffer(
                        distance=0.5,
                        join_style=2,  # Mitre保持尖角
                        mitre_limit=5.0
                    )

                # ========== 孔洞处理 ==========
                is_hole = False
                if hierarchy is not None and hierarchy[0][idx][3] != -1:
                    is_hole = True

                polygons.append((poly, is_hole))

            except Exception as e:
                print(f"[WOODBLOCK] Warning: Polygon creation failed: {e}")
                continue

        #STEP 6: 布尔运算合并
        outer_polys = [p for p, is_hole in polygons if not is_hole]
        hole_polys = [p for p, is_hole in polygons if is_hole]

        if len(outer_polys) == 0:
            continue

        # 合并外轮廓
        if len(outer_polys) > 1:
            merged = unary_union(outer_polys)
        else:
            merged = outer_polys[0]

        for hole in hole_polys:
            merged = merged.difference(hole)

        final_polygons = []
        if isinstance(merged, Polygon):
            final_polygons = [merged]
        elif isinstance(merged, MultiPolygon):
            final_polygons = list(merged.geoms)

        #STEP 7: 挤出为3D网格
        for poly in final_polygons:
            if poly.area < 0.01:
                continue

            try:
                mesh = trimesh.creation.extrude_polygon(poly, height=z_height)
                mesh.apply_translation([0, 0, start_z])
                all_meshes.append(mesh)

            except Exception as e:
                print(f"[WOODBLOCK] Warning: Extrusion failed: {e}")
                continue

    if not all_meshes:
        return None

    #STEP 8: 合并与清理
    combined = trimesh.util.concatenate(all_meshes)
    combined.process()

    print(f"[WOODBLOCK] Mat {mat_id}: Final mesh - {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    return combined


def create_preview_mesh(matched_rgb, mask_solid, total_layers):

    height, width = matched_rgb.shape[:2]
    total_pixels = width * height

    DISABLE_THRESHOLD = 2_000_000
    SIMPLIFY_THRESHOLD = 500_000
    TARGET_PIXELS = 300_000

    if total_pixels > DISABLE_THRESHOLD:
        print(f"[PREVIEW] Model too large ({total_pixels:,} pixels = {total_pixels*12:,} triangles)")
        print(f"[PREVIEW] Browser WebGL limit is typically 2-3M triangles")
        print(f"[PREVIEW] ⚠️ 3D preview disabled to prevent crash")
        print(f"[PREVIEW] ✅ 3MF file is ready - view it in your slicer!")
        return None

    if total_pixels > SIMPLIFY_THRESHOLD:
        scale_factor = int(np.sqrt(total_pixels / TARGET_PIXELS))
        scale_factor = max(2, min(scale_factor, 16))  # Clamp to 2-16x

        print(f"[PREVIEW] Large model detected ({width}×{height} = {total_pixels:,} pixels)")
        print(f"[PREVIEW] Downsampling preview by {scale_factor}× for browser compatibility")
        print(f"[PREVIEW] Final preview: ~{total_pixels//(scale_factor**2):,} pixels")
        print(f"[PREVIEW] ℹ️ Note: 3MF output retains full {total_pixels:,} pixel quality!")

        new_height = height // scale_factor
        new_width = width // scale_factor

        matched_rgb_small = cv2.resize(
            matched_rgb,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        mask_solid_small = cv2.resize(
            mask_solid.astype(np.uint8),
            (new_width, new_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        matched_rgb = matched_rgb_small
        mask_solid = mask_solid_small
        height, width = new_height, new_width

        shrink = 0.05 * scale_factor
    else:
        shrink = 0.05

    vertices = []
    faces = []
    face_colors = []

    for y in range(height):
        for x in range(width):
            if not mask_solid[y, x]:
                continue

            rgb = matched_rgb[y, x]
            rgba = [int(rgb[0]), int(rgb[1]), int(rgb[2]), 255]

            world_y = (height - 1 - y)
            x0, x1 = x + shrink, x + 1 - shrink
            y0, y1 = world_y + shrink, world_y + 1 - shrink
            z0, z1 = 0, total_layers

            base_idx = len(vertices)
            vertices.extend([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
            ])

            cube_faces = [
                [0, 2, 1], [0, 3, 2],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],  # front
                [1, 2, 6], [1, 6, 5],  # right
                [2, 3, 7], [2, 7, 6],  # back
                [3, 0, 4], [3, 4, 7]   # left
            ]

            for f in cube_faces:
                faces.append([v + base_idx for v in f])
                face_colors.append(rgba)

    if not vertices:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.visual.face_colors = np.array(face_colors, dtype=np.uint8)

    print(f"[PREVIEW] Generated: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    return mesh


def convert_image_to_3d(image_path, lut_path, target_width_mm, spacer_thick,
                         structure_mode, auto_bg, bg_tol, color_mode,
                         add_loop, loop_width, loop_length, loop_hole, loop_pos,
                         modeling_mode="vector", quantize_colors=32):

    if image_path is None:
        return None, None, None, "❌ 请上传图片"
    if lut_path is None:
        return None, None, None, "⚠️ 请上传 .npy 校准文件！"

    # ========== Normalize modeling mode input ==========
    mode_str = str(modeling_mode).lower()
    use_vector_mode = "vector" in mode_str or "矢量" in mode_str
    use_woodblock_mode = "woodblock" in mode_str or "版画" in mode_str

    if use_woodblock_mode:
        mode_name_zh = "版画细节"
        mode_name_en = "Woodblock"
        if not WOODBLOCK_AVAILABLE:
            print("[WARNING] Woodblock mode unavailable, falling back to Vector mode")
            use_woodblock_mode = False
            use_vector_mode = True
            mode_name_zh = "矢量平滑"
            mode_name_en = "Vector"
    elif use_vector_mode:
        mode_name_zh = "矢量平滑"
        mode_name_en = "Vector"
    else:
        mode_name_zh = "像素方块"
        mode_name_en = "Voxel"

    print(f"[INFO] Modeling mode detected: {mode_name_en} (from input: {modeling_mode})")

    color_conf = ColorSystem.get(color_mode)

    # Load LUT
    lut_rgb, ref_stacks, msg = load_calibrated_lut(lut_path.name)
    if lut_rgb is None:
        return None, None, None, msg
    tree = KDTree(lut_rgb)

    print(f"[QUANT] Loading and resizing image...")
    img = Image.open(image_path).convert('RGBA')


    if use_vector_mode or use_woodblock_mode:

        PIXELS_PER_MM = 10
        target_w = int(target_width_mm * PIXELS_PER_MM)
        pixel_to_mm_scale = 1.0 / PIXELS_PER_MM  # 0.1 mm per pixel

        if use_woodblock_mode:
            print(f"[WOODBLOCK] High-res mode: {PIXELS_PER_MM} pixels/mm, scale={pixel_to_mm_scale}mm/px")
        else:
            print(f"[VECTOR] High-res mode: {PIXELS_PER_MM} pixels/mm, scale={pixel_to_mm_scale}mm/px")
    else:

        target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
        pixel_to_mm_scale = PrinterConfig.NOZZLE_WIDTH
        print(f"[VOXEL] Low-res mode: {1.0/pixel_to_mm_scale:.2f} pixels/mm, scale={pixel_to_mm_scale}mm/px")

    target_h = int(target_w * img.height / img.width)

    print(f"[INFO] Target resolution: {target_w}×{target_h}px ({target_w*pixel_to_mm_scale:.1f}×{target_h*pixel_to_mm_scale:.1f}mm)")

    if use_vector_mode or use_woodblock_mode:
        # High-quality resampling for smooth modes
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:

        img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
        print(f"[VOXEL] Using NEAREST resampling to preserve pixel boundaries")

    img_arr = np.array(img)
    rgb_arr, alpha_arr = img_arr[:, :, :3], img_arr[:, :, 3]


    if use_vector_mode or use_woodblock_mode:

        print(f"[QUANT] Applying bilateral filter for edge-preserving denoising...")
        rgb_denoised = cv2.bilateralFilter(rgb_arr.astype(np.uint8), d=5, sigmaColor=50, sigmaSpace=50)



        print(f"[DENOISE] Applying spatial denoising (median blur) BEFORE quantization...")

        rgb_denoised = cv2.medianBlur(rgb_denoised, 7)

        print(f"[DENOISE] Spatial denoising complete! Ready for color quantization")

        print(f"[QUANT] Quantizing image to {quantize_colors} dominant colors...")
        h, w = rgb_denoised.shape[:2]
        pixels = rgb_denoised.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        flags = cv2.KMEANS_PP_CENTERS

        _, labels, centers = cv2.kmeans(pixels, quantize_colors, None, criteria, 10, flags)

        centers = centers.astype(np.uint8)
        quantized_pixels = centers[labels.flatten()]
        quantized_image = quantized_pixels.reshape(h, w, 3)

        print(f"[QUANT] Quantization complete! Image now has {quantize_colors} distinct colors (was ~{h*w} unique pixels)")

        print(f"[QUANT] Finding unique colors from quantized image...")

        unique_colors = np.unique(quantized_image.reshape(-1, 3), axis=0)
        print(f"[QUANT] Found {len(unique_colors)} unique colors to match")

        print(f"[QUANT] Matching {len(unique_colors)} colors to LUT...")
        _, unique_indices = tree.query(unique_colors.astype(float))

        color_to_stack = {}
        color_to_rgb = {}
        for i, color in enumerate(unique_colors):
            color_key = tuple(color)
            color_to_stack[color_key] = ref_stacks[unique_indices[i]]
            color_to_rgb[color_key] = lut_rgb[unique_indices[i]]

        print(f"[QUANT] Mapping results back to full image...")
        matched_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        best_stacks = np.zeros((target_h, target_w, PrinterConfig.COLOR_LAYERS), dtype=int)

        for y in range(target_h):
            for x in range(target_w):
                color_key = tuple(quantized_image[y, x])
                matched_rgb[y, x] = color_to_rgb[color_key]
                best_stacks[y, x] = color_to_stack[color_key]

        print(f"[QUANT] Color matching complete!")

        bg_reference = quantized_image

    else:

        print(f"[VOXEL] Using direct pixel-level color matching (no smoothing)")

        flat_rgb = rgb_arr.reshape(-1, 3)
        _, indices = tree.query(flat_rgb)

        matched_rgb = lut_rgb[indices].reshape(target_h, target_w, 3)
        best_stacks = ref_stacks[indices].reshape(target_h, target_w, PrinterConfig.COLOR_LAYERS)

        print(f"[VOXEL] Direct matching complete!")

        bg_reference = rgb_arr

    mask_transparent = alpha_arr < 10
    if auto_bg:
        bg_color = bg_reference[0, 0]
        diff = np.sum(np.abs(bg_reference - bg_color), axis=-1)
        mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)

    best_stacks[mask_transparent] = -1

    preview_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    mask_solid = ~mask_transparent
    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
    preview_rgba[mask_solid, 3] = 255

    loop_info = None
    loop_color_id = 0

    print(f"[DEBUG] add_loop={add_loop}, loop_pos={loop_pos}, loop_width={loop_width}, loop_length={loop_length}, loop_hole={loop_hole}")

    if add_loop:
        solid_rows = np.any(mask_solid, axis=1)
        if np.any(solid_rows):
            if loop_pos is not None and len(loop_pos) == 2:
                click_x, click_y = loop_pos
                attach_col = int(click_x)
                attach_row = int(click_y)
                attach_col = max(0, min(target_w - 1, attach_col))
                attach_row = max(0, min(target_h - 1, attach_row))

                col_mask = mask_solid[:, attach_col]
                if np.any(col_mask):
                    solid_rows_in_col = np.where(col_mask)[0]
                    distances = np.abs(solid_rows_in_col - attach_row)
                    nearest_idx = np.argmin(distances)
                    top_row = solid_rows_in_col[nearest_idx]
                else:
                    top_row = np.argmax(solid_rows)
                    solid_cols_in_top = np.where(mask_solid[top_row])[0]
                    if len(solid_cols_in_top) > 0:
                        distances = np.abs(solid_cols_in_top - attach_col)
                        nearest_idx = np.argmin(distances)
                        attach_col = solid_cols_in_top[nearest_idx]
            else:
                top_row = np.argmax(solid_rows)
                solid_cols_in_top = np.where(mask_solid[top_row])[0]
                if len(solid_cols_in_top) > 0:
                    attach_col = int(np.mean(solid_cols_in_top))
                else:
                    attach_col = target_w // 2

            attach_col = max(0, min(target_w - 1, attach_col))

            search_area = best_stacks[max(0, top_row-2):top_row+3,
                                     max(0, attach_col-3):attach_col+4]
            search_area = search_area[search_area >= 0]
            if len(search_area) > 0:
                unique, counts = np.unique(search_area, return_counts=True)
                for mat_id in unique[np.argsort(-counts)]:
                    if mat_id != 0:
                        loop_color_id = int(mat_id)
                        break

            loop_info = {
                'attach_x_mm': attach_col * PrinterConfig.NOZZLE_WIDTH,
                'attach_y_mm': (target_h - 1 - top_row) * PrinterConfig.NOZZLE_WIDTH,
                'width_mm': loop_width,
                'length_mm': loop_length,
                'hole_dia_mm': loop_hole,
                'color_id': loop_color_id
            }

            preview_pil = Image.fromarray(preview_rgba, mode='RGBA')
            draw = ImageDraw.Draw(preview_pil)
            loop_color_rgba = tuple(color_conf['preview'][loop_color_id][:3]) + (255,)

            loop_w_px = int(loop_width / PrinterConfig.NOZZLE_WIDTH)
            loop_h_px = int(loop_length / PrinterConfig.NOZZLE_WIDTH)
            hole_r_px = int(loop_hole / 2 / PrinterConfig.NOZZLE_WIDTH)
            circle_r_px = loop_w_px // 2

            loop_bottom = top_row
            loop_top = top_row - loop_h_px
            loop_left = attach_col - loop_w_px // 2
            loop_right = attach_col + loop_w_px // 2

            rect_h_px = loop_h_px - circle_r_px
            rect_bottom = loop_bottom
            rect_top = loop_bottom - rect_h_px

            circle_center_y = rect_top
            circle_center_x = attach_col

            if rect_h_px > 0:
                draw.rectangle([loop_left, rect_top, loop_right, rect_bottom], fill=loop_color_rgba)

            draw.ellipse([circle_center_x - circle_r_px, circle_center_y - circle_r_px,
                          circle_center_x + circle_r_px, circle_center_y + circle_r_px],
                         fill=loop_color_rgba)

            hole_center_y = circle_center_y
            draw.ellipse([circle_center_x - hole_r_px, hole_center_y - hole_r_px,
                          circle_center_x + hole_r_px, hole_center_y + hole_r_px],
                         fill=(0, 0, 0, 0))

            preview_rgba = np.array(preview_pil)

    preview_img = Image.fromarray(preview_rgba, mode='RGBA')

    # Voxel construction
    bottom_voxels = np.transpose(best_stacks, (2, 0, 1))
    spacer_layers = max(1, int(round(spacer_thick / PrinterConfig.LAYER_HEIGHT)))

    if "双面" in structure_mode:
        top_voxels = np.transpose(best_stacks[..., ::-1], (2, 0, 1))
        total_layers = 5 + spacer_layers + 5
        full_matrix = np.full((total_layers, target_h, target_w), -1, dtype=int)
        full_matrix[0:5] = bottom_voxels

        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(5, 5 + spacer_layers):
            full_matrix[z] = spacer
        full_matrix[5 + spacer_layers:] = top_voxels
    else:
        total_layers = 5 + spacer_layers
        full_matrix = np.full((total_layers, target_h, target_w), -1, dtype=int)
        full_matrix[0:5] = bottom_voxels

        spacer = np.full((target_h, target_w), -1, dtype=int)
        spacer[~mask_transparent] = 0
        for z in range(5, total_layers):
            full_matrix[z] = spacer

    scene = trimesh.Scene()

    transform = np.eye(4)
    transform[0, 0] = pixel_to_mm_scale  # X: pixels to mm
    transform[1, 1] = pixel_to_mm_scale  # Y: pixels to mm
    transform[2, 2] = PrinterConfig.LAYER_HEIGHT  # Z: layers to mm

    print(f"[INFO] Transform scale: XY={pixel_to_mm_scale}mm/px, Z={PrinterConfig.LAYER_HEIGHT}mm/layer")

    preview_colors = color_conf['preview']
    slot_names = color_conf['slots']

    if use_woodblock_mode:
        mesh_creator = create_woodblock_mesh
    elif use_vector_mode:
        mesh_creator = create_vector_mesh
    else:
        mesh_creator = create_slab_mesh

    print(f"[INFO] Using {mode_name_en.upper()} mode for mesh generation")

    for mat_id in range(4):
        mesh = mesh_creator(full_matrix, mat_id, target_h)
        if mesh:
            mesh.apply_transform(transform)
            mesh.visual.face_colors = preview_colors[mat_id]
            mesh.metadata['name'] = slot_names[mat_id]
            scene.add_geometry(mesh, node_name=slot_names[mat_id], geom_name=slot_names[mat_id])

    loop_added = False
    print(f"[DEBUG] Before loop creation: add_loop={add_loop}, loop_info={loop_info}")
    if add_loop and loop_info is not None:
        try:
            loop_thickness = total_layers * PrinterConfig.LAYER_HEIGHT
            print(f"[DEBUG] Creating loop mesh with thickness={loop_thickness}")

            loop_mesh = create_keychain_loop(
                width_mm=loop_info['width_mm'],
                length_mm=loop_info['length_mm'],
                hole_dia_mm=loop_info['hole_dia_mm'],
                thickness_mm=loop_thickness,
                attach_x_mm=loop_info['attach_x_mm'],
                attach_y_mm=loop_info['attach_y_mm']
            )

            print(f"[DEBUG] loop_mesh created: {loop_mesh is not None}")

            if loop_mesh is not None:
                loop_mesh.visual.face_colors = preview_colors[loop_info['color_id']]
                loop_mesh.metadata['name'] = "Keychain_Loop"
                scene.add_geometry(loop_mesh, node_name="Keychain_Loop", geom_name="Keychain_Loop")
                slot_names_with_loop = slot_names + ["Keychain_Loop"]
                loop_added = True
                print(f"[DEBUG] Loop added to scene successfully")
        except Exception as e:
            print(f"挂孔创建失败: {e}")
            import traceback
            traceback.print_exc()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(tempfile.gettempdir(), f"{base_name}_Lumina.3mf")
    scene.export(out_path)

    preview_mesh = create_preview_mesh(matched_rgb, mask_solid, total_layers)

    if preview_mesh:
        preview_mesh.apply_transform(transform)

    print(f"[DEBUG] preview_mesh={preview_mesh is not None}, loop_added={loop_added}, loop_info={loop_info is not None}")
    if preview_mesh and loop_added and loop_info is not None:
        try:
            loop_thickness = total_layers * PrinterConfig.LAYER_HEIGHT
            preview_loop = create_keychain_loop(
                width_mm=loop_info['width_mm'],
                length_mm=loop_info['length_mm'],
                hole_dia_mm=loop_info['hole_dia_mm'],
                thickness_mm=loop_thickness,
                attach_x_mm=loop_info['attach_x_mm'],
                attach_y_mm=loop_info['attach_y_mm']
            )
            print(f"[DEBUG] preview_loop created: {preview_loop is not None}")
            if preview_loop is not None:
                # 设置挂孔颜色
                loop_color = preview_colors[loop_info['color_id']]
                preview_loop.visual.face_colors = [loop_color] * len(preview_loop.faces)

                # 合并mesh（两者都已经是mm单位）
                preview_mesh = trimesh.util.concatenate([preview_mesh, preview_loop])
                print(f"[DEBUG] preview_mesh merged with loop")
        except Exception as e:
            print(f"预览挂孔创建失败: {e}")
            import traceback
            traceback.print_exc()

    if preview_mesh:
        glb_path = os.path.join(tempfile.gettempdir(), f"{base_name}_Preview.glb")
        preview_mesh.export(glb_path)
    else:
        glb_path = None

    names_to_fix = slot_names_with_loop if loop_added else slot_names
    safe_fix_3mf_names(out_path, names_to_fix)

    Stats.increment("conversions")

    msg = f"✅ 转换完成 ({mode_name_zh} {mode_name_en})！分辨率: {target_w}×{target_h}px | 爱你喵"
    if loop_added:
        msg += f" | 挂孔: {slot_names[loop_info['color_id']]}"

    # Add preview status info
    total_pixels = target_w * target_h
    if glb_path is None and total_pixels > 2_000_000:
        msg += " | ⚠️ 模型过大，已禁用3D预览（请在切片软件中查看3MF）"
    elif glb_path and total_pixels > 500_000:
        msg += " | ℹ️ 3D预览已简化（3MF为完整高质量）"

    return out_path, glb_path, preview_img, msg


def generate_preview_cached(image_path, lut_path, target_width_mm,
                            auto_bg, bg_tol, color_mode):
    """生成预览并缓存数据"""
    if image_path is None:
        return None, None, "❌ 请上传图片"
    if lut_path is None:
        return None, None, "⚠️ 请上传校准文件"

    color_conf = ColorSystem.get(color_mode)
    lut_rgb, ref_stacks, msg = load_calibrated_lut(lut_path.name)
    if lut_rgb is None:
        return None, None, msg
    tree = KDTree(lut_rgb)

    img = Image.open(image_path).convert('RGBA')
    target_w = int(target_width_mm / PrinterConfig.NOZZLE_WIDTH)
    target_h = int(target_w * img.height / img.width)

    img = img.resize((target_w, target_h), Image.Resampling.NEAREST)
    img_arr = np.array(img)
    rgb_arr, alpha_arr = img_arr[:, :, :3], img_arr[:, :, 3]

    flat_rgb = rgb_arr.reshape(-1, 3)
    _, indices = tree.query(flat_rgb)
    matched_rgb = lut_rgb[indices].reshape(target_h, target_w, 3)
    best_stacks = ref_stacks[indices].reshape(target_h, target_w, PrinterConfig.COLOR_LAYERS)

    mask_transparent = alpha_arr < 10
    if auto_bg:
        bg_color = rgb_arr[0, 0]
        diff = np.sum(np.abs(rgb_arr - bg_color), axis=-1)
        mask_transparent = np.logical_or(mask_transparent, diff < bg_tol)

    mask_solid = ~mask_transparent

    preview_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    preview_rgba[mask_solid, :3] = matched_rgb[mask_solid]
    preview_rgba[mask_solid, 3] = 255

    cache = {
        'target_w': target_w, 'target_h': target_h,
        'mask_solid': mask_solid, 'best_stacks': best_stacks,
        'matched_rgb': matched_rgb, 'preview_rgba': preview_rgba.copy(),
        'color_conf': color_conf
    }

    display = render_preview(preview_rgba, None, 0, 0, 0, 0, False, color_conf)

    return display, cache, f"✅ 预览 ({target_w}×{target_h}px) | 点击图片放置挂孔"


def render_preview(preview_rgba, loop_pos, loop_width, loop_length, loop_hole, loop_angle, loop_enabled, color_conf):
    """渲染带挂孔和坐标网格的预览图"""

    h, w = preview_rgba.shape[:2]
    new_w, new_h = w * PREVIEW_SCALE, h * PREVIEW_SCALE

    margin = PREVIEW_MARGIN
    canvas_w = new_w + margin
    canvas_h = new_h + margin

    canvas = Image.new('RGBA', (canvas_w, canvas_h), (240, 240, 245, 255))
    draw = ImageDraw.Draw(canvas)

    grid_color = (220, 220, 225, 255)
    grid_color_main = (200, 200, 210, 255)

    grid_step = 10 * PREVIEW_SCALE
    main_step = 50 * PREVIEW_SCALE

    for x in range(margin, canvas_w, grid_step):
        draw.line([(x, margin), (x, canvas_h)], fill=grid_color, width=1)
    for y in range(margin, canvas_h, grid_step):
        draw.line([(margin, y), (canvas_w, y)], fill=grid_color, width=1)

    for x in range(margin, canvas_w, main_step):
        draw.line([(x, margin), (x, canvas_h)], fill=grid_color_main, width=1)
    for y in range(margin, canvas_h, main_step):
        draw.line([(margin, y), (canvas_w, y)], fill=grid_color_main, width=1)

    axis_color = (100, 100, 120, 255)
    draw.line([(margin, margin), (margin, canvas_h)], fill=axis_color, width=2)  # Y轴
    draw.line([(margin, canvas_h - 1), (canvas_w, canvas_h - 1)], fill=axis_color, width=2)  # X轴

    label_color = (80, 80, 100, 255)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for i, x in enumerate(range(margin, canvas_w, main_step)):
        px_value = i * 50
        if font:
            draw.text((x - 5, canvas_h - margin + 5), str(px_value), fill=label_color, font=font)

    for i, y in enumerate(range(margin, canvas_h, main_step)):
        px_value = i * 50
        if font:
            draw.text((5, y - 5), str(px_value), fill=label_color, font=font)

    pil_img = Image.fromarray(preview_rgba, mode='RGBA')
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.NEAREST)

    canvas.paste(pil_img, (margin, 0), pil_img)

    if loop_enabled and loop_pos is not None:
        canvas = draw_loop_on_image(canvas, loop_pos, loop_width, loop_length, loop_hole, loop_angle, color_conf, margin)

    return np.array(canvas)


def draw_loop_on_image(pil_img, loop_pos, loop_width, loop_length, loop_hole, loop_angle, color_conf, margin=None):


    if margin is None:
        margin = PREVIEW_MARGIN

    loop_w_px = int(loop_width / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
    loop_h_px = int(loop_length / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
    hole_r_px = int(loop_hole / 2 / PrinterConfig.NOZZLE_WIDTH * PREVIEW_SCALE)
    circle_r_px = loop_w_px // 2

    cx = int(loop_pos[0] * PREVIEW_SCALE) + margin
    cy = int(loop_pos[1] * PREVIEW_SCALE)

    loop_size = max(loop_w_px, loop_h_px) * 2 + 20
    loop_layer = Image.new('RGBA', (loop_size, loop_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(loop_layer)

    lc = loop_size // 2
    rect_h = max(1, loop_h_px - circle_r_px)

    loop_color = (220, 60, 60, 200)
    outline_color = (255, 255, 255, 255)

    draw.rectangle([lc - loop_w_px//2, lc, lc + loop_w_px//2, lc + rect_h],
                  fill=loop_color, outline=outline_color, width=2)

    draw.ellipse([lc - circle_r_px, lc - circle_r_px,
                 lc + circle_r_px, lc + circle_r_px],
                fill=loop_color, outline=outline_color, width=2)

    draw.ellipse([lc - hole_r_px, lc - hole_r_px,
                 lc + hole_r_px, lc + hole_r_px],
                fill=(0, 0, 0, 0))

    if loop_angle != 0:
        loop_layer = loop_layer.rotate(-loop_angle, center=(lc, lc),
                                       expand=False, resample=Image.BICUBIC)

    paste_x = cx - lc
    paste_y = cy - lc - rect_h // 2
    pil_img.paste(loop_layer, (paste_x, paste_y), loop_layer)

    return pil_img


def on_preview_click(cache, loop_pos, evt: gr.SelectData):

    if evt is None or cache is None:
        return loop_pos, False, "点击无效 - 请先生成预览"

    click_x, click_y = evt.index

    click_x = click_x - PREVIEW_MARGIN

    orig_x = click_x / PREVIEW_SCALE
    orig_y = click_y / PREVIEW_SCALE

    target_w = cache['target_w']
    target_h = cache['target_h']
    orig_x = max(0, min(target_w - 1, orig_x))
    orig_y = max(0, min(target_h - 1, orig_y))

    pos_info = f"位置: ({orig_x:.1f}, {orig_y:.1f}) px"
    return (orig_x, orig_y), True, pos_info

def update_preview_with_loop(cache, loop_pos, add_loop,
                            loop_width, loop_length, loop_hole, loop_angle):
    if cache is None:
        return None

    preview_rgba = cache['preview_rgba'].copy()
    color_conf = cache['color_conf']

    display = render_preview(
        preview_rgba,
        loop_pos if add_loop else None,
        loop_width, loop_length, loop_hole, loop_angle,
        add_loop, color_conf
    )
    return display

def on_remove_loop():
    return None, False, 0, "已移除挂孔"

def generate_final_model(image_path, lut_path, target_width_mm, spacer_thick,
                        structure_mode, auto_bg, bg_tol, color_mode,
                        add_loop, loop_width, loop_length, loop_hole, loop_pos,
                        modeling_mode="vector", quantize_colors=64):
    return convert_image_to_3d(
        image_path, lut_path, target_width_mm, spacer_thick,
        structure_mode, auto_bg, bg_tol, color_mode,
        add_loop, loop_width, loop_length, loop_hole, loop_pos,
        modeling_mode, quantize_colors
    )
