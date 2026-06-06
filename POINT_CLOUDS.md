# Point Clouds — Inventory, Sizes & Data Paths

_Report generated 2026-06-01 for `molmo-motion-visual`._

## 1. Location

All point-cloud data lives in a single directory:

```
/weka/prior-default/jianingz/home/visual/molmo-motion-visual/static/data/
```

Each clip is stored as a pair:
- `*_pc.bin`  — the point cloud (binary)
- `*.json`    — scene / track / camera metadata for that clip

Some clips also carry auxiliary binaries (`_full_pc.bin`, `_pc_dist.bin`, `_pc_lastframe.bin`, `_pc_f33.bin`, `_cam.bin`, `_gt3d*.bin`) — see §5.

## 2. Binary format

Each `*_pc.bin` is a flat little-endian binary blob:

| Offset | Type            | Meaning                          |
|--------|-----------------|----------------------------------|
| 0      | `uint32` N      | number of points                 |
| 4      | `float32[N][3]` | XYZ world coordinates            |
| 4+12N  | `uint8[N][3]`   | RGB color (0–255)                |

So **15 bytes per point** plus a 4-byte header → `filesize = 4 + 15·N`.

Reference writer: `build/rebake_pc_stride1.py:69-74`:

```python
def write_pc_binary(path, xyz, rgb_u8):
    N = xyz.shape[0]
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())     # header: point count
        f.write(xyz.astype(np.float32).tobytes())   # N x 3 float32
        f.write(rgb_u8.astype(np.uint8).tobytes())  # N x 3 uint8
```

Point clouds are produced by back-projecting per-pixel depth into world space
(`build/rebake_pc_stride1.py`, `build/regen_hot3d_dense_scene_pc.py`,
`build/build_scene_moge_lastframe.py`, etc.). The dense/full clips are baked at
stride = 1 (every pixel projected).

## 3. Totals

| Metric                                   | Value          |
|------------------------------------------|----------------|
| Core point-cloud files (`*_pc.bin` / `*_full_pc.bin`) | **36**         |
| Total points across all core files       | **7,478,287**  |
| Total size of core point clouds          | **112.2 MB** (112,174,449 bytes) |
| Auxiliary binaries                        | 19 files, **79.0 MB** |
| Entire `static/data/` directory           | **~251 MB**    |

## 4. Point-cloud files (full list)

Sorted by point count (descending). Sizes in MB (decimal).

| Points  | Size (MB) | File (relative to `static/data/`) |
|--------:|----------:|-----------------------------------|
| 409,920 | 6.15 | `basic_pick_place_14851_t29_pc.bin` |
| 409,920 | 6.15 | `basic_pick_place_14851_t29_full_pc.bin` |
| 409,920 | 6.15 | `clean_surface_1713_t10_pc.bin` |
| 409,920 | 6.15 | `clean_surface_1713_t10_full_pc.bin` |
| 409,920 | 6.15 | `clean_surface_3603_t12_pc.bin` |
| 409,920 | 6.15 | `insert_remove_utensils_534_t185_pc.bin` |
| 409,920 | 6.15 | `part4_pour_1027_pc.bin` |
| 223,022 | 3.35 | `AUTOLab_5d05c5aa_2023-07-07-18h-52m-04s_22008760_pc.bin` |
| 222,743 | 3.34 | `CLVR_236539bc_2023-06-25-18h-35m-31s_20655732_object_t19_pc.bin` |
| 220,900 | 3.31 | `hot3d_clip002562_obj2_s008_e025_pc.bin` |
| 220,900 | 3.31 | `hot3d_clip002658_obj3_s038_e055_pc.bin` |
| 220,900 | 3.31 | `hot3d_clip003020_obj0_s023_e056_pc.bin` |
| 220,900 | 3.31 | `hot3d_clip003137_obj0_s055_e074_pc.bin` |
| 220,739 | 3.31 | `AUTOLab_44bb9c36_2023-11-23-19h-41m-33s_22008760_pc.bin` |
| 216,023 | 3.24 | `GuptaLab_553d1bd5_2023-04-20-12h-41m-59s_22246076_object_t33_pc.bin` |
| 215,711 | 3.24 | `REAL_de601749_2023-06-17-11h-34m-24s_20540549_object_t25_pc.bin` |
| 215,426 | 3.23 | `PennPAL_c5f808b7_2023-10-09-21h-10m-27s_27085680_object_t14_pc.bin` |
| 215,321 | 3.23 | `AUTOLab_0d4edc83_2023-11-03-16h-52m-04s_22008760_pc.bin` |
| 215,235 | 3.23 | `AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334_object_t27_pc.bin` |
| 215,178 | 3.23 | `AUTOLab_0d4edc83_2023-11-03-15h-58m-46s_22008760_pc.bin` |
| 214,786 | 3.22 | `AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334_pc.bin` |
| 213,619 | 3.20 | `AUTOLab_0d4edc83_2023-10-21-19h-16m-18s_22008760_pc.bin` |
| 212,156 | 3.18 | `AUTOLab_0d4edc83_2023-10-21-19h-46m-27s_22008760_pc.bin` |
| 211,742 | 3.18 | `AUTOLab_84bd5053_2023-07-14-14h-56m-46s_24400334_object_t23_pc.bin` |
| 209,065 | 3.14 | `PennPAL_c5f808b7_2023-10-30-00h-50m-14s_25455306_object_t67_pc.bin` |
| 208,431 | 3.13 | `AUTOLab_0d4edc83_2023-10-21-20h-20m-00s_22008760_pc.bin` |
| 207,779 | 3.12 | `AUTOLab_0d4edc83_2023-10-21-20h-16m-31s_22008760_object_t26_pc.bin` |
| 79,524  | 1.19 | `hot3d_clip1995_clip1996_pc.bin` |
| 45,600  | 0.68 | `pick_food_2372_t20_pc.bin` |
| 45,600  | 0.68 | `pick_food_2372_t20_full_pc.bin` |
| 25,600  | 0.38 | `P05-20240425-171455-251_pc.bin` |
| 25,600  | 0.38 | `P05-20240427-145526-105_pc.bin` |
| 25,600  | 0.38 | `P06-20240510-100047-225_pc.bin` |
| 22,491  | 0.34 | `flamingo_flamingo_t2_pc.bin` |
| 14,966  | 0.22 | `camel_camal_t2_pc.bin` |
| 3,290   | 0.05 | `car-turn_silver_car_t2_pc.bin` |

### Size tiers

- **Dense robot clips (~410K pts, 6.15 MB each):** `basic_pick_place`,
  `clean_surface` (×2), `insert_remove_utensils`, `part4_pour`. Baked at stride=1.
- **DROID lab + HOT3D clips (~205–223K pts, 3.1–3.3 MB):** `AUTOLab`, `CLVR`,
  `GuptaLab`, `PennPAL`, `REAL`, and the four `hot3d_clip0025xx/0030xx` clips.
- **Small clips (3K–80K pts):** DAVIS-style (`camel`, `car-turn`, `flamingo`),
  egocentric `P05`/`P06` (25.6K each), `pick_food` (45.6K),
  `hot3d_clip1995_clip1996` (79.5K).

## 5. Auxiliary binaries (not raw point clouds)

These accompany certain clips (mostly HOT3D). 19 files, ~79.0 MB total.

| Size (bytes) | File | Notes |
|-------------:|------|-------|
| 35,808,022 | `hot3d_clip1995_clip1996_gt3d_a.bin` | ground-truth 3D track A |
| 35,808,022 | `hot3d_clip1995_clip1996_gt3d_b.bin` | ground-truth 3D track B |
| 1,192,864  | `hot3d_clip1995_clip1996_pc_lastframe.bin` | last-frame scene PC (same format as `_pc.bin`) |
| 883,600    | `hot3d_clip002562_obj2_s008_e025_pc_dist.bin` | per-point distance buffer |
| 883,600    | `hot3d_clip002658_obj3_s038_e055_pc_dist.bin` | per-point distance buffer |
| 883,600    | `hot3d_clip003020_obj0_s023_e056_pc_dist.bin` | per-point distance buffer |
| 883,600    | `hot3d_clip003137_obj0_s055_e074_pc_dist.bin` | per-point distance buffer |
| 684,004    | `part4_pour_1027_pc_f33.bin` | point cloud at frame 33 |
| 484,044    | `hot3d_clip003020_obj0_s023_e056_gt3d.bin` | ground-truth 3D track |
| 318,096    | `hot3d_clip1995_clip1996_pc_dist.bin` | per-point distance buffer |
| 318,096    | `hot3d_clip1995_clip1996_pc_lastframe_dist.bin` | distance buffer (last frame) |
| 288,044    | `hot3d_clip003137_obj0_s055_e074_gt3d.bin` | ground-truth 3D track |
| 260,044    | `hot3d_clip002562_obj2_s008_e025_gt3d.bin` | ground-truth 3D track |
| 260,044    | `hot3d_clip002658_obj3_s038_e055_gt3d.bin` | ground-truth 3D track |
| 6,812      | `hot3d_clip1995_clip1996_cam.bin` | camera intrinsics/poses |
| 2,204      | `hot3d_clip003020_obj0_s023_e056_cam.bin` | camera intrinsics/poses |
| 1,308      | `hot3d_clip003137_obj0_s055_e074_cam.bin` | camera intrinsics/poses |
| 1,180      | `hot3d_clip002562_obj2_s008_e025_cam.bin` | camera intrinsics/poses |
| 1,180      | `hot3d_clip002658_obj3_s038_e055_cam.bin` | camera intrinsics/poses |

## 6. How to read a point cloud (Python)

```python
import numpy as np

def load_pc(path):
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype="<u4")[0]
        xyz = np.frombuffer(f.read(12 * n), dtype="<f4").reshape(n, 3)
        rgb = np.frombuffer(f.read(3 * n), dtype=np.uint8).reshape(n, 3)
    return xyz, rgb

xyz, rgb = load_pc("static/data/clean_surface_1713_t10_pc.bin")
print(xyz.shape, rgb.shape)   # (409920, 3) (409920, 3)
```
