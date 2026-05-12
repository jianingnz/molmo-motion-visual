"""
Re-bake a clip bundle's scene PC at stride=1.

Reuses the existing JSON's metadata to figure out:
  - which vipe artifact stem to read   (camera.video_stem)
  - which raw video frame to project   (pc_bin.frame_indices_original)
  - which raw video frame for full PC  (full_pc_bin.frame_index_original, if present)
  - where to write                      (pc_bin.url, full_pc_bin.url)

Backprojects depth at stride=1 (every pixel) and OVERWRITES the existing
_pc.bin / _full_pc.bin. Touches NOTHING ELSE in the bundle — no chrono,
no clip mp4, no hi-res strips, no JSON edits (except updating
`pc_bin.subsample = 1` and `n_points`).

Runs entirely on CPU (no MoGe — just reads pre-baked vipe depth zips).
~5-10s per clip for a typical 480p depth map.

Usage:
  python rebake_pc_stride1.py <clip-id> [<clip-id> ...]
"""
import argparse, json, sys, tempfile, os, zipfile
from pathlib import Path
import numpy as np
import cv2
import OpenEXR, Imath


REPO_ROOT  = Path("/weka/prior-default/jianingz/home/visual/molmo-motion-visual")
VIPE_ROOT  = Path("/weka/prior-default/jianingz/home/project/_GenTraj/vipe/vipe_results")


def load_exr_depth(raw_bytes: bytes) -> np.ndarray:
    """Decode a single-channel EXR depth file (vipe convention: channel 'Z')."""
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
        f.write(raw_bytes); fname = f.name
    try:
        exr = OpenEXR.InputFile(fname)
        dw = exr.header()['dataWindow']
        H = dw.max.y - dw.min.y + 1
        W = dw.max.x - dw.min.x + 1
        ch = 'Z' if 'Z' in exr.header()['channels'] else list(exr.header()['channels'].keys())[0]
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        buf = exr.channel(ch, pt)
        depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()
    finally:
        os.unlink(fname)
    return depth


def backproject_stride1(depth: np.ndarray, rgb: np.ndarray,
                        c2w: np.ndarray, intr: np.ndarray):
    """Backproject every depth pixel → world-space XYZ + RGB."""
    H, W = depth.shape
    fx, fy, cx, cy = intr
    uu, vv = np.meshgrid(np.arange(W, dtype=np.int32),
                         np.arange(H, dtype=np.int32))
    uu, vv = uu.ravel(), vv.ravel()
    z = depth[vv, uu]
    valid = (z > 0) & np.isfinite(z)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    xc = (uu.astype(np.float32) - cx) / fx * z
    yc = (vv.astype(np.float32) - cy) / fy * z
    pts_cam = np.stack([xc, yc, z, np.ones_like(z)], axis=1)
    xyz = (c2w @ pts_cam.T).T[:, :3].astype(np.float32)
    cols = rgb[vv, uu].astype(np.uint8)
    return xyz, cols


def write_pc_binary(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray):
    N = int(xyz.shape[0])
    with open(path, "wb") as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())


def rebake_one(clip_id: str):
    json_path = REPO_ROOT / "static" / "data" / f"{clip_id}.json"
    if not json_path.exists():
        print(f"  ✗ {clip_id}: bundle JSON missing"); return False
    bundle = json.loads(json_path.read_text())
    stem = bundle.get("camera", {}).get("video_stem")
    if not stem:
        print(f"  ✗ {clip_id}: camera.video_stem missing"); return False

    # Locate vipe artifacts.
    pose_p  = VIPE_ROOT / "pose"        / f"{stem}.npz"
    intr_p  = VIPE_ROOT / "intrinsics"  / f"{stem}.npz"
    depth_p = VIPE_ROOT / "depth"       / f"{stem}.zip"
    rgb_p   = VIPE_ROOT / "rgb"         / f"{stem}.mp4"
    for p in [pose_p, intr_p, depth_p, rgb_p]:
        if not p.exists():
            print(f"  ✗ {clip_id}: missing {p}"); return False

    poses = np.load(pose_p)['data'].astype(np.float32)
    intrs = np.load(intr_p)['data'].astype(np.float32)

    # Open RGB mp4 once.
    cap = cv2.VideoCapture(str(rgb_p))

    def read_rgb(fi: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {fi} from {rgb_p}")
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

    out_lines = []
    with zipfile.ZipFile(depth_p) as zf:
        depth_names = sorted(zf.namelist())

        def backproject_frame(fi: int):
            depth = load_exr_depth(zf.read(depth_names[int(fi)]))
            xyz, cols = backproject_stride1(depth, read_rgb(fi),
                                            poses[int(fi)], intrs[int(fi)])
            return xyz, cols

        # 1) clip-window PC.
        pcbin = bundle.get("pc_bin", {})
        if pcbin and "frame_indices_original" in pcbin and "url" in pcbin:
            fi_list = pcbin["frame_indices_original"]
            xyz_all, col_all = [], []
            for fi in fi_list:
                xyz, cols = backproject_frame(fi)
                xyz_all.append(xyz); col_all.append(cols)
            xyz = np.concatenate(xyz_all, axis=0)
            cols = np.concatenate(col_all, axis=0)
            out_path = REPO_ROOT / pcbin["url"]
            write_pc_binary(out_path, xyz, cols)
            pcbin["n_points"]   = int(xyz.shape[0])
            pcbin["subsample"]  = 1
            bundle["pc_bin"]    = pcbin
            out_lines.append(f"    wrote {out_path.name}  {xyz.shape[0]} pts  ({out_path.stat().st_size//1024} KB)")

        # 2) full-video frame-0 PC (EgoDex / HD-EPIC always ship one).
        fullpc = bundle.get("full_pc_bin", {})
        if fullpc and "url" in fullpc:
            fi = fullpc.get("frame_index_original", 0)
            xyz, cols = backproject_frame(int(fi))
            out_path = REPO_ROOT / fullpc["url"]
            write_pc_binary(out_path, xyz, cols)
            fullpc["n_points"]  = int(xyz.shape[0])
            fullpc["subsample"] = 1
            bundle["full_pc_bin"] = fullpc
            out_lines.append(f"    wrote {out_path.name}  {xyz.shape[0]} pts  ({out_path.stat().st_size//1024} KB)")

    cap.release()
    # Write updated bundle JSON.
    json_path.write_text(json.dumps(bundle, indent=2))
    print(f"  ✓ {clip_id}:")
    for line in out_lines: print(line)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip_ids", nargs="+",
                    help="Clip id(s) (without .json). Example: clean_surface_1713_t10")
    args = ap.parse_args()
    ok = 0
    for cid in args.clip_ids:
        if rebake_one(cid): ok += 1
    print(f"\nDone: {ok}/{len(args.clip_ids)} clips re-baked at stride=1.")


if __name__ == "__main__":
    main()
