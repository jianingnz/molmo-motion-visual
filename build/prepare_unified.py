#!/usr/bin/env python3
"""
prepare_unified.py — build a self-contained viewer bundle from the unified
`molmo-motion-1m` release (the dataset under
/weka/oe-training-default/jianingz/molmo-motion-1m).

Unlike the legacy `prepare_clip.py` (which consumed a pre-existing motion5
gt/pred JSON + a separate vipe tree), this script reads the unified release
directly:

  <root>/<dataset>/annotations/<dataset>_clips.json   (caption, fps, motion ranges)
  <root>/<dataset>/tracks/...                          (3D point trajectories, world frame)
  <root>/<dataset>/camera/...                          (c2w pose + intrinsics)
  <root>/<dataset>/depth/...                           (EXR-in-zip or .npz depth)
  <root>/<dataset>/videos/<file>.mp4

It produces the same viewer bundle contract as the other build scripts:
a clip JSON + `<clip_id>_pc.bin` (depth-backprojected scene point cloud) +
trimmed mp4 + chronophotography jpg.

Key facts verified empirically against the release (2026-06-04):
  * The stored camera pose `data` / `cam_poses` is **cam-to-world (c2w)** for
    egodex, ytvis, hepic, molmospaces — reprojecting a 3D track point with
    `K @ inv(c2w)` reproduces the released 2D tracks exactly. (The top-level
    README calls egodex/ytvis pose "world-to-camera"; that label is wrong for
    the released arrays — trust the reprojection.)
  * 3D tracks are in the **world frame**; the scene PC is backprojected into
    the same world frame via c2w, so tracks and PC are aligned.
  * 2D is **derived by projecting the 3D tracks** through (pose, intrinsics).
    This gives a 2D set that is aligned 1:1 with the 3D points (needed for
    per-point color sampling, chrono stamps, and the 2D overlay). hepic's
    shipped 2D is a separate AllTracker grid that does NOT correspond to the
    per-object 3D, so projection is the only consistent source.

This release ships **ground-truth training tracks only** (no model
prediction). The bundle therefore sets `pred_3d == gt_3d` and emits
`viewer_defaults.showPred = false` so the figure shows the training tracks
cleanly. The user can still toggle the pred layer on in the sidebar.

Per-dataset 3D track layout (normalized internally to (K, T, 3)):
  egodex      tracks/object/{file}_3d.npz  points_3d (K,T,3) array
  ytvis       tracks/{file}_3d.npz         points_3d  -> 0-d object -> {obj:(K,T,3)}
  hepic       tracks/{file}_3d.npz         points_3d  -> 0-d object -> {obj:(K,T,3)}
  molmospaces camera/{file}.npz            points_3d (T,K,3) + body_ids (K,)
"""

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_BIN = "ffmpeg"

DATASETS = ("egodex", "ytvis", "hepic", "molmospaces")


# ───────────────────────────── depth decode ─────────────────────────────

def load_exr_depth(raw_bytes: bytes) -> np.ndarray:
    """Decode a single-channel ('Z') EXR byte string to float32 (H, W)."""
    import OpenEXR
    import Imath
    with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
        f.write(raw_bytes); fname = f.name
    try:
        exr = OpenEXR.InputFile(fname)
        dw = exr.header()['dataWindow']
        W = dw.max.x - dw.min.x + 1
        H = dw.max.y - dw.min.y + 1
        buf = exr.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT))
        depth = np.frombuffer(buf, dtype=np.float32).reshape(H, W).copy()
        exr.close()
    finally:
        os.unlink(fname)
    return depth


def backproject_depth_to_world(depth, rgb_bgr, c2w, intr_fxfycxcy, subsample=1):
    """Regular-grid sample of depth → world-space (xyz float32, rgb uint8).

    `rgb_bgr` must already be the same (H, W) as `depth`. Colors are returned
    as RGB uint8 (the bin format stores RGB). c2w is cam→world.
    """
    H, W = depth.shape
    fx, fy, cx, cy = intr_fxfycxcy
    us = np.arange(0, W, subsample, dtype=np.int32)
    vs = np.arange(0, H, subsample, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    uu, vv = uu.ravel(), vv.ravel()
    z = depth[vv, uu]
    valid = (z > 0) & np.isfinite(z)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    xc = (uu.astype(np.float32) - cx) / fx * z
    yc = (vv.astype(np.float32) - cy) / fy * z
    pts_cam = np.stack([xc, yc, z, np.ones_like(z)], axis=1)
    xyz = (c2w @ pts_cam.T).T[:, :3].astype(np.float32)
    bgr = rgb_bgr[vv, uu]
    rgb = bgr[:, ::-1].astype(np.uint8)
    return xyz, rgb


def write_pc_binary(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray):
    """4-byte LE uint32 N | N*12 float32 xyz | N*3 uint8 rgb."""
    N = int(xyz.shape[0])
    with open(path, 'wb') as f:
        f.write(np.uint32(N).tobytes())
        f.write(xyz.astype(np.float32).tobytes())
        f.write(rgb_u8.astype(np.uint8).tobytes())


# ───────────────────────────── projection ─────────────────────────────

def project_world_to_px(X_world, c2w, intr_fxfycxcy):
    """Project a world point to pixel (u, v) + camera-space z. Returns (u,v,z)."""
    w2c = np.linalg.inv(c2w)
    Xc = w2c @ np.array([X_world[0], X_world[1], X_world[2], 1.0])
    z = float(Xc[2])
    if z <= 1e-6:
        return None
    fx, fy, cx, cy = intr_fxfycxcy
    u = fx * Xc[0] / Xc[2] + cx
    v = fy * Xc[1] / Xc[2] + cy
    return float(u), float(v), z


# ───────────────────────────── color / frames ─────────────────────────────

def sample_color_px(img_bgr, u_px, v_px, half=3):
    """Mean RGB over a small patch around pixel (u,v). img is BGR."""
    H, W = img_bgr.shape[:2]
    cx = int(round(u_px)); cy = int(round(v_px))
    if not (0 <= cx < W and 0 <= cy < H):
        return None
    x0 = max(0, cx - half); x1 = min(W, cx + half + 1)
    y0 = max(0, cy - half); y1 = min(H, cy + half + 1)
    patch = img_bgr[y0:y1, x0:x1].reshape(-1, 3)
    m = patch.mean(axis=0)
    return [int(m[2]), int(m[1]), int(m[0])]


def grab_frame_bgr(mp4_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_idx} from {mp4_path}")
    return frame


def grab_frames_bgr(mp4_path: Path, frame_indices) -> list:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    out = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {fi} from {mp4_path}")
        out.append(frame)
    cap.release()
    return out


# ───────────────────────────── chronophotography ─────────────────────────────

def build_object_stamps_chrono(bg, stamp_frames, stamp_pts2d_px, dilate_px=2, edge_blur=3):
    """Composite an opaque object cutout (convex hull of the object's 2D pts)
    per stamp frame onto `bg`. `stamp_pts2d_px` are lists of (u_px, v_px) | None.
    """
    H, W = bg.shape[:2]
    out = bg.astype(np.float32).copy()
    for frame, pts in zip(stamp_frames, stamp_pts2d_px):
        coords = []
        for pt in pts:
            if pt is None:
                continue
            u, v = pt
            if not (0 <= u < W and 0 <= v < H):
                continue
            coords.append([int(round(u)), int(round(v))])
        if len(coords) < 3:
            continue
        hull = cv2.convexHull(np.array(coords, dtype=np.int32))
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        if edge_blur > 1:
            soft = cv2.GaussianBlur(mask, (edge_blur, edge_blur), 0).astype(np.float32) / 255.0
        else:
            soft = mask.astype(np.float32) / 255.0
        soft = soft[..., None]
        out = out * (1.0 - soft) + frame.astype(np.float32) * soft
    return out.clip(0, 255).astype(np.uint8)


# ───────────────────────────── mp4 trim ─────────────────────────────

def trim_mp4_ffmpeg(src: Path, dst: Path, start_frame: int, end_frame: int, fps: float):
    vf = (f"select=between(n\\,{start_frame}\\,{end_frame}),"
          f"setpts=PTS-STARTPTS,fps={fps}")
    cmd = [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", vf, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           str(dst)]
    subprocess.run(cmd, check=True)


# ───────────────────────────── per-dataset loader ─────────────────────────────

class UnifiedClip:
    """Loads one clip's tracks / camera / depth / video from the unified release.

    Normalizes 3D tracks to a dict { obj_name -> (K, T, 3) } plus a parallel
    visibility dict { obj_name -> (K, T) bool }. Exposes c2w poses (T,4,4),
    intrinsics (fx,fy,cx,cy) per frame, depth (H,W) per frame, and BGR video
    frames (resized to depth resolution).
    """

    def __init__(self, dataset: str, root: Path, file: str):
        if dataset not in DATASETS:
            raise NotImplementedError(f"dataset {dataset} not supported")
        self.dataset = dataset
        self.root = root
        self.file = file
        ds_dir = root / dataset
        clips = json.loads((ds_dir / "annotations" / f"{dataset}_clips.json").read_text())
        ent = [e for e in clips if e["file"] == file]
        if not ent:
            raise RuntimeError(f"{file} not found in {dataset}_clips.json")
        self.entry = ent[0]
        self.caption = self.entry.get("caption", "")
        self.fps = float(self.entry["fps"])
        self.num_frames = int(self.entry["num_frames"])
        self.clips_by_object = self.entry["clips_by_object"]

        self.video_path = ds_dir / "videos" / f"{file}.mp4"
        if not self.video_path.exists():
            raise RuntimeError(f"missing video {self.video_path}")

        if dataset == "molmospaces":
            self._load_molmospaces(ds_dir)
        else:
            self._load_vipe(ds_dir, dataset)

        self.depth_H, self.depth_W = self._depth_HW()

    # -- vipe-style datasets: egodex / ytvis / hepic --
    def _load_vipe(self, ds_dir, dataset):
        if dataset == "egodex":
            t3 = np.load(ds_dir / "tracks" / "object" / f"{self.file}_3d.npz", allow_pickle=True)
        else:
            t3 = np.load(ds_dir / "tracks" / f"{self.file}_3d.npz", allow_pickle=True)
        p3 = t3["points_3d"]
        v3 = t3["visibility"]
        objs = {}
        viss = {}
        if p3.shape == ():            # 0-d object → per-object dict (ytvis, hepic)
            pdict = p3.item(); vdict = v3.item()
            for k in pdict:
                objs[k] = np.asarray(pdict[k], dtype=np.float64)            # (K,T,3)
                viss[k] = np.asarray(vdict[k]).reshape(objs[k].shape[0], objs[k].shape[1]).astype(bool)
        else:                          # flat single-object array (egodex)
            objs["object"] = np.asarray(p3, dtype=np.float64)              # (K,T,3)
            viss["object"] = np.asarray(v3).reshape(p3.shape[0], p3.shape[1]).astype(bool)
        self.tracks3d = objs
        self.vis3d = viss
        pose = np.load(ds_dir / "camera" / "pose" / f"{self.file}.npz")["data"].astype(np.float64)
        intr = np.load(ds_dir / "camera" / "intrinsics" / f"{self.file}.npz")["data"].astype(np.float64)
        self.c2w = pose                                                    # (T,4,4) cam→world
        self.intr = intr                                                   # (T,4) fx,fy,cx,cy
        self._depth_zip = ds_dir / "depth" / f"{self.file}.zip"
        if not self._depth_zip.exists():
            raise RuntimeError(f"missing depth zip {self._depth_zip}")
        self._depth_kind = "exr_zip"

    def _load_molmospaces(self, ds_dir):
        cam = np.load(ds_dir / "camera" / f"{self.file}.npz", allow_pickle=True)
        p3 = np.asarray(cam["points_3d"], dtype=np.float64)                # (T,K,3)
        vis = np.asarray(cam["visibility"]).astype(bool)                   # (T,K)
        body_ids = np.asarray(cam["body_ids"]).astype(int)                 # (K,)
        p3 = np.transpose(p3, (1, 0, 2))                                   # → (K,T,3)
        vis = np.transpose(vis, (1, 0))                                    # → (K,T)
        objs = {}; viss = {}
        for key in self.clips_by_object:                                   # e.g. "body_315"
            bid = int(key.split("_")[1])
            sel = np.where(body_ids == bid)[0]
            if len(sel) == 0:
                raise RuntimeError(f"no points with body id {bid} for {self.file}")
            objs[key] = p3[sel]
            viss[key] = vis[sel]
        self.tracks3d = objs
        self.vis3d = viss
        self.c2w = np.asarray(cam["cam_poses"], dtype=np.float64)          # (T,4,4) cam→world
        K = np.asarray(cam["intrinsics"], dtype=np.float64)               # (3,3) constant
        T = self.c2w.shape[0]
        intr1 = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)
        self.intr = np.tile(intr1, (T, 1))
        self._depth_frames = np.asarray(cam["depth_frames"], dtype=np.float32)  # (T,H,W)
        self._depth_kind = "npz_array"

    def _depth_HW(self):
        if self._depth_kind == "npz_array":
            return int(self._depth_frames.shape[1]), int(self._depth_frames.shape[2])
        with zipfile.ZipFile(self._depth_zip) as zf:
            names = sorted(zf.namelist())
            d0 = load_exr_depth(zf.read(names[0]))
        return int(d0.shape[0]), int(d0.shape[1])

    def intr_at(self, f):
        return tuple(float(x) for x in self.intr[int(f)])

    def c2w_at(self, f):
        return self.c2w[int(f)]

    def depth_at(self, f):
        if self._depth_kind == "npz_array":
            return self._depth_frames[int(f)]
        with zipfile.ZipFile(self._depth_zip) as zf:
            names = sorted(zf.namelist())
            return load_exr_depth(zf.read(names[int(f)]))

    def rgb_at(self, f):
        """BGR video frame at depth resolution."""
        img = grab_frame_bgr(self.video_path, f)
        if img.shape[0] != self.depth_H or img.shape[1] != self.depth_W:
            img = cv2.resize(img, (self.depth_W, self.depth_H), interpolation=cv2.INTER_AREA)
        return img


# ───────────────────────────── bundle assembly ─────────────────────────────

def select_objects(uc: UnifiedClip, objs_arg: str):
    """Return [(obj_name, (s, e))] for the chosen objects. Default: the single
    object whose longest motion range is longest (most to show)."""
    cbo = uc.clips_by_object
    if objs_arg == "all":
        keys = list(cbo.keys())
    elif objs_arg:
        keys = [k.strip() for k in objs_arg.split(",")]
        for k in keys:
            if k not in cbo:
                raise RuntimeError(f"object {k} not in clips_by_object {list(cbo)}")
    else:
        def longest(k):
            return max(e - s for s, e in cbo[k])
        keys = [max(cbo, key=longest)]
    out = []
    for k in keys:
        ranges = cbo[k]
        s, e = max(ranges, key=lambda r: r[1] - r[0])   # longest motion range for this obj
        out.append((k, (int(s), int(e))))
    return out


def build_config_for_object(uc, obj_name, s, e, video_for_color):
    """Build one viewer `config` dict for object `obj_name` over clip-local
    window [s, e] (inclusive). gt_2d is derived by projecting gt_3d."""
    P = uc.tracks3d[obj_name]      # (K, T, 3)
    V = uc.vis3d[obj_name]         # (K, T)
    K = P.shape[0]
    n_clip = e - s + 1
    gt_3d, gt_2d, vis = [], [], []
    for t_local in range(n_clip):
        f = s + t_local
        intr = uc.intr_at(f); c2w = uc.c2w_at(f)
        row3, row2, rowv = [], [], []
        for k in range(K):
            visible = bool(V[k, f])
            if not visible:
                row3.append(None); row2.append(None); rowv.append(False); continue
            X = P[k, f]
            if not np.all(np.isfinite(X)):
                row3.append(None); row2.append(None); rowv.append(False); continue
            row3.append([float(X[0]), float(X[1]), float(X[2])])
            pr = project_world_to_px(X, c2w, intr)
            if pr is None:
                row2.append(None)
            else:
                u, v, _ = pr
                row2.append([u / uc.depth_W, v / uc.depth_H])
            rowv.append(True)
        gt_3d.append(row3); gt_2d.append(row2); vis.append(rowv)

    # per-point color from the first clip frame's projected 2D
    f0 = s
    img0 = video_for_color
    intr0 = uc.intr_at(f0); c2w0 = uc.c2w_at(f0)
    colors = []
    for k in range(K):
        if not bool(V[k, f0]) or not np.all(np.isfinite(P[k, f0])):
            colors.append(None); continue
        pr = project_world_to_px(P[k, f0], c2w0, intr0)
        if pr is None:
            colors.append(None); continue
        u, v, _ = pr
        colors.append(sample_color_px(img0, u, v))

    n_hist = 1
    all_frames = list(range(n_clip))
    cfg = {
        "obj_name": obj_name,
        "t": int(s),
        "n_hist": n_hist,
        "hist_frames": all_frames[:n_hist],
        "future_frames": all_frames[n_hist:],
        "all_frames": all_frames,
        "gt_3d": gt_3d,
        "pred_3d": gt_3d,             # GT-only release: pred mirrors gt (hidden by default)
        "vis": vis,
        "gt_2d": gt_2d,
        "pred_2d": gt_2d,
        "pt_colors_rgb": colors,
        "color_sample_frame": 0,
        "l2": 0.0,
    }
    return cfg


def merge_configs(configs, obj_name):
    """Concatenate several per-object configs (sharing the same frame window)
    into one config so all objects render together in a single view."""
    n_frames = len(configs[0]["all_frames"])
    for c in configs:
        if len(c["all_frames"]) != n_frames:
            raise RuntimeError("merge requires configs over the same frame window")
    merged = {
        "obj_name": obj_name,
        "t": configs[0]["t"],
        "n_hist": configs[0]["n_hist"],
        "hist_frames": configs[0]["hist_frames"],
        "future_frames": configs[0]["future_frames"],
        "all_frames": configs[0]["all_frames"],
        "color_sample_frame": configs[0]["color_sample_frame"],
        "l2": 0.0,
        "gt_3d": [], "pred_3d": [], "gt_2d": [], "pred_2d": [], "vis": [],
        "pt_colors_rgb": [],
    }
    for c in configs:
        merged["pt_colors_rgb"].extend(c["pt_colors_rgb"])
    for t in range(n_frames):
        for key in ("gt_3d", "pred_3d", "gt_2d", "pred_2d", "vis"):
            row = []
            for c in configs:
                row.extend(c[key][t])
            merged[key].append(row)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--file", required=True, help="video_id / slug (matches clips.json 'file')")
    ap.add_argument("--clip-id", required=True, help="output bundle id")
    ap.add_argument("--objs", default="",
                    help="comma list of clips_by_object keys, or 'all'. "
                         "Default: the single object with the longest motion range.")
    ap.add_argument("--root", type=Path,
                    default=Path("/weka/oe-training-default/jianingz/molmo-motion-1m"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/weka/prior-default/jianingz/home/visual/molmo-motion-visual"))
    ap.add_argument("--pc-subsample", type=int, default=1,
                    help="pixel stride when backprojecting depth (1 = every pixel = highest res)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap the clip window length (frames from its start). "
                         "0 = no cap. Useful for datasets with very long motion ranges.")
    ap.add_argument("--pc-frame", type=str, default="start",
                    help="which clip frame to backproject for the scene PC: "
                         "'start' (clip first frame) or an integer clip-local index.")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate all selected objects into ONE config so they "
                         "render together (e.g. two parrots in one view).")
    ap.add_argument("--merge-name", default="objects",
                    help="obj_name for the merged config (with --merge)")
    ap.add_argument("--n-stamps", type=int, default=4)
    ap.add_argument("--dilate-px", type=int, default=2)
    ap.add_argument("--edge-blur", type=int, default=3)
    args = ap.parse_args()

    uc = UnifiedClip(args.dataset, args.root, args.file)
    objs = select_objects(uc, args.objs)
    # clip window = union of selected objects' chosen ranges
    s = min(s for _, (s, e) in objs)
    e = max(e for _, (s, e) in objs)
    if args.max_frames and (e - s + 1) > args.max_frames:
        e = s + args.max_frames - 1
        objs = [(o, (max(rs, s), min(re, e))) for o, (rs, re) in objs]
    n_clip = e - s + 1
    fps = uc.fps
    print(f"[{args.dataset}] {args.file}  caption={uc.caption!r}")
    print(f"  objects: {[(o, r) for o, r in objs]}  → clip window [{s}, {e}] ({n_clip} frames @ {fps:g}fps)")
    print(f"  depth {uc.depth_H}x{uc.depth_W}")

    out_data = args.out_dir / "static" / "data"
    out_video = args.out_dir / "static" / "videos"
    out_data.mkdir(parents=True, exist_ok=True)
    out_video.mkdir(parents=True, exist_ok=True)

    # color-sample frame (clip start), shared by all configs
    color_img = uc.rgb_at(s)

    configs = []
    for obj_name, (os_, oe_) in objs:
        cfg = build_config_for_object(uc, obj_name, s, e, color_img)
        # frame fields are already 0-based within the trimmed mp4 (t_local
        # indexes from s); `t` stays as the source start frame for provenance.
        configs.append(cfg)
        nvis0 = sum(1 for c in cfg["pt_colors_rgb"] if c is not None)
        print(f"    cfg obj={obj_name}: {len(cfg['gt_3d'][0])} pts, {nvis0} colored @ frame0")

    if args.merge and len(configs) > 1:
        configs = [merge_configs(configs, args.merge_name)]
        print(f"    merged → 1 config '{args.merge_name}': {len(configs[0]['gt_3d'][0])} pts")

    # ── scene point cloud (highest-res depth backprojection) ──
    if args.pc_frame == "start":
        pc_f = s
    else:
        pc_f = s + int(args.pc_frame)
    depth = uc.depth_at(pc_f)
    rgb = uc.rgb_at(pc_f)
    c2w = uc.c2w_at(pc_f)
    intr = uc.intr_at(pc_f)
    xyz, col = backproject_depth_to_world(depth, rgb, c2w, intr, subsample=args.pc_subsample)
    pc_path = out_data / f"{args.clip_id}_pc.bin"
    write_pc_binary(pc_path, xyz, col)
    print(f"  PC: {xyz.shape[0]} points (subsample={args.pc_subsample}) from clip-frame {pc_f - s} (src {pc_f})")

    # ── chronophotography (object stamps on last clip frame) ──
    cfg0 = configs[0]
    n_stamps = min(args.n_stamps, n_clip)
    stamp_local = sorted(set(int(round(i)) for i in np.linspace(0, n_clip - 1, n_stamps)))
    if stamp_local and stamp_local[-1] == n_clip - 1:
        stamp_local = stamp_local[:-1]
    stamp_src = [s + k for k in stamp_local]
    bg = grab_frame_bgr(uc.video_path, e)
    if bg.shape[0] != uc.depth_H or bg.shape[1] != uc.depth_W:
        bg = cv2.resize(bg, (uc.depth_W, uc.depth_H), interpolation=cv2.INTER_AREA)
    stamp_frames = [uc.rgb_at(f) for f in stamp_src]
    # per stamp, gather pixel-space 2D of ALL selected objects at that frame
    stamp_pts = []
    for k in stamp_local:
        pts = []
        for cfg in configs:
            for uv in cfg["gt_2d"][k]:
                if uv is not None:
                    pts.append((uv[0] * uc.depth_W, uv[1] * uc.depth_H))
        stamp_pts.append(pts)
    chrono = build_object_stamps_chrono(bg, stamp_frames, stamp_pts,
                                        dilate_px=args.dilate_px, edge_blur=args.edge_blur)
    chrono_path = out_video / f"{args.clip_id}_chrono.jpg"
    cv2.imwrite(str(chrono_path), chrono, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # ── trim mp4 ──
    mp4_dst = out_video / f"{args.clip_id}.mp4"
    trim_mp4_ffmpeg(uc.video_path, mp4_dst, s, e, fps=fps)

    # ── assemble bundle ──
    bundle = {
        "configs": configs,
        "n_configs": len(configs),
        "num_frames": n_clip,
        "fps": fps,
        "video_fps_mult": 1,
        "caption": uc.caption,
        "mse": 0.0,
        "l2": 0.0,
        "chrono": {
            "image_url": f"static/videos/{args.clip_id}_chrono.jpg",
            "frame_indices": stamp_local + [n_clip - 1],
            "mode": "object_stamps_on_last_frame_convex_hull",
            "dilate_px": args.dilate_px,
        },
        "pc_bin": {
            "url": f"static/data/{args.clip_id}_pc.bin",
            "n_points": int(xyz.shape[0]),
            "format": "uint32 N | float32 N*3 xyz | uint8 N*3 rgb",
            "n_concat_frames": 1,
            "subsample": args.pc_subsample,
            "frame_indices_original": [int(pc_f)],
        },
        "camera": {
            "c2w_frame0": np.asarray(c2w).astype(float).tolist(),
            "intrinsics_frame0": [float(x) for x in intr],
            "video_stem": args.file,
        },
        "source": {
            "release": str(args.root),
            "dataset": args.dataset,
            "file": args.file,
            "objects": [o for o, _ in objs],
            "src_window": [int(s), int(e)],
            "note": "GT training tracks from molmo-motion-1m; 2D derived by projecting 3D; "
                    "pred==gt (no model prediction in this release).",
        },
        "viewer_defaults": {
            "showPred": False,
        },
    }
    out_json = out_data / f"{args.clip_id}.json"
    out_json.write_text(json.dumps(bundle))
    print(f"  wrote {out_json}  ({out_json.stat().st_size // 1024} KB)")
    print(f"  wrote {mp4_dst}  ({mp4_dst.stat().st_size // 1024} KB)")
    print(f"  wrote {chrono_path}  ({chrono_path.stat().st_size // 1024} KB)")
    print(f"  wrote {pc_path}  ({pc_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
