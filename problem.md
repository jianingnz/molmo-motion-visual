# Mistakes / bugs / solutions log

> Append-only. Each entry: short title, what was wrong, why, the fix.
> Read this before debugging anything in this repo so we don't re-run
> the same wall-bumps.

## 2026-05-04 — extract_hires_frames.py emitted wrong JSON URL prefix
**Symptom**: After the first run, the bundle JSON's `clip_frames_hires[].url`
came out as `videos/<clip_id>_clipframes/00.jpg`. The viewer expects URLs
relative to the repo root (it serves `static/...` paths everywhere else),
so loading the page would 404 the strip frames.
**Root cause**: `write_jpgs` computed `path.relative_to(out_dir.parent.parent)`,
which evaluated to `out_dir / static / videos / ..` → only `videos/...`
because `out_dir` was already the repo root. The intent was repo-root
relative.
**Fix**: pass `repo_root` explicitly into `write_jpgs` and compute
`path.relative_to(repo_root)` (`build/extract_hires_frames.py:125-136`).
**Prevent**: when generating URLs for the JSON bundle, always derive them
from the explicit repo root, never from the output directory's `parent`s.

## 2026-05-04 — first EgoDex full-strip run failed on out-of-bounds frame
**Symptom**: `failed to read frame 302 of /weka/.../egodex/.../1713.mp4`
(the source mp4 has 300 frames; vipe-domain → 30-fps index calculation
generated 302).
**Root cause**: The vipe-domain `n_frames` from the bundle (152) maps to a
30-fps source range of `[0, 2·151] = [0, 302]`, but the source mp4 actually
has frame indices `[0, 299]`. We ignored that the source mp4 may run a
half-frame short of `2 × n_vipe`.
**Fix**: clamp the chosen 30-fps src indices to `[0, n_src - 1]` after
opening the source mp4 and reading `CV_CAP_PROP_FRAME_COUNT`
(`build/extract_hires_frames.py` — see `build_egodex_full_strip`).
**Prevent**: any time we map between fps domains for a finite-length mp4,
read the source's `CAP_PROP_FRAME_COUNT` and clamp before seeking.

## 2026-05-11 — repo combined from motion-teaser-viz + motion-teaser-viz2
**Symptom**: not a bug — record that this repo (`molmo-motion-visual`) was
created by combining two predecessor repos so future readers know where to
look for older history.
**Source**: base = `motion-teaser-viz2` (newer JS, has DROID-specific build
scripts and the full 17-clip DROID example set). Patched in: viz1's
**non-DROID** examples (EgoDex `clean_surface_3603_t12`; DAVIS
`flamingo_flamingo_t2`, `car-turn_silver_car_t2`; HOT3D `clip002658`,
`clip003137`, `clip002562`) plus viz1-only build scripts
(`bake_motion5_pred_into_hot3d.py`, `prepare_davis_singleclip.py`).
**Prevent**: when adding a new example, drop its data/video files in
`static/data` and `static/videos` and add one `<option>` in `index.html`
under the right `<optgroup>` — pattern-based dataset detection in the JS
will pick it up automatically.

## 2026-06-04 — adding examples from the unified `molmo-motion-1m` release
**Symptom**: not a bug — record the non-obvious facts about the unified
training-data release at `/weka/oe-training-default/jianingz/molmo-motion-1m`
so future adds don't re-derive them. New builder: `build/prepare_unified.py`.
**Gotchas verified empirically (trust reprojection over the release README)**:
- **Camera pose is cam→world (c2w), NOT world→camera.** The top-level README
  labels egodex/ytvis pose `data` as "world-to-camera", but projecting a 3D
  track with `K @ inv(pose)` reproduces the released 2D *exactly* only when
  `pose` is treated as **c2w**. Same for hepic (`pose/`) and molmospaces
  (`cam_poses`). Backproject with `world = pose @ cam_pt`.
- **3D track NPZ layout varies.** egodex `tracks/object/{id}_3d.npz` →
  `points_3d (K,T,3)` flat array. ytvis/hepic `tracks/{id}_3d.npz` →
  `points_3d` is a **0-d object array** wrapping `{obj_name: (K,T,3)}` (use
  `.item()`). molmospaces lives entirely in `camera/{slug}.npz`
  (`points_3d (T,K,3)` — T-first! — + `body_ids`, `cam_poses`, `intrinsics`,
  `depth_frames`); the `tracks/*_3d.npz` are empty 0-d stubs.
- **hepic shipped 2D ≠ 3D point set.** `_2d.npz tracks` is a single
  AllTracker grid `(T,N,2)`; the 3D is a separate per-object dict. They do not
  correspond. **Derive 2D by projecting 3D** through (pose, intrinsics) — this
  is what `prepare_unified.py` does for ALL datasets, giving a 2D set aligned
  1:1 with the 3D (needed for color sampling / chrono / overlay).
- **Invisible points carry garbage 3D** (e.g. hepic `-20522`). Always gate on
  the per-point per-frame visibility mask; emit `null` for invisible samples.
  The viewer tolerates `null` (HOT3D sparse-stub path: `trailVisible` /
  `pointMaxJumpScores` both guard `Number.isFinite`).
- **This release is GT-only** (no model prediction). Bundles set
  `pred_3d == gt_3d` + `viewer_defaults.showPred=false` to show training
  tracks cleanly.
- **Use the right conda env**: `gentraj` (has OpenEXR + cv2 + imageio_ffmpeg).
**Prevent**: when adding more examples, run `build/prepare_unified.py
--dataset <ds> --file <video_id> --clip-id <id>` (add `--objs all --merge` for
multi-object scenes, `--max-frames N` to cap long motion ranges), then add one
`<option>` to `index.html` under the right `<optgroup>`.

## 2026-06-04 — "highest-resolution PC" rebake
**Symptom**: HD-EPIC PCs were 25.6 K pts (built at 160×160), DAVIS PCs 3–22 K.
**Fix**: HD-EPIC re-generated from unified hepic depth at full 512² (262 K pts,
stride-1) via `prepare_unified.py` (PC + tracks together → guaranteed aligned).
DAVIS re-baked to stride-1 854×480 (409 K pts) via `rebake_pc_stride1.py` after
patching `camera.video_stem` + `pc_bin.frame_indices_original=[0]` into the
legacy bundles (their PCs were "pre-baked from motion5-viz", no camera meta).
Verified the vipe world frame == the DAVIS track frame (tracks fall inside the
backprojected bbox) before swapping. EgoDex/DROID/molmospaces/ytvis already at
native max (≈stride-1). HOT3D left at stride-3 (~221 K): full 1408² ≈ 2 M
pts/clip is impractical for the web viewer and HOT3D is not part of this
training release.
**Prevent**: `prepare_unified.py` bakes at `--pc-subsample 1` by default;
`rebake_pc_stride1.py` needs `camera.video_stem` + `pc_bin.frame_indices_original`.

### Template
```
## <YYYY-MM-DD> — short title
**Symptom**: what looked wrong from the outside.
**Root cause**: the actual bug.
**Fix**: code change made (file:line).
**Prevent**: what to look for next time so it isn't re-introduced.
```
