# motion-teaser-viz2 — handoff

Single-file reference for everything done so far in this fork plus open
items. Read this first when picking up the project; the rest of the
docs (`high_level_idea.md`, `code_workthrough.md`, `problem.md`,
`README.md`) describe the upstream `motion-teaser-viz` and remain
accurate for the viewer architecture.

---

## 1. Live site

- Repo: <https://github.com/jianingnz/motion-teaser-viz2>
- GitHub Pages: <https://jianingnz.github.io/motion-teaser-viz2/>
- Local working copy: `/weka/prior-default/jianingz/home/visual/motion-teaser-viz2`
- Original (don't touch): `/weka/prior-default/jianingz/home/visual/motion-teaser-viz`
- Pages source: branch `main`, path `/`. First build was enabled via
  `gh api repos/jianingnz/motion-teaser-viz2/pages -X POST --field
  'source[branch]=main' --field 'source[path]=/'`. Builds usually live
  in 1–3 min after a push; hard-refresh (Cmd-Shift-R) once deployed.
- One large file warning at push time:
  `static/data/hot3d_clip1995_clip1996_gt3d.bin` (58.6 MB) exceeds
  GitHub's 50 MB soft limit but pushes fine. Consider migrating to Git
  LFS if the warning becomes annoying or the file grows further.

---

## 2. The 16 DROID clips

The DROID `<optgroup>` in [index.html](index.html#L425) lists exactly
these 16. Two distinct prediction sources, both reproducible via
`build/swap_droid_predictions.py`.

### Group A — rollout5 (8 clips, suffix `_object_tNN`)

Predictions from
`/weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results/
rollout5_droid_test_byclass_s*/predictions.jsonl`. Raw 3D only; we
swap pred_3d/pred_2d in place over the existing motion5-viz bundle's
gt + history.

| clip | t0 | task |
|---|---|---|
| `GuptaLab_553d1bd5_2023-04-20-12h-41m-59s_22246076_object_t33` | 33 | Move object |
| `PennPAL_c5f808b7_2023-10-09-21h-10m-27s_27085680_object_t14`  | 14 | Move object |
| `AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334_object_t27`  | 27 | Put / Place |
| `REAL_de601749_2023-06-17-11h-34m-24s_20540549_object_t25`     | 25 | Cover / Lid |
| `AUTOLab_0d4edc83_2023-10-21-20h-16m-31s_22008760_object_t26`  | 26 | Open / Close |
| `PennPAL_c5f808b7_2023-10-30-00h-50m-14s_25455306_object_t67`  | 67 | Open / Close |
| `CLVR_236539bc_2023-06-25-18h-35m-31s_20655732_object_t19`     | 19 | Hang / Unhang |
| `AUTOLab_84bd5053_2023-07-14-14h-56m-46s_24400334_object_t23`  | 23 | Stack |

### Group B — traj3d_v1 (8 clips, bare stem)

Predictions from
`/weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results/
traj3d_droid_v1_ft_f16_per_ds_droid/`. The corresponding bundle-shaped
JSONs live at
`/weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz/static/
data/modeling_json/droid/droid_v1_ft_f16/<stem>.json` — that's the
direct route the swap script uses (re-runs `prepare_clip_simple.py`).

| clip | t | task |
|---|---|---|
| `AUTOLab_0d4edc83_2023-10-21-19h-16m-18s_22008760` | 13 | Open / Close (eval calls it "Put brick in drawer shelf and close drawer") |
| `AUTOLab_0d4edc83_2023-10-21-19h-46m-27s_22008760` | 11 | Open / Close |
| `AUTOLab_0d4edc83_2023-10-21-20h-20m-00s_22008760` | 14 | Open / Close |
| `AUTOLab_0d4edc83_2023-11-03-15h-58m-46s_22008760` | 14 | Move object |
| `AUTOLab_0d4edc83_2023-11-03-16h-52m-04s_22008760` | 11 | Move object |
| `AUTOLab_44bb9c36_2023-11-23-19h-41m-33s_22008760` | 13 | Clean / Wipe (eval: "Take cloth pieces out of container.") |
| `AUTOLab_5d05c5aa_2023-07-07-18h-52m-04s_22008760` | 26 | Move object |
| `AUTOLab_5d05c5aa_2023-07-13-10h-59m-53s_24400334` | 23 | Put / Place |

### Captions / model prompts

Both eval result types store the prompt the model saw, just under
different field names — `caption` in rollout5, `expression` in
traj3d_v1. The 8 rollout5 bundles already carry the correct caption
(motion5-viz preprocess copied the same DROID
`aggregated-annotations-030724.json` text). The 8 traj3d_v1 bundles
do not have a `caption` field — if you want one shown in the page
header, run a stamper that copies `expression` into `caption`. The
helper was drafted as `build/stamp_droid_captions.py` but the user
reverted it; recreate from this README's table if needed.

### Two stems renamed (don't be surprised)

Earlier dropdown entries `…_object_t20` (44bb9c36) and `…_object_t24`
(5d05c5aa_07-13_24400334) were rolloff5 substitutions for missing-from-
motion5-viz bare-stem requests. Now that we use `droid_v1_ft_f16` for
both, the bare-stem clip-ids carry the prediction. The old files
were deleted from `static/`; the dropdown was updated.

---

## 3. Build pipeline (deterministic re-run order)

Everything below runs from the repo root with the `moge` env active:

```bash
source /weka/prior-default/jianingz/home/anaconda3/etc/profile.d/conda.sh
conda activate moge
cd /weka/prior-default/jianingz/home/visual/motion-teaser-viz2
```

| Step | Script | What it does |
|---|---|---|
| 1 | `python build/bake_droid16.py` | Bake all 16 DROID clips from scratch via `prepare_clip_simple.py`. mp4 source = raw 720p (`find_raw_mp4` resolves stem → run dir via `metadata_<UUID>.json`). |
| 2 | `python build/dense_pc_droid16.py` | Re-bake every `*_pc.bin` at stride-1 over the full 360×640 depth grid. ~210k pts/clip. |
| 3 | `python build/swap_droid_predictions.py` | One-shot pipeline: re-bake rollout5 group at 720p → swap pred from byclass → remove stale t20/t24 → re-bake traj3d_v1 from droid_v1_ft_f16 → dense PC re-run → stamp `viewer_defaults.show2DGt=false` on all 16. |
| 4 | `python build/regen_chrono_first_frame.py` | Replace `<clip>_chrono.jpg` with frame 0 of the served mp4 (no ghost stamps). DROID-only by default; pass `--all` to do every clip. |

Useful flags on `swap_droid_predictions.py`:
- `--skip-rollout5` skips Group A.
- `--skip-traj3d` skips Group B.
- `--skip-densify` skips the dense PC re-run.
- `--skip-viewer-defaults` skips the show2DGt stamp.

To reproduce the current site from scratch:
```bash
python build/swap_droid_predictions.py
python build/regen_chrono_first_frame.py
```

---

## 4. External data paths the build depends on

| What | Path |
|---|---|
| Raw DROID 1.0.1 release (1280×720 @ 60 fps mp4s) | `/weka/oe-training-default/jianingz/dataset/droid/1.0.1` |
| DROID depth h5 + camera intrinsics | `/weka/oe-training-default/chenhaoz/droid_pointworld/droid_all/<UUID>/{depth,cameras,*camframe.npz}` |
| motion5-viz source bundles (DROID rollout5) | `/weka/prior-default/jianingz/home/visual/motion5-viz/static/data/modeling_json/droid/test/` |
| motion3-viz source bundles (droid_v1_ft_f16) | `/weka/prior-default/chenhaoz/home/MotionPlanner/motion3-viz/static/data/modeling_json/droid/droid_v1_ft_f16/` |
| Eval result root | `/weka/prior-default/chenhaoz/home/MotionPlanner/molmo2/eval_results/` |
| `droid-rgb-viz` raw mp4 mirror (referenced by manifest) | `/weka/prior-default/jianingz/home/visual/droid-rgb-viz/` |

`find_raw_mp4(stem)` in [build/swap_droid_predictions.py](build/swap_droid_predictions.py)
and [build/bake_droid16.py](build/bake_droid16.py) walks
`/1.0.1/<lab>/{success,failure}/<date>/*/metadata_<UUID>.json` — DROID
run-dir names use a different timestamp format
(`Sat_Dec__9_15:45:51_2023`) than our stems
(`2023-12-09-15h-45m-51s`), which is why the metadata file is the
authoritative join key.

---

## 5. Viewer changes vs upstream `motion-teaser-viz`

### 5.1 Per-role colormap (GT vs Pred)

Settings keys: `cmapPreset` / `cmapReverse` (everyone except Pred),
`cmapPresetPred` / `cmapReversePred` (Pred only). `_useColormapFor(role)`
in [index.html:2114](index.html#L2114) routes `role==='pred'` to the
Pred pair, everything else to the global pair. Sidebar's
"Time-gradient palette" section now has two chip rows + two reverse
buttons. `_capturePreset` / `_applyPreset` and `_applyPalette` were
updated to capture/restore both halves; chip-active state is scoped to
the matching row only.

### 5.2 Endpoint spheres

- Pred-end coloured separately from GT-end so they don't fully occlude
  each other when the prediction lands on the GT — Pred uses the Pred
  trail color (Pred LUT[1.0] in colormap mode, else `TRAIL_COLORS.pred`).
  Per-instance color is carried in `records[i][3..5]` and assigned at
  push() time.
- For non-HOT3D bundles (EgoDex / DROID / HD-EPIC / DAVIS) only the
  **start** marker is drawn (frame 0). End markers were removed
  because they appeared at the future end mid-playback and stacked on
  top of each other when Pred was close to GT.
- Endpoints are filtered to only the tracks that have ≥2 valid frames
  in the same `[0, lastFi+1]` window the trail builder uses, so a
  filtered-out track no longer leaves orphaned spheres in the scene.

### 5.3 Track picker

[index.html ≈5503](index.html#L5503) — `_enableMultiSelectClickToggle`
and `fillTrajectoryPicker`.

- Plain click in the GT/Raw picker = single-select (replaces the
  selection so picking one track shows just that one).
- Modifier click (Ctrl / Cmd / Shift) = toggle (multi-select).
- `sel.focus()` was removed in the mousedown handler; Chrome's auto-
  scroll-to-first-selected-option triggered on focus arrival was the
  cause of the user-reported "jumps to start" complaint.
- `scrollTop` is pinned across click using a synchronous restore +
  `requestAnimationFrame` + `setTimeout(…, 0)` — the rAF + timeout
  defeat Chrome's post-layout snap that runs after our handler
  returns.
- A `focus` listener on the picker also re-pins to a tracked
  `lastUserScroll` so Tab-into-picker keyboard nav stays put.
- `fillTrajectoryPicker` saves and restores `scrollTop` across the
  `innerHTML = ''` rebuild so refresh paths (n-keep slider, ↺ reset)
  don't bounce the listbox either.

### 5.4 2D overlay knobs (panels ① + ②)

`Settings.video2D` (bag of numbers) drives both `drawVideoOverlay`
(panel ①) and `drawChronoOverlay` (panel ②):

| key | default | meaning |
|---|---|---|
| `gtWidth` | 1.5 | GT trail line width (px) |
| `gtDotR` | 4.2 | GT current-frame soft dot radius |
| `gtAlpha` | 1.0 | multiplier on the 0.55→0.95 GT pale-vivid ramp |
| `predWidth` | 1.5 | Pred trail line width |
| `predDashOn` | true | Pred trail dashed `[4, 3]` (false → solid) |
| `predDotR` | 5.0 | Pred ring radius |
| `predDotW` | 1.6 | Pred ring stroke width |
| `predAlpha` | 1.0 | multiplier on the 0.55→0.95 Pred ramp |

Sidebar section: "2D overlays". `strokeGradientPath` grew an
`alphaMul` parameter; `ringDot` grew a `lineWidth` parameter. Both
default to the pre-change hard values so the chrono / 3D-panel call
sites that did not get an explicit value remain byte-identical.

Colours still come from the global `col-gt` / `col-pred` pickers via
`TRAIL_RGB`; if you want per-overlay colours, split `Settings.video2D`
into `videoOverlay` and `chronoOverlay` halves.

### 5.5 Hide GT in 2D overlays

`Settings.show2DGt` (default true) gates the GT trail + GT dots in
both `drawVideoOverlay` and `drawChronoOverlay`. The 3D panels keep
showing GT via the existing `t-gt` toggle, so the two scopes are
independent.

`applyBundleViewerDefaults` honours `viewer_defaults.show2DGt`.
`build/swap_droid_predictions.py` stamps `show2DGt: false` on every
DROID bundle as the final step, matching the user's "only show the
pred we have on the 2D overlay" ask.

### 5.6 Save / Load full config (single JSON)

Two buttons under the localStorage preset row in the sidebar:

- 💾 **Save config** → downloads `mt_config_<clip>_<timestamp>.json`.
- 📂 **Load config** → file picker, parses, applies.

Captures: `clip_url`, `obj`, `inputs` (every PRESET_INPUTS slider /
colour / dropdown value), `toggles` (every PRESET_TOGGLES on/off
state), `cmap` (global GT preset + Pred preset + reverse flags +
intensity), `camera` (yaw/pitch/dist/target xyz), `picker` (gt + raw
selected indices), `pc_hidden` (click-to-hide PC index set).

Apply order is chosen so each later step can overwrite the previous
one: clip URL first (page reload via `sessionStorage` stash if it
differs), then inputs (n-keep re-runs `recomputeGtIdx`, but that's
overwritten by the saved picker selection two steps later), toggles,
cmap, camera, picker, pc_hidden, finally `rebuildAllStatic()` so the
hidden-PC + camera changes paint.

When the saved clip differs from the current page, the rest of the
config is stashed in `sessionStorage` (`mt_pending_config_v1`) and we
bounce through `location.href` so the bundle has time to re-hydrate;
the receiving end picks up the stash inside `bindConfigSaveLoad` on
the next animation frame, after `init()` has finished populating
`cfg` + `clipData`.

### 5.7 Panel ② chrono = first-frame-only background

`build/regen_chrono_first_frame.py` overwrites every DROID
`<clip>_chrono.jpg` with `frame 0` of the served mp4. The convex-hull
ghost stamps are gone — the prediction trail handles the storytelling.

---

## 6. Settings keys reference (post-changes)

```
Settings.cmapPreset        | cmapReverse        | cmapIntensity   — global colormap
Settings.cmapPresetPred    | cmapReversePred                      — Pred-only colormap override
Settings.show2DGt                                                  — gates GT in 2D overlays
Settings.video2D.gtWidth   | gtDotR | gtAlpha                      — 2D overlay GT knobs
Settings.video2D.predWidth | predDotR | predDotW
                           | predAlpha | predDashOn                — 2D overlay Pred knobs
Settings.endpointColor     | endpointScale | endpointAlpha         — start-marker spheres
```

`viewer_defaults` schema honoured by `applyBundleViewerDefaults`:
`showPC`, `showGT`, `showPred`, `showRaw`, `showBalls`,
`showEndpoints`, `show2DGt`, `objMaskRadiusPx`, `maxSceneDepth`,
`objectCloud`, `objCloudPointPx`, `trackSmoothWindow`,
`staticObjFrameMode`. Unrecognised fields are silently ignored.

---

## 7. Build scripts on disk (`build/`)

| File | Purpose |
|---|---|
| `prepare_clip_simple.py` | Generic clip-bundle builder (upstream). Trims mp4, samples colours, writes JSON + `_pc.bin` + chrono.jpg. Accepts `--src-json`, `--src-mp4`, `--out-dir`, `--clip-id`. |
| `prepare_clip.py` | Older, vipe-aware variant (upstream — unused for DROID). |
| `prepare_full_video.py`, `prepare_hot3d.py`, `prepare_hdepic.py` | Other dataset preps (upstream). |
| `build_scene_moge_lastframe.py`, `build_scene_monst3r.py` | Alternate scene-PC sources (upstream). |
| `rebuild_pc_droid_dense.py` | Stride-1 dense-PC tool for one DROID clip. Reads depth h5 + camframe.npz, projects, writes `_pc.bin`, patches `pc_bin.n_points` in served JSON. |
| `regen_hot3d_dense_scene_pc.py`, `regen_hot3d_dense_tracks.py` | HOT3D-only (upstream). |
| `extract_hires_frames.py` | EgoDex + HOT3D paper figures (upstream). |
| `pick_best_clip.py` | Old helper (upstream). |
| **`bake_droid16.py`** | Driver that bakes all 16 DROID clips end-to-end via `prepare_clip_simple`. Sources mp4 from raw DROID 1.0.1 (720p) via `find_raw_mp4`. |
| **`dense_pc_droid16.py`** | Driver that runs `rebuild_pc_droid_dense.py` for all 16 DROID clips with stride-1 + no max-points cap. Passes `--uuid`/`--cam` explicitly so non-AUTOLab labs work. |
| **`swap_droid_predictions.py`** | The full one-shot pipeline. See §3. |
| **`regen_chrono_first_frame.py`** | Replaces `<clip>_chrono.jpg` with frame 0 of the served mp4. |

Bold = added in this fork.

---

## 8. Things still open / parking lot

- **Captions for the 8 traj3d_v1 clips**. They have an
  `expression` field in the eval row but no `caption` in the served
  bundle. Stamper script (`build/stamp_droid_captions.py`) was
  drafted then reverted at user request — they wanted the text only,
  not a site change. Re-create the script from §2's table if you
  want the page header to show the prompt for those 8 too.
- **Per-overlay colour overrides**. `Settings.video2D` shares
  colour with the global `col-gt` / `col-pred` pickers. If panel ①
  needs its own palette, split `video2D` into `videoOverlay` /
  `chronoOverlay` halves (or add `colorGt` / `colorPred` to each).
- **`endpointColorPred`**. Currently the Pred-end sphere colour is
  derived from the Pred trail (LUT[1.0] or `TRAIL_COLORS.pred`).
  Splitting it from the GT yellow into its own user-pickable swatch
  is a 5-line sidebar addition + one Setting key.
- **Re-bake other datasets at 720p**. EgoDex / HD-EPIC / HOT3D
  bundles still ship at whatever resolution motion5-viz produced.
  Only DROID was touched.
- **Git LFS** for `static/data/hot3d_clip1995_clip1996_gt3d.bin`
  (58.6 MB) — over GitHub's 50 MB warning threshold. Optional.
- **Track picker scroll bug**. The user has reported one more time
  that picker still bounces; the latest fix
  (focus-drop + triple-pin + focus-listener) shipped in commit
  `eee9609`. If reports persist, instrument the listbox with a
  scroll-event log to find which pass is moving it.

---

## 9. Conventions & gotchas

- **No Claude coauthor footer in commits**. The user rejects the
  default `Co-Authored-By: Claude ...` line. Saved as a memory at
  `~/.claude/projects/-weka/memory/feedback_no_claude_coauthor.md`.
- **No wide `/weka` filesystem scans**. Pin to specific deep paths
  or ask. `find /weka ...` from the root or recursive grep over the
  whole filesystem is too slow. Saved as a memory.
- **DROID frame conventions**: bundle `cfg.t` = last hist frame =
  first future frame's index − 1. So `_object_t27` means
  `hist_frames=[25,26,27]`, `future_frames=[28..]`. The eval result
  rows expose this as `t0` (rollout5) or `t` (traj3d_v1) — same
  semantics, different field name.
- **Bundle gt_3d frame**: OpenCV camera frame (X right, Y down, Z
  forward). The viewer renders at `[x, -y, -z]` to convert to
  three.js's right-handed Y-up. Projection through K from
  `<UUID>/<CAM>_smoothed_camframe.npz` recovers gt_2d to ~1×10⁻⁴
  rms in normalised image space — confirmed via least-squares in
  `swap_droid_predictions.fit_projection`.
- **`prepare_clip_simple.py` does not resize the mp4**. It only
  trims and (when `video_fps_mult > 1`) sub-samples the temporal
  stream. Output mp4 inherits source resolution — that's why pinning
  `--src-mp4` to the raw 720p DROID was the right move for high-res
  panel ①.
- **`pred[0..n_hist] == gt[0..n_hist]` invariant**. Held across all
  16 DROID clips. The viewer renders Pred from frame 0 to lastFi,
  so the prediction trail visually starts on the GT history before
  diverging. Don't break this when adding new bundles.

---

## 10. How to pick this up next session

1. `cd /weka/prior-default/jianingz/home/visual/motion-teaser-viz2`
2. `git pull` (or `git status` if you've been editing locally).
3. `cat HANDOFF.md` (this file).
4. If you need to re-bake everything:
   ```bash
   source /weka/prior-default/jianingz/home/anaconda3/etc/profile.d/conda.sh
   conda activate moge
   python build/swap_droid_predictions.py
   python build/regen_chrono_first_frame.py
   git add -A && git commit -m "<msg>" && git push mtv2 main
   ```
5. Pages live at <https://jianingnz.github.io/motion-teaser-viz2/>
   within ~1–3 min. Hard-refresh.
