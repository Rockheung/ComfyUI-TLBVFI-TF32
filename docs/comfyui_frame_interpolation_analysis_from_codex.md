# ComfyUI Frame Interpolation: RIFE vs FILM

Research snapshot of the memory management, caching, and ComfyUI integration strategies used in **ComfyUI-Frame-Interpolation** for the RIFE and FILM VFI nodes. Sources reference commit history as of 2024-11-21.

## Repository Overview

- Repo: `https://github.com/Fannovel16/ComfyUI-Frame-Interpolation`
- Key modules:
  - `vfi_utils.py`: shared utilities (model downloads, frame loops, cache handling)
  - `vfi_models/rife/__init__.py`: RIFE node entry point
  - `vfi_models/rife/rife_arch.py`: IFNet architecture
  - `vfi_models/film/__init__.py`: FILM node entry point

---

## 1. RIFE VFI Implementation

### Model Loading & Caching

- `RIFE_VFI.vfi` downloads the checkpoint via `load_file_from_github_release`, then instantiates IFNet and loads weights every invocation (`vfi_models/rife/__init__.py:89-104`).
- Checkpoints cached on disk under `./ckpts/rife/…` (shared helper in `vfi_utils.py:62-109`).
- No resident in-memory model cache; relies on Python re-instantiation for each workflow run.

### ComfyUI Integration

- Pulls active device through `comfy.model_management.get_torch_device` (`vfi_models/rife/__init__.py:94`), ensuring it honours Comfy’s global device choice.
- Registered in `NODE_CLASS_MAPPINGS` with label “RIFE VFI (recommend rife47 and rife49)” (`__init__.py:29-43`).

### Frame Processing Pipeline

```
frames (IMAGE tensor)
 └─ preprocess_frames → NCHW (CPU)  (`vfi_utils.py:114-118`)
 └─ generic_frame_loop (CPU orchestrator) (`vfi_utils.py:124-207`)
     ├─ for each pair (frame_i, frame_{i+1})
     │    ├─ send to DEVICE as float32
     │    ├─ run IFNet with timestep per middle frame
     │    └─ detach to CPU immediately
     └─ preallocated CPU tensor stores results
 └─ postprocess_frames → NHWC (CPU) (`vfi_models/rife/__init__.py:95-107`)
```

- Processing is strictly sequential per frame pair; middle frames computed one timestep at a time unless list-based multiplier is provided.

### Memory Management

- Middle-frame tensors detached and moved to CPU right after inference (`vfi_utils.py:170-184`).
- `clear_cache_after_n_frames` argument increments a counter and calls `soft_empty_cache()` once threshold is hit (`vfi_utils.py:186-206`).
- Uses `gc.collect()` every iteration to release Python references.
- `backwarp_tenGrid` holds cached sampling grids keyed by device/shape, reducing repeated allocations (`vfi_models/rife/rife_arch.py:16-70`).
- No `torch.no_grad()` guard around model invocation, so autograd metadata is briefly created before `.detach()`—slightly higher VRAM overhead than necessary.

---

## 2. FILM VFI Implementation

### Model Loading & Caching

- Weights provided as TorchScript (`film_net_fp32.pt`).
- Loaded via `torch.jit.load(..., map_location='cpu')` then moved to Comfy’s device (`vfi_models/film/__init__.py:63-77`) to avoid GPU spikes during load.
- Like RIFE, no in-process cache; model is reloaded for each run.

### ComfyUI Integration

- Uses `get_torch_device` & `soft_empty_cache` from `comfy.model_management` but no advanced lifecycle hooks (`vfi_models/film/__init__.py:2`).
- Node registered as “FILM VFI” in `NODE_CLASS_MAPPINGS` (`__init__.py:32`).

### Frame Processing Pipeline

```
frames (IMAGE tensor)
 └─ preprocess_frames → NCHW
 └─ loop pairs (no generic_frame_loop reuse)
     ├─ convert to DEVICE float32 tensors
     ├─ run TorchScript FILM under torch.no_grad() (`vfi_models/film/__init__.py:31-36`)
     ├─ results inserted using bisect strategy (`vfi_models/film/__init__.py:12-42`)
     └─ detach & move to CPU list right away
 └─ concat CPU frames at end → NHWC postprocess
```

- Supports either scalar or per-pair multiplier arrays; sequential evaluation without tensor preallocation.

### Memory Management

- `torch.no_grad()` wraps each inference call (contrast to RIFE).
- Results clipped, detached, and moved to CPU before being stored (`vfi_models/film/__init__.py:92-107`).
- Same `clear_cache_after_n_frames` counter calling `soft_empty_cache` (`vfi_models/film/__init__.py:97-112`).
- Explicit `gc.collect()` each iteration; final `soft_empty_cache()` after concatenation.

---

## 3. Integration Patterns with ComfyUI

- Both nodes rely on Comfy’s model management only for device selection and soft cache clearing; neither uses global model caching or context hooks.
- Model assets live under module-local `./ckpts/<model_type>` path so downloads are reused automatically between runs.
- Frame skipping handled via `InterpolationStateList` objects passed as optional inputs (`vfi_utils.py:27-59`), enabling workflow-driven control without extra state.
- Each workflow execution reloads the model from disk and reinitializes architecture classes.

---

## 4. Memory Management Deep Dive

| Aspect | RIFE | FILM |
|---|---|---|
| VRAM residency | Only current frame pair + model; outputs immediately moved to CPU (`vfi_utils.py:147-184`) | Same, but no preallocated buffer (list accumulation on CPU) |
| Cache clearing | Manual counter, calls `soft_empty_cache` (`vfi_utils.py:186-206`) | Identical counter mechanism (`vfi_models/film/__init__.py:97-112`) |
| CPU↔GPU transfers | `.to(DEVICE)` per frame pair; `.detach().cpu()` back (`vfi_utils.py:170-184`) | Mirror behaviour with `torch.no_grad()` wrapper (`vfi_models/film/__init__.py:92-107`) |
| Batch vs streaming | Sequential; preallocates CPU tensor sized for multiplier × frames (`vfi_utils.py:147-207`) | Sequential; streams into list, final `torch.cat` |
| Cache clearing granularity | User-tunable `clear_cache_after_n_frames` (default 10) (`vfi_models/rife/__init__.py:41-87`) | Same default and controls |
| OOM mitigation | Reliant on user tuning + frequent `gc.collect()` | Same; TorchScript load on CPU avoids GPU spikes |

Other notable safeguards:

- RIFE keeps reusable flow grid cache per resolution to prevent repeated grid computation (`vfi_models/rife/rife_arch.py:16-70`).
- Both nodes end with a courtesy `soft_empty_cache()` after final output.

---

## 5. Comparison vs. TLBVFI (Our Node)

| Topic | ComfyUI RIFE/FILM | Our TLBVFI (`tlbvfi_node.py`) |
|---|---|---|
| Model caching | Reloads every run; no global cache | Loads once per execution, then explicitly unloads at end (`tlbvfi_node.py:304-479`) |
| Cache clearing strategy | Fixed counter using `soft_empty_cache` | Adaptive `calculate_cleanup_interval` uses actual VRAM stats to set cleanup cadence (`tlbvfi_node.py:86-177`) |
| CPU streaming | Immediate CPU detach per frame, but RIFE preallocates giant tensor | Streams in batches with non-blocking CPU transfer; batch size determined by available RAM (`tlbvfi_node.py:182-215`, `tlbvfi_node.py:399-459`) |
| GPU cleanup | `soft_empty_cache` only | Uses `torch.cuda.empty_cache` after synchronization and unloads model to CPU (`tlbvfi_node.py:431-482`) |
| Autograd handling | RIFE lacks `torch.no_grad()` | TLBVFI wraps sampling in `torch.no_grad()` (`tlbvfi_node.py:402-405`), matching FILM’s safer pattern |

### Lessons & Best Practices

1. **Detaching early matters**: Both models immediately move outputs to CPU, keeping VRAM steady. Our streaming pipeline is aligned, but we can ensure every mid-frame path uses `no_grad()` for parity with FILM.
2. **Expose manual cache interval**: RIFE/FILM expose an easy `clear_cache_after_n_frames` knob. Adding an override on top of our adaptive cleanup would give power users more control.
3. **Model load optimisation**: RIFE/FILM re-instantiate weights each run. We could improve our own cold start by adding a module-level cache keyed by model path (or use Comfy’s `model_management` registries) while still supporting explicit unloads.
4. **TorchScript deployment**: FILM’s TorchScript packaging makes loading simpler and CPU-friendly. Investigating scripted/safe-load variants of TLBVFI could reduce start-up overhead.

### Recommended Actions for TLBVFI

1. Add a `clear_cache_after_n_segments` user input that bypasses the auto-calculated interval when set, mirroring ComfyUI-Frame-Interpolation behaviour.
2. Introduce `torch.no_grad()` guards wherever we call `model.sample`, ensuring we never leak grad metadata in long runs.
3. Evaluate a lightweight model cache (dict keyed by `model_path`) to reuse loaded weights across workflow executions when running inside the same Comfy session.
4. Consider a TorchScript export of the inference path to speed future loads, inspired by FILM’s single-file deployment.

---

## File References

- `vfi_models/rife/__init__.py:31-107`
- `vfi_models/rife/rife_arch.py:1-206`
- `vfi_models/film/__init__.py:1-113`
- `vfi_utils.py:14-274`
- `__init__.py:13-44`
- `tlbvfi_node.py:1-493`

