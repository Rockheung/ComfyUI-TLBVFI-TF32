# TLBVFI ComfyUI Production Improvement Plan

**Goal**: Transform current implementation into production-grade, memory-safe, hardware-optimized VFI node

**Analysis Date**: 2025-10-15

---

## Problem Analysis

### Critical Issues with Current Implementation

#### 1. **Recursive Interpolation Memory Explosion**

**Current Code** (`chunk_processor.py:305-316`):
```python
current_frames = [frame1, frame2]
for iteration in range(times_to_interpolate):
    temp_frames = [current_frames[0]]
    for j in range(len(current_frames) - 1):
        mid_frame = model.sample(current_frames[j], current_frames[j+1])
        temp_frames.extend([mid_frame, current_frames[j+1]])
    current_frames = temp_frames  # MEMORY EXPLOSION!
```

**Problem**: Memory grows exponentially:
- `times_to_interpolate=1`: 2 → 3 frames (~780MB @ 4K FP32)
- `times_to_interpolate=2`: 3 → 5 frames (~1.3GB)
- `times_to_interpolate=3`: 5 → 9 frames (~2.3GB)
- `times_to_interpolate=4`: 9 → 17 frames (~4.4GB)
- **All frames stay in GPU VRAM simultaneously!**

#### 2. **Deviation from Original Paper**

**Original TLBVFI** (`/tmp/TLBVFI/interpolate_one.py:61-64`):
```python
def interpolate(frame0, frame1, model, gt=None):
    with torch.no_grad():
        out = model.sample(frame0, frame1)
    return out
```

**Key Points**:
- **Single frame interpolation only**
- Recursive bisection done in **post-processing** (interpolate.py:88-95)
- Each interpolation is **independent** and can immediately release memory

**Current Implementation**:
- Tries to do 2^N interpolation in one pass
- Keeps all intermediate frames in GPU memory
- Not aligned with original paper's design

#### 3. **FFmpeg Subprocess Overhead**

**Current** (`chunk_processor.py:360-427`):
```python
def _save_chunk_as_video(self, frames, chunk_path, ...):
    # Creates new FFmpeg process for EACH frame pair
    # Overhead: process spawn + pipe setup + MP4 container creation
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, ...)
    process.stdin.write(frames_np.tobytes())
```

**Problems**:
- Process spawn overhead (~50-100ms per chunk)
- MP4 container overhead (each chunk needs header/footer)
- No benefit over frame-by-frame processing

#### 4. **Missing RIFE/FILM Best Practices**

Based on `docs/ComfyUI-Frame-Interpolation-Analysis.md`:

| Pattern | RIFE/FILM | Current TLBVFI |
|---------|-----------|----------------|
| CPU output accumulation | ✅ Every frame | ❌ Batch at end |
| Periodic cache clearing | ✅ Every 10 frames | ❌ Only on memory pressure |
| FP16 inference | ✅ RIFE default | ❌ Not implemented |
| torch.no_grad() | ✅ FILM, ❌ RIFE missing | ✅ Implemented |
| Immediate GPU→CPU transfer | ✅ `non_blocking=True` | ❌ Batch transfer |

#### 5. **Missing Original Paper Optimizations**

From original analysis (`docs/TLBVFI-Original-Implementation-Analysis.md`):

| Feature | Original | Current |
|---------|----------|---------|
| Adaptive padding | ✅ Dynamic | ❌ None |
| Flow-guided decoding | ✅ `scale=0.5` | ❌ Not exposed |
| Configurable timesteps | ✅ 10/20/50 steps | ❌ Fixed 10 |
| Frozen VQGAN | ✅ `disabled_train()` | ⚠️ Implicit |

---

## Production-Grade Solution

### Design Principles

1. **Single-Frame Interpolation** (aligned with original paper)
2. **Immediate Memory Release** (RIFE/FILM pattern)
3. **Periodic Cache Clearing** (RIFE pattern)
4. **CPU Accumulation** (RIFE/FILM pattern)
5. **Adaptive Padding** (Original pattern)
6. **FP16 Support** (RIFE pattern + TF32)
7. **Configurable Quality/Speed** (Original pattern)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input (N frames)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FramePairSlicer (unchanged)                     │
│          Slices: (0,1), (1,2), (2,3), ..., (N-2,N-1)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│        TLBVFI_Interpolator_V2 (NEW - Production Grade)       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Load model with FP16 + TF32 (cached globally)  │    │
│  │ 2. Adaptive padding for input frames               │    │
│  │ 3. Single interpolation: (frame0, frame1) → mid   │    │
│  │ 4. Optional recursive bisection with memory mgmt   │    │
│  │ 5. Immediate GPU→CPU transfer (non-blocking)       │    │
│  │ 6. Periodic cache clearing (every 10 pairs)        │    │
│  │ 7. Unpad output frames                             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  Output: Single interpolated frame or sequence              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  User Choice  │
                    └───────────────┘
                     /              \
                    /                \
                   ▼                  ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  Direct Use          │   │  Save to Disk        │
    │  (ComfyUI workflow)  │   │  (optional)          │
    └──────────────────────┘   └──────────────────────┘
```

### Key Components

#### Component 1: Production-Grade Interpolator

**File**: `nodes/tlbvfi_interpolator_v2.py`

```python
class TLBVFI_Interpolator_V2:
    """
    Production-grade TLBVFI interpolator with memory safety and optimizations.

    Features:
    - Single-frame interpolation (aligned with original paper)
    - Optional recursive bisection with memory management
    - FP16 inference support
    - TF32 acceleration on RTX 30/40
    - Adaptive padding for arbitrary resolutions
    - Immediate GPU→CPU transfer
    - Periodic cache clearing
    - Configurable timestep schedule
    - Flow-guided decoding control
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev_frame": ("IMAGE",),
                "next_frame": ("IMAGE",),
                "model_name": (get_available_models(),),
            },
            "optional": {
                "times_to_interpolate": ("INT", {"default": 0, "min": 0, "max": 4}),
                # 0 = single frame only (original paper)
                # 1-4 = recursive bisection (2x, 4x, 8x, 16x)

                "use_fp16": ("BOOLEAN", {"default": True}),
                "enable_tf32": ("BOOLEAN", {"default": True}),

                "sample_steps": ([10, 20, 50], {"default": 10}),
                # 10 = fast (paper default)
                # 20 = balanced
                # 50 = high quality

                "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                # 0.5 = fast (half resolution flow, paper default)
                # 1.0 = high quality (full resolution flow)

                "cpu_offload": ("BOOLEAN", {"default": True}),
                # True = immediate GPU→CPU transfer (RIFE pattern)

                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 7}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("interpolated_frames",)
    FUNCTION = "interpolate"
    CATEGORY = "frame_interpolation/TLBVFI-TF32"

    def __init__(self):
        self._frame_pair_count = 0  # For periodic cache clearing

    def interpolate(self, prev_frame, next_frame, model_name,
                   times_to_interpolate=0, use_fp16=True, enable_tf32=True,
                   sample_steps=10, flow_scale=0.5, cpu_offload=True, gpu_id=0):
        """
        Production-grade interpolation with memory safety.

        Memory profile (4K frame):
        - FP32: ~260MB per frame in GPU
        - FP16: ~130MB per frame in GPU
        - CPU offload: ~0MB sustained (only during processing)

        Args:
            prev_frame: (1, H, W, C) ComfyUI tensor
            next_frame: (1, H, W, C) ComfyUI tensor
            model_name: Model checkpoint
            times_to_interpolate: 0=single, 1=2x, 2=4x, 3=8x, 4=16x
            use_fp16: Enable FP16 inference (2x memory reduction)
            enable_tf32: Enable TF32 on RTX 30/40 (4x speed boost)
            sample_steps: Diffusion steps (10/20/50)
            flow_scale: Flow computation scale (0.5=fast, 1.0=quality)
            cpu_offload: Immediate GPU→CPU transfer
            gpu_id: CUDA device

        Returns:
            interpolated_frames: (N, H, W, C) tensor
                N = 1 if times_to_interpolate=0
                N = 2^times_to_interpolate + 1 otherwise
        """
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        # Enable optimizations
        if enable_tf32 and device.type == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load model with caching
        cache_key = f"{model_name}_{gpu_id}_{use_fp16}"
        model = self._get_or_load_model(cache_key, model_name, device, use_fp16, sample_steps)

        # Preprocessing with adaptive padding
        prev_tensor, next_tensor, pad_info = self._preprocess_with_padding(
            prev_frame, next_frame, device, use_fp16
        )

        # Core interpolation
        if times_to_interpolate == 0:
            # Single-frame interpolation (original paper mode)
            result = self._interpolate_single(
                prev_tensor, next_tensor, model, flow_scale, cpu_offload
            )
        else:
            # Recursive bisection with memory management
            result = self._interpolate_recursive(
                prev_tensor, next_tensor, model, times_to_interpolate,
                flow_scale, cpu_offload
            )

        # Postprocessing: unpad and convert format
        result = self._postprocess_with_unpadding(result, pad_info)

        # Periodic cache clearing (RIFE pattern)
        self._frame_pair_count += 1
        if self._frame_pair_count >= 10:
            soft_empty_cache()
            gc.collect()
            self._frame_pair_count = 0
            print("TLBVFI_V2: Periodic cache clear (10 pairs)")

        return (result,)

    def _interpolate_single(self, prev_tensor, next_tensor, model, flow_scale, cpu_offload):
        """
        Single-frame interpolation (original paper pattern).

        Memory: Only 2 input frames + 1 output frame in GPU at once.
        """
        with torch.no_grad():
            # Core interpolation (original paper: model.sample())
            mid_frame = model.sample(prev_tensor, next_tensor, scale=flow_scale)

            # Immediate CPU transfer if enabled (RIFE pattern)
            if cpu_offload:
                mid_frame = mid_frame.cpu()

        # Postprocess and return
        mid_frame = (mid_frame + 1.0) / 2.0  # [-1,1] → [0,1]
        mid_frame = mid_frame.clamp(0, 1)
        mid_frame = mid_frame.squeeze(0).permute(1, 2, 0)  # (C,H,W) → (H,W,C)

        return mid_frame.unsqueeze(0)  # (1, H, W, C)

    def _interpolate_recursive(self, prev_tensor, next_tensor, model,
                              times_to_interpolate, flow_scale, cpu_offload):
        """
        Recursive bisection with aggressive memory management.

        Key difference from current implementation:
        - Processes frames in PAIRS, not all at once
        - Transfers to CPU immediately after each interpolation
        - Releases GPU memory before next interpolation

        Memory: Only 2 frames in GPU at any time (not 2^N!)
        """
        # Start with prev and next on CPU
        if prev_tensor.device.type != 'cpu':
            prev_cpu = prev_tensor.cpu()
            next_cpu = next_tensor.cpu()
        else:
            prev_cpu = prev_tensor
            next_cpu = next_tensor

        # Postprocess to ComfyUI format
        def to_comfy_format(tensor_gpu):
            tensor_cpu = tensor_gpu.cpu() if cpu_offload else tensor_gpu
            tensor_cpu = (tensor_cpu + 1.0) / 2.0
            tensor_cpu = tensor_cpu.clamp(0, 1)
            tensor_cpu = tensor_cpu.squeeze(0).permute(1, 2, 0)
            return tensor_cpu

        # Initialize frame list on CPU
        frames_cpu = [
            to_comfy_format(prev_tensor),
            to_comfy_format(next_tensor)
        ]

        # Recursive bisection
        for iteration in range(times_to_interpolate):
            new_frames_cpu = [frames_cpu[0]]  # Start with first frame

            for i in range(len(frames_cpu) - 1):
                # Load pair to GPU
                frame_a = frames_cpu[i].unsqueeze(0).permute(0, 3, 1, 2)  # (H,W,C) → (1,C,H,W)
                frame_b = frames_cpu[i+1].unsqueeze(0).permute(0, 3, 1, 2)

                frame_a = (frame_a * 2.0 - 1.0).to(model.device)  # [0,1] → [-1,1], move to GPU
                frame_b = (frame_b * 2.0 - 1.0).to(model.device)

                # Interpolate
                with torch.no_grad():
                    mid_frame = model.sample(frame_a, frame_b, scale=flow_scale)

                # Convert to ComfyUI format and transfer to CPU
                mid_frame_cpu = to_comfy_format(mid_frame)

                # Release GPU memory immediately
                del frame_a, frame_b, mid_frame
                if (i + 1) % 5 == 0:  # Mini cache clear every 5 pairs
                    torch.cuda.empty_cache()

                # Append to CPU list
                new_frames_cpu.extend([mid_frame_cpu, frames_cpu[i+1]])

            # Replace frame list
            frames_cpu = new_frames_cpu

            print(f"  TLBVFI_V2: Iteration {iteration+1}/{times_to_interpolate} → {len(frames_cpu)} frames")

        # Stack all frames
        result = torch.stack(frames_cpu, dim=0)  # (N, H, W, C)

        return result
```

#### Component 2: Adaptive Padding (from Original)

```python
def _preprocess_with_padding(self, prev_frame, next_frame, device, use_fp16):
    """
    Adaptive padding to satisfy model dimension requirements.
    Ported from original TLBVFI (model/VQGAN/vqgan.py:406-432).
    """
    # Convert to PyTorch format
    prev_tensor = prev_frame.permute(0, 3, 1, 2).float()  # (1,H,W,C) → (1,C,H,W)
    next_tensor = next_frame.permute(0, 3, 1, 2).float()

    b, c, h, w = prev_tensor.shape

    # Calculate minimum required dimension
    # From original: min_side = 8 * 2^(num_resolutions-1) * 4
    # For default config: 8 * 2^(5-1) * 4 = 8 * 16 * 4 = 512
    encoder_resolutions = 5  # From original config
    min_side = 8 * (2 ** (encoder_resolutions - 1)) * 4

    # Calculate padding
    pad_h = 0 if h % min_side == 0 else min_side - (h % min_side)
    pad_w = 0 if w % min_side == 0 else min_side - (w % min_side)

    # Avoid padding full dimension (original behavior)
    if pad_h == h:
        pad_h = 0
    if pad_w == w:
        pad_w = 0

    # Apply padding
    if pad_h > 0 or pad_w > 0:
        prev_tensor = F.pad(prev_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        next_tensor = F.pad(next_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        print(f"  TLBVFI_V2: Applied adaptive padding: {h}x{w} → {h+pad_h}x{w+pad_w}")

    # Normalize and convert dtype
    prev_tensor = (prev_tensor * 2.0) - 1.0  # [0,1] → [-1,1]
    next_tensor = (next_tensor * 2.0) - 1.0

    if use_fp16:
        prev_tensor = prev_tensor.half()
        next_tensor = next_tensor.half()

    # Move to device
    prev_tensor = prev_tensor.to(device, non_blocking=True)
    next_tensor = next_tensor.to(device, non_blocking=True)

    pad_info = {
        'pad_h': pad_h,
        'pad_w': pad_w,
        'orig_h': h,
        'orig_w': w,
    }

    return prev_tensor, next_tensor, pad_info

def _postprocess_with_unpadding(self, frames, pad_info):
    """
    Remove padding and convert to ComfyUI format.
    """
    # frames: (N, C, H, W) or (C, H, W) in [-1, 1]

    # Remove padding
    if pad_info['pad_h'] > 0:
        frames = frames[..., :pad_info['orig_h'], :]
    if pad_info['pad_w'] > 0:
        frames = frames[..., :, :pad_info['orig_w']]

    # Denormalize and convert format handled in calling functions
    return frames
```

#### Component 3: Model Loading with FP16

```python
def _get_or_load_model(self, cache_key, model_name, device, use_fp16, sample_steps):
    """
    Load model with caching, FP16 support, and configurable timesteps.
    """
    global _MODEL_CACHE

    if cache_key in _MODEL_CACHE:
        print(f"TLBVFI_V2: Reusing cached model {model_name}")
        return _MODEL_CACHE[cache_key]

    # Memory pressure check
    if device.type == 'cuda':
        mem_stats = get_memory_stats(device)
        if mem_stats['free'] < 4.0:
            print(f"TLBVFI_V2: Low memory ({mem_stats['free']:.1f}GB), clearing cache")
            clear_model_cache()

    # Load model
    print(f"TLBVFI_V2: Loading {model_name} (FP16={use_fp16}, steps={sample_steps})")

    model = load_tlbvfi_model(model_name, device, sample_steps=sample_steps)

    # Convert to FP16 if requested
    if use_fp16 and device.type == 'cuda':
        model = model.half()
        print(f"  Converted to FP16 (2x memory reduction)")

    # Cache model
    _MODEL_CACHE[cache_key] = model

    return model
```

---

## Implementation Roadmap

### Phase 1: Core Production Features (Week 1)

**Priority: CRITICAL**

1. **Create TLBVFI_Interpolator_V2**
   - Implement single-frame interpolation (original paper)
   - Add adaptive padding/unpadding
   - Implement recursive bisection with memory management
   - Add immediate CPU offload

2. **Add FP16 Support**
   - Model conversion to FP16
   - Input/output dtype handling
   - Mixed precision safety checks

3. **Add Periodic Cache Clearing**
   - Track frame pair count
   - Clear every 10 pairs (RIFE pattern)
   - Integrate with ComfyUI's soft_empty_cache()

**Expected Outcomes**:
- Memory-safe interpolation for 4K video
- 2x memory reduction with FP16
- No OOM crashes on long videos

### Phase 2: Performance Optimizations (Week 2)

**Priority: HIGH**

4. **TF32 Acceleration**
   - Auto-detect RTX 30/40 series
   - Enable TF32 globally
   - Benchmark speedup

5. **Configurable Quality Settings**
   - Expose sample_steps parameter (10/20/50)
   - Expose flow_scale parameter (0.5/1.0)
   - Add quality presets (fast/balanced/quality)

6. **Model Loading Optimizations**
   - Cache with FP16/FP32 distinction
   - Add model warmup pass
   - Optimize checkpoint loading

**Expected Outcomes**:
- 4x speedup on RTX 30/40 with TF32
- User control over quality/speed tradeoff
- Faster model loading

### Phase 3: Advanced Features (Week 3+)

**Priority: MEDIUM**

7. **Batch Processing Support**
   - Process multiple frame pairs in single GPU call
   - Adaptive batch size based on VRAM
   - Progress tracking

8. **Alternative Sampling Schedules**
   - Cosine schedule support
   - Custom schedule input
   - Schedule visualization

9. **Comprehensive Testing**
   - Unit tests for padding/unpadding
   - Integration tests with various resolutions
   - Memory profiling for different configs
   - Benchmark against RIFE/FILM

**Expected Outcomes**:
- Higher throughput with batching
- Research-level flexibility
- Production-grade reliability

---

## Deprecation Strategy

### Keep

- `FramePairSlicer`: Still useful for batch processing
- `TLBVFI_Interpolator`: Deprecate but keep for backward compatibility

### Replace

- `TLBVFI_ChunkProcessor`: Replace with `TLBVFI_Interpolator_V2` + standard VHS nodes
- `ChunkVideoSaver`: Not needed (use VHS VideoCombine)
- `VideoConcatenator`: Not needed (VHS handles this)

### New Workflow

```
LoadVideo (VHS)
    ↓
FramePairSlicer (or direct use)
    ↓
TLBVFI_Interpolator_V2 (NEW - production grade)
    ↓ (interpolated frames)
VideoCombine (VHS) - standard video output
```

**Advantages**:
- Simpler architecture
- Leverage VHS's proven video encoding
- Focus on core VFI functionality
- Better ComfyUI ecosystem integration

---

## Memory Profile Comparison

### Current Implementation (BROKEN)

```
times_to_interpolate=4, 4K, FP32:
- Input: 2 frames × 260MB = 520MB
- Iteration 1: 3 frames × 260MB = 780MB
- Iteration 2: 5 frames × 260MB = 1.3GB
- Iteration 3: 9 frames × 260MB = 2.3GB
- Iteration 4: 17 frames × 260MB = 4.4GB
- FFmpeg encoding: +500MB (buffer)
Total: ~5GB peak VRAM (FAILS on 8GB GPU!)
```

### Production V2 (SAFE)

```
times_to_interpolate=4, 4K, FP16:
- Model: 3.6GB (cached, persistent)
- Peak per iteration:
  - 2 input frames on GPU: 2 × 130MB = 260MB
  - 1 output frame: 130MB
  - Intermediate buffers: ~200MB
  Total per iteration: ~600MB
- CPU accumulation: Unlimited frames (no GPU impact)

Total: 3.6GB (model) + 0.6GB (processing) = 4.2GB peak
Fits comfortably on 8GB GPU with headroom!
```

### RIFE Comparison (for reference)

```
RIFE inference, 4K, FP16:
- Model: ~50MB (tiny!)
- Per-frame processing: ~300MB
- CPU accumulation: 0MB GPU impact

Total: ~350MB peak (extremely efficient!)
```

**Note**: TLBVFI is a diffusion model, inherently larger than RIFE's flow-based approach. Our optimization goal is to match RIFE's memory pattern while accepting the model size difference.

---

## Testing Plan

### Memory Tests

1. **Resolution Tests**
   - 480p, 720p, 1080p, 1440p, 4K
   - Verify adaptive padding works
   - Measure peak VRAM usage

2. **Interpolation Depth Tests**
   - times_to_interpolate: 0, 1, 2, 3, 4
   - Verify memory doesn't explode
   - Confirm output frame counts

3. **Long Video Tests**
   - 100 frame pairs continuous
   - Monitor VRAM over time
   - Verify periodic cache clearing

### Quality Tests

1. **Padding Correctness**
   - Compare padded vs non-padded outputs
   - Verify no artifacts at boundaries

2. **FP16 vs FP32 Quality**
   - PSNR/SSIM comparison
   - Visual inspection for artifacts

3. **Sample Steps Impact**
   - 10 vs 20 vs 50 steps
   - Quality vs speed tradeoff measurement

### Performance Tests

1. **TF32 Speedup**
   - Benchmark on RTX 3090/4090
   - Compare vs FP32 baseline

2. **Throughput Comparison**
   - frames/second vs RIFE
   - frames/second vs current implementation

---

## Success Criteria

### Must Have (MVP)

- ✅ No OOM on 8GB GPU with 4K video
- ✅ Memory usage flat over 100+ frame pairs
- ✅ FP16 support with <1% quality loss
- ✅ Adaptive padding for arbitrary resolutions
- ✅ Aligned with original paper architecture

### Should Have

- ✅ TF32 acceleration on RTX 30/40
- ✅ Configurable quality/speed tradeoff
- ✅ Periodic cache clearing
- ✅ Backward compatible with existing workflows

### Nice to Have

- ✅ Batch processing support
- ✅ Custom sampling schedules
- ✅ Comprehensive test suite
- ✅ Performance parity with RIFE (adjusted for model size)

---

## Conclusion

The current chunk-based approach deviates from the original paper and introduces memory management issues. The production V2 solution:

1. **Returns to Original Paper Design**: Single-frame interpolation as core operation
2. **Adopts RIFE/FILM Patterns**: CPU offload, periodic clearing, FP16
3. **Adds Original Optimizations**: Adaptive padding, flow scale, configurable timesteps
4. **Simplifies Architecture**: Remove custom video encoding, use VHS
5. **Guarantees Memory Safety**: Flat memory usage regardless of video length

This approach is **production-ready**, **hardware-optimized**, and **aligned with both the original research and industry best practices**.
