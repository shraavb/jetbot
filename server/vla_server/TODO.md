# VLA Server - TODO / Known Issues

## Critical: Inference Timeout Issue

The test script (`scripts/test_vla_server.py`) times out because VLA inference is extremely slow on Apple Silicon MPS.

### Root Cause
- **First inference on MPS takes ~457 seconds (~7.6 minutes)** due to Metal shader compilation/JIT
- Subsequent inferences are also slow (need more testing to determine exact time)
- The test script default timeout is 120 seconds, which is insufficient

### Fixes Needed

1. **Add warmup inference during server startup**
   - Run a dummy inference when the server starts so users don't experience the slow first inference
   - Location: `server/vla_server/server.py` in `VLAServer.__init__()` after model loading
   - Example:
     ```python
     # After self.vla = OpenVLAWrapper(...)
     print("Running warmup inference (this may take several minutes)...", flush=True)
     dummy_image = Image.new('RGB', (224, 224), color='gray')
     self.vla.predict(dummy_image, "warmup")
     print("Warmup complete.", flush=True)
     ```

2. **Increase default timeout in test script**
   - Change default from 120s to 600s (10 minutes) or more
   - Location: `scripts/test_vla_server.py` line 126

3. **Consider using a smaller/quantized model**
   - The 7B parameter model is too large for efficient MPS inference
   - Options:
     - Use 4-bit quantization (CUDA only currently)
     - Find a smaller OpenVLA variant
     - Use CPU with optimized settings (still slow but more predictable)

## Minor Issues (Already Fixed)

These were fixed during debugging:

1. **Output buffering** - Added `flush=True` to print statements in:
   - `server/vla_server/openvla_wrapper.py`
   - `server/vla_server/server.py`

2. **Circular import warning** - Fixed `server/vla_server/__init__.py` to use lazy imports

## Dependency Warnings

The model expects older library versions:
```
Expected: transformers==4.40.1, tokenizers==0.19.1
Got: transformers==4.57.3, tokenizers==0.22.1
```

This may cause inference regressions. Consider pinning to the expected versions if issues persist.

## Performance Benchmarks Needed

- [ ] Measure subsequent inference times on MPS after warmup
- [ ] Compare CPU vs MPS performance for this model
- [ ] Test with different batch sizes
- [ ] Profile to identify bottlenecks

## Hardware Tested

- Apple Silicon Mac (MPS backend)
- ~15GB memory usage for model
- First inference: ~457 seconds
