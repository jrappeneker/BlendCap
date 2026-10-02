# Experimental Apple Silicon setup

This is a community compatibility path, not official macOS support. Preserve the upstream licensing and notices. Model licenses and access requirements are separate from the source-code license. Never commit model weights, access tokens, or personal footage.

## What works and what changed

The tested combination is **ONNX Runtime/CoreML for DINOv3 + CPU for PyTorch**. The bundled MHR TorchScript skeleton solver calls `.double()`, requiring float64 tensors. MPS cannot represent those tensors, and `PYTORCH_ENABLE_MPS_FALLBACK=1` does not fix the conversion. The local change in `save_mhr_data.py` makes `_get_device()` select CPU on Apple Silicon. ONNX provider selection remains independent; selecting CPU for PyTorch does not force the DINO backbone off CoreML.

Native MPS person detection worked, but full MPS body capture failed. Simply seeing MPS available or CoreML listed is not an end-to-end compatibility test. CoreML sessions can also fall back to CPU for unsupported operations.

Tested 2 October 2026: Apple Silicon, Blender 5.2.0 LTS, Python 3.12 ARM64, torch 2.7.1, torchvision 0.22.1, ONNX Runtime 1.26.0, upstream commit `e3238699507c00e34b6c948931c4742915b8dc42`.

Verified: five-frame body capture on real portrait phone footage, finite output for 127 joints, BVH export, and Blender import. Full-clip capture was still running when this guide was prepared. Facial capture, hand refinement, other Macs, and other Blender versions have not been validated. On one single-person clip, subsequent frames took about 2.2 seconds after startup; this is not a performance guarantee. First-frame camera estimation took roughly a minute on CPU.

## Install the extension

Prerequisites: Blender 4.2+ (only 5.2 tested), native ARM64 Python 3.12, Git, Xcode command-line tools, internet access, and approximately 32 GB free space. Python 3.12 can be installed with Homebrew: `brew install python@3.12`.

Clone this repository. Build a source archive from tracked files so local videos, environments, models, and credentials are excluded:

```sh
git archive --format=zip --output=../BlendCap-macos.zip HEAD
```

In Blender, choose **Edit → Preferences → Add-ons → menu → Install from Disk**, select the ZIP, and enable BlendCap. Save Preferences in that same Blender session. In BlendCap preferences, use **Open Install Folder**. Run all subsequent commands from that installed folder: the addon looks for `venv/` and `checkpoints/` beside its source.

Avoid configuring preferences in a separate background Blender process while your main Blender session is running: a later preference save in the main session can overwrite those changes.

## Python environment

```sh
python3.12 -m venv venv
./venv/bin/python -m pip install --upgrade pip setuptools wheel
./venv/bin/python -m pip install -r requirements-macos.txt
./venv/bin/python -m pip check
./venv/bin/python - <<'PY'
import platform, torch, onnxruntime as ort
assert platform.machine() == 'arm64', 'Use native ARM64 Python, not Rosetta'
print('Torch:', torch.__version__)
print('MPS available (not used for body rig):', torch.backends.mps.is_available())
print('ORT:', ort.__version__, ort.get_available_providers())
assert 'CoreMLExecutionProvider' in ort.get_available_providers()
PY
```

Do not install the CUDA-specific `requirements.txt` on a Mac. The Mac file preserves upstream runtime pins while replacing CUDA packages with native wheels. Transitive dependencies are not fully locked.

## Model downloads

First request access and accept the terms yourself at [Meta's SAM 3D Body model page](https://huggingface.co/facebook/sam-3d-body-dinov3). Authenticate locally; never paste a token into a repository file or issue:

```sh
./venv/bin/hf auth login
./venv/bin/hf download facebook/sam-3d-body-dinov3 \
  model.ckpt model_config.yaml assets/mhr_model.pt \
  --local-dir checkpoints/sam-3d-body-dinov3
./venv/bin/hf download Ruicheng/moge-2-vits-normal model.pt \
  --local-dir checkpoints/moge-2-vits-normal
./venv/bin/hf download Blendcap/blendcap-onnx-exports \
  backbone_dinov3_fp16.onnx backbone_dinov3_fp16.onnx_data yolo11m_pose.onnx \
  --local-dir checkpoints/onnx
mkdir -p checkpoints/yolo
./venv/bin/python - <<'PY'
from ultralytics import YOLO
YOLO('checkpoints/yolo/yolo11m-pose.pt')
PY
```

Both DINO files are required. Leave `USE_ORT_MOGE` unset: upstream deliberately disables that export by default because of focal-estimation accuracy concerns. Face models download on first use, but that path is untested here.

## Blender settings

Restart Blender after installing dependencies. In **Edit → Preferences → Add-ons → BlendCap**:

- **Capture Backend:** Non-NVIDIA – DirectML / ONNX (experimental). Despite its Windows-oriented label, this enables ONNX and allows CoreML on macOS.
- **Refine Hand Tracking:** Always off for the first run.

In **3D Viewport → N → BlendCap → Capture**:

- Body on, Face off for the tested configuration.
- **FOV Mode:** Sample first frame only. Estimating every frame is expensive on CPU.
- **Capture Skip:** 0 for every frame. Higher values trade motion accuracy for speed.

**CPU Only** forces both PyTorch and ONNX onto CPU, disabling the CoreML backbone. It is a diagnostic fallback, not the tested accelerated setup. Settings changes require cancelling and restarting an existing capture to take effect.

Save the Blender project, select a video under **Video Source → File Path**, click **Run Capture**, then **BVH Settings → Generate/Reload BVH**. The second step creates and imports the animated armature. Retarget that source armature onto a character rig separately.

## Validate before a long capture

```sh
USE_ORT=1 ./venv/bin/python save_mhr_data.py \
  --video '/absolute/path/to/video.mov' \
  --output /tmp/blendcap-mac-test.npz --max-frames 5 --no-hands
./venv/bin/python npz_to_bvh.py \
  --input /tmp/blendcap-mac-test.npz \
  --output /tmp/blendcap-mac-test.bvh --no-hands
```

Expect `Device: cpu`, the Apple Silicon compatibility message, and `ORT backbone active: CoreMLExecutionProvider`. Require valid frames, not just a successful startup. Remove `--max-frames 5` for the whole clip. Keep logs local unless reviewed for personal paths and other private information before sharing.

## Lightweight regression tests

```sh
python3.12 -m unittest discover -s tests -v
```

These tests exercise device routing without loading torch or any model. They cover the Apple float64 workaround, CUDA priority, explicit CPU override, DirectML hybrid behavior, and CPU fallback. They do not replace hardware inference tests.
