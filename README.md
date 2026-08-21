# PDSI-RTDETR

**Structure-Aware Dynamic Alignment for Enhanced Small Object Detection in UAV Imagery**

This repository provides the reference implementation of PDSI-RTDETR. If you use this code in your research, please cite this work.

## Architecture

PDSI-RTDETR extends RT-DETR-L with two lightweight modules:

- **ConvSimAMPP**: Structure-aware attention (DWConv + SimAM + learnable gate) inserted at P3/P4/P5 backbone levels
- **ResDySample**: Flow-guided dynamic upsampler replacing nearest-neighbor interpolation in the neck

```
RT-DETR-L backbone -> ConvSimAMPP(P3/P4/P5) -> HybridEncoder(AIFI+CCFM) -> RTDETRDecoder
                                                      | ResDySample (x2)
```

## Module Overview

- **SimAM**: Parameter-free energy-based attention (ICML 2021)
- **ConvSimAMPP**: DWConv -> BN -> SiLU -> SimAM -> gated residual fusion with learnable gate
- **ResDySample**: Offset prediction -> tanh constraint -> pixel shuffle -> grid_sample -> alpha fusion with static bilinear branch

## Configurations

| Config | Modules | Params |
|--------|---------|--------|
| `rtdetr-l-visdrone.yaml` | Baseline RT-DETR-L | 32.83M |
| `rtdetr-l-visdrone-convsimam.yaml` | +ConvSimAMPP | 34.69M |
| `rtdetr-l-visdrone-resdysample.yaml` | +ResDySample | 34.02M |
| `rtdetr-l-visdrone-pdsi.yaml` | PDSI-RTDETR (both modules) | 35.88M |

All configs are based on the standard RT-DETR-L backbone.

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Setup

1. Copy custom modules into your Ultralytics installation:

```bash
cp modules/conv_simampp.py <ultralytics_dir>/ultralytics/nn/modules/
cp modules/res_dysample.py <ultralytics_dir>/ultralytics/nn/modules/
```

2. Register modules in `<ultralytics_dir>/ultralytics/nn/modules/__init__.py`:

```python
from .conv_simampp import SimAM, ConvSimAMPP
from .res_dysample import ResDySample
```

Add `SimAM`, `ConvSimAMPP`, and `ResDySample` to both `__all__` and the `base_modules` frozenset in `tasks.py`.

3. Copy model configs:

```bash
cp cfg/*.yaml <ultralytics_dir>/ultralytics/cfg/models/rt-detr/
```

4. Train using the standard Ultralytics training interface with one of the provided configs.

## Citation

If you use this code in your research, please cite this work:

```bibtex
@misc{fu2026pdsi,
  title={Structure-Aware Dynamic Alignment for Enhanced Small Object Detection in UAV Imagery},
  author={Fu, Jiadi and Zhou, Hangyu and Wu, Jiawei},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
