# DSINE Benchmark for RDR2 Surface Normals

This project runs the DSINE surface-normal model on the RDR2 image and
ground-truth normal-map dataset. The benchmark reports angular-error metrics
overall and by daytime, and can save predictions and visual comparisons.

## Requirements

- Linux with an NVIDIA GPU and a CUDA-compatible NVIDIA driver.
- Python 3.10.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for environment
  and dependency management.
- The RDR2 benchmark dataset and the pretrained DSINE checkpoint (download
  instructions below).

The project is in `DSINE/DSINE/` in this repository. Open a terminal at the
repository root and enter that directory:

```bash
cd DSINE/DSINE
```

The `pyproject.toml` configures CUDA 12.1 PyTorch wheels. A sufficiently recent
NVIDIA driver is required; the CUDA toolkit does not need to be installed
separately.

## Create and sync the environment

Create a Python 3.10 virtual environment and sync the exact dependencies from
`pyproject.toml` and `uv.lock`:

```bash
uv venv --prompt DSINE --python 3.10 .venv
uv sync --locked
```

`--locked` ensures uv uses the committed lock file and does not silently update
it. You can run commands with `uv run` without activating the environment:

```bash
uv run --locked python benchmark_rdr2.py --help
```

Alternatively, activate it first:

```bash
source .venv/bin/activate
python benchmark_rdr2.py --help
```

## Download the model checkpoint

Download the DSINE model weights from the
[official Google Drive folder](https://drive.google.com/drive/folders/1t3LMJIIrSnCGwOEf53Cyg0lkSXd3M4Hm?usp=drive_link).
Preserve the folder structure and place the CVPR 2024 checkpoint at:

```text
projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt
```

This is the default checkpoint expected by the supplied DSINE experiment
configuration. If you save the weights elsewhere, provide their location with
`--checkpoint`.

## Prepare the dataset

The dataset root must contain both `PNG/` and `NormalMap/` directories:

```text
<dataset-root>/
├── PNG/
│   └── o_<id>_<daytime>.png
└── NormalMap/
    └── n_<id>.exr
```

Every PNG must have a matching EXR for its ID. The benchmark evaluates each
daytime image against that ID's normal map. The dataset is not included in this
repository; set `<dataset-root>` to the location where you have stored it.

## Run the benchmark

From `DSINE/DSINE/`, run the full evaluation and choose an output directory:

```bash
uv run --locked python benchmark_rdr2.py \
  --dataset-root /path/to/RDR2_dataset_processed_test \
  --output-dir /path/to/rdr2-dsine-results
```

The script displays a progress bar with an estimated time remaining while
processing images. Model setup and checkpoint loading happen before the bar
starts. A short run is useful to check that the environment, checkpoint, and
dataset paths are valid:

```bash
uv run --locked python benchmark_rdr2.py \
  --dataset-root /path/to/RDR2_dataset_processed_test \
  --output-dir /path/to/rdr2-dsine-smoke-test \
  --max-images 5 \
  --metrics-only
```

The main options are:

| Option | Description |
| --- | --- |
| `--dataset-root PATH` | Dataset directory containing `PNG/` and `NormalMap/`. |
| `--output-dir PATH` | Required directory for metrics, predictions, and visualizations. |
| `--config PATH` | DSINE experiment config; defaults to the included CVPR 2024 config. |
| `--checkpoint PATH` | Override the checkpoint path in the config. |
| `--device DEVICE` | CUDA device, default `cuda:0`. |
| `--fov-degrees DEGREES` | Assumed field of view, default `60`. |
| `--ground-truth-encoding {signed,zero-to-one}` | EXR vector encoding; default `zero-to-one`. |
| `--max-images N` | Process only the first N paired images. |
| `--visualize-images N` | Save side-by-side PNGs for the first N images. |
| `--metrics-only` | Do not save per-image prediction arrays. |

Use `--ground-truth-encoding signed` if the EXR normals are stored in
`[-1, 1]` rather than remapped to `[0, 1]`. OpenCV reads EXR channels in BGR
order; the script reorders them to XYZ. The benchmark uses the configured
60-degree field of view unless `--fov-degrees` is specified.

## Results

The output directory contains:

- `per_image_metrics.csv` — metrics for each image.
- `daytime_metrics.csv` — metrics grouped by daytime, plus an analytic
  random-guess baseline.
- `summary.json` — run configuration and overall/daytime metrics.
- `predictions/` — float32 XYZ prediction arrays with shape
  `(height, width, 3)`, unless `--metrics-only` was used.
- `visualizations/` — input, ground-truth, and prediction panels when
  `--visualize-images N` is greater than zero.

Prediction arrays are uncompressed and can use about 42 MiB per
2560x1440 image. Consider using `--metrics-only` when disk space is limited.

## Upstream DSINE project

For the original DSINE paper, project details, training, and other demos, see
[the upstream DSINE README](README_DSINE.md).