#!/usr/bin/env python3
"""Run DSINE on the RDR2 image/normal-map dataset and report angular metrics."""

import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = Path("/media/rdr2/RDR2_dataset_processed_test")
DSINE_PROJECT_DIR = SCRIPT_DIR / "projects/dsine"
DEFAULT_CONFIG = SCRIPT_DIR / "projects/dsine/experiments/exp001_cvpr2024/dsine.txt"
IMAGE_PATTERN = re.compile(r"o_(\d+)_(\d+)\Z")
METRIC_NAMES = (
    "mean",
    "median",
    "rmse",
    "accuracy_5",
    "accuracy_7_5",
    "accuracy_11_25",
    "accuracy_22_5",
    "accuracy_30",
)
THRESHOLDS = (5.0, 7.5, 11.25, 22.5, 30.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSINE inference and evaluate RDR2 normal maps."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root containing PNG/ and NormalMap/ (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for prediction maps and benchmark metrics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="DSINE experiment config (default: official CVPR 2024 config).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override the checkpoint path from the DSINE config.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device (DSINE's upstream model currently assumes CUDA).",
    )
    parser.add_argument(
        "--fov-degrees",
        type=float,
        default=60.0,
        help="Assumed horizontal/vertical field of view when intrinsics are unavailable.",
    )
    parser.add_argument(
        "--ground-truth-encoding",
        choices=("signed", "zero-to-one"),
        default="zero-to-one",
        help="EXR vector encoding: signed XYZ in [-1,1], or XYZ remapped to [0,1].",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Optionally process only the first N paired images (useful for a smoke test).",
    )
    parser.add_argument(
        "--visualize-images",
        type=int,
        default=0,
        help="Save side-by-side input/ground-truth/prediction PNGs for the first N images.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Do not save per-image prediction maps.",
    )
    return parser.parse_args()


def find_samples(dataset_root):
    image_dir = dataset_root / "PNG"
    normal_dir = dataset_root / "NormalMap"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not normal_dir.is_dir():
        raise FileNotFoundError(f"Normal-map directory not found: {normal_dir}")

    images = sorted(image_dir.glob("o_*.png"))
    if not images:
        raise FileNotFoundError(f"No o_<id>_<daytime>.png images found in {image_dir}")

    samples = []
    missing = []
    for image_path in images:
        match = IMAGE_PATTERN.fullmatch(image_path.stem)
        if match is None:
            raise ValueError(f"Unexpected image filename: {image_path.name}")
        identifier, _daytime = match.groups()
        normal_path = normal_dir / f"n_{identifier}.exr"
        if normal_path.is_file():
            samples.append((image_path, normal_path))
        else:
            missing.append((image_path.name, normal_path.name))

    if missing:
        examples = ", ".join(f"{image} -> {normal}" for image, normal in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} image(s) have no matching normal map; examples: {examples}"
        )
    return samples


def load_dsine_args(config_path):
    import projects.dsine.config as dsine_config

    original_argv = sys.argv
    original_experiment_dir = dsine_config.EXPERIMENT_DIR
    with tempfile.TemporaryDirectory(prefix="dsine-rdr2-config-") as temporary_dir:
        try:
            dsine_config.EXPERIMENT_DIR = temporary_dir
            sys.argv = [str(SCRIPT_DIR / "benchmark_rdr2.py"), str(config_path)]
            return dsine_config.get_args(test=True)
        finally:
            sys.argv = original_argv
            dsine_config.EXPERIMENT_DIR = original_experiment_dir


def create_model(dsine_args, checkpoint_path, device):
    import utils.utils as utils

    if dsine_args.NNET_architecture == "v00":
        from models.dsine.v00 import DSINE_v00 as DSINE
    elif dsine_args.NNET_architecture == "v01":
        from models.dsine.v01 import DSINE_v01 as DSINE
    elif dsine_args.NNET_architecture == "v02":
        from models.dsine.v02 import DSINE_v02 as DSINE
    elif dsine_args.NNET_architecture == "v02_kappa":
        from models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise ValueError(
            f"Unsupported DSINE architecture: {dsine_args.NNET_architecture}"
        )

    model = DSINE(dsine_args).to(device)
    dsine_args.ckpt_path = str(checkpoint_path)
    model = utils.load_checkpoint(dsine_args.ckpt_path, model)
    model.eval()
    return model


def load_ground_truth(path, cv2, encoding):
    normal_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if normal_bgr is None:
        raise RuntimeError(
            f"Could not read {path} as OpenEXR. Ensure OpenCV has EXR support "
            "and OPENCV_IO_ENABLE_OPENEXR=1 is set."
        )
    if normal_bgr.ndim != 3 or normal_bgr.shape[2] < 3:
        raise ValueError(
            f"Expected at least three EXR channels in {path}, got {normal_bgr.shape}"
        )

    # OpenCV returns color channels in BGR order; DSINE uses XYZ in RGB order.
    normal_xyz = normal_bgr[..., :3][..., ::-1].astype("float32", copy=True)
    if encoding == "zero-to-one":
        normal_xyz = normal_xyz * 2.0 - 1.0
    return normal_xyz


def get_metrics(errors):
    import numpy as np

    return {
        "mean": float(errors.mean()),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "accuracy_5": float(100.0 * (errors < 5.0).mean()),
        "accuracy_7_5": float(100.0 * (errors < 7.5).mean()),
        "accuracy_11_25": float(100.0 * (errors < 11.25).mean()),
        "accuracy_22_5": float(100.0 * (errors < 22.5).mean()),
        "accuracy_30": float(100.0 * (errors < 30.0).mean()),
    }


def random_guess_metrics():
    """Expected angular metrics for an independent uniform random unit vector."""
    return {
        "mean": 90.0,
        "median": 90.0,
        "rmse": math.sqrt(math.pi**2 / 2.0 - 2.0) * 180.0 / math.pi,
        "accuracy_5": 100.0 * (1.0 - math.cos(math.radians(5.0))) / 2.0,
        "accuracy_7_5": 100.0 * (1.0 - math.cos(math.radians(7.5))) / 2.0,
        "accuracy_11_25": 100.0 * (1.0 - math.cos(math.radians(11.25))) / 2.0,
        "accuracy_22_5": 100.0 * (1.0 - math.cos(math.radians(22.5))) / 2.0,
        "accuracy_30": 100.0 * (1.0 - math.cos(math.radians(30.0))) / 2.0,
    }


def create_metric_accumulator(np, histogram_edges):
    return {
        "histogram": np.zeros(histogram_edges.size - 1, dtype=np.int64),
        "error_sum": 0.0,
        "squared_error_sum": 0.0,
        "valid_pixels": 0,
        "threshold_counts": np.zeros(len(THRESHOLDS), dtype=np.int64),
    }


def update_metric_accumulator(accumulator, errors, histogram_edges, np):
    accumulator["histogram"] += np.histogram(errors, bins=histogram_edges)[0]
    accumulator["error_sum"] += float(errors.sum(dtype=np.float64))
    accumulator["squared_error_sum"] += float(
        np.square(errors.astype(np.float64)).sum()
    )
    accumulator["valid_pixels"] += errors.size
    accumulator["threshold_counts"] += np.asarray(
        [(errors < threshold).sum() for threshold in THRESHOLDS], dtype=np.int64
    )


def finalize_metric_accumulator(accumulator, histogram_edges, np):
    valid_pixels = accumulator["valid_pixels"]
    median_bin = int(
        np.searchsorted(
            np.cumsum(accumulator["histogram"]), (valid_pixels + 1) // 2
        )
    )
    metrics = {
        "mean": accumulator["error_sum"] / valid_pixels,
        "median": float(
            (histogram_edges[median_bin] + histogram_edges[median_bin + 1]) / 2
        ),
        "rmse": float((accumulator["squared_error_sum"] / valid_pixels) ** 0.5),
    }
    metrics.update(
        {
            name: float(100.0 * count / valid_pixels)
            for name, count in zip(
                METRIC_NAMES[3:], accumulator["threshold_counts"]
            )
        }
    )
    return metrics


def main():
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_override = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else None
    )

    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be greater than zero")
    if args.visualize_images < 0:
        raise ValueError("--visualize-images cannot be negative")
    if not 0.0 < args.fov_degrees < 180.0:
        raise ValueError("--fov-degrees must be between 0 and 180")
    if not config_path.is_file():
        raise FileNotFoundError(f"DSINE config not found: {config_path}")

    samples = find_samples(dataset_root)
    if args.max_images is not None:
        samples = samples[: args.max_images]
    visualize_count = min(args.visualize_images, len(samples))

    sys.path.insert(0, str(SCRIPT_DIR))
    dsine_args = load_dsine_args(config_path)
    checkpoint_path = checkpoint_override or Path(dsine_args.ckpt_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (DSINE_PROJECT_DIR / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"DSINE checkpoint not found: {checkpoint_path}\n"
            "Download the official exp001_cvpr2024/dsine.pt checkpoint and place "
            "it at projects/dsine/checkpoints/exp001_cvpr2024/dsine.pt, or pass "
            "--checkpoint."
        )

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw
    from tqdm import tqdm
    from models.dsine.submodules import get_pixel_coords
    from utils.projection import intrins_from_fov
    import utils.utils as utils

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DSINE's upstream implementation requires CUDA. Install a CUDA-enabled "
            "PyTorch build and verify that a GPU is available."
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device, such as cuda:0")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {device} requested, but only "
            f"{torch.cuda.device_count()} device(s) are available"
        )

    model = create_model(dsine_args, checkpoint_path, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    if not args.metrics_only:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    if visualize_count:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    normalize_mean = torch.tensor(
        (0.485, 0.456, 0.406), device=device
    ).view(1, 3, 1, 1)
    normalize_std = torch.tensor(
        (0.229, 0.224, 0.225), device=device
    ).view(1, 3, 1, 1)

    histogram_edges = np.linspace(0.0, 180.0, 18001, dtype=np.float64)
    global_accumulator = create_metric_accumulator(np, histogram_edges)
    daytime_accumulators = {}

    csv_path = output_dir / "per_image_metrics.csv"
    fieldnames = (
        "image",
        "ground_truth",
        "prediction",
        "valid_pixels",
        *METRIC_NAMES,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with torch.inference_mode():
            for index, (image_path, normal_path) in enumerate(
                tqdm(samples, desc="Benchmarking", unit="image"), start=1
            ):
                image_match = IMAGE_PATTERN.fullmatch(image_path.stem)
                if image_match is None:
                    raise ValueError(f"Unexpected image filename: {image_path.name}")
                identifier, daytime = image_match.groups()

                with Image.open(image_path) as image_file:
                    image_rgb = np.asarray(image_file.convert("RGB"), dtype=np.float32)
                image_rgb /= 255.0
                image = (
                    torch.from_numpy(image_rgb)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )

                _, _, height, width = image.shape
                padding = utils.get_padding(height, width)
                image = F.pad(image, padding, mode="constant", value=0.0)
                image = (image - normalize_mean) / normalize_std
                padded_height, padded_width = image.shape[-2:]

                if hasattr(model, "pixel_coords"):
                    coord_height, coord_width = model.pixel_coords.shape[-2:]
                    if (
                        coord_height < padded_height
                        or coord_width < padded_width
                        or model.pixel_coords.device != device
                    ):
                        model.pixel_coords = get_pixel_coords(
                            h=max(coord_height, padded_height),
                            w=max(coord_width, padded_width),
                        ).to(device)

                intrins = intrins_from_fov(
                    new_fov=args.fov_degrees,
                    H=height,
                    W=width,
                    device=device,
                ).unsqueeze(0)
                intrins[:, 0, 2] += padding[0]
                intrins[:, 1, 2] += padding[2]

                pred = model(image, intrins=intrins, mode="test")[-1][:, :3]
                pred = pred[
                    :,
                    :,
                    padding[2] : padding[2] + height,
                    padding[0] : padding[0] + width,
                ]
                prediction = (
                    pred[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
                )

                visualize = index <= visualize_count
                ground_truth = load_ground_truth(
                    normal_path, cv2, args.ground_truth_encoding
                )
                if ground_truth.shape[:2] != prediction.shape[:2]:
                    raise ValueError(
                        f"Image/normal dimensions differ for {image_path.name}: "
                        f"{prediction.shape[:2]} vs {ground_truth.shape[:2]}"
                    )

                pred_length = np.linalg.norm(prediction, axis=2)
                gt_length = np.linalg.norm(ground_truth, axis=2)
                valid = (
                    np.isfinite(prediction).all(axis=2)
                    & np.isfinite(ground_truth).all(axis=2)
                    & (pred_length > 1e-8)
                    & (gt_length > 1e-8)
                )
                if not np.any(valid):
                    raise ValueError(f"No valid ground-truth normals in {normal_path}")

                pred_unit = prediction[valid] / pred_length[valid, None]
                gt_unit = ground_truth[valid] / gt_length[valid, None]
                cosine = np.clip(np.sum(pred_unit * gt_unit, axis=1), -1.0, 1.0)
                errors = np.degrees(np.arccos(cosine))
                image_metrics = get_metrics(errors)

                if visualize:
                    gt_visualization = np.zeros_like(ground_truth, dtype=np.float32)
                    prediction_visualization = np.zeros_like(
                        prediction, dtype=np.float32
                    )
                    gt_visualization[valid] = gt_unit
                    prediction_visualization[valid] = pred_unit
                    panel_width = width * 3
                    header_height = 36
                    visualization = Image.new(
                        "RGB", (panel_width, height + header_height), "white"
                    )
                    draw = ImageDraw.Draw(visualization)
                    draw.text((8, 10), "Input RGB", fill="black")
                    draw.text((width + 8, 10), "Ground truth normal", fill="black")
                    draw.text((width * 2 + 8, 10), "DSINE prediction", fill="black")
                    visualization.paste(
                        Image.fromarray(
                            np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8),
                            mode="RGB",
                        ),
                        (0, header_height),
                    )
                    for panel_index, normal in enumerate(
                        (gt_visualization, prediction_visualization), start=1
                    ):
                        normal_rgb = np.clip((normal + 1.0) * 127.5, 0, 255).astype(
                            np.uint8
                        )
                        normal_rgb[~valid] = 0
                        visualization.paste(
                            Image.fromarray(normal_rgb, mode="RGB"),
                            (panel_index * width, header_height),
                        )
                    visualization.save(
                        visualization_dir / f"{image_path.stem}.png"
                    )

                update_metric_accumulator(
                    global_accumulator, errors, histogram_edges, np
                )
                if daytime not in daytime_accumulators:
                    daytime_accumulators[daytime] = {
                        "metrics": create_metric_accumulator(np, histogram_edges),
                        "images": 0,
                        "identifiers": set(),
                    }
                daytime_accumulator = daytime_accumulators[daytime]
                update_metric_accumulator(
                    daytime_accumulator["metrics"], errors, histogram_edges, np
                )
                daytime_accumulator["images"] += 1
                daytime_accumulator["identifiers"].add(identifier)

                prediction_path = ""
                if not args.metrics_only:
                    prediction_path = str(prediction_dir / f"{image_path.stem}.npy")
                    np.save(prediction_path, prediction.astype(np.float32, copy=False))

                writer.writerow(
                    {
                        "image": image_path.name,
                        "ground_truth": normal_path.name,
                        "prediction": prediction_path,
                        "valid_pixels": int(errors.size),
                        **image_metrics,
                    }
                )
                csv_file.flush()

    global_metrics = finalize_metric_accumulator(
        global_accumulator, histogram_edges, np
    )
    valid_pixel_count = global_accumulator["valid_pixels"]
    daytime_metrics = {}
    daytime_csv_path = output_dir / "daytime_metrics.csv"
    with daytime_csv_path.open("w", newline="", encoding="utf-8") as daytime_csv_file:
        daytime_fieldnames = (
            "daytime",
            "images",
            "unique_ids",
            "valid_pixels",
            *METRIC_NAMES,
        )
        writer = csv.DictWriter(daytime_csv_file, fieldnames=daytime_fieldnames)
        writer.writeheader()
        for daytime in sorted(daytime_accumulators, key=int):
            accumulator = daytime_accumulators[daytime]
            metrics = finalize_metric_accumulator(
                accumulator["metrics"], histogram_edges, np
            )
            ids = sorted(accumulator["identifiers"], key=int)
            valid_pixels = accumulator["metrics"]["valid_pixels"]
            daytime_metrics[daytime] = {
                "images": accumulator["images"],
                "unique_ids": len(ids),
                "valid_pixels": valid_pixels,
                "metrics": metrics,
            }
            writer.writerow(
                {
                    "daytime": daytime,
                    "images": accumulator["images"],
                    "unique_ids": len(ids),
                    "valid_pixels": valid_pixels,
                    **metrics,
                }
            )
        baseline_metrics = random_guess_metrics()
        writer.writerow(
            {
                "daytime": "random_guess_uniform_sphere",
                "images": 0,
                "unique_ids": 0,
                "valid_pixels": 0,
                **baseline_metrics,
            }
        )
    summary = {
        "dataset_root": str(dataset_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "fov_degrees": args.fov_degrees,
        "ground_truth_encoding": args.ground_truth_encoding,
        "images_processed": len(samples),
        "valid_pixels": valid_pixel_count,
        "global_median_method": "0.01-degree histogram (within 0.01 degrees)",
        "metrics": global_metrics,
        "random_guess_baseline": {
            "assumption": "Independent uniform random unit vector on the sphere",
            "metrics": baseline_metrics,
        },
        "daytime_metrics": daytime_metrics,
        "visualized_images": visualize_count,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, allow_nan=False)
        summary_file.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Per-image metrics: {csv_path}")
    print(f"Per-daytime metrics: {daytime_csv_path}")
    if not args.metrics_only:
        print(f"Predictions: {prediction_dir}")
    if visualize_count:
        print(f"Visualizations ({visualize_count}): {visualization_dir}")


if __name__ == "__main__":
    main()
