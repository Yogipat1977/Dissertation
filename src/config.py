"""
config.py — Load and resolve YAML configuration.
"""

import yaml
import os
from datetime import datetime
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load a YAML config file and resolve all relative paths against
    the project root directory."""
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Project root = directory containing the configs/ folder
    project_root = config_path.parent.parent
    cfg["_project_root"] = project_root
    cfg["_config_path"] = config_path

    # Resolve relative paths against project root
    cfg["data"]["data_dir"] = str(project_root / cfg["data"]["data_dir"])
    if cfg["data"].get("cache_dir"):
        cfg["data"]["cache_dir"] = str(project_root / cfg["data"]["cache_dir"])
    else:
        cfg["data"]["cache_dir"] = ""
    cfg["paths"]["models_dir"] = str(project_root / cfg["paths"]["models_dir"])
    cfg["paths"]["results_dir"] = str(project_root / cfg["paths"]["results_dir"])

    # Ensure ROI size is a tuple
    cfg["data"]["roi_size"] = tuple(cfg["data"]["roi_size"])

    # Generate a unique run name
    cfg["_run_name"] = _generate_run_name(cfg)

    # Create output directories
    run_dir = Path(cfg["paths"]["models_dir"]) / cfg["_run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg["_run_dir"] = str(run_dir)

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    return cfg


def _generate_run_name(cfg: dict) -> str:
    """Generate a descriptive run name from config values."""
    arch = cfg["model"]["architecture"]
    filters = cfg["model"]["init_filters"]
    dropout = cfg["model"]["dropout_prob"]
    lr = cfg["training"]["learning_rate"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{arch}_f{filters}_d{dropout}_lr{lr}_{timestamp}"


def save_config(cfg: dict, output_dir: str):
    """Save a copy of the config alongside the checkpoint for
    reproducibility. Strips internal keys (prefixed with _)."""
    out_path = Path(output_dir) / "config.yaml"
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(out_path, "w") as f:
        yaml.dump(clean_cfg, f, default_flow_style=False, sort_keys=False)
