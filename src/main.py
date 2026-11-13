import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import LEVIRCDDataset
from models.unet import build_model
from training.trainer import Trainer


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BASE_DIR / "configs" / "base.yaml"


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Change Detection Training")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    config["data"]["root"] = resolve_path(config["data"]["root"])
    config["paths"]["log_dir"] = resolve_path(config["paths"]["log_dir"])
    config["paths"]["checkpoint_dir"] = resolve_path(config["paths"]["checkpoint_dir"])

    device = get_device()
    pin_memory = device == "cuda"

    print(f"Device: {device}")
    print(f"Config: {resolve_path(args.config)}")
    print(f"Data root: {config['data']['root']}")

    # Датасеты
    train_ds = LEVIRCDDataset(
        root=config["data"]["root"],
        split="train",
        img_size=config["data"]["img_size"],
    )
    val_ds = LEVIRCDDataset(
        root=config["data"]["root"],
        split="val",
        img_size=config["data"]["img_size"],
    )

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=pin_memory,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    # Проверка одного примера
    x, mask = train_ds[0]
    print(f"Input shape: {x.shape}")        # [6, 256, 256]
    print(f"Mask shape: {mask.shape}")       # [1, 256, 256]
    print(f"Mask values: {mask.unique().tolist()}")

    # Модель
    model = build_model(config)
    print(f"Model: {model.__class__.__name__}")

    # Обучение
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()