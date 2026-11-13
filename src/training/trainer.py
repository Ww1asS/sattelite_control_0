import os

import torch
from tqdm import tqdm

from src.training.losses import CombinedLoss
from src.training.metrics import ChangeDetectionMetrics


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = config["training"]["epochs"]
        self.threshold = config["training"]["threshold"]
        self.patience = config["training"]["patience"]

        self.criterion = CombinedLoss()
        self.metrics = ChangeDetectionMetrics(threshold=self.threshold).to(device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        self.checkpoint_dir = config["paths"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_iou = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs} [train]",
            leave=False,
        )

        for x, mask in pbar:
            x = x.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0

        for x, mask in self.val_loader:
            x = x.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, mask)

            total_loss += loss.item()
            self.metrics.update(logits, mask)

        results = self.metrics.compute()
        results["loss"] = total_loss / len(self.val_loader)
        return results

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_results = self.validate()

            self.scheduler.step(val_results["iou"])

            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"Val IoU: {val_results['iou']:.4f} | "
                f"Val F1: {val_results['f1']:.4f}"
            )

            if val_results["iou"] > self.best_val_iou:
                self.best_val_iou = val_results["iou"]
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_results)
                print(f"Checkpoint saved: {self.checkpoint_dir}/best_model_epoch_{epoch}.pth")
            else:
                self.patience_counter += 1

        print("Training complete.")

    def save_checkpoint(self, epoch, val_results):
        path = os.path.join(self.checkpoint_dir, f"best_model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_iou": val_results["iou"],
                "val_f1": val_results["f1"],
            },
            path,
        )