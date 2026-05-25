import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Import configs & models
from configs.config import config

class ExponentialMovingAverage:
    """Helper class to compute and maintain Exponential Moving Average of model parameters."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

class AegisTrainer:
    """
    NASA-grade Training Engine supporting mixed-precision (AMP),
    gradient accumulation, SWA, EMA, and telemetry logging.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        model_name: str,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        grad_accum_steps: int = 1,
        mixed_precision: bool = True,
        use_swa: bool = True,
        ema_decay: float = 0.999
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.mixed_precision = mixed_precision
        self.use_swa = use_swa
        
        # Configure output paths
        self.checkpoint_dir = Path(config.PATHS.CHECKPOINTS_DIR) / model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(Path(config.PATHS.LOGS_DIR) / model_name))
        
        # Setup Optimizer and LR Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # AMP Scaler
        self.scaler = GradScaler(enabled=self.mixed_precision)
        
        # EMA Settings
        self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
        
        # SWA Settings
        if self.use_swa:
            self.swa_model = AveragedModel(self.model).to(device)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=learning_rate * 0.1)
        else:
            self.swa_model = None
            self.swa_scheduler = None
            
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        # Compile if PyTorch >= 2.0 (for production Speedups)
        if hasattr(torch, "compile") and int(torch.__version__.split(".")[0]) >= 2:
            try:
                print(f"[{model_name}] Compiling model via torch.compile() for accelerated training graph...")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"[Warning] torch.compile not available or failed: {e}. Running in eager mode.")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "ema_shadow": self.ema.shadow
        }
        if self.use_swa and self.swa_model is not None:
            checkpoint["swa_model_state_dict"] = self.swa_model.state_dict()
            
        filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, filename)
        
        # Maintain latest symlink
        latest_file = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_file)
        
        if is_best:
            best_file = self.checkpoint_dir / "model_best.pt"
            torch.save(checkpoint, best_file)
            print(f"[*] Saved new best model checkpoint to: {best_file}")

    def resume_checkpoint(self, checkpoint_path: Optional[str] = None) -> bool:
        if checkpoint_path is None:
            checkpoint_path = str(self.checkpoint_dir / "checkpoint_latest.pt")
            
        if not os.path.exists(checkpoint_path):
            print(f"[Trainer] No checkpoint found at {checkpoint_path}. Starting training from scratch.")
            return False
            
        print(f"[*] Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.start_epoch = checkpoint["epoch"] + 1
        self.ema.shadow = checkpoint["ema_shadow"]
        
        if self.use_swa and "swa_model_state_dict" in checkpoint:
            self.swa_model.load_state_dict(checkpoint["swa_model_state_dict"])
            
        print(f"[Trainer] Checkpoint loaded. Resuming from epoch {self.start_epoch}.")
        return True

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress):
            # Unpack batch inputs
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch["input"], batch["target"]
                
            inputs = inputs.to(self.device, non_blocking=True)
            
            # Label conversion if target is label idx
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device, non_blocking=True)
            elif isinstance(targets, dict):
                targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                
            # Forward pass under AutoCast mixed-precision context
            with autocast(enabled=self.mixed_precision):
                preds = self.model(inputs)
                loss = self.loss_fn(preds, targets)
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
                
            self.scaler.scale(loss).backward()
            
            if (step + 1) % self.grad_accum_steps == 0:
                # Unscale and clip gradients
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update EMA weights
                self.ema.update()
                
            epoch_loss += loss.item() * self.grad_accum_steps
            progress.set_postfix(loss=loss.item() * self.grad_accum_steps)
            
        avg_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)
        return avg_loss

    def validate(self, epoch: int) -> float:
        # Apply EMA parameters to evaluation model if available
        self.ema.apply_shadow()
        self.model.eval()
        
        val_loss = 0.0
        progress = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in progress:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch["input"], batch["target"]
                    
                inputs = inputs.to(self.device, non_blocking=True)
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device, non_blocking=True)
                elif isinstance(targets, dict):
                    targets = {k: v.to(self.device, non_blocking=True) for k, v in targets.items()}
                    
                with autocast(enabled=self.mixed_precision):
                    preds = self.model(inputs)
                    loss = self.loss_fn(preds, targets)
                    
                val_loss += loss.item()
                progress.set_postfix(loss=loss.item())
                
        avg_loss = val_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Val", avg_loss, epoch)
        
        # Restore normal weights from backup
        self.ema.restore()
        return avg_loss

    def fit(self, epochs: int):
        print(f"[*] Starting AegisTrainer training loop for {self.model_name}...")
        print(f"[*] Training epochs: {self.start_epoch} -> {epochs}")
        print(f"[*] Hardware Acceleration Device: {self.device}")
        
        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # Step the standard scheduler
            self.scheduler.step()
            
            # Check SWA eligibility (typically during final 20% of epochs)
            if self.use_swa and epoch >= int(epochs * 0.8):
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                
            # Log VRAM metrics
            if self.device.type == "cuda":
                max_vram = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                self.writer.add_scalar("GPU/VRAM_MB", max_vram, epoch)
                print(f"Epoch {epoch} - VRAM Peak: {max_vram:.2f} MB")
                
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            # Save checkpoints regularly
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
                
        # Final SWA update if enabled
        if self.use_swa and self.swa_model is not None:
            print("[*] SWA training complete. Updating SWA BatchNorm running statistics...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            # Save final SWA model
            torch.save({"model_state_dict": self.swa_model.state_dict()}, self.checkpoint_dir / "model_swa.pt")
            print("[*] SWA model checkpoints exported successfully.")
            
        self.writer.close()
