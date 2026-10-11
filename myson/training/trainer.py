"""
MY SON TRAINER - GPU Training Pipeline
=======================================

Training loop with:
- Mixed precision (FP16) for your RTX 4070
- Gradient checkpointing to save VRAM
- Cosine learning rate schedule
- Checkpoint saving/loading
- TensorBoard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Iterator
from pathlib import Path
import time
import json
import os
import math


@dataclass
class TrainerConfig:
    """Configuration for training."""
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    lr_scheduler_type: str = "cosine"
    
    # Batch
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    
    # Mixed precision
    fp16: bool = True
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 5
    
    # Logging
    logging_steps: int = 10
    log_dir: str = "logs"
    
    # Evaluation
    eval_steps: int = 500
    
    # LoRA Configuration (train 0.1% of params, fit 7B on 12GB VRAM)
    use_lora: bool = False
    lora_r: int = 16              # LoRA rank (4-64, higher = more capacity)
    lora_alpha: int = 32          # LoRA scaling factor
    lora_dropout: float = 0.05    # Dropout for LoRA layers
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # QLoRA Configuration (4-bit quantization + LoRA)
    use_qlora: bool = False       # Use 4-bit quantization (fit 13B on 12GB)
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class MySonTrainer:
    """
    GPU Training Pipeline for My Son AI.
    
    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation for larger effective batches
    - Cosine learning rate with warmup
    - Automatic checkpointing
    - TensorBoard integration
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        train_dataloader: Iterator,
        eval_dataloader: Optional[Iterator] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Apply LoRA if enabled (BEFORE moving to device for QLoRA)
        if config.use_lora or config.use_qlora:
            self.model = self._apply_lora(self.model)
        
        self.model = self.model.to(self.device)
        print(f"[Trainer] Using device: {self.device}")
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Parameters: {trainable_params:,} trainable / {total_params:,} total ({100*trainable_params/total_params:.2f}%)")
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if config.fp16 and torch.cuda.is_available() else None
        
        # Logging
        self.writer = SummaryWriter(config.log_dir)
        
        # State
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[Trainer] Initialized")
        print(f"  - Effective batch size: {config.effective_batch_size}")
        print(f"  - Max steps: {config.max_steps}")
        print(f"  - FP16: {config.fp16}")
        print(f"  - LoRA: {config.use_lora} | QLoRA: {config.use_qlora}")
    
    def _apply_lora(self, model: nn.Module) -> nn.Module:
        """Apply LoRA adapters to the model for efficient fine-tuning."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            model = get_peft_model(model, lora_config)
            print(f"[Trainer] LoRA applied: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            model.print_trainable_parameters()
            return model
            
        except ImportError:
            print("[Trainer] WARNING: peft not installed. Install with: pip install peft")
            print("[Trainer] Falling back to full fine-tuning")
            return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
        # Separate weight decay for different parameter types
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return optim.AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(1, self.config.max_steps - self.config.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast(device_type='cuda'):
                logits, loss = self.model(input_ids, labels=labels)
                loss = loss / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            logits, loss = self.model(input_ids, labels=labels)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
        
        return {"loss": loss.item() * self.config.gradient_accumulation_steps}
    
    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")
        
        self.model.train()
        total_loss = 0.0
        start_time = time.time()
        
        accumulation_counter = 0
        
        for batch in self.train_dataloader:
            # Training step
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            accumulation_counter += 1
            
            # Gradient accumulation
            if accumulation_counter >= self.config.gradient_accumulation_steps:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                accumulation_counter = 0
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / self.config.logging_steps
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    
                    print(f"Step {self.global_step:>6} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Steps/s: {steps_per_sec:.2f}")
                    
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    
                    total_loss = 0.0
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Evaluation
                if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    print(f"  → Eval Loss: {eval_loss:.4f}")
                    self.writer.add_scalar("eval/loss", eval_loss, self.global_step)
                    
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint("best")
                
                # Check if done
                if self.global_step >= self.config.max_steps:
                    break
        
        # Final save
        self.save_checkpoint("final")
        self.writer.close()
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"  Total steps: {self.global_step}")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"{'='*60}")
        
        return {
            "total_steps": self.global_step,
            "best_loss": self.best_loss,
        }
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)
            
            logits, loss = self.model(input_ids, labels=labels)
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 50:  # Limit eval batches
                break
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Save model checkpoint."""
        if tag is None:
            tag = f"step_{self.global_step}"
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{tag}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")
        
        # Save optimizer state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }, checkpoint_path / "training_state.pt")
        
        # Save config
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)
        
        print(f"  → Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        self.model.load_state_dict(
            torch.load(checkpoint_path / "model.pt", map_location=self.device)
        )
        
        # Load training state
        state = torch.load(checkpoint_path / "training_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler and state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_loss = state["best_loss"]
        
        print(f"[Trainer] Loaded checkpoint from {checkpoint_path}")
        print(f"  - Step: {self.global_step}")
        print(f"  - Best loss: {self.best_loss:.4f}")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint_step_")],
            key=lambda x: int(x.name.split("_")[-1])
        )
        
        # Keep only the most recent ones
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
            print(f"  → Removed old checkpoint: {oldest}")




if __name__ == "__main__":
    print("="*60)
    print("MY SON TRAINER - TEST")
    print("="*60)
    
    # Create dummy model for testing
    from model import create_400m_model
    model = create_400m_model()
    
    # Create dummy data
    def dummy_dataloader(batch_size=4, seq_len=128, num_batches=100):
        for _ in range(num_batches):
            yield {
                "input_ids": torch.randint(0, 32000, (batch_size, seq_len)),
            }
    
    # Create trainer
    config = TrainerConfig(
        max_steps=20,
        logging_steps=5,
        save_steps=10,
        gradient_accumulation_steps=2,
    )
    
    trainer = MySonTrainer(
        model=model,
        config=config,
        train_dataloader=dummy_dataloader(),
    )
    
    # Train
    results = trainer.train()
    
    print(f"\nResults: {results}")
    print("\n" + "="*60)
    print("✅ TRAINER READY")
    print("="*60)
