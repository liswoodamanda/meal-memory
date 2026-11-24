"""Reusable training utilities for Meal Memory.

AI attribution: Drafted with assistance from OpenAI ChatGPT.
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run an epoch while keeping backbone running statistics fixed."""
    training = optimizer is not None
    model.eval()

    if training:
        model.classifier.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered.")

            if training:
                loss.backward()
                optimizer.step()

            count = labels.size(0)
            total_images += count
            total_loss += loss.item() * count
            total_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()

    return {
        "loss": total_loss / total_images,
        "accuracy": total_correct / total_images,
    }


def train_experiment(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    run_dir,
    results_path,
    classes,
    config,
    train_criterion=None,
):
    """Train a fresh experiment, preserving its best and latest states."""
    run_dir = Path(run_dir)
    results_path = Path(results_path)

    if any(run_dir.glob("*.pt")) or results_path.exists():
        raise FileExistsError(
            "Experiment outputs already exist; use a new run name "
            "or explicitly resume the existing experiment."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)

    initial = run_epoch(model, val_loader, criterion, device)
    best_loss = initial["loss"]
    best_epoch = 0
    stale_epochs = 0
    history = []

    def save_checkpoint(path, epoch):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "classes": list(classes),
            "class_to_idx": {
                name: index for index, name in enumerate(classes)
            },
            "config": dict(config),
            "history": list(history),
            "initial_validation": initial,
            "best_val_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": stale_epochs,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else []
            ),
            "loader_rng_state": (
                train_loader.generator.get_state()
                if train_loader.generator is not None else None
            ),
            "python_rng_state": random.getstate(),
        }, path)

    # Preserve the starting checkpoint if training never improves it.
    save_checkpoint(run_dir / "best.pt", epoch=0)

    for epoch in range(1, config["max_epochs"] + 1):
        started = time.perf_counter()
        print(
            f"Epoch {epoch}/{config['max_epochs']} starting...",
            flush=True,
        )

        train = run_epoch(
            model,
            train_loader,
            criterion if train_criterion is None else train_criterion,
            device,
            optimizer,
        )
        val = run_epoch(model, val_loader, criterion, device)

        improved = val["loss"] < best_loss
        if improved:
            best_loss = val["loss"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "seconds": time.perf_counter() - started,
        })

        save_checkpoint(run_dir / "last.pt", epoch)
        if improved:
            save_checkpoint(run_dir / "best.pt", epoch)

        pd.DataFrame(history).to_csv(results_path, index=False)

        print(
            f"Train loss {train['loss']:.4f} | "
            f"accuracy {train['accuracy']:.1%} | "
            f"Val loss {val['loss']:.4f} | "
            f"accuracy {val['accuracy']:.1%}"
            + (" | New best" if improved else ""),
            flush=True,
        )

        patience = config.get("patience")
        if patience is not None and stale_epochs >= patience:
            print("Early stopping.")
            break

    print(f"Finished. Best validation-loss checkpoint: epoch {best_epoch}")
    return pd.DataFrame(history)
