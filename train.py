"""
End-to-end supervised fine-tuning of DermLIP on HAM10000 (7-class classification).
- Distributed via PyTorch DDP across N nodes x G GPUs.
- Auto-detects single-process mode for local smoke testing (no DDP/NCCL).
- Saves per-epoch checkpoints + best-by-balanced-accuracy.
- Auto-resumes from latest checkpoint on restart (spot-preemption safe).
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import open_clip


# ----------------------- Args -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--metadata-csv", type=str, default="HAM10000_metadata.csv")
    p.add_argument("--image-subdirs", nargs="+",
                   default=["HAM10000_images_part_1", "HAM10000_images_part_2"])
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--model-name", type=str, default="hf-hub:redlessone/DermLIP_ViT-B-16")
    p.add_argument("--finetune-mode", choices=["full", "linear", "lora"], default="full")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nodes", type=int, default=int(os.environ.get("AZUREML_NODE_COUNT", "1")))
    p.add_argument("--gpus-per-node", type=int, default=torch.cuda.device_count())
    p.add_argument("--output-dir", type=str, default="./outputs")
    return p.parse_args()


# ----------------------- Dataset -----------------------
HAM_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(HAM_CLASSES)}


class HAM10000(Dataset):
    def __init__(self, df, image_index, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.image_index[row["image_id"]]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, CLASS_TO_IDX[row["dx"]]


def build_image_index(data_root, subdirs):
    idx = {}
    for sd in subdirs:
        for p in Path(data_root, sd).glob("*.jpg"):
            idx[p.stem] = str(p)
    return idx


# ----------------------- Model -----------------------
class DermLIPClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(model_name, num_classes, mode, lora_rank):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    visual = model.visual
    embed_dim = getattr(visual, "output_dim", None) or visual.proj.shape[1]

    if mode == "linear":
        for p in visual.parameters():
            p.requires_grad = False
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model
        for p in visual.parameters():
            p.requires_grad = False
        cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_rank * 2, lora_dropout=0.1,
            target_modules=["qkv", "proj"], bias="none",
        )
        visual = get_peft_model(visual, cfg)
    return DermLIPClassifier(visual, embed_dim, num_classes), preprocess


# ----------------------- Distributed helpers -----------------------
def setup_ddp():
    """Returns (local_rank, rank, world_size, distributed)."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 0, 1, False
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), True


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg, flush=True)


# ----------------------- Train / Eval -----------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, epoch):
    model.train()
    running, steps = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()
        steps += 1
        if steps % 50 == 0:
            log(f"epoch {epoch} step {steps} loss {loss.item():.4f} "
                f"lr {scheduler.get_last_lr()[0]:.2e}")
    return running / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, world_size, distributed):
    model.eval()
    preds, labs = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(imgs)
        preds.append(logits.argmax(1).cpu())
        labs.append(labels)
    preds = torch.cat(preds).numpy()
    labs = torch.cat(labs).numpy()
    acc = accuracy_score(labs, preds)
    bal = balanced_accuracy_score(labs, preds)
    f1 = f1_score(labs, preds, average="macro")

    if distributed:
        t = torch.tensor([acc, bal, f1], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
        return t[0].item(), t[1].item(), t[2].item()
    return acc, bal, f1


# ----------------------- Checkpointing -----------------------
def save_ckpt(path, epoch, model, optimizer, scheduler, scaler, best_bal_acc, args, distributed):
    state = model.module.state_dict() if distributed else model.state_dict()
    torch.save({
        "epoch": epoch,
        "model": state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_bal_acc": best_bal_acc,
        "args": vars(args),
    }, path)


def maybe_resume(out_dir, model, optimizer, scheduler, scaler, local_rank, distributed):
    ckpt_path = out_dir / "latest.pt"
    if not ckpt_path.exists():
        return 0, 0.0
    log(f"[resume] loading {ckpt_path}")
    map_location = {"cuda:0": f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=map_location)
    target = model.module if distributed else model
    target.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_bal_acc = ckpt["best_bal_acc"]
    log(f"[resume] starting epoch={start_epoch} best_bal_acc={best_bal_acc:.4f}")
    return start_epoch, best_bal_acc


# ----------------------- Main -----------------------
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    local_rank, rank, world_size, distributed = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    out_dir = Path(args.output_dir)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[setup] nodes={args.nodes} gpus_per_node={args.gpus_per_node} "
        f"world_size={world_size} distributed={distributed} mode={args.finetune_mode} device={device}")

    # --- Data ---
    df = pd.read_csv(Path(args.data) / args.metadata_csv)
    df = df[df["dx"].isin(HAM_CLASSES)].copy()
    train_df, val_df = train_test_split(
        df, test_size=args.val_frac, stratify=df["dx"], random_state=args.seed,
    )
    image_index = build_image_index(args.data, args.image_subdirs)
    log(f"[data] train={len(train_df)} val={len(val_df)} images={len(image_index)}")

    # --- Model ---
    model, preprocess = build_model(
        args.model_name, num_classes=len(HAM_CLASSES),
        mode=args.finetune_mode, lora_rank=args.lora_rank,
    )
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # --- Transforms ---
    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        preprocess.transforms[-2],
        preprocess.transforms[-1],
    ])
    val_tf = preprocess

    train_ds = HAM10000(train_df, image_index, train_tf)
    val_ds = HAM10000(val_df, image_index, val_tf)

    if distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    # --- Optim / Loss ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    class_counts = train_df["dx"].value_counts().reindex(HAM_CLASSES).values
    class_weights = torch.tensor(
        (1.0 / class_counts) * (class_counts.sum() / len(HAM_CLASSES)),
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Resume if possible ---
    start_epoch, best_bal_acc = maybe_resume(
        out_dir, model, optimizer, scheduler, scaler, local_rank, distributed
    )

    # --- Train loop ---
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion, device, epoch
        )
        val_acc, val_bal, val_f1 = evaluate(model, val_loader, device, world_size, distributed)

        log(f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_acc={val_acc:.4f} val_bal_acc={val_bal:.4f} val_macro_f1={val_f1:.4f}")

        if is_main():
            save_ckpt(out_dir / "latest.pt", epoch, model, optimizer, scheduler,
                      scaler, max(best_bal_acc, val_bal), args, distributed)
            if val_bal > best_bal_acc:
                best_bal_acc = val_bal
                save_ckpt(out_dir / "best.pt", epoch, model, optimizer, scheduler,
                          scaler, best_bal_acc, args, distributed)
                with open(out_dir / "best_metrics.json", "w") as f:
                    json.dump({"epoch": epoch, "val_acc": val_acc,
                               "val_bal_acc": val_bal, "val_macro_f1": val_f1}, f, indent=2)

        if distributed:
            dist.barrier()

    log(f"[done] best balanced acc = {best_bal_acc:.4f}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
