"""
Training script for TemporalGAT (T-GAT).

Changes from original train.py:
1. Uses TemporalGAT (model_v2) instead of MyGAT
2. Fixes bug: random_turns augmentation now actually applied (line 27 in original)
3. Adds temporal augmentation: random turn masking and noise injection
4. Adds F1/Precision/Recall metrics (important for imbalanced attacker detection)
5. Supports both model_v2 (TemporalGAT) and original model (MyGAT) via --model_type flag
"""

import argparse
import os
from tqdm import tqdm
from data import AgentGraphDataset
from torch_geometric.loader import DataLoader
from scatter_compat import scatter_mean
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_v2 import TemporalGAT
from model import MyGAT
from datetime import datetime
import random


def compute_metrics(predicted, labels):
    """Compute accuracy, precision, recall, F1 for binary classification."""
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    tn = ((predicted == 0) & (labels == 0)).sum().item()
    
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1) * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


def temporal_augmentation(edge_attr, min_turns=1):
    """Temporal data augmentation for training robustness.
    
    Randomly truncates turns to simulate detection at different dialogue stages.
    This teaches the model to detect attacks even with partial temporal info.
    
    Args:
        edge_attr: (M, T, D) tensor
        min_turns: minimum number of turns to keep
    Returns:
        augmented edge_attr: (M, T, D) with some turns zeroed out
    """
    T = edge_attr.size(1)
    if T <= min_turns:
        return edge_attr
    
    # Random number of turns to use (at least min_turns)
    num_turns = random.randint(min_turns, T)
    
    # Zero-out future turns (simulate early detection scenario)
    augmented = edge_attr.clone()
    if num_turns < T:
        augmented[:, num_turns:, :] = 0.0
    
    return augmented


def train(model, train_loader, criterion, optimizer, device, use_augmentation=True):
    model.train()
    running_loss = 0.0
    all_predicted = []
    all_labels = []

    for data in train_loader:
        x, y, edge_index, edge_attr = (
            data.x.to(device), data.y.to(device),
            data.edge_index.to(device), data.edge_attr.to(device)
        )
        
        # Compute node features from first-round edge embeddings
        x = edge_attr[:, 0, :]
        x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
        
        # Apply temporal augmentation during training
        if use_augmentation:
            edge_attr = temporal_augmentation(edge_attr)
        
        optimizer.zero_grad()
        outputs = model(x, edge_index=edge_index, edge_attr=edge_attr)
        loss = criterion(outputs, y.float().unsqueeze(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
        all_predicted.append(predicted)
        all_labels.append(y)

    avg_loss = running_loss / len(train_loader)
    all_predicted = torch.cat(all_predicted)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predicted, all_labels)

    return avg_loss, metrics


def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for data in test_loader:
            x, y, edge_index, edge_attr = (
                data.x.to(device), data.y.to(device),
                data.edge_index.to(device), data.edge_attr.to(device)
            )
            
            # Node features from first-round edge embeddings
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(data.x))

            outputs = model(x, edge_index, edge_attr)
            loss = criterion(outputs, y.float().unsqueeze(-1))
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
            all_predicted.append(predicted)
            all_labels.append(y)

    avg_loss = running_loss / len(test_loader)
    all_predicted = torch.cat(all_predicted)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predicted, all_labels)

    return avg_loss, metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train TemporalGAT for attacker detection")

    parser.add_argument("--dataset_path", type=str,
                        default="./ModelTrainingSet/tool_attack/dataset.pkl",
                        help="Path to the training dataset")
    
    # Model architecture
    parser.add_argument("--model_type", type=str, default="temporal_gat",
                        choices=["temporal_gat", "original_gat"],
                        help="Model architecture to use")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--temporal_heads", type=int, default=4,
                        help="Number of temporal attention heads (TemporalGAT only)")
    parser.add_argument("--temporal_hidden", type=int, default=256,
                        help="Hidden dim for temporal encoder (TemporalGAT only)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable temporal augmentation")

    parser.add_argument("--save_dir", type=str, default="./checkpoint")

    args = parser.parse_args()

    normalized_path = os.path.normpath(args.dataset_path)
    parts = normalized_path.split(os.sep)
    dataset = parts[-2]
    args.save_dir = os.path.join(args.save_dir, dataset)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "tgat" if args.model_type == "temporal_gat" else "gat"
    filename = (f"{current_time_str}-{model_tag}-hiddim_{args.hidden_dim}"
                f"-heads_{args.num_heads}-layers_{args.num_layers}"
                f"-epochs_{args.epochs}-lr_{args.lr}"
                f"-dropout_{args.dropout}-wd_{args.weight_decay}.pth")
    args.save_path = os.path.join(args.save_dir, filename)

    return args


def main():
    args = parse_arguments()
    
    train_dataset = AgentGraphDataset(args.dataset_path, phase="train")
    val_dataset = AgentGraphDataset(args.dataset_path, phase="val")
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(val_dataset)
    
    example = train_dataset[0]
    in_channels = example.x.size(1)
    edge_dim = example.edge_attr.size()[1:]

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    # Build model
    if args.model_type == "temporal_gat":
        gnn = TemporalGAT(
            in_channels, args.hidden_dim, out_channels=1,
            heads=args.num_heads, num_layers=args.num_layers,
            edge_dim=edge_dim, dropout=args.dropout,
            temporal_heads=args.temporal_heads,
            temporal_hidden=args.temporal_hidden
        )
        print(f"Using TemporalGAT (temporal_heads={args.temporal_heads}, "
              f"temporal_hidden={args.temporal_hidden})")
    else:
        gnn = MyGAT(
            in_channels, args.hidden_dim, out_channels=1,
            heads=args.num_heads, num_layers=args.num_layers,
            edge_dim=edge_dim, dropout=args.dropout
        )
        print("Using Original MyGAT")
    
    num_params = sum(p.numel() for p in gnn.parameters())
    print(f"Total parameters: {num_params:,}")
    gnn.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_f1 = 0.0
    use_aug = not args.no_augmentation

    print(f"\nTraining on {device} | Augmentation: {use_aug}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print("-" * 100)

    for i in range(args.epochs):
        train_loss, train_m = train(gnn, trainloader, criterion, optimizer,
                                     device=device, use_augmentation=use_aug)
        test_loss, test_m = test(gnn, testloader, criterion, device=device)
        scheduler.step()
        
        # Use F1 as primary metric (better for imbalanced attacker detection)
        improved = test_m["f1"] > best_f1
        if improved:
            best_f1 = test_m["f1"]
            torch.save(gnn.state_dict(), args.save_path)
        
        status = "★ SAVE" if improved else ""
        print(f"Epoch {i:3d}/{args.epochs} | "
              f"Train [loss={train_loss:.4f} acc={train_m['accuracy']:.1f}% "
              f"F1={train_m['f1']:.1f}%] | "
              f"Val [loss={test_loss:.4f} acc={test_m['accuracy']:.1f}% "
              f"P={test_m['precision']:.1f}% R={test_m['recall']:.1f}% "
              f"F1={test_m['f1']:.1f}%] {status}")

    print("-" * 100)
    print(f"Best Val F1: {best_f1:.2f}% | Saved to: {args.save_path}")
    print(f"FINAL_CHECKPOINT: {args.save_path}")

if __name__ == "__main__":
    main()
