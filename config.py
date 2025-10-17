"""
Configuration file for GIL-DDI model training
"""
import argparse
import torch


def get_config():
    """Get configuration for training"""
    parser = argparse.ArgumentParser(description='GIL-DDI: Multi-view Graph Invariant Learning for DDI Prediction')

    # Training parameters
    parser.add_argument("--task", type=int, choices=[1, 2, 3], default=1,
                        help="Task number: 1=known-known, 2=known-new, 3=new-new")
    parser.add_argument("--epochs", type=int, default=120,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024,
                        choices=[128, 256, 512, 1024, 2048],
                        help="Batch size for training")
    parser.add_argument("--learning_rate", "--lr", type=float, default=0.001,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-8,
                        help="Weight decay (L2 regularization)")

    # Model parameters
    parser.add_argument("--embedding_dim", type=int, default=256,
                        choices=[32, 64, 128, 256],
                        help="Dimension of drug embeddings")
    parser.add_argument("--neighborhood_size", type=int, default=6,
                        choices=[4, 6, 8, 10, 16],
                        help="Size of neighborhood sample")
    parser.add_argument("--attention_heads", type=int, default=8,
                        choices=[1, 2, 4, 8],
                        help="Number of attention heads")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout rate for regularization")

    # Data parameters
    parser.add_argument("--n_drug", type=int, default=572,
                        help="Number of drugs in dataset")
    parser.add_argument("--event_num", type=int, default=65,
                        help="Number of DDI event types")

    # System parameters
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: 'auto', 'cpu', 'cuda', or 'cuda:X'")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")

    # Logging and saving
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for saving logs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory for saving model checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def print_config(args):
    """Print configuration"""
    print("\n" + "="*50)
    print("GIL-DDI Configuration")
    print("="*50)
    for arg in vars(args):
        print(f"{arg:.<30} {getattr(args, arg)}")
    print("="*50 + "\n")

