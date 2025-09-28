import argparse
import torch
from train_cost_coeffs import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=10, help="horizon length")
    parser.add_argument("--n_batch", type=int, default=12, help="batch count")
    parser.add_argument("--n_epochs", type=int, default=50, help="epoch count")
    parser.add_argument("--n_train", type=int, default=64, help="train samples count")
    parser.add_argument("--n_test", type=int, default=8, help="test samples count")
    parser.add_argument("--learn_rate", type=float, default=5e-3, help="optimizer lr")
    parser.add_argument(
        "--freeze_idx",
        type=int,
        nargs="*",
        default=[0, 2],  # 0 - X, 2 - cos theta
        help="index of goal wts to not backprop",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("USING DEVICE:", device)

    train(
        args.T,
        args.n_batch,
        args.n_epochs,
        args.n_train,
        args.n_test,
        args.learn_rate,
        args.freeze_idx,
        device,
    )

# Run 1: RMSProp LR: 5e-3
# Start at:
# Epoch Loss:  2.9116614818573
# Weights:  tensor([0.1000, 0.1000, 0.1099, 0.1000, 0.1049], grad_fn=<AddBackward0>)
# Weights:  tensor([X, 0.1000, 0.1099, X, 0.1049], grad_fn=<AddBackward0>)
# Breaken at:
# Epoch Loss:  1.4234720468521118
# Weights:  tensor([0.1000, 0.0810, 0.0869, 0.1000, 0.3558], grad_fn=<AddBackward0>)
# Weights:  tensor([X, 0.0810, 0.0869, X, 0.3558], grad_fn=<AddBackward0>)
