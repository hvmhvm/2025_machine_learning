import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#  Utilities 

def set_seed(seed:int=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)

# Runge function & derivative 

def runge_np(x: np.ndarray) -> np.ndarray:
    """f(x) = 1 / (1 + 25 x^2)"""
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_prime_np(x: np.ndarray) -> np.ndarray:
    """f'(x) = -50 x / (1 + 25 x^2)^2"""
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

def runge_torch(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_prime_torch(x: torch.Tensor) -> torch.Tensor:
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

# Model 

class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=64, out_dim=1, depth=2, activation='tanh'):
        super().__init__()
        acts = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
        }
        act = acts.get(activation, nn.Tanh())
        layers = []
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(act)
        for _ in range(max(0, depth-1)):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(act)
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Training / Evaluation 

def make_dataset(n_train=256, n_val=256, n_test=512, xmin=-1.0, xmax=1.0, device=torch.device('cpu')):
    # Sample uniformly in [-1,1]
    x_train = np.random.uniform(xmin, xmax, size=(n_train, 1)).astype(np.float32)
    x_val   = np.random.uniform(xmin, xmax, size=(n_val,   1)).astype(np.float32)
    x_test  = np.linspace(xmin, xmax, num=n_test, dtype=np.float32).reshape(-1, 1)

    y_train = runge_np(x_train)
    y_val   = runge_np(x_val)
    y_test  = runge_np(x_test)

    dy_train = runge_prime_np(x_train)
    dy_val   = runge_prime_np(x_val)
    dy_test  = runge_prime_np(x_test)

    data = {
        'train': (to_tensor(x_train, device), to_tensor(y_train, device), to_tensor(dy_train, device)),
        'val':   (to_tensor(x_val,   device), to_tensor(y_val,   device), to_tensor(dy_val,   device)),
        'test':  (to_tensor(x_test,  device), to_tensor(y_test,  device), to_tensor(dy_test,  device)),
        'test_np': (x_test, y_test, dy_test)
    }
    return data

def compute_derivative_nn(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return d/dx of model(x). x must require grad."""
    # Ensure requires_grad
    x_req = x.clone().detach().requires_grad_(True)
    y_hat = model(x_req)  # (N,1)
    # sum to scalar so autograd works; then grad wrt x
    grad = torch.autograd.grad(outputs=y_hat.sum(), inputs=x_req, create_graph=False)[0]
    return grad

def evaluate(model, data_split, lambda_d=1.0):
    """Compute MSE_f, MSE_df, and combined on a split (x, y, dy)."""
    x, y, dy = data_split
    # For derivative we need autograd; do a fresh forward with grad enabled
    x_req = x.clone().detach().requires_grad_(True)
    y_hat = model(x_req)
    dy_hat = torch.autograd.grad(y_hat.sum(), x_req, create_graph=False)[0]
    mse = nn.MSELoss()
    mse_f  = mse(y_hat, y)
    mse_df = mse(dy_hat, dy)
    loss = mse_f + lambda_d * mse_df
    return float(mse_f.item()), float(mse_df.item()), float(loss.item())

def train(model, data, epochs=3000, lr=1e-3, weight_decay=0.0, lambda_d=1.0, print_every=200):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    x_tr, y_tr, dy_tr = data['train']
    x_val, y_val, dy_val = data['val']

    history = {'train_f': [], 'train_df': [], 'val_f': [], 'val_df': []}

    for ep in range(1, epochs+1):
        model.train()
        # forward with gradient on input for derivative term
        x_req = x_tr.clone().detach().requires_grad_(True)
        y_hat = model(x_req)
        dy_hat = torch.autograd.grad(y_hat.sum(), x_req, create_graph=True)[0]

        loss_f  = mse(y_hat, y_tr)
        loss_df = mse(dy_hat, dy_tr)
        loss = loss_f + lambda_d * loss_df

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # log
        if ep % print_every == 0 or ep == 1 or ep == epochs:
            model.eval()
            tr_f, tr_df, tr_total = evaluate(model, (x_tr, y_tr, dy_tr), lambda_d)
            va_f, va_df, va_total = evaluate(model, (x_val, y_val, dy_val), lambda_d)
            history['train_f'].append(tr_f); history['train_df'].append(tr_df)
            history['val_f'].append(va_f);   history['val_df'].append(va_df)
            print(f"[{ep:04d}] Train: MSE_f={tr_f:.6e}  MSE_df={tr_df:.6e}  |  Val: MSE_f={va_f:.6e}  MSE_df={va_df:.6e}")

    return history

# Plotting 



def plot_fit(model, data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    x_np, y_np, dy_np = data['test_np']
    device = next(model.parameters()).device
    x = to_tensor(x_np, device)

    # Forward (pred) and derivative pred
    with torch.no_grad():
        y_hat = model(x).cpu().numpy().reshape(-1)
    x_req = x.clone().detach().requires_grad_(True)
    y_pred = model(x_req)
    dy_hat = torch.autograd.grad(y_pred.sum(), x_req, create_graph=False)[0].detach().cpu().numpy().reshape(-1)

    # Function main
    plt.figure()
    plt.title("Runge function fit")
    plt.plot(x_np.reshape(-1), y_np.reshape(-1), label="true f(x)", color="blue", linewidth=2.5)
    plt.plot(x_np.reshape(-1), y_hat, label="NN f_hat(x)", color="orange", linestyle="--", linewidth=2.0)
    plt.legend(); plt.xlabel("x"); plt.ylabel("y")
    f_func = os.path.join(out_dir, "fit_function.png")
    plt.savefig(f_func, dpi=160, bbox_inches="tight"); plt.close()

    # Function error
    plt.figure()
    plt.title("Function error: true f(x) - NN f_hat(x)")
    plt.plot(x_np.reshape(-1), y_np.reshape(-1) - y_hat, label="error", linewidth=2.0)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend(); plt.xlabel("x"); plt.ylabel("error")
    f_func_err = os.path.join(out_dir, "fit_function_error.png")
    plt.savefig(f_func_err, dpi=160, bbox_inches="tight"); plt.close()

    # Function zoomed
    plt.figure()
    plt.title("Runge function fit (zoomed [-0.2,0.2])")
    plt.plot(x_np.reshape(-1), y_np.reshape(-1), label="true f(x)", color="blue", linewidth=2.5)
    plt.plot(x_np.reshape(-1), y_hat, label="NN f_hat(x)", color="orange", linestyle="--", linewidth=2.0)
    plt.xlim(-0.2, 0.2)
    plt.legend(); plt.xlabel("x"); plt.ylabel("y")
    f_func_zoom = os.path.join(out_dir, "fit_function_zoom.png")
    plt.savefig(f_func_zoom, dpi=160, bbox_inches="tight"); plt.close()

    #  Derivative main 
    plt.figure()
    plt.title("Runge derivative fit")
    plt.plot(x_np.reshape(-1), dy_np.reshape(-1), label="true f'(x)", color="blue", linewidth=2.5)
    plt.plot(x_np.reshape(-1), dy_hat, label="NN d/dx f_hat(x)", color="orange", linestyle="--", linewidth=2.0)
    plt.legend(); plt.xlabel("x"); plt.ylabel("dy/dx")
    f_deriv = os.path.join(out_dir, "fit_derivative.png")
    plt.savefig(f_deriv, dpi=160, bbox_inches="tight"); plt.close()

    # Derivative error
    plt.figure()
    plt.title("Derivative error: true f'(x) - NN d/dx f_hat(x)")
    plt.plot(x_np.reshape(-1), dy_np.reshape(-1) - dy_hat, label="error", linewidth=2.0)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.legend(); plt.xlabel("x"); plt.ylabel("error")
    f_deriv_err = os.path.join(out_dir, "fit_derivative_error.png")
    plt.savefig(f_deriv_err, dpi=160, bbox_inches="tight"); plt.close()

    # Derivative zoomed
    plt.figure()
    plt.title("Runge derivative fit (zoomed [-0.2,0.2])")
    plt.plot(x_np.reshape(-1), dy_np.reshape(-1), label="true f'(x)", color="blue", linewidth=2.5)
    plt.plot(x_np.reshape(-1), dy_hat, label="NN d/dx f_hat(x)", color="orange", linestyle="--", linewidth=2.0)
    plt.xlim(-0.2, 0.2)
    plt.legend(); plt.xlabel("x"); plt.ylabel("dy/dx")
    f_deriv_zoom = os.path.join(out_dir, "fit_derivative_zoom.png")
    plt.savefig(f_deriv_zoom, dpi=160, bbox_inches="tight"); plt.close()

    return f_func, f_deriv

#  Main Function

def main():
    parser = argparse.ArgumentParser(description="Approximate Runge function and its derivative with an NN")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2, help="number of hidden layers")
    parser.add_argument("--activation", type=str, default="tanh", choices=["tanh","relu","gelu","sigmoid"])
    parser.add_argument("--lambda_d", type=float, default=1.0, help="weight on derivative MSE")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--n_train", type=int, default=256)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./assignment3_outputs")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = make_dataset(n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, device=device)

    model = MLP(in_dim=1, hidden=args.hidden, out_dim=1, depth=args.depth, activation=args.activation).to(device)

    _ = train(model, data, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
              lambda_d=args.lambda_d, print_every=max(1, args.epochs//15))

    # Final evaluation (test set)
    test_f, test_df, test_total = evaluate(model, data['test'], lambda_d=args.lambda_d)
    print("\n=== Test metrics ===")
    print(f"MSE_f (function):   {test_f:.6e}")
    print(f"MSE_df (derivative): {test_df:.6e}")
    print(f"Combined (λ={args.lambda_d:g}): {test_total:.6e}")

    # Save plots
    f_fig, df_fig = plot_fit(model, data, args.out_dir)
    print(f"\nSaved plots to:\n  {f_fig}\n  {df_fig}")

    # Save a tiny markdown report
    report_path = os.path.join(args.out_dir, "report.md")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Assignment 3: Runge Function + Derivative\n")
        f.write(f"- Hidden: {args.hidden}, Depth: {args.depth}, Activation: {args.activation}\n")
        f.write(f"- λ_derivative: {args.lambda_d}, lr: {args.lr}, epochs: {args.epochs}\n\n")
        f.write("## Test MSE\n")
        f.write(f"- Function MSE: {test_f:.6e}\n")
        f.write(f"- Derivative MSE: {test_df:.6e}\n")
        f.write(f"- Combined: {test_total:.6e}\n")
        f.write("\nFigures saved alongside this report.\n")
    print(f"Report written to: {report_path}")

if __name__ == "__main__":
    main()


