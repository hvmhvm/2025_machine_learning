# 111652041 assignment2 for programming
# 用神經網路逼近 Runge 函數 f(x) = 1 / (1 + 25 x^2)
# 產生圖：real function vs NN ：預測、訓練/驗證損失
# output：測試 MSE 與最大誤差、Markdown 


import os
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim

# step0) 基本設定 
# 設定隨機種子，讓實驗可重現（numpy 與 torch 分別設定）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 輸出資料夾（圖與報告將存這裡）
OUT_DIR = "./runge_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# using CPU
DEVICE = torch.device("cpu")

# step1) 定義目標函數（Runge function）
def runge_np(x: np.ndarray) -> np.ndarray:
    """
    以 NumPy 計算 Runge 函數，方便在資料產生與畫圖時使用
    參數：
        x: shape = (N, 1) 或 (N,)
    回傳：
        f(x) = 1 / (1 + 25 x^2)，形狀與 x 對齊
    """
    return 1.0 / (1.0 + 25.0 * np.square(x))

def runge_torch(x: torch.Tensor) -> torch.Tensor:
    """
    以 PyTorch 計算 Runge 函數（雖然訓練時不直接用到，但可驗證/比對）
    """
    return 1.0 / (1.0 + 25.0 * x**2)

# step 2) 產生資料集
# 我們在區間 [-1, 1] 上均勻抽樣 N 個 x，並計算 y = f(x)
N = 800  # 總樣本數（可以調整）
X_all = np.random.uniform(-1.0, 1.0, size=(N, 1)).astype(np.float32)  # shape = (N, 1)
y_all = runge_np(X_all).astype(np.float32)                             # shape = (N, 1)

# 將資料切成：訓練 60% / 驗證 20% / 測試 20%
perm = np.random.permutation(N)  # 打亂索引index
n_train = int(0.6 * N)
n_val   = int(0.2 * N)
idx_tr  = perm[:n_train]
idx_va  = perm[n_train:n_train + n_val]
idx_te  = perm[n_train + n_val:]

X_tr, y_tr = X_all[idx_tr], y_all[idx_tr]
X_va, y_va = X_all[idx_va], y_all[idx_va]
X_te, y_te = X_all[idx_te], y_all[idx_te]

# 轉成 PyTorch Tensor，並放到裝置（DEVICE）
Xtr_t = torch.from_numpy(X_tr).to(DEVICE)   # shape (n_train, 1)
ytr_t = torch.from_numpy(y_tr).to(DEVICE)   # shape (n_train, 1)
Xva_t = torch.from_numpy(X_va).to(DEVICE)
yva_t = torch.from_numpy(y_va).to(DEVICE)
Xte_t = torch.from_numpy(X_te).to(DEVICE)
yte_t = torch.from_numpy(y_te).to(DEVICE)

# step3) 建立神經網路（MLP）
# 選擇 tanh 當 activation：因為目標函數平滑、連續，tanh 的平滑特性很適合逼近
# 結構：1 -> 64 -> 64 -> 1
model = nn.Sequential(
    nn.Linear(1, 64),  # 輸入維度 1（x），第一層 64 個神經元
    nn.Tanh(),         # 非線性轉換（提供非線性能力）
    nn.Linear(64, 64), # 第二層 64 個神經元
    nn.Tanh(),
    nn.Linear(64, 1)   # 輸出維度 1（預測 y）
).to(DEVICE)

# Loss fuction（MSE）：L(θ) = 平均 (y_hat - y)^2
criterion = nn.MSELoss()

# 最佳化器：Adam，學習率 1e-3，並加 weight_decay 做 L2 正則化以防過度擬合
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# step4) 訓練流程（含早停 Early Stopping） 
# 我們在每個 epoch：
#   1. 前向傳播 -> 計算訓練損失 -> 反向傳播 -> 更新權重
#   2. 在驗證集計算 vloss 以觀察泛化；若 vloss 連續多次沒有明顯進步，就停止（早停）
train_losses = []   # 紀錄每個 epoch 的訓練損失
val_losses   = []   # 紀錄每個 epoch 的驗證損失

best_val = float("inf")   # 目前最好的驗證損失
best_state = None         # 目前最佳模型參數
patience = 150            # 容忍沒有進步的最大連續 epoch 數
wait = 0                  # 目前已連續沒有進步的次數
max_epochs = 2000         # 訓練最多跑多少 epoch（早停通常會更早停止）

for epoch in range(max_epochs):
    # 1.訓練模式（啟用 dropout/bn 等訓練行為；這裡沒有用到，但習慣這麼做） 
    model.train()
    optimizer.zero_grad()         # 將上次反向傳播累積的梯度清零
    yhat_tr = model(Xtr_t)        # 前向傳播得到訓練預測（shape 對齊 ytr_t）
    loss_tr = criterion(yhat_tr, ytr_t)  # 計算訓練 MSE 損失
    loss_tr.backward()            # 反向傳播計算梯度
    optimizer.step()              # 依梯度更新權重

    # 2.驗證模式（關閉訓練態行為，僅 forward，不做梯度計算）
    model.eval()
    with torch.no_grad():
        yhat_va = model(Xva_t)               # 驗證集預測
        loss_va = criterion(yhat_va, yva_t)  # 驗證損失（用於早停）

    # 3.紀錄損失
    train_losses.append(loss_tr.item())
    val_losses.append(loss_va.item())

    # 4.早停邏輯 
    # 若驗證損失有明顯改善（小於目前最佳值一點點），更新最佳模型與計數器
    if loss_va.item() < best_val - 1e-7:
        best_val = loss_va.item()
        # 儲存目前最佳權重（複製到 CPU，避免被覆寫）
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1

    # 若已經連續多個 epoch 沒進步，停止訓練
    if wait > patience:
        print(f"[Early Stopping] epoch={epoch+1}, best_val={best_val:.6e}")
        break

# 訓練完成後，把最佳權重載回模型（確保後續評估/繪圖用到最佳點）
if best_state is not None:
    model.load_state_dict(best_state)

# step5) 在測試集上計算指標 
model.eval()
with torch.no_grad():
    yhat_te = model(Xte_t).cpu().numpy()  # 預測結果轉回 NumPy，便於計算
# MSE：平均平方誤差
mse_test = float(np.mean((yhat_te - y_te)**2))
# Max |error|：最大絕對誤差
max_err_test = float(np.max(np.abs(yhat_te - y_te)))
print(f"[Test] MSE = {mse_test:.6e},  Max |error| = {max_err_test:.6e}")

# 6) 繪圖並且輸出：真實函數 vs NN predict 
# 產生密集 x 網格，讓曲線看起來平滑
x_plot = np.linspace(-1, 1, 800, dtype=np.float32).reshape(-1, 1)  # 800 個點
with torch.no_grad():
    y_pred_plot = model(torch.from_numpy(x_plot).to(DEVICE)).cpu().numpy().reshape(-1)
y_true_plot = runge_np(x_plot).reshape(-1)

plt.figure(figsize=(6, 4))
plt.plot(x_plot.reshape(-1), y_true_plot, label="True f(x)")     # 真實函數曲線
plt.plot(x_plot.reshape(-1), y_pred_plot, label="NN prediction") # 神經網路預測曲線
plt.scatter(X_tr.reshape(-1), y_tr.reshape(-1), s=6, alpha=0.3, label="Train pts")  # 訓練樣本點
plt.title("Runge function vs. Neural Network Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "true_vs_pred.png")
plt.savefig(fig1_path, dpi=160)
plt.close()

# step7) 繪圖：訓練/驗證損失曲線 
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(train_losses)+1), train_losses, label="Train loss")
plt.plot(np.arange(1, len(val_losses)+1),   val_losses,   label="Val loss")
plt.title("Training and Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "loss_curves.png")
plt.savefig(fig2_path, dpi=160)
plt.close()


# 1) 想在邊界更準確：可以把取樣在 |x| 接近 1 的區域加密（或用 Chebyshev nodes）
# 2) 想更低誤差：把隱藏層寬度從 64 提到 128，或增加層數，但記得調整早停/正則
# 3) 如果 loss 波動：把學習率從 1e-3 降到 5e-4；或增大 patience
# 4) 也可以嘗試其他 activation（如 GELU/SiLU），但對這題 tanh 很穩 

