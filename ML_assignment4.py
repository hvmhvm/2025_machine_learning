#Something ask for ChatGpt
from __future__ import annotations
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
# Assignment configuration
N_LON, N_LAT = 67, 120               # 67 (lon) × 120 (lat)
DLON = DLAT = 0.03
LON0, LAT0 = 120.00, 21.88
INVALID = -999.0

# 需要圖表就設 True（需安裝 matplotlib）
GENERATE_PLOTS = True

# ---------- 解析 XML ----------
def parse_xml_to_grid(xml_path: Path) -> np.ndarray:
    """從 <Content> 解析 67×120 科學記號數字，回傳 shape=(120,67) 的網格陣列。"""
    raw = xml_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<Content>(.*?)</Content>", raw, re.S)
    if not m:
        raise RuntimeError("No <Content> found in XML.")
    # 抓像 2.345E+01、-1.23E+00 這樣的數字
    nums = re.findall(r'[-+]?\d+\.\d+E[+-]?\d+', m.group(1))
    vals = np.array([float(x) for x in nums], dtype=float)
    if vals.size != N_LON * N_LAT:
        raise RuntimeError(f"Expected {N_LON*N_LAT} values, got {vals.size}.")
    return vals.reshape(N_LAT, N_LON)

def lonlat_mesh() -> tuple[np.ndarray, np.ndarray]:
    lons = LON0 + DLON * np.arange(N_LON)
    lats = LAT0 + DLAT * np.arange(N_LAT)
    lon2d, lat2d = np.meshgrid(lons, lats)    # (N_LAT, N_LON)
    return lon2d, lat2d

# ---------- 建立資料集 ----------
def build_datasets(grid: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    回傳四份資料：
      1) classification_df: (lon, lat, label)   # 1=有效, 0=無效
      2) regression_df:     (lon, lat, value)   # 僅有效值
      3) full_with_invalid: (lon, lat, value)   # 含 -999
      4) binary_class_df:   (lon, lat, is_valid)
    """
    lon2d, lat2d = lonlat_mesh()
    is_valid = (grid != INVALID)

    classification_df = pd.DataFrame({
        "lon": lon2d.ravel(),
        "lat": lat2d.ravel(),
        "label": is_valid.ravel().astype(int),
    })

    regression_df = pd.DataFrame({
        "lon": lon2d[is_valid].ravel(),
        "lat": lat2d[is_valid].ravel(),
        "value": grid[is_valid].ravel(),
    })

    full_with_invalid = pd.DataFrame({
        "lon": lon2d.ravel(),
        "lat": lat2d.ravel(),
        "value": grid.ravel(),  # 保留 -999
    })

    binary_class_df = pd.DataFrame({
        "lon": lon2d.ravel(),
        "lat": lat2d.ravel(),
        "is_valid": is_valid.ravel().astype(int),
    })

    return classification_df, regression_df, full_with_invalid, binary_class_df

# ---------- 訓練 + 產出指標與圖 ----------
def train_and_report(cls: pd.DataFrame, reg: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 分類：Logistic Regression ---
    Xc = cls[["lon", "lat"]].to_numpy()
    yc = cls["label"].to_numpy()
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        Xc, yc, test_size=0.2, random_state=42, stratify=yc
    )
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    clf.fit(Xc_tr, yc_tr)
    y_pred_c = clf.predict(Xc_te)
    acc = accuracy_score(yc_te, y_pred_c)
    cm = confusion_matrix(yc_te, y_pred_c)

    # --- 回歸：Linear Regression ---
    Xr = reg[["lon", "lat"]].to_numpy()
    yr = reg["value"].to_numpy()
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )
    reg_model = make_pipeline(StandardScaler(), LinearRegression())
    reg_model.fit(Xr_tr, yr_tr)
    y_pred_r = reg_model.predict(Xr_te)
    rmse = float(np.sqrt(mean_squared_error(yr_te, y_pred_r)))

    # --- metrics.txt ---
    (outdir / "metrics.txt").write_text(
        "Classification (Logistic Regression)\n"
        f"- Accuracy: {acc:.4f}\n"
        f"- Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}\n\n"
        "Regression (Linear Regression)\n"
        f"- RMSE: {rmse:.4f} °C\n",
        encoding="utf-8"
    )

    # --- Optional plots ---
    if GENERATE_PLOTS:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[warn] matplotlib not available, skip plots: {e}")
            return

        # Confusion matrix heatmap (純 matplotlib)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, origin="upper")
        ax.set_title(f"Classification Confusion Matrix (Acc={acc:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(outdir / "classification_confusion_matrix.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

        # True vs Predicted scatter for regression
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(yr_te, y_pred_r, alpha=0.5)
        mins, maxs = float(min(yr_te.min(), y_pred_r.min())), float(max(yr_te.max(), y_pred_r.max()))
        ax.plot([mins, maxs], [mins, maxs], linestyle="--")
        ax.set_xlabel("True Temperature (°C)")
        ax.set_ylabel("Predicted Temperature (°C)")
        ax.set_title(f"Regression: True vs Predicted (RMSE={rmse:.2f} °C)")
        plt.tight_layout()
        plt.savefig(outdir / "regression_true_vs_pred.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

# ---------- 找 XML ----------
def resolve_xml_path(argv: list[str]) -> Path:
    """優先用參數；否則找同層 O-A0038-003.xml；否則在專案底下 rglob 搜尋。"""
    here = Path(__file__).resolve().parent
    if len(argv) > 1:
        p = Path(argv[1]).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"指定的路徑不存在：{p}")

    p = here / "O-A0038-003.xml"
    if p.exists():
        return p

    matches = list(here.rglob("O-A0038-003*.xml"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "找不到 XML，請把檔案放到與 ML_assignment4_complete.py 同層，"
        "或用參數指定完整路徑，例如：\n"
        'python ML_assignment4_complete.py "/Users/你/路徑/O-A0038-003.xml"'
    )

# ---------- main ----------
def main():
    here = Path(__file__).resolve().parent
    xml_path = resolve_xml_path(sys.argv)
    print(f"使用 XML：{xml_path}")

    grid = parse_xml_to_grid(xml_path)
    cls, reg, full, binary = build_datasets(grid)

    outdir = here / "artifacts"      # 輸出固定在程式同層
    outdir.mkdir(parents=True, exist_ok=True)

    # 輸出四份 CSV（對應作業(1)）
    cls.to_csv(outdir / "classification.csv", index=False)
    reg.to_csv(outdir / "regression.csv", index=False)
    full.to_csv(outdir / "full_with_invalid.csv", index=False)
    binary.to_csv(outdir / "binary_classification.csv", index=False)

    # 訓練 + 指標 + 圖（對應作業(2)）
    train_and_report(cls, reg, outdir)

    print("✅ Done. 請到 artifacts/ 查看：")
    print(" - classification.csv / binary_classification.csv")
    print(" - regression.csv / full_with_invalid.csv")
    print(" - metrics.txt")
    if GENERATE_PLOTS:
        print(" - classification_confusion_matrix.png")
        print(" - regression_true_vs_pred.png")

if __name__ == "__main__":
    main()






