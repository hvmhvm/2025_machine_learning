# week6_all_in_one.py
# 完整版：資料讀取(自動從 XML 產 CSV) + QDA 分類 + 多項式 Ridge 回歸 + h(x) 輸出
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xml.etree.ElementTree as ET

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from numpy.linalg import pinv

OUTDIR = Path("outputs")
CSV_PATH = OUTDIR / "taiwan_grid.csv"
XML_PATH = Path("O-A0038-003.xml")
MISSING = -999.0

# ---------------------------
# (A) 如果沒有 CSV，就從 XML 產生
# ---------------------------
def parse_grid_from_xml(xml_path: Path, n_lat=120, n_lon=67):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # 嘗試抓 namespace
    ns_uri = None
    for k in root.attrib:
        if "}" in k:
            ns_uri = k.split("}")[0].strip("{")
            break
    ns = {"cwa": ns_uri or "urn:cwa:gov:tw:cwacommon:0.1"}
    content_el = root.find(".//cwa:Content", ns)
    if content_el is None or not (content_el.text and content_el.text.strip()):
        raise ValueError("XML 檔沒有 <Content> 或內容為空。")
    text = content_el.text

    # 篩掉含 '~' 的非數據行，串起來後用正則抓數字(含科學記號)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    data_lines = [ln for ln in lines if ("," in ln and "~" not in ln)]
    if not data_lines:
        data_lines = [max(lines, key=lambda s: s.count(","))]
    joined = ",".join(data_lines)
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?", joined)
    vals = [float(s) for s in nums]

    need = n_lat * n_lon
    if len(vals) < need:
        raise ValueError(f"從 XML 擷取數字 {len(vals)} 個，少於期望 {need} 個。")
    vals = vals[-need:]  # 取最後 need 個（避免前面雜訊數字）
    arr = np.array(vals, float).reshape(n_lat, n_lon)
    return arr

def ensure_csv(csv_path: Path, xml_path: Path):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    if not xml_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}，也沒有 {xml_path} 可用來產生資料。請至少提供其一。"
        )
    print("[INFO] 找不到 CSV，改由 XML 自動產生 taiwan_grid.csv ...")
    data = parse_grid_from_xml(xml_path, n_lat=120, n_lon=67)
    lon0, lat0, d = 120.0, 21.88, 0.03
    lon = lon0 + np.arange(67) * d
    lat = lat0 + np.arange(120) * d
    lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
    df = pd.DataFrame({"lon": lon2d.ravel(), "lat": lat2d.ravel(), "value": data.ravel()})
    df.to_csv(csv_path, index=False)
    print(f"[INFO] 已從 XML 產生 CSV -> {csv_path}")

# ---------------------------
# (B) 回歸相關工具
# ---------------------------
def geo_poly_features(X, degree:int):
    """
    地理多項式特徵：
    [1, lon, lat, lon*lat, (lon-c)^2, (lat-c)^2, r^2, ...(到三階)]
    """
    X = np.asarray(X, float)
    lon, lat = X[:,0], X[:,1]
    lon_c, lat_c = lon - lon.mean(), lat - lat.mean()
    feats = [np.ones(len(X)), lon, lat]
    if degree >= 2:
        feats += [lon*lat, lon_c**2, lat_c**2, lon_c**2 + lat_c**2]
    if degree >= 3:
        feats += [lon_c**3, lat_c**3, (lon_c**2)*lat_c, lon_c*(lat_c**2)]
    return np.column_stack(feats)

def ridge_fit(Phi, y, lam:float):
    I = np.eye(Phi.shape[1]); I[0,0] = 0.0  # 不懲罰 bias
    theta = pinv(Phi.T @ Phi + lam*I) @ (Phi.T @ y)
    return theta

def ridge_predict(Phi, theta):
    return Phi @ theta

def rmse(y_true, y_pred):
    # 相容舊版 sklearn：不用 squared=False
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------------------
# (C) 主流程
# ---------------------------
def main():
    ensure_csv(CSV_PATH, XML_PATH)

    df = pd.read_csv(CSV_PATH)
    X = df[["lon","lat"]].values
    y_class = (df["value"].values != MISSING).astype(int)  # 1=陸地, 0=海
    y_reg = df["value"].values

    lon_min, lon_max = X[:,0].min(), X[:,0].max()
    lat_min, lat_max = X[:,1].min(), X[:,1].max()

    # 1) QDA 分類（= GDA 不共享 Σ），先標準化
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)
    qda = QuadraticDiscriminantAnalysis(store_covariance=True)
    qda.fit(Z, y_class)
    y_pred_cls = qda.predict(Z)
    acc = float((y_pred_cls == y_class).mean())
    print(f"[Classification] QDA accuracy = {acc:.4f}")

    # 決策邊界圖
    gx, gy = np.meshgrid(
        np.linspace(lon_min, lon_max, 400),
        np.linspace(lat_min, lat_max, 400),
    )
    grid_xy = np.c_[gx.ravel(), gy.ravel()]
    grid_z = scaler.transform(grid_xy)
    Zhat = qda.predict(grid_z).reshape(gx.shape)

    plt.figure(figsize=(7,10))
    plt.contour(gx, gy, Zhat, levels=[0.5], colors="red", linewidths=1.6)
    plt.scatter(X[y_class==0,0], X[y_class==0,1], s=4, c="blue", label="Class 0 (sea)")
    plt.scatter(X[y_class==1,0], X[y_class==1,1], s=4, c="green", label="Class 1 (land)")
    plt.title("QDA Decision Boundary (unshared Σ)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.gca().set_aspect("equal"); plt.legend(markerscale=2.5)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(OUTDIR/"w6_decision_boundary.png", dpi=170); plt.close()
    print("[Saved] outputs/w6_decision_boundary.png")

    # 2) 回歸（只用陸地）
    mask_land = (y_class == 1)
    X_land = X[mask_land]
    y_land = y_reg[mask_land]

    cands_degree = [1,2,3]
    cands_lam = [1e-6, 1e-5, 1e-4, 1e-3]
    best_rmse = 1e9; best_deg = None; best_lam = None; best_theta = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for deg in cands_degree:
        Phi = geo_poly_features(X_land, degree=deg)
        for lam in cands_lam:
            rmses = []
            for tr_idx, va_idx in kf.split(Phi):
                th = ridge_fit(Phi[tr_idx], y_land[tr_idx], lam)
                pred = ridge_predict(Phi[va_idx], th)
                rmses.append(rmse(y_land[va_idx], pred))
            mean_rmse = float(np.mean(rmses))
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse; best_deg = deg; best_lam = lam
                best_theta = ridge_fit(Phi, y_land, lam)

    print(f"[Regression] best degree={best_deg}, ridge={best_lam:g}, CV-RMSE={best_rmse:.4f}")

    # 用第一折做展示 holdout RMSE
    tr0, va0 = next(iter(KFold(n_splits=5, shuffle=True, random_state=0).split(X_land)))
    Phi_all = geo_poly_features(X_land, best_deg)
    theta_hold = ridge_fit(Phi_all[tr0], y_land[tr0], best_lam)
    rmse_hold = rmse(y_land[va0], ridge_predict(Phi_all[va0], theta_hold))
    print(f"[Regression] holdout RMSE = {rmse_hold:.4f}")

    # 3) 組合 h(x)（海=-999，陸=回歸預測）
    grid_cls = qda.predict(grid_z)
    grid_reg = ridge_predict(geo_poly_features(grid_xy, best_deg), best_theta)
    h = np.where(grid_cls==1, grid_reg, MISSING).reshape(gx.shape)

    # 先鋪海（藍色）
    plt.figure(figsize=(7,10))
    sea_cmap = ListedColormap(["blue"])
    mask_sea = (h == MISSING)
    plt.pcolormesh(gx, gy, np.where(mask_sea, 1, np.nan), shading="auto", cmap=sea_cmap)
    # 疊陸地（綠色漸層）
    im = plt.pcolormesh(gx, gy, np.ma.masked_where(mask_sea, h), shading="auto", cmap="Greens")
    plt.colorbar(im, label="h(x) / Predicted Temp (°C)")
    plt.title("Combined model h(x): R(x) if C(x)=1 else -999")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.gca().set_aspect("equal")
    plt.tight_layout(); plt.savefig(OUTDIR/"w6_combined_heatmap.png", dpi=170); plt.close()
    print("[Saved] outputs/w6_combined_heatmap.png")

    # 4) 指標存檔
    (OUTDIR/"w6_metrics.txt").write_text(
        "=== Week 6 Metrics ===\n"
        f"Classification (QDA unshared Σ)\n"
        f"- Accuracy: {acc:.4f}\n\n"
        f"Regression (Ridge OLS, polynomial features)\n"
        f"- Best degree: {best_deg}\n"
        f"- Best ridge λ: {best_lam:g}\n"
        f"- CV RMSE: {best_rmse:.4f}\n"
        f"- Holdout RMSE (1 fold): {rmse_hold:.4f}\n\n"
        "Combined model:\n"
        "h(x) = R(x) if C(x)=1 else -999\n"
    )
    print("[Saved] outputs/w6_metrics.txt")

if __name__ == "__main__":
    main()














