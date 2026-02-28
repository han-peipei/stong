import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # 或 ":4096:8"
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm,BoundaryNorm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde  # 若没有，可 pip 安装
import matplotlib.ticker as ticker
# import cmaps

from model import Direct_Conv3D_GRU
from data_3_B import standardize
# from data_3_B import build_dataset_batched, standardize

import random
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
out_dir = "/kaggle/working/2324_drop/"
os.makedirs(out_dir, exist_ok=True)
##########################################################################################
def stitch_overlapping_forecasts(y_windows):
    """
    将形如 [N, F] 的滑动窗口序列拼接成一条连续序列（对重叠处做平均）。
    返回:
      y_stitched: [N+F-1]
      counts    : [N+F-1] 每个时刻被多少个窗口覆盖
    """
    N, F = y_windows.shape
    T = N + F - 1
    acc = np.zeros(T, dtype=float)
    cnt = np.zeros(T, dtype=int)
    for i in range(N):
        acc[i:i+F] += y_windows[i]
        cnt[i:i+F] += 1
    y_stitched = acc / np.maximum(cnt, 1)
    return y_stitched, cnt
def plot_timeseries_stitched(y_true_windows, y_pred_windows, time=None, unit='m/s', tag='direct'):

    yt_st, _ = stitch_overlapping_forecasts(y_true_windows)
    yp_st, cnt = stitch_overlapping_forecasts(y_pred_windows)
    T = len(yt_st)

    x = np.arange(T) if time is None else time[:T]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.spines['left'].set_position('zero')    # y轴穿过x=0
    ax.spines['bottom'].set_position('zero')  # x轴穿过y=0
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 让坐标轴箭头更明显（可选）
    # ax.spines['left'].set_linewidth(1.2)
    # ax.spines['bottom'].set_linewidth(1.2)
    # plt.figure(figsize=(12,4))
    ax.plot(x[:1000], yt_st[:1000], label='True', lw=1.2)
    ax.plot(x[:1000], yp_st[:1000], '--', label='Pred', lw=1.0)
    ax.set_xlabel('Time'); ax.set_ylabel(f'Wind speed [{unit}]')
    # ax.title('Sliding-window Forecast')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}timeseries_stitched.png"); plt.close()
##########################################################################################
def chain_rows_by_step(y_windows, time=None, auto_time=True):
    y_windows = np.asarray(y_windows)
    N, F0 = y_windows.shape
    # print(F0)
    # F0=24
    idx_list = list(range(0, N, F0))
    chunks, tchunks = [], []
    for i in idx_list:
        seg = y_windows[i, :F0]  # 整段窗口
        if time is None:
            chunks.append(seg)
        else:
            gidx = i + np.arange(F0)      # 该窗口映射到全局时间的索引
            mask = gidx < len(time)       # 越界保护
            chunks.append(seg[mask])
            tchunks.append(np.asarray(time)[gidx[mask]])

    y_chain = np.concatenate(chunks, axis=0) if chunks else np.array([], dtype=float)
    t_chain = (np.concatenate(tchunks, axis=0) if (time is not None and tchunks) else None)
    return y_chain, t_chain, np.asarray(idx_list)
def plot_chain_rows_by_step(y_true_windows, y_pred_windows, time=None,
                            unit='m/s', tag='direct', auto_time=True):
    yt, tt, idx = chain_rows_by_step(y_true_windows, time=time,  auto_time=auto_time)
    yp, tp, _   = chain_rows_by_step(y_pred_windows, time=time,  auto_time=auto_time)

    fig, ax = plt.subplots(figsize=(12,4))
    x = np.arange(len(yt))
    ax.plot(x[:1000], yt[:1000], label='True', lw=1.2)
    ax.plot(x[:1000], yp[:1000], '--', label='Pred', lw=1.0)
    ax.set_xlabel('Time'); ax.set_ylabel(f'Wind speed [{unit}]')
    ax.spines['left'].set_position('zero')     # y轴穿过 x=0
    ax.spines['bottom'].set_position('zero')   # x轴穿过 y=0
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # plt.title(f'Chained series: rows 1, 1+{step}, 1+2×{step}, ... (full window)')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}chain_rows_by_step_{tag}.png')
    plt.close()

    # return yt, yp, (tt if tt is not None else None), idx

###########################################################################################
def plot_scatter_by_leads(y_true_windows, y_pred_windows,  unit='m/s', tag='direct'):
    yt, _ = stitch_overlapping_forecasts(y_true_windows)
    yp, cnt = stitch_overlapping_forecasts(y_pred_windows)
    # yt = y_true_windows.reshape(-1)
    # yp = y_pred_windows.reshape(-1)

    # 坐标范围
    vmin = float(min(yt.min(), yp.min()))
    vmax = float(max(yt.max()+1.5, yp.max()+1.5))

    # 绘制
    plt.figure(figsize=(6,6))
    plt.scatter(yt, yp, s=3, alpha=0.3)
    plt.plot([vmin, vmax+1.5], [vmin, vmax+1.5], 'k--', lw=1)
    plt.xlim(0, vmax+1.5)
    plt.ylim(0, vmax+1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(f'True [{unit}]')
    plt.ylabel(f'Predicted [{unit}]')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}scatter_all_{tag}.png")
    plt.close()
#######################################################################################################
def get_bins_interval(bins):
    """
    用于生成左闭右开的区间字符串
    """
    freq_bins_left = bins[0:-1] # 左区间
    freq_bins_right = bins[1:] # 右区间
    freq_bins_str = []
    for interval_ind in range(0,len(freq_bins_left)):
        if (interval_ind == len(freq_bins_left) - 1 ):
            str_temp = '[' + str(freq_bins_left[interval_ind]) + "," + str(freq_bins_right[interval_ind]) + "]"
        else:
            str_temp = '[' + str(freq_bins_left[interval_ind]) + "," + str(freq_bins_right[interval_ind]) + ")"
        freq_bins_str.append(str_temp)
    return freq_bins_str
def plot_pdf_1d(y_true_windows,y_pred_windows):
    y_true, _ = stitch_overlapping_forecasts(y_true_windows)
    y_pred, cnt = stitch_overlapping_forecasts(y_pred_windows)

    bins2 = np.arange(0, 28, 2)  # [0,1,2,...,14]
    labels = get_bins_interval(bins2)

    fig, ax = plt.subplots(figsize=(10,4), dpi=120)
    # 直方图密度（柱形不表示频数，而是密度；面积=1）
    y_true2 = np.histogram(y_true, bins=bins2, range=None, weights=None, density=None)
    y_pred2 = np.histogram(y_pred, bins=bins2, range=None, weights=None, density=None)
    histogramt = y_true2[0] / y_true2[0].sum() # 在降水日中进行归一化
    histogramp = y_pred2[0] / y_pred2[0].sum() # 在降水日中进行归一化

    ax.plot(bins2[0:-1],histogramt, label='True')
    ax.plot(bins2[0:-1],histogramp, label='Pred')
    ax.xaxis.set_major_locator(ticker.FixedLocator(bins2[0:-1]))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter((labels))) 
    ax.spines['bottom'].set_position('zero')   # x轴穿过 y=0
    # ax.set_xlim(bins2[0], bins2[-1])  # 显示完整区间
    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig(f"{out_dir}pdf.png")
    plt.show()

#######################################################################################################
def plot_kde2d_full(y_true_windows, y_pred_windows, unit='m/s', tag='direct',
                    nx=200, ny=200, 
                    show_contour=True,
                    bw_method=None):
    yt, _ = stitch_overlapping_forecasts(y_true_windows)
    yp, cnt = stitch_overlapping_forecasts(y_pred_windows)
    # yt = y_true_windows.reshape(-1).astype(float)
    # yp = y_pred_windows.reshape(-1).astype(float)

    vmin = min(yt.min(), yp.min())
    vmax = max(yt.max(), yp.max())

    # KDE
    kde = gaussian_kde(np.vstack([yt, yp]), bw_method=bw_method)  # 形状 [2, N]
    x = np.linspace(vmin, vmax, nx)
    y = np.linspace(vmin, vmax, ny)
    X, Y = np.meshgrid(x, y)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(ny, nx)  # 概率密度（积分≈1）
    
    fig, ax = plt.subplots(figsize=(6,6))
    # bounds = np.arange(0.1,0.6,0.01)
    # norm = BoundaryNorm(bounds, ncolors=plt.get_cmap(cmap).N, clip=True)
    # cmap = 'gray_r'
    # cmap = 'pink_r'
    cmap = 'Blues'
    # cmap = 'viridis'
    pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap=plt.get_cmap(cmap))
    cbar = fig.colorbar(pcm, ax=ax,fraction=0.04)
    cbar.set_label('Probability density')

    ax.set_xlim(0, vmax)
    ax.set_ylim(0, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f'True [{unit}]')
    ax.set_ylabel(f'Predicted [{unit}]')
    ax.grid(True, alpha=0.25)

    # ax.spines['left'].set_position('zero')     # y轴穿过 x=0
    # ax.spines['bottom'].set_position('zero')   # x轴穿过 y=0
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    fig.tight_layout()
    fig.savefig(f"{out_dir}scatter_kde2d_full_{tag}.png", dpi=160)
    plt.close(fig)
###############################################################################################################
def plot_pcolor(y_true_windows, y_pred_windows, unit='m/s', tag='direct',
                    nx=100, ny=100, 
                    show_contour=True):
    y_true, _ = stitch_overlapping_forecasts(y_true_windows)
    y_pred, cnt = stitch_overlapping_forecasts(y_pred_windows)
    # y_true = np.asarray(y_true_windows, dtype=float).ravel()
    # y_pred = np.asarray(y_pred_windows, dtype=float).ravel()
    # xedges = np.linspace(xe[0],28,nx + 1)
    # yedges = np.linspace(0,28,ny + 1)

    # H, xe, ye = np.histogram2d(y_true, y_pred, bins=[xedges, yedges], density=False)
    H, xe, ye = np.histogram2d(y_true, y_pred, bins=100, density=False)

    Z = H.T  # (ny, nx)
    cmap='Blues'
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    im = ax.pcolormesh(xe, ye, Z, cmap=cmap)

    ax.plot([0, 28], [0, 28], '--', lw=1, color='gray', zorder=3, label='y = x')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    ax.set_xlabel(f"True [{unit}]")
    ax.set_ylabel(f"Predicted [{unit}]")
    ax.set_title(f"True vs Pred ({tag})")

    # ax.set_xlim(0, 28)
    # ax.set_ylim(0, 28)

    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    fig.tight_layout()
    fig.savefig(f"{out_dir}pcolor_{tag}.png", dpi=160)
###############################################################################################################
def plot_residual_hist_all(y_true_windows, y_pred_windows, bins=40, unit='m/s', tag='direct'):
    yt, _ = stitch_overlapping_forecasts(y_true_windows)
    yp, cnt = stitch_overlapping_forecasts(y_pred_windows)
    # yt = y_true_windows.reshape(-1)
    # yp = y_pred_windows.reshape(-1)
    err = yp - yt
    weights = np.ones(err.shape, dtype=float) / err.size * 100.0

    plt.figure(figsize=(7,4.5))
    plt.hist(err, bins=bins, alpha=0.85, edgecolor='k',weights=weights)
    plt.xlabel(f'Error (Pred - True) [{unit}]')
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}residual_hist_ln.png")
    plt.close()
##################################################################################################
def bin_percentages(y_true_windows, y_pred_windows, edges=(-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12), include_outside=True):
    yt, _ = stitch_overlapping_forecasts(y_true_windows)
    yp, cnt = stitch_overlapping_forecasts(y_pred_windows)
    err=yp - yt
    edges = np.asarray(edges, dtype=int)
    err = np.asarray(err, dtype=float)
    if include_outside:
        bins = np.concatenate(([-np.inf], edges, [np.inf]))
        counts, _ = np.histogram(err, bins=bins)
        labels = [f"< {edges[0]:.0f}"]
        labels += [f"({edges[i-1]:.0f}, {edges[i]:.0f}]" for i in range(1, len(edges))]
        labels += [f"> {edges[-1]:.0f}"]
    else:
        counts, _ = np.histogram(err, bins=edges)
        labels = [f"[{edges[0]:.0f}, {edges[1]:.0f})"] + \
                 [f"[{edges[i]:.0f}, {edges[i+1]:.0f})" for i in range(1, len(edges)-2)] + \
                 [f"[{edges[-2]:.0f}, {edges[-1]:.0f}]"]  # 最后一箱右闭

    # inside_count = counts.sum()
    total = len(err)
    # print(total)
    # print(err)
    # print(inside_count)
    # labels = []
    # for i in range(len(edges)-1):
    #     a, b = edges[i], edges[i+1]
    #     if i == 0:
    #         labels.append(f"[{a},{b}]")
    #     else:
    #         labels.append(f"({a},{b}]")
    percents = (counts / total) * 100.0
    return labels, percents, counts, total
def plot_bin_percentages(labels, percents, tag='direct'):
    plt.figure(figsize=(9,3.5))
    x = np.arange(len(labels))
    plt.bar(x, percents,width=0.4)
    plt.xticks(x, labels, rotation=45, fontsize=9)
    plt.ylabel('Percentage (%)')
    plt.title('error distribution (log scale)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.yscale('log') 
    plt.savefig(f'{out_dir}error_bin_percent_{tag}.png')
    plt.close()
##########################################################################################
##########################################################################################
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 若版本支持，强制确定性（某些算子会报错，可改 warn_only=True）
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

set_seed(2025)

seed = 2025
g = torch.Generator()
g.manual_seed(seed)

def _worker_init_fn(wid):
    np.random.seed(seed + wid)
    random.seed(seed + wid)

def train_and_evaluate_from_npy(
    hist_train,        # [B, H_hist]          
    nwp_train,         # [B, F, C, H, W]     
    y_train,           # [B, F]               
    hist_val,          # [Bv, H_hist]
    nwp_val,           # [Bv, F, C, H, W]
    y_val,             # [Bv, F]
    coords_tr, # [B, d]  (d=2/3)
    coords_va,   # [Bv, d]
    num_epochs=1000, batch_size=32, patience=3,
    device=torch.device('cuda'),
):
    hist_train = np.asarray(hist_train, dtype=np.float32)   # [B, H]
    nwp_train  = np.asarray(nwp_train,  dtype=np.float32)   # [B, F, C, H, W]
    y_train    = np.asarray(y_train,    dtype=np.float32)   # [B, F]

    hist_val   = np.asarray(hist_val,   dtype=np.float32)
    nwp_val    = np.asarray(nwp_val,    dtype=np.float32)
    y_val      = np.asarray(y_val,      dtype=np.float32)

    B, H_hist = hist_train.shape
    Bv        = hist_val.shape[0]
    F, C, H, W = nwp_train.shape[1], nwp_train.shape[2], nwp_train.shape[3], nwp_train.shape[4]

    coords_train = np.asarray(coords_tr, dtype=np.float32)
    coords_val   = np.asarray(coords_va,   dtype=np.float32)
    

    def normalize_coords_trig(coords, elev_mean=None, elev_std=None,
                            fix_lon=True, eps=1e-8, return_stats=False):
        """
        coords: [B,2]或[B,3]，列为 [lat, lon, (elev)]
        输出: [B, 4] 或 [B, 5]  -> [sin(lat), cos(lat), sin(lon), cos(lon), (elev_z)]
        训练集：不传 elev_mean/std，会在内部计算并返回
        验证/测试：把训练集的 elev_mean/std 传入，保持一致
        """
        coords = np.asarray(coords, dtype=np.float32)
        # if coords.ndim != 2 or coords.shape[1] < 2:
        #     raise ValueError(f"coords 形状应为 [B,2] 或 [B,3]，收到 {coords.shape}")

        lat_deg = coords[:, 0].copy()
        lon_deg = coords[:, 1].copy()
        # if fix_lon:
        #     lat_deg, lon_deg = _fix_lat_lon(lat_deg, lon_deg)

        lat_rad = np.deg2rad(lat_deg)
        lon_rad = np.deg2rad(lon_deg)

        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)

        # if coords.shape[1] >= 3:
        elev = coords[:, 2].astype(np.float32)
        if elev_mean is None or elev_std is None:
            # 训练集：拟合
            elev_mean = float(elev.mean())
            elev_std  = float(elev.std()) + eps
        elev_z = (elev - elev_mean) / max(elev_std, eps)
        out = np.stack([sin_lat, cos_lat, sin_lon, cos_lon, elev_z], axis=1)
        if return_stats:
            return out, elev_mean, elev_std
        return out
        # else:
        #     out = np.stack([sin_lat, cos_lat, sin_lon, cos_lon], axis=1)
        #     if return_stats:
        #         # 没有海拔，返回 None
        #         return out, None, None
        #     return out
    
    coords_tr_norm, elev_mean, elev_std = normalize_coords_trig(
        coords_train, return_stats=True
    )
    coord_dim    = coords_tr_norm.shape[1]
    # 验证/测试复用训练统计量
    coords_va_norm = normalize_coords_trig(
        coords_val, elev_mean=elev_mean, elev_std=elev_std
    )
    print(coords_tr_norm.shape)
    print(hist_train.shape)
    print(nwp_train.shape)
    print(y_train.shape)
    np.save(f"{out_dir}coords_elev_mean.npy", np.array([elev_mean], dtype=np.float32))
    np.save(f"{out_dir}coords_elev_std.npy",  np.array([elev_std],  dtype=np.float32))


    y_train, y_mean, y_std = standardize(y_train)
    y_val = (y_val - y_mean)/y_std
    
    y_mean_t = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std_t  = torch.as_tensor(y_std,  dtype=torch.float32, device=device)
    
    np.save(f"{out_dir}y_mean.npy", y_mean.astype(np.float32))
    np.save(f"{out_dir}y_std.npy", y_std.astype(np.float32))
########################
    # eps = 1e-8

    # h_mean = float(hist_train.mean())
    # h_std = float(hist_train.std()) + eps
    # hist_train = (hist_train - h_mean) / h_std
    # hist_val = (hist_val - h_mean) / h_std
    # np.save("hist_y_mean.npy", np.array([h_mean], dtype=np.float32))
    # np.save("hist_y_std.npy",  np.array([h_std],  dtype=np.float32))

    # # ===== 3) 标准化 nwp（按通道 C 做；只用训练集）=====
    # # nwp_train: [B, F, C, H, W]
    # nwp_train_mean = nwp_train.mean(axis=(0,1,3,4), keepdims=True)  # [1,1,C,1,1]
    # nwp_train_std  = nwp_train.std(axis=(0,1,3,4), keepdims=True) + eps

    # nwp_train = (nwp_train - nwp_train_mean) / nwp_train_std
    # nwp_val   = (nwp_val   - nwp_train_mean) / nwp_train_std

    # np.save("nwp_mean.npy", nwp_train_mean.astype(np.float32).squeeze())
    # np.save("nwp_std.npy",  nwp_train_std.astype(np.float32).squeeze())
########################
    hist_train_3d = hist_train[..., None]   # [B, H_hist, 1]
    hist_val_3d   = hist_val[...,   None]   # [Bv, H_hist, 1]
    pin = (device.type == "cuda")
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(hist_train_3d).float(),   # [B, H, 1]
            torch.tensor(nwp_train).float(),       # [B, F, C, H, W]
            torch.tensor(coords_tr_norm).float(),    # [B, d]
            torch.tensor(y_train).float(),       # [B, F]
        ),
        batch_size=batch_size, shuffle=True,
        generator=g, worker_init_fn=_worker_init_fn, num_workers=0,pin_memory=pin,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(hist_val_3d).float(),
            torch.tensor(nwp_val).float(),
            torch.tensor(coords_va_norm).float(),
            torch.tensor(y_val).float(),
        ),
        batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=pin,
    )

    # ===== 3) 建模 =====
    model = Direct_Conv3D_GRU(
        in_channels=C,
        forecast_hours=F,
        coord_dim=coord_dim,
        hist_input_size=1,
        hidden_size=32,
        coord_feat_dim=16,
        kt=3
    ).to(device)

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_val, no_imp = float('inf'), 0
    train_losses, val_losses = [], []

    # === 4. 训练 ===
    # if epoch==0 and tot==0:
    for epoch in range(num_epochs):
        model.train()
        tot = 0
        for hist_b, nwp_b, coord_b, y_b in train_loader:
            # print("hist_b shape =", hist_b.shape)
            # print("nwp_b  shape =", nwp_b.shape)
            # print("coord_b shape =", coord_b.shape)
            # print("hist_b shape =", hist_b.shape)
            # print("nwp_b  shape =", nwp_b.shape)
            # print("coord_b shape =", coord_b.shape)
            hist_b, nwp_b, coord_b, y_b = hist_b.to(device), nwp_b.to(device), coord_b.to(device), y_b.to(device)
            y_b = y_b.squeeze(-1) 
            out = model(coord_b, hist_b, nwp_b)
            
            y_true_phys = y_b * y_std_t + y_mean_t   # [B, F]，单位 m/s
            # 分段权重（你可以自己调）
            w = torch.ones_like(y_true_phys)
            w = torch.where(y_true_phys >= 6.0,  torch.full_like(w, 1.5), w)
            w = torch.where(y_true_phys >= 10.8, torch.full_like(w, 4.0), w)
            w = torch.where(y_true_phys >= 14.0, torch.full_like(w, 8.0), w)
            # 分段加权 L1
            # err = torch.abs(out - y_b)              # 仍在标准化空间算误差（没问题）
            # loss = (w * err).sum() / (w.sum() + 1e-8)
            err_phys = torch.abs((out - y_b) * y_std_t)   # 误差单位 m/s
            loss = (w * err_phys).sum() / (w.sum() + 1e-8)
              
            # loss = criterion(out, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
        train_losses.append(tot/len(train_loader))

        # ------- 验证 -------
        model.eval()
        totv = 0
        with torch.no_grad():
            for hist_b, nwp_b, coord_b, y_b in val_loader:
                hist_b, nwp_b, coord_b, y_b = hist_b.to(device), nwp_b.to(device), coord_b.to(device), y_b.to(device)
                y_b = y_b.squeeze(-1) 
                out = model(coord_b, hist_b, nwp_b)
            
                y_true_phys = y_b * y_std_t + y_mean_t   # [B, F]，单位 m/s
                # 分段权重（你可以自己调）
                w = torch.ones_like(y_true_phys)
                w = torch.where(y_true_phys >= 6.0,  torch.full_like(w, 1.5), w)
                w = torch.where(y_true_phys >= 10.8, torch.full_like(w, 4.0), w)
                w = torch.where(y_true_phys >= 14.0, torch.full_like(w, 8.0), w)
                # 分段加权 L1
                # err = torch.abs(out - y_b)              # 仍在标准化空间算误差（没问题）
                # loss = (w * err).sum() / (w.sum() + 1e-8)
                err_phys = torch.abs((out - y_b) * y_std_t)   # 误差单位 m/s
                loss = (w * err_phys).sum() / (w.sum() + 1e-8)
                
                # loss = criterion(out, y_b)
                totv += loss.item()
        avg_val = totv/len(val_loader)
        val_losses.append(avg_val)

        print(f"Epoch {epoch+1}: Train {train_losses[-1]:.4f}, Val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            no_imp = 0
            torch.save(model.state_dict(), f'{out_dir}best_direct_model.pth')
        else:
            no_imp += 1
        if no_imp >= patience:
            print(f"Early Stop at epoch {epoch+1}")
            break

    # === 5. 可视化Loss ===
    plt.figure()
    plt.plot(train_losses,label='train')
    plt.plot(val_losses,label='val')
    plt.ylim(0.5,0.75)
    plt.xlim(0,13)
    plt.legend()
    plt.savefig(f"{out_dir}loss_curve_direct.png")
    plt.close()

    # === 6. 评估 ===
    model.load_state_dict(torch.load(f"{out_dir}best_direct_model.pth", weights_only=True))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for hist_b, nwp_b, coord_b, y_b in val_loader:
            out = model(coord_b.to(device), hist_b.to(device), nwp_b.to(device))
            pred = (out.cpu().numpy()*y_std + y_mean)
            true = (y_b.numpy()*y_std + y_mean)
            preds.append(pred) 
            trues.append(true)
    preds = np.concatenate(preds,0)
    trues = np.concatenate(trues,0)
    np.save(os.path.join(out_dir, "val_preds_windows.npy"), preds.astype(np.float32))
    np.save(os.path.join(out_dir, "val_trues_windows.npy"), trues.astype(np.float32))
#########################################################################
    time_val = None   # 或者用 pandas.date_range(...) 构造

    # 1) 连续时间序列（拼接）
    plot_timeseries_stitched(trues, preds, time=time_val, unit='m/s', tag='val')
    plot_kde2d_full(trues, preds, unit='m/s', tag='val',
                nx=200, ny=200, 
                show_contour=True,
                bw_method=None)  
    plot_scatter_by_leads(trues, preds,  unit='m/s', tag='val')
    plot_chain_rows_by_step(trues, preds, tag='val', auto_time=True)
    plot_residual_hist_all(trues, preds,bins=40, unit='m/s', tag='val')
    plot_pdf_1d(trues,preds)
    plot_pcolor(trues, preds, unit='m/s', tag='val', nx=150, ny=150)
    # yt = trues.reshape(-1)
    # yp = preds.reshape(-1)
    # err = yp - yt
    labels, perc, cnt, total = bin_percentages(
        trues,preds,
        edges=(-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12),
        include_outside=True,   # 同时给出区间外比例
        
    )
    plot_bin_percentages(labels, perc, tag='direct')
##########################################################################
    # 每小时MSE
    forecast_hours=24
    hourly_mse = [mean_squared_error(trues[:,i], preds[:,i]) for i in range(forecast_hours)]
    hourly_mae  = [mean_absolute_error(trues[:,i], preds[:,i]) for i in range(forecast_hours)]
    plt.plot(hourly_mse,'o-')
    plt.xlim(0,25)
    # plt.ylim(0,2.5)
    plt.savefig(f"{out_dir}hourly_mse_direct.png")
    plt.close()
    plt.plot(hourly_mae,'o-')
    # plt.ylim(1,2.5)
    plt.xlim(0,25)
    plt.savefig(f"{out_dir}hourly_mae_direct.png")
    plt.close()

    # 单个样本
    plt.figure()
    plt.plot(trues[0],label='True')
    plt.plot(preds[0],label='Pred')
    # plt.ylim(0,8)
    plt.xlim(0,25)
    plt.legend()
    plt.savefig(f"{out_dir}example_direct0.png")
    plt.close()
    plt.plot(trues[1],label='True')
    plt.plot(preds[1],label='Pred')
    # plt.ylim(0,8)
    plt.xlim(0,25)
    plt.legend()
    plt.savefig(f"{out_dir}example_direct1.png")
    plt.close()
    plt.plot(trues[10],label='True')
    plt.plot(preds[10],label='Pred')
    # plt.ylim(0,8)
    plt.xlim(0,25)
    plt.legend()
    plt.savefig(f"{out_dir}example_direct10.png")
    plt.close()

    print("Total MSE:", mean_squared_error(trues.flatten(), preds.flatten()))
    print("Total MAE:", mean_absolute_error(trues.flatten(), preds.flatten()))
    return model
