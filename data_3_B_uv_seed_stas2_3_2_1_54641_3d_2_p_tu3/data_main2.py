# -*- coding: utf-8 -*-
import os, re
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from train_3_B import train_and_evaluate_from_npy

# ---------------- 必要的小工具函数（保留） ----------------
def to_array(x):
    if isinstance(x, list): 
        try:
            return np.stack([np.asarray(e) for e in x], axis=0)
        except Exception:
            return np.asarray(x)
    return np.asarray(x)

# def ensure_4d(a):
#     # if a.ndim == 3:
#     T, H, W = a.shape
#     return a.reshape(T, 1, H, W)
#     # raise ValueError(f"Unsupported model shape {a.shape}")

# def ensure_BT(y):
#     """
#     统一到 [B, T]
#     - [T] -> [1, T]
#     - [B, T] -> 原样
#     - [B, T, 1] -> [B, T]
#     """
#     y = np.asarray(y)
#     if y.ndim == 2:
#         return y
#     if y.ndim == 1:
#         T = y.shape[0]
#         return y.reshape(1, T)
#     if y.ndim == 3 and y.shape[-1] == 1:
#         return y[..., 0]
#     raise ValueError(f"Unsupported labels shape {y.shape}")

# ---------------- 配置区 ----------------
ROOT = "/kaggle/input/datasets/niaosilius/stations-2324-train-drop-title/stations_2324_train_drop"   # 
# ROOT = "/thfs1/home/qx_hyt/hpp/data/station_AI/train_data2"   
csv_path = "2023.csv"  
VARS = ("10u", "10v")                                        
# VARS = ("10u", "10v",'2DPT','2RH','CAPE','MSL','2T','VIS')                                       
TI = "02"                                                     # 起报时（当前正则里只匹配02）
_var_pat = "|".join(map(re.escape, VARS))
_station_pat = r'(?:[A-Za-z]\d{4}|\d{5})'
# _station_pat = r'(?:[A]\d{4})'
# _station_pat ='A2662'
_data_re  = re.compile(rf'^(train|val)_data_({_var_pat})_({_station_pat})_{TI}\.npy$')
_label_re = re.compile(rf'^(train|val)_labels_({_station_pat})_{TI}\.npy$')

data_idx = defaultdict(dict)  
labels_idx = {}                
stations_set = set()

for fn in os.listdir(ROOT):
    m = _data_re.match(fn)
    if m:
        split, var, station = m.groups()
        data_idx[(split, station, var)] = os.path.join(ROOT, fn)
        stations_set.add(station)
        continue
    m = _label_re.match(fn)
    if m:
        split, station = m.groups()
        labels_idx[(split, station)] = os.path.join(ROOT, fn)
        stations_set.add(station)

stations = sorted(stations_set)
if not stations:
    raise RuntimeError("未在目录中匹配到任何站点文件，请检查ROOT与命名规则/起报时。")

# ---------------- 逐站加载并拼接（train/val） ----------------
hist_train, nwp_train, y_train = [], [], []
hist_val,   nwp_val,   y_val   = [], [], []
meta_tr, meta_va = [], []  

Hhist = 24   # 历史窗口长度（上一个24小时）
F     = 24   # 预测窗口长度（下一个24小时）
S     = 1 
for split in ("train", "val"):
    for station in stations:
        # 检查变量与标签是否齐全
        missing = [v for v in VARS if (split, station, v) not in data_idx]
        if missing or (split, station) not in labels_idx:
            print(f"[{split}][{station}] 缺文件，跳过。缺失变量: {missing}, 有标签? {(split, station) in labels_idx}")
            continue

        try:
            var_arrs  = []
            for v in VARS:
                p = data_idx[(split, station, v)]
                modle = np.load(p, allow_pickle=True)
                Tlen, H, W = modle.shape
                modle = modle.reshape(Tlen, 1, H, W)  
                var_arrs.append(modle)
                # a = ensure_4d(to_array(a).astype(np.float32))
            X_full = np.concatenate(var_arrs, axis=1) 
            # 3) 读取标签，整理为 (T,)
            p_lab = labels_idx[(split, station)]
            lab_np = np.load(p_lab, allow_pickle=True)       # 可能是 (T,) 或 (1,T) 或 (B,T,1)
            lab_np = to_array(lab_np).astype(np.float32)
            y_full = lab_np.reshape(-1) if lab_np.ndim > 1 else lab_np
            # 4) 沿 T 做滑窗
            Tlen = X_full.shape[0]
            need = Hhist + F
            nwin = (Tlen - need) // S + 1
            H_list, N_list, Y_list = [], [], []
            for k in range(nwin):
                t0 = k * S
                t1 = t0 + Hhist
                t2 = t1 + F
                hist = y_full[t0:t1]  
                nwp  = X_full[t1:t2]  
                y    = y_full[t1:t2]  
                H_list.append(hist)
                N_list.append(nwp)
                Y_list.append(y)

            H_win = np.stack(H_list, axis=0)  # (nwin, Hhist)
            N_win = np.stack(N_list, axis=0)  # (nwin, F, C, H, W)
            Y_win = np.stack(Y_list, axis=0)  # (nwin, F)

            if split == "train":
                hist_train.append(H_win)
                nwp_train.append(N_win)
                y_train.append(Y_win)
                meta_tr.append((station, nwin))  
                print(f"[train][{station}] -> hist{H_win.shape}, nwp{N_win.shape}, y{Y_win.shape}, nwin={nwin}")
            else:
                hist_val.append(H_win)
                nwp_val.append(N_win)
                y_val.append(Y_win)
                meta_va.append((station, nwin)) 
                print(f"[val  ][{station}] -> hist{H_win.shape}, nwp{N_win.shape}, y{Y_win.shape}, nwin={nwin}")

        except Exception as e:
            print(f"[{split}][{station}] 载入失败: {e}")

# if not train_models and not val_models:
#     raise RuntimeError("没有成功载入的 train/val 数据。")


hist_train_all   = np.concatenate(hist_train, axis=0) if hist_train else None
model_train_all  = np.concatenate(nwp_train,   axis=0) if nwp_train   else None
obs_train_all    = np.concatenate(y_train,  axis=0) if y_train  else None

hist_val_all     = np.concatenate(hist_val,     axis=0) if hist_val     else None
model_val_all    = np.concatenate(nwp_val,     axis=0) if nwp_val     else None
obs_val_all      = np.concatenate(y_val,     axis=0) if y_val     else None

print("hist_train_all:", None if hist_train_all is None else hist_train_all.shape)
print("model_train_all:", None if model_train_all is None else model_train_all.shape)
print("obs_train_all  :", None if obs_train_all   is None else obs_train_all.shape)
print("hist_val_all  :", None if hist_val_all   is None else hist_val_all.shape)
print("model_val_all  :", None if model_val_all   is None else model_val_all.shape)
print("obs_val_all    :", None if obs_val_all     is None else obs_val_all.shape)
def _check_np(name, arr):
    arr = np.asarray(arr)
    print(f"[CHECK] {name}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"        finite={np.isfinite(arr).all()}, "
          f"nan={np.isnan(arr).sum()}, inf={np.isinf(arr).sum()}, "
          f"min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}")

_check_np("hist_train_all",  hist_train_all)
_check_np("model_train_all", model_train_all)
_check_np("obs_train_all",   obs_train_all)
_check_np("hist_val_all",    hist_val_all)
_check_np("model_val_all",   model_val_all)
_check_np("obs_val_all",     obs_val_all)

# ---------------- 读取站点经纬高程并生成 coords ----------------
df = pd.read_csv(csv_path, dtype={'Station_Id_C': str}, low_memory=False)
df['Station_Id_C'] = df['Station_Id_C'].str.strip().str.upper()

dfu = (df.drop_duplicates('Station_Id_C', keep='last')
         .set_index('Station_Id_C')[['Lat','Lon','Alti']])

# 构建查找表
station_lut = {sid: (float(row['Lat']), float(row['Lon']), float(row['Alti']))
               for sid, row in dfu.iterrows()}

def coords_from_meta(meta_list, lut):
    blocks = []
    for sid, nwin in meta_list:
        key = str(sid).strip().upper()
        if key not in lut:
            raise KeyError(f"站点 {key} 不在站点表里")
        lat, lon, elev = lut[key]
        blocks.append(np.repeat([[lat, lon, elev]], nwin, axis=0).astype(np.float32))
    return np.concatenate(blocks, axis=0) if blocks else np.zeros((0,3), np.float32)

coords_tr = coords_from_meta(meta_tr, station_lut)   # (sum(nwin_train), 3)
coords_va = coords_from_meta(meta_va, station_lut)   # (sum(nwin_val), 3)
_check_np("coords_tr",       coords_tr)
_check_np("coords_va",       coords_va)
print("\n========== 开始统一训练模型 ==========\n")

# 你的训练函数应接受四个输入（不需要 hour_codes）
model = train_and_evaluate_from_npy(
    hist_train_all, model_train_all,obs_train_all,  
    hist_val_all,model_val_all,obs_val_all,
    coords_tr,  coords_va,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
