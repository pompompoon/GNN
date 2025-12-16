# -*- coding: utf-8 -*-
"""
create_graph_top_strategy.py
TOP Strategy GNN用グラフ構築（角度パラメータ化 + 感度比率エッジ特徴量）

★修正版:
- 角度閾値をコマンドライン引数で指定可能
- エッジ特徴量に感度比率（外側/内側）を追加

使用例:
  python create_graph_top_strategy.py --angle 45
  python create_graph_top_strategy.py --angle 30 --adj-angle 60
  python create_graph_top_strategy.py --angle 60 --tolerance 2

ノード特徴量:
  - GAP感度（1回測定の結果）★主要入力
  - 偏心度（中心からの距離）
  - 角度（極座標）

エッジ特徴量:
  - 相関係数（位置間の感度相関）
  - 角度類似度
  - 距離重み
  - 感度比率（外側/内側）★NEW

教師データ:
  - HFA感度（Gold Standard）
"""

# OpenMP競合を回避
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import pickle
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 設定ファイル読み込み
try:
    from config_top_strategy import (
        ANGLE_THRESHOLD_SAME_RING_DEG, ANGLE_THRESHOLD_ADJACENT_RING_DEG,
        ECCENTRICITY_TOLERANCE, RING_RELAXATION_FACTOR,
        SENSITIVITY_MIN, SENSITIVITY_MAX,
        SENSITIVITY_RATIO_MIN, SENSITIVITY_RATIO_MAX, SENSITIVITY_EPSILON,
        USE_CORRELATION, USE_ANGLE_SIMILARITY,
        USE_SENSITIVITY_RATIO, USE_SENSITIVITY_RATIO_INVERSE, USE_DISTANCE_WEIGHT,
        PATTERN_NAME_MAP, PATTERN_ID_MAP, EYE_MAP,
        get_output_suffix
    )
except ImportError:
    # デフォルト値
    ANGLE_THRESHOLD_SAME_RING_DEG = 60.0
    ANGLE_THRESHOLD_ADJACENT_RING_DEG = 90.0
    ECCENTRICITY_TOLERANCE = 1
    RING_RELAXATION_FACTOR = 0.5
    SENSITIVITY_MIN = 0.0
    SENSITIVITY_MAX = 40.0
    SENSITIVITY_RATIO_MIN = 0.1
    SENSITIVITY_RATIO_MAX = 10.0
    SENSITIVITY_EPSILON = 1.0
    USE_CORRELATION = False
    USE_ANGLE_SIMILARITY = True
    USE_SENSITIVITY_RATIO = True
    USE_SENSITIVITY_RATIO_INVERSE = True
    USE_DISTANCE_WEIGHT = True
    PATTERN_NAME_MAP = {
        '中心30-2 閾値テスト': 'Pattern30-2',
        '中心24-2 閾値テスト': 'Pattern24-2',
        '中心10-2 閾値テスト': 'Pattern10-2',
    }
    PATTERN_ID_MAP = {'Pattern30-2': 0, 'Pattern24-2': 1, 'Pattern10-2': 2}
    EYE_MAP = {0: 'Left', 1: 'Right'}
    def get_output_suffix(angle_deg):
        return f"_angle{int(angle_deg)}"

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

# データパス
DATA_ROOT = Path("G:/共有ドライブ/GAP_Analysis/Data/GAP2_KyodaiClinical")


# ==================== 標準グリッド定義 ====================

def get_10_2_grid(eye_side=0):
    """10-2パターンの標準グリッド（68点）- 左右同じ"""
    points = [
        [-1, 9], [1, 9],
        [-5, 7], [-3, 7], [-1, 7], [1, 7], [3, 7], [5, 7],
        [-7, 5], [-5, 5], [-3, 5], [-1, 5], [1, 5], [3, 5], [5, 5], [7, 5],
        [-7, 3], [-5, 3], [-3, 3], [-1, 3], [1, 3], [3, 3], [5, 3], [7, 3],
        [-9, 1], [-7, 1], [-5, 1], [-3, 1], [-1, 1], [1, 1], [3, 1], [5, 1], [7, 1], [9, 1],
        [-9, -1], [-7, -1], [-5, -1], [-3, -1], [-1, -1], [1, -1], [3, -1], [5, -1], [7, -1], [9, -1],
        [-7, -3], [-5, -3], [-3, -3], [-1, -3], [1, -3], [3, -3], [5, -3], [7, -3],
        [-7, -5], [-5, -5], [-3, -5], [-1, -5], [1, -5], [3, -5], [5, -5], [7, -5],
        [-5, -7], [-3, -7], [-1, -7], [1, -7], [3, -7], [5, -7],
        [-1, -9], [1, -9],
    ]
    return np.array(points, dtype=np.float32)


def get_24_2_grid(eye_side=0):
    """24-2パターンの標準グリッド（54点）- ★左右で異なる"""
    if eye_side == 0:  # Left eye - 右端が27
        points = [
            [-9, 21], [-3, 21], [3, 21], [9, 21],
            [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
            [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
            [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
            [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
            [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
            [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
            [-9, -21], [-3, -21], [3, -21], [9, -21],
        ]
    else:  # Right eye - 左端が-27
        points = [
            [-9, 21], [-3, 21], [3, 21], [9, 21],
            [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
            [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
            [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3],
            [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3],
            [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
            [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
            [-9, -21], [-3, -21], [3, -21], [9, -21],
        ]
    return np.array(points, dtype=np.float32)


def get_30_2_grid(eye_side=0):
    """30-2パターンの標準グリッド（76点）- 左右同じ"""
    points = [
        [-9, 27], [-3, 27], [3, 27], [9, 27],
        [-15, 21], [-9, 21], [-3, 21], [3, 21], [9, 21], [15, 21],
        [-21, 15], [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15], [21, 15],
        [-27, 9], [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9], [27, 9],
        [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
        [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
        [-27, -9], [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9], [27, -9],
        [-21, -15], [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15], [21, -15],
        [-15, -21], [-9, -21], [-3, -21], [3, -21], [9, -21], [15, -21],
        [-9, -27], [-3, -27], [3, -27], [9, -27],
    ]
    return np.array(points, dtype=np.float32)


def get_mariotte_positions(eye_side):
    """マリオット盲点の位置"""
    if eye_side == 0:  # Left eye - 盲点は耳側（左）
        return [(-15, 3), (-15, -3)]
    else:  # Right eye - 盲点は耳側（右）
        return [(15, 3), (15, -3)]


def get_standard_grid(pattern_name, eye_side=0, exclude_mariotte=True):
    """標準グリッドを取得"""
    if 'Pattern30-2' in pattern_name:
        grid = get_30_2_grid(eye_side)
    elif 'Pattern24-2' in pattern_name:
        grid = get_24_2_grid(eye_side)
    elif 'Pattern10-2' in pattern_name:
        grid = get_10_2_grid(eye_side)
    else:
        grid = get_30_2_grid(eye_side)
    
    if exclude_mariotte and 'Pattern10-2' not in pattern_name:
        mariotte = get_mariotte_positions(eye_side)
        mask = np.ones(len(grid), dtype=bool)
        for mx, my in mariotte:
            distances = np.sqrt((grid[:, 0] - mx)**2 + (grid[:, 1] - my)**2)
            mask &= (distances > 1.0)
        grid = grid[mask]
    
    return grid


def match_data_to_grid(df_subset, standard_grid, value_col='Sensitivity', tolerance=3.0):
    """実データを標準グリッドにマッチング"""
    n_points = len(standard_grid)
    values = np.full(n_points, np.nan, dtype=np.float32)
    
    if len(df_subset) == 0:
        return values
    
    data_x = df_subset['InspectionAngleX'].values
    data_y = df_subset['InspectionAngleY'].values
    data_vals = df_subset[value_col].values
    
    for i, (gx, gy) in enumerate(standard_grid):
        distances = np.sqrt((data_x - gx)**2 + (data_y - gy)**2)
        min_idx = np.argmin(distances)
        
        if distances[min_idx] <= tolerance:
            values[i] = data_vals[min_idx]
    
    return values


# ==================== データ読み込み ====================

def load_gap_data(subject_folder):
    """GAP測定データを読み込む"""
    response_files = list(subject_folder.glob("response_*.csv"))
    
    if len(response_files) == 0:
        return None
    
    try:
        df = pd.read_csv(response_files[0])
        
        required_cols = ['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'IsMariotte']
        if not all(col in df.columns for col in required_cols):
            return None
        
        if 'IsInspectionPointer' in df.columns:
            df = df[df['IsInspectionPointer'] == True].copy()
        
        # GAP感度カラムを探す
        gap_cols = ['EstimatedSensitivity', 'Sensitivity', 'ThresholdEstimate', 'MeasuredSensitivity']
        gap_col = None
        for col in gap_cols:
            if col in df.columns:
                gap_col = col
                break
        
        if gap_col is None:
            return None
        
        df['GAPSensitivity'] = df[gap_col]
        return df
        
    except:
        return None


def load_hfa_data(subject_folder):
    """HFAデータを読み込む"""
    hfa_folder = subject_folder / "HFAMatchData"
    
    if not hfa_folder.exists():
        return None, None
    
    hfa_result_files = list(hfa_folder.glob("hfa_result_*.csv"))
    hfa_option_files = list(hfa_folder.glob("hfa_option_*.csv"))
    
    if len(hfa_result_files) == 0 or len(hfa_option_files) == 0:
        return None, None
    
    try:
        df_hfa = pd.read_csv(hfa_result_files[0])
        df_option = pd.read_csv(hfa_option_files[0])
        
        required_cols = ['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'Sensitivity']
        if not all(col in df_hfa.columns for col in required_cols):
            return None, None
        
        if 'TestPattern' not in df_option.columns:
            return None, None
        
        test_pattern = df_option['TestPattern'].iloc[0]
        return df_hfa, test_pattern
        
    except:
        return None, None


# ==================== 極座標・隣接計算 ====================

def compute_polar_coordinates(positions):
    """極座標を計算"""
    x = positions[:, 0]
    y = positions[:, 1]
    eccentricity = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    return eccentricity, angle


def get_eccentricity_ring(eccentricity, pattern_name):
    """偏心度からリング番号を取得"""
    if 'Pattern10-2' in pattern_name:
        ring = np.round(eccentricity / 2).astype(int)
    else:
        ring = np.round(eccentricity / 6).astype(int)
    return ring


def compute_angular_adjacency(positions, pattern_name, 
                               angle_same_deg, angle_adj_deg,
                               tolerance, relaxation_factor):
    """
    角度ベースの隣接を計算（パラメータ化版）
    
    Args:
        positions: 測定点の座標
        pattern_name: パターン名
        angle_same_deg: 同一リング内の角度閾値（度）
        angle_adj_deg: 隣接リング間の角度閾値（度）
        tolerance: 偏心度リングの許容差
        relaxation_factor: 緩和係数
    
    Returns:
        edge_list: エッジリスト
        edge_info: エッジ情報 [angle_diff, distance, ring_diff]
        eccentricity: 偏心度
        angle: 角度
    """
    n_points = len(positions)
    eccentricity, angle = compute_polar_coordinates(positions)
    ring = get_eccentricity_ring(eccentricity, pattern_name)
    
    angle_same_rad = np.deg2rad(angle_same_deg)
    angle_adj_rad = np.deg2rad(angle_adj_deg)
    
    edge_list = []
    edge_info = []
    
    for i in range(n_points):
        for j in range(n_points):
            if i == j:
                continue
            
            ring_diff = abs(ring[i] - ring[j])
            if ring_diff > tolerance:
                continue
            
            # 角度差の計算
            a_diff = abs(angle[i] - angle[j])
            if a_diff > np.pi:
                a_diff = 2 * np.pi - a_diff
            
            # 閾値の決定
            if ring_diff == 0:
                # 同一リング
                threshold = angle_same_rad
            else:
                # 隣接リング（緩和係数を適用）
                threshold = angle_adj_rad * (1.0 + relaxation_factor * (ring_diff - 1))
            
            if a_diff <= threshold:
                dist = np.sqrt((positions[i, 0] - positions[j, 0])**2 + 
                              (positions[i, 1] - positions[j, 1])**2)
                edge_list.append([i, j])
                edge_info.append([a_diff, dist, ring_diff])
    
    return edge_list, edge_info, eccentricity, angle


# ==================== グラフ作成 ====================

def compute_sensitivity_ratio(gap_sens_i, gap_sens_j, ecc_i, ecc_j):
    """
    感度比率を計算（両方向）
    
    Args:
        gap_sens_i: ノードiのGAP感度
        gap_sens_j: ノードjのGAP感度
        ecc_i: ノードiの偏心度
        ecc_j: ノードjの偏心度
    
    Returns:
        (outer_inner, inner_outer): 両方向の感度比率
        - outer_inner: 外側÷内側（中心から外への変化）
        - inner_outer: 内側÷外側（外から中心への変化）
        被験者ごとに異なる値！
    """
    # 外側と内側を判定
    if ecc_j > ecc_i:
        # j が外側
        outer_sens = gap_sens_j
        inner_sens = gap_sens_i
    elif ecc_i > ecc_j:
        # i が外側
        outer_sens = gap_sens_i
        inner_sens = gap_sens_j
    else:
        # 同じ偏心度の場合は1.0
        return 1.0, 1.0
    
    # ゼロ除算を回避
    inner_safe = max(inner_sens, SENSITIVITY_EPSILON)
    outer_safe = max(outer_sens, SENSITIVITY_EPSILON)
    
    # 外側÷内側
    ratio_outer_inner = outer_sens / inner_safe
    ratio_outer_inner = np.clip(ratio_outer_inner, SENSITIVITY_RATIO_MIN, SENSITIVITY_RATIO_MAX)
    
    # 内側÷外側
    ratio_inner_outer = inner_sens / outer_safe
    ratio_inner_outer = np.clip(ratio_inner_outer, SENSITIVITY_RATIO_MIN, SENSITIVITY_RATIO_MAX)
    
    return ratio_outer_inner, ratio_inner_outer


def create_graph_top_strategy(df_gap, df_hfa, eye_side, pattern_name, correlation_dict,
                               angle_same_deg, angle_adj_deg, tolerance, relaxation_factor):
    """
    TOP Strategy用のグラフを作成（標準グリッド使用 + 感度比率エッジ特徴量）
    
    Args:
        df_gap: GAPデータ
        df_hfa: HFAデータ
        eye_side: 眼側（0=Left, 1=Right）
        pattern_name: パターン名
        correlation_dict: 位置間の相関辞書
        angle_same_deg: 同一リング角度閾値
        angle_adj_deg: 隣接リング角度閾値
        tolerance: 偏心度許容差
        relaxation_factor: 緩和係数
    
    Returns:
        PyTorch Geometric Data オブジェクト
    """
    
    # 標準グリッドを取得
    standard_grid = get_standard_grid(pattern_name, eye_side, exclude_mariotte=True)
    n_points = len(standard_grid)
    
    # GAPデータをマッチング
    df_gap_eye = df_gap[df_gap['EyeSide'] == eye_side].copy()
    gap_sens = match_data_to_grid(df_gap_eye, standard_grid, value_col='GAPSensitivity')
    
    # HFAデータをマッチング
    df_hfa_eye = df_hfa[df_hfa['EyeSide'] == eye_side].copy()
    hfa_sens = match_data_to_grid(df_hfa_eye, standard_grid, value_col='Sensitivity')
    
    # 両方のデータがある点のみ使用
    valid_mask = np.isfinite(gap_sens) & np.isfinite(hfa_sens)
    if valid_mask.sum() < 5:
        return None
    
    positions = standard_grid[valid_mask]
    gap_valid = np.clip(gap_sens[valid_mask], SENSITIVITY_MIN, SENSITIVITY_MAX)
    hfa_valid = np.clip(hfa_sens[valid_mask], SENSITIVITY_MIN, SENSITIVITY_MAX)
    
    # 極座標
    eccentricity, angle = compute_polar_coordinates(positions)
    
    # ノード特徴量: [NormX, NormY, GAP感度(正規化), 偏心度(正規化), 角度(正規化)]
    max_ecc = max(eccentricity.max(), 1.0)
    features = np.column_stack([
        positions[:, 0] / 30.0,           # NormX
        positions[:, 1] / 30.0,           # NormY
        gap_valid / SENSITIVITY_MAX,       # GAP感度
        eccentricity / max_ecc,            # 偏心度
        (angle + np.pi) / (2 * np.pi)      # 角度
    ]).astype(np.float32)
    
    # 隣接計算（パラメータ化版）
    edge_list, edge_info, ecc_arr, _ = compute_angular_adjacency(
        positions, pattern_name,
        angle_same_deg, angle_adj_deg,
        tolerance, relaxation_factor
    )
    
    if len(edge_list) == 0:
        return None
    
    # バリデーション
    edge_array = np.array(edge_list)
    if edge_array.max() >= len(positions) or edge_array.min() < 0:
        return None
    
    # エッジ特徴量（設定により次元が変わる）
    edge_attrs = []
    for idx, (i, j) in enumerate(edge_list):
        angle_diff, dist, ring_diff = edge_info[idx]
        
        attr = []
        
        # 1. 相関係数（オプション）
        if USE_CORRELATION:
            pos_i = tuple(np.round(positions[i]).astype(int))
            pos_j = tuple(np.round(positions[j]).astype(int))
            corr = correlation_dict.get((pos_i, pos_j), 0.5)
            corr_norm = (corr + 1) / 2  # [-1, 1] → [0, 1]
            attr.append(corr_norm)
        
        # 2. 角度類似度（オプション）
        if USE_ANGLE_SIMILARITY:
            angle_sim = 1.0 - (angle_diff / np.pi)
            attr.append(angle_sim)
        
        # 感度比率を計算（両方向）
        ratio_outer_inner, ratio_inner_outer = compute_sensitivity_ratio(
            gap_valid[i], gap_valid[j],
            ecc_arr[i], ecc_arr[j]
        )
        
        # 3. ★感度比率（外側÷内側）
        if USE_SENSITIVITY_RATIO:
            sens_ratio_norm = np.log10(ratio_outer_inner) / np.log10(SENSITIVITY_RATIO_MAX)
            sens_ratio_norm = np.clip(sens_ratio_norm, -1.0, 1.0)
            sens_ratio_norm = (sens_ratio_norm + 1.0) / 2.0
            attr.append(sens_ratio_norm)
        
        # 4. ★感度比率（内側÷外側）
        if USE_SENSITIVITY_RATIO_INVERSE:
            sens_ratio_inv_norm = np.log10(ratio_inner_outer) / np.log10(SENSITIVITY_RATIO_MAX)
            sens_ratio_inv_norm = np.clip(sens_ratio_inv_norm, -1.0, 1.0)
            sens_ratio_inv_norm = (sens_ratio_inv_norm + 1.0) / 2.0
            attr.append(sens_ratio_inv_norm)
        
        # 5. 距離重み
        if USE_DISTANCE_WEIGHT:
            dist_weight = 1.0 / (1.0 + dist / 10.0)
            attr.append(dist_weight)
        
        edge_attrs.append(attr)
    
    # グラフ作成
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    graph = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(hfa_valid, dtype=torch.float32),
        gap_sensitivity=torch.tensor(gap_valid, dtype=torch.float32),
        pos=torch.tensor(positions, dtype=torch.float32),
        eye_side=eye_side
    )
    
    return graph


# ==================== 相関計算 ====================

def compute_correlation_matrix(all_sensitivities):
    """位置間の感度相関を計算"""
    positions = list(all_sensitivities.keys())
    n_positions = len(positions)
    correlation_dict = {}
    
    for i in range(n_positions):
        for j in range(i+1, n_positions):
            pos_i = positions[i]
            pos_j = positions[j]
            
            subjects_i = set(all_sensitivities[pos_i].keys())
            subjects_j = set(all_sensitivities[pos_j].keys())
            common = subjects_i & subjects_j
            
            if len(common) >= 5:
                sens_i = np.array([all_sensitivities[pos_i][s] for s in common])
                sens_j = np.array([all_sensitivities[pos_j][s] for s in common])
                
                valid = np.isfinite(sens_i) & np.isfinite(sens_j)
                if valid.sum() >= 5:
                    corr = np.corrcoef(sens_i[valid], sens_j[valid])[0, 1]
                    if np.isfinite(corr):
                        correlation_dict[(pos_i, pos_j)] = corr
                        correlation_dict[(pos_j, pos_i)] = corr
    
    return correlation_dict


def collect_correlation_data(subject_folders):
    """相関計算用データを収集"""
    print("\nCollecting data for correlation computation...")
    
    data_by_pattern_eye = {
        f"{eye}_{pattern}": defaultdict(dict)
        for eye in ['Left', 'Right']
        for pattern in ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    }
    
    for subject_folder in tqdm(subject_folders, desc="Collecting"):
        subject_id = subject_folder.name
        
        df_gap = load_gap_data(subject_folder)
        if df_gap is None:
            continue
        
        df_hfa, test_pattern = load_hfa_data(subject_folder)
        if df_hfa is None or test_pattern not in PATTERN_NAME_MAP:
            continue
        
        pattern_name = PATTERN_NAME_MAP[test_pattern]
        
        for eye_side in [0, 1]:
            eye_name = EYE_MAP[eye_side]
            key = f"{eye_name}_{pattern_name}"
            
            standard_grid = get_standard_grid(pattern_name, eye_side, exclude_mariotte=True)
            df_gap_eye = df_gap[df_gap['EyeSide'] == eye_side]
            gap_sens = match_data_to_grid(df_gap_eye, standard_grid, value_col='GAPSensitivity')
            
            for i, (gx, gy) in enumerate(standard_grid):
                if np.isfinite(gap_sens[i]):
                    pos = (int(gx), int(gy))
                    data_by_pattern_eye[key][pos][subject_id] = gap_sens[i]
    
    # 相関計算
    correlation_matrices = {}
    for key, data in data_by_pattern_eye.items():
        if len(data) > 0:
            corr_dict = compute_correlation_matrix(data)
            correlation_matrices[key] = corr_dict
            print(f"  {key}: {len(corr_dict)} correlation pairs")
    
    return correlation_matrices


# ==================== メイン処理 ====================

def process_all_subjects(args):
    """全被験者を処理"""
    
    # 出力パス設定
    output_suffix = args.output_suffix if args.output_suffix else get_output_suffix(args.angle)
    OUTPUT_PATH = GNN_PROJECT_PATH / "data" / f"by_eye_pattern_top{output_suffix}"
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    if not DATA_ROOT.exists():
        print(f"\n❌ Data root not found: {DATA_ROOT}")
        return
    
    subject_folders = [f for f in DATA_ROOT.iterdir() if f.is_dir()]
    print(f"\nFound {len(subject_folders)} subject folders")
    
    # パラメータ表示
    print("\n★ Angular Adjacency Parameters:")
    print(f"  Same ring angle threshold: {args.angle}°")
    print(f"  Adjacent ring angle threshold: {args.adj_angle}°")
    print(f"  Eccentricity tolerance: {args.tolerance} rings")
    print(f"  Relaxation factor: {args.relaxation}")
    print(f"  Output suffix: {output_suffix}")
    
    # 期待される点数を表示
    print("\n★ Expected point counts:")
    for pattern in ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']:
        for eye_side in [0, 1]:
            eye_name = EYE_MAP[eye_side]
            grid = get_standard_grid(pattern, eye_side, exclude_mariotte=True)
            print(f"  {eye_name}_{pattern}: {len(grid)} points")
    
    # 相関行列を計算（USE_CORRELATIONがTrueの場合のみ）
    if USE_CORRELATION:
        correlation_matrices = collect_correlation_data(subject_folders)
    else:
        correlation_matrices = {}
        print("\n※ Correlation計算をスキップ（USE_CORRELATION=False）")
    
    # グラフ作成
    graphs_by_eye_pattern = {
        f"{EYE_MAP[eye]}_{pattern}": []
        for eye in [0, 1]
        for pattern in ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']
    }
    
    stats = {
        'total_subjects': 0,
        'gap_hfa_matched': 0,
        'graphs_created': 0
    }
    
    print("\nCreating graphs...")
    
    for subject_folder in tqdm(subject_folders, desc="Processing"):
        stats['total_subjects'] += 1
        
        df_gap = load_gap_data(subject_folder)
        if df_gap is None:
            continue
        
        df_hfa, test_pattern = load_hfa_data(subject_folder)
        if df_hfa is None or test_pattern not in PATTERN_NAME_MAP:
            continue
        
        pattern_name = PATTERN_NAME_MAP[test_pattern]
        stats['gap_hfa_matched'] += 1
        
        for eye_side in [0, 1]:
            key = f"{EYE_MAP[eye_side]}_{pattern_name}"
            corr_dict = correlation_matrices.get(key, {})
            
            graph = create_graph_top_strategy(
                df_gap, df_hfa, eye_side, pattern_name, corr_dict,
                args.angle, args.adj_angle, args.tolerance, args.relaxation
            )
            
            if graph is not None:
                graphs_by_eye_pattern[key].append(graph)
                stats['graphs_created'] += 1
    
    # 統計表示
    print(f"\n{'='*70}")
    print("Statistics")
    print(f"{'='*70}")
    print(f"Total subjects: {stats['total_subjects']}")
    print(f"GAP-HFA matched: {stats['gap_hfa_matched']}")
    print(f"Graphs created: {stats['graphs_created']}")
    
    # パラメータをファイルに保存
    params_file = OUTPUT_PATH / "angular_params.txt"
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("Angular Adjacency Parameters (TOP Strategy)\n")
        f.write("="*50 + "\n")
        f.write(f"Same ring angle threshold: {args.angle}°\n")
        f.write(f"Adjacent ring angle threshold: {args.adj_angle}°\n")
        f.write(f"Eccentricity tolerance: {args.tolerance} rings\n")
        f.write(f"Relaxation factor: {args.relaxation}\n")
        f.write(f"Output suffix: {output_suffix}\n")
        f.write("\n【Edge Features】\n")
        idx = 1
        if USE_CORRELATION:
            f.write(f"  {idx}. Correlation (normalized)\n")
            idx += 1
        if USE_ANGLE_SIMILARITY:
            f.write(f"  {idx}. AngleSimilarity\n")
            idx += 1
        if USE_SENSITIVITY_RATIO:
            f.write(f"  {idx}. SensRatio_OuterInner: 外側÷内側 ★被験者ごとに異なる\n")
            idx += 1
        if USE_SENSITIVITY_RATIO_INVERSE:
            f.write(f"  {idx}. SensRatio_InnerOuter: 内側÷外側 ★被験者ごとに異なる\n")
            idx += 1
        if USE_DISTANCE_WEIGHT:
            f.write(f"  {idx}. DistanceWeight\n")
    print(f"\n✓ Saved parameters: {params_file}")
    
    # 保存
    print(f"\n{'='*70}")
    print("Saving graph data")
    print(f"{'='*70}")
    
    for key, graph_list in graphs_by_eye_pattern.items():
        if len(graph_list) == 0:
            print(f"\n{key}: No data (skipped)")
            continue
        
        point_counts = [g.num_nodes for g in graph_list]
        edge_counts = [g.num_edges for g in graph_list]
        
        print(f"\n{key}:")
        print(f"  Graphs: {len(graph_list)}")
        print(f"  Points per graph: min={min(point_counts)}, max={max(point_counts)}, mean={np.mean(point_counts):.1f}")
        print(f"  Edges per graph: min={min(edge_counts)}, max={max(edge_counts)}, mean={np.mean(edge_counts):.1f}")
        
        n_features = graph_list[0].x.shape[1]
        edge_dim = graph_list[0].edge_attr.shape[1]
        
        # GAP-HFA相関
        all_gap = []
        all_hfa = []
        for g in graph_list:
            all_gap.extend(g.gap_sensitivity.numpy().tolist())
            all_hfa.extend(g.y.numpy().tolist())
        gap_hfa_corr = np.corrcoef(all_gap, all_hfa)[0, 1]
        
        print(f"  GAP-HFA correlation: {gap_hfa_corr:.3f}")
        print(f"  Node features: {n_features}")
        print(f"  Edge features: {edge_dim}")
        
        # エッジ特徴量名を動的に生成
        edge_feature_names = []
        if USE_CORRELATION:
            edge_feature_names.append('Correlation')
        if USE_ANGLE_SIMILARITY:
            edge_feature_names.append('AngleSimilarity')
        if USE_SENSITIVITY_RATIO:
            edge_feature_names.append('SensRatio_OuterInner')
        if USE_SENSITIVITY_RATIO_INVERSE:
            edge_feature_names.append('SensRatio_InnerOuter')
        if USE_DISTANCE_WEIGHT:
            edge_feature_names.append('DistanceWeight')
        
        print(f"  Edge feature names: {edge_feature_names}")
        
        eye_name, pattern_name = key.split('_', 1)
        
        pattern_data = {
            'graph_list': graph_list,
            'n_features': n_features,
            'edge_attr_dim': edge_dim,
            'eye_side': int(eye_name == 'Right'),
            'pattern_id': PATTERN_ID_MAP[pattern_name],
            'pattern_name': pattern_name,
            'feature_names': ['NormX', 'NormY', 'GAP_Sensitivity', 'Eccentricity', 'Angle'],
            'edge_feature_names': edge_feature_names,
            'target_name': 'HFA_Sensitivity',
            'adjacency_type': 'angular_top_strategy',
            'sensitivity_range': [SENSITIVITY_MIN, SENSITIVITY_MAX],
            'use_correlation': USE_CORRELATION,
            'use_angle_similarity': USE_ANGLE_SIMILARITY,
            'use_sensitivity_ratio': USE_SENSITIVITY_RATIO,
            'use_sensitivity_ratio_inverse': USE_SENSITIVITY_RATIO_INVERSE,
            'use_distance_weight': USE_DISTANCE_WEIGHT,
            'angular_params': {
                'angle_same_deg': args.angle,
                'angle_adj_deg': args.adj_angle,
                'tolerance': args.tolerance,
                'relaxation': args.relaxation
            }
        }
        
        output_file = OUTPUT_PATH / f"graph_data_{key}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(pattern_data, f)
        
        print(f"  ✓ Saved: {output_file.name}")
    
    print(f"\n{'='*70}")
    print("Graph Data Creation Completed!")
    print(f"{'='*70}")
    print(f"\n★ TOP Strategy Design:")
    print(f"  Input: GAP感度（1回測定）+ 空間情報")
    print(f"  Output: HFA感度（最終閾値）")
    print(f"  GNN: 隣接点のGAP感度を集約して閾値を補正")
    print(f"\n★ Edge Features (4D):")
    print(f"  1. Correlation: 位置間の感度相関")
    print(f"  2. AngleSimilarity: 角度の類似度")
    print(f"  3. DistanceWeight: 距離による重み")
    print(f"  4. SensitivityRatio: 外側/内側の感度比 ★NEW")
    print(f"\n★ Output: {OUTPUT_PATH}")


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description='TOP Strategy GNN用グラフ構築（角度パラメータ化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python create_graph_top_strategy.py --angle 45
  python create_graph_top_strategy.py --angle 30 --adj-angle 60
  python create_graph_top_strategy.py --angle 60 --tolerance 2 --output-suffix _custom
        """
    )
    
    parser.add_argument('--angle', '-a', type=float, 
                        default=ANGLE_THRESHOLD_SAME_RING_DEG,
                        help=f'同一リングの角度閾値（度）。デフォルト: {ANGLE_THRESHOLD_SAME_RING_DEG}')
    
    parser.add_argument('--adj-angle', '-A', type=float, 
                        default=ANGLE_THRESHOLD_ADJACENT_RING_DEG,
                        help=f'隣接リングの角度閾値（度）。デフォルト: {ANGLE_THRESHOLD_ADJACENT_RING_DEG}')
    
    parser.add_argument('--tolerance', '-t', type=int, 
                        default=ECCENTRICITY_TOLERANCE,
                        choices=[0, 1, 2],
                        help=f'偏心度リングの許容差。デフォルト: {ECCENTRICITY_TOLERANCE}')
    
    parser.add_argument('--relaxation', '-r', type=float, 
                        default=RING_RELAXATION_FACTOR,
                        help=f'リング緩和係数。デフォルト: {RING_RELAXATION_FACTOR}')
    
    parser.add_argument('--output-suffix', '-o', type=str, 
                        default=None,
                        help='出力ディレクトリのサフィックス。未指定時は_angle{角度}')
    
    return parser.parse_args()


if __name__ == "__main__":
    print("="*70)
    print("Creating Graph Data for TOP Strategy")
    print("(Angular Parameterized + Sensitivity Ratio Edge Feature)")
    print("="*70)
    
    args = parse_args()
    
    print("\n★ Using Standard Grid Definitions:")
    print("  10-2: 68 points")
    print("  24-2: 54 points (52 after Mariotte exclusion)")
    print("  30-2: 76 points (74 after Mariotte exclusion)")
    print("\n★ Goal: Use neighboring points to refine threshold estimates")
    print("\n【Node Features】")
    print("  - GAP Sensitivity (initial measurement) ← Main input")
    print("  - Eccentricity (distance from center)")
    print("  - Angle (polar coordinate)")
    print("\n【Edge Features】★4次元")
    print("  - Correlation (how similar are neighboring points)")
    print("  - Angle similarity")
    print("  - Distance weight")
    print("  - Sensitivity Ratio (outer/inner) ★NEW")
    print("\n【Target】")
    print("  - HFA Sensitivity (gold standard)")
    print("="*70)
    
    process_all_subjects(args)