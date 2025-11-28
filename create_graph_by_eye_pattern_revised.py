# -*- coding: utf-8 -*-
"""
create_graph_by_eye_pattern.py
左右眼別・パターン別にグラフデータを作成（マリオット盲点除外版）

修正内容:
- response*.csvからIsMariotte列を読み込み（1=マリオット盲点）
- マリオット盲点の計測点をグラフから除外
- エッジ属性: 2次元 [距離の逆数, 感度類似性]
- 教師データ: HFA_Sensitivity（機能点のみ）
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# パス設定
SCRIPT_DIR = Path(__file__).parent
GNN_PROJECT_PATH = SCRIPT_DIR

# データパス（Googleドライブのパス）
DATA_ROOT = Path("G:/共有ドライブ/GAP_Analysis/Data/GAP2_KyodaiClinical")
OUTPUT_PATH = GNN_PROJECT_PATH / "data" / "by_eye_pattern"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"\nPaths:")
print(f"  Project: {GNN_PROJECT_PATH}")
print(f"  Data Root: {DATA_ROOT}")
print(f"  Output: {OUTPUT_PATH}")

# パターン定義
PATTERN_MAP = {
    0: 'Pattern30-2',
    1: 'Pattern24-2',
    2: 'Pattern10-2'
}

EYE_MAP = {
    0: 'Left',
    1: 'Right'
}


def load_response_data(subject_folder):
    """
    response*.csvからマリオット盲点情報を読み込む
    
    Returns:
        pd.DataFrame or None: 応答データ（IsMariotte列を含む）
    """
    # response*.csv を探す
    response_files = list(subject_folder.glob("response_*.csv"))
    
    if len(response_files) == 0:
        return None
    
    # 最初のファイルを使用
    response_file = response_files[0]
    
    try:
        df_response = pd.read_csv(response_file)
        
        # 必須カラムの確認
        required_cols = ['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'IsMariotte']
        
        if not all(col in df_response.columns for col in required_cols):
            print(f"  Warning: Missing required columns in {response_file.name}")
            return None
        
        # 計測点のみをフィルタ（IsInspectionPointer == True）
        if 'IsInspectionPointer' in df_response.columns:
            df_response = df_response[df_response['IsInspectionPointer'] == True].copy()
        
        return df_response
        
    except Exception as e:
        print(f"  Error reading {response_file.name}: {e}")
        return None


def load_hfa_data(subject_folder):
    """
    HFAデータを読み込む
    
    Returns:
        pd.DataFrame or None: HFAの結果データ
    """
    hfa_folder = subject_folder / "HFAMatchData"
    
    if not hfa_folder.exists():
        return None
    
    # hfa_result_*.csv を探す
    hfa_result_files = list(hfa_folder.glob("hfa_result_*.csv"))
    
    if len(hfa_result_files) == 0:
        return None
    
    # 最初のファイルを使用
    hfa_file = hfa_result_files[0]
    
    try:
        df_hfa = pd.read_csv(hfa_file)
        
        # 必須カラムの確認
        required_cols = ['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'Sensitivity']
        
        if not all(col in df_hfa.columns for col in required_cols):
            print(f"  Warning: Missing required columns in {hfa_file.name}")
            return None
        
        return df_hfa
        
    except Exception as e:
        print(f"  Error reading {hfa_file.name}: {e}")
        return None


def match_mariotte_flags(df_hfa, df_response):
    """
    HFAデータとresponseデータを座標でマッチングし、マリオット盲点フラグを設定
    
    Args:
        df_hfa: pd.DataFrame, HFAデータ
        df_response: pd.DataFrame, responseデータ（IsMariotte列を含む）
    
    Returns:
        pd.DataFrame: マリオット盲点フラグが追加されたHFAデータ
    """
    df_hfa = df_hfa.copy()
    df_hfa['IsMariotte'] = False  # デフォルトはFalse
    
    # 各HFAの点について、responseデータから対応する点を探す
    for idx, row in df_hfa.iterrows():
        eye_side = row['EyeSide']
        x = row['InspectionAngleX']
        y = row['InspectionAngleY']
        
        # responseデータから対応する点を探す（座標で±1度以内）
        match_mask = (
            (df_response['EyeSide'] == eye_side) &
            (np.abs(df_response['InspectionAngleX'] - x) <= 1) &
            (np.abs(df_response['InspectionAngleY'] - y) <= 1)
        )
        
        matched_points = df_response[match_mask]
        
        if len(matched_points) > 0:
            # マリオット盲点フラグを取得（1ならTrue）
            is_mariotte = matched_points['IsMariotte'].iloc[0]
            df_hfa.at[idx, 'IsMariotte'] = bool(is_mariotte == 1 or is_mariotte == True)
    
    return df_hfa


def compute_edge_attributes(positions, sensitivities):
    """
    エッジ属性を計算: [距離の逆数, 感度類似性]
    
    Args:
        positions: np.ndarray, shape (N, 2)
        sensitivities: np.ndarray, shape (N,)
    
    Returns:
        edge_index: torch.Tensor, shape (2, E)
        edge_attr: torch.Tensor, shape (E, 2)
    """
    n_points = len(positions)
    
    if n_points < 2:
        # ノードが少なすぎる場合
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
        return edge_index, edge_attr
    
    # k近傍グラフ（k=6）
    k = min(6, n_points - 1)
    
    edge_list = []
    edge_attrs = []
    
    for i in range(n_points):
        # 距離計算
        distances = np.linalg.norm(positions - positions[i], axis=1)
        distances[i] = np.inf  # 自己ループを除外
        
        # k近傍のインデックス
        nearest_indices = np.argsort(distances)[:k]
        
        for j in nearest_indices:
            # エッジを追加
            edge_list.append([i, j])
            
            # エッジ属性1: 距離の逆数（正規化）
            dist = distances[j]
            dist_weight = 1.0 / (dist + 1e-6)
            
            # エッジ属性2: 感度の類似性
            sens_diff = abs(sensitivities[i] - sensitivities[j])
            sens_similarity = np.exp(-sens_diff / 10.0)
            
            edge_attrs.append([dist_weight, sens_similarity])
    
    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    
    return edge_index, edge_attr


def create_graph_for_eye_pattern(df_hfa, eye_side, pattern_id):
    """
    特定の眼・パターンのグラフを作成（マリオット盲点除外版）
    
    Args:
        df_hfa: pd.DataFrame, HFAデータ（IsMariotte列を含む）
        eye_side: int, 0=左眼, 1=右眼
        pattern_id: int, パターンID
    
    Returns:
        Data or None: PyG Dataオブジェクト（マリオット盲点は除外）
    """
    # パターン範囲の定義
    pattern_ranges = {
        0: 30,  # Pattern30-2
        1: 24,  # Pattern24-2
        2: 10   # Pattern10-2
    }
    
    max_range = pattern_ranges.get(pattern_id, 30)
    
    # フィルタリング：指定の眼・パターン範囲・マリオット盲点でない点のみ
    mask = (
        (df_hfa['EyeSide'] == eye_side) &
        (df_hfa['InspectionAngleX'].abs() <= max_range) &
        (df_hfa['InspectionAngleY'].abs() <= max_range) &
        (~df_hfa['IsMariotte'])  # ★ マリオット盲点を除外
    )
    
    df_subset = df_hfa[mask].copy()
    
    if len(df_subset) < 5:
        return None
    
    # データ抽出
    positions = df_subset[['InspectionAngleX', 'InspectionAngleY']].values.astype(np.float32)
    sensitivities = df_subset['Sensitivity'].values.astype(np.float32)
    
    # 特徴量作成（座標 + 感度）
    features = np.column_stack([
        positions,  # X, Y座標
        sensitivities,  # 感度
    ])
    
    # エッジとエッジ属性を計算
    edge_index, edge_attr = compute_edge_attributes(positions, sensitivities)
    
    # PyG Dataオブジェクト作成
    graph = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(sensitivities, dtype=torch.float32),
        pos=torch.tensor(positions, dtype=torch.float32),
        eye_side=eye_side,
        pattern_id=pattern_id
    )
    
    return graph


def process_all_subjects():
    """すべての被験者データを処理"""
    
    if not DATA_ROOT.exists():
        print(f"\n❌ Data root not found: {DATA_ROOT}")
        print("Please check the path and mount Google Drive if necessary")
        return
    
    # 被験者フォルダを取得
    subject_folders = [f for f in DATA_ROOT.iterdir() if f.is_dir()]
    
    print(f"\nFound {len(subject_folders)} subject folders")
    
    # 眼・パターンごとにグラフを集約
    graphs_by_eye_pattern = {
        f"{EYE_MAP[eye]}_{PATTERN_MAP[pattern]}": []
        for eye in [0, 1]
        for pattern in [0, 1, 2]
    }
    
    # 統計情報
    total_points = 0
    total_mariotte_excluded = 0
    
    # 各被験者を処理
    for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
        # responseデータを読み込み（マリオット盲点情報）
        df_response = load_response_data(subject_folder)
        
        if df_response is None:
            continue
        
        # HFAデータを読み込み（教師データ）
        df_hfa = load_hfa_data(subject_folder)
        
        if df_hfa is None:
            continue
        
        # マリオット盲点フラグをマッチング
        df_hfa = match_mariotte_flags(df_hfa, df_response)
        
        # 統計
        total_points += len(df_hfa)
        total_mariotte_excluded += df_hfa['IsMariotte'].sum()
        
        # 各眼・パターンでグラフ作成
        for eye_side in [0, 1]:
            for pattern_id in [0, 1, 2]:
                graph = create_graph_for_eye_pattern(df_hfa, eye_side, pattern_id)
                
                if graph is not None:
                    key = f"{EYE_MAP[eye_side]}_{PATTERN_MAP[pattern_id]}"
                    graphs_by_eye_pattern[key].append(graph)
    
    # 全体統計
    print(f"\n{'='*70}")
    print("Overall Statistics")
    print(f"{'='*70}")
    print(f"Total points processed: {total_points}")
    print(f"Mariotte blind spot points excluded: {total_mariotte_excluded}")
    print(f"Functional points used: {total_points - total_mariotte_excluded}")
    print(f"Exclusion rate: {100 * total_mariotte_excluded / total_points:.1f}%")
    
    # 結果を保存
    print(f"\n{'='*70}")
    print("Saving graph data by eye and pattern")
    print(f"{'='*70}")
    
    for key, graph_list in graphs_by_eye_pattern.items():
        if len(graph_list) == 0:
            print(f"\n{key}: No data (skipped)")
            continue
        
        print(f"\n{key}: {len(graph_list)} graphs")
        
        # 統計情報
        n_features = graph_list[0].x.shape[1]
        avg_nodes = np.mean([g.num_nodes for g in graph_list])
        avg_edges = np.mean([g.edge_index.shape[1] for g in graph_list])
        
        print(f"  Features: {n_features}")
        print(f"  Avg nodes per graph: {avg_nodes:.1f}")
        print(f"  Avg edges per graph: {avg_edges:.1f}")
        print(f"  Note: All points are functional (Mariotte excluded)")
        
        # Pickle形式で保存
        pattern_data = {
            'graph_list': graph_list,
            'n_features': n_features,
            'eye_side': int(key.split('_')[0] == 'Right'),
            'pattern_id': list(PATTERN_MAP.keys())[list(PATTERN_MAP.values()).index('_'.join(key.split('_')[1:]))],
            'edge_attr_dim': 2,
            'feature_names': ['InspectionAngleX', 'InspectionAngleY', 'Sensitivity'],
            'mariotte_excluded': True  # マリオット盲点が除外されていることを明示
        }
        
        output_file = OUTPUT_PATH / f"graph_data_{key}.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(pattern_data, f)
        
        print(f"  ✓ Saved: {output_file.name}")
    
    print(f"\n{'='*70}")
    print("Graph Data Creation Completed!")
    print(f"{'='*70}")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"\n★ Important: Mariotte blind spot points have been EXCLUDED from all graphs")


# メイン処理
if __name__ == "__main__":
    print("="*70)
    print("Creating Graph Data by Eye and Pattern")
    print("★ MARIOTTE BLIND SPOT POINTS ARE EXCLUDED")
    print("Data source: response*.csv (IsMariotte column)")
    print("Teacher data: HFA_Sensitivity (functional points only)")
    print("Edge attributes: [distance_weight, sensitivity_similarity]")
    print("="*70)
    
    process_all_subjects()