# -*- coding: utf-8 -*-
"""
generate_sample_data.py
TOP Strategy GNNパイプライン用サンプルデータ生成

使用例:
  python generate_sample_data.py
  python generate_sample_data.py --n-subjects 50
  python generate_sample_data.py --output-dir ./sample_data --pattern 30-2

生成されるデータ構造:
  {output_dir}/
    ├── Subject_001/
    │   ├── response_001.csv          # GAP測定データ
    │   └── HFAMatchData/
    │       ├── hfa_result_001.csv    # HFA測定結果
    │       └── hfa_option_001.csv    # HFAテストオプション
    ├── Subject_002/
    │   └── ...
    └── ...

各CSVの構成:
  response_*.csv:
    - EyeSide: 0(左眼) or 1(右眼)
    - InspectionAngleX, InspectionAngleY: 測定位置（度）
    - IsMariotte: マリオット盲点フラグ
    - IsInspectionPointer: 検査ポインタフラグ
    - EstimatedSensitivity: GAP推定感度 (dB)

  hfa_result_*.csv:
    - EyeSide, InspectionAngleX, InspectionAngleY: 上と同じ
    - Sensitivity: HFA測定感度 (dB) ← Gold Standard

  hfa_option_*.csv:
    - TestPattern: テストパターン名（例: '中心30-2 閾値テスト'）
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ==================== 標準グリッド定義 ====================
# create_graph_top_strategy.py と同じグリッド定義を使用

def get_10_2_grid():
    """10-2パターンの標準グリッド（68点）"""
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
    """24-2パターンの標準グリッド（54点）- 左右で異なる"""
    if eye_side == 0:  # Left eye
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
    else:  # Right eye
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


def get_30_2_grid():
    """30-2パターンの標準グリッド（76点）"""
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


# ==================== 感度生成モデル ====================

def generate_normal_hill_of_vision(positions, base_sensitivity=32.0, decay_rate=0.15,
                                    noise_std=2.0, rng=None):
    """
    正常眼の「島状視野 (Hill of Vision)」を模擬する感度を生成
    
    中心部ほど感度が高く、周辺部に向かって低下する。
    
    Args:
        positions: (N, 2) 測定位置
        base_sensitivity: 中心感度のベース値 (dB)
        decay_rate: 偏心度あたりの感度低下率
        noise_std: ノイズの標準偏差
        rng: numpy RandomState
    
    Returns:
        sensitivities: (N,) 感度値 (dB)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    eccentricity = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    
    # 基本的な感度プロファイル（中心高、周辺低）
    sensitivities = base_sensitivity - decay_rate * eccentricity
    
    # 個人差（全体的なオフセット）
    individual_offset = rng.normal(0, 3.0)
    sensitivities += individual_offset
    
    # 測定ノイズ
    sensitivities += rng.normal(0, noise_std, size=len(positions))
    
    # クリッピング（0〜40 dB）
    sensitivities = np.clip(sensitivities, 0.0, 40.0)
    
    return sensitivities


def generate_glaucoma_sensitivity(positions, base_sensitivity=30.0, decay_rate=0.2,
                                   defect_center=None, defect_radius=10.0, 
                                   defect_depth=15.0, noise_std=3.0, rng=None):
    """
    緑内障パターンの感度を生成（局所的な感度低下あり）
    
    Args:
        positions: 測定位置
        defect_center: 暗点の中心位置 (x, y)
        defect_radius: 暗点の半径
        defect_depth: 暗点の深さ (dB)
        noise_std: ノイズの標準偏差
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # ベースライン（正常プロファイル）
    sensitivities = generate_normal_hill_of_vision(
        positions, base_sensitivity, decay_rate, noise_std * 0.5, rng
    )
    
    # 局所暗点を追加
    if defect_center is not None:
        dist_to_defect = np.sqrt(
            (positions[:, 0] - defect_center[0])**2 + 
            (positions[:, 1] - defect_center[1])**2
        )
        defect_mask = dist_to_defect < defect_radius
        defect_weight = np.exp(-0.5 * (dist_to_defect / (defect_radius * 0.5))**2)
        sensitivities -= defect_depth * defect_weight
    
    # 追加ノイズ
    sensitivities += rng.normal(0, noise_std * 0.5, size=len(positions))
    sensitivities = np.clip(sensitivities, 0.0, 40.0)
    
    return sensitivities


def generate_gap_from_hfa(hfa_sensitivity, noise_std=3.0, bias=0.0, rng=None):
    """
    HFA感度を基にGAP推定感度を生成
    
    GAP（1回測定）はHFA（最終閾値）と相関があるが、ノイズが大きい。
    
    Args:
        hfa_sensitivity: HFA感度（Gold Standard）
        noise_std: GAP固有のノイズ
        bias: 系統的バイアス
    """
    if rng is None:
        rng = np.random.RandomState()
    
    gap_sensitivity = hfa_sensitivity + bias + rng.normal(0, noise_std, size=len(hfa_sensitivity))
    gap_sensitivity = np.clip(gap_sensitivity, 0.0, 40.0)
    
    return gap_sensitivity


# ==================== マリオット盲点 ====================

def get_mariotte_positions(eye_side):
    """マリオット盲点の位置を返す"""
    if eye_side == 0:  # Left eye
        return [(-15, 3), (-15, -3)]
    else:  # Right eye
        return [(15, 3), (15, -3)]


def is_mariotte(x, y, eye_side, tolerance=1.0):
    """指定位置がマリオット盲点かどうか"""
    mariotte = get_mariotte_positions(eye_side)
    for mx, my in mariotte:
        if np.sqrt((x - mx)**2 + (y - my)**2) <= tolerance:
            return True
    return False


# ==================== 被験者データ生成 ====================

def generate_subject_data(subject_id, eye_sides, pattern_name, pattern_jp,
                          grid_func, condition='normal', rng=None):
    """
    1被験者分のGAP + HFAデータを生成
    
    Args:
        subject_id: 被験者ID
        eye_sides: 測定する眼側のリスト [0], [1], [0, 1]
        pattern_name: パターン名（英語）
        pattern_jp: パターン名（日本語テストパターン名）
        grid_func: グリッド生成関数
        condition: 'normal' or 'glaucoma'
        rng: RandomState
    
    Returns:
        df_gap: GAP測定データ DataFrame
        df_hfa: HFA測定結果 DataFrame
        df_option: HFAオプション DataFrame
    """
    if rng is None:
        rng = np.random.RandomState()
    
    gap_rows = []
    hfa_rows = []
    
    for eye_side in eye_sides:
        # グリッド取得（24-2は左右で異なる）
        if pattern_name in ['Pattern24-2']:
            grid = grid_func(eye_side)
        else:
            grid = grid_func()
        
        # HFA感度を生成（Gold Standard）
        if condition == 'normal':
            hfa_sens = generate_normal_hill_of_vision(
                grid, base_sensitivity=32.0, decay_rate=0.15, noise_std=1.5, rng=rng
            )
        else:
            # 緑内障：ランダムな位置に暗点
            defect_x = rng.choice([-15, -9, 9, 15])
            defect_y = rng.choice([-15, -9, 9, 15])
            hfa_sens = generate_glaucoma_sensitivity(
                grid, base_sensitivity=28.0, decay_rate=0.2,
                defect_center=(defect_x, defect_y),
                defect_radius=rng.uniform(8, 15),
                defect_depth=rng.uniform(10, 25),
                noise_std=2.0, rng=rng
            )
        
        # GAP感度を生成（HFAからノイズ付き）
        gap_sens = generate_gap_from_hfa(
            hfa_sens, noise_std=rng.uniform(2.0, 5.0), 
            bias=rng.uniform(-1.0, 1.0), rng=rng
        )
        
        # マリオット盲点の処理（10-2以外）
        mariotte_positions = get_mariotte_positions(eye_side)
        
        for i, (x, y) in enumerate(grid):
            is_mar = is_mariotte(x, y, eye_side)
            
            # GAP行
            gap_rows.append({
                'EyeSide': eye_side,
                'InspectionAngleX': float(x),
                'InspectionAngleY': float(y),
                'IsMariotte': is_mar,
                'IsInspectionPointer': True,
                'EstimatedSensitivity': float(gap_sens[i]),
            })
            
            # HFA行
            hfa_rows.append({
                'EyeSide': eye_side,
                'InspectionAngleX': float(x),
                'InspectionAngleY': float(y),
                'Sensitivity': float(hfa_sens[i]),
            })
        
        # GAPデータに非検査ポインタ行も追加（フィルタリングテスト用）
        for _ in range(5):
            gap_rows.append({
                'EyeSide': eye_side,
                'InspectionAngleX': float(rng.uniform(-30, 30)),
                'InspectionAngleY': float(rng.uniform(-30, 30)),
                'IsMariotte': False,
                'IsInspectionPointer': False,
                'EstimatedSensitivity': 0.0,
            })
    
    df_gap = pd.DataFrame(gap_rows)
    df_hfa = pd.DataFrame(hfa_rows)
    df_option = pd.DataFrame({'TestPattern': [pattern_jp]})
    
    return df_gap, df_hfa, df_option


# ==================== メイン ====================

def generate_all_data(output_dir, n_subjects=100, patterns=None, seed=42):
    """
    全被験者のサンプルデータを生成
    
    Args:
        output_dir: 出力ディレクトリ
        n_subjects: 被験者数
        patterns: 生成するパターンのリスト（Noneなら全パターン）
        seed: ランダムシード
    """
    rng = np.random.RandomState(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # パターン定義
    pattern_configs = {
        '30-2': {
            'name': 'Pattern30-2',
            'jp': '中心30-2 閾値テスト',
            'grid_func': get_30_2_grid,
            'n_points': 76,
        },
        '24-2': {
            'name': 'Pattern24-2',
            'jp': '中心24-2 閾値テスト',
            'grid_func': get_24_2_grid,
            'n_points': 54,
        },
        '10-2': {
            'name': 'Pattern10-2',
            'jp': '中心10-2 閾値テスト',
            'grid_func': get_10_2_grid,
            'n_points': 68,
        },
    }
    
    if patterns is None:
        patterns = list(pattern_configs.keys())
    
    print("=" * 70)
    print("Generating Sample Data for TOP Strategy GNN Pipeline")
    print("=" * 70)
    print(f"\nOutput directory: {output_path}")
    print(f"Number of subjects: {n_subjects}")
    print(f"Patterns: {patterns}")
    print(f"Random seed: {seed}")
    
    # 被験者ごとにパターンと状態をランダム割り当て
    total_created = 0
    pattern_counts = {p: 0 for p in patterns}
    condition_counts = {'normal': 0, 'glaucoma': 0}
    
    for i in range(n_subjects):
        subject_id = f"Subject_{i+1:03d}"
        subject_dir = output_path / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        hfa_dir = subject_dir / "HFAMatchData"
        hfa_dir.mkdir(parents=True, exist_ok=True)
        
        # パターンをランダムに選択
        pattern_key = rng.choice(patterns)
        config = pattern_configs[pattern_key]
        
        # 正常 or 緑内障
        condition = 'normal' if rng.random() < 0.6 else 'glaucoma'
        
        # 両眼 or 片眼
        if rng.random() < 0.8:
            eye_sides = [0, 1]  # 両眼
        else:
            eye_sides = [rng.choice([0, 1])]  # 片眼
        
        # データ生成
        df_gap, df_hfa, df_option = generate_subject_data(
            subject_id, eye_sides, config['name'], config['jp'],
            config['grid_func'], condition, rng
        )
        
        # CSV保存
        df_gap.to_csv(subject_dir / f"response_{i+1:03d}.csv", index=False)
        df_hfa.to_csv(hfa_dir / f"hfa_result_{i+1:03d}.csv", index=False)
        df_option.to_csv(hfa_dir / f"hfa_option_{i+1:03d}.csv", index=False)
        
        total_created += 1
        pattern_counts[pattern_key] += 1
        condition_counts[condition] += 1
    
    # 統計表示
    print(f"\n{'=' * 70}")
    print("Generation Complete!")
    print(f"{'=' * 70}")
    print(f"\nTotal subjects: {total_created}")
    print(f"\nPattern distribution:")
    for p, count in pattern_counts.items():
        config = pattern_configs[p]
        print(f"  {config['jp']} ({config['name']}): {count} subjects ({config['n_points']} points/eye)")
    
    print(f"\nCondition distribution:")
    for c, count in condition_counts.items():
        print(f"  {c}: {count} subjects")
    
    print(f"\nDirectory structure:")
    print(f"  {output_path}/")
    print(f"    ├── Subject_001/")
    print(f"    │   ├── response_001.csv")
    print(f"    │   └── HFAMatchData/")
    print(f"    │       ├── hfa_result_001.csv")
    print(f"    │       └── hfa_option_001.csv")
    print(f"    ├── Subject_002/")
    print(f"    │   └── ...")
    print(f"    └── Subject_{n_subjects:03d}/")
    
    # サンプルデータの品質チェック
    print(f"\n{'=' * 70}")
    print("Data Quality Check")
    print(f"{'=' * 70}")
    
    sample_dir = output_path / "Subject_001"
    df_check = pd.read_csv(list(sample_dir.glob("response_*.csv"))[0])
    df_inspection = df_check[df_check['IsInspectionPointer'] == True]
    
    print(f"\nSample (Subject_001):")
    print(f"  Total rows: {len(df_check)}")
    print(f"  Inspection points: {len(df_inspection)}")
    print(f"  EyeSides: {sorted(df_inspection['EyeSide'].unique())}")
    print(f"  Sensitivity range: [{df_inspection['EstimatedSensitivity'].min():.1f}, {df_inspection['EstimatedSensitivity'].max():.1f}] dB")
    
    df_hfa_check = pd.read_csv(list((sample_dir / "HFAMatchData").glob("hfa_result_*.csv"))[0])
    print(f"  HFA Sensitivity range: [{df_hfa_check['Sensitivity'].min():.1f}, {df_hfa_check['Sensitivity'].max():.1f}] dB")
    
    # GAP-HFA相関を計算
    if len(df_inspection) > 0 and len(df_hfa_check) > 0:
        merged = pd.merge(
            df_inspection[['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'EstimatedSensitivity']],
            df_hfa_check[['EyeSide', 'InspectionAngleX', 'InspectionAngleY', 'Sensitivity']],
            on=['EyeSide', 'InspectionAngleX', 'InspectionAngleY']
        )
        if len(merged) > 2:
            corr = np.corrcoef(merged['EstimatedSensitivity'], merged['Sensitivity'])[0, 1]
            print(f"  GAP-HFA correlation: {corr:.3f}")
    
    print(f"\n✓ Sample data is ready!")
    print(f"  To use with the pipeline, update DATA_ROOT in create_graph_top_strategy.py:")
    print(f"  DATA_ROOT = Path(\"{output_path}\")")
    
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description='TOP Strategy GNNパイプライン用サンプルデータ生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python generate_sample_data.py
  python generate_sample_data.py --n-subjects 50
  python generate_sample_data.py --output-dir ./my_sample_data --pattern 30-2 24-2
  python generate_sample_data.py --seed 123

生成データの特徴:
  - 正常眼（60%）と緑内障眼（40%）を混合
  - GAP感度はHFA感度にノイズを付加して生成（相関 r≈0.7-0.9）
  - 両眼（80%）と片眼（20%）をランダムに割り当て
  - マリオット盲点フラグを正しく設定
        """
    )
    
    parser.add_argument('--output-dir', '-o', type=str, 
                        default='./sample_data/GAP2_KyodaiClinical',
                        help='出力ディレクトリ。デフォルト: ./sample_data/GAP2_KyodaiClinical')
    
    parser.add_argument('--n-subjects', '-n', type=int, default=100,
                        help='被験者数。デフォルト: 100')
    
    parser.add_argument('--pattern', '-p', nargs='+', 
                        choices=['30-2', '24-2', '10-2'],
                        default=None,
                        help='生成するパターン（未指定時は全パターン）。例: --pattern 30-2 24-2')
    
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='ランダムシード。デフォルト: 42')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    generate_all_data(
        output_dir=args.output_dir,
        n_subjects=args.n_subjects,
        patterns=args.pattern,
        seed=args.seed
    )
