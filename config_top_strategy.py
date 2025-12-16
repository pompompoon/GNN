# -*- coding: utf-8 -*-
"""
config_top_strategy.py
TOP Strategy GNN用の設定ファイル

コマンドライン引数で上書き可能なパラメータを一元管理
"""

# ==================== 隣接定義パラメータ ====================

# 同一偏心度リング内での角度閾値（度）
ANGLE_THRESHOLD_SAME_RING_DEG = 60.0

# 隣接偏心度リング間での角度閾値（度）
ANGLE_THRESHOLD_ADJACENT_RING_DEG = 90.0

# 偏心度リングの許容差（何リング離れた点まで接続するか）
# 0 = 同一リングのみ, 1 = 隣接リングまで, 2 = 2リング先まで
ECCENTRICITY_TOLERANCE = 1

# 隣接リングでの閾値緩和係数
# 1リング離れるごとに閾値を (1 + RING_RELAXATION_FACTOR * ring_diff) 倍
RING_RELAXATION_FACTOR = 0.5


# ==================== 感度設定 ====================

SENSITIVITY_MIN = 0.0
SENSITIVITY_MAX = 40.0


# ==================== 訓練設定 ====================

BATCH_SIZE = 1
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 30
WEIGHT_DECAY = 1e-4


# ==================== Monte Carlo Dropout ====================

MC_SAMPLES = 30


# ==================== 重要度の重み ====================

# 4指標の重み（合計=1.0）
WEIGHT_PRED_STD = 0.30           # 不確実性
WEIGHT_PREDICTION_ERROR = 0.25   # 予測誤差
WEIGHT_GNN_CORRECTION = 0.25     # GNN補正量
WEIGHT_LEAVE_ONE_OUT = 0.20      # 除外影響度


# ==================== パターンマッピング ====================

PATTERN_NAME_MAP = {
    '中心30-2 閾値テスト': 'Pattern30-2',
    '中心24-2 閾値テスト': 'Pattern24-2',
    '中心10-2 閾値テスト': 'Pattern10-2',
}

PATTERN_ID_MAP = {
    'Pattern30-2': 0,
    'Pattern24-2': 1,
    'Pattern10-2': 2,
}

EYE_MAP = {0: 'Left', 1: 'Right'}


# ==================== エッジ特徴量設定 ====================

# エッジ特徴量の構成
# True = 使用する, False = 使用しない
USE_CORRELATION = True              # 位置間の感度相関係数 ← 無効化
USE_ANGLE_SIMILARITY = True          # 角度の類似度 (1 - angle_diff/π)
USE_SENSITIVITY_RATIO = True         # 感度比率（外側/内側）★被験者ごとに異なる
USE_SENSITIVITY_RATIO_INVERSE = True # 感度比率（内側/外側）★逆方向も追加
USE_DISTANCE_WEIGHT = True           # 距離による重み (1 / (1 + dist/10))

# 感度比率のクリッピング範囲
SENSITIVITY_RATIO_MIN = 0.1
SENSITIVITY_RATIO_MAX = 10.0

# 感度の最小値（0除算防止）
SENSITIVITY_EPSILON = 1.0  # dB


# ==================== ヘルパー関数 ====================

def get_output_suffix(angle_deg):
    """角度に基づく出力ディレクトリサフィックスを生成"""
    return f"_angle{int(angle_deg)}"


def print_config():
    """現在の設定を表示"""
    print("\n" + "="*50)
    print("TOP Strategy Configuration")
    print("="*50)
    print(f"\n【隣接定義】")
    print(f"  同一リング角度閾値: {ANGLE_THRESHOLD_SAME_RING_DEG}°")
    print(f"  隣接リング角度閾値: {ANGLE_THRESHOLD_ADJACENT_RING_DEG}°")
    print(f"  偏心度許容差: {ECCENTRICITY_TOLERANCE} リング")
    print(f"  緩和係数: {RING_RELAXATION_FACTOR}")
    print(f"\n【感度設定】")
    print(f"  範囲: [{SENSITIVITY_MIN}, {SENSITIVITY_MAX}] dB")
    print(f"\n【訓練設定】")
    print(f"  バッチサイズ: {BATCH_SIZE}")
    print(f"  エポック数: {NUM_EPOCHS}")
    print(f"  学習率: {LEARNING_RATE}")
    print(f"  早期終了: {EARLY_STOPPING_PATIENCE} epochs")
    print(f"\n【重要度の重み】")
    print(f"  pred_std: {WEIGHT_PRED_STD}")
    print(f"  prediction_error: {WEIGHT_PREDICTION_ERROR}")
    print(f"  gnn_correction: {WEIGHT_GNN_CORRECTION}")
    print(f"  leave_one_out: {WEIGHT_LEAVE_ONE_OUT}")
    print("="*50 + "\n")


if __name__ == "__main__":
    print_config()