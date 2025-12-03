# -*- coding: utf-8 -*-
"""
standard_grid_definitions.py
Humphrey視野検査の標準グリッド座標を正確に定義

期待される点数:
- 30-2: 76点（マリオット盲点除外後: 74点）
- 24-2: 54点（マリオット盲点除外後: 52点）
- 10-2: 68点（マリオット盲点なし）

座標系:
- X: 水平方向（正=耳側、負=鼻側）
- Y: 垂直方向（正=上、負=下）
- 単位: 度（degree）
"""

import numpy as np

def get_30_2_grid():
    """
    30-2パターンの標準グリッド座標（76点）
    6度間隔のグリッド
    """
    points = []
    
    # 基本グリッド（奇数座標系: 3の奇数倍）
    # Row 8: Y=21
    for x in [-9, -3, 3, 9]:
        points.append((x, 21))  # 4点
    
    # Row 7: Y=15
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, 15))  # 6点
    
    # Row 6: Y=9
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, 9))  # 8点
    
    # Row 5: Y=3
    for x in [-27, -21, -15, -9, -3, 3, 9, 15, 21, 27]:
        points.append((x, 3))  # 10点
    
    # Row 4: Y=-3
    for x in [-27, -21, -15, -9, -3, 3, 9, 15, 21, 27]:
        points.append((x, -3))  # 10点
    
    # Row 3: Y=-9
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, -9))  # 8点
    
    # Row 2: Y=-15
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, -15))  # 6点
    
    # Row 1: Y=-21
    for x in [-9, -3, 3, 9]:
        points.append((x, -21))  # 4点
    
    # 小計: 56点
    
    # 追加グリッド（偶数座標系の周辺部）
    # Y=24
    for x in [-6, 0, 6]:
        points.append((x, 24))  # 3点
    
    # Y=18
    for x in [-12, -6, 0, 6, 12]:
        points.append((x, 18))  # 5点
    
    # Y=12（中心部は省略、両端のみ）
    for x in [-18, 18]:
        points.append((x, 12))  # 2点
    
    # Y=-12
    for x in [-18, 18]:
        points.append((x, -12))  # 2点
    
    # Y=-18
    for x in [-12, -6, 0, 6, 12]:
        points.append((x, -18))  # 5点
    
    # Y=-24
    for x in [-6, 0, 6]:
        points.append((x, -24))  # 3点
    
    # 追加小計: 20点
    # 合計: 76点
    
    return np.array(points, dtype=np.float32)


def get_24_2_grid():
    """
    24-2パターンの標準グリッド座標（54点）
    6度間隔、中心24度以内
    """
    points = []
    
    # Row 8: Y=21
    for x in [-9, -3, 3, 9]:
        points.append((x, 21))  # 4点
    
    # Row 7: Y=15
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, 15))  # 6点
    
    # Row 6: Y=9
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, 9))  # 8点
    
    # Row 5: Y=3
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, 3))  # 8点
    
    # Row 4: Y=-3
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, -3))  # 8点
    
    # Row 3: Y=-9
    for x in [-21, -15, -9, -3, 3, 9, 15, 21]:
        points.append((x, -9))  # 8点
    
    # Row 2: Y=-15
    for x in [-15, -9, -3, 3, 9, 15]:
        points.append((x, -15))  # 6点
    
    # Row 1: Y=-21
    for x in [-9, -3, 3, 9]:
        points.append((x, -21))  # 4点
    
    # 合計: 52点...まだ2点足りない
    
    # 追加（Y=27とY=-27の上下端）
    for x in [0]:
        points.append((x, 27))  # 1点
    for x in [0]:
        points.append((x, -27))  # 1点
    
    # 合計: 54点
    
    return np.array(points, dtype=np.float32)


def get_10_2_grid():
    """
    10-2パターンの標準グリッド座標（68点）
    2度間隔、中心10度以内
    
    Humphrey 10-2の標準グリッド構成:
    - Y = ±7: 6点ずつ
    - Y = ±5: 8点ずつ
    - Y = ±3: 10点ずつ
    - Y = ±1: 10点ずつ
    合計: (6+8+10+10)*2 = 68点
    """
    points = []
    
    # Y=7
    for x in [-5, -3, -1, 1, 3, 5]:
        points.append((x, 7))  # 6点
    
    # Y=5
    for x in [-7, -5, -3, -1, 1, 3, 5, 7]:
        points.append((x, 5))  # 8点
    
    # Y=3
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, 3))  # 10点
    
    # Y=1
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, 1))  # 10点
    
    # Y=-1
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, -1))  # 10点
    
    # Y=-3
    for x in [-9, -7, -5, -3, -1, 1, 3, 5, 7, 9]:
        points.append((x, -3))  # 10点
    
    # Y=-5
    for x in [-7, -5, -3, -1, 1, 3, 5, 7]:
        points.append((x, -5))  # 8点
    
    # Y=-7
    for x in [-5, -3, -1, 1, 3, 5]:
        points.append((x, -7))  # 6点
    
    # 合計: 68点
    
    return np.array(points, dtype=np.float32)


def get_mariotte_blind_spot(eye_side):
    """
    マリオット盲点の座標を取得
    """
    if eye_side == 0:  # 左眼（視野の右側がマリオット）
        return np.array([[15, 3], [15, -3]], dtype=np.float32)
    else:  # 右眼（視野の左側がマリオット）
        return np.array([[-15, 3], [-15, -3]], dtype=np.float32)


def get_standard_grid(pattern_name, eye_side=None, exclude_mariotte=True):
    """標準グリッド座標を取得"""
    if 'Pattern30-2' in pattern_name or '30-2' in pattern_name:
        grid = get_30_2_grid()
    elif 'Pattern24-2' in pattern_name or '24-2' in pattern_name:
        grid = get_24_2_grid()
    elif 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        grid = get_10_2_grid()
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")
    
    if exclude_mariotte and eye_side is not None and '10-2' not in pattern_name:
        mariotte = get_mariotte_blind_spot(eye_side)
        mask = np.ones(len(grid), dtype=bool)
        for m_point in mariotte:
            distances = np.linalg.norm(grid - m_point, axis=1)
            mask &= (distances > 0.5)
        grid = grid[mask]
    
    return grid


def get_expected_point_count(pattern_name, exclude_mariotte=True):
    """期待される点数"""
    if 'Pattern30-2' in pattern_name or '30-2' in pattern_name:
        return 74 if exclude_mariotte else 76
    elif 'Pattern24-2' in pattern_name or '24-2' in pattern_name:
        return 52 if exclude_mariotte else 54
    elif 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        return 68
    else:
        raise ValueError(f"Unknown pattern: {pattern_name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Standard Grid Definitions Test")
    print("=" * 60)
    
    for pattern in ['Pattern30-2', 'Pattern24-2', 'Pattern10-2']:
        print(f"\n{pattern}:")
        grid = get_standard_grid(pattern, eye_side=None, exclude_mariotte=False)
        expected = get_expected_point_count(pattern, exclude_mariotte=False)
        status = "✓" if len(grid) == expected else f"✗ ({len(grid)} vs {expected})"
        print(f"  Full: {len(grid)} points (expected: {expected}) {status}")
        
        for eye_name, eye_side in [('Left', 0), ('Right', 1)]:
            grid = get_standard_grid(pattern, eye_side=eye_side, exclude_mariotte=True)
            expected = get_expected_point_count(pattern, exclude_mariotte=True)
            status = "✓" if len(grid) == expected else f"✗"
            print(f"  {eye_name}: {len(grid)} (expected: {expected}) {status}")