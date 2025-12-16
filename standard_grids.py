# -*- coding: utf-8 -*-
"""
standard_grids.py
視野検査の標準グリッド定義

実測データに基づく正確な座標定義（左右の目で分離）:
- 10-2: 68点（左右同じ）
- 24-2: 54点（左右で異なる - y=±3の27/-27）
- 30-2: 76点（左右同じ）

マリオット盲点:
- 左目: (15, 3), (15, -3) - 右側（鼻側）
- 右目: (-15, 3), (-15, -3) - 左側（鼻側）
"""

import numpy as np


def get_10_2_grid(eye_side=0):
    """
    10-2パターンの標準グリッド（68点）
    左右の目で同じ
    
    Args:
        eye_side: 0=Left, 1=Right (このパターンでは使用しない)
    """
    points = [
        # y=9
        [-1, 9], [1, 9],
        # y=7
        [-5, 7], [-3, 7], [-1, 7], [1, 7], [3, 7], [5, 7],
        # y=5
        [-7, 5], [-5, 5], [-3, 5], [-1, 5], [1, 5], [3, 5], [5, 5], [7, 5],
        # y=3
        [-7, 3], [-5, 3], [-3, 3], [-1, 3], [1, 3], [3, 3], [5, 3], [7, 3],
        # y=1
        [-9, 1], [-7, 1], [-5, 1], [-3, 1], [-1, 1], [1, 1], [3, 1], [5, 1], [7, 1], [9, 1],
        # y=-1
        [-9, -1], [-7, -1], [-5, -1], [-3, -1], [-1, -1], [1, -1], [3, -1], [5, -1], [7, -1], [9, -1],
        # y=-3
        [-7, -3], [-5, -3], [-3, -3], [-1, -3], [1, -3], [3, -3], [5, -3], [7, -3],
        # y=-5
        [-7, -5], [-5, -5], [-3, -5], [-1, -5], [1, -5], [3, -5], [5, -5], [7, -5],
        # y=-7
        [-5, -7], [-3, -7], [-1, -7], [1, -7], [3, -7], [5, -7],
        # y=-9
        [-1, -9], [1, -9],
    ]
    return np.array(points, dtype=np.float32)


def get_24_2_grid(eye_side=0):
    """
    24-2パターンの標準グリッド（54点）
    ★左右の目で異なる（y=±3の端点が異なる）
    
    Args:
        eye_side: 0=Left (27が右側), 1=Right (-27が左側)
    """
    if eye_side == 0:  # Left eye
        points = [
            # y=21
            [-9, 21], [-3, 21], [3, 21], [9, 21],
            # y=15
            [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
            # y=9
            [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
            # y=3 (左目: 右端が27)
            [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
            # y=-3 (左目: 右端が27)
            [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
            # y=-9
            [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
            # y=-15
            [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
            # y=-21
            [-9, -21], [-3, -21], [3, -21], [9, -21],
        ]
    else:  # Right eye
        points = [
            # y=21
            [-9, 21], [-3, 21], [3, 21], [9, 21],
            # y=15
            [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15],
            # y=9
            [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9],
            # y=3 (右目: 左端が-27)
            [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3],
            # y=-3 (右目: 左端が-27)
            [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3],
            # y=-9
            [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9],
            # y=-15
            [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15],
            # y=-21
            [-9, -21], [-3, -21], [3, -21], [9, -21],
        ]
    return np.array(points, dtype=np.float32)


def get_30_2_grid(eye_side=0):
    """
    30-2パターンの標準グリッド（76点）
    左右の目で同じ
    
    Args:
        eye_side: 0=Left, 1=Right (このパターンでは使用しない)
    """
    points = [
        # y=27
        [-9, 27], [-3, 27], [3, 27], [9, 27],
        # y=21
        [-15, 21], [-9, 21], [-3, 21], [3, 21], [9, 21], [15, 21],
        # y=15
        [-21, 15], [-15, 15], [-9, 15], [-3, 15], [3, 15], [9, 15], [15, 15], [21, 15],
        # y=9
        [-27, 9], [-21, 9], [-15, 9], [-9, 9], [-3, 9], [3, 9], [9, 9], [15, 9], [21, 9], [27, 9],
        # y=3
        [-27, 3], [-21, 3], [-15, 3], [-9, 3], [-3, 3], [3, 3], [9, 3], [15, 3], [21, 3], [27, 3],
        # y=-3
        [-27, -3], [-21, -3], [-15, -3], [-9, -3], [-3, -3], [3, -3], [9, -3], [15, -3], [21, -3], [27, -3],
        # y=-9
        [-27, -9], [-21, -9], [-15, -9], [-9, -9], [-3, -9], [3, -9], [9, -9], [15, -9], [21, -9], [27, -9],
        # y=-15
        [-21, -15], [-15, -15], [-9, -15], [-3, -15], [3, -15], [9, -15], [15, -15], [21, -15],
        # y=-21
        [-15, -21], [-9, -21], [-3, -21], [3, -21], [9, -21], [15, -21],
        # y=-27
        [-9, -27], [-3, -27], [3, -27], [9, -27],
    ]
    return np.array(points, dtype=np.float32)


def get_mariotte_positions(eye_side):
    """
    マリオット盲点の位置
    
    解剖学的事実:
    - 視神経乳頭は網膜の鼻側にある
    - 視野検査では像が反転するため、視野上では耳側に現れる
    - 左目: 耳側 = 左側 (-15°付近)
    - 右目: 耳側 = 右側 (+15°付近)
    
    Args:
        eye_side: 0=Left, 1=Right
    
    Returns:
        マリオット盲点座標のリスト
    """
    if eye_side == 0:  # Left eye - 盲点は耳側（左）
        return [(-15, 3), (-15, -3)]
    else:  # Right eye - 盲点は耳側（右）
        return [(15, 3), (15, -3)]


def get_standard_grid(pattern_name, eye_side=0, exclude_mariotte=True):
    """
    標準グリッドを取得
    
    Args:
        pattern_name: 'Pattern10-2', 'Pattern24-2', 'Pattern30-2'
        eye_side: 0=Left, 1=Right
        exclude_mariotte: マリオット盲点を除外するかどうか
    
    Returns:
        標準グリッド座標の配列
    """
    if 'Pattern30-2' in pattern_name or '30-2' in pattern_name:
        grid = get_30_2_grid(eye_side)
    elif 'Pattern24-2' in pattern_name or '24-2' in pattern_name:
        grid = get_24_2_grid(eye_side)  # ★左右で異なる
    elif 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        grid = get_10_2_grid(eye_side)
    else:
        grid = get_30_2_grid(eye_side)
    
    # 10-2はマリオット盲点が範囲外なので除外不要
    if exclude_mariotte and 'Pattern10-2' not in pattern_name and '10-2' not in pattern_name:
        mariotte = get_mariotte_positions(eye_side)
        mask = np.ones(len(grid), dtype=bool)
        for mx, my in mariotte:
            distances = np.sqrt((grid[:, 0] - mx)**2 + (grid[:, 1] - my)**2)
            mask &= (distances > 1.0)
        grid = grid[mask]
    
    return grid


def get_expected_point_count(pattern_name, exclude_mariotte=True):
    """
    期待される点数を取得
    """
    if 'Pattern30-2' in pattern_name or '30-2' in pattern_name:
        return 74 if exclude_mariotte else 76
    elif 'Pattern24-2' in pattern_name or '24-2' in pattern_name:
        return 52 if exclude_mariotte else 54
    elif 'Pattern10-2' in pattern_name or '10-2' in pattern_name:
        return 68
    return 0


# 検証用
if __name__ == "__main__":
    print("Standard Grid Point Counts:")
    print("="*60)
    
    for pattern in ['Pattern10-2', 'Pattern24-2', 'Pattern30-2']:
        print(f"\n{pattern}:")
        
        for eye_side, eye_name in [(0, 'Left'), (1, 'Right')]:
            grid_full = get_standard_grid(pattern, eye_side=eye_side, exclude_mariotte=False)
            grid_excl = get_standard_grid(pattern, eye_side=eye_side, exclude_mariotte=True)
            expected = get_expected_point_count(pattern, exclude_mariotte=True)
            
            print(f"  {eye_name} eye:")
            print(f"    Full grid: {len(grid_full)} points")
            print(f"    After Mariotte exclusion: {len(grid_excl)} points")
            print(f"    Expected: {expected} points")
            print(f"    {'✓ Match' if len(grid_excl) == expected else '✗ Mismatch!'}")
    
    # 24-2の左右差異を確認
    print("\n" + "="*60)
    print("24-2 Left vs Right difference:")
    left_24 = set(map(tuple, get_24_2_grid(eye_side=0)))
    right_24 = set(map(tuple, get_24_2_grid(eye_side=1)))
    
    only_left = left_24 - right_24
    only_right = right_24 - left_24
    
    print(f"  Only in Left:  {sorted(only_left)}")
    print(f"  Only in Right: {sorted(only_right)}")