# -*- coding: utf-8 -*-
"""
validate_and_fix_graphs.py
グラフデータの検証と修正（マリオット盲点情報保持版）
"""

import torch
import pickle
from pathlib import Path
from torch_geometric.data import Data

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "by_eye_pattern"

print("="*70)
print("Graph Data Validator and Fixer")
print("★ Preserving Mariotte blind spot information")
print("="*70)

if not DATA_PATH.exists():
    print(f"\n❌ Data path not found: {DATA_PATH}")
    exit(1)

data_files = list(DATA_PATH.glob("graph_data_*.pkl"))
print(f"\nFound {len(data_files)} data files")

total_graphs = 0
total_fixed = 0
total_invalid = 0

for data_file in sorted(data_files):
    eye_pattern_name = data_file.stem.replace('graph_data_', '')
    
    print(f"\n{'='*70}")
    print(f"Processing: {eye_pattern_name}")
    print(f"{'='*70}")
    
    try:
        with open(data_file, 'rb') as f:
            pattern_data = pickle.load(f)
        
        graph_list = pattern_data['graph_list']
        print(f"  Graphs: {len(graph_list)}")
        
        fixed_graphs = []
        file_fixed = 0
        file_invalid = 0
        
        for i, graph in enumerate(graph_list):
            num_nodes = graph.x.size(0)
            edge_index = graph.edge_index
            
            # エッジインデックスの検証
            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                min_idx = edge_index.min().item()
                
                if max_idx >= num_nodes or min_idx < 0:
                    # 不正なエッジを除去
                    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & \
                           (edge_index[0] >= 0) & (edge_index[1] >= 0)
                    
                    valid_edges = edge_index[:, mask]
                    
                    if valid_edges.shape[1] < edge_index.shape[1]:
                        # エッジ属性も修正
                        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                            valid_edge_attr = graph.edge_attr[mask]
                        else:
                            valid_edge_attr = None
                        
                        # 新しいグラフを作成
                        fixed_graph = Data(
                            x=graph.x,
                            edge_index=valid_edges,
                            edge_attr=valid_edge_attr,
                            y=graph.y,
                            pos=graph.pos if hasattr(graph, 'pos') else None
                        )
                        
                        # メタデータをコピー（★ is_mariotte を含む）
                        if hasattr(graph, 'eye_side'):
                            fixed_graph.eye_side = graph.eye_side
                        if hasattr(graph, 'pattern_id'):
                            fixed_graph.pattern_id = graph.pattern_id
                        if hasattr(graph, 'feature_names'):
                            fixed_graph.feature_names = graph.feature_names
                        if hasattr(graph, 'is_mariotte'):  # ★ 追加
                            fixed_graph.is_mariotte = graph.is_mariotte
                        
                        fixed_graphs.append(fixed_graph)
                        file_fixed += 1
                        
                        if i < 3:
                            print(f"  Graph {i}: Fixed edges {edge_index.shape[1]} -> {valid_edges.shape[1]}")
                    else:
                        fixed_graphs.append(graph)
                else:
                    fixed_graphs.append(graph)
            else:
                # エッジがないグラフは除外
                print(f"  Graph {i}: No edges (skipping)")
                file_invalid += 1
        
        print(f"\n  Summary:")
        print(f"    Total: {len(graph_list)}")
        print(f"    Fixed: {file_fixed}")
        print(f"    Invalid: {file_invalid}")
        print(f"    Valid: {len(fixed_graphs)}")
        
        # マリオット盲点情報の統計
        if len(fixed_graphs) > 0 and hasattr(fixed_graphs[0], 'is_mariotte'):
            n_mariotte = sum([g.is_mariotte.sum().item() for g in fixed_graphs])
            n_functional = sum([(~g.is_mariotte).sum().item() for g in fixed_graphs])
            print(f"    Mariotte points: {n_mariotte}")
            print(f"    Functional points: {n_functional}")
        
        total_graphs += len(graph_list)
        total_fixed += file_fixed
        total_invalid += file_invalid
        
        # 修正したデータを保存
        if file_fixed > 0 or file_invalid > 0:
            pattern_data['graph_list'] = fixed_graphs
            
            with open(data_file, 'wb') as f:
                pickle.dump(pattern_data, f)
            
            print(f"  ✓ Saved fixed data")
        else:
            print(f"  ✓ No fixes needed")
            
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"  Total graphs: {total_graphs}")
print(f"  Fixed: {total_fixed}")
print(f"  Invalid (removed): {total_invalid}")

if total_fixed > 0 or total_invalid > 0:
    print(f"\n✓ Data has been fixed and saved")
    print(f"\nYou can now run:")
    print(f"  python train_by_eye_pattern_revised.py")
else:
    print(f"\n✓ All data is valid")

print(f"{'='*70}")