# -*- coding: utf-8 -*-
"""
diagnose_graphs.py
グラフデータの詳細診断
"""

import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "by_eye_pattern"

print("="*70)
print("Graph Data Diagnostic")
print("="*70)

test_pattern = "Left_Pattern30-2"
data_file = DATA_PATH / f"graph_data_{test_pattern}.pkl"

if not data_file.exists():
    print(f"\n❌ Data file not found: {data_file}")
    exit(1)

with open(data_file, 'rb') as f:
    pattern_data = pickle.load(f)

graph_list = pattern_data['graph_list']

print(f"\nPattern: {test_pattern}")
print(f"Total graphs: {len(graph_list)}")

print(f"\n{'='*70}")
print("Analyzing first 5 graphs...")
print(f"{'='*70}")

for i, graph in enumerate(graph_list[:5]):
    print(f"\nGraph {i}:")
    print(f"  x.shape: {graph.x.shape}")
    print(f"  y.shape: {graph.y.shape}")
    print(f"  edge_index.shape: {graph.edge_index.shape}")
    
    if hasattr(graph, 'edge_attr'):
        print(f"  edge_attr.shape: {graph.edge_attr.shape}")
    
    if hasattr(graph, 'pos'):
        print(f"  pos.shape: {graph.pos.shape}")
    
    # 検証
    num_nodes = graph.x.size(0)
    
    if graph.edge_index.numel() > 0:
        max_idx = graph.edge_index.max().item()
        min_idx = graph.edge_index.min().item()
        
        print(f"  num_nodes: {num_nodes}")
        print(f"  edge_index range: [{min_idx}, {max_idx}]")
        
        if max_idx >= num_nodes:
            print(f"  ❌ ERROR: max_idx ({max_idx}) >= num_nodes ({num_nodes})")
        else:
            print(f"  ✓ Valid edge indices")
        
        if min_idx < 0:
            print(f"  ❌ ERROR: min_idx ({min_idx}) < 0")
        else:
            print(f"  ✓ Non-negative indices")
    else:
        print(f"  ⚠️  No edges!")

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")

valid_count = 0
invalid_count = 0
no_edges_count = 0

for i, graph in enumerate(graph_list):
    num_nodes = graph.x.size(0)
    
    if graph.edge_index.numel() == 0:
        no_edges_count += 1
        continue
    
    max_idx = graph.edge_index.max().item()
    min_idx = graph.edge_index.min().item()
    
    if max_idx >= num_nodes or min_idx < 0:
        invalid_count += 1
    else:
        valid_count += 1

print(f"\nValid graphs: {valid_count}")
print(f"Invalid graphs: {invalid_count}")
print(f"No edges: {no_edges_count}")

if invalid_count > 0:
    print(f"\n❌ Found {invalid_count} invalid graphs!")
    print("\nAction required:")
    print("  python recreate_graphs_safe.py")
else:
    print(f"\n✓ All graphs are valid!")
    print("\nThe problem might be in the model's forward pass.")
    print("Check models_revised.py EdgeWeightedGATConv implementation.")

print(f"{'='*70}")