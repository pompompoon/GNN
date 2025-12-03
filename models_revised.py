# -*- coding: utf-8 -*-
"""
models_revised.py
2次元エッジ属性対応のGNNモデル

エッジ属性: [距離の逆数, 感度相関]
教師データ: HFA_Sensitivity
目的: 周辺感度予測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear


class EdgeWeightedGCNConv(MessagePassing):
    """エッジ属性を重みとして使用するGCN"""
    
    def __init__(self, in_channels, out_channels, edge_dim=2):
        super().__init__(aggr='add')
        self.lin = Linear(in_channels, out_channels)
        # エッジ属性を統合的な重みに変換
        self.edge_weight_mlp = nn.Sequential(
            Linear(edge_dim, 16),
            nn.ReLU(),
            Linear(16, 1),
            nn.Sigmoid()  # 0-1の重み
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)
        
        # エッジ重みを計算
        if edge_attr is not None:
            edge_weight = self.edge_weight_mlp(edge_attr).squeeze(-1)
        else:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class EdgeWeightedGATConv(MessagePassing):
    """エッジ属性を考慮するGAT"""
    
    def __init__(self, in_channels, out_channels, heads=4, edge_dim=2, concat=True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        
        self.lin = Linear(in_channels, heads * out_channels)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))
        
        # エッジ特徴量の埋め込み
        self.edge_encoder = nn.Sequential(
            Linear(edge_dim, heads * out_channels // 4),
            nn.ReLU()
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att)
    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)
        
        if edge_attr is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = torch.zeros(
                edge_index.size(1), 
                self.heads * self.out_channels // 4,
                device=x.device
            )
        
        return self.propagate(edge_index, x=x, edge_embedding=edge_embedding)
    
    def message(self, x_i, x_j, edge_index_i, edge_embedding):
        # Attention mechanism with edge features
        alpha = (torch.cat([x_i, x_j, edge_embedding.view(-1, self.heads, self.out_channels // 4)
                           .expand(-1, -1, self.out_channels)[:, :, :edge_embedding.size(-1) // self.heads]], 
                          dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = F.softmax(alpha, dim=0)
        
        return (x_j * alpha.unsqueeze(-1)).flatten(1)
    
    def update(self, aggr_out):
        if self.concat:
            return aggr_out
        else:
            return aggr_out.mean(dim=1)


class VisibilityGNN(nn.Module):
    """
    視標可視性予測用のGNNモデル（2次元エッジ属性対応）
    
    エッジ属性: [距離の逆数, 感度相関]
    教師データ: HFA_Sensitivity（ノードレベル回帰）
    目的: 周辺感度予測
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels=2, 
                 num_layers=4, dropout=0.4, conv_type='gat', edge_dim=2):
        super(VisibilityGNN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim
        
        # グラフ畳み込み層
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 最初の層
        if conv_type == 'gcn':
            self.convs.append(EdgeWeightedGCNConv(in_channels, hidden_channels, edge_dim=edge_dim))
        elif conv_type == 'gat':
            # 標準のGATv2Convを使用（エッジ属性サポート）
            self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=4, 
                                        edge_dim=edge_dim, concat=True))
            hidden_channels = hidden_channels * 4
        elif conv_type == 'sage':
            # SAGEConvはエッジ属性を直接サポートしないため、GCNを使用
            self.convs.append(EdgeWeightedGCNConv(in_channels, hidden_channels, edge_dim=edge_dim))
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 中間層
        for _ in range(num_layers - 2):
            if conv_type == 'gcn':
                self.convs.append(EdgeWeightedGCNConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
            elif conv_type == 'gat':
                self.convs.append(GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, 
                                            edge_dim=edge_dim, concat=True))
            elif conv_type == 'sage':
                self.convs.append(EdgeWeightedGCNConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # 最終層
        if conv_type == 'gcn':
            self.convs.append(EdgeWeightedGCNConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
        elif conv_type == 'gat':
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels // 4, heads=4, 
                                        edge_dim=edge_dim, concat=True))
        elif conv_type == 'sage':
            self.convs.append(EdgeWeightedGCNConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # HFA感度予測ヘッド（回帰）
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 平均と不確実性を予測
        self.mean_predictor = nn.Linear(hidden_channels // 4, 1)
        self.std_predictor = nn.Sequential(
            nn.Linear(hidden_channels // 4, 1),
            nn.Softplus()  # 正の値を保証
        )
        
        # 可視性分類ヘッド（オプション）
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # グラフ畳み込み
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index, edge_attr)
            
            # BatchNormは訓練時のみ適用（推論時はスキップ）
            if self.training:
                x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(x)
        
        # HFA感度予測（ノードレベル回帰）
        reg_features = self.regression_head(x)
        pred_mean = self.mean_predictor(reg_features).squeeze(-1)
        pred_std = self.std_predictor(reg_features).squeeze(-1)
        
        # 可視性分類（オプション）
        vis_logits = self.classification_head(x)
        
        return pred_mean, pred_std, vis_logits


def create_model(model_name, in_channels, hidden_channels, out_channels=2,
                num_layers=4, dropout=0.4, edge_dim=2):
    """
    モデル作成
    
    Args:
        model_name: 'gcn', 'gat', 'sage'
        in_channels: 入力特徴量の次元
        hidden_channels: 隠れ層の次元
        out_channels: 出力クラス数（分類用）
        num_layers: レイヤー数
        dropout: ドロップアウト率
        edge_dim: エッジ属性の次元（デフォルト2: [距離の逆数, 感度相関]）
    
    Returns:
        VisibilityGNN: モデルインスタンス
    """
    return VisibilityGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=model_name,
        edge_dim=edge_dim
    )