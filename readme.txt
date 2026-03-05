GNN — 視野重要度マップ作成システム

■概要
GAP（Gaze-based Automated Perimetry）の測定データから、GNNを用いて隣接点の情報を集約し、HFA（Humphrey Field Analyzer）相当の最終閾値を推定するパイプライン。

■目的
視野検査では通常54〜76点の計測座標について 視野感度（感度値） を測定するが、すべての点を実測すると検査時間が長くなる。
本システムは GNN（Graph Neural Network） を用いて各計測点の視野感度を予測し、「周辺の点から感度を予測しにくい＝実測が不可欠な点」を特定して、
重要度マップ（Importance Map） として出力する。



重要度スコアの定義
各計測点の重要度スコアは、以下の 3指標の加重平均 で算出する。


指標　　　　　　　意味　　　　　　　　　　　　　内容
pred_std　　　　　モデルの不確実性　　　　　　　モデルが出力する予測の標準偏差（予測の不確実性）

pred_error　　　　実際の予測精度　　　　　　　　|predicted - ground_truth|（実際の予測誤差）

leave_one_out　　 その点の構造的重要性　　　 　　その点を除外したときのMAE変化量（除外影響度）


■実行手順
個別実行
以下の4ステップを順番に実行することで、重要度マップを生成する。
グラフ構築 → 訓練 → 重要度計算 → 可視化


ステップ1: グラフデータを作成
マリオット盲点情報を考慮したグラフ構造を構築する。角度ベースの隣接関係と感度比率エッジ特徴量を含む。
python create_graph_top_strategy.py
ステップ2: モデル訓練
GATConvベースのGNNモデルを訓練する。GAP感度からHFA感度を予測し、隣接点情報による補正効果を学習する。
python train_top_strategy.py
ステップ3: 重要度マップ計算
訓練済みモデルを用いて、各計測点の重要度スコアを4指標（pred_std, prediction_error, gnn_correction, leave_one_out）で算出する。
python compute_importance_top_strategy.py
ステップ4: 可視化
重要度マップを視覚的に出力する。総合スコア図、5パネル図、左右眼比較図を生成する。
python visualize_importance_top_strategy.py

一括実行
上記ステップ1〜4をまとめて実行する場合：
run_top_strategy_analysis.py

※サンプルデータ生成用
python generate_sample_data.py

■出力ディレクトリ構成
{プロジェクトルート}/
├── data/
│   └── by_eye_pattern_top_angle60/       # グラフデータ（角度閾値で命名）
│       ├── graph_data_Left_Pattern30-2.pkl
│       ├── graph_data_Right_Pattern30-2.pkl
│       ├── graph_data_Left_Pattern24-2.pkl
│       └── angular_params.txt
├── models/
│   └── top_strategy_angle60/             # 訓練済みモデル
│       ├── best_model_Left_Pattern30-2.pt
│       └── ...
├── results/
│   └── top_strategy_angle60/             # 重要度スコア・訓練結果
│       ├── analysis_summary_Left_Pattern30-2.csv
│       ├── importance_map_Left_Pattern30-2.pkl
│       ├── training_results_top_strategy.csv
│       └── analysis_summary.txt
└── visualizations/
    └── importance_top_angle60_{timestamp}/  # 可視化画像
        ├── combined_score_Left_Pattern30-2.png
        ├── importance_map_Left_Pattern30-2.png
        ├── comparison_Pattern30-2.png
        └── analysis_summary.txt



今後の方針：
下記のように、計測アルゴリズムを設定する。

・重要度の高い点 → 優先的に実測し、正確な感度値を取得
・重要度の低い点 → 実測した高重要度点の感度値を入力として、GNNが感度を予測し、その予測値を提示感度として採用（計測を省略）

これにより、感度測定の精度を維持しつつ検査時間を大幅に短縮する。
