GNN(1) — 視野重要度マップ作成システム
■システム概要
視野検査では通常54〜76点の計測座標について 視野感度（感度値） を測定するが、すべての点を実測すると検査時間が長くなる。
本システムは GNN（Graph Neural Network） を用いて各計測点の視野感度を予測し、「周辺の点から感度を予測しにくい＝実測が不可欠な点」を特定して、
重要度マップ（Importance Map） として出力する。

基本方針：

・重要度の高い点 → 優先的に実測し、正確な感度値を取得
・重要度の低い点 → 実測した高重要度点の感度値を入力として、GNNが感度を予測し、その予測値を提示感度として採用（計測を省略）

これにより、感度測定の精度を維持しつつ検査時間を大幅に短縮する。

重要度スコアの定義
各計測点の重要度スコアは、以下の 3指標の加重平均 で算出する。


指標　　　　　　　意味　　　　　　　　　　　　　内容
pred_std　　　　　モデルの不確実性　　　　　　　モデルが出力する予測の標準偏差（予測の不確実性）

pred_error　　　　実際の予測精度　　　　　　　　|predicted - ground_truth|（実際の予測誤差）

leave_one_out　　 その点の構造的重要性　　　 　　その点を除外したときのMAE変化量（除外影響度）


■実行手順
個別実行
以下の5ステップを順番に実行することで、重要度マップを生成する。

ステップ1: グラフデータを作成
マリオット盲点情報を考慮したグラフ構造を構築する。
bashpython create_graph_by_eye_pattern.py
ステップ2：データ検証（オプション）
作成したグラフデータの整合性を検証・修正する。
bashpython validate_and_fix_graphs.py
ステップ3: モデル訓練
GNNモデルを訓練する。
bashpython train_by_eye_pattern_revised.py
ステップ4: 重要度マップ計算
訓練済みモデルを用いて、各計測点の重要度スコアを算出する。
bashpython compute_importance_by_eye_pattern.py
ステップ5: 可視化
重要度マップを視覚的に出力する。
bashpython visualize_importance_by_eye_pattern.py


一括実行
上記ステップ1〜5をまとめて実行する場合：
python run_importance_analysis_by_eye_pattern.py


