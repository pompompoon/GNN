
GNNで視野計測点の視野感度を予測して、importancemapを作成するシステム

■importanceについて

importanceの点数は下記のとおり

・pred_std（モデルが出力する予測の標準偏差）
予測した感度の不確実性（予測値の標準偏差）
予測値の標準偏差が大きい点では不確実性（std）を大きくすることを学習
予測が当たる点では不確実性を小さくすることを学習

・|predicted - ground_truth|（実際の予測誤差）

・Leave-one-out（除外影響度）
その点を除外したときのMAEの変化

上記３つの重ね付き平均和




■下記の順番で実行すると、importancemap作成

ステップ1: グラフデータを作成（マリオット盲点情報）
bashpython create_graph_by_eye_pattern.py
ステップ2: データ検証（オプション）
bashpython validate_and_fix_graphs.py
ステップ3: モデル訓練
bashpython train_by_eye_pattern_revised.py
ステップ4: 重要度マップ計算
bashpython compute_importance_by_eye_pattern.py
ステップ5: 可視化
bashpython visualize_importance_by_eye_pattern.py





■一括でimportancemap作成

run_importance_analysis_by_eye_pattern.py