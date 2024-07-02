# normal-method

従来手法から実験するためのリポジトリ
normal-method/
├── data/
│ ├── raw/ # 生データ（未加工データ）
│ └── processed/ # 前処理済みデータ
├── notebooks/ # Jupyter ノートブック
├── src/normal_method # ソースコード
│ ├── init.py
│ ├── data/ # データ処理用スクリプト
│ │ ├── init.py
│ │ └── make_dataset.py
│ ├── features/ # 特徴量エンジニアリング用スクリプト
│ │ ├── init.py
│ │ └── build_features.py
│ ├── models/ # モデル関連スクリプト
│ │ ├── init.py
│ │ ├── train_model.py
│ │ └── predict_model.py
│ ├── visualization/ # 可視化用スクリプト
│ │ ├── init.py
│ │ └── visualize.py
│ └── utils/ # ユーティリティスクリプト
│ ├── init.py
│ └── utils.py
├── tests/ # テストコード
│ ├── init.py
│ └── test_make_dataset.py
├── scripts/ # 実行用スクリプト
│ ├── run_training.py
│ └── run_prediction.py
├── requirements.txt # 依存パッケージ
├── README.md # プロジェクト概要
└── setup.py # パッケージ設定
