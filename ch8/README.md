# 第8章 監視と運用 サンプルコード

第8章「監視と運用 - LLMアプリケーションの健全性管理」のサンプルコードです。

## セットアップ

### 前提条件

- Python 3.11以上
- uv (パッケージマネージャー)
- OpenAI APIキー

### 環境変数の設定

リポジトリルートで設定済みの `.env` をコピーする方法（推奨）：

```
cp ../.env .env
```

または、章固有のテンプレートからコピーすることもできます：

```
cp .env.template .env
```

`.env` ファイルにAPIキーを設定してください。

### インストール

```
make setup
```

### MLflow Tracking Serverの起動

```
uv run mlflow server --host 0.0.0.0 --port 5000
```

## 実行

各スクリプトは独立して実行できますが、順番に実行することを推奨します。

### 一括実行

```
make all
```

### 個別実行

| コマンド | 対応セクション | 内容 |
|---------|--------------|------|
| `make tracing` | 8.1 | 本番トレーシング設定、メタデータ追加、トレース検索 |
| `make cost` | 8.2 | トークン使用量の自動追跡、コスト計算・集計 |
| `make feedback` | 8.3 | ユーザーフィードバックの記録と検索 |
| `make eval` | 8.3-8.4 | LLM-as-a-Judgeによる品質評価パイプライン |
| `make otel` | 8.5 | OpenTelemetry連携の設定確認 |

## ファイル構成

```
ch8/
├── monitoring/
│   ├── 01_tracing_setup.py    # 本番トレーシングの設定
│   ├── 02_token_and_cost.py   # トークン使用量・コスト可視化
│   ├── 03_feedback.py         # フィードバック収集
│   ├── 04_evaluation.py       # 品質評価パイプライン
│   ├── 05_otel_export.py      # OpenTelemetry設定
│   └── cost_calculator.py     # コスト計算ユーティリティ
├── agents/                    # QAエージェント(第4章と同一)
└── scripts/                   # データ取り込み(第4章と同一)
```

## 本文との対応

- `01_tracing_setup.py`: 8.1.4〜8.1.7 の設定を一通り確認
- `02_token_and_cost.py`: 8.2.2〜8.2.3 のトークン追跡とコスト計算を実践
- `03_feedback.py`: 8.3.3 のフィードバック収集を実践
- `04_evaluation.py`: 8.3.4〜8.3.5 の品質評価パイプラインを実践
- `05_otel_export.py`: 8.5.3 のOTLP設定方法を確認
- `cost_calculator.py`: 8.2.3 で「堅牢な実装を推奨」と記載のユーティリティ

## Databricksでの実行

Databricksノートブックで実行する場合は、各スクリプトの内容をセルにコピーして実行してください。
以下の点が異なります。

- `mlflow.set_tracking_uri()` は不要(自動設定)
- `mlflow.set_experiment()` のパスは `/Users/<email>/ch8-monitoring-quickstart` の形式
- トレースアーカイブが利用可能(`mlflow.enable_databricks_trace_archival()`)
- リアルタイムスコアラーが利用可能(8.3.4節参照)
