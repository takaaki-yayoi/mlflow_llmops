# 第8章 監視と運用 - サンプルコード

本ディレクトリには、第8章「監視と運用 - LLMアプリケーションの健全性管理」で解説した内容の実装サンプルが含まれています。

## ディレクトリ構成

```
chapter8_samples/
├── README.md
├── 01_production_tracing/
│   ├── basic_setup.py          # 本番トレーシングの基本設定
│   ├── graceful_shutdown.py    # グレースフルシャットダウン
│   ├── metadata_tagging.py     # メタデータとタグ付け
│   └── sampling_strategies.py  # サンプリング戦略
├── 02_cost_tracking/
│   ├── token_tracking.py       # トークン使用量の追跡
│   ├── cost_calculator.py      # コスト計算クラス
│   └── cost_analysis.sql       # コスト分析SQLクエリ
├── 03_quality_monitoring/
│   ├── feedback_collection.py  # ユーザーフィードバック収集
│   ├── scorer_setup.py         # スコアラーの登録と開始
│   ├── scorer_lifecycle.py     # スコアラーライフサイクル管理
│   └── backfill_scorers.py     # 過去トレースへのバックフィル
├── 04_alerting/
│   ├── alert_config.py         # アラート設定
│   ├── notification_handlers.py # 通知ハンドラー(Slack/PagerDuty)
│   └── rollback.py             # ロールバック戦略
└── 05_opentelemetry/
    ├── otlp_setup.py           # OTLP設定
    └── otel-collector-config.yaml # OpenTelemetry Collector設定
```

## 前提条件

### 必須パッケージ

**開発環境:**
```bash
pip install mlflow>=3.0.0
```

**本番環境(軽量SDK):**
```bash
pip install mlflow-tracing
```

> **注意:** `mlflow`と`mlflow-tracing`は同時にインストールしないでください。

### Databricks環境

以下の機能はDatabricks環境でのみ利用可能です:

- **スコアラーの継続監視**: `Scorer.register()`, `start()`, `stop()`, `update()`
- **トレースアーカイブ**: `enable_databricks_trace_archival()`
- **バックフィル**: `backfill_scorers()`
- **組み込みスコアラー**: `Safety`, `RetrievalRelevance` (Databricks専用)

Databricks固有APIを使用するには `databricks-agents` パッケージが必要です:

```bash
pip install databricks-agents
```

### OSS MLflow環境

OSS MLflow環境では以下の機能が利用可能です:

- **開発時評価**: `mlflow.genai.evaluate()` でスコアラーを実行
- **トレーシング**: `mlflow.trace`, `@mlflow.trace` デコレータ
- **トレース検索**: `mlflow.search_traces()`
- **フィードバック**: `mlflow.log_feedback()`

## 環境変数

以下の環境変数を設定してください:

```bash
# MLflow Tracking Server
export MLFLOW_TRACKING_URI="databricks"  # または自前サーバーのURL

# 非同期ログ記録(本番環境で推奨)
export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING="true"

# サービス識別
export OTEL_SERVICE_NAME="your-service-name"

# OpenTelemetry(オプション)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
```

## クイックスタート

### 1. 本番トレーシングの設定

```python
from chapter8_samples.production_tracing import setup_production_tracing

setup_production_tracing(
    experiment_name="/production/my-llm-app",
    service_name="my-llm-app",
    enable_async=True
)
```

### 2. スコアラーによる品質監視

```python
from chapter8_samples.quality_monitoring import setup_quality_scorers

setup_quality_scorers(
    experiment_name="/production/my-llm-app",
    safety_sample_rate=1.0,
    quality_sample_rate=0.5
)
```

### 3. コスト追跡

```python
from chapter8_samples.cost_tracking import CostCalculator

calculator = CostCalculator()
cost = calculator.calculate(
    model="gpt-4o",
    input_tokens=1000,
    output_tokens=500
)
```

## 注意事項

- 料金情報は2024年12月時点の参考値です。最新の料金は各プロバイダーの公式サイトで確認してください。
- サンプルコードは教育目的であり、本番環境では適切なエラーハンドリングとセキュリティ対策を追加してください。
- Databricks固有の機能を使用する場合は、適切なワークスペース設定が必要です。

## 関連ドキュメント

- [MLflow Tracing Documentation](https://mlflow.org/docs/latest/genai/tracing/)
- [Databricks Production Monitoring](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/production-monitoring)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
