.PHONY: help setup tracing cost feedback eval otel cost-builtin otel-semconv all

# MLflow 3.16以降、import mlflow のたびにコーディングエージェント向けの案内
# (mlflow.agent.hint) がログに出るため、サンプル実行時は抑制する
export MLFLOW_DISABLE_AGENT_HINT := 1

help:
	@echo "ch8 監視と運用"
	@echo ""
	@echo "使用方法:"
	@echo "  make setup    依存関係のインストールと.envの初期化"
	@echo "  make tracing  8.1: トレーシングの設定とメタデータ追加"
	@echo "  make cost     8.2: トークン使用量とコストの可視化"
	@echo "  make feedback 8.3: フィードバック収集"
	@echo "  make eval     8.3: 品質評価パイプライン"
	@echo "  make otel     8.5: OpenTelemetry設定確認"
	@echo "  make all      全ステップ実行 (tracing cost feedback eval)"
	@echo ""
	@echo "本書以降の新機能:"
	@echo "  make cost-builtin  組み込みのコスト trace.info.cost (MLflow 3.10以降)"
	@echo "  make otel-semconv  OTel GenAIセマンティック規約でのエクスポート (MLflow 3.11以降)"

setup:
	uv sync
	cp -n .env.template .env 2>/dev/null || true
	@echo "✅ セットアップ完了。.envファイルにAPIキーを設定してください。"

# 8.1: トレーシングの設定とメタデータ追加
tracing:
	uv run python monitoring/01_tracing_setup.py

# 8.2: トークン使用量とコストの可視化
cost:
	uv run python monitoring/02_token_and_cost.py

# 8.3: フィードバック収集
feedback:
	uv run python monitoring/03_feedback.py

# 8.3: 品質評価パイプライン
eval:
	uv run python monitoring/04_evaluation.py

# 8.5: OpenTelemetry設定確認
otel:
	uv run python monitoring/05_otel_export.py

# 本書以降の新機能: 組み込みのコスト (MLflow 3.10以降)
cost-builtin:
	uv run python monitoring/06_builtin_cost.py

# 本書以降の新機能: OTel GenAIセマンティック規約でのエクスポート (MLflow 3.11以降)
otel-semconv:
	uv run python monitoring/07_otel_genai_semconv.py

# 全ステップ実行
all: tracing cost feedback eval
	@echo "✅ 全ステップ完了。MLflow UI (http://localhost:5000) で結果を確認してください。"
