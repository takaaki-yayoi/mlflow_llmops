.PHONY: help setup tracing cost feedback eval otel all

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

# 全ステップ実行
all: tracing cost feedback eval
	@echo "✅ 全ステップ完了。MLflow UI (http://localhost:5000) で結果を確認してください。"
