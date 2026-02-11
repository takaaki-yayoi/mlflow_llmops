.PHONY: setup tracing cost feedback eval otel all

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

# 8.4: 品質評価パイプライン
eval:
	uv run python monitoring/04_evaluation.py

# 8.5: OpenTelemetry設定確認
otel:
	uv run python monitoring/05_otel_export.py

# 全ステップ実行
all: tracing cost feedback eval
	@echo "✅ 全ステップ完了。MLflow UI (http://localhost:5000) で結果を確認してください。"
