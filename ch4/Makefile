.PHONY: cli install ingest log-level clean help

# MLflow 3.16以降、import mlflow のたびにコーディングエージェント向けの案内
# (mlflow.agent.hint) がログに出るため、サンプル実行時は抑制する
export MLFLOW_DISABLE_AGENT_HINT := 1

help:
	@echo "MLflow Sample Agent"
	@echo ""
	@echo "使用方法:"
	@echo "  make install  依存関係をインストールする"
	@echo "  make ingest   データ取り込みスクリプトを実行する"
	@echo "  make cli      CLIを起動する"
	@echo "  make clean    生成されたファイルを削除する"
	@echo ""
	@echo "本書以降の新機能:"
	@echo "  make log-level  スパンのログレベル (MLflow 3.13以降)"

install:
	uv sync

cli:
	uv run python -m cli.main

ingest:
	uv run python scripts/web_ingest.py

# 本書以降の新機能: スパンのログレベル (MLflow 3.13以降)
log-level:
	uv run python -m tracing.span_log_level

clean:
	rm -rf data/milvus.db
	rm -rf .scrapy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
