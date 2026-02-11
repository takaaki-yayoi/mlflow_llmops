.PHONY: cli install ingest clean help

help:
	@echo "MLflow Sample Agent"
	@echo ""
	@echo "使用方法:"
	@echo "  make install  依存関係をインストールする"
	@echo "  make ingest   データ取り込みスクリプトを実行する"
	@echo "  make cli      CLIを起動する"
	@echo "  make clean    生成されたファイルを削除する"

install:
	uv sync

cli:
	uv run python -m cli.main

ingest:
	uv run python scripts/web_ingest.py

clean:
	rm -rf data/milvus.db
	rm -rf .scrapy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
