.PHONY: install ingest vibe-check test-standard test-custom register eval sim eval-all cli clean help

help:
	@echo "第5章: 改善サイクルを加速する - 評価の仕組み"
	@echo ""
	@echo "セットアップ:"
	@echo "  make install        依存関係をインストール"
	@echo "  make ingest         ドキュメントをベクトルストアに取り込み"
	@echo ""
	@echo "原稿順に実行:"
	@echo "  make vibe-check     5.2: テスト質問を実行してトレースを確認"
	@echo "  make test-standard  5.4.5: 標準スコアラーの個別テスト"
	@echo "  make test-custom    5.4.6: カスタムスコアラーの個別テスト"
	@echo "  make register       5.4.7: スコアラーをMLflowに登録"
	@echo "  make eval           5.5-5.6: 自動評価の実行 (メインスクリプト)"
	@echo "  make sim            5.7: [応用] 会話シミュレーション"
	@echo ""
	@echo "一括実行:"
	@echo "  make eval-all       vibe-check → eval を順番に実行"
	@echo ""
	@echo "その他:"
	@echo "  make cli            CLIエージェントを起動"
	@echo "  make clean          生成されたファイルを削除"

install:
	uv sync

ingest:
	uv run python scripts/web_ingest.py

cli:
	uv run python -m cli.main

# 5.2: Vibe Check
vibe-check:
	uv run python evaluation/01_vibe_check.py

# 5.4.5: 標準スコアラーのテスト
test-standard:
	uv run python evaluation/02_standard_scorers.py

# 5.4.6: カスタムスコアラーのテスト
test-custom:
	uv run python evaluation/03_custom_scorers.py

# 5.4.7: スコアラーの登録
register:
	uv run python evaluation/04_register_scorers.py

# 5.5-5.6: 自動評価の実行
eval:
	uv run python evaluation/05_run_evaluation.py

# 5.7: 会話シミュレーション
sim:
	uv run python evaluation/06_conversation_sim.py

# 一括実行
eval-all: vibe-check eval
	@echo "評価完了。MLflow UI (http://localhost:5000) で結果を確認してください。"

clean:
	rm -rf data/milvus.db
	rm -rf .scrapy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
