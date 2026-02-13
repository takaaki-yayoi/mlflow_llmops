.PHONY: install register version alias tags model-config system-prompt structured eval optimize all clean help

help:
	@echo "第6章: Prompt Registry - プロンプトエンジニアリングの体系化"
	@echo ""
	@echo "セットアップ:"
	@echo "  make install          依存関係をインストール"
	@echo ""
	@echo "原稿順に実行 (01-05はOPENAI_API_KEY不要):"
	@echo "  make register         6.2.1: プロンプトの登録"
	@echo "  make version          6.2.2: バージョン更新と不変性"
	@echo "  make alias            6.2.3: エイリアス管理"
	@echo "  make tags             6.2.4: タグとメタデータ"
	@echo "  make model-config     6.2.6: モデルパラメータの保存"
	@echo "  make system-prompt    6.4: QAエージェントのシステムプロンプト登録(第7章連携)"
	@echo "  make structured       6.2.7: 構造化出力 (OPENAI_API_KEY必要)"
	@echo "  make eval             6.3.1: プロンプトのオフライン評価 (OPENAI_API_KEY必要)"
	@echo "  make optimize         6.3.4: プロンプト自動最適化 (OPENAI_API_KEY必要, コスト注意)"
	@echo ""
	@echo "一括実行:"
	@echo "  make all              01→07を順番に実行 (08は除外)"
	@echo ""
	@echo "その他:"
	@echo "  make clean            MLflowデータを削除"

install:
	uv sync

# 6.2.1: プロンプトの登録
register:
	uv run python prompts/01_register_prompt.py

# 6.2.2: バージョン更新
version:
	uv run python prompts/02_version_update.py

# 6.2.3: エイリアス管理
alias:
	uv run python prompts/03_alias_management.py

# 6.2.4: タグとメタデータ
tags:
	uv run python prompts/04_tags_metadata.py

# 6.2.6: モデルパラメータの保存
model-config:
	uv run python prompts/05_model_config.py

# 6.4: QAエージェントのシステムプロンプト登録(第7章連携)
system-prompt:
	uv run python prompts/09_register_system_prompt.py

# 6.2.7: 構造化出力
structured:
	uv run python prompts/06_structured_output.py

# 6.3.1: プロンプトのオフライン評価
eval:
	uv run python prompts/07_evaluate_prompt.py

# 6.3.4: プロンプト自動最適化
optimize:
	uv run python prompts/08_optimize_prompt.py

# 一括実行 (optimizeは除外: コストが高いため)
all: register version alias tags model-config system-prompt structured eval
	@echo "完了。MLflow UI (http://localhost:5000) のPromptsタブで結果を確認してください。"

clean:
	rm -rf mlflow.db mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
