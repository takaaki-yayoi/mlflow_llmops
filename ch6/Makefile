.PHONY: install register version alias eval optimize-meta optimize-gepa deploy model-config structured all clean help

help:
	@echo "第6章: Prompt Registry - プロンプトエンジニアリングの体系化"
	@echo ""
	@echo "セットアップ:"
	@echo "  make install          依存関係をインストール"
	@echo ""
	@echo "基本フロー (01-04, OPENAI_API_KEYは04から必要):"
	@echo "  make register         6.2: プロンプトの登録"
	@echo "  make version          6.2: バージョン更新(改善版プロンプト)"
	@echo "  make alias            6.2: エイリアスによるライフサイクル管理"
	@echo "  make eval             6.3: 改善したプロンプトで評価 (OPENAI_API_KEY必要)"
	@echo ""
	@echo "応用 (OPENAI_API_KEY必要):"
	@echo "  make optimize-meta    6.5: MetaPromptによる構造改善"
	@echo "  make optimize-gepa    6.5: GEPAによる反復最適化 (コスト注意)"
	@echo "  make deploy           6.6: 段階的デプロイとロールバック"
	@echo "  make model-config     6.7: モデルパラメータの紐付け"
	@echo "  make structured       6.7: 構造化出力 (OPENAI_API_KEY必要)"
	@echo ""
	@echo "一括実行:"
	@echo "  make all              基本フロー + deploy + model-config + structured"
	@echo ""
	@echo "その他:"
	@echo "  make clean            MLflowデータを削除"

install:
	uv sync

# 6.2: プロンプトの登録
register:
	uv run python prompts/01_register_prompt.py

# 6.2: バージョン更新
version:
	uv run python prompts/02_version_update.py

# 6.2: エイリアス管理
alias:
	uv run python prompts/03_alias_management.py

# 6.3: プロンプトの評価
eval:
	uv run python prompts/04_evaluate_prompt.py

# 6.5: MetaPromptによる最適化
optimize-meta:
	uv run python prompts/05_optimize_metaprompt.py

# 6.5: GEPAによる最適化
optimize-gepa:
	uv run python prompts/06_optimize_gepa.py

# 6.6: 段階的デプロイとロールバック
deploy:
	uv run python prompts/07_deploy_lifecycle.py

# 6.7: モデルパラメータの紐付け
model-config:
	uv run python prompts/08_model_config.py

# 6.7: 構造化出力
structured:
	uv run python prompts/09_structured_output.py

# 一括実行 (optimize-meta/optimize-gepaは除外: コストが高いため)
all: register version alias eval deploy model-config structured
	@echo "完了。MLflow UI (http://localhost:5000) のPromptsタブで結果を確認してください。"

clean:
	rm -rf mlflow.db mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
