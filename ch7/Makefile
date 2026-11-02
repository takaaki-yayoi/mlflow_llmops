.PHONY: install ingest serve test-request eval log-model test-gateway gateway-legacy clean help

# MLflow 3.16以降、import mlflow のたびにコーディングエージェント向けの案内
# (mlflow.agent.hint) がログに出るため、サンプル実行時は抑制する
export MLFLOW_DISABLE_AGENT_HINT := 1

help:
	@echo "MLflow Sample Agent - サービング（第7章）"
	@echo ""
	@echo "使用方法:"
	@echo "  make install        依存関係をインストールする"
	@echo "  make ingest         Milvusにドキュメントを取り込む"
	@echo "  make serve          Agent Serverを起動する（7.2節）"
	@echo "  make test-request   サービングエンドポイントにテストリクエストを送信する（7.2節）"
	@echo "  make eval           サービング環境のエージェントを評価する（7.3節、Agent Server停止後に実行）"
	@echo "  make log-model      QAエージェントをMLflowに記録する（任意、Note参照）"
	@echo "  make test-gateway   AI Gateway経由でテストリクエストを送信する（7.4節 新方式）"
	@echo "  make gateway-legacy AI Gatewayを起動する（7.4節 Legacy方式）"
	@echo "  make clean          生成されたファイルを削除する"

install:
	uv sync

ingest:
	uv run python scripts/web_ingest.py

serve:
	uv run python serving/start_server.py --reload --port 5005

test-request:
	curl -s -X POST http://localhost:5005/invocations \
		-H "Content-Type: application/json" \
		-d '{"input": [{"role": "user", "content": "MLflow Tracingとは何ですか?"}]}' | python3 -m json.tool --no-ensure-ascii

# Agent Serverと同じ@invoke関数をin-processで実行して評価する
# Milvusのファイルロック競合を避けるため、Agent Serverを停止してから実行すること
eval:
	@echo "⚠️  Agent Serverが起動中の場合はCtrl+Cで停止してから実行してください（Milvusロック競合）"
	uv run python -m serving.eval_serving

# Model Registry方式を使う場合（任意）
# Databricks Model Servingへのデプロイ時などに利用（Note・第10章参照）
log-model:
	uv run python -m serving.log_model

# 新AI Gateway（Tracking Server統合型）のテスト
# 事前にMLflow UI (http://localhost:5000/#/gateway) でエンドポイントを作成してください
test-gateway:
	@echo "AI Gateway経由でテストリクエストを送信します..."
	@echo "※ 事前にMLflow UI (http://localhost:5000/#/gateway) で"
	@echo "  qa-agent-llmエンドポイントを作成してください"
	@echo ""
	curl -s -X POST http://localhost:5000/gateway/qa-agent-llm/mlflow/invocations \
		-H "Content-Type: application/json" \
		-d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}' | python3 -m json.tool --no-ensure-ascii

# Legacy方式: 別プロセスとしてAI Gatewayを起動（YAML設定ファイル使用）
# 新方式（Tracking Server統合型）ではこのターゲットは不要です
gateway-legacy:
	set -a && . ./.env && set +a && uv run mlflow gateway start --config-path gateway/gateway_config.yaml --port 5010

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
