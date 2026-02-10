.PHONY: install ingest log-model serve test-request eval gateway clean help

help:
	@echo "MLflow Sample Agent - サービング（第7章）"
	@echo ""
	@echo "使用方法:"
	@echo "  make install       依存関係をインストールする"
	@echo "  make ingest        Milvusにドキュメントを取り込む"
	@echo "  make log-model     QAエージェントをMLflowに記録する"
	@echo "  make serve         Agent Serverを起動する"
	@echo "  make test-request  サービングエンドポイントにテストリクエストを送信する"
	@echo "  make eval          サービング中のエージェントを評価する"
	@echo "  make gateway       AI Gatewayを起動する"
	@echo "  make clean         生成されたファイルを削除する"

install:
	uv sync

ingest:
	uv run python scripts/web_ingest.py

log-model:
	uv run python -m serving.log_model

serve:
	uv run python serving/start_server.py --reload --port 5005

test-request:
	curl -s -X POST http://localhost:5005/invocations \
		-H "Content-Type: application/json" \
		-d '{"input": [{"role": "user", "content": "MLflow Tracingとは何ですか?"}]}' | python3 -m json.tool --no-ensure-ascii

eval:
	uv run python -m serving.eval_serving

gateway:
	set -a && . ./.env && set +a && uv run mlflow gateway start --config-path gateway/gateway_config.yaml --port 5010

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
