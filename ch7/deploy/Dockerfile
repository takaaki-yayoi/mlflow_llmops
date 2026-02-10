# QAエージェント サービング用Dockerfile（7.5.2節）
#
# 使用方法:
#   docker build -t qa-agent-server -f deploy/Dockerfile .
#   docker run -p 5005:8080 \
#       -e OPENAI_API_KEY=$OPENAI_API_KEY \
#       -e EXA_API_KEY=$EXA_API_KEY \
#       -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
#       qa-agent-server

FROM python:3.11-slim

WORKDIR /app

# システム依存関係
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY pyproject.toml .
RUN pip install --no-cache-dir "mlflow[genai]>=3.9.0" \
    "langgraph>=1.0.0" \
    "langchain>=1.0.0" \
    "langchain-openai>=1.0.0" \
    "langchain-community>=0.3.0" \
    "langchain-milvus>=0.2.0" \
    "milvus-lite>=2.4.0" \
    "exa-py>=1.0.0" \
    "uvicorn>=0.30.0" \
    "fastapi>=0.110.0" \
    "python-dotenv>=1.0.0"

# アプリケーションコード
# 第4章のagentsパッケージと第7章のservingパッケージの両方をコピー
COPY agents/ agents/
COPY serving/ serving/
COPY data/ data/

# 環境変数のデフォルト値
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true
ENV OTEL_SERVICE_NAME=qa-agent-production

EXPOSE 8080

# Agent Serverを起動
CMD ["python", "serving/start_server.py", "--port", "8080", "--workers", "4"]
