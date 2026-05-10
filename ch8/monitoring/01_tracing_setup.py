"""
第8章 - 8.1 本番トレーシングの基本設定

本番環境でのトレーシング設定と、メタデータの追加方法を確認します。

実行方法:
  make tracing
  または
  uv run python monitoring/01_tracing_setup.py
"""

from dotenv import load_dotenv

load_dotenv()

import mlflow
import os
import uuid
from openai import OpenAI

# === 1. 本番向け設定 ===
# 非同期ログ記録を有効化(本番環境推奨)
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
os.environ["OTEL_SERVICE_NAME"] = "qa-agent"

# エクスペリメントの設定
mlflow.set_experiment("ch8-monitoring-quickstart")

# OpenAI自動トレーシングを有効化
mlflow.openai.autolog()

client = OpenAI()


# === 2. メタデータ付きのLLM呼び出し ===
# 本書 リスト8.7
@mlflow.trace
def handle_chat_request(message: str, user_id: str, session_id: str) -> str:
    """メタデータを付与したリクエスト処理"""
    # トレースにコンテキスト情報を追加
    mlflow.update_current_trace(
        tags={
            # 標準タグ
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            "mlflow.trace.request_id": str(uuid.uuid4()),
            # カスタムタグ
            "environment": "production",
            "service.version": os.getenv("SERVICE_VERSION", "unknown"),
            "deployment.region": os.getenv("DEPLOYMENT_REGION", "unknown"),
        }
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "あなたはMLflowの専門家です。簡潔に回答してください。",
            },
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content


# === 3. 複数リクエストを実行 ===
questions = [
    "MLflow Tracingとは何ですか?",
    "MLflowでモデルをサーブする方法は?",
    "MLflow Evaluateの主な機能は?",
]

print("=== トレース生成 ===")
session_id = str(uuid.uuid4())
for i, q in enumerate(questions):
    user_id = f"user-{(i % 2) + 1}"
    answer = handle_chat_request(q, user_id=user_id, session_id=session_id)
    print(f"\nQ: {q}")
    print(f"A: {answer[:100]}...")

# 非同期バッファをフラッシュ
mlflow.flush_trace_async_logging()

# === 4. トレース検索 ===
# 本書 リスト8.8
print("\n=== トレース検索 ===")

# 全トレースを取得（set_experimentで設定済みのアクティブエクスペリメントを検索）
all_traces = mlflow.search_traces(max_results=10)
print(f"総トレース数: {len(all_traces)}")

# 特定ユーザーのトレースを検索
user_traces = mlflow.search_traces(
    filter_string="tags.`mlflow.trace.user` = 'user-1'",
    max_results=100,
)
print(f"user-1のトレース数: {len(user_traces)}")

# 特定セッションのトレースを時系列で取得
session_traces = mlflow.search_traces(
    filter_string=f"tags.`mlflow.trace.session` = '{session_id}'",
    order_by=["timestamp_ms ASC"],
)
print(f"このセッションのトレース数: {len(session_traces)}")

print("\n✅ MLflow UI でトレースの詳細を確認してください。")
print("   各トレースに user, session, request_id タグが設定されています。")
