"""
第8章 - 8.1 本番トレーシングの基本設定

本番環境でのトレーシング設定と、メタデータの追加方法を確認します。

実行方法:
  make tracing
  または
  uv run python monitoring/01_tracing_setup.py
"""

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
@mlflow.trace
def handle_request(message: str, user_id: str, session_id: str) -> str:
    """メタデータを付与したリクエスト処理"""
    # トレースにコンテキスト情報を追加
    mlflow.update_current_trace(
        tags={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            "mlflow.trace.request_id": str(uuid.uuid4()),
            "environment": "development",  # 本番では "production"
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
    answer = handle_request(q, user_id=user_id, session_id=session_id)
    print(f"\nQ: {q}")
    print(f"A: {answer[:100]}...")

# 非同期バッファをフラッシュ
mlflow.flush_trace_async_logging()

# === 4. トレース検索 ===
print("\n=== トレース検索 ===")

# 全トレースを取得
all_traces = mlflow.search_traces(
    experiment_names=["ch8-monitoring-quickstart"],
    max_results=10,
)
print(f"総トレース数: {len(all_traces)}")

# 特定ユーザーのトレースを検索
user1_traces = mlflow.search_traces(
    experiment_names=["ch8-monitoring-quickstart"],
    filter_string="tags.`mlflow.trace.user` = 'user-1'",
)
print(f"user-1のトレース数: {len(user1_traces)}")

print("\n✅ MLflow UI でトレースの詳細を確認してください。")
print("   各トレースに user, session, request_id タグが設定されています。")
