"""
第8章 - 8.3 フィードバック収集

トレースに対してフィードバックを記録し、検索する方法を確認します。

実行方法:
  make feedback
  または
  uv run python monitoring/03_feedback.py
"""

from dotenv import load_dotenv

load_dotenv()

import mlflow
from openai import OpenAI
from mlflow.entities import AssessmentSource, AssessmentSourceType

mlflow.set_experiment("ch8-monitoring-quickstart")
mlflow.openai.autolog()

client = OpenAI()

# === 1. LLM呼び出し(フィードバック対象のトレースを生成) ===
print("=== フィードバック対象のトレース生成 ===\n")

questions_and_feedback = [
    {
        "question": "MLflowでモデルをデプロイする方法を教えてください。",
        "thumbs_up": True,
        "rating": 5,
        "comment": "正確で分かりやすい回答でした",
    },
    {
        "question": "MLflowとKubeflowの違いは?",
        "thumbs_up": True,
        "rating": 3,
        "comment": "概要は良いが、もう少し具体例が欲しい",
    },
    {
        "question": "量子コンピューティングについて教えてください。",
        "thumbs_up": False,
        "rating": 1,
        "comment": "MLflowとは無関係な質問に回答してしまっている",
    },
]

for item in questions_and_feedback:
    # LLM呼び出し
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたはMLflowの専門家です。"},
            {"role": "user", "content": item["question"]},
        ],
    )

    trace_id = mlflow.get_last_active_trace_id()
    answer = response.choices[0].message.content
    print(f"Q: {item['question']}")
    print(f"A: {answer[:80]}...")

    # === 2. フィードバックを記録 ===
    # サムズアップ/ダウン
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_feedback",
        value=item["thumbs_up"],
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="demo-user-1",
        ),
        rationale=item["comment"],
    )

    # 評価スコア (1-5)
    mlflow.log_feedback(
        trace_id=trace_id,
        name="rating",
        value=item["rating"],
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="demo-user-1",
        ),
    )
    thumbs = "\U0001f44d" if item["thumbs_up"] else "\U0001f44e"
    print(f"   -> feedback: {thumbs}, rating: {item['rating']}/5")
    print()

# === 3. フィードバック付きトレースの確認 ===
print("=== フィードバック確認 ===\n")

experiment = mlflow.get_experiment_by_name("ch8-monitoring-quickstart")
traces = mlflow.search_traces(
    experiment_ids=[experiment.experiment_id],
    max_results=10,
)

for _, row in traces.head(3).iterrows():
    trace = mlflow.get_trace(trace_id=row["trace_id"])
    assessments = trace.search_assessments(type="feedback")
    if assessments:
        print(f"Trace: {row['trace_id'][:16]}...")
        for a in assessments:
            print(f"  {a.name}: {a.feedback.value} (by {a.source.source_id})")
        print()

print(
    "\u2705 MLflow UIの各トレースで [Assessments] タブからフィードバックを確認できます。"
)
