"""
第8章 - 8.3/8.4 品質評価パイプライン

蓄積されたトレースに対して、LLM-as-a-Judgeで品質評価を実行します。
5章で開発時に使ったmlflow.genai.evaluate()を、本番トレースに適用する形です。

実行方法:
  make eval
  または
  uv run python monitoring/04_evaluation.py

前提:
  先に make tracing, make cost, make feedback を実行して
  トレースを蓄積しておいてください。
"""

import mlflow
from mlflow.genai.scorers import (
    Safety,
    RelevanceToQuery,
    Guidelines,
)

mlflow.set_experiment("ch8-monitoring-quickstart")

# === 1. 本番トレースの取得 ===
print("=== 本番トレースの取得 ===\n")

traces = mlflow.search_traces(
    experiment_names=["ch8-monitoring-quickstart"],
    max_results=20,
)

print(f"評価対象トレース数: {len(traces)}")

if len(traces) == 0:
    print("⚠️ トレースが見つかりません。先に make tracing を実行してください。")
    exit(1)

# === 2. スコアラーで評価 ===
print("\n=== LLM-as-a-Judge 評価実行 ===\n")

results = mlflow.genai.evaluate(
    data=traces,
    scorers=[
        RelevanceToQuery(),  # 質問との関連性
        Safety(),  # 有害コンテンツチェック
        Guidelines(
            name="helpfulness",
            guidelines="回答はユーザーの質問に対して具体的で実用的な情報を提供している必要があります。",
        ),
    ],
)

# === 3. 結果の確認 ===
print("=== 評価結果サマリー ===\n")
for metric_name, value in results.metrics.items():
    print(f"  {metric_name}: {value:.3f}")

print(f"\n✅ 評価完了。MLflow UI のエクスペリメント画面で詳細を確認できます。")
print("   Evaluation タブに各トレースのスコアが記録されています。")
