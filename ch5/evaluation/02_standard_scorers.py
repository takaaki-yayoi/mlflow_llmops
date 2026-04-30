"""5.4.4節: 標準の評価指標(ToolCallCorrectness, Correctness)を個別にテストする。

既存のトレースを使って標準スコアラーを試すデモスクリプト。
01_vibe_check.pyで生成されたトレースを自動取得して評価します。

実行: make test-standard
前提: make vibe-check でトレースが生成済みであること

# 本書本文との差分

本書では「MLflow UIで質問1のトレースIDをコピーして mlflow.get_trace() で取得する」
流れで説明していますが、本スクリプトでは get_latest_traces() で最新のトレースを
自動取得します。そのため:

- 本書リスト5.2 の判定理由 "Missing: {'doc_search'}; Unexpected: {'web_search'}" は、
  質問1のトレースで web_search が使われた場合の例であり、本スクリプトの実行結果は
  使用されるトレースに応じて変動します。
- 本書リスト5.3 の expected_response は LangGraph トークン使用量に関する長文ですが、
  本スクリプトではフレームワーク対応に関する短文を使用しているため、判定結果が
  本書とは異なります。

本書と同じ出力を再現したい場合は、コード内の `traces[0]` を該当質問のトレースIDに
差し替えてください。詳細は ch5/CHAPTER_NOTES.md を参照してください。
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.genai.scorers import Correctness, ToolCallCorrectness

# MLflow接続設定（エージェントを使わずスコアラーのみテストするため明示的に設定）
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")


def get_latest_traces(experiment_name: str = "MLflow QAエージェント", max_results: int = 5):
    """最新のトレースを取得するヘルパー関数。

    Args:
        experiment_name: 実験名
        max_results: 取得する最大件数

    Returns:
        トレースのリスト
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        max_results=max_results,
        order_by=["timestamp DESC"],
        return_type="list",
    )
    return traces


def test_tool_call_correctness(trace):
    """ToolCallCorrectnessスコアラーをテストする。"""
    print("--- ToolCallCorrectness ---")

    scorer = ToolCallCorrectness()
    expected_tools = [{"name": "doc_search"}]

    result = scorer(
        trace=trace,
        expectations={"expected_tool_calls": expected_tools},
    )

    print(f"  name: {result.name}")
    print(f"  value: {result.value}")
    if hasattr(result, "rationale") and result.rationale:
        print(f"  rationale: {result.rationale}")
    print()


def test_correctness(trace):
    """Correctnessスコアラーをテストする。"""
    print("--- Correctness ---")

    scorer = Correctness()
    expected_response = (
        "MLflowトレーシングは、LangChain、LangGraph、LlamaIndex、"
        "OpenAI SDK、Anthropic SDK、AWS Bedrock SDKなどの"
        "主要なフレームワークに対応しています。"
    )

    result = scorer(
        trace=trace,
        expectations={"expected_response": expected_response},
    )

    print(f"  name: {result.name}")
    print(f"  value: {result.value}")
    if hasattr(result, "rationale") and result.rationale:
        print(f"  rationale: {result.rationale}")
    print()


def main():
    """標準スコアラーの個別テストを実行する。"""
    print("=" * 60)
    print("5.4.4節: 標準スコアラーの個別テスト")
    print("=" * 60)

    try:
        traces = get_latest_traces(max_results=3)
    except Exception as e:
        print(f"\nMLflow Tracking Serverに接続できません: {e}")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)

    if not traces:
        print("\nトレースが見つかりません。")
        print("  先に 'make vibe-check' を実行してトレースを生成してください。")
        sys.exit(1)

    # 最新のトレースを使用
    trace = traces[0]
    print(f"\nトレースID: {trace.info.trace_id}")
    print(f"トレース数: {len(traces)}件取得\n")

    test_tool_call_correctness(trace)
    test_correctness(trace)

    print("=" * 60)
    print("標準スコアラーのテスト完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
