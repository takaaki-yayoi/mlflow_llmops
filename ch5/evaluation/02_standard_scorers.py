"""5.4.4節: 標準の評価指標(ToolCallCorrectness, Correctness)を個別にテストする。

既存のトレースを使って標準スコアラーを試すデモスクリプト。
01_vibe_check.pyで生成されたトレースを自動取得して評価します。

実行: make test-standard
前提: make vibe-check でトレースが生成済みであること

# 本書本文との差分

本書では「MLflow UIで質問2のトレースIDをコピーして mlflow.get_trace() で取得する」
流れで説明していますが、本スクリプトではUI操作なしで実行できるように、
get_latest_traces() で取得した最新トレースの中から、本書と同じ
質問2 (LangGraphエージェントのトークン使用量) のトレースを質問文で検索して採点します。

本書リスト5.2 の判定理由 "Missing: {'doc_search'}; Unexpected: {'web_search'}" は、
質問2のトレースで web_search が使われた場合の例です。エージェントがどのツールを
選ぶかは実行ごとに変動するため、doc_search が使われた場合は yes と判定されます。

なお、01_vibe_check.py の質問の並び順は本書 リスト5.1 と同じで、
質問1が「実験管理」、質問2が「LangGraphエージェントのトークン使用量」、
質問3が「MLflowトレーシングの対応フレームワーク」です。
詳細は ch5/CHAPTER_NOTES.md を参照してください。
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.genai.scorers import Correctness, ToolCallCorrectness

# MLflow接続設定（エージェントを使わずスコアラーのみテストするため明示的に設定）
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")

# 本書 リスト5.2/5.3 が採点対象にしている質問 (01_vibe_check.py の質問2)。
# 最新トレースを無条件に使うと質問3のトレースが選ばれ、下の expected_response
# (質問2用の正解データ) と噛み合わずに Correctness が必ず no になってしまう。
TARGET_QUESTION = "LangGraphエージェントのトークン使用量"


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


def extract_question(trace) -> str:
    """トレースからユーザーの質問文を取り出す。"""
    preview = getattr(trace.info, "request_preview", None)
    if preview:
        return str(preview)
    spans = getattr(trace.data, "spans", None)
    if spans:
        return str(spans[0].inputs)
    return ""


def select_target_trace(traces, target: str = TARGET_QUESTION):
    """本書と同じ質問のトレースを選ぶ。見つからなければ最新トレースを返す。

    Args:
        traces: 新しい順に並んだトレースのリスト
        target: 探したい質問文に含まれるキーワード

    Returns:
        (トレース, 本書と同じ質問のトレースが見つかったか) のタプル
    """
    for trace in traces:
        if target in extract_question(trace):
            return trace, True
    return traces[0], False


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
    # 本書 リスト5.3 と同じ正解データ。質問2 (LangGraphエージェントのトークン使用量) のトレースで使う前提。
    # デフォルトではMLflowはOpenAIのGPTモデルを使用します。他のモデルを使用する場合は、
    # modelパラメータを<provider>:/<model_name>の形式で指定してください。
    # 例: correctness = Correctness(model="anthropic:/claude-sonnet-4-20250514")
    # 例: correctness = Correctness(model="google:/gemini-2.0-flash")
    expected_response = (
        "LangGraphエージェントのトークン使用量をMLflowで可視化するには、"
        "MLflowのトレーシング機能が利用できます。`mlflow.langchain.autolog()`"
        "APIをコードに追加することで、エージェントを実行するたびにトレースが生成され、"
        "呼び出しごとのトークンの使用量が記録されます。"
        "ダッシュボードで使用量の推移をグラフで確認することも可能です。"
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
        traces = get_latest_traces(max_results=5)
    except Exception as e:
        print(f"\nMLflow Tracking Serverに接続できません: {e}")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)

    if not traces:
        print("\nトレースが見つかりません。")
        print("  先に 'make vibe-check' を実行してトレースを生成してください。")
        sys.exit(1)

    # 本書 リスト5.2/5.3 と同じ質問2のトレースを採点対象にする
    trace, found = select_target_trace(traces)
    print(f"\nトレース数: {len(traces)}件取得")
    print(f"トレースID: {trace.info.trace_id}")

    if found:
        print(f"採点対象: 質問2「{TARGET_QUESTION}...」のトレース (本書 リスト5.2/5.3 と同じ)\n")
    else:
        print("採点対象: 最新のトレース")
        print(f"  ※「{TARGET_QUESTION}」を含むトレースが見つからなかったため、")
        print("    最新トレースで採点します。下の expected_response は質問2用の正解データなので、")
        print("    Correctness が no になることがあります。")
        print("    本書と同じ結果を見たい場合は 'make vibe-check' を実行し直してください。\n")

    test_tool_call_correctness(trace)
    test_correctness(trace)

    print("=" * 60)
    print("標準スコアラーのテスト完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
