"""5.4.5節: カスタムの評価指標を実装してテストする。

ルールベース(@scorer)、Guidelines、make_judgeの3種類のカスタムスコアラーを
個別にテストするデモスクリプト。

実行: make test-custom
前提: make vibe-check でトレースが生成済みであること
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow

from evaluation.scorers import (
    contains_code_block,
    has_reference_link,
    appropriate_katakana,
    katakana_judge,
)

# MLflow接続設定
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")


def get_latest_traces(experiment_name: str = "MLflow QAエージェント", max_results: int = 5):
    """最新のトレースを取得するヘルパー関数。"""
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


def print_result(result, name: str = ""):
    """スコアラーの結果を表示する。"""
    # @scorerデコレータの関数を直接呼び出すと素のPython値が返る
    if not hasattr(result, "name"):
        print(f"  name: {name}")
        print(f"  value: {result}")
        print()
        return
    print(f"  name: {result.name}")
    print(f"  value: {result.value}")
    if hasattr(result, "rationale") and result.rationale:
        rationale = result.rationale
        if len(rationale) > 200:
            rationale = rationale[:200] + "..."
        print(f"  rationale: {rationale}")
    print()


def main():
    """カスタムスコアラーの個別テストを実行する。"""
    print("=" * 60)
    print("5.4.6節: カスタムスコアラーの個別テスト")
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

    trace = traces[0]
    print(f"\nトレースID: {trace.info.trace_id}\n")

    # 1. ルールベース: contains_code_block
    print("--- contains_code_block (@scorerデコレータ) ---")
    # トレースの出力を取得してテスト
    outputs = trace.data.spans[0].outputs
    output_text = str(outputs) if not isinstance(outputs, str) else outputs
    result = contains_code_block(outputs=output_text)
    print_result(result, name="contains_code_block")

    # 2. Guidelinesベース: has_reference_link
    print("--- has_reference_link (Guidelines) ---")
    result = has_reference_link(trace=trace)
    print_result(result)

    # 3. Guidelinesベース: appropriate_katakana
    print("--- appropriate_katakana (Guidelines) ---")
    result = appropriate_katakana(trace=trace)
    print_result(result)

    # 4. make_judgeベース: katakana_judge
    print("--- katakana_judge (make_judge) ---")
    result = katakana_judge(trace=trace)
    print_result(result)

    print("=" * 60)
    print("カスタムスコアラーのテスト完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()
