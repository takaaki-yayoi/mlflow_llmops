"""第5章 補足: 評価を pytest のリグレッションテストとして実行する (MLflow 3.14以降)。

本書の5.5〜5.6節では、mlflow.genai.evaluate() で評価を実行し、結果を MLflow UI で
確認しました。MLflow 3.14 で追加された @mlflow.test を使うと、同じ評価を pytest の
テストとして書けます。スコアラーが1つでも不合格になるとテストが失敗するので、
プロンプトやモデルを変えたときの品質の後退 (リグレッション) を CI で止められます。

実行: make test-regression
  (または uv run pytest tests -v)

前提:
  - MLflow Tracking Server が http://localhost:5000 で起動していること
  - make ingest でドキュメントが取り込み済みであること

本書の対象版 (MLflow 3.9) には @mlflow.test がないため、このテストは 3.14 以降で動きます。
"""
import mlflow
import pytest
from mlflow.genai.scorers import RelevanceToQuery

from agents.langgraph import LangGraphAgent
from agents.thread import Thread
from evaluation.scorers import has_reference_link


@pytest.fixture(scope="module")
def agent():
    return LangGraphAgent()


def make_predict_fn(agent):
    """05_run_evaluation.py と同じ predict_fn。質問ごとに新しいスレッドで回答させる。"""

    def predict_fn(question: str) -> str:
        return agent.process_query(query=question, thread=Thread())

    return predict_fn


# 品質を落としたくない代表的な質問
#
# 注意: MLflow 3.17.2 時点では、データに expectations (expected_response など) を含めると
# result.passed が常に False になります。期待値の列もスコアラーの結果として判定されて
# しまうためです (mlflow/mlflow#25701、修正PR #26151 は未マージ)。そのため、このテストでは
# expectations を使わず、期待値なしで判定できるスコアラーだけを使っています。
CRITICAL_QUESTIONS = [
    "MLflowとは何ですか？",
    "実験管理はどのように始めれば良いですか？",
]


@mlflow.test
@pytest.mark.parametrize("question", CRITICAL_QUESTIONS, ids=["what_is_mlflow", "start_tracking"])
def test_answer_quality(agent, question):
    """回答が質問に沿っていて、参照リンクを含むこと。"""
    result = mlflow.genai.evaluate(
        data=[{"inputs": {"question": question}}],
        predict_fn=make_predict_fn(agent),
        scorers=[RelevanceToQuery(), has_reference_link],
    )
    # すべてのスコアラーが合格 (yes / True) なら passed が True になる。
    # 不合格のときは、どのスコアラーがなぜ落ちたかが reason に入る
    assert result.passed, result.reason
