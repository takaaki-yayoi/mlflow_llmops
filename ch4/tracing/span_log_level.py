"""第4章 補足: スパンのログレベル (MLflow 3.13以降)。

エージェントのトレースには、LLM 呼び出しやツール呼び出しのほかに、LangGraph の
内部処理などの細かいスパンが大量に並びます。MLflow 3.13 で、スパンに Python の
logging と同じ考え方のログレベル (DEBUG / INFO / WARNING / ERROR / CRITICAL) が
付くようになりました。MLflow UI のトレース画面で「Minimum log level」を INFO にすると、
DEBUG のスパンが隠れ、見たいスパンだけが残ります。

ログレベルの決まり方:
- 何も指定しない場合は、スパンの種類から決まる。LLM / CHAT_MODEL / TOOL / RETRIEVER /
  AGENT / EMBEDDING は INFO、それ以外 (CHAIN や種類なし) は DEBUG
- @mlflow.trace(log_level=...) や span.set_log_level(...) で明示的に指定できる
- スパンの中で例外が起きると ERROR になる

このスクリプトは、第4章の QA エージェントを自前のスパンで包んで2回実行し、
記録されたスパンとログレベルを一覧表示します。

実行: make log-level
前提: MLflow Tracking Server が起動し、make ingest でドキュメントを取り込み済みであること

本書の対象版 (MLflow 3.9) にはログレベルがないため、3.13 以降で動きます。
"""
import re

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.entities import SpanType

from agents.langgraph.agent import LangGraphAgent
from agents.thread import Thread

agent = LangGraphAgent()


@mlflow.trace(log_level="DEBUG")
def normalize_question(question: str) -> str:
    """前後の空白を取るだけの前処理。普段は見なくてよいので DEBUG にする。"""
    return question.strip()


@mlflow.trace
def validate_question(question: str) -> None:
    """空の質問を弾く。例外が起きるとこのスパンは自動で ERROR になる。"""
    if not question:
        raise ValueError("質問が空です")


@mlflow.trace(log_level="INFO")
def check_citation(answer: str) -> bool:
    """回答に URL が含まれるかを確認する。含まれないときは WARNING に上げる。"""
    has_url = bool(re.search(r"https?://", answer))
    if not has_url:
        mlflow.get_current_active_span().set_log_level("WARNING")
    return has_url


@mlflow.trace(name="qa_request", span_type=SpanType.AGENT)
def answer_question(question: str) -> str:
    question = normalize_question(question)
    validate_question(question)
    answer = agent.process_query(question, Thread())
    check_citation(answer)
    return answer


def print_spans(trace_id: str) -> None:
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    for span in trace.data.spans:
        level = span.log_level.name if span.log_level else "-"
        print(f"  {level:<8} {span.span_type or '':<11} {span.name}")


print("\n=== 1. 通常の質問 ===")
answer_question("  MLflowのトレースとは何ですか？  ")
print_spans(mlflow.get_last_active_trace_id())

print("\n=== 2. 空の質問 (validate_question で例外) ===")
try:
    answer_question("   ")
except ValueError as e:
    print(f"例外: {e}")
print_spans(mlflow.get_last_active_trace_id())

print("\nMLflow UI (http://localhost:5000) のトレース画面で「Minimum log level」を")
print("INFO に切り替えると、DEBUG のスパンが非表示になります。")
