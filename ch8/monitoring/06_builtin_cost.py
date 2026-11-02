"""
第8章 補足 - トレースに記録される組み込みのコスト (MLflow 3.10以降)

8.2節 (02_token_and_cost.py) では、トレースからトークン数を取り出し、
自前の料金表 (cost_calculator.py) でコストを計算してタグに記録しました。

MLflow 3.10 以降は、自動トレーシングで記録した LLM 呼び出しのコストを MLflow が
計算し、trace.info.cost に入れてくれます。本書 8.2.4 で紹介した Overview ダッシュボードの
Usage ビューに表示されるコストと同じものを、コードから取り出せます。
料金は litellm の料金データから引かれます。MLflow 3.11 以降は litellm が入っていない
環境でも、MLflow に同梱された料金データで計算されます。

このスクリプトは、同じ呼び出しについて組み込みのコストと cost_calculator.py の
計算結果を並べ、最後にエクスペリメント内のトレースのコストを合計します。

実行方法:
  make cost-builtin
  または
  uv run python monitoring/06_builtin_cost.py

本書の対象版 (MLflow 3.9) には trace.info.cost がないため、3.10 以降で動きます。
"""

from dotenv import load_dotenv

load_dotenv()

import mlflow
from openai import OpenAI
from cost_calculator import calculate_cost

mlflow.set_experiment("ch8-monitoring-quickstart")
mlflow.openai.autolog()

client = OpenAI()

prompts = [
    "Pythonのデコレータを簡潔に説明してください。",
    "MLflowのトレーシング機能について、主な利点を3つ挙げてください。",
]

# === 1. 組み込みのコストと自前の計算を比べる ===
print("=== 組み込みのコスト (trace.info.cost) と cost_calculator.py の比較 ===\n")

for prompt in prompts:
    model = "gpt-4o-mini"
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    trace_id = mlflow.get_last_active_trace_id()
    # トレースは非同期で書き込まれるため、取得前にフラッシュする
    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id=trace_id)

    usage = trace.info.token_usage or {}
    builtin = trace.info.cost  # {"input_cost", "output_cost", "total_cost"} (USD)
    manual = calculate_cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    print(f"プロンプト: {prompt[:40]}...")
    print(f"  トークン: 入力 {usage.get('input_tokens')} / 出力 {usage.get('output_tokens')}")
    if builtin:
        print(f"  組み込み: ${builtin['total_cost']:.6f} "
              f"(入力 ${builtin['input_cost']:.6f} + 出力 ${builtin['output_cost']:.6f})")
    else:
        print("  組み込み: なし (料金データにないモデル名か、トークン数が記録されていない)")
    print(f"  cost_calculator.py: ${manual:.6f}")
    print()

# === 2. エクスペリメント内のトレースのコストを合計する ===
print("=== コスト集計 (直近のトレース) ===\n")

traces = mlflow.search_traces(max_results=50, return_type="list")
total = 0.0
counted = 0
for trace in traces:
    if trace.info.cost:
        total += trace.info.cost["total_cost"]
        counted += 1

print(f"コストが記録されたトレース: {counted}件 / {len(traces)}件")
print(f"合計コスト: ${total:.6f}")
print()
print("MLflow UI のトレース一覧でも、トレースごとのコストを確認できます。")
print("自前の料金表 (cost_calculator.py) は、契約価格や自社ホストのモデルなど、")
print("MLflow の料金データにない単価で計算したい場合に使えます。")
