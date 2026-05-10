"""サービング環境での評価スクリプト（7.3節）。

Agent Serverと同じ@invoke関数をin-processで呼び出し、
第5章と同じ評価フレームワークを適用します。

注意: Agent Serverを停止してから実行してください。
    エージェント初期化時にMilvusデータベースを開くため、
    Agent Serverが起動中だとファイルロックが競合します。
    この制約は本書本文には明記されていません。詳細は ch7/CHAPTER_NOTES.md を参照してください。

使用方法:
    make eval
    # または
    uv run python -m serving.eval_serving
"""

# 本書 リスト7.4
import asyncio
import os

import dotenv

dotenv.load_dotenv()

import mlflow
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

import serving.agent  # noqa: F401

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("QAエージェント - サービング評価")
TRACKING_URI = mlflow.get_tracking_uri()


# 評価データセット（第5章と共通の形式）
eval_dataset = [
    {
        "inputs": {
            "request": {
                "input": [
                    {"role": "user", "content": "MLflow Tracingとは何ですか?"}
                ]
            }
        },
        "expected_response": (
            "MLflow Tracingは、LLMアプリケーションの実行フローを可視化するための"
            "機能です。各ステップの入出力、レイテンシ、トークン使用量を記録し、"
            "デバッグや性能分析に活用できます。"
        ),
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {"role": "user", "content": "MLflowでプロンプトをバージョン管理する方法は?"}
                ]
            }
        },
        "expected_response": (
            "MLflowのPrompt Registryを使ってプロンプトをバージョン管理できます。"
            "mlflow.genai.register_prompt()で登録し、"
            "エイリアス(@production, @latestなど)で環境ごとに使い分けられます。"
        ),
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {"role": "user", "content": "MLflowの評価機能でLLMの品質をどう測定しますか?"}
                ]
            }
        },
        "expected_response": (
            "mlflow.genai.evaluate()を使用し、LLM-as-a-Judgeスコアラーで"
            "関連性、安全性、正確性などの品質指標を自動的に評価できます。"
        ),
    },
]


def sync_invoke_fn(request: dict) -> ResponsesAgentResponse:
    """Agent Serverの@invoke関数を同期的に呼び出すラッパー。"""
    invoke_fn = get_invoke_function()
    return asyncio.run(invoke_fn(ResponsesAgentRequest(**request)))


def main():
    """サービング環境のエージェントを評価する。"""
    print("=" * 50)
    print("サービング環境での評価（第7章）")
    print("=" * 50)

    # 第5章と同じスコアラーで評価
    scorers = [
        RelevanceToQuery(),
        Safety(),
        Guidelines(
            name="uses_sources",
            guidelines=(
                "回答にはMLflow公式ドキュメントや検索結果に基づく"
                "具体的な情報を含む必要があります。"
            ),
        ),
    ]

    print(f"評価データセット: {len(eval_dataset)} 件")
    print(f"スコアラー: {', '.join(s.name for s in scorers)}")
    print()

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=sync_invoke_fn,
        scorers=scorers,
    )

    # 結果の表示
    print(f"評価完了: {len(eval_dataset)} 件")
    for metric_name, value in results.metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    print("\n詳細はMLflow UIで確認できます:")
    print(f"  {TRACKING_URI}")


if __name__ == "__main__":
    main()
