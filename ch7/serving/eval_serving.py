"""サービング中エージェントの評価スクリプト（7.3.3節）。

Agent Server経由で動作するQAエージェントに対して、
第5章と同じ評価フレームワークを適用します。

使用方法:
    # Agent Serverが起動している状態で実行
    make eval
    # または
    uv run python -m serving.eval_serving
"""

import asyncio
import os

import dotenv

dotenv.load_dotenv()

import mlflow
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# @invokeデコレータの登録に必要
import serving.agent  # noqa: F401

# --- MLflow設定 ---
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("QAエージェント - サービング評価")


# 評価データセット（第5章のデータセットと共通の形式）
EVAL_DATASET = [
    {
        "inputs": {
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "MLflow Tracingとは何ですか?",
                    }
                ]
            }
        },
        "expected_response": (
            "MLflow Tracingは、LLMアプリケーションの実行フローを可視化するための"
            "機能です。プロンプト、検索結果、ツール呼び出し、モデルの応答を"
            "記録し、デバッグや品質改善に活用できます。"
        ),
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "MLflowでプロンプトをバージョン管理する方法は?",
                    }
                ]
            }
        },
        "expected_response": (
            "MLflowのプロンプトレジストリを使うことで、プロンプトの"
            "バージョン管理とエイリアスによるライフサイクル管理が可能です。"
        ),
    },
    {
        "inputs": {
            "request": {
                "input": [
                    {
                        "role": "user",
                        "content": "MLflowの評価機能でLLMの品質をどう測定しますか?",
                    }
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
    """Agent Serverの@invoke関数を同期的に呼び出すラッパー。

    mlflow.genai.evaluate()はpredict_fnに同期関数を要求するため、
    非同期の@invoke関数をラップします。
    """
    invoke_fn = get_invoke_function()
    return asyncio.run(invoke_fn(ResponsesAgentRequest(**request)))


def main():
    """サービング中のエージェントを評価する。"""
    print("=" * 50)
    print("サービング中エージェントの評価（第7章）")
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

    print(f"評価データセット: {len(EVAL_DATASET)} 件")
    print(f"スコアラー: {', '.join(s.name for s in scorers)}")
    print()

    results = mlflow.genai.evaluate(
        data=EVAL_DATASET,
        predict_fn=sync_invoke_fn,
        scorers=scorers,
    )

    # 結果の表示
    print("\n--- 評価結果 ---")
    for metric_name, value in results.metrics.items():
        print(f"  {metric_name}: {value:.3f}")

    print(f"\n評価完了: {len(EVAL_DATASET)} 件")
    print("詳細はMLflow UIで確認できます:")
    print(f"  {TRACKING_URI}")


if __name__ == "__main__":
    main()
