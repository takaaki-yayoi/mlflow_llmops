"""5.7節: [応用] ConversationSimulatorを使った会話シミュレーションと評価。

LLMでユーザーの発話をシミュレートし、複数ターンの会話を
自動生成・評価するデモスクリプト。

実行: make sim
注意: 会話生成に追加のLLM呼び出しが発生するため、APIコストに注意。
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.genai.simulators import ConversationSimulator
from mlflow.genai.scorers import ConversationCompleteness, UserFrustration

from agents.langgraph import LangGraphAgent
from agents.thread import Thread

# --- テストケースの定義 ---
test_cases = [
    {
        "goal": "MLflowのトレーシング機能の使い方を理解し、LangChainでの具体的な実装方法を学ぶ",
    },
    {
        "goal": "MLflowのモデルバージョニングについて学ぶ",
        "persona": "Pythonは書けるがMLflow初心者のデータサイエンティスト。丁寧な説明を求める。",
    },
    {
        "goal": "本番環境でのモデルデプロイでエラーが発生している問題を解決する",
        "persona": "急いでいるエンジニア。簡潔な回答を好む。",
    },
]


def main():
    """会話シミュレーションと評価を実行する。"""
    print("=" * 60)
    print("5.7節: [応用] 会話シミュレーション")
    print("=" * 60)

    # --- エージェントの初期化 ---
    try:
        agent = LangGraphAgent()
    except ConnectionError:
        print("\nMLflow Tracking Serverに接続できません。")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)

    # --- シミュレーターの作成 ---
    simulator = ConversationSimulator(
        test_cases=test_cases,
        max_turns=5,
    )

    # --- セッションIDごとにThreadを管理 ---
    threads: dict[str, Thread] = {}

    def predict_fn(input: list[dict], **kwargs) -> str:
        """会話シミュレーション用のpredict関数。

        Args:
            input: Chat Completions形式のメッセージリスト
                [{"role": "user", "content": "..."}, ...]
        """
        latest_message = input[-1]["content"]
        session_id = kwargs.get("mlflow_session_id", "default")

        if session_id not in threads:
            threads[session_id] = Thread(session_id)

        return agent.process_query(query=latest_message, thread=threads[session_id])

    # --- evaluate()の実行 ---
    print(f"\nテストケース: {len(test_cases)}件")
    print("最大ターン数: 5")
    print("スコアラー: ConversationCompleteness, UserFrustration")
    print("\nシミュレーションを開始します...\n")

    try:
        results = mlflow.genai.evaluate(
            data=simulator,
            predict_fn=predict_fn,
            scorers=[
                ConversationCompleteness(),
                UserFrustration(),
            ],
        )
    except Exception as e:
        print(f"\nシミュレーション中にエラーが発生しました: {e}")
        sys.exit(1)

    # --- 結果の表示 ---
    print("\n" + "=" * 60)
    print("会話シミュレーション完了!")
    print("=" * 60)
    print(f"\nMLflow UI で詳細を確認してください。")
    print(f"  http://localhost:5000")


if __name__ == "__main__":
    main()
