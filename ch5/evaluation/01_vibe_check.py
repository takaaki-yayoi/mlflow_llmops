"""5.2節: テスト用の質問を用意して実行し、トレースを確認する。

Vibe Check - エージェントを実際に動かして振る舞いを観察する最初のステップ。
3つのテスト質問をエージェントに投げ、MLflow UIでトレースを確認します。

実行: make vibe-check
前提: MLflow Tracking Serverが起動していること (uv run mlflow server --port 5000)
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from agents.langgraph import LangGraphAgent
from agents.thread import Thread

# テスト質問リスト
TEST_QUESTIONS = [
    "実験管理はどのように始めれば良いですか？",
    "LangGraphエージェントのトークン使用量を追跡するにはどうすればよいですか？",
    "MLflowトレーシングはどのフレームワークに対応していますか？",
]


def main():
    """テスト質問を実行してトレースを生成する。"""
    print("=" * 60)
    print("5.2節: Vibe Check - テスト質問の実行")
    print("=" * 60)

    try:
        agent = LangGraphAgent()
    except ConnectionError:
        print("\nMLflow Tracking Serverに接続できません。")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)

    print(f"\n{len(TEST_QUESTIONS)}件のテスト質問を実行します...\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"--- 質問 {i}/{len(TEST_QUESTIONS)} ---")
        print(f"Q: {question}")

        # 質問ごとに新しいThreadを作成
        thread = Thread()
        try:
            answer = agent.process_query(query=question, thread=thread)
            # 回答を最大300文字で表示
            display_answer = answer[:300] + "..." if len(answer) > 300 else answer
            print(f"A: {display_answer}")
        except Exception as e:
            print(f"エラー: {e}")

        print()

    print("=" * 60)
    print("Vibe Check完了!")
    print("MLflow UI (http://localhost:5000) のTracesタブで")
    print("トレースを確認してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
