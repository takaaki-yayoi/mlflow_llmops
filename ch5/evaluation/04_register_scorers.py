"""5.4.6節: 評価指標をMLflowに登録してバージョン管理する。

GuidelinesスコアラーをMLflowのJudgesタブに登録し、
get_scorerで取得するデモスクリプト。

実行: make register
前提: MLflow Tracking Serverが起動していること
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow

from evaluation.scorers import has_reference_link, appropriate_katakana

# MLflow接続設定
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")


def main():
    """スコアラーをMLflowに登録する。"""
    print("=" * 60)
    print("5.4.6節: スコアラーの登録")
    print("=" * 60)

    try:
        # has_reference_link を登録
        print("\n--- has_reference_link の登録 ---")
        has_reference_link.register()
        print("  登録完了")

        # appropriate_katakana を登録
        print("\n--- appropriate_katakana の登録 ---")
        appropriate_katakana.register()
        print("  登録完了")

        # 登録済みスコアラーの取得テスト
        # versionを省略すると最新バージョンを取得。特定バージョンの指定も可能: get_scorer(name="...", version=1)
        print("\n--- 登録済みスコアラーの取得テスト ---")
        loaded_scorer = mlflow.genai.get_scorer(name="has_reference_link")
        print(f"  取得成功: {loaded_scorer.name}")

        loaded_scorer2 = mlflow.genai.get_scorer(name="appropriate_katakana")
        print(f"  取得成功: {loaded_scorer2.name}")

    except ConnectionError:
        print("\nMLflow Tracking Serverに接続できません。")
        print("  'uv run mlflow server --port 5000' を実行してください。")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("スコアラーの登録完了!")
    print("MLflow UI (http://localhost:5000) のJudgesタブで")
    print("登録されたスコアラーを確認してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
