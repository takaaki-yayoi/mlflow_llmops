"""QAエージェントのモデル記録スクリプト（7.2節）。

第4章で構築したLangGraphエージェントをMLflowに記録し、
モデルレジストリに登録します。

使用方法:
    make log-model
    # または
    uv run python -m serving.log_model
"""

import os
from pathlib import Path

import dotenv

dotenv.load_dotenv()

import mlflow
from mlflow import MlflowClient


# --- MLflow設定 ---
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "QAエージェント - サービング"
REGISTERED_MODEL_NAME = "qa-agent"

# models-from-code用のモデル定義ファイル
MODEL_CODE_PATH = str(Path(__file__).parent / "model_code.py")


def log_agent():
    """QAエージェントをMLflowに記録する。

    MLflow v3ではmodels-from-codeパターンを使用し、
    モデル定義コードのパスを指定して記録します。

    Returns:
        記録されたモデルの情報
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="qa-agent-serving") as run:
        # models-from-codeパターンでエージェントを記録
        model_info = mlflow.langchain.log_model(
            lc_model=MODEL_CODE_PATH,
            name="qa-agent",                                # アーティファクト名
            registered_model_name=REGISTERED_MODEL_NAME,    # レジストリに自動登録
        )

        # メタデータをタグとして記録
        mlflow.set_tags(
            {
                "agent_type": "langgraph",
                "tools": "doc_search,web_search,open_url",
                "chapter": "7",
                "base_chapter": "4",
            }
        )

        print(f"モデルを記録しました:")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Model URI: {model_info.model_uri}")
        print(f"  Registered Model: {REGISTERED_MODEL_NAME}")

        return model_info, run


def set_champion_alias(model_info):
    """log_model()で自動登録されたモデルにchampionエイリアスを設定する。

    Args:
        model_info: mlflow.langchain.log_model()の戻り値
    """
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # log_model(registered_model_name=...)で自動登録されたバージョンを取得
    model_version = model_info.registered_model_version

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=model_version,
    )

    print(f"championエイリアスを設定しました:")
    print(f"  名前: {REGISTERED_MODEL_NAME}")
    print(f"  バージョン: {model_version}")
    print(f"  エイリアス: champion")


def verify_model():
    """記録したモデルをロードして動作確認する。"""
    print("\n--- モデルの動作確認 ---")

    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    print(f"モデルをロード中: {model_uri}")

    loaded_model = mlflow.langchain.load_model(model_uri)

    test_query = "MLflow Tracingの主な機能を教えてください"
    print(f"テストクエリ: {test_query}")

    result = loaded_model.invoke(
        {
            "messages": [
                {"role": "user", "content": test_query}
            ]
        },
        config={"configurable": {"thread_id": "verify-test"}},
    )

    # 最後のAIメッセージを取得
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            print(f"回答: {msg.content[:200]}...")
            break

    print("動作確認: OK")


def main():
    """メインの実行フロー。"""
    print("=" * 50)
    print("QAエージェントのモデル記録（第7章）")
    print("=" * 50)

    # 1. エージェントをMLflowに記録・レジストリに自動登録
    model_info, run = log_agent()

    # 2. championエイリアスを設定
    set_champion_alias(model_info)

    # 3. 動作確認
    verify_model()

    print("\n完了! 次のコマンドでサービングを開始できます:")
    print("  make serve")


if __name__ == "__main__":
    main()
