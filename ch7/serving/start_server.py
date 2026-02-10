"""Agent Server起動スクリプト（7.3節）。

QAエージェントをResponses APIエンドポイントとしてホストします。

使用方法:
    make serve
    # または
    uv run python serving/start_server.py --reload --port 5005

テスト:
    curl -X POST http://localhost:5005/invocations \
        -H "Content-Type: application/json" \
        -d '{"input": [{"role": "user", "content": "MLflow Tracingとは何ですか?"}]}'
"""

# agent.py内の@invokeデコレータを登録するためにインポートが必要
import serving.agent  # noqa: F401

from mlflow.genai.agent_server import (
    AgentServer,
    setup_mlflow_git_based_version_tracking,
)

agent_server = AgentServer("QAAgent")
app = agent_server.app

# Gitコミットとトレースを紐付け（任意）
# リポジトリのルートで実行した場合、トレースにコミットハッシュが記録されます
setup_mlflow_git_based_version_tracking()


def main():
    # app_import_stringを指定することで複数ワーカーをサポート
    agent_server.run(app_import_string="serving.start_server:app")


if __name__ == "__main__":
    main()
