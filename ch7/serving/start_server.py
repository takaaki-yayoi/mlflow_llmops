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

# 本書 リスト7.3
import serving.agent  # noqa: F401 (デコレータの登録に必要)

from mlflow.genai.agent_server import (
    AgentServer,
    setup_mlflow_git_based_version_tracking,
)

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Gitコミットとトレースをひも付け（7.3.1で解説）
setup_mlflow_git_based_version_tracking()


def main():
    agent_server.run(app_import_string="serving.start_server:app")


if __name__ == "__main__":
    main()
