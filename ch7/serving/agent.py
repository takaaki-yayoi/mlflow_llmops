"""Agent Server用エージェント定義（7.3節）。

第4章のLangGraphAgentをAgent Serverで公開するためのラッパーです。
@invoke デコレータでResponses APIエンドポイントとして登録します。

変更点（第4章からの差分）:
- システムプロンプトをプロンプトレジストリから取得（6章との連携）
- Responses API形式でリクエスト/レスポンスを処理
- Agent Serverの自動トレーシングを活用
"""

import os
import uuid

import dotenv

dotenv.load_dotenv()

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

# 第4章のエージェントを再利用
from agents.langgraph.agent import LangGraphAgent
from agents.thread import Thread

# --- MLflow設定 ---
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("QAエージェント - サービング")

# エージェントのインスタンスを作成（サーバー起動時に1回だけ初期化）
agent = LangGraphAgent()


def _load_system_prompt() -> str:
    """プロンプトレジストリからシステムプロンプトを取得する。

    第6章でプロンプトレジストリに登録したプロンプトを使用します。
    レジストリが利用できない場合は、第4章のデフォルトプロンプトにフォールバックします。
    """
    try:
        prompt = mlflow.genai.load_prompt("prompts:/qa-system-prompt@production")
        return prompt.template
    except Exception:
        # プロンプトレジストリが未設定の場合は第4章のデフォルトを使用
        from agents.langgraph.agent import SYSTEM_PROMPT

        return SYSTEM_PROMPT


@invoke()
async def handle_request(request) -> ResponsesAgentResponse:
    """QAエージェントへのリクエストを処理する。

    Responses APIのメッセージ形式を受け取り、第4章のLangGraphAgentで処理し、
    Responses API形式で返します。

    Args:
        request: Responses API形式のリクエスト（dictまたはResponsesAgentRequest）

    Returns:
        Responses API形式のレスポンス
    """
    # dictの場合はResponsesAgentRequestに変換
    if isinstance(request, dict):
        request = ResponsesAgentRequest(**request)

    # Responses APIのメッセージからユーザーの質問を抽出
    user_message = None
    for msg in request.input:
        msg_dict = msg.model_dump() if hasattr(msg, "model_dump") else msg
        if msg_dict.get("role") == "user":
            # contentがリストの場合はテキストを結合
            content = msg_dict.get("content", "")
            if isinstance(content, list):
                user_message = " ".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "input_text"
                )
            else:
                user_message = content

    if not user_message:
        return ResponsesAgentResponse(
            output=[
                {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "質問を入力してください。",
                        }
                    ],
                }
            ]
        )

    # 第4章のエージェントで処理
    # リクエストごとに新しいスレッドを作成（ステートレスサービング）
    thread = Thread()
    response_text = agent.process_query(user_message, thread)

    # Responses API形式でレスポンスを返す
    return ResponsesAgentResponse(
        output=[
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": response_text,
                    }
                ],
            }
        ]
    )
