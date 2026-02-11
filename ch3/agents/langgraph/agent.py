"""LangGraphを使用したエージェント実装。

このモジュールは、LangGraphフレームワークを使用してMLflowに関する質問に
回答するエージェントを実装しています。ツール呼び出しとメモリ管理を
サポートしています。
"""
from typing import List, Optional
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

from agents.thread import Message, Thread
from .tools import doc_search, web_search, open_url


# システムプロンプト: エージェントの役割と振る舞いを定義
SYSTEM_PROMPT = """あなたはMLflowに関する質問に答える専門アシスタントです。
MLflowは機械学習のライフサイクルを管理するためのオープンソースプラットフォームです。

あなたの責務:
- MLflowの機能、API、ベストプラクティスに関する質問に回答する
- MLflowの概念を説明し、問題のトラブルシューティングを支援する
- 適切なリソースやドキュメントへユーザーを案内する

利用可能なツールを使用して、正確で最新の情報を取得してください。
ツールから取得した情報を提供する際は、必ずURLを含む引用を記載してください。
"""


class LangGraphAgent:
    """MLflowに関する質問に回答するLangGraphエージェント。

    このクラスは、リポジトリ内のすべてのエージェント実装が提供すべき
    インターフェースを提供します。CLIとAPIは、基盤となるフレームワークに
    関係なく、このクラスを使用してエージェントと対話します。
    """

    def __init__(self):
        """エージェントを初期化する。"""
        # 利用可能なツールのリストを作成（Noneのツールは除外）
        self.tools = [t for t in [doc_search, web_search, open_url] if t is not None]
        # LangGraphのグラフを構築
        self.executor = self._build_graph()

        print(f"LangGraphエージェントを初期化しました（ツール数: {len(self.tools)}）:")
        for tool in self.tools:
            print(f"  - {tool.name}")

    def _tools_condition(self, messages: List[BaseMessage]) -> str:
        """最後のメッセージにツール呼び出しがあれば'tools'、なければ'end'を返す。"""
        if messages:
            last = messages[-1]
            # AIメッセージにツール呼び出しがあるかチェック
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", []):
                return "tools"
        return "end"

    def _build_graph(self):
        """LangGraphエージェントのグラフを構築する。

        Returns:
            コンパイル済みのLangGraphグラフ
        """
        model = ChatOpenAI(model=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
        # ツールがあればモデルにバインド
        model_with_tools = model.bind_tools(self.tools) if self.tools else model

        def call_model(state: MessagesState):
            """モデルを呼び出してレスポンスを取得するノード。"""
            response = model_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        # グラフを構築
        graph = StateGraph(MessagesState)
        graph.add_node("agent", call_model)

        if self.tools:
            # ツールノードを追加し、条件付きエッジを設定
            tool_node = ToolNode(self.tools)
            graph.add_node("tools", tool_node)
            graph.add_conditional_edges(
                "agent",
                lambda state: self._tools_condition(state["messages"]),
                {"tools": "tools", "end": END},
            )
            # ツール実行後はエージェントに戻る
            graph.add_edge("tools", "agent")
        else:
            # ツールがない場合は直接終了
            graph.add_edge("agent", END)

        # エントリーポイントを設定してコンパイル
        graph.set_entry_point("agent")
        # MemorySaverで会話履歴を保持
        return graph.compile(checkpointer=MemorySaver())

    def _extract_last_ai_message(self, messages) -> Optional[AIMessage]:
        """メッセージリストから最後のAIメッセージを抽出する。"""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    def process_query(self, query: str, thread: Thread) -> str:
        """ユーザーのクエリをエージェントで処理する。

        これは、リポジトリ内のすべてのエージェント実装が提供すべき
        統一インターフェースです。CLIとAPIは、基盤となるフレームワーク
        （LangGraph、Pydantic AI、OpenAI Agents SDKなど）に関係なく、
        このメソッドを使用してエージェントと対話します。

        Args:
            query: ユーザーの入力メッセージ
            thread: コンテキストを維持するための会話スレッド

        Returns:
            エージェントの回答（文字列）
        """
        print(f"クエリを処理中: {query}")

        incoming_messages = []

        # 新しい会話の場合はシステムプロンプトを追加
        if not thread.messages:
            incoming_messages.append(SystemMessage(content=SYSTEM_PROMPT))
            thread.messages.append(Message(role="system", content=SYSTEM_PROMPT))

        # ユーザーメッセージを追加
        incoming_messages.append(HumanMessage(content=query))
        thread.messages.append(Message(role="user", content=query))

        # エージェントの実行
        config = {"configurable": {"thread_id": thread.id}}
        result_state = self.executor.invoke({"messages": incoming_messages}, config=config)

        # AIの回答を抽出
        ai_message = self._extract_last_ai_message(result_state["messages"])
        if not ai_message:
            raise RuntimeError("エージェントからの回答がありませんでした。")

        # 回答をスレッドに保存
        thread.messages.append(Message(role="assistant", content=ai_message.content))
        return ai_message.content
