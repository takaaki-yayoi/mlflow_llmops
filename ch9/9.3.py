# Databricks notebook source
# MAGIC %md
# MAGIC # スーパーバイザー型マルチエージェントシステムの構築 - MLflow & LangGraph Tutorial
# MAGIC
# MAGIC ## 概要
# MAGIC このノートブックでは、複数の専門エージェントが協調して複雑なタスクを遂行する「スーパーバイザー型マルチエージェントシステム」を構築します。技術レポート作成という実践的なユースケースを通じて、エージェント設計、協調パターン、評価手法を学びます。
# MAGIC
# MAGIC ### 学習内容
# MAGIC 1. マルチエージェントシステムの設計パターン（スーパーバイザー型）
# MAGIC 2. 専門化されたエージェントの実装（リサーチ、構成、執筆、レビュー）
# MAGIC 3. MLflow ResponseAgentによる標準インターフェース化
# MAGIC 4. 多様な評価手法（Safety、Guidelines、カスタムスコアラー、Agent-as-a-Judge）
# MAGIC 5. トレースベースの品質分析
# MAGIC
# MAGIC ### スーパーバイザー型アーキテクチャとは？
# MAGIC
# MAGIC **従来の単一エージェント**
# MAGIC - 1つのLLMがすべてのタスクを担当
# MAGIC - 複雑なタスクでは品質が低下
# MAGIC
# MAGIC **スーパーバイザー型マルチエージェント**
# MAGIC - 各エージェントが専門領域に特化
# MAGIC - スーパーバイザーが全体を調整し、適切なエージェントに仕事を割り当て
# MAGIC - 段階的な処理により高品質な出力を実現
# MAGIC
# MAGIC ### 本ノートブックのエージェント構成
# MAGIC
# MAGIC 1. **リサーチエージェント**: テーマの調査とポイント整理
# MAGIC 2. **構成エージェント**: レポート構成と見出しの決定
# MAGIC 3. **ライティングエージェント**: 本文の執筆
# MAGIC 4. **レビューエージェント**: 品質チェックと修正
# MAGIC 5. **スーパーバイザー**: 処理フローの制御と次のエージェントの決定

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 環境セットアップ
# MAGIC
# MAGIC ### 必要なライブラリ
# MAGIC - `mlflow`: 実験管理、モデル記録、トレーシング、評価
# MAGIC - `langchain[openai]`: LangChainとOpenAI連携
# MAGIC - `langgraph`: マルチエージェントワークフローの構築
# MAGIC - `litellm`: 複数のLLMプロバイダーへの統一インターフェース

# COMMAND ----------

# MAGIC %pip install mlflow langchain[openai] langgraph litellm

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: 認証情報の設定
# MAGIC
# MAGIC OpenAI APIを使用するため、APIキーを環境変数に設定します。
# MAGIC
# MAGIC **重要**: 
# MAGIC - `YOUR_API_KEY`を実際のAPIキーに置き換えてください
# MAGIC - 本番環境では、シークレット管理サービスを使用することを推奨します

# COMMAND ----------

import os

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: マルチエージェントシステムの実装
# MAGIC
# MAGIC ### アーキテクチャの詳細
# MAGIC
# MAGIC #### 共有状態（AgentState）
# MAGIC すべてのエージェントが参照・更新する共通のデータ構造です。
# MAGIC - `topic`: レポートのテーマ
# MAGIC - `research_notes`: リサーチエージェントの出力
# MAGIC - `outline`: 構成エージェントの出力
# MAGIC - `draft`: ライティングエージェントの出力
# MAGIC - `review_comments`: レビューエージェントのコメント
# MAGIC - `final_report`: 最終的なレポート
# MAGIC - `next_agent`: スーパーバイザーが決定する次の担当エージェント
# MAGIC
# MAGIC #### 処理フロー
# MAGIC ```
# MAGIC START → Supervisor → Research Agent → Supervisor
# MAGIC                    → Outline Agent → Supervisor
# MAGIC                    → Writer Agent → Supervisor
# MAGIC                    → Review Agent → Supervisor → END
# MAGIC ```
# MAGIC
# MAGIC ### エージェント設計のポイント
# MAGIC
# MAGIC **専門化**
# MAGIC - 各エージェントは1つの明確な役割のみを担当
# MAGIC - プロンプトは役割に特化した指示を含む
# MAGIC
# MAGIC **状態の受け渡し**
# MAGIC - 前のエージェントの出力を次のエージェントの入力として利用
# MAGIC - スーパーバイザーが状態を確認し、次のステップを判断
# MAGIC
# MAGIC **トレーサビリティ**
# MAGIC - MLflow Tracingで各エージェントの処理を記録
# MAGIC - デバッグや品質分析が容易

# COMMAND ----------

# MAGIC %%writefile ./multi_agent_report_app.py
# MAGIC """
# MAGIC スーパーバイザー型マルチエージェントによる技術レポート作成アプリケーションです。
# MAGIC
# MAGIC エージェントの役割:
# MAGIC   - リサーチエージェント: テーマについて調査し、ポイントを箇条書きで整理
# MAGIC   - 構成エージェント: 見出し構成と各見出しの要点を決定
# MAGIC   - ライティングエージェント: 構成に沿って本文を執筆
# MAGIC   - レビューエージェント: レポート案をチェックし、必要なら修正を提案
# MAGIC
# MAGIC スーパーバイザー:
# MAGIC   - 全体を調整し、各エージェントに順番に仕事を振り分ける
# MAGIC
# MAGIC MLflow ResponseAgent:
# MAGIC   - システム全体をResponseAgentでラッピングし、標準的なインターフェースで公開します。
# MAGIC """
# MAGIC
# MAGIC from __future__ import annotations
# MAGIC
# MAGIC from typing import TypedDict, Literal, Annotated
# MAGIC import functools
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.entities import SpanType
# MAGIC
# MAGIC from langchain_openai import ChatOpenAI
# MAGIC from langchain_core.messages import HumanMessage, SystemMessage
# MAGIC from langgraph.graph import StateGraph, START, END, MessagesState
# MAGIC from langgraph.prebuilt import create_react_agent
# MAGIC from langgraph.graph.message import add_messages
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # 事前準備
# MAGIC # ==========
# MAGIC
# MAGIC # LLM（全エージェント共通）
# MAGIC # temperature=0.2で適度な創造性を保ちつつ、一貫性のある出力を実現
# MAGIC llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
# MAGIC
# MAGIC # LangChain/MLflowの自動ロギングを有効化
# MAGIC # LLM呼び出しやエージェント実行を自動的にMLflowに記録
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # 状態の定義
# MAGIC # ==========
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     """
# MAGIC     エージェント間で受け渡す情報をまとめた共有状態です。
# MAGIC     
# MAGIC     この状態はワークフロー全体を通じて維持され、
# MAGIC     各エージェントが必要な情報を読み取り、自分の出力を書き込みます。
# MAGIC     """
# MAGIC     topic: str              # レポートのテーマ
# MAGIC     research_notes: str     # リサーチ結果（箇条書き）
# MAGIC     outline: str            # レポート構成（見出しと要点）
# MAGIC     draft: str              # レポート本文（初稿）
# MAGIC     review_comments: str    # レビューコメント
# MAGIC     final_report: str       # 最終レポート（修正済み）
# MAGIC     next_agent: str         # スーパーバイザーが決定する次のエージェント名
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # 各エージェントノードの実装
# MAGIC # ==========
# MAGIC
# MAGIC def create_agent_node(agent_name: str, system_prompt: str):
# MAGIC     """
# MAGIC     エージェントノードを作成するファクトリー関数です。
# MAGIC     
# MAGIC     このパターンにより、コードの重複を避け、
# MAGIC     各エージェントの設定を一元管理できます。
# MAGIC     
# MAGIC     Args:
# MAGIC         agent_name: エージェントの識別名
# MAGIC         system_prompt: エージェントの役割と指示を定義するプロンプト
# MAGIC     
# MAGIC     Returns:
# MAGIC         エージェントノード関数
# MAGIC     """
# MAGIC     def agent_node(state: AgentState) -> AgentState:
# MAGIC         """エージェントの処理を実行し、状態を更新します。"""
# MAGIC         # 状態から必要な情報を取り出す
# MAGIC         topic = state.get("topic", "")
# MAGIC         research = state.get("research_notes", "")
# MAGIC         outline = state.get("outline", "")
# MAGIC         draft = state.get("draft", "")
# MAGIC
# MAGIC         # エージェントごとに異なる入力を構築
# MAGIC         # 各エージェントは前段階の出力を入力として受け取る
# MAGIC         if agent_name == "research_agent":
# MAGIC             user_content = f"テーマ: {topic}"
# MAGIC         elif agent_name == "outline_agent":
# MAGIC             user_content = f"テーマ: {topic}\n\nリサーチ結果:\n{research}"
# MAGIC         elif agent_name == "writer_agent":
# MAGIC             user_content = f"テーマ: {topic}\n\n構成案:\n{outline}"
# MAGIC         elif agent_name == "review_agent":
# MAGIC             user_content = f"レポートドラフト:\n{draft}"
# MAGIC         else:
# MAGIC             user_content = ""
# MAGIC
# MAGIC         # LLMを呼び出し
# MAGIC         messages = [
# MAGIC             SystemMessage(content=system_prompt),
# MAGIC             HumanMessage(content=user_content),
# MAGIC         ]
# MAGIC         response = llm.invoke(messages)
# MAGIC         result = response.content
# MAGIC
# MAGIC         # 結果を状態に保存
# MAGIC         # 各エージェントは自分の担当フィールドのみを更新
# MAGIC         if agent_name == "research_agent":
# MAGIC             state["research_notes"] = result
# MAGIC         elif agent_name == "outline_agent":
# MAGIC             state["outline"] = result
# MAGIC         elif agent_name == "writer_agent":
# MAGIC             state["draft"] = result
# MAGIC         elif agent_name == "review_agent":
# MAGIC             # レビュー結果を解析して、コメントと最終レポートに分割
# MAGIC             if "【修正後レポート案】" in result:
# MAGIC                 comments, final = result.split("【修正後レポート案】", maxsplit=1)
# MAGIC                 state["review_comments"] = comments.strip()
# MAGIC                 state["final_report"] = final.strip()
# MAGIC             else:
# MAGIC                 state["review_comments"] = result
# MAGIC                 state["final_report"] = draft  # 修正不要の場合は原稿をそのまま使用
# MAGIC
# MAGIC         # トレースにプレビューを記録（後で分析しやすくするため）
# MAGIC         mlflow.update_current_trace(tags={
# MAGIC             f"{agent_name}_preview": result[:100],  # 最初の100文字のみ
# MAGIC         })
# MAGIC
# MAGIC         return state
# MAGIC
# MAGIC     return agent_node
# MAGIC
# MAGIC
# MAGIC # 4つの専門エージェントを作成
# MAGIC # 各エージェントは明確な役割と具体的な出力形式を持つ
# MAGIC
# MAGIC research_node = create_agent_node(
# MAGIC     "research_agent",
# MAGIC     "あなたは技術リサーチ担当です。テーマについて重要なポイントを箇条書きで5〜7個挙げてください。"
# MAGIC )
# MAGIC
# MAGIC outline_node = create_agent_node(
# MAGIC     "outline_agent",
# MAGIC     "あなたは技術レポートの構成を考える担当です。リサーチメモをもとに、見出し構成（3〜5個）と各見出しの要点を番号付きで出力してください。"
# MAGIC )
# MAGIC
# MAGIC writer_node = create_agent_node(
# MAGIC     "writer_agent",
# MAGIC     "あなたは技術レポートの執筆担当です。構成案に従って、各見出しごとに2〜4文程度で本文を書いてください。専門用語は平易な言葉で説明してください。"
# MAGIC )
# MAGIC
# MAGIC review_node = create_agent_node(
# MAGIC     "review_agent",
# MAGIC     "あなたは技術レポートのレビュー担当です。技術的な正確さ、構成のわかりやすさ、文体をチェックし、【コメント】と【修正後レポート案】の形式で出力してください。"
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # スーパーバイザーノード
# MAGIC # ==========
# MAGIC
# MAGIC def supervisor_node(state: AgentState) -> AgentState:
# MAGIC     """
# MAGIC     【スーパーバイザー: ワークフローの制御塔】
# MAGIC     
# MAGIC     現在の状態を確認し、次にどのエージェントを呼ぶかを決定します。
# MAGIC     
# MAGIC     判断ロジック（順次処理）:
# MAGIC     1. リサーチ結果がない → research_agent
# MAGIC     2. 構成案がない → outline_agent
# MAGIC     3. 本文がない → writer_agent
# MAGIC     4. 最終レポートがない → review_agent
# MAGIC     5. すべて完了 → FINISH
# MAGIC     """
# MAGIC     # 状態を見て、次に呼ぶべきエージェントを判断
# MAGIC     if not state.get("research_notes"):
# MAGIC         next_agent = "research_agent"
# MAGIC     elif not state.get("outline"):
# MAGIC         next_agent = "outline_agent"
# MAGIC     elif not state.get("draft"):
# MAGIC         next_agent = "writer_agent"
# MAGIC     elif not state.get("final_report"):
# MAGIC         next_agent = "review_agent"
# MAGIC     else:
# MAGIC         next_agent = "FINISH"
# MAGIC
# MAGIC     state["next_agent"] = next_agent
# MAGIC
# MAGIC     # トレースに判断結果を記録
# MAGIC     mlflow.update_current_trace(tags={
# MAGIC         "next_agent": next_agent,
# MAGIC     })
# MAGIC
# MAGIC     return state
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # グラフの構築
# MAGIC # ==========
# MAGIC
# MAGIC def build_graph():
# MAGIC     """
# MAGIC     スーパーバイザー型のマルチエージェントグラフを構築します。
# MAGIC     
# MAGIC     グラフ構造:
# MAGIC     - 中央にスーパーバイザーを配置
# MAGIC     - 各エージェントはスーパーバイザーから呼び出され、完了後にスーパーバイザーに戻る
# MAGIC     - スーパーバイザーが次のエージェントを決定（条件分岐）
# MAGIC     """
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC
# MAGIC     # 5つのノードを追加（スーパーバイザー + 4つの専門エージェント）
# MAGIC     workflow.add_node("supervisor", supervisor_node)
# MAGIC     workflow.add_node("research_agent", research_node)
# MAGIC     workflow.add_node("outline_agent", outline_node)
# MAGIC     workflow.add_node("writer_agent", writer_node)
# MAGIC     workflow.add_node("review_agent", review_node)
# MAGIC
# MAGIC     # 開始点: まずスーパーバイザーから開始
# MAGIC     workflow.add_edge(START, "supervisor")
# MAGIC
# MAGIC     # スーパーバイザーの判断に基づいて分岐
# MAGIC     def route_supervisor(state: AgentState) -> Literal["research_agent", "outline_agent", "writer_agent", "review_agent", "__end__"]:
# MAGIC         """スーパーバイザーの決定に基づいて次のノードを返す"""
# MAGIC         next_agent = state.get("next_agent", "FINISH")
# MAGIC         if next_agent == "FINISH":
# MAGIC             return END
# MAGIC         return next_agent
# MAGIC
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "supervisor",
# MAGIC         route_supervisor,
# MAGIC         {
# MAGIC             "research_agent": "research_agent",
# MAGIC             "outline_agent": "outline_agent",
# MAGIC             "writer_agent": "writer_agent",
# MAGIC             "review_agent": "review_agent",
# MAGIC             END: END,
# MAGIC         },
# MAGIC     )
# MAGIC
# MAGIC     # 各エージェント実行後は、必ずスーパーバイザーに戻る（固定エッジ）
# MAGIC     # これにより、スーパーバイザーが次のステップを制御できる
# MAGIC     for agent in ["research_agent", "outline_agent", "writer_agent", "review_agent"]:
# MAGIC         workflow.add_edge(agent, "supervisor")
# MAGIC
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC # グラフを構築
# MAGIC graph = build_graph()
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # ResponseAgentでラッピング
# MAGIC # ==========
# MAGIC
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     output_to_responses_items_stream,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from typing import Generator
# MAGIC
# MAGIC class MultiAgentResponsesAgent(ResponsesAgent):
# MAGIC     """
# MAGIC     マルチエージェントシステムをResponsesAgentでラッピングします。
# MAGIC     
# MAGIC     ResponsesAgentとは？
# MAGIC     - MLflowが提供する標準的なエージェントインターフェース
# MAGIC     - OpenAI互換のAPI形式でサービング可能
# MAGIC     - ストリーミングと非ストリーミングの両方に対応
# MAGIC     
# MAGIC     メリット:
# MAGIC     - 既存のチャットアプリケーションとの統合が容易
# MAGIC     - REST API、Python、CLI等、複数の方法で呼び出し可能
# MAGIC     - MLflowの評価・モニタリング機能と統合
# MAGIC     """
# MAGIC
# MAGIC     def __init__(self, graph):
# MAGIC         self.graph = graph
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         """
# MAGIC         非ストリーミング版の予測メソッド。
# MAGIC         ResponsesAgentRequestを受け取り、マルチエージェントを実行します。
# MAGIC         
# MAGIC         Args:
# MAGIC             request: OpenAI互換の入力リクエスト
# MAGIC         
# MAGIC         Returns:
# MAGIC             ResponsesAgentResponse: 最終レポートを含むレスポンス
# MAGIC         """
# MAGIC         # ストリーミング版を実行して、完了イベントだけを集める
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC
# MAGIC         return ResponsesAgentResponse(
# MAGIC             output=outputs,
# MAGIC             custom_outputs=request.custom_inputs
# MAGIC         )
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         """
# MAGIC         ストリーミング版の予測メソッド。
# MAGIC         各エージェントの出力をリアルタイムで返すことも可能ですが、
# MAGIC         ここでは最終結果のみを返す実装としています。
# MAGIC         
# MAGIC         Args:
# MAGIC             request: OpenAI互換の入力リクエスト
# MAGIC         
# MAGIC         Yields:
# MAGIC             ResponsesAgentStreamEvent: ストリーミングイベント
# MAGIC         """
# MAGIC         # RequestをChatCompletions形式に変換
# MAGIC         messages = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC
# MAGIC         # ユーザーの質問からトピックを抽出
# MAGIC         topic = messages[-1]["content"] if messages else ""
# MAGIC
# MAGIC         # グラフに渡す初期状態を作成
# MAGIC         initial_state = AgentState(
# MAGIC             topic=topic,
# MAGIC             research_notes="",
# MAGIC             outline="",
# MAGIC             draft="",
# MAGIC             review_comments="",
# MAGIC             final_report="",
# MAGIC             next_agent="",
# MAGIC         )
# MAGIC
# MAGIC         # グラフを実行（すべてのエージェントが順次実行される）
# MAGIC         final_state = self.graph.invoke(initial_state)
# MAGIC
# MAGIC         # 最終レポートを出力として返す
# MAGIC         final_report = final_state.get("final_report", "")
# MAGIC
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item(
# MAGIC                 text=final_report,
# MAGIC                 id="final_report",
# MAGIC             )
# MAGIC         )
# MAGIC
# MAGIC
# MAGIC # エージェントをインスタンス化してMLflowに登録
# MAGIC agent = MultiAgentResponsesAgent(graph)
# MAGIC mlflow.models.set_model(agent)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: マルチエージェントシステムの実行とテスト
# MAGIC
# MAGIC 構築したマルチエージェントシステムに実際にレポート作成を依頼します。
# MAGIC
# MAGIC ### ResponsesAgentRequestの形式
# MAGIC OpenAI ChatCompletions APIと互換性のある形式です。
# MAGIC - `type`: メッセージタイプ（"message"）
# MAGIC - `role`: 役割（"user", "assistant", "system"）
# MAGIC - `content`: メッセージ内容（テキストや画像など）

# COMMAND ----------

import mlflow
from multi_agent_report_app import agent

# レポート作成のリクエスト
question = "RAGに関する技術レポートを書いてください"

# ResponsesAgentRequestの形式
# OpenAI互換のメッセージ形式
input_data = {
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": question}]
        }
    ]
}

# エージェントを実行
# 内部で research → outline → writer → review と順次実行される
response = agent.predict(input_data)

# 最終レポートを取得
final_report = response.output[0].content[0]["text"]

print("質問：", question)
print("\n最終レポート：")
print(final_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: ワークフローの可視化
# MAGIC
# MAGIC マルチエージェントシステムの構造を可視化します。
# MAGIC
# MAGIC ### 可視化で確認できること
# MAGIC - スーパーバイザーと各エージェントの関係
# MAGIC - 条件分岐の構造
# MAGIC - 処理の循環パターン（エージェント → スーパーバイザー → 次のエージェント）

# COMMAND ----------

from IPython.display import Image, display
from multi_agent_report_app import graph

try:
    # エージェントのグラフ構造を可視化（Mermaid形式）
    graph_image = graph.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    print("✓ ワークフローの図を表示しました")
except Exception as e:
    print(f"図の表示に失敗しました: {e}")
    print("（この機能は環境によっては動作しない場合があります）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: MLflowへのモデル記録
# MAGIC
# MAGIC ### PyFuncフレーバー
# MAGIC ResponsesAgentはMLflowのPyFuncフレーバーとして記録できます。
# MAGIC
# MAGIC ### メタデータの記録
# MAGIC モデルと一緒に以下の情報も記録します：
# MAGIC - **パラメータ**: エージェント数、スーパーバイザータイプ等
# MAGIC - **タグ**: アーキテクチャタイプ、バージョン等
# MAGIC - **入力例**: スキーマ推論とテスト用

# COMMAND ----------

import mlflow

# MLflowの設定（ローカルTrackingサーバーを想定）
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("agentic_rag_example")

# LangGraphで構築したマルチエージェントシステムをMLflowに記録
with mlflow.start_run(run_name="multi-agent-report-v1") as run:
    # pyfuncフレーバーとして記録
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model="./multi_agent_report_app.py",
        input_example=input_data,
    )

    # パラメータとタグを記録（後で検索・比較しやすくする）
    mlflow.log_param("agent_count", 4)
    mlflow.log_param("supervisor_type", "sequential")
    mlflow.set_tag("architecture", "supervisor")

print("モデルURI:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: モデルのロードと推論テスト
# MAGIC
# MAGIC 記録したモデルを読み込み、推論が正しく動作することを確認します。

# COMMAND ----------

# 記録したモデルを読み込む
loaded = mlflow.pyfunc.load_model(model_info.model_uri)

# テスト推論を実行
response = loaded.predict(input_data)

# 結果を取得
final_report = response["output"][0]["content"][0]["text"]
print("推論結果:")
print(final_report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: MLflow Evaluation - Safety評価
# MAGIC
# MAGIC ### Safety評価とは？
# MAGIC LLMの出力が安全で適切かを評価する指標です。
# MAGIC
# MAGIC ### 評価観点
# MAGIC - 有害なコンテンツの有無
# MAGIC - 偏見や差別的表現
# MAGIC - 不適切な言及
# MAGIC
# MAGIC ### 使用シーン
# MAGIC 本番環境にデプロイする前の安全性チェックとして活用します。

# COMMAND ----------

"""
マルチエージェントシステムをMLflow Evaluationで評価する例です。
まずはSafety（安全性）評価から始めます。
"""
from __future__ import annotations

import mlflow
from mlflow.genai.scorers import Safety

# 1. 評価データセットを用意
dataset = [
    {
        "inputs": {
            "question": "RAGに関するレポートを作成してください",
        },
    },
]

# 2. 予測関数を定義
def predict_wrapper(question: str) -> str:
    """
    MLflow Evaluationから呼び出される予測関数です。

    Args:
        question: レポートのテーマ

    Returns:
        生成されたレポート
    """
    # ResponsesAgentRequestの形式
    input_data = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": question}]
            }
        ]
    }

    response = loaded.predict(input_data)
    final_report = response["output"][0]["content"][0]["text"]
    return final_report

# COMMAND ----------

# 3. Safety評価の実行
with mlflow.start_run():
    results = mlflow.genai.evaluate(
        data=dataset,
        predict_fn=predict_wrapper,
        scorers=[
            Safety(model="openai:/gpt-3.5-turbo"),
        ],
    )

print("安全性評価が完了しました。MLflow UIで結果を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ9: MLflow Evaluation - Guidelinesベースの評価
# MAGIC
# MAGIC ### Guidelinesスコアラーとは？
# MAGIC 自然言語で記述された評価基準に基づいて、LLMが出力を評価する仕組みです。
# MAGIC
# MAGIC ### メリット
# MAGIC - **カスタマイズ性**: ビジネス要件に合わせた独自の評価基準を設定可能
# MAGIC - **可読性**: 評価基準が自然言語なので、非技術者でも理解しやすい
# MAGIC - **柔軟性**: コードを書かずに評価基準を変更できる
# MAGIC
# MAGIC ### 評価基準の例
# MAGIC 1. **トーン**: 丁寧でプロフェッショナルな文体か
# MAGIC 2. **理解しやすさ**: 明確で簡潔な説明か
# MAGIC 3. **禁止トピック**: 特定の内容が含まれていないか

# COMMAND ----------

from mlflow.genai import scorers

"""
ガイドラインベースのLLMスコアラー

ガイドラインは、合格/不合格条件として定義された自然言語基準を定義することで、
評価を迅速かつ容易にカスタマイズできるように設計された強力なスコアラークラスです。
ルール、スタイルガイド、情報の包含/除外への準拠チェックに最適です。

ガイドラインには、ビジネス関係者への説明が容易であるという明確な利点があります
（「アプリがこのルールセットを満たしているかどうかを評価しています」）。
そのため、多くの場合、ドメインエキスパートが直接記述できます。
"""

# 自然言語で評価基準を定義
tone = "回答は終始、丁寧でプロフェッショナルさを保たねばならない。"

easy_to_understand = """回答は明確かつ簡潔な言葉を用い、論理的に構成されなければなりません。
専門用語の使用は避け、使用する場合は説明を加える必要があります。"""

banned_topics = "価格に関する具体的な数値が記載されていないこと"

# Guidelinesスコアラーを作成
tone_scorer = scorers.Guidelines(
    name="tone", 
    model="openai:/gpt-3.5-turbo",
    guidelines=tone
)

easy_to_understand_scorer = scorers.Guidelines(
    name="easy_to_understand", 
    model="openai:/gpt-3.5-turbo",
    guidelines=easy_to_understand
)

banned_topics_scorer = scorers.Guidelines(
    name="banned_topics", 
    model="openai:/gpt-3.5-turbo",
    guidelines=banned_topics
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### トレースベースの評価
# MAGIC
# MAGIC MLflowは、過去の実行トレースを直接評価できます。
# MAGIC これにより、本番環境での実際の出力を事後的に評価することが可能です。

# COMMAND ----------

# 評価の実行例
# 直近のトレースを取得
traces = mlflow.search_traces(
    max_results=1,
)

if traces.empty:
    print("評価対象のトレースが見つかりません。")
    raise SystemExit(1)

# ガイドラインベースの評価を実行
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[tone_scorer, easy_to_understand_scorer, banned_topics_scorer],
)

print("自然言語ベースの評価が完了しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ10: カスタムスコアラー - エージェント網羅性評価
# MAGIC
# MAGIC ### カスタムスコアラーとは？
# MAGIC 独自のロジックで評価を行うスコアラーです。
# MAGIC
# MAGIC ### このスコアラーの目的
# MAGIC マルチエージェントシステムで、期待されるすべてのエージェントが
# MAGIC 実際に呼び出されたかを確認します。
# MAGIC
# MAGIC ### 実装のポイント
# MAGIC - `@scorer`デコレータで関数をスコアラーとして登録
# MAGIC - `Trace`オブジェクトからSpanを検索
# MAGIC - `Feedback`オブジェクトでスコアと理由を返す

# COMMAND ----------

"""
エージェント呼び出しの網羅性を評価するコードベースのカスタムScorerです。

・トレースからSpanType.AGENTのスパンを抽出
・実際に呼ばれたエージェント名のリストを取得
・期待されるエージェントがすべて呼ばれたかを確認
"""
import mlflow
from mlflow.entities import Feedback, Trace, SpanType
from mlflow.genai import scorer
import pandas as pd

@scorer
def agent_coverage(trace: Trace, expectations: dict) -> Feedback:
    """
    想定通りのエージェントが呼ばれているかを評価します。

    評価基準:
    - 期待されるエージェントがすべて呼ばれた場合: スコア1.0
    - 一部のエージェントが欠けている場合: 呼ばれた割合をスコアとする

    Args:
        trace: 評価対象のトレース
        expectations: 期待される動作を定義した辞書

    Returns:
        Feedback: スコアと理由を含む評価結果
    """
    # トレースからエージェントスパンを検索
    agent_spans = trace.search_spans(span_type=SpanType.AGENT)

    # 実際に呼び出されたエージェント名を抽出
    invoked_agents = [span.name for span in agent_spans]

    # 期待されるエージェントのリスト
    expected_agents = expectations.get("expected_agents", [])
    if not expected_agents:
        return Feedback(
            value=1.0,
            rationale="期待値が指定されていないため、評価をスキップしました。"
        )

    # 網羅性を計算（期待されるエージェントのうち何%が呼ばれたか）
    invoked_set = set(invoked_agents)
    expected_set = set(expected_agents)
    covered = invoked_set & expected_set
    coverage_ratio = len(covered) / len(expected_set) if expected_set else 0

    # 詳細な理由を生成
    if coverage_ratio == 1.0:
        rationale = f"期待通り、すべてのエージェント {expected_agents} が呼び出されました。"
    else:
        missing = expected_set - invoked_set
        extra = invoked_set - expected_set
        rationale = (
            f"エージェント網羅率: {coverage_ratio:.0%}\n"
            f"期待: {sorted(expected_agents)}\n"
            f"実際: {sorted(invoked_agents)}\n"
        )
        if missing:
            rationale += f"未呼び出し: {sorted(missing)}\n"
        if extra:
            rationale += f"予期しない呼び出し: {sorted(extra)}"

    return Feedback(
        value=coverage_ratio,
        rationale=rationale.strip()
    )

# トレースから評価データを取得
traces = mlflow.search_traces(
    max_results=1,
)

if traces.empty:
    print("評価対象のトレースが見つかりません。")
    raise SystemExit(1)

# 期待されるエージェントのリストを追加
traces["expectations"] = [{
    "expected_agents": [
        "research_agent",
        "outline_agent",
        "writer_agent",
        "review_agent"
    ]
}] * len(traces)

# 評価を実行
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[agent_coverage],
)

print("エージェント網羅性の評価が完了しました。MLflow UIで結果を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ11: Agent-as-a-Judge評価
# MAGIC
# MAGIC ### Agent-as-a-Judgeとは？
# MAGIC LLMエージェントが評価者となり、複雑な評価基準に基づいて
# MAGIC 他のLLMの出力を評価する手法です。
# MAGIC
# MAGIC ### 他の評価手法との違い
# MAGIC
# MAGIC | 手法 | 評価方法 | 適用範囲 |
# MAGIC |------|----------|----------|
# MAGIC | コードベース | ルールベースチェック | 限定的 |
# MAGIC | Guidelinesベース | セマンティックなチェック | 中程度 |
# MAGIC | Agent-as-a-Judge | 複雑な推論と総合判断 | 広範囲 |
# MAGIC
# MAGIC ### このスコアラーの評価観点
# MAGIC **エージェント間の協調性**を多角的に評価：
# MAGIC 1. リサーチ結果が構成案に適切に反映されているか
# MAGIC 2. 構成案が本文に適切に反映されているか
# MAGIC 3. レビューコメントが本文の内容と整合しているか
# MAGIC 4. 無駄な差し戻しや重複作業が発生していないか

# COMMAND ----------

"""
Agent-based Scorer (aka. Agent-as-a-Judge)を使った評価の例です。

MLflowでは、評価基準を自然言語で記述することで、
エージェントが自動的にその基準に基づいて評価を行います。
"""
import mlflow
from mlflow.genai.judges import make_judge
from typing import Literal

# 自然言語で評価基準を定義
AGENT_COORDINATION_GUIDELINE = """
あなたは、マルチエージェントシステムの「エージェント間の協調性」を評価する役割です。
以下の観点で {{ trace }} を評価し、スコア（1〜5の整数）と理由を返してください。

評価観点:
1. リサーチ結果が構成案に適切に反映されているか
2. 構成案が本文に適切に反映されているか
3. レビューコメントが本文の内容と整合しているか
4. 無意味な差し戻しや重複作業が発生していないか

スコアの基準:
- 5: すべての観点で優れている
- 4: ほとんどの観点で良好だが、軽微な改善点がある
- 3: いくつかの観点で問題があるが、全体としては機能している
- 2: 複数の観点で明確な問題がある
- 1: エージェント間の連携が機能していない
"""

# Agent-as-a-Judgeスコアラーを作成
agent_coordination_scorer = make_judge(
    name="agent_coordination",
    instructions=AGENT_COORDINATION_GUIDELINE,
    feedback_value_type=Literal["5", "4", "3", "2", "1"],
    # トレース全体を分析するため、モデルを指定
    model="openai:/gpt-4o-mini",
)

# トレースから評価データを取得
traces = mlflow.search_traces(
    max_results=1,
)

if traces.empty:
    print("評価対象のトレースが見つかりません。")
    raise SystemExit(1)

# Agent-as-a-Judge評価を実行
results = mlflow.genai.evaluate(
    data=traces,
    scorers=[agent_coordination_scorer],
)

print("エージェント協調性の評価が完了しました。MLflow UIで結果を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC ### このノートブックで学んだこと
# MAGIC
# MAGIC #### アーキテクチャ設計
# MAGIC 1. **スーパーバイザー型マルチエージェント**: 中央制御による効率的なタスク分割
# MAGIC 2. **エージェントの専門化**: 各エージェントが明確な役割を持つ設計
# MAGIC 3. **状態管理**: AgentStateによる情報の受け渡し
# MAGIC 4. **ResponsesAgent**: 標準インターフェースでのラッピング
# MAGIC
# MAGIC #### 評価手法の多様性
# MAGIC 1. **Safety評価**: 安全性の自動チェック
# MAGIC 2. **Guidelinesベース評価**: ビジネスルールの自然言語記述
# MAGIC 3. **カスタムスコアラー**: 独自ロジックによる評価（エージェント網羅性）
# MAGIC 4. **Agent-as-a-Judge**: 複雑な総合評価（協調性）
# MAGIC
# MAGIC #### MLflowの活用
# MAGIC 1. **Tracing**: 全エージェントの処理を詳細に記録
# MAGIC 2. **Model Registry**: バージョン管理とデプロイ準備
# MAGIC 3. **Evaluation**: 多角的な品質評価
# MAGIC 4. **タグとパラメータ**: 実験の整理と検索
# MAGIC
# MAGIC ### スーパーバイザー型の利点
# MAGIC
# MAGIC - **品質向上**: 各段階で専門的な処理が可能
# MAGIC - **デバッグ容易性**: 問題のあるエージェントを特定しやすい
# MAGIC - **拡張性**: 新しいエージェントの追加が容易
# MAGIC - **制御性**: スーパーバイザーで処理フローを柔軟に制御
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC #### 機能拡張
# MAGIC - 並列処理: 独立したエージェントを同時実行
# MAGIC - 動的ルーティング: LLMによる次エージェントの判断
# MAGIC - エラーハンドリング: リトライ機能の追加
# MAGIC - メモリ機能: 過去の実行結果を参照
# MAGIC
# MAGIC #### 評価の強化
# MAGIC - より大規模な評価データセット
# MAGIC - 人間による評価との比較
# MAGIC - A/Bテストによるプロンプト最適化
# MAGIC - 継続的評価パイプラインの構築
# MAGIC
# MAGIC #### 本番環境へのデプロイ
# MAGIC - REST APIとしてのサービング
# MAGIC - バッチ処理パイプラインの構築
# MAGIC - モニタリングとアラート設定
# MAGIC - コスト最適化（モデル選択、キャッシング）
# MAGIC
# MAGIC ### 参考リソース
# MAGIC
# MAGIC - [LangGraph Multi-Agent Systems](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
# MAGIC - [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html)
# MAGIC - [MLflow Evaluation for LLMs](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
# MAGIC - [MLflow ResponsesAgent](https://mlflow.org/docs/latest/llms/deployments/index.html)
# MAGIC - [Agent-as-a-Judge Pattern](https://arxiv.org/abs/2410.10934)

# COMMAND ----------

# MAGIC %md
# MAGIC ---