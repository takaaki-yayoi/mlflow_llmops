# Databricks notebook source
# MAGIC %md
# MAGIC # エージェント型RAGシステムの構築 - MLflow & LangGraph Tutorial
# MAGIC
# MAGIC ## 概要
# MAGIC このノートブックでは、LangGraphとMLflowを組み合わせて、自律的に判断し行動するエージェント型RAG（Retrieval-Augmented Generation）システムを構築する方法を学びます。
# MAGIC
# MAGIC ### 学習内容
# MAGIC 1. ベクトルデータベース（Chroma）の構築と文書の格納
# MAGIC 2. LangGraphによるマルチノードワークフローの設計
# MAGIC 3. エージェント型RAGの実装（質問のルーティング、検索、再試行ロジック）
# MAGIC 4. MLflow Tracingによる処理の可観測性向上
# MAGIC 5. MLflow Evaluationによる自動評価
# MAGIC 6. モデルの記録と本番環境へのデプロイ準備
# MAGIC
# MAGIC ### エージェント型RAGとは？
# MAGIC 従来のRAGは「必ず検索→回答」という固定フローでしたが、エージェント型RAGは：
# MAGIC - 質問内容に応じて検索の要否を自動判断
# MAGIC - 検索結果の品質を評価し、不十分な場合は質問を改善して再検索
# MAGIC - 複数の処理経路を動的に選択
# MAGIC
# MAGIC これにより、より柔軟で精度の高い回答生成が可能になります。

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 環境セットアップ
# MAGIC
# MAGIC ### 必要なライブラリ
# MAGIC - `mlflow`: 実験管理、モデル記録、トレーシング
# MAGIC - `langchain[openai]`: LangChainとOpenAI連携
# MAGIC - `langgraph`: ワークフローグラフの構築（状態管理、条件分岐）
# MAGIC - `chromadb`: ベクトルデータベース
# MAGIC - `langchain-community`: コミュニティ拡張（ベクトルストア等）
# MAGIC - `langchain-text-splitters`: テキスト分割ツール

# COMMAND ----------

# MAGIC %pip install "mlflow" "langchain[openai]" langgraph chromadb langchain-community langchain-text-splitters

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
# MAGIC ## ステップ3: ベクトルデータベースの構築
# MAGIC
# MAGIC ### RAGにおけるベクトルデータベースの役割
# MAGIC テキストを数値ベクトル（埋め込み）に変換して保存し、意味的に類似した文書を高速に検索できるようにします。
# MAGIC
# MAGIC ### このステップで行うこと
# MAGIC 1. **サンプル文書の準備**: 社内技術文書を想定したテキストデータ
# MAGIC 2. **テキスト分割（チャンキング）**: 長い文書を適切なサイズに分割
# MAGIC 3. **埋め込み生成**: OpenAIの埋め込みモデルでベクトル化
# MAGIC 4. **Chromaへの保存**: ベクトルデータベースに永続化
# MAGIC
# MAGIC ### チャンキングのパラメータ
# MAGIC - `chunk_size=300`: 各チャンクの最大文字数
# MAGIC - `chunk_overlap=50`: チャンク間のオーバーラップ（文脈の連続性を保つため）

# COMMAND ----------

"""
RAGで使うベクトルデータベース（Chroma）を準備するスクリプトです。
・社内技術文書などを想定したサンプルテキストをいくつか登録します。
・LangChainのTextSplitterでチャンクに分割し、Chromaに保存します。
"""

from __future__ import annotations

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def build_vectorstore(persist_dir: str = "./chroma_store") -> Chroma:
    """
    ベクトルデータベースを構築する関数

    Args:
        persist_dir: ベクトルデータベースの保存先ディレクトリ

    Returns:
        構築されたChromaベクトルストア
    """
    # サンプルの社内技術文書（本番ではファイルやデータベースから読み込む）
    raw_docs = [
        Document(
            page_content=(
                "MLflowは機械学習および生成AIの実験管理、モデル管理、"
                "そして可観測性を提供するオープンソースのプラットフォームです。"
            ),
            metadata={"source": "mlflow_intro"},
        ),
        Document(
            page_content=(
                "RAG（Retrieval-Augmented Generation）は、外部のナレッジベースから"
                "関連文書を検索し、その内容をもとに回答を生成する手法です。"
            ),
            metadata={"source": "rag_intro"},
        ),
        Document(
            page_content=(
                "Chromaはシンプルに使えるベクトルデータベースであり、"
                "Pythonからの利用に適しています。我が社ではRAG用のデフォルトDBとして使用されています。"
            ),
            metadata={"source": "chroma_intro"},
        ),
    ]

    # テキストをチャンクに分割
    # RecursiveCharacterTextSplitter: 段落、文、単語の順で自然な区切りを探す
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,      # 各チャンクの最大文字数
        chunk_overlap=50,    # チャンク間のオーバーラップ（文脈保持のため）
    )
    splits = splitter.split_documents(raw_docs)

    # OpenAIの埋め込みモデルを使ってChromaを構築
    # 埋め込み: テキストを高次元ベクトルに変換（意味的類似性の計算に使用）
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    # ディスクに永続化（プログラム終了後も利用可能）
    vectordb.persist()
    return vectordb

# ベクトルストアを構築
build_vectorstore()
print("Chromaストアを作成しました。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: エージェント型RAGアプリケーションの実装
# MAGIC
# MAGIC ### LangGraphとは？
# MAGIC 状態を持つマルチエージェントワークフローを構築するためのフレームワークです。
# MAGIC 複雑な処理フローを「ノード（処理単位）」と「エッジ（遷移）」のグラフとして表現します。
# MAGIC
# MAGIC ### このアプリケーションの構成
# MAGIC
# MAGIC #### 5つのノード
# MAGIC 1. **router**: 質問を分析し、検索の要否を判定
# MAGIC 2. **retrieve**: ベクトルDBから関連文書を検索
# MAGIC 3. **check**: 検索結果の品質を評価
# MAGIC 4. **rewrite**: 質問を検索に適した形に改善
# MAGIC 5. **answer**: 最終回答を生成
# MAGIC
# MAGIC #### 処理フロー
# MAGIC ```
# MAGIC START → router → [検索必要？]
# MAGIC              ├─ YES → retrieve → check → [品質十分？]
# MAGIC              │                        ├─ YES → answer → END
# MAGIC              │                        └─ NO → rewrite → router（再試行）
# MAGIC              └─ NO → answer → END（LLMの知識のみで回答）
# MAGIC ```
# MAGIC
# MAGIC ### MLflow Tracingの活用
# MAGIC 各ノードの処理内容、入出力、実行時間をトレースとして記録し、デバッグや性能分析を容易にします。

# COMMAND ----------

# MAGIC %%writefile ./agentic_rag_app.py
# MAGIC """
# MAGIC 最小構成のエージェント型RAGアプリケーションの例です。
# MAGIC
# MAGIC ・LangGraphで次の5ノードを持つグラフを作成します:
# MAGIC   - router: 質問を見て「検索すべきかどうか」を決める
# MAGIC   - retrieve: Chromaから関連文書を検索する
# MAGIC   - check: 検索結果が十分かどうかを判定する
# MAGIC   - rewrite: 質問を少し言い換える
# MAGIC   - answer: コンテキスト＋質問から最終回答を生成する
# MAGIC
# MAGIC ・MLflow Tracingで、各ノードの処理をスパンとして記録します。
# MAGIC """
# MAGIC
# MAGIC from __future__ import annotations
# MAGIC
# MAGIC from typing import Literal, Dict, Any
# MAGIC
# MAGIC import mlflow
# MAGIC from mlflow.entities import SpanType
# MAGIC
# MAGIC from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# MAGIC from langchain_community.vectorstores import Chroma
# MAGIC from langchain_core.messages import HumanMessage, AIMessage
# MAGIC from langgraph.graph import StateGraph, MessagesState, START, END
# MAGIC
# MAGIC from typing import TypedDict, Annotated
# MAGIC from langgraph.graph.message import add_messages
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # 事前準備
# MAGIC # ==========
# MAGIC
# MAGIC # LangChainのLLM（温度0で決定的な出力）
# MAGIC llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
# MAGIC
# MAGIC # Chromaベクトルストア（前のステップで作成したものを読み込み）
# MAGIC PERSIST_DIR = "./chroma_store"
# MAGIC vectordb = Chroma(
# MAGIC     embedding_function=OpenAIEmbeddings(),
# MAGIC     persist_directory=PERSIST_DIR,
# MAGIC )
# MAGIC retriever = vectordb.as_retriever()
# MAGIC
# MAGIC # LangChain/MLflowの自動ロギングを有効化
# MAGIC # LLM呼び出しやチェーン実行を自動的にMLflowに記録
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # 検索関数
# MAGIC # ==========
# MAGIC
# MAGIC def retrieve_docs(query: str) -> str:
# MAGIC     """
# MAGIC     質問文から関連文書を検索し、テキストを1つの文字列として返します。
# MAGIC     
# MAGIC     Args:
# MAGIC         query: 検索クエリ（ユーザーの質問）
# MAGIC     
# MAGIC     Returns:
# MAGIC         検索結果の文書を改行で連結した文字列
# MAGIC     """
# MAGIC     docs = retriever.invoke(query)
# MAGIC     return "\n\n".join([d.page_content for d in docs])
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # カスタム状態の定義
# MAGIC # ==========
# MAGIC
# MAGIC class AgenticRAGState(TypedDict):
# MAGIC     """
# MAGIC     LangGraphのワークフロー全体で共有される状態
# MAGIC     
# MAGIC     Attributes:
# MAGIC         messages: 会話履歴（質問、中間メッセージ、回答など）
# MAGIC         route: routerノードの判定結果（'rag' or 'llm_only'）
# MAGIC         context: 検索結果のテキスト
# MAGIC         check_result: checkノードの判定結果（'answer' or 'rewrite'）
# MAGIC     """
# MAGIC     messages: Annotated[list, add_messages]  # 自動的にメッセージを追加
# MAGIC     route: str
# MAGIC     context: str
# MAGIC     check_result: str
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # ノード定義
# MAGIC # ==========
# MAGIC
# MAGIC def router_node(state: AgenticRAGState) -> Dict[str, Any]:
# MAGIC     """
# MAGIC     【ノード1: ルーター】
# MAGIC     質問を見て「検索すべきかどうか」を判定するノードです。
# MAGIC     
# MAGIC     判定ロジック:
# MAGIC     - 社内文書に関する質問 → 'rag'（検索が必要）
# MAGIC     - 一般常識や外部知識の質問 → 'llm_only'（LLMの知識のみで回答）
# MAGIC     """
# MAGIC     question = state["messages"][-1].content
# MAGIC
# MAGIC     prompt = (
# MAGIC         "次の質問に答えるために、社内の技術文書を検索した方がよいかを判定してください。\n"
# MAGIC         "検索した方がよい場合は 'rag'、不要な場合は 'llm_only' とだけ答えてください。\n\n"
# MAGIC         f"質問: {question}"
# MAGIC     )
# MAGIC     res = llm.invoke([HumanMessage(content=prompt)])
# MAGIC     decision = res.content.strip().lower()
# MAGIC
# MAGIC     # トレースにルート情報をタグとして残す（後で分析しやすくするため）
# MAGIC     mlflow.update_current_trace(tags={"route_decision": decision})
# MAGIC
# MAGIC     return {"messages": [AIMessage(content=f"[route={decision}]")], "route": decision}
# MAGIC
# MAGIC
# MAGIC def retrieve_node(state: AgenticRAGState) -> Dict[str, Any]:
# MAGIC     """
# MAGIC     【ノード2: 検索】
# MAGIC     Chromaベクトルデータベースから関連文書を検索するノードです。
# MAGIC     
# MAGIC     処理内容:
# MAGIC     1. ユーザーの質問（最初のメッセージ）を取得
# MAGIC     2. ベクトル検索を実行
# MAGIC     3. 検索結果をstateのcontextに格納
# MAGIC     """
# MAGIC     question = state["messages"][0].content
# MAGIC     context = retrieve_docs(question)
# MAGIC
# MAGIC     # 検索結果をstateに追加
# MAGIC     return {"messages": [AIMessage(content=context)], "context": context}
# MAGIC
# MAGIC
# MAGIC def check_node(state: AgenticRAGState) -> Dict[str, Any]:
# MAGIC     """
# MAGIC     【ノード3: 品質チェック】
# MAGIC     検索結果のテキスト（context）が質問に十分関連しているかどうかを判定します。
# MAGIC     
# MAGIC     判定基準:
# MAGIC     - 関連性が高い → 'answer'（そのまま回答生成へ）
# MAGIC     - 関連性が低い → 'rewrite'（質問を改善して再検索）
# MAGIC     """
# MAGIC     question = state["messages"][0].content
# MAGIC     context = state.get("context", "")
# MAGIC
# MAGIC     prompt = (
# MAGIC         "あなたは、検索で得られた文書が質問に関連しているかどうかを判定する役割です。\n"
# MAGIC         "関連していれば 'yes'、ほとんど関係なければ 'no' とだけ回答してください。\n\n"
# MAGIC         f"質問: {question}\n\n"
# MAGIC         f"検索結果: {context[:2000]}\n"  # 長すぎる場合は最初の2000文字のみ使用
# MAGIC     )
# MAGIC     res = llm.invoke([HumanMessage(content=prompt)])
# MAGIC     score = res.content.strip().lower()
# MAGIC
# MAGIC     decision = "answer" if score == "yes" else "rewrite"
# MAGIC
# MAGIC     # トレースに判定結果をタグとして保存
# MAGIC     mlflow.update_current_trace(tags={"check_decision": decision})
# MAGIC
# MAGIC     # 判定結果をstateに保存
# MAGIC     return {"check_result": decision}
# MAGIC
# MAGIC
# MAGIC def rewrite_node(state: AgenticRAGState) -> Dict[str, Any]:
# MAGIC     """
# MAGIC     【ノード4: 質問の改善】
# MAGIC     質問を検索に適した形に言い換えるノードです。
# MAGIC     
# MAGIC     目的:
# MAGIC     - 曖昧な表現を具体化
# MAGIC     - 検索に適したキーワードを含める
# MAGIC     - 元の意図は保持
# MAGIC     """
# MAGIC     question = state["messages"][0].content
# MAGIC     prompt = (
# MAGIC         "次の質問文を、検索に適した形になるように言い換えてください。\n"
# MAGIC         "ただし、意味や意図は変えないでください。\n\n"
# MAGIC         f"元の質問: {question}"
# MAGIC     )
# MAGIC     res = llm.invoke([HumanMessage(content=prompt)])
# MAGIC
# MAGIC     # 新しい質問をMessagesStateに追加して、再度routerに戻します
# MAGIC     return {"messages": [HumanMessage(content=res.content)]}
# MAGIC
# MAGIC
# MAGIC def answer_node(state: AgenticRAGState) -> Dict[str, Any]:
# MAGIC     """
# MAGIC     【ノード5: 回答生成】
# MAGIC     検索結果のコンテキストと質問を使って最終回答を生成するノードです。
# MAGIC     contextがない場合は、LLMの知識のみで回答します。
# MAGIC     
# MAGIC     2つのモード:
# MAGIC     1. RAGモード: 検索結果に基づいて回答（ハルシネーション防止）
# MAGIC     2. LLMモード: LLMの内部知識のみで回答
# MAGIC     """
# MAGIC     question = state["messages"][0].content
# MAGIC     context = state.get("context", "")
# MAGIC
# MAGIC     if context:
# MAGIC         # RAGルート: コンテキストに基づいて回答
# MAGIC         prompt = (
# MAGIC             "以下のコンテキストに基づいて質問に答えてください。\n"
# MAGIC             "コンテキストに書かれていないことは推測せず、「わかりません」と答えてください。\n"
# MAGIC             "3文以内で、簡潔でわかりやすい日本語で答えてください。\n\n"
# MAGIC             f"質問: {question}\n\n"
# MAGIC             f"コンテキスト:\n{context}"
# MAGIC         )
# MAGIC     else:
# MAGIC         # LLM単体ルート: LLMの知識で直接回答
# MAGIC         prompt = (
# MAGIC             "以下の質問に、あなたの知識に基づいて答えてください。\n"
# MAGIC             "3文以内で、簡潔でわかりやすい日本語で答えてください。\n\n"
# MAGIC             f"質問: {question}"
# MAGIC         )
# MAGIC
# MAGIC     res = llm.invoke([HumanMessage(content=prompt)])
# MAGIC     return {"messages": [res]}
# MAGIC
# MAGIC
# MAGIC # ==========
# MAGIC # グラフ定義
# MAGIC # ==========
# MAGIC
# MAGIC def build_agentic_rag_graph():
# MAGIC     """
# MAGIC     LangGraphでエージェント型RAGのワークフローグラフを組み立てます。
# MAGIC     
# MAGIC     グラフ構造:
# MAGIC     - ノード: 処理単位（router, retrieve, check, rewrite, answer）
# MAGIC     - エッジ: ノード間の遷移（固定エッジと条件付きエッジ）
# MAGIC     - 条件分岐: routerとcheckの判定結果に応じて次のノードを動的に決定
# MAGIC     """
# MAGIC     workflow = StateGraph(AgenticRAGState)
# MAGIC
# MAGIC     # 5つのノードを登録
# MAGIC     workflow.add_node("router", router_node)
# MAGIC     workflow.add_node("retrieve", retrieve_node)
# MAGIC     workflow.add_node("check", check_node)
# MAGIC     workflow.add_node("rewrite", rewrite_node)
# MAGIC     workflow.add_node("answer", answer_node)
# MAGIC
# MAGIC     # 開始点: まずrouterノードから開始
# MAGIC     workflow.add_edge(START, "router")
# MAGIC
# MAGIC     # routerの判定結果に応じて分岐
# MAGIC     def route_decision(state: AgenticRAGState) -> Literal["retrieve", "answer"]:
# MAGIC         """routerの判定結果に基づいて次のノードを決める"""
# MAGIC         route = state.get("route", "rag")
# MAGIC         return "retrieve" if route == "rag" else "answer"
# MAGIC
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "router",
# MAGIC         route_decision,
# MAGIC         {
# MAGIC             "retrieve": "retrieve",  # 検索が必要 → retrieveノードへ
# MAGIC             "answer": "answer",      # 検索不要 → answerノードへ
# MAGIC         },
# MAGIC     )
# MAGIC
# MAGIC     # retrieve後は必ずcheckに進む（固定エッジ）
# MAGIC     workflow.add_edge("retrieve", "check")
# MAGIC
# MAGIC     # checkの判定結果に応じて分岐
# MAGIC     def check_decision(state: AgenticRAGState) -> Literal["answer", "rewrite"]:
# MAGIC         """checkの判定結果に基づいて次のノードを決める"""
# MAGIC         return state.get("check_result", "answer")
# MAGIC
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "check",
# MAGIC         check_decision,
# MAGIC         {
# MAGIC             "answer": "answer",    # 品質OK → 回答生成へ
# MAGIC             "rewrite": "rewrite",  # 品質不足 → 質問改善へ
# MAGIC         },
# MAGIC     )
# MAGIC
# MAGIC     # rewrite後は再度routerへ戻る（再試行ループ）
# MAGIC     workflow.add_edge("rewrite", "router")
# MAGIC
# MAGIC     # answer後は終了
# MAGIC     workflow.add_edge("answer", END)
# MAGIC
# MAGIC     # グラフをコンパイルして実行可能な状態にする
# MAGIC     return workflow.compile()
# MAGIC
# MAGIC
# MAGIC # グラフを構築
# MAGIC graph = build_agentic_rag_graph()
# MAGIC
# MAGIC # MLflowのModels from Codeパターンで登録できるようにする
# MAGIC mlflow.models.set_model(graph)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: エージェント型RAGの実行とテスト
# MAGIC
# MAGIC 構築したエージェントに質問を投げて、実際の動作を確認します。
# MAGIC
# MAGIC ### 実行フロー
# MAGIC 1. 質問をメッセージ形式で準備
# MAGIC 2. `graph.invoke()`で実行
# MAGIC 3. 最終状態から回答を取得
# MAGIC 4. MLflow Tracingで処理の詳細を記録

# COMMAND ----------

# ==========
# エージェント型RAGの実行
# ==========
import mlflow
from agentic_rag_app import graph

# テスト用の質問
question = "Chromaの弊社での位置付けを教えて"

# トレースに質問内容をタグとして保存（後から分析しやすくするため）
mlflow.update_current_trace(tags={"question": question})

# LangGraphに渡す初期State（messagesにユーザー質問を入れる）
input_example = {"messages": [{"role": "user", "content": question}]}

# graph.invokeで実行し、最終状態を取得
# 内部で router → retrieve → check → answer と処理が進む
final_state = graph.invoke(input_example)

# 最後のメッセージを最終回答とみなす
last_msg = final_state["messages"][-1]
print("質問:", question)
print("回答:", last_msg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: ワークフローの可視化
# MAGIC
# MAGIC LangGraphの強力な機能の1つが、グラフ構造の可視化です。
# MAGIC ノード間の関係、条件分岐、ループ構造を視覚的に確認できます。
# MAGIC
# MAGIC ### 可視化のメリット
# MAGIC - ワークフローの全体像を把握
# MAGIC - デバッグ時に処理フローを追跡
# MAGIC - チームメンバーとの共有・ドキュメント化

# COMMAND ----------

from IPython.display import Image, display

try:
    # エージェントのグラフ構造を可視化（Mermaid形式で描画）
    graph_image = graph.get_graph().draw_mermaid_png()
    display(Image(graph_image))
    print("✓ ワークフローの図を表示しました")
except Exception as e:
    print(f"図の表示に失敗しました: {e}")
    print("（この機能は環境によっては動作しない場合があります）")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: MLflowへのモデル記録
# MAGIC
# MAGIC ### LangChainフレーバー
# MAGIC MLflowは、LangChainで構築したアプリケーションを専用のフレーバーとして記録できます。
# MAGIC これにより、依存関係、入力スキーマ、実行環境がすべて保存されます。
# MAGIC
# MAGIC ### 記録内容
# MAGIC - グラフの定義（agentic_rag_app.py）
# MAGIC - 依存パッケージ（自動推論）
# MAGIC - 入力例（スキーマ推論用）
# MAGIC - 実行環境の情報

# COMMAND ----------

import mlflow

# MLflowの設定（ローカルTrackingサーバーを想定）
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("agentic_rag_example")

# LangGraphで構築したグラフをMLflowに記録
with mlflow.start_run(run_name="my-agentic-rag") as run:
    # LangChainフレーバーとして記録
    model_info = mlflow.langchain.log_model(
        lc_model="./agentic_rag_app.py",  # LangGraphアプリのコード
        artifact_path="model",             # モデルの保存先
        input_example=input_example,       # 入力スキーマ推論用
    )

print("モデルURI:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: モデルのロードと推論テスト
# MAGIC
# MAGIC 記録したモデルを読み込み、推論が正しく動作することを確認します。
# MAGIC
# MAGIC ### PyFuncとして読み込むメリット
# MAGIC - 統一されたインターフェース（predict()メソッド）
# MAGIC - フレームワーク非依存のデプロイが可能
# MAGIC - REST APIとして簡単にサービング可能

# COMMAND ----------

# 記録したモデルを読み込む
loaded = mlflow.pyfunc.load_model(model_info.model_uri)

# テスト推論を実行
result = loaded.predict(input_example)
print("推論結果:", result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ9: MLflow Evaluationによる自動評価
# MAGIC
# MAGIC ### MLflow Evaluationとは？
# MAGIC モデルの品質を自動的に評価する機能です。LLM-as-a-Judgeパターンを使用し、
# MAGIC 別のLLMが回答の品質を採点します。
# MAGIC
# MAGIC ### 評価指標
# MAGIC - **Correctness**: 回答の正しさ（期待回答との一致度）
# MAGIC - **RetrievalSufficiency**: 検索結果の十分性（質問に答えるのに十分な情報があるか）
# MAGIC
# MAGIC ### 評価データセット
# MAGIC 質問と期待回答のペアを用意し、モデルの出力と比較します。

# COMMAND ----------

"""
エージェント型RAGアプリケーションをMLflow Evaluationで評価する例です。
・小さな評価データセットを用意し、
・エージェント型RAG関数をpredict_fnとして渡し、
・LLM-as-a-Judgeのスコア（正しさ・関連性）を計算します。
"""
from __future__ import annotations

import mlflow
from mlflow.genai.scorers import Correctness, RetrievalSufficiency

# 1. 評価データセットを用意
# 各エントリは入力（質問）と期待される出力のペア
dataset = [
    {
        "inputs": {
            "question": "RAGとは何か、簡単に説明してください。",
        },
        "expectations": {
            "expected_response": "外部の文書を検索して、その内容に基づいて回答を生成する仕組みであること",
        },
    },
    {
        "inputs": {
            "question": "Chromaの弊社での位置付けを教えて",
        },
        "expectations": {
            "expected_response": "Chromaは、弊社ではRAG用のデフォルトDBとして位置付けられています",
        },
    },
]

# 2. エージェント型RAGを呼び出すpredict_wrapperを定義
def predict_wrapper(question: str) -> str:
    """
    MLflow Evaluationから呼び出される予測関数です。

    Args:
        question: ユーザーの質問文

    Returns:
        エージェント型RAGからの回答テキスト
    """
    # トレースに質問内容をタグとして保存
    mlflow.update_current_trace(tags={"question": question})

    # LangGraphに渡す初期State
    input_example = {"messages": [{"role": "user", "content": question}]}

    # graph.invokeで実行し、最終状態を取得
    final_state = loaded.predict(input_example)

    # 最後のメッセージを最終回答とみなす
    last_msg = final_state[0]["messages"][-1]
    return last_msg

# 3. Evaluationの実行
# 評価結果はMLflowに自動的に記録されます
with mlflow.start_run():
    results = mlflow.genai.evaluate(
        data=dataset,                  # 評価データセット
        predict_fn=predict_wrapper,    # 予測関数
        scorers=[                      # 評価指標のリスト
            Correctness(),             # 正しさの評価
            RetrievalSufficiency(),    # 検索の十分性評価
        ],
    )

print("評価が完了しました。MLflow UIで結果を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ10: モデルレジストリへの登録
# MAGIC
# MAGIC ### モデルレジストリとは？
# MAGIC 本番環境へのデプロイ準備として、モデルに名前を付けて一元管理する仕組みです。
# MAGIC
# MAGIC ### メリット
# MAGIC - バージョン管理: 複数バージョンの並行管理
# MAGIC - エイリアス設定: champion, staging等のラベル付け
# MAGIC - デプロイ追跡: どのバージョンが本番稼働中かを記録
# MAGIC - チーム共有: モデルの中央リポジトリとして機能

# COMMAND ----------

# モデルをモデルレジストリに登録
# 登録すると、バージョン番号が自動的に付与されます
mlflow.register_model(
    model_uri=model_info.model_uri, 
    name="agentic-rag-model"
)

print("モデルを 'agentic-rag-model' として登録しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC ### このノートブックで学んだこと
# MAGIC
# MAGIC 1. **ベクトルデータベース構築**: Chromaを使った文書の埋め込みと保存
# MAGIC 2. **LangGraphによるワークフロー設計**: 複雑な処理フローをグラフで表現
# MAGIC 3. **エージェント型RAG**: 動的な判断と再試行を含む高度なRAGシステム
# MAGIC 4. **MLflow Tracing**: 処理の可観測性向上とデバッグ支援
# MAGIC 5. **MLflow Evaluation**: LLM-as-a-Judgeによる自動評価
# MAGIC 6. **モデル管理**: LangChainモデルの記録、バージョン管理、デプロイ準備
# MAGIC
# MAGIC ### エージェント型RAGの利点
# MAGIC
# MAGIC - **適応的な検索**: 質問に応じて検索の要否を自動判断
# MAGIC - **品質管理**: 検索結果の妥当性を評価し、必要に応じて再試行
# MAGIC - **柔軟性**: 複数の処理経路を持ち、状況に応じた最適な回答生成
# MAGIC - **可観測性**: MLflow Tracingで全処理を追跡可能
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC - より大規模な文書コレクションでの実験
# MAGIC - 他のベクトルデータベース（Pinecone, Weaviate等）の試用
# MAGIC - カスタム評価指標の追加
# MAGIC - 本番環境へのデプロイ（REST API、バッチ処理等）
# MAGIC - A/Bテストによる異なるプロンプト戦略の比較
# MAGIC - エラーハンドリングとフォールバック機能の強化
# MAGIC
# MAGIC ### 参考リソース
# MAGIC
# MAGIC - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
# MAGIC - [MLflow Tracing](https://mlflow.org/docs/latest/llms/tracing/index.html)
# MAGIC - [MLflow Evaluation for LLMs](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
# MAGIC - [Chroma Vector Database](https://docs.trychroma.com/)