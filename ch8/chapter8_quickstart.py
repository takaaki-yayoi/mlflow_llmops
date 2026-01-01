# Databricks notebook source
# MAGIC %md
# MAGIC # 第8章 監視と運用 - クイックスタート
# MAGIC 
# MAGIC 本ノートブックでは、第8章で解説した監視・運用機能の動作を確認できます。
# MAGIC 
# MAGIC **前提条件:**
# MAGIC - Databricks Runtime 15.0 ML以上
# MAGIC - OpenAI APIキーがDatabricksシークレットに設定済み

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. セットアップ

# COMMAND ----------

# 必要なパッケージのインストール
# MLflow 3.1以降でtrace.info.token_usageが利用可能
# databricks-agentsはスコアラー機能に必要
%pip install openai "mlflow>=3.1" databricks-agents -q
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import time
from openai import OpenAI
from mlflow.entities import SpanType, Document

# OpenAI APIキーをシークレットから取得
# TODO: 実際の環境に合わせてスコープとキー名を変更してください
import os
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="your-scope", key="openai-api-key")

# エクスペリメントを設定
# TODO: 実際の環境に合わせてパスを変更してください
# 注意: ノートブック名と異なる名前にしてください(同じだとノートブックエクスペリメントになります)
# 例: /Users/your-email@example.com/experiments/chapter8-monitoring
#     /Shared/team-name/experiments/llm-monitoring
experiment_path = "/Shared/experiments/chapter8-monitoring"
mlflow.set_experiment(experiment_path)

# OpenAI自動トレーシングを有効化
mlflow.openai.autolog()

print(f"セットアップ完了: {experiment_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. 効果的なスパン設計(手動トレーシング)
# MAGIC 
# MAGIC 本文 8.1.2 に対応
# MAGIC 
# MAGIC 本番運用で分析しやすいトレースを設計するためのベストプラクティス:
# MAGIC - SpanTypeを適切に設定
# MAGIC - デバッグに役立つ属性を記録
# MAGIC - エラー時のコンテキストを充実

# COMMAND ----------

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str, top_k: int = 3) -> list[dict]:
    """検索スパンの実装例 - 本文8.1.2に対応"""
    span = mlflow.get_current_active_span()
    
    # 入力パラメータを記録
    span.set_inputs({"query": query, "top_k": top_k})
    
    # 検索実行(ダミー実装)
    start_time = time.time()
    results = [
        {"content": "MLflowはMLライフサイクル管理プラットフォームです。", "uri": "docs/mlflow/intro.md"},
        {"content": "トレーシングで実行フローを可視化できます。", "uri": "docs/mlflow/tracing.md"},
        {"content": "スコアラーで品質を継続監視できます。", "uri": "docs/mlflow/scorers.md"},
    ][:top_k]
    search_time = time.time() - start_time
    
    # デバッグに役立つ属性を記録
    span.set_attributes({
        "retriever.search_time_ms": search_time * 1000,
        "retriever.num_results": len(results),
        "retriever.index_name": "demo_index",
    })
    
    # 出力をMLflow Document形式で記録
    span.set_outputs([
        Document(page_content=doc["content"], metadata={"doc_uri": doc["uri"]})
        for doc in results
    ])
    
    return results

@mlflow.trace(span_type=SpanType.LLM)
def generate_with_context(query: str, context: list[dict]) -> str:
    """LLMスパンの実装例 - エラー時のコンテキスト記録"""
    span = mlflow.get_current_active_span()
    
    context_text = "\n".join([doc["content"] for doc in context])
    prompt = f"以下のコンテキストに基づいて質問に答えてください。\n\nコンテキスト:\n{context_text}\n\n質問: {query}"
    
    span.set_attributes({
        "llm.model": "gpt-4o",
        "llm.prompt_length": len(prompt),
        "llm.context_docs": len(context),
    })
    
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        span.set_attributes({
            "error.type": type(e).__name__,
            "error.message": str(e),
        })
        raise

@mlflow.trace(span_type=SpanType.CHAIN, name="rag_pipeline")
def rag_query(query: str) -> str:
    """RAGパイプライン全体のトレース"""
    # 検索
    docs = retrieve_documents(query)
    # 生成
    response = generate_with_context(query, docs)
    return response

# 実行
result = rag_query("MLflowのトレーシング機能について教えてください")
print(result)
print("\nMLflow UIでトレースを確認すると、rag_pipeline > retrieve_documents, generate_with_contextの階層が表示されます")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2.1 トレースへのメタデータ追加
# MAGIC 
# MAGIC 本文 8.1.6 に対応
# MAGIC 
# MAGIC 本番環境でのデバッグと分析を効率化するため、トレースにコンテキスト情報を追加します。

# COMMAND ----------

import uuid

@mlflow.trace
def handle_request_with_metadata(message: str, user_id: str = "user_123"):
    """メタデータ付きのリクエスト処理"""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    request_id = str(uuid.uuid4())
    
    # トレースにコンテキスト情報を追加
    mlflow.update_current_trace(tags={
        # 標準タグ - MLflow UIでフィルタリングに使用可能
        "mlflow.trace.user": user_id,
        "mlflow.trace.session": session_id,
        "mlflow.trace.request_id": request_id,
        # カスタムタグ - 環境やバージョン情報
        "environment": "development",
        "app_version": "1.0.0",
        "feature_flags": "new_model_enabled",
    })
    
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}]
    )
    return response.choices[0].message.content

# 実行
result = handle_request_with_metadata("MLflowのメタデータ機能について教えてください")
print(result[:300] + "...")
print("\nMLflow UIのトレース詳細 → Tagsセクションでメタデータを確認できます")
print("mlflow.search_traces()でタグによるフィルタリングも可能です")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. LLM呼び出し(自動トレーシング)
# MAGIC 
# MAGIC 本文 8.2.2 に対応

# COMMAND ----------

client = OpenAI()

# 通常通りAPIを呼び出すだけで、トークン使用量が自動記録される
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "フランスの首都はどこですか?"}
    ]
)

print(response.choices[0].message.content)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. トークン使用量の確認
# MAGIC 
# MAGIC 本文 8.2.2 に対応

# COMMAND ----------

# トレースからトークン使用量を取得
last_trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id=last_trace_id)

token_usage = trace.info.token_usage
print(f"入力トークン: {token_usage.get('input_tokens')}")
print(f"出力トークン: {token_usage.get('output_tokens')}")
print(f"合計トークン: {token_usage.get('total_tokens')}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4.1 コスト計算
# MAGIC 
# MAGIC 本文 8.2.3 に対応

# COMMAND ----------

# カスタム実装例: コスト計算関数
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """モデルとトークン数からコストを計算(カスタム実装)"""
    # 注意: 料金は2025年12月時点の参考値です
    # 最新の料金は各プロバイダーの公式サイトで確認してください
    pricing = {
        # OpenAI Models (per 1K tokens)
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        # Anthropic Models
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
    }
    
    if model not in pricing:
        return 0.0
    
    p = pricing[model]
    return (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]

# トレースから取得したトークン使用量でコスト計算
cost = calculate_cost(
    model="gpt-4o",
    input_tokens=token_usage.get('input_tokens', 0),
    output_tokens=token_usage.get('output_tokens', 0)
)
print(f"推定コスト: ${cost:.6f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4.2 トレースのDelta Tableアーカイブ
# MAGIC 
# MAGIC 本文 8.1.5 に対応
# MAGIC 
# MAGIC 大規模なトレースデータをUnity Catalog Delta Tableに自動同期し、SQLで分析可能にします。
# MAGIC 
# MAGIC **注意:** 実行前に、Unity Catalogのカタログ・スキーマ・テーブル名を環境に合わせて変更してください。

# COMMAND ----------

from mlflow.tracing.archival import enable_databricks_trace_archival, disable_databricks_trace_archival

# アーカイブの設定
# TODO: 実際の環境に合わせてカタログ・スキーマ・テーブル名を変更してください
CATALOG = "your_catalog"
SCHEMA = "your_schema"
TABLE = "trace_archive"

# 現在のエクスペリメントIDを取得
experiment = mlflow.get_experiment_by_name(experiment_path)
experiment_id = experiment.experiment_id

# アーカイブを有効化(コメントを外して実行)
# enable_databricks_trace_archival(
#     delta_table_fullname=f"{CATALOG}.{SCHEMA}.{TABLE}",
#     experiment_id=experiment_id,
# )
# print(f"アーカイブ有効化: {CATALOG}.{SCHEMA}.{TABLE}")
# print("約15分間隔でトレースがDelta Tableに同期されます")

# アーカイブを停止する場合
# disable_databricks_trace_archival(experiment_id=experiment_id)
# print("アーカイブ停止")

print("アーカイブを有効にするには、上記のコメントを外して実行してください")
print("有効化後、SQLでトレースデータを分析できます:")
print(f"  SELECT * FROM {CATALOG}.{SCHEMA}.{TABLE} LIMIT 10")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. フィードバックの記録
# MAGIC 
# MAGIC 本文 8.3.3 に対応

# COMMAND ----------

from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType

trace_id = last_trace_id  # 上で取得したトレースIDを使用

# フィードバックをトレースに記録
mlflow.log_feedback(
    trace_id=trace_id,
    name="user_satisfaction",
    value=True,
    rationale="ユーザーが役立ったと評価",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="user@example.com"
    )
)

# 評価スコアを記録
mlflow.log_feedback(
    trace_id=trace_id,
    name="quality_rating",
    value=4,  # 1-5スケール
    rationale="概ね良いが、もう少し詳しく説明してほしい",
    source=AssessmentSource(
        source_type=AssessmentSourceType.HUMAN,
        source_id="user@example.com"
    )
)

print("フィードバック記録完了")
print("MLflow UIのトレース詳細 → Assessmentsタブで確認できます")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. スコアラーの登録・開始
# MAGIC 
# MAGIC 本文 8.3.4 に対応

# COMMAND ----------

from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig

# 1. スコアラーを登録(名前はエクスペリメント内で一意)
safety_scorer = Safety().register(name="safety_check")

# 2. サンプリング設定を指定して監視を開始
safety_scorer = safety_scorer.start(
    sampling_config=ScorerSamplingConfig(sample_rate=1.0)  # 100%のトレースを評価
)

# カスタムガイドラインも同様に登録・開始
tone_scorer = Guidelines(
    name="professional_tone",
    guidelines="回答は専門的で丁寧なトーンである必要があります。"
).register(name="tone_check")

tone_scorer = tone_scorer.start(
    sampling_config=ScorerSamplingConfig(sample_rate=0.5)  # 50%をサンプリング
)

print("スコアラー登録完了")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. 登録済みスコアラーの確認
# MAGIC 
# MAGIC 本文 8.3.4 に対応

# COMMAND ----------

from mlflow.genai.scorers import list_scorers

# 登録済みスコアラーの一覧
all_scorers = list_scorers()
for s in all_scorers:
    print(f"{s.name}: sample_rate={s.sample_rate}")

print("\nMLflow UIのエクスペリメント → Scorersタブで確認できます")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. クリーンアップ(オプション)

# COMMAND ----------

from mlflow.genai.scorers import delete_scorer

# 登録したスコアラーを削除する場合は以下のコメントを外してください
# delete_scorer(name="safety_check")
# delete_scorer(name="tone_check")
# print("スコアラー削除完了")
