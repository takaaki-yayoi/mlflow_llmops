# Databricks notebook source
# MAGIC %md
# MAGIC # 第8章 監視と運用 - サンプルノートブック
# MAGIC 
# MAGIC 第8章で解説している監視・運用機能の実践的なサンプルコードです。
# MAGIC 
# MAGIC ## 内容
# MAGIC 1. セットアップ
# MAGIC 2. トレースベース監視 (8.1)
# MAGIC 3. コスト可視化 (8.2)
# MAGIC 4. 品質追跡 (8.3)
# MAGIC 5. クリーンアップ
# MAGIC 
# MAGIC ## 前提条件
# MAGIC - Databricks Runtime 15.4 LTS ML以降
# MAGIC - MLflow 3.x (Databricks Runtime MLに含まれる)
# MAGIC - Unity Catalog対応ワークスペース
# MAGIC - OpenAI APIキー (Databricksシークレットに格納)

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. セットアップ

# COMMAND ----------

# 設定変数（環境に合わせて変更）
CATALOG = "my_catalog"
SCHEMA = "my_schema"
TRACE_TABLE = "archived_traces"
SERVICE_NAME = "chapter8-demo"

# OpenAI APIキー（Databricksシークレットから取得）
# OPENAI_API_KEY = dbutils.secrets.get(scope="llm-keys", key="openai-api-key")

# COMMAND ----------

import mlflow
import os
from datetime import datetime
import uuid

# 非同期ログ記録を有効化
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
os.environ["OTEL_SERVICE_NAME"] = SERVICE_NAME

# エクスペリメントの設定
experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/chapter8_demo"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

print(f"エクスペリメント: {experiment_name}")
print(f"エクスペリメントID: {experiment_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. トレースベース監視 (8.1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 効果的なスパン設計 (8.1.2)

# COMMAND ----------

from mlflow.entities import SpanType
import time

@mlflow.trace(span_type=SpanType.CHAIN)
def rag_pipeline(query: str) -> str:
    """RAGパイプラインのサンプル実装"""
    
    # 検索スパン
    context = retrieve_documents(query)
    
    # 生成スパン
    response = generate_response(query, context)
    
    return response

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_documents(query: str) -> list:
    """検索スパンの例"""
    span = mlflow.get_current_active_span()
    
    start_time = time.time()
    # 実際の検索処理（ダミー）
    documents = [
        {"content": "MLflowはMLOpsプラットフォームです。", "doc_uri": "doc://mlflow/intro"},
        {"content": "トレーシングで可観測性を向上できます。", "doc_uri": "doc://mlflow/tracing"},
    ]
    search_time = time.time() - start_time
    
    # デバッグに役立つ属性を記録
    span.set_attributes({
        "retriever.search_time_ms": search_time * 1000,
        "retriever.num_results": len(documents),
        "retriever.index_name": "demo_index",
    })
    
    return documents

@mlflow.trace(span_type=SpanType.LLM)
def generate_response(query: str, context: list) -> str:
    """LLM呼び出しスパンの例"""
    span = mlflow.get_current_active_span()
    
    # 実際のLLM呼び出し（ダミー）
    response = f"クエリ '{query}' に対する回答: コンテキストに基づいて回答します。"
    
    # トークン使用量を記録（ダミー値）
    span.set_attributes({
        "llm.model": "gpt-4o-mini",
        "llm.temperature": 0.7,
        "llm.input_tokens": 150,
        "llm.output_tokens": 50,
    })
    
    return response

# 実行
result = rag_pipeline("MLflowのトレーシング機能について教えてください")
print(f"結果: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 トレースへのメタデータ追加 (8.1.6)

# COMMAND ----------

@mlflow.trace
def handle_request_with_metadata(message: str, user_id: str = "user_123"):
    """メタデータ付きリクエスト処理"""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    request_id = str(uuid.uuid4())
    
    # トレースにメタデータを追加
    mlflow.update_current_trace(tags={
        # 標準タグ
        "mlflow.trace.user": user_id,
        "mlflow.trace.session": session_id,
        "mlflow.trace.request_id": request_id,
        # カスタムタグ
        "environment": "development",
        "app_version": "1.0.0",
        "feature_flags": "new_model_enabled",
    })
    
    # RAGパイプラインを呼び出し
    response = rag_pipeline(message)
    
    return response

# 実行
result = handle_request_with_metadata(
    message="トレーシングのベストプラクティスは？",
    user_id="demo_user"
)
print(f"結果: {result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 トレースのDelta Tableアーカイブ (8.1.5)
# MAGIC 
# MAGIC 大規模トラフィック向けに、トレースをDelta Tableにアーカイブする設定例です。

# COMMAND ----------

# from mlflow.tracing.archival import enable_databricks_trace_archival

# アーカイブの有効化（コメントアウト - 実行時はコメント解除）
# enable_databricks_trace_archival(
#     delta_table_fullname=f"{CATALOG}.{SCHEMA}.{TRACE_TABLE}",
#     experiment_id=experiment_id,
# )

print(f"アーカイブ先: {CATALOG}.{SCHEMA}.{TRACE_TABLE}")
print("※ 実行するには上記のコメントを解除してください")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. コスト可視化 (8.2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 トークン使用量の確認 (8.2.2)

# COMMAND ----------

# 最新のトレースを取得
last_trace_id = mlflow.get_last_active_trace_id()
if last_trace_id:
    trace = mlflow.get_trace(trace_id=last_trace_id)
    
    print(f"トレースID: {last_trace_id}")
    print(f"ステータス: {trace.info.status}")
    
    # トークン使用量（MLflow 3.1+）
    if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
        token_usage = trace.info.token_usage
        print(f"入力トークン: {token_usage.get('input_tokens', 'N/A')}")
        print(f"出力トークン: {token_usage.get('output_tokens', 'N/A')}")
        print(f"合計トークン: {token_usage.get('total_tokens', 'N/A')}")
    else:
        print("トークン使用量: 自動トレーシング未使用のため取得できません")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 コスト計算 (8.2.3)

# COMMAND ----------

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """モデルとトークン数からコストを計算"""
    # 注意: 料金は頻繁に変更されるため、最新の公式料金を確認してください
    # 2024年12月時点の参考価格（per 1K tokens）
    pricing = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
    }
    
    if model not in pricing:
        return 0.0
    
    p = pricing[model]
    return (input_tokens / 1000) * p["input"] + (output_tokens / 1000) * p["output"]

# 使用例
cost = calculate_cost("gpt-4o-mini", input_tokens=150, output_tokens=50)
print(f"推定コスト: ${cost:.6f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 SQLによるコスト分析 (8.2.4)
# MAGIC 
# MAGIC 以下のクエリは、Delta Tableにアーカイブされたトレースに対して実行します。
# MAGIC アーカイブが有効になっていない場合は、まず2.3節の設定を行ってください。

# COMMAND ----------

# MAGIC %md
# MAGIC ### テーブル存在確認

# COMMAND ----------

FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TRACE_TABLE}"

# テーブルが存在するか確認
try:
    df = spark.table(FULL_TABLE_NAME)
    print(f"✓ テーブル {FULL_TABLE_NAME} が存在します")
    print(f"  レコード数: {df.count()}")
    df.printSchema()
except Exception as e:
    print(f"✗ テーブル {FULL_TABLE_NAME} が存在しないか、アクセスできません")
    print(f"  エラー: {e}")
    print("\n以下のクエリはテーブル作成後に実行してください")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 日次コストサマリー

# COMMAND ----------

# 日次コストサマリークエリ（テーブル存在時のみ実行）
daily_cost_query = f"""
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request,
    SUM(token_usage.input_tokens) as total_input_tokens,
    SUM(token_usage.output_tokens) as total_output_tokens
FROM {FULL_TABLE_NAME}
WHERE timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
GROUP BY DATE(timestamp)
ORDER BY date DESC
"""

print("日次コストサマリークエリ:")
print(daily_cost_query)

# テーブルが存在する場合は以下を実行
# spark.sql(daily_cost_query).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### モデル別コスト内訳

# COMMAND ----------

# モデル別コスト内訳クエリ
model_cost_query = f"""
SELECT 
    tags['cost.model'] as model,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    ROUND(SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) * 100.0 / 
          SUM(SUM(CAST(tags['cost.total_usd'] AS DOUBLE))) OVER (), 2) as cost_percentage
FROM {FULL_TABLE_NAME}
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND tags['cost.model'] IS NOT NULL
GROUP BY tags['cost.model']
ORDER BY total_cost_usd DESC
"""

print("モデル別コスト内訳クエリ:")
print(model_cost_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ユーザー別コストTop 10

# COMMAND ----------

# ユーザー別コストTop 10クエリ
user_cost_query = f"""
SELECT 
    tags['mlflow.trace.user'] as user_id,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request
FROM {FULL_TABLE_NAME}
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND tags['mlflow.trace.user'] IS NOT NULL
GROUP BY tags['mlflow.trace.user']
ORDER BY total_cost_usd DESC
LIMIT 10
"""

print("ユーザー別コストTop 10クエリ:")
print(user_cost_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 時間帯別コスト分析

# COMMAND ----------

# 時間帯別コスト分析クエリ
hourly_cost_query = f"""
SELECT 
    HOUR(timestamp) as hour_of_day,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request
FROM {FULL_TABLE_NAME}
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
GROUP BY HOUR(timestamp)
ORDER BY hour_of_day
"""

print("時間帯別コスト分析クエリ:")
print(hourly_cost_query)

# COMMAND ----------

# MAGIC %md
# MAGIC ### コスト異常検出

# COMMAND ----------

# コスト異常検出クエリ（平均+3標準偏差を超えるものを検出）
anomaly_query = f"""
WITH daily_stats AS (
    SELECT 
        DATE(timestamp) as date,
        SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as daily_cost
    FROM {FULL_TABLE_NAME}
    WHERE timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
    GROUP BY DATE(timestamp)
),
stats AS (
    SELECT 
        AVG(daily_cost) as avg_cost,
        STDDEV(daily_cost) as stddev_cost
    FROM daily_stats
)
SELECT 
    d.date,
    d.daily_cost,
    s.avg_cost,
    CASE 
        WHEN (d.daily_cost - s.avg_cost) / NULLIF(s.stddev_cost, 0) > 2 THEN '異常(高)'
        WHEN (d.daily_cost - s.avg_cost) / NULLIF(s.stddev_cost, 0) < -2 THEN '異常(低)'
        ELSE '正常'
    END as status
FROM daily_stats d
CROSS JOIN stats s
ORDER BY d.date DESC
"""

print("コスト異常検出クエリ:")
print(anomaly_query)

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. 品質追跡 (8.3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 ユーザーフィードバックの記録 (8.3.3)

# COMMAND ----------

from mlflow.entities import AssessmentSource, AssessmentSourceType

# 最新のトレースIDを取得
trace_id = mlflow.get_last_active_trace_id()

if trace_id:
    # サムズアップ/ダウン形式のフィードバック
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_feedback",
        value=True,  # True = thumbs up, False = thumbs down
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="demo_user"
        ),
        rationale="回答が参考になりました"
    )
    
    # 評価スコア形式のフィードバック
    mlflow.log_feedback(
        trace_id=trace_id,
        name="rating",
        value=4,  # 1-5スケール
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="demo_user"
        ),
        rationale="概ね良いが、もう少し具体例が欲しい"
    )
    
    print(f"フィードバックを記録しました: {trace_id}")
else:
    print("トレースがありません。先にセクション2を実行してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 スコアラーの登録・開始 (8.3.4)

# COMMAND ----------

from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig

# Safetyスコアラーの登録と開始
try:
    safety_scorer = Safety().register(name="safety_check")
    safety_scorer = safety_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0)
    )
    print("✓ Safetyスコアラーを登録・開始しました")
except Exception as e:
    print(f"Safetyスコアラー: {e}")

# カスタムガイドラインスコアラーの登録と開始
try:
    tone_scorer = Guidelines(
        name="professional_tone",
        guidelines="回答は専門的で丁寧なトーンである必要があります。"
    ).register(name="tone_check")
    tone_scorer = tone_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=0.5)
    )
    print("✓ Toneスコアラーを登録・開始しました")
except Exception as e:
    print(f"Toneスコアラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 登録済みスコアラーの確認 (8.3.4)

# COMMAND ----------

from mlflow.genai.scorers import list_scorers

# 登録済みスコアラーの一覧
try:
    all_scorers = list_scorers()
    print("登録済みスコアラー:")
    for s in all_scorers:
        print(f"  - {s.name}: sample_rate={getattr(s, 'sample_rate', 'N/A')}")
except Exception as e:
    print(f"スコアラー一覧取得エラー: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. クリーンアップ

# COMMAND ----------

# スコアラーの削除（必要に応じて）
from mlflow.genai.scorers import delete_scorer

cleanup_scorers = False  # Trueに変更して実行

if cleanup_scorers:
    try:
        delete_scorer(name="safety_check")
        print("✓ safety_check を削除しました")
    except Exception as e:
        print(f"safety_check 削除: {e}")
    
    try:
        delete_scorer(name="tone_check")
        print("✓ tone_check を削除しました")
    except Exception as e:
        print(f"tone_check 削除: {e}")
else:
    print("スコアラーを削除するには cleanup_scorers = True に変更してください")

# COMMAND ----------

# アーカイブの停止（必要に応じて）
# from mlflow.tracing.archival import disable_databricks_trace_archival
# disable_databricks_trace_archival(experiment_id=experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC # 参考: 本章のセクション対応表
# MAGIC 
# MAGIC | ノートブックセクション | 本文セクション | 内容 |
# MAGIC |----------------------|----------------|------|
# MAGIC | 2.1 | 8.1.2 | 効果的なスパン設計 |
# MAGIC | 2.2 | 8.1.6 | トレースへのメタデータ追加 |
# MAGIC | 2.3 | 8.1.5 | トレースのDelta Tableアーカイブ |
# MAGIC | 3.1 | 8.2.2 | トークン使用量の確認 |
# MAGIC | 3.2 | 8.2.3 | コスト計算 |
# MAGIC | 3.3 | 8.2.4 | SQLによるコスト分析 |
# MAGIC | 4.1 | 8.3.3 | ユーザーフィードバックの記録 |
# MAGIC | 4.2 | 8.3.4 | スコアラーの登録・開始 |
# MAGIC | 4.3 | 8.3.4 | 登録済みスコアラーの確認 |
# MAGIC 
# MAGIC ## 注意事項
# MAGIC 
# MAGIC - **コスト分析クエリ**: Delta Tableへのアーカイブが有効になっている必要があります
# MAGIC - **料金情報**: 記載の料金は参考値です。最新の公式料金を確認してください
# MAGIC - **スコアラー**: 実際のLLM呼び出しが必要な機能です。ダミーデータでは動作しない場合があります
