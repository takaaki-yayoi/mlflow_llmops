"""
第8章 8.1.5: トレースへのメタデータ追加

本番環境でのデバッグと分析を効率化するため、
トレースにコンテキスト情報(タグ)を追加します。
"""

import os
import uuid
from typing import Optional
from dataclasses import dataclass
import mlflow


@dataclass
class RequestContext:
    """リクエストのコンテキスト情報"""
    user_id: str
    session_id: str
    request_id: Optional[str] = None
    environment: str = "production"
    service_version: Optional[str] = None
    deployment_region: Optional[str] = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


def add_trace_metadata(context: RequestContext) -> None:
    """
    現在のトレースにメタデータを追加します。
    
    Args:
        context: リクエストのコンテキスト情報
    """
    tags = {
        # MLflow標準タグ
        "mlflow.trace.user": context.user_id,
        "mlflow.trace.session": context.session_id,
        "mlflow.trace.request_id": context.request_id,
        
        # カスタムタグ
        "environment": context.environment,
    }
    
    # オプションのタグを追加
    if context.service_version:
        tags["service.version"] = context.service_version
    if context.deployment_region:
        tags["deployment.region"] = context.deployment_region
    
    mlflow.update_current_trace(tags=tags)


@mlflow.trace
def handle_chat_request(
    message: str,
    user_id: str,
    session_id: str,
    **kwargs
) -> str:
    """
    チャットリクエストを処理する関数の例
    
    Args:
        message: ユーザーからのメッセージ
        user_id: ユーザーID
        session_id: セッションID
        **kwargs: 追加のコンテキスト情報
    
    Returns:
        レスポンスメッセージ
    """
    # コンテキストを作成
    context = RequestContext(
        user_id=user_id,
        session_id=session_id,
        environment=kwargs.get("environment", "production"),
        service_version=os.getenv("SERVICE_VERSION", "unknown"),
        deployment_region=os.getenv("DEPLOYMENT_REGION", "unknown"),
    )
    
    # トレースにメタデータを追加
    add_trace_metadata(context)
    
    # 実際の処理 (ここではダミー)
    response = process_message(message)
    
    return response


def process_message(message: str) -> str:
    """メッセージ処理のダミー実装"""
    return f"Response to: {message}"


# タグを使った検索例
def search_traces_examples():
    """
    タグを使ったトレース検索の例
    """
    # 特定ユーザーのトレースを検索
    user_traces = mlflow.search_traces(
        filter_string="tags.`mlflow.trace.user` = 'user-123'",
        max_results=100
    )
    print(f"User traces: {len(user_traces)}")
    
    # 本番環境のエラートレースを検索
    error_traces = mlflow.search_traces(
        filter_string="tags.environment = 'production' AND status = 'ERROR'",
        order_by=["timestamp_ms DESC"]
    )
    print(f"Error traces: {len(error_traces)}")
    
    # 特定セッションのトレースを時系列で取得
    session_traces = mlflow.search_traces(
        filter_string="tags.`mlflow.trace.session` = 'session-456'",
        order_by=["timestamp_ms ASC"]
    )
    print(f"Session traces: {len(session_traces)}")
    
    # 複合条件での検索
    filtered_traces = mlflow.search_traces(
        filter_string="""
            tags.environment = 'production' 
            AND tags.`deployment.region` = 'ap-northeast-1'
            AND status = 'OK'
        """,
        order_by=["timestamp_ms DESC"],
        max_results=50
    )
    print(f"Filtered traces: {len(filtered_traces)}")
    
    return {
        "user_traces": user_traces,
        "error_traces": error_traces,
        "session_traces": session_traces,
        "filtered_traces": filtered_traces,
    }


# RAGパイプラインでのスパンレベル属性の例
@mlflow.trace
def rag_pipeline(query: str, user_id: str, session_id: str) -> str:
    """
    RAGパイプラインの例: スパンレベルでカスタム属性を記録
    """
    # トレースレベルのメタデータ
    context = RequestContext(user_id=user_id, session_id=session_id)
    add_trace_metadata(context)
    
    # 検索スパン
    with mlflow.start_span(name="retrieve_context") as span:
        span.set_inputs({"query": query})
        
        # ダミーの検索処理
        documents = [{"id": "doc1", "content": "..."}, {"id": "doc2", "content": "..."}]
        
        # スパン属性を記録
        span.set_attributes({
            "retriever.num_documents": len(documents),
            "retriever.vector_store": "pinecone",
            "retriever.similarity_threshold": 0.7,
        })
        span.set_outputs({"num_documents": len(documents)})
    
    # 生成スパン
    with mlflow.start_span(name="generate_response") as span:
        span.set_inputs({
            "query": query,
            "context_length": len(documents),
        })
        
        # ダミーの生成処理
        response = f"Answer based on {len(documents)} documents"
        
        # スパン属性を記録
        span.set_attributes({
            "llm.model": "gpt-4o",
            "llm.temperature": 0.7,
            "llm.max_tokens": 1000,
        })
        span.set_outputs({
            "response": response,
            "response_length": len(response),
        })
    
    return response


# 使用例
if __name__ == "__main__":
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/test/metadata-example")
    
    # チャットリクエストの処理
    response = handle_chat_request(
        message="What is MLflow?",
        user_id="user-123",
        session_id="session-456",
    )
    print(f"Response: {response}")
    
    # RAGパイプラインの実行
    rag_response = rag_pipeline(
        query="How to use MLflow Tracing?",
        user_id="user-123",
        session_id="session-456",
    )
    print(f"RAG Response: {rag_response}")
