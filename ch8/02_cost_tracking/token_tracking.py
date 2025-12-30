"""
第8章 8.2.2: トークン使用量の自動追跡

MLflow Tracingによるトークン使用量の自動追跡と
トレースからのトークン情報取得方法を示します。
"""

import mlflow
from openai import OpenAI


def setup_token_tracking():
    """
    トークン追跡のセットアップ
    
    対応フレームワーク:
        - OpenAI: mlflow.openai.autolog()
        - Anthropic: mlflow.anthropic.autolog()
        - LangChain: mlflow.langchain.autolog()
        - LangGraph: mlflow.langgraph.autolog()
        - LlamaIndex: mlflow.llama_index.autolog()
        - 全自動検出: mlflow.autolog()
    """
    # OpenAIの自動トレーシングを有効化
    mlflow.openai.autolog()


def get_token_usage_from_trace(trace_id: str) -> dict:
    """
    トレースからトークン使用量を取得
    
    Args:
        trace_id: トレースID
    
    Returns:
        トークン使用量の辞書
    """
    trace = mlflow.get_trace(trace_id=trace_id)
    token_usage = trace.info.token_usage
    
    return {
        "input_tokens": token_usage.get("input_tokens", 0),
        "output_tokens": token_usage.get("output_tokens", 0),
        "total_tokens": token_usage.get("total_tokens", 0),
    }


def get_last_trace_token_usage() -> dict:
    """
    最後のトレースからトークン使用量を取得
    
    Returns:
        トークン使用量の辞書
    """
    trace_id = mlflow.get_last_active_trace_id()
    if trace_id is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    return get_token_usage_from_trace(trace_id)


def demo_token_tracking():
    """
    トークン追跡のデモ
    """
    # セットアップ
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/test/token-tracking")
    setup_token_tracking()
    
    # OpenAI APIを呼び出し
    client = OpenAI()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    
    # トークン使用量を取得
    token_usage = get_last_trace_token_usage()
    
    print(f"Input tokens: {token_usage['input_tokens']}")
    print(f"Output tokens: {token_usage['output_tokens']}")
    print(f"Total tokens: {token_usage['total_tokens']}")
    
    return token_usage


def aggregate_token_usage_from_traces(
    experiment_name: str,
    filter_string: str = None,
    max_results: int = 1000,
) -> dict:
    """
    複数のトレースからトークン使用量を集計
    
    Args:
        experiment_name: エクスペリメント名
        filter_string: フィルタ条件 (オプション)
        max_results: 最大取得件数
    
    Returns:
        集計結果
    """
    traces = mlflow.search_traces(
        experiment_names=[experiment_name],
        filter_string=filter_string,
        max_results=max_results,
    )
    
    total_input = 0
    total_output = 0
    total_tokens = 0
    trace_count = 0
    
    for _, trace_row in traces.iterrows():
        trace_id = trace_row.get("trace_id")
        if trace_id:
            try:
                trace = mlflow.get_trace(trace_id=trace_id)
                usage = trace.info.token_usage or {}
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)
                trace_count += 1
            except Exception as e:
                print(f"Warning: Could not get trace {trace_id}: {e}")
    
    return {
        "trace_count": trace_count,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "avg_input_tokens": total_input / trace_count if trace_count > 0 else 0,
        "avg_output_tokens": total_output / trace_count if trace_count > 0 else 0,
        "avg_total_tokens": total_tokens / trace_count if trace_count > 0 else 0,
    }


# 使用例
if __name__ == "__main__":
    # デモを実行 (OpenAI APIキーが必要)
    # token_usage = demo_token_tracking()
    
    # 集計の例
    # stats = aggregate_token_usage_from_traces(
    #     experiment_name="/production/my-app",
    #     filter_string="tags.environment = 'production'",
    # )
    # print(f"Aggregated stats: {stats}")
    pass
