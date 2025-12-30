"""
第8章 8.1.2-8.1.3: 本番トレーシングの基本設定

本番環境でMLflow Tracingを設定するための基本的なコード例です。
軽量SDK(mlflow-tracing)の使用を前提としています。
"""

import os
import mlflow


def setup_production_tracing(
    experiment_name: str,
    service_name: str = "llm-app",
    tracking_uri: str = "databricks",
    enable_async: bool = True,
    deployment_region: str = None,
    service_version: str = None,
) -> None:
    """
    本番環境向けのMLflow Tracing設定を行います。
    
    Args:
        experiment_name: MLflowエクスペリメント名 (例: "/production/customer-support-bot")
        service_name: サービス識別名
        tracking_uri: MLflow Tracking ServerのURI ("databricks" or URL)
        enable_async: 非同期ログ記録を有効にするか
        deployment_region: デプロイリージョン (オプション)
        service_version: サービスバージョン (オプション)
    """
    # 1. 非同期ログ記録の設定
    # → トレース送信完了を待たずにレスポンスを返せる
    if enable_async:
        os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
    
    # 2. サービス名の設定
    # → 複数サービスのトレースを識別可能にする
    os.environ["OTEL_SERVICE_NAME"] = service_name
    
    # 3. リソース属性の追加 (オプション)
    resource_attrs = []
    if service_version:
        resource_attrs.append(f"service.version={service_version}")
    if deployment_region:
        resource_attrs.append(f"deployment.environment={deployment_region}")
    
    if resource_attrs:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)
    
    # 4. トラッキングサーバーとエクスペリメントの設定
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    print(f"Production tracing configured:")
    print(f"  - Experiment: {experiment_name}")
    print(f"  - Service: {service_name}")
    print(f"  - Async logging: {enable_async}")


def enable_autologging(providers: list[str] = None) -> None:
    """
    指定されたプロバイダーの自動トレーシングを有効化します。
    
    Args:
        providers: 有効化するプロバイダーのリスト
                   None の場合は全プロバイダーを自動検出
    
    対応プロバイダー:
        - openai
        - anthropic
        - langchain
        - langgraph
        - llama_index
    """
    if providers is None:
        # 全プロバイダーを自動検出して有効化
        mlflow.autolog()
        print("Autologging enabled for all supported providers")
        return
    
    provider_map = {
        "openai": mlflow.openai.autolog,
        "anthropic": mlflow.anthropic.autolog,
        "langchain": mlflow.langchain.autolog,
        "langgraph": mlflow.langgraph.autolog,
        "llama_index": mlflow.llama_index.autolog,
    }
    
    for provider in providers:
        if provider in provider_map:
            provider_map[provider]()
            print(f"Autologging enabled for {provider}")
        else:
            print(f"Warning: Unknown provider '{provider}'")


# 使用例
if __name__ == "__main__":
    # 本番環境の設定
    setup_production_tracing(
        experiment_name="/production/customer-support-bot/2024-12",
        service_name="customer-support-bot",
        tracking_uri="databricks",
        enable_async=True,
        service_version="1.2.0",
        deployment_region="ap-northeast-1",
    )
    
    # OpenAIの自動トレーシングを有効化
    enable_autologging(providers=["openai"])
    
    # 以降、通常通りLLM APIを呼び出すとトレースが自動記録される
    # from openai import OpenAI
    # client = OpenAI()
    # response = client.chat.completions.create(...)
