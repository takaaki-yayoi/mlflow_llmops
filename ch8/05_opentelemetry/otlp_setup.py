"""
第8章 8.5.2: OpenTelemetry (OTLP) 設定

MLflow TracingをOpenTelemetryバックエンドにエクスポートする設定例です。

重要: MLflowはトレースを単一の宛先にのみエクスポートします。
OTEL_EXPORTER_OTLP_ENDPOINTが設定されている場合、
MLflow Tracking Serverにはトレースがエクスポートされません。
"""

import os


def setup_otlp_export(
    endpoint: str,
    service_name: str,
    protocol: str = "grpc",
    service_version: str = None,
    deployment_environment: str = None,
    service_namespace: str = None,
) -> None:
    """
    OTLP エクスポートを設定
    
    Args:
        endpoint: OTLPエンドポイント (例: "http://otel-collector:4317")
        service_name: サービス名
        protocol: プロトコル ("grpc" or "http/protobuf")
        service_version: サービスバージョン (オプション)
        deployment_environment: デプロイ環境 (オプション)
        service_namespace: サービス名前空間 (オプション)
    """
    # OTLPエンドポイントの設定
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = protocol
    
    # サービス名の設定
    os.environ["OTEL_SERVICE_NAME"] = service_name
    
    # リソース属性の構築
    resource_attrs = []
    
    if service_version:
        resource_attrs.append(f"service.version={service_version}")
    if deployment_environment:
        resource_attrs.append(f"deployment.environment={deployment_environment}")
    if service_namespace:
        resource_attrs.append(f"service.namespace={service_namespace}")
    
    if resource_attrs:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)
    
    print(f"OTLP export configured:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Protocol: {protocol}")
    print(f"  Service: {service_name}")
    if resource_attrs:
        print(f"  Attributes: {', '.join(resource_attrs)}")
    
    print("\n⚠️  WARNING: MLflow UI will NOT show traces when OTLP is configured.")
    print("   Traces are exported only to the OTLP endpoint.")


def setup_otlp_for_datadog(
    service_name: str,
    dd_api_key: str = None,
    dd_site: str = "datadoghq.com",
) -> None:
    """
    Datadog向けのOTLP設定
    
    Args:
        service_name: サービス名
        dd_api_key: Datadog APIキー (環境変数DD_API_KEYからも取得可能)
        dd_site: Datadogサイト (デフォルト: datadoghq.com)
    """
    api_key = dd_api_key or os.environ.get("DD_API_KEY")
    if not api_key:
        raise ValueError("Datadog API key is required")
    
    # Datadog OTLP エンドポイント
    endpoint = f"https://trace.agent.{dd_site}:443"
    
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
    os.environ["OTEL_SERVICE_NAME"] = service_name
    
    # Datadog固有のヘッダー
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"DD-API-KEY={api_key}"
    
    print(f"Datadog OTLP export configured:")
    print(f"  Service: {service_name}")
    print(f"  Site: {dd_site}")


def setup_otlp_for_grafana_tempo(
    service_name: str,
    tempo_endpoint: str = "http://tempo:4317",
) -> None:
    """
    Grafana Tempo向けのOTLP設定
    
    Args:
        service_name: サービス名
        tempo_endpoint: Tempoエンドポイント
    """
    setup_otlp_export(
        endpoint=tempo_endpoint,
        service_name=service_name,
        protocol="grpc",
    )


def setup_otlp_for_jaeger(
    service_name: str,
    jaeger_endpoint: str = "http://jaeger:4317",
) -> None:
    """
    Jaeger向けのOTLP設定
    
    Args:
        service_name: サービス名
        jaeger_endpoint: Jaegerエンドポイント
    """
    setup_otlp_export(
        endpoint=jaeger_endpoint,
        service_name=service_name,
        protocol="grpc",
    )


def setup_otlp_via_collector(
    service_name: str,
    collector_endpoint: str = "http://otel-collector:4317",
    service_version: str = None,
    deployment_environment: str = "production",
) -> None:
    """
    OpenTelemetry Collector経由での設定
    
    Collectorを使用することで、複数のバックエンドに同時にエクスポートできます。
    
    Args:
        service_name: サービス名
        collector_endpoint: Collectorエンドポイント
        service_version: サービスバージョン
        deployment_environment: デプロイ環境
    """
    setup_otlp_export(
        endpoint=collector_endpoint,
        service_name=service_name,
        protocol="grpc",
        service_version=service_version,
        deployment_environment=deployment_environment,
        service_namespace="llm-apps",
    )
    
    print("\n📡 Using OpenTelemetry Collector for multi-backend export.")
    print("   Configure the collector to forward traces to your backends.")


def disable_otlp_export() -> None:
    """
    OTLPエクスポートを無効化し、MLflowに戻す
    """
    keys_to_remove = [
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_OTLP_HEADERS",
    ]
    
    for key in keys_to_remove:
        if key in os.environ:
            del os.environ[key]
    
    print("OTLP export disabled. Traces will be sent to MLflow Tracking Server.")


# GenAI Semantic Conventions
GENAI_SEMANTIC_CONVENTIONS = {
    # システム/プロバイダー
    "gen_ai.system": "LLMプロバイダー (openai, anthropic, etc.)",
    
    # リクエスト
    "gen_ai.request.model": "リクエストしたモデル名",
    "gen_ai.request.temperature": "温度パラメータ",
    "gen_ai.request.max_tokens": "最大トークン数",
    "gen_ai.request.top_p": "Top-Pパラメータ",
    
    # レスポンス
    "gen_ai.response.model": "実際に使用されたモデル",
    "gen_ai.response.id": "レスポンスID",
    "gen_ai.response.finish_reasons": "完了理由",
    
    # 使用量
    "gen_ai.usage.input_tokens": "入力トークン数",
    "gen_ai.usage.output_tokens": "出力トークン数",
}


def print_semantic_conventions():
    """GenAI Semantic Conventionsを表示"""
    print("=== GenAI Semantic Conventions ===")
    for attr, desc in GENAI_SEMANTIC_CONVENTIONS.items():
        print(f"  {attr}: {desc}")


# 使用例
if __name__ == "__main__":
    print("=== OTLP Setup Examples ===\n")
    
    # OpenTelemetry Collector経由
    # setup_otlp_via_collector(
    #     service_name="customer-support-bot",
    #     service_version="1.2.0",
    #     deployment_environment="production",
    # )
    
    # Datadog直接
    # setup_otlp_for_datadog(
    #     service_name="customer-support-bot",
    #     dd_api_key="your-api-key",
    # )
    
    # Grafana Tempo
    # setup_otlp_for_grafana_tempo(
    #     service_name="customer-support-bot",
    # )
    
    # Semantic Conventions
    print_semantic_conventions()
