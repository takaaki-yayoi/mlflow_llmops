"""
第8章 - 8.5 OpenTelemetry連携の設定確認

MLflow TracingをOpenTelemetryプロトコル(OTLP)でエクスポートする
設定方法を確認します。

注意: 実際にOTLPエンドポイントに送信するには、
OpenTelemetry Collectorが稼働している必要があります。
このスクリプトは設定方法の確認用です。

実行方法:
  make otel
  または
  uv run python monitoring/05_otel_export.py
"""

# === 1. OSS MLflow: デュアルエクスポート設定 ===
print("=== OSS MLflow デュアルエクスポート設定 ===\n")

oss_config = {
    "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT": "true",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://localhost:4317/v1/traces",
    "OTEL_SERVICE_NAME": "qa-agent",
    "OTEL_RESOURCE_ATTRIBUTES": (
        "service.version=1.0.0,"
        "deployment.environment=production,"
        "service.namespace=llm-apps"
    ),
}

print("OSS MLflow で MLflow + OTel Collector の両方にトレースを送信する設定:")
print()
for key, value in oss_config.items():
    print(f'  export {key}="{value}"')

# === 2. Databricks: デュアルエクスポート設定 ===
print("\n=== Databricks デュアルエクスポート設定 ===\n")

databricks_config = {
    "MLFLOW_ENABLE_DUAL_EXPORT": "true",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://localhost:4317/v1/traces",
    "OTEL_SERVICE_NAME": "qa-agent",
}

print(
    "Databricks で Databricks MLflow + OTel Collector の両方にトレースを送信する設定:"
)
print()
for key, value in databricks_config.items():
    print(f'  export {key}="{value}"')

# === 3. 動作確認(Collector不要) ===
print("\n=== 動作確認(実際の送信は行いません) ===\n")

import mlflow

# 設定の確認のみ(実際のエクスポートはしない)
mlflow.set_experiment("ch8-monitoring-quickstart")
mlflow.openai.autolog()

print("現在のMLflow Tracking URI:", mlflow.get_tracking_uri())
print(
    "現在のエクスペリメント:",
    mlflow.get_experiment_by_name("ch8-monitoring-quickstart"),
)
print()
print("✅ OpenTelemetry連携の設定方法を確認しました。")
print("   実際にエクスポートするには、上記の環境変数を設定した上で")
print("   OpenTelemetry Collector を起動してください。")
print()
print("   参考: OpenTelemetry Collector Docker イメージ")
print(
    "   docker run -p 4317:4317 otel/opentelemetry-collector-contrib:latest"
)
