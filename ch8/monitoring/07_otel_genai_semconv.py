"""
第8章 補足 - OpenTelemetry GenAI セマンティック規約でのエクスポート (MLflow 3.11以降)

8.5節では、MLflow のトレースを OTLP で OpenTelemetry Collector にも送る
デュアルエクスポートの設定を確認しました。そのとき送られるスパンの属性名は
MLflow 独自のもの (mlflow.spanInputs など) です。

MLflow 3.11 で追加された MLFLOW_ENABLE_OTEL_GENAI_SEMCONV を有効にすると、
OpenTelemetry の GenAI セマンティック規約 (gen_ai.* 属性) に変換して送ります。
Datadog や Grafana など、GenAI 規約に対応した監視基盤でそのまま扱えるようになります。

このスクリプトは Collector の代わりに、受け取ったスパンを表示するだけの
小さな OTLP 受信サーバーを起動し、同じ LLM 呼び出しを
「変換なし」「変換あり」の2通りで送って属性を見比べます。
MLflow Tracking Server にも従来どおりトレースが記録されます。

実行方法:
  make otel-semconv
  または
  uv run python monitoring/07_otel_genai_semconv.py

本書の対象版 (MLflow 3.9) にはこの設定がないため、3.11 以降で動きます。
"""

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

load_dotenv()

RECEIVER_PORT = 4318

# MLflow を import する前に、OTLP の送信先とデュアルエクスポートを設定する
os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = f"http://localhost:{RECEIVER_PORT}/v1/traces"
os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
os.environ["MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT"] = "true"
os.environ["OTEL_SERVICE_NAME"] = "qa-agent"

import mlflow
from openai import OpenAI
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

# === 1. 受け取ったスパンを貯めておく OTLP 受信サーバー ===
received_spans: list = []


class OtlpReceiver(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                received_spans.extend(scope_spans.spans)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


server = HTTPServer(("localhost", RECEIVER_PORT), OtlpReceiver)
threading.Thread(target=server.serve_forever, daemon=True).start()


def attribute_value(value) -> str:
    """OTLP の AnyValue を表示用の文字列にする。"""
    kind = value.WhichOneof("value")
    if kind == "array_value":
        text = "[" + ", ".join(attribute_value(v) for v in value.array_value.values) + "]"
    else:
        text = str(getattr(value, kind))
    return text if len(text) <= 70 else text[:70] + "..."


def call_llm_and_collect(semconv: bool) -> list:
    """LLM を1回呼び出し、OTLP で受け取った LLM スパンの属性を返す。"""
    # この環境変数はスパンの終了時に読まれるので、呼び出しごとに切り替えられる
    os.environ["MLFLOW_ENABLE_OTEL_GENAI_SEMCONV"] = "true" if semconv else "false"
    received_spans.clear()

    client.chat.completions.create(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": "MLflow Tracingを一文で説明してください。"}],
    )

    # BatchSpanProcessor は一定間隔でまとめて送るので、届くまで待つ
    for _ in range(50):
        if received_spans:
            break
        time.sleep(0.2)
    return received_spans[:]


mlflow.set_experiment("ch8-monitoring-quickstart")
mlflow.openai.autolog()
client = OpenAI()

# === 2. 変換なし (MLflow 独自の属性名) ===
print("=== 変換なし: MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=false ===\n")
for span in call_llm_and_collect(semconv=False):
    print(f"スパン名: {span.name}")
    for attr in span.attributes:
        print(f"  {attr.key} = {attribute_value(attr.value)}")

# === 3. 変換あり (GenAI セマンティック規約) ===
print("\n=== 変換あり: MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true ===\n")
for span in call_llm_and_collect(semconv=True):
    print(f"スパン名: {span.name}")
    for attr in span.attributes:
        print(f"  {attr.key} = {attribute_value(attr.value)}")

server.shutdown()
print()
print("変換ありでは、モデル名やトークン数が gen_ai.* の属性名で送られています。")
print("MLflow UI (http://localhost:5000) には、どちらの呼び出しも通常どおり記録されています。")
