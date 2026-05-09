# 本書脚注 / 注釈で参照されている外部 URL の要約

本書では、MLflow 公式ドキュメント等の外部 URL を脚注で示している箇所が複数あります。ここでは、本書の流れを止めずに該当ページの要点を把握できるよう、**章ごとに URL を分類し、それぞれに 2〜3 行の要約と「本書のどの手順の前に読むと迷わないか」のヒント**をまとめています。

URL の網羅性は段階的に高めていきます。本ドキュメントに未収録の脚注 URL でフォローが必要なものを見つけられた場合は、GitHub Issues に `errata` ラベルを付けて報告いただければ随時追記します。

# 第3章 LLMアプリケーションの構築

## LangGraph 公式

URL: https://langchain-ai.github.io/langgraph/

**要約**: LangGraph はステートフルなマルチアクターアプリケーションを LLM で構築するためのフレームワーク。`StateGraph` でノード間の状態遷移を定義する。

**読みどころ**: 本書 3 章のエージェント構築前に、`StateGraph` / `MessagesState` / `ToolNode` の概念に目を通しておくと、ch3 の `agents/langgraph/agent.py` の `_build_graph()` の流れが追いやすくなります。

## Milvus Lite

URL: https://milvus.io/docs/milvus_lite.md

**要約**: Milvus の組み込み版。サーバ起動不要でローカル ファイル (`*.db`) としてベクトルデータベースを扱える。

**読みどころ**: 本書ではフルの Milvus サーバを前提とした記述がありますが、本リポジトリは `milvus-lite` を使用しているため、別途サーバ起動は不要です。

# 第4章 可観測性の確保

## MLflow Tracing 概要

URL: https://mlflow.org/docs/latest/llms/tracing/index.html

**要約**: LLM アプリの実行を span のツリーとして可視化する MLflow の機能。autolog による自動トレース、`@mlflow.trace` デコレータによる手動計装の 2 系統がある。

**読みどころ**: ch4 では autolog のみを使用。手動計装を試したい場合に最初に読むページ。

## mlflow.langchain.autolog

URL: https://mlflow.org/docs/latest/llms/langchain/autologging.html

**要約**: LangChain / LangGraph の各種コール (LLM 呼び出し、ツール呼び出し、各ノード) を MLflow に自動記録する。

**読みどころ**: ch4 で 3 行追加するだけでトレースが取れる仕組みの内訳を把握したいときに。

# 第5章 評価の仕組み

## mlflow.genai.evaluate

URL: https://mlflow.org/docs/latest/llms/genai/evaluation.html

**要約**: LLM アプリ向けの評価エントリポイント。スコアラーのリストとデータセットを渡すと、各サンプルに対するスコアと総合結果を MLflow に記録する。

**読みどころ**: ch5 の `make eval` の実体。本書のリスト 5.11〜5.13 を読む前に、`evaluate()` の引数と戻り値の構造を確認しておくと混乱しません。

## 標準スコアラー (Built-in Scorers)

URL: https://mlflow.org/docs/latest/llms/genai/scorers/built-in.html

**要約**: ToolUsage、Correctness、Safety、Guidelines、ConversationCompleteness、UserFrustration などが MLflow に同梱されている。

**読みどころ**: 本書 5.4 節で個別に解説されているスコアラーの一覧と使い分けの早見表として。

## カスタムスコアラー (`@scorer` / `make_judge`)

URL: https://mlflow.org/docs/latest/llms/genai/scorers/custom.html

**要約**: ルールベースのスコアラーは `@scorer` デコレータ、LLM ジャッジは `make_judge()` で定義する。

**読みどころ**: 本書 リスト5.4 (`contains_code_block`)、リスト5.6 (Guidelines)、リスト5.8 (`make_judge`) の前提として一読を推奨。

## 会話シミュレーション (`mlflow.genai.simulate_session`)

URL: https://mlflow.org/docs/latest/llms/genai/simulation.html

**要約**: ペルソナと目標を与えると、LLM が複数ターンの会話を生成してくれる API。会話評価のテストデータを自動生成できる。

**読みどころ**: ch5 の `make sim` の前提知識。コストが発生するため事前に料金感を把握しておくと安心。

## サードパーティ評価ライブラリの統合

URL: https://mlflow.org/docs/latest/llms/genai/scorers/third-party.html

**要約**: DeepEval、RAGAS のスコアラーを MLflow の評価パイプラインに組み込める。MLflow 3.8.0 以降で利用可能。

**読みどころ**: 本書 リスト5.23 を試したいとき。`uv add deepeval ragas` で導入してから `evaluate()` の `scorers=[...]` に追加する形になります。

# 第6章 Prompt Registry

## Prompt Registry の概要

URL: https://mlflow.org/docs/latest/llms/prompt-registry/index.html

**要約**: MLflow にプロンプトをバージョン管理対象として登録できる機能。エイリアス (`production`, `staging` 等) でライフサイクルを管理する。

**読みどころ**: ch6 の `make register` 〜 `make alias` のフロー全体の前に。

## MetaPrompt / GEPA による自動最適化

URL: https://mlflow.org/docs/latest/llms/prompt-registry/optimization.html

**要約**: 評価データセットと評価関数を与えると、MLflow がプロンプトを反復改善する。MetaPrompt は構造改善、GEPA は遺伝的最適化。

**読みどころ**: ch6 の `make optimize-meta` / `make optimize-gepa` を実行する前に、それぞれのアルゴリズムの違いと API コスト感を把握するため。

## モデルパラメータの紐付け

URL: https://mlflow.org/docs/latest/llms/prompt-registry/model-config.html

**要約**: プロンプトに `temperature`、`max_tokens` などのモデルパラメータを紐付けて登録できる。プロンプトと推論設定を一体で管理できる。

**読みどころ**: 本書 6.2.6 節 (`make model-config`) の前提。

# 第7章 サービングとデプロイメント

## Agent Server

URL: https://mlflow.org/docs/latest/llms/agents/serving.html

**要約**: `@invoke` でエージェントを定義し、`mlflow agent serve` で REST エンドポイントとして公開できる仕組み。

**読みどころ**: ch7 の `make serve` の実体。

## AI Gateway (Tracking Server 統合型 / 新方式)

URL: https://mlflow.org/docs/latest/llms/gateway/index.html

**要約**: 複数 LLM プロバイダへのアクセスを 1 つのエンドポイントに集約する機能。コスト管理・レート制限・キー管理をゲートウェイ側に寄せられる。MLflow 3.10 以降は Tracking Server に統合されている。

**読みどころ**: 本書 7.4 節は新方式が前提。MLflow 3.9.0 以下で動かす場合は、本リポジトリの `gateway/gateway_config.yaml` を使う Legacy 方式 (下記) を参照。

## AI Gateway (Legacy 方式)

URL: https://mlflow.org/docs/3.9.0/llms/gateway/index.html

**要約**: `mlflow gateway start --config-path <yaml>` で別プロセス起動する旧方式。MLflow 3.9.x まで。

**読みどころ**: 本リポジトリ ch7 の `make gateway-legacy` で起動する方式の出典。

## Databricks Agent Framework

URL: https://docs.databricks.com/en/generative-ai/agent-framework/index.html

**要約**: Databricks 環境専用の Agent Framework。`databricks.agents` モジュールでマネージド MLflow へのデプロイ、評価、モニタリングを統合的に扱える。

**読みどころ**: 本書 リスト7.9 (`make log-model` 後の Databricks デプロイ) の前提。OSS リポジトリには含まれていません。

# 第8章 監視と運用

## トレースのメタデータ追加

URL: https://mlflow.org/docs/latest/llms/tracing/api.html#metadata

**要約**: `mlflow.update_current_trace()` でトレースに `tags`、`request_id`、`session_id` などのメタデータを付与できる。本番運用での集計・絞り込みに必須。

**読みどころ**: ch8 8.1 節 (`make tracing`) の前提。

## トークン使用量とコスト

URL: https://mlflow.org/docs/latest/llms/tracing/cost.html

**要約**: トレースから `usage` 情報 (input/output tokens) と推定コストを取り出すことができる。プロバイダ別にコストレートが事前定義されている。

**読みどころ**: ch8 8.2 節 (`make cost`) の実体。

## OpenTelemetry エクスポート

URL: https://mlflow.org/docs/latest/llms/tracing/opentelemetry.html

**要約**: MLflow のトレースを OTLP 経由で任意の OTel バックエンド (Datadog、New Relic、Grafana Tempo など) に送出できる。

**読みどころ**: ch8 8.5 節 (`make otel`) の前提。社内の APM に統合したい場合に。

# 第9章 チュートリアル

## Models from Code

URL: https://mlflow.org/docs/latest/models/models-from-code.html

**要約**: Python ファイルそのものを MLflow Model として登録できる機能。サービング時にコードがそのまま実行される。

**読みどころ**: 本書 9.1 ノートブックの「リスト9.3 (リスト9.2をモデルとして登録)」の前提。

## カスタム PyFunc モデル

URL: https://mlflow.org/docs/latest/models/python_function.html

**要約**: 任意の Python 関数を MLflow Model として包む方法。シグネチャ、入力例、依存関係を一緒に保存できる。

**読みどころ**: 9.1 ノートブックでの推論ロジック登録の汎用版として。

## ResponsesAgent インターフェース

URL: https://mlflow.org/docs/latest/llms/agents/responses-agent.html

**要約**: マルチエージェントを単一エンドポイントの背後に置くための統一インターフェース。OpenAI Responses API 互換のレスポンスを返す。

**読みどころ**: 9.3 ノートブックでスーパーバイザー型マルチエージェントを実装する前に。

# 全章共通

## MLflow リリースノート

URL: https://github.com/mlflow/mlflow/releases

**要約**: MLflow の各バージョンで追加 / 変更された機能の一覧。本書記述と挙動が違う場合、まずリリースノートで該当機能のバージョンを確認するのが近道。

**読みどころ**: 本書執筆時点は MLflow 3.9.0 前後を基準にしています。3.10 以降の機能 (`mlflow.search_sessions`、新方式 AI Gateway 等) は本書未収録ですが、リリースノートでフォロー可能。

## MLflow GenAI 全体ドキュメント

URL: https://mlflow.org/docs/latest/llms/index.html

**要約**: GenAI 関連機能 (Tracing / Evaluation / Prompt Registry / AI Gateway) のエントリポイント。各章の上位概念を俯瞰したいときに。

**読みどころ**: 本書を読み終えたあと、機能横断で振り返りたいタイミングで。

# 更新ポリシー

- 本ドキュメントは段階的に追記します。本書の脚注 / 注釈で「URL のみ」が記載されている箇所のうち、要約があると読者の手が止まらない箇所を優先的に拡充していきます。
- MLflow のバージョンアップに伴って URL 構造や機能名が変わった場合、できる限り追従します。
- 不足や誤りを発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ反映します。
