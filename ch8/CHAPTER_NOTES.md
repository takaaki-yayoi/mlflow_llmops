# 第8章 リスト ↔ リポジトリ対応表 と 既知の挙動差分

本書 第8章「監視と運用―― LLMアプリケーションの健全性管理」の各リストとリポジトリ実装の対応表です。リポジトリのコードは可能な限り本書本文に寄せるように整備されています。ファイルと節の対応は本書 表8.1 にも掲載されています。

| ファイル | 対応する節 | 実行 |
| --- | --- | --- |
| `monitoring/01_tracing_setup.py` | 8.1.4〜8.1.7 | `make tracing` |
| `monitoring/02_token_and_cost.py` | 8.2.2〜8.2.3 | `make cost` |
| `monitoring/cost_calculator.py` | 8.2.3 | (02 から利用) |
| `monitoring/03_feedback.py` | 8.3.3 | `make feedback` |
| `monitoring/04_evaluation.py` | 8.3.4 | `make eval` |
| `monitoring/05_otel_export.py` | 8.5.3 | `make otel` |
| `monitoring/06_builtin_cost.py` | 8.2 の補足 (MLflow 3.10以降) | `make cost-builtin` |
| `monitoring/07_otel_genai_semconv.py` | 8.5 の補足 (MLflow 3.11以降) | `make otel-semconv` |

# 8.1 本番環境でのトレースベース監視

## 8.1.2 効果的なトレース設計の指針

### リスト8.1 スパンの手動設定
### リスト8.2 スパンへ属性を追加
### リスト8.3 エラー発生時の状況情報・診断情報を調整

- **対応ファイル**: 本リポジトリ未収録 (本書中の独立スニペット)
- **理由**: スパン操作は MLflow Tracing API の使い方を示す例示のため、独立した実行スクリプトとしては提供していません。
- **試したい場合**: `monitoring/01_tracing_setup.py` を雛形として、本書のコードを追加して試してください。

## 8.1.3 MLflow 3軽量トレーシングSDK / 8.1.4 本番トレーシングの基本設定

### リスト8.4 mlflow-tracingの基本設定

- **対応ファイル**: `monitoring/01_tracing_setup.py`
- **実行**: `make tracing`
- **差分**: リポジトリでは `mlflow` フルパッケージで動作確認しています。本番環境で軽量SDKに切り替える場合は `uv add mlflow-tracing` でインストールしてください (本書記載どおり、API互換性あり)。

### リスト8.5 FastAPI/Flaskアプリでのトレーシングの設定
### リスト8.6 グレースフルシャットダウン

- **対応ファイル**: 本リポジトリ未収録 (FastAPI/Flask 統合の例示)
- **理由**: 本リポジトリは Agent Server (第7章) を中心とした構成で、FastAPI/Flask の統合例は提供していません。
- **試したい場合**: 既存の FastAPI/Flask アプリケーションに本書のコードを追加してください。グレースフルシャットダウン (リスト8.6) はスタンドアロンの Python アプリケーション向けの実装です。

## 8.1.6 トレースへのメタデータ追加

### リスト8.7 タグを使ってトレースにメタデータを記録させる
### リスト8.8 タグを使ってトレースを検索

- **対応ファイル**: `monitoring/01_tracing_setup.py` 内のメタデータ追加部分
- **差分**: 本書は機能ごとに分割した抜粋。リポジトリでは1ファイルにまとめています。
- **アライメント済み**: 関数名 (`handle_chat_request`) 、標準タグ + カスタムタグ (`environment`, `service.version`, `deployment.region`) の構成、search_traces のクエリ例 (ユーザー検索・セッション時系列取得) を本書通りに揃えました。

# 8.2 トークン使用量とコストの可視化

## 8.2.2 Overviewダッシュボード / 8.2.3 トークン使用量の自動追跡

- **対応ファイル**: `monitoring/02_token_and_cost.py` (トークン使用量の自動追跡、コスト計算・集計)、`monitoring/cost_calculator.py` (コスト計算ユーティリティ)
- **実行**: `make cost`
- **差分**: Overview ダッシュボード (8.2.2) は MLflow UI の機能で、コードはありません。コスト計算のロジック (8.2.3) は本書ではリストとして掲載せず、表8.1 でファイルを参照する形になっています。単価は `cost_calculator.py` 内に定義しており、モデルの価格改定に合わせて適宜更新が必要です。

### リスト8.9 自動トレーシングの有効化

- **対応ファイル**: 各章の `agents/langgraph/` 配下のエージェント実装に自動トレーシングが組み込み済み
- **差分**: 自動トレーシング (autolog) はエージェント側で常に有効化されているため、本章の独立スクリプトには含まれません。

# 8.3 品質メトリクスのリアルタイム追跡

## 8.3.3 ユーザーフィードバックの収集

### リスト8.10 ユーザーフィードバックをトレースにひも付ける

- **対応ファイル**: `monitoring/03_feedback.py`
- **実行**: `make feedback`

## 8.3.4 LLMジャッジによる自動評価

### リスト8.11 本番環境のトレースに評価を実行
### リスト8.12 リアルタイムスコアラーの設定

- **対応ファイル**: `monitoring/04_evaluation.py`
- **実行**: `make eval` (第8章のディレクトリで実行)
- **アライメント済み**: 本書 リスト8.11 通りに `RetrievalGroundedness` / `RelevanceToQuery` / `Safety` / `Guidelines(name="professional_tone")` の組み合わせと、`tags.environment = 'production'` でのフィルタリング、`max_results=100` を反映しました。
- **残る差分**:
  - リスト8.12 (リアルタイムスコアラー) は Databricks 環境前提のため、OSS の本リポジトリには未収録です。
  - OSS環境では `make eval` を Cron や定期ジョブで回す運用が代替手段になります (8.3.5 継続的評価パイプラインの構築)。

# 8.4 アラート設定とインシデント対応

8.4.1〜8.4.4 (アラート戦略、閾値、通知、インシデント対応フロー) は設計の解説が中心で、リストはありません。

## 8.4.5 ロールバック戦略

### リスト8.13 モデルのロールバック

- **対応ファイル**: 本リポジトリ未収録 (単独スニペット)
- **試したい場合**: 本書のコードをそのまま実行できます。`previous_version` には実際のバージョン番号を指定してください。第7章リスト7.10 と内容はほぼ同一です。

# 8.5 OpenTelemetryとの統合

## 8.5.3 MLflow TracingのOTLP設定 / 8.5.4 OpenTelemetry Collectorの設定

### リスト8.14 トレースをOTLPエンドポイントにエクスポートする設定
### リスト8.15 デュアルエクスポート (OSS MLflowの場合)
### リスト8.16 デュアルエクスポート (Databricksの場合)
### リスト8.17 OpenTelemetry Collectorの設定

- **対応ファイル**: `monitoring/05_otel_export.py` (リスト8.14, 8.15, 8.16の設定確認用)
- **実行**: `make otel`
- **重要な制約**:
  - リポジトリのスクリプトは**設定方法を確認するためのもの**であり、実際にOTLPエンドポイントに送信するには **OpenTelemetry Collector が稼働している必要があります**。
  - リスト8.17 (OpenTelemetry Collector の設定YAML) はリポジトリには含まれていません。本書のコードを `otel-collector-config.yaml` として保存し、`docker run otel/opentelemetry-collector` などで起動してください。
- **環境変数の差分**:
  - OSS MLflow: `MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT=true`
  - Databricks: `MLFLOW_ENABLE_DUAL_EXPORT=true`
  - 環境によって変数名が異なる点は本書の表8.16にも記載されています。

# MLflow 3.10以降の変更への対応 (2026年11月)

## 非同期トレースログ (MLflow 3.12) と 02 / 03 の修正

MLflow 3.12 で、OSS の MLflow でもトレースが既定で非同期に書き込まれるようになりました (mlflow/mlflow#22304)。そのため、LLM を呼び出した直後に `mlflow.get_trace()` や `mlflow.log_feedback()` を呼ぶと、トレースがまだ Tracking Server に届いていないことがあります。修正前のコードを 3.12 以降で実行すると、次のエラーで止まります。

- `make cost`: `AttributeError: 'NoneType' object has no attribute 'info'` (`get_trace()` が None を返す)
- `make feedback`: `RESOURCE_DOES_NOT_EXIST: Trace with ID '...' not found.`

`02_token_and_cost.py` と `03_feedback.py` では、トレース ID を取得した直後に `mlflow.flush_trace_async_logging()` を呼ぶように修正しました。本書のリスト8.9 のトークン使用量を取得する部分を写して動かす場合は、`mlflow.get_trace()` の前に同じ1行を足してください。リスト8.10 も、LLM を呼び出した直後にフィードバックを記録する場合は同じです。

この関数は MLflow 3.9 にもあり、修正後のコードは 3.9 でも動きます。ただし 3.9 では非同期ログが無効のため、`Failed to flush trace async logging` という ERROR のログが出ます。処理には影響しません。

## 本書以降の新機能の補足サンプル

### 組み込みのコスト `trace.info.cost` (MLflow 3.10以降)

- **対応ファイル**: `monitoring/06_builtin_cost.py`
- **実行**: `make cost-builtin`
- **内容**: 8.2.3 では、トークン数から自前の料金表 (`cost_calculator.py`) でコストを計算しました。3.10 以降は、自動トレーシングで記録した LLM 呼び出しのコストを MLflow が計算し、`trace.info.cost` (`input_cost` / `output_cost` / `total_cost`、単位は USD) に入れます。8.2.4 で紹介した Overview ダッシュボードの Usage ビューのコストを、コードから取り出す方法にあたります。スクリプトでは、同じ呼び出しについて組み込みのコストと `cost_calculator.py` の結果を並べて表示します
- **料金データ**: litellm の料金データから引かれます。3.11 以降は litellm が入っていない環境でも、MLflow に同梱された料金データで計算されます (mlflow/mlflow#22046)
- **`cost_calculator.py` の扱い**: 契約価格や自社ホストのモデルなど、MLflow の料金データにない単価で計算したい場合のために残しています

### OpenTelemetry GenAI セマンティック規約でのエクスポート (MLflow 3.11以降)

- **対応ファイル**: `monitoring/07_otel_genai_semconv.py`
- **実行**: `make otel-semconv`
- **内容**: 8.5 のデュアルエクスポートで送られるスパンの属性名は、`mlflow.spanInputs` などの MLflow 独自のものです。環境変数 `MLFLOW_ENABLE_OTEL_GENAI_SEMCONV=true` を設定すると、OpenTelemetry の GenAI セマンティック規約 (`gen_ai.request.model`、`gen_ai.usage.input_tokens` など) に変換して送ります。スクリプトは Collector の代わりに受信したスパンを表示するだけの小さな OTLP 受信サーバーを起動し、同じ LLM 呼び出しを変換なしと変換ありの2通りで送って属性を見比べます。Collector や Docker は不要です

### 問題検出 (Issue detection、MLflow 3.11以降)

- **対応ファイル**: なし (MLflow UI の機能)
- **内容**: 8.3 では、LLM ジャッジやユーザーフィードバックでトレースの品質を測りました。3.11 で追加された問題検出は、選んだトレースを LLM がまとめて読み、失敗のパターンを「問題 (Issue)」として分類して、深刻度、影響を受けたトレースの数、考えられる原因と一緒に示します。どのジャッジを用意すればよいか分からない段階で、まず何が起きているかをつかむのに使えます
- **手順**:
  1. `make tracing`、`make cost`、`make feedback` などでトレースを作る
  2. MLflow UI で実験「ch8-monitoring-quickstart」の Traces タブを開き、右上の「Detect Issues」を押す
  3. ダイアログでモデル (既定は OpenAI) と対象のトレースを確かめ、「Run Analysis」を押す。初回は API キーを求められ、入力したキーは Settings > LLM Connections に保存される。`.env` の `OPENAI_API_KEY=` の部分を含めずに、キーだけを入力する
  4. 完了すると、Evaluation runs に種類が「Issue detection」のランができる。Overview タブに要約、Issues タブに問題の一覧が出る。それぞれ Resolve (対応済み) または Reject (問題ではない) に振り分けられる
- **費用**: 対象のトレースの数に応じて LLM を呼ぶ。本リポジトリの ch8 のトレース10件で、ダイアログの見積もりは $0.03〜$0.10、実際の費用は $0.13 だった

# 全体的な注意事項

- 本書の本文コードは「読んで理解するため」の抜粋です。実行可能な完全版は本リポジトリを参照してください。
- 8.5節 (OpenTelemetry連携) は外部ツール (OpenTelemetry Collector) の起動が前提となるため、リポジトリ単体では完結しません。本番運用で必要となった際の設定リファレンスとしてご利用ください。
- 本ドキュメントに未記載の挙動差分や実装上の不整合を発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ随時更新します。
