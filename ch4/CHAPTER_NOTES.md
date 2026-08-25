# 第4章 リスト ↔ リポジトリ対応表 と 既知の挙動差分

本書 第4章「可観測性の確保 ──トレーシングの導入」の各リスト (コード片) と、本リポジトリの実装の対応表です。

# 章の位置づけ

第4章は第3章 (ch3) で構築した QA エージェントに **MLflow Tracing を追加する**章です。本リポジトリの ch4 は ch3 のコピーで、`agents/langgraph/agent.py` の冒頭に以下 4 行 (import 1 行 + 設定 3 行) が追加されている点だけが実装上の差分です。本書 4.4.1 の記載どおりです。

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")
mlflow.langchain.autolog()
```

ch3 と共通の差分メモ (ツールの有効・無効、埋め込みモデル、`make clean` など) は [ch3/CHAPTER_NOTES.md](../ch3/CHAPTER_NOTES.md) を参照してください。本ドキュメントでは **トレーシング固有の注意点と、本書に掲載されていてリポジトリに未収録のコード** に絞って解説します。

# 4.3 トレーシングを有効にする / 4.4 MLflow QAエージェントへのトレーシング導入

## 4.4.1 自動トレーシングの有効化

- **対応箇所**: `agents/langgraph/agent.py` 冒頭の 4 行 (上記)
- **差分・注意**:
  * リポジトリでは最初からこの 4 行が入った状態です。本書のように ch3 に自分で追記する必要はありません。
  * **モジュール読み込み時に Tracking Server の設定を行う**実装のため、Tracking Server (3.2.5 で起動) を起動していない状態で `make cli` を実行すると、トレース送信時に接続エラーの警告が出ます。エージェント自体は動作しますが、トレースは記録されません。本書の手順どおり、先に Tracking Server を起動してください。
  * 本書は「環境変数 `MLFLOW_TRACKING_URI` と `MLFLOW_EXPERIMENT_NAME` で設定する方法もある」と述べていますが、リポジトリの `agent.py` は `set_tracking_uri()` / `set_experiment()` を直接呼んでいるため、`.env` に `MLFLOW_TRACKING_URI` を書いても無視されます。別のサーバに送りたい場合は `agent.py` の 1 行目を書き換えてください。

## 4.4.2 実行してトレースを確認

- **対応**: `make cli` の実行後、MLflow UI (http://localhost:5000) の Traces タブで確認
- **差分**: なし。本書の説明手順がそのまま使えます。スクリーンショット (図4.4〜4.10) は執筆時点の MLflow の UI であり、バージョンによって表示は多少異なります。

## 4.4.3 エラーのデバッグと修正 (リスト4.1)

- **対応ファイル**: `agents/langgraph/tools/doc_search.py`
- **差分・注意**:
  * 本書はデバッグ手順を体験するために `doc_search` 関数に `raise ValueError(...)` を追加して意図的にエラーを起こしています。リポジトリにはこのエラー行は入っていないので、試す場合は自分で追加し、確認後に元に戻してください。
  * リスト4.1 (リトライロジック) はリポジトリの `doc_search.py` に実装済みです。本書は `@tool` デコレータ以下の要点を抜粋しており、`_reset_milvus_connection()` などの補助関数はリポジトリで確認してください。

# 4.5 トレースに追加情報を付与する

## 4.5.1 タグの追加

- **対応箇所**: `agents/langgraph/agent.py` の `process_query()` メソッド
- **差分・注意**:
  * 本書は `process_query()` 内で `mlflow.update_current_trace(tags={...})` を呼び出す手順を示していますが、**リポジトリの ch4 にはこのコードは入っていません**。試す場合は本書の記載どおり `process_query()` の冒頭に追記してください。
  * ch5 以降の `agent.py` にもタグ追加のコードは含まれていません (各章の autolog 設定は ch4 と同じです)。

## 4.5.2 タグを使ったトレースの検索 (リスト4.2)

- **対応**: リポジトリ未収録。本書のコードを単体のスクリプトとして実行してください (実験 ID は MLflow UI または `mlflow.get_experiment_by_name("MLflow QAエージェント").experiment_id` で取得)。

## 4.5.3 会話セッションの管理

- **対応箇所**: `agents/langgraph/agent.py` の `process_query()` 内 `config = {"configurable": {"thread_id": thread.id}}`
- **差分**: なし。本書の説明どおり、LangGraph に渡した `thread_id` を MLflow が自動的にセッションとして扱うため、追加コードは不要です。CLI の `/new` で新しいスレッドを開始すると、MLflow UI 上でも別セッションになります。

# 4.6 トレーシングの仕組みと実装 (リスト4.3〜4.5)

- **対応**: リポジトリでは `mlflow.langchain.autolog()` のみを使用しており、`@mlflow.trace` デコレータによる手動トレーシングは行っていません。
- **試したい場合**:
  * リスト4.3 / 4.4 (手動トレーシングの例) は単体のスクリプトとして実行できます。
  * リスト4.5 (`process_query()` への `@mlflow.trace(span_type=SpanType.AGENT)` の付与) は、本書の記載どおり `agent.py` に `from mlflow.entities import SpanType` の import とデコレータを追加すると動作を確認できます。4.5.1 のタグ追加と組み合わせた形がリスト4.5 です。

# 4.7〜4.10 応用編 (リスト4.6〜4.15)

以下は本書で解説されていますが、**リポジトリには収録していません**。いずれも QA エージェント本体とは独立した内容です。

| 節 | 内容 | リスト | 備考 |
| --- | --- | --- | --- |
| 4.7 応用 (1) | TypeScript アプリケーションのトレーシング (MLflow TypeScript SDK / Vercel AI SDK) | 4.6, 4.7 | Node.js 環境が別途必要 |
| 4.8 応用 (2) | `mlflow.search_traces()` によるトレースの検索 | 4.8〜4.10 | 単体スクリプトとして実行可能 |
| 4.9 応用 (3) | OpenTelemetry との連携 (FastAPI からの送信、OTLP 取り込み、Collector) | 4.11, 4.12 | OpenTelemetry 連携のサンプルは第8章 (ch8) に収録 |
| 4.10 応用 (4) | 並行実行とスレッド安全性 (async / マルチスレッド) | 4.13〜4.15 | 単体スクリプトとして実行可能 |

# トラブルシューティング

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| MLflow UI にトレースが出ない | Tracking Server が未起動、または別のポートで起動している | 3.2.5 の手順で `http://localhost:5000` に起動する |
| `Connection refused` の警告が出るが応答は返る | Tracking Server 未起動。トレース記録は失敗するが、エージェントの動作自体は継続する | Tracking Server を起動する |
| ブラウザで `http://0.0.0.0:5000` を開いても表示されない | `--host 0.0.0.0` は待ち受けアドレスであり、ブラウザからのアクセス先ではない | `http://localhost:5000` を開く |
| `.env` の `MLFLOW_TRACKING_URI` を変えても反映されない | `agent.py` が `set_tracking_uri()` で直接指定している | `agent.py` の該当行を書き換える |
| 既存の experiment と重複している | `set_experiment("MLflow QAエージェント")` で同名の experiment を再利用 | 既存の experiment は維持されます。問題ありません |

# 全体的な注意事項

- 本章は ch3 とのコード差分が 4 行のみのため、本書とリポジトリの間で混乱が起きにくい章です。
- ただし「Tracking Server を起動していないと autolog の効果が見えない」点が、ch3 から ch4 に進んだ読者がつまずきやすいポイントです。
- 本ドキュメントに未記載の挙動差分や実装上の不整合を発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ随時更新します。
