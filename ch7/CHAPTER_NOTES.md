# 第7章 リスト ↔ リポジトリ対応表 と 既知の挙動差分

本書 第7章「本番環境で動かす―― サービングとデプロイメント」の各リストとリポジトリ実装の対応表です。リポジトリのコードは可能な限り本書本文に寄せるように整備されています。

# 7.2 エージェントのサービング準備

## リスト7.1 Agent Server用エージェント定義
## リスト7.2 プロンプトの動的なロード
## リスト7.3 Agent Serverの起動スクリプト

- **対応ファイル**:
  - `serving/agent.py` (リスト7.1, 7.2 — エージェント定義とPrompt Registryからの動的ロード)
  - `serving/start_server.py` (リスト7.3 — サーバー起動スクリプト)
- **実行**: `make serve`
- **前提**: `make log-model` でモデルを登録、Prompt Registryに `qa-agent-system-prompt` が登録済みであること (第6章実行後)。
- **アライメント済み**: `serving/agent.py` のdocstringの「プロンプトレジストリ」を本書通り「Prompt Registry」に揃え、import 文の整理・コメントの統一を行いました。`start_server.py` の構造とコメントも本書通りに揃えました。

# 7.3 サービング中のエージェントの評価

## リスト7.4 サービング中のエージェントに評価を実行

- **対応ファイル**: `serving/eval_serving.py`
- **実行**: `make eval`
- **重要な制約**: Agent Server が稼働中だと Milvus データベースのファイルロックが競合するため、**Agent Serverを停止してから実行してください** (リポジトリの docstring に記載)。本書には明記されていない運用上の注意点です。
- **アライメント済み**: 変数名 (`eval_dataset` 小文字)、各 `expected_response` の文言を本書 リスト7.4 と完全一致させました (リスト6.8 の `expected_answer` を踏襲) 。

# 7.4 AI Gateway の活用

## リスト7.5 QAエージェントのAI Gateway対応
## リスト7.6 AI Gateway対応 (OpenAI SDK)

- **対応ファイル**: 本リポジトリ未収録 (エージェントから AI Gateway を呼び出す側のコード)
- **代替**: ゲートウェイ設定として `gateway/gateway_config.yaml` (Legacy方式) を提供しています。
- **重要な差分**:
  - 本書では `http://localhost:5000/gateway/mlflow/v1` のように **Tracking Server統合型 (新方式)** の AI Gateway を前提に解説しています。
  - リポジトリの `gateway_config.yaml` は **Gateway Server (Legacy方式)** で、`mlflow gateway start --config-path gateway/gateway_config.yaml --port 5010` のように別ポートで起動します。
  - 本書の説明 (新方式) は MLflow 3.10 以降が必要です。MLflow 3.9.0 では Legacy 方式 (`gateway_config.yaml`) を使用してください。
  - 詳細は `gateway/gateway_config.yaml` 冒頭のコメントを参照してください。

# 7.5 本番デプロイメント

## リスト7.7 Agent Serverを直接起動するDockerfile

- **対応ファイル**: `deploy/Dockerfile`
- **差分**: 本書は抜粋。リポジトリの完全版を参照してください。
- **本書側のerrata候補**: 本書 リスト7.7 では `EMBEDDING_MODEL=text-embedding-small` と記載されていますが、OpenAIのモデル名としては `text-embedding-3-small` が正しく、本書 リスト7.8 (Kubernetes) でも `text-embedding-3-small` が使用されています。リポジトリは `text-embedding-3-small` を採用しています。

## リスト7.8 デプロイのためのKubernetesマニフェスト

- **対応ファイル**: `deploy/k8s/deployment.yaml`、`deploy/k8s/service.yaml`
- **差分**: 本書は1つのリストとして連結提示、リポジトリでは2ファイルに分離しています。
- **動作には影響なし**: `kubectl apply -f deploy/k8s/` でディレクトリごと適用すれば本書通りに動作します。

## リスト7.9 マネージドMLflowでのデプロイ

- **対応ファイル**: 本リポジトリ未収録 (Databricks環境前提)
- **理由**: `databricks.agents` モジュールは Databricks 環境でのみ動作するため、OSS の本リポジトリには含めていません。
- **試したい場合**: Databricks 環境で本書のコードをそのまま使用してください。詳細は本書 注7.14 (Databricks Agent Framework のドキュメント) を参照してください。

## リスト7.10 QAエージェントのロールバック

- **対応ファイル**: 本リポジトリ未収録 (単独スニペット)
- **試したい場合**: 本書のコードをそのまま実行できます。`previous_version` には実際のバージョン番号を指定してください。

# 7.6 応用 (1): ストリーミングと ResponsesAgent
# 7.7 応用 (2): カスタムアプリケーション統合

## リスト7.11 ストリーミングの実装
## リスト7.12 QAエージェントのFastAPIラッパー
## リスト7.13 GradioでデモのためのUIを構築

- **対応ファイル**: 本リポジトリ未収録
- **理由**: 本書 7.6節冒頭に「**本節と次節 (7.6、7.7) のコードはサンプルリポジトリに含まれていません。概念と実装パターンの紹介として参照してください。**」と明記されています。
- **試したい場合**:
  - リスト7.11 (ストリーミング): `serving/agent.py` に `@stream()` デコレータの実装を追加してください。
  - リスト7.12 (FastAPIラッパー): 別途 FastAPI プロジェクトを起動し、AI Gateway もしくは Agent Server を呼び出してください。
  - リスト7.13 (Gradio): `pip install gradio` 後、本書のコードをそのまま実行できます。

# 全体的な注意事項

- 本書の本文コードは「読んで理解するため」の抜粋です。実行可能な完全版は本リポジトリを参照してください。
- 7.4節 (AI Gateway) は MLflow のバージョンによって構成が大きく異なります。MLflow 3.10 以降では Tracking Server に統合された新方式が利用できますが、本リポジトリは互換性のため Legacy 方式の設定ファイルも提供しています。
- 7.6-7.7節 (応用編) は意図的にリポジトリ未収録です。
- 本ドキュメントに未記載の挙動差分や実装上の不整合を発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ随時更新します。
