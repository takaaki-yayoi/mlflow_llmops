# CLAUDE.md - 第7章サンプルコード開発ガイド

## プロジェクト概要

MLflow LLMOps書籍の第7章「本番環境に届ける - サービングとデプロイメント」のサンプルコードリポジトリ。

第4章で構築したMLflow QAエージェントを、Agent Serverでサービングし、AI Gatewayでプロバイダー管理し、本番デプロイする一連の実装を提供する。


## 最重要原則: 各章フォルダは自己完結

各章のサンプルコードは、そのフォルダ内だけで動作が完結すること。他の章のフォルダへのシンボリックリンクや相対パス参照は禁止。

- 第4章の`agents/`パッケージ(エージェント本体、ツール群)は第7章フォルダ内にコピーして含める
- `data/`ディレクトリの構築手順(ingest)も第7章フォルダ内で完結させる
- `.env.template`は第7章で必要なキーをすべて含める
- `pyproject.toml`の依存関係に他章のローカルパスを含めない


## 第4章からコピーすべきファイル

第4章サンプル(`section-4-add-tracing`)から以下を第7章フォルダにそのまま含める。

```
agents/
├── __init__.py
├── thread.py
└── langgraph/
    ├── __init__.py
    ├── agent.py
    └── tools/
        ├── __init__.py
        ├── doc_search.py
        ├── web_search.py
        └── open_url.py
scripts/
└── web_ingest.py
```

コピー時の注意:
- `agents/langgraph/agent.py`内の`mlflow.set_tracking_uri()`や`mlflow.set_experiment()`、`mlflow.langchain.autolog()`は第7章のサービング文脈で適切に動作するか確認する。サービング用スクリプト側で設定を上書きする設計でよい。
- `SYSTEM_PROMPT`のハードコードはそのまま残す。第7章の`serving/agent.py`でプロンプトレジストリからの取得を試み、失敗時にこのデフォルトにフォールバックする。


## ディレクトリ構成(最終形)

```
section-7-serving/
├── CLAUDE.md              # このファイル
├── README.md              # セットアップ・実行手順
├── pyproject.toml         # 依存関係(自己完結)
├── Makefile               # コマンド一覧
├── .env.template          # 環境変数テンプレート
├── .gitignore
│
├── agents/                # ← 第4章からコピー(変更なし)
│   ├── __init__.py
│   ├── thread.py
│   └── langgraph/
│       ├── __init__.py
│       ├── agent.py
│       └── tools/
│           ├── __init__.py
│           ├── doc_search.py
│           ├── web_search.py
│           └── open_url.py
│
├── scripts/               # ← 第4章からコピー(変更なし)
│   └── web_ingest.py
│
├── serving/               # ★ 第7章の新規コード
│   ├── __init__.py
│   ├── log_model.py       # 7.2: モデル記録・レジストリ登録
│   ├── agent.py           # 7.3: Agent Server用ラッパー
│   ├── start_server.py    # 7.3: Agent Server起動
│   └── eval_serving.py    # 7.3.3: サービング中エージェントの評価
│
├── gateway/               # ★ 第7章の新規設定
│   ├── gateway_config.yaml    # 7.4: AI Gateway基本設定
│   └── gateway_ab_test.yaml   # 7.4.4: A/Bテスト設定
│
└── deploy/                # ★ 第7章の新規設定
    ├── Dockerfile         # 7.5.2: Docker
    └── k8s/
        ├── deployment.yaml    # 7.5.3: Kubernetes Deployment
        └── service.yaml       # 7.5.3: Kubernetes Service
```


## 原稿との対応

原稿ファイル: Google Drive上の改訂版第7章原稿
原稿タイトル: 「本番環境に届ける - サービングとデプロイメント」

| 原稿セクション | 内容 | 対応するコード |
|---|---|---|
| 7.1 LLMサービングの課題 | 概念説明(コードなし) | なし |
| 7.2.1 サービング用のモデル記録 | `mlflow.langchain.log_model()` | `serving/log_model.py` の `log_agent()` |
| 7.2.2 プロンプトレジストリとの統合 | `mlflow.genai.load_prompt()` | `serving/agent.py` の `_load_system_prompt()` |
| 7.2.3 モデルレジストリへの登録 | `client.create_model_version()`, エイリアス設定 | `serving/log_model.py` の `register_model()` |
| 7.2.4 モデルの動作確認 | `mlflow.langchain.load_model()` で推論テスト | `serving/log_model.py` の `verify_model()` |
| 7.3.1 mlflow models serve | CLIコマンドのみ(コード不要) | `Makefile` の `serve-cli` ターゲット(任意) |
| 7.3.2 Agent Serverによるサービング | `@invoke`デコレータ、`AgentServer` | `serving/agent.py`, `serving/start_server.py` |
| 7.3.3 サービング環境での評価 | `mlflow.genai.evaluate()` | `serving/eval_serving.py` |
| 7.4.2 QAエージェント向けのGateway設定 | YAML設定 | `gateway/gateway_config.yaml` |
| 7.4.3 エージェントからGatewayを利用する | `base_url`の変更 | 原稿のコード例のみ(agent.pyへの組込は任意) |
| 7.4.4 A/Bテストとトラフィックルーティング | YAML設定 | `gateway/gateway_ab_test.yaml` |
| 7.5.2 Dockerコンテナとしてのデプロイ | Dockerfile | `deploy/Dockerfile` |
| 7.5.3 Kubernetesへのデプロイ | マニフェスト | `deploy/k8s/` |
| 7.5.4 Databricks Model Serving | Databricks固有(ローカル実行不可) | なし |
| 7.6 ストリーミングとResponsesAgent | `@stream`デコレータ | 応用。実装する場合は`serving/agent.py`に追加 |
| 7.7 カスタムアプリケーション統合 | FastAPI/Gradio | 応用。実装する場合は`serving/custom_app.py`等 |


## 技術仕様

### 使用ポート

| ポート | 用途 |
|---|---|
| 5000 | MLflow Tracking Server |
| 5005 | Agent Server(QAエージェント) |
| 5010 | AI Gateway |

### 必須環境変数

```
OPENAI_API_KEY          # LLM呼び出し(GPT-4o-mini)とEmbedding
EXA_API_KEY             # Web検索ツール
LLM_MODEL               # デフォルト: gpt-4o-mini
EMBEDDING_MODEL         # デフォルト: text-embedding-3-small
MLFLOW_TRACKING_URI     # デフォルト: http://localhost:5000
```

### 主要依存パッケージ

```
mlflow>=3.9.0           # Agent Server, プロンプトレジストリ, 評価
langgraph>=1.0.0        # エージェントフレームワーク
langchain>=1.0.0
langchain-openai>=1.0.0
langchain-milvus>=0.2.0 # ベクトル検索
milvus-lite>=2.4.0
exa-py>=1.0.0           # Web検索
uvicorn>=0.30.0         # Agent Server実行
fastapi>=0.110.0
python-dotenv>=1.0.0
```

### MLflow最低バージョンの根拠

- `mlflow.genai.agent_server` (Agent Server): MLflow 3.6.0で追加
- `mlflow.genai.load_prompt()` (プロンプトレジストリ): MLflow 3.x
- `mlflow.genai.evaluate()` + スコアラー: MLflow 3.x
- `ResponsesAgentRequest/Response`: MLflow 3.6.0で追加


## Makefileターゲット

```makefile
make install        # uv sync
make ingest         # Milvusにドキュメントを取り込み(第4章と同じ)
make log-model      # QAエージェントをMLflowに記録(7.2)
make serve          # Agent Serverを起動(7.3)
make test-request   # curlでテストリクエスト送信(7.3)
make eval           # サービング中エージェントを評価(7.3.3)
make gateway        # AI Gatewayを起動(7.4)
make clean          # 生成ファイルの削除
```


## 動作確認フロー

以下の順序で検証する。各ステップが成功してから次に進む。

1. `make install` → 依存関係エラーがないこと
2. `make ingest` → `data/milvus.db`が生成されること
3. MLflowサーバー起動(`mlflow server --port 5000`)
4. `make log-model` → モデル記録・レジストリ登録・推論テストがすべてパスすること
5. `make serve` → Agent Serverが5005ポートで起動すること
6. `make test-request` → JSON形式のレスポンスが返ること
7. MLflow UI → Tracesタブにサービング経由トレースが記録されていること
8. `make eval` → 3件の評価がすべて完了し、スコアが表示されること
9. `make gateway`(任意) → AI Gatewayが5010ポートで起動すること


## コード品質の基準

- すべてのPythonファイルにモジュールdocstringを含める
- docstringで原稿セクション番号を明記する(例: 「7.2節」)
- 第4章からコピーしたファイルには手を加えない(差分管理のため)
- 第7章の新規コード(`serving/`, `gateway/`, `deploy/`)のみ編集対象
- エラーメッセージは日本語で、原因と対処法を含める
