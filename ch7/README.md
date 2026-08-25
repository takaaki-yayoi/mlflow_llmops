# 第7章 サンプルコード: 本番環境に届ける - サービングとデプロイメント

第4章で構築したMLflow QAエージェントを、Agent Serverでサービングし、AI Gatewayでプロバイダー管理し、本番デプロイする一連の実装を提供します。

本ディレクトリは**自己完結型**です。第4章のコード（`agents/`、`scripts/`）を内包しており、他の章のフォルダへの依存はありません。

## 前提条件

- Python 3.10以上
- [uv](https://docs.astral.sh/uv/) がインストール済みであること
- OpenAI APIキー（`OPENAI_API_KEY`）
- Exa APIキー（`EXA_API_KEY`） - [exa.ai](https://exa.ai) で取得

## ディレクトリ構成

```
ch7/
├── README.md
├── pyproject.toml          # 依存関係（自己完結）
├── Makefile                # コマンド一覧
├── .env.template           # 環境変数テンプレート
├── .gitignore
│
├── agents/                 # 第4章からコピー（変更なし）
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
├── scripts/                # 第4章からコピー（変更なし）
│   └── web_ingest.py
│
├── serving/                # 第7章の新規コード
│   ├── __init__.py
│   ├── agent.py            # 7.2: Agent Server用エージェント定義
│   ├── start_server.py     # 7.2: Agent Server起動
│   ├── eval_serving.py     # 7.3: サービング環境での評価
│   ├── model_code.py       # Note: models-from-code用モデル定義（任意）
│   └── log_model.py        # Note: モデル記録・レジストリ登録（任意）
│
├── gateway/                # 7.4: AI Gateway設定（Legacy方式用）
│   ├── gateway_config.yaml     # 基本設定
│   └── gateway_ab_test.yaml    # A/Bテスト設定
│
└── deploy/                 # 7.5: 本番デプロイメント
    ├── Dockerfile              # 7.5.2: Docker
    └── k8s/
        ├── deployment.yaml     # 7.5.3: Kubernetes Deployment
        └── service.yaml        # 7.5.3: Kubernetes Service
```

## セットアップ

### 1. 環境変数の設定

リポジトリルートで設定済みの `.env` をコピーする方法（推奨）：

```bash
cp ../.env .env
```

または、章固有のテンプレートからコピーすることもできます：

```bash
cp .env.template .env
```

`.env` を編集して `OPENAI_API_KEY` と `EXA_API_KEY` を入力してください。

> **注意**: 書籍3.4節の手順でチャット用モデルをAnthropicなど他プロバイダーに変更しても、埋め込みモデル（`scripts/web_ingest.py` と `agents/langgraph/tools/doc_search.py` の `OpenAIEmbeddings`）はOpenAIのままのため、`OPENAI_API_KEY` は引き続き必要です。埋め込みモデルも変更する場合は [ch3/CHAPTER_NOTES.md](../ch3/CHAPTER_NOTES.md) の3.4節の項を参照してください。

### 2. 依存関係のインストール

```bash
make install
```

### 3. ドキュメントの取り込み（MilvusベクトルDB構築）

```bash
make ingest
```

`data/milvus.db` が生成されれば成功です。全ページ取得には時間がかかるので、テスト用にページ数を制限する場合：

```bash
uv run python scripts/web_ingest.py --max-pages 20
```

## 実行手順

### ステップ1: MLflow Tracking Serverの起動（別ターミナル）

```bash
uv run mlflow server --port 5000
```

http://localhost:5000 でUIが表示されることを確認してください。

### ステップ2: Agent Serverの起動（7.2節、別ターミナル）

```bash
make serve
```

ポート5005でResponses APIエンドポイントが起動します。`@invoke()`デコレータで登録されたエージェント関数が`/invocations`で公開され、トレーシングも自動的に有効になります。

### ステップ3: テストリクエスト送信（7.2節）

```bash
make test-request
```

JSON形式のレスポンスが日本語で返れば成功です。MLflow UI（http://localhost:5000）の「Traces」タブにトレースが記録されていることも確認してください。

### ステップ4: サービング環境での評価（7.3節、任意）

Agent Serverと同じ`@invoke`関数をin-processで実行して評価します。Milvusデータベースのファイルロックが競合するため、**Agent Serverを停止してから**実行してください。

```bash
# ステップ2で起動したAgent ServerをCtrl+Cで停止してから実行
make eval
```

3件の評価データで関連性・安全性・ガイドラインのスコアが表示されます。評価後、Agent Serverは`make serve`で再起動できます。

### （任意）モデル記録・レジストリ登録

Databricks Model Servingへのデプロイ時など、Model Registryを使う場合に実行します。Agent Serverのみで運用する場合は不要です。詳細は本書のNote「models-from-codeとModel Registry」および第10章を参照してください。

```bash
make log-model
```

以下が順に実行されます：
- QAエージェントをMLflowに記録（models-from-codeパターン）
- モデルレジストリに登録し `champion` エイリアスを設定
- ロードして推論テストで動作確認

### ステップ5: AI Gatewayのセットアップ（7.4節、任意）

本書の7.4節では、MLflow 3.10で刷新された**新AI Gateway（Tracking Server統合型）**を解説しています。新AI GatewayではMLflow UIからエンドポイントの作成・管理を行います。

#### 新方式（推奨）: MLflow UIからセットアップ

ステップ1で起動したTracking Serverには、AI Gatewayが組み込まれています。

1. ブラウザで http://localhost:5000/#/gateway にアクセス
2. **API Keys** タブで OpenAI APIキーを登録:
   - 名前: `openai-key`
   - プロバイダー: OpenAI
   - APIキーの値を入力
3. **Endpoints** タブでエンドポイントを作成:
   - `qa-agent-llm` — OpenAI / gpt-4o-mini / openai-key
   - `qa-agent-embedding` — OpenAI / text-embedding-3-small / openai-key
4. 動作確認:

```bash
make test-gateway
```

#### Legacy方式: YAML設定ファイルからセットアップ

UIを使わずにコマンドラインでセットアップしたい場合や、設定をバージョン管理したい場合は、YAML設定ファイルによるLegacy方式も利用できます。

```bash
make gateway-legacy
```

ポート5010でAI Gatewayが別プロセスとして起動します。動作確認：

```bash
curl -s -X POST http://localhost:5010/gateway/qa-agent-llm/invocations \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}'
```

> **注**: 新方式とLegacy方式ではアクセスURLが異なります。
> - 新方式: `http://localhost:5000/gateway/qa-agent-llm/mlflow/invocations`
> - Legacy: `http://localhost:5010/gateway/qa-agent-llm/invocations`

## ポート一覧

| ポート | 用途 |
|--------|------|
| 5000 | MLflow Tracking Server（AI Gateway含む） |
| 5005 | Agent Server（QAエージェント） |
| 5010 | AI Gateway Legacy方式（`make gateway-legacy`使用時のみ） |

## Makefileターゲット一覧

```
make install          # uv sync
make ingest           # Milvusにドキュメントを取り込み
make serve            # Agent Serverを起動（7.2）
make test-request     # curlでテストリクエスト送信（7.2）
make eval             # サービング環境のエージェントを評価（7.3、Agent Server停止後に実行）
make log-model        # QAエージェントをMLflowに記録（任意、Note参照）
make test-gateway     # AI Gateway経由でテストリクエスト送信（7.4 新方式）
make gateway-legacy   # AI Gatewayを起動（7.4 Legacy方式）
make clean            # 生成ファイルの削除
```

## 注意事項

### setuptools について

`milvus-lite` が `pkg_resources`（setuptools の一部）に依存していますが、Python 3.12 以降では setuptools が標準インストールに含まれません。さらに setuptools v78 以降では `pkg_resources` が削除されています。そのため `pyproject.toml` で `setuptools<75` を明示的に指定しています。

### make ingest が終了しない場合

デフォルトではMLflowドキュメント全体をクロールするため時間がかかります。テスト用には `--max-pages` オプションを使用してください：

```bash
uv run python scripts/web_ingest.py --max-pages 20
```

### make eval でMilvusエラーが出る場合

`make eval`はAgent Serverと同じエージェントコードをin-processで実行します。Milvus Liteはファイルレベルのロックを使用するため、Agent Serverが`data/milvus.db`を開いている状態では評価スクリプトからアクセスできません。Agent Serverを停止してから`make eval`を実行してください。

### AI Gateway のバックアップエンドポイント（Legacy方式）

`gateway/gateway_config.yaml` にAnthropicをフォールバック用に設定するエンドポイントがコメントアウトされています。利用する場合は `ANTHROPIC_API_KEY` を `.env` に追加し、コメントを解除してください。新方式では、フォールバックはMLflow UIのエンドポイント設定画面から構成します。

### 第4章コードとの関係

`agents/` と `scripts/` は第4章からそのままコピーしたものです。第7章固有の変更は `serving/`、`gateway/`、`deploy/` にのみ含まれます。
