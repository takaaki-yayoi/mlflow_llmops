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
│   ├── model_code.py       # 7.2: models-from-code用モデル定義
│   ├── log_model.py        # 7.2: モデル記録・レジストリ登録
│   ├── agent.py            # 7.3: Agent Server用ラッパー
│   ├── start_server.py     # 7.3: Agent Server起動
│   └── eval_serving.py     # 7.3.3: サービング中エージェントの評価
│
├── gateway/                # 第7章の新規設定
│   ├── gateway_config.yaml     # 7.4: AI Gateway基本設定
│   └── gateway_ab_test.yaml    # 7.4.4: A/Bテスト設定
│
└── deploy/                 # 第7章の新規設定
    ├── Dockerfile              # 7.5.2: Docker
    └── k8s/
        ├── deployment.yaml     # 7.5.3: Kubernetes Deployment
        └── service.yaml        # 7.5.3: Kubernetes Service
```

## セットアップ

### 1. 環境変数の設定

```bash
cp .env.template .env
```

`.env` を編集して `OPENAI_API_KEY` と `EXA_API_KEY` を入力してください。

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

### ステップ2: モデル記録・レジストリ登録（7.2節）

```bash
make log-model
```

以下が順に実行されます：
- QAエージェントをMLflowに記録（models-from-codeパターン）
- モデルレジストリに登録し `champion` エイリアスを設定
- ロードして推論テストで動作確認

### ステップ3: Agent Serverの起動（7.3節、別ターミナル）

```bash
make serve
```

ポート5005でResponses APIエンドポイントが起動します。

### ステップ4: テストリクエスト送信

```bash
make test-request
```

JSON形式のレスポンスが日本語で返れば成功です。

### ステップ5: サービング中エージェントの評価（7.3.3節、任意）

Agent Serverが起動している状態で実行してください。

```bash
make eval
```

3件の評価データで関連性・安全性・ガイドラインのスコアが表示されます。

### ステップ6: AI Gatewayの起動（7.4節、任意、別ターミナル）

```bash
make gateway
```

ポート5010でAI Gatewayが起動します。動作確認：

```bash
curl -s -X POST http://localhost:5010/gateway/qa-agent-llm/invocations \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}'
```

## ポート一覧

| ポート | 用途 |
|--------|------|
| 5000 | MLflow Tracking Server |
| 5005 | Agent Server（QAエージェント） |
| 5010 | AI Gateway |

## Makefileターゲット一覧

```
make install        # uv sync
make ingest         # Milvusにドキュメントを取り込み
make log-model      # QAエージェントをMLflowに記録（7.2）
make serve          # Agent Serverを起動（7.3）
make test-request   # curlでテストリクエスト送信（7.3）
make eval           # サービング中エージェントを評価（7.3.3）
make gateway        # AI Gatewayを起動（7.4）
make clean          # 生成ファイルの削除
```

## 注意事項

### setuptools について

`milvus-lite` が `pkg_resources`（setuptools の一部）に依存していますが、Python 3.12 以降では setuptools が標準インストールに含まれません。さらに setuptools v78 以降では `pkg_resources` が削除されています。そのため `pyproject.toml` で `setuptools<75` を明示的に指定しています。

### make ingest が終了しない場合

デフォルトではMLflowドキュメント全体をクロールするため時間がかかります。テスト用には `--max-pages` オプションを使用してください：

```bash
uv run python scripts/web_ingest.py --max-pages 20
```

### AI Gateway のバックアップエンドポイント

`gateway/gateway_config.yaml` にAnthropicをフォールバック用に設定するエンドポイントがコメントアウトされています。利用する場合は `ANTHROPIC_API_KEY` を `.env` に追加し、コメントを解除してください。

### 第4章コードとの関係

`agents/` と `scripts/` は第4章からそのままコピーしたものです。第7章固有の変更は `serving/`、`gateway/`、`deploy/` にのみ含まれます。
