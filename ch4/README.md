# 第4章 可観測性の確保 サンプルコード

第4章「可観測性の確保 - トレーシングと評価」のサンプルコードです。第3章で構築したQAエージェントにMLflow Tracingを追加し、評価パイプラインを構築します。

## 概要

第3章のQAエージェントに対して、以下を追加しています。

- **MLflow Tracing**: `mlflow.langchain.autolog()` によるLLM呼び出し・ツール実行の自動トレース
- **MLflow Tracking Server**: トレース結果の記録・可視化

第3章との差分は `agents/langgraph/agent.py` の3行のみです。

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")
mlflow.langchain.autolog()
```

## セットアップ

### 前提条件

- Python 3.10以上
- uv（パッケージマネージャー）
- OpenAI APIキー
- Exa APIキー（Web検索を使用する場合）

### インストール

```bash
make install
```

### 環境変数の設定

```bash
cp .env.template .env
```

`.env` ファイルに以下のAPIキーを設定してください。

| 環境変数 | 用途 | 必須 |
|---------|------|------|
| `OPENAI_API_KEY` | LLM呼び出し・Embedding | はい |
| `EXA_API_KEY` | Web検索ツール | いいえ（`ENABLE_WEB_SEARCH=false`で無効化可） |

## 実行

### 1. ドキュメントの取り込み

```bash
make ingest
```

`data/milvus.db` が生成されれば成功です。

### 2. MLflow Tracking Serverの起動

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

### 3. CLIの起動

```bash
make cli
```

エージェントとの対話中、MLflow UI（http://localhost:5000）のTracesタブでトレースをリアルタイムに確認できます。

## コマンド一覧

| コマンド | 説明 |
|---------|------|
| `make install` | 依存関係をインストール |
| `make ingest` | ドキュメントをベクトルストアに取り込み |
| `make cli` | CLIエージェントを起動 |
| `make clean` | 生成されたファイルを削除 |

## ファイル構成

```
ch4/
├── agents/
│   ├── __init__.py
│   ├── thread.py              # スレッド・メッセージ管理
│   └── langgraph/
│       ├── __init__.py
│       ├── agent.py           # LangGraphエージェント本体（★トレーシング追加）
│       └── tools/
│           ├── __init__.py
│           ├── doc_search.py  # Milvusベクトル検索
│           ├── web_search.py  # Exa Web検索
│           └── open_url.py    # URL コンテンツ取得
├── cli/
│   └── main.py                # CLIインターフェース
├── scripts/
│   └── web_ingest.py          # ドキュメント取り込みスクリプト
├── Makefile
├── pyproject.toml
└── .env.template
```
