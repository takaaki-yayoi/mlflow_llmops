# 第3章 LLMアプリケーションの構築 サンプルコード

第3章「LLMアプリケーションの構築」のサンプルコードです。LangGraphを使用したRAG対応QAエージェントを構築します。

## 概要

MLflowドキュメントを検索ソースとしたQAエージェントの実装です。以下の3つのツールを備えています。

- **doc_search**: Milvusベクトルストアによるドキュメント検索
- **web_search**: Exa APIによるWeb検索
- **open_url**: 指定URLのコンテンツ取得

本章のエージェントはMLflowトレーシングなしのベースライン実装です。第4章でMLflow Tracingを追加します。

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

リポジトリルートで設定済みの `.env` をコピーする方法（推奨）：

```bash
cp ../.env .env
```

または、章固有のテンプレートからコピーすることもできます：

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

MLflowドキュメントをクロールしてMilvusベクトルストアに格納します。

```bash
make ingest
```

`data/milvus.db` が生成されれば成功です。

### 2. CLIの起動

```bash
make cli
```

対話的にエージェントに質問できます。

## コマンド一覧

| コマンド | 説明 |
|---------|------|
| `make install` | 依存関係をインストール |
| `make ingest` | ドキュメントをベクトルストアに取り込み |
| `make cli` | CLIエージェントを起動 |
| `make clean` | 生成されたファイルを削除 |

## ファイル構成

```
ch3/
├── agents/
│   ├── __init__.py
│   ├── thread.py              # スレッド・メッセージ管理
│   └── langgraph/
│       ├── __init__.py
│       ├── agent.py           # LangGraphエージェント本体
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

## 第4章との関係

本章（ch3）はトレーシングなしのベースラインです。第4章（ch4）では `agents/langgraph/agent.py` に以下の3行を追加してMLflow Tracingを有効化します。

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")
mlflow.langchain.autolog()
```
