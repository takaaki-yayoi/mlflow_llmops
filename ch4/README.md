# MLflow サンプルエージェント

LangGraphを使用したAIエージェントのサンプル実装です。

## 概要

このリポジトリは、LangGraphフレームワークを使用したAIエージェントのリファレンス実装を提供します。

## ディレクトリ構成

```
sample-agent/
├── agents/
│   ├── thread.py           # スレッド管理
│   └── langgraph/          # LangGraph エージェント実装
│       ├── agent.py        # メインエージェントクラス
│       └── tools/          # ツール定義
├── cli/
│   └── main.py             # CLI インターフェース
├── scripts/                # ユーティリティスクリプト
├── data/                   # ベクトルストアデータ
├── Makefile                # コマンド
├── pyproject.toml          # 依存関係
└── .env.template           # 設定テンプレート
```

## クイックスタート

```bash
# 依存関係のインストール
make install

# 環境変数の設定
cp .env.template .env
# .env ファイルにAPIキーを設定してください

# (オプション) ドキュメントのRAG用取り込み
make ingest

# CLI の起動
make cli
```

## コマンド

| コマンド | 説明 |
|---------|------|
| `make install` | 依存関係をインストール |
| `make cli` | CLI エージェントを起動 |
| `make ingest` | ドキュメントをベクトルストアに取り込み |
| `make clean` | 生成されたファイルを削除 |
