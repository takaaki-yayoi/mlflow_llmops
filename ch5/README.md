# 第5章 改善サイクルを加速する - 評価の仕組み サンプルコード

第5章「改善サイクルを加速する - 評価の仕組み」のサンプルコードです。第4章のQAエージェントにMLflow GenAIの評価機能を追加し、エージェントの品質を体系的に評価する仕組みを構築します。

## 概要

第4章のQAエージェントに対して、以下の評価機能を追加しています。

- **Vibe Check**: テスト質問を実行してトレースを目視確認（5.2節）
- **標準スコアラー**: Correctness、ToolCallCorrectnessによる自動評価（5.4.4節）
- **カスタムスコアラー**: ルールベース、Guidelines、make_judgeによる独自評価指標（5.4.5節）
- **スコアラー登録**: MLflowへの評価指標のバージョン管理（5.4.6節）
- **自動評価**: データセット定義とmlflow.genai.evaluate()の実行（5.5-5.6節）
- **会話シミュレーション**: ConversationSimulatorによる複数ターン評価（5.7節）

第4章との差分は `evaluation/` ディレクトリの追加が主な変更点です。`agents/`、`cli/`、`scripts/` は第4章と同一です。

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

リポジトリルートの共通テンプレートからコピーする方法（推奨）：

```bash
cp ../.env.template .env
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

### MLflow Tracking Serverの起動

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

### ドキュメントの取り込み

```bash
make ingest
```

`data/milvus.db` が生成されれば成功です。

## 実行方法

原稿のセクション順に実行してください。

### 5.2節: Vibe Check

テスト質問を実行してMLflow UIでトレースを確認します。

```bash
make vibe-check
```

### 5.4.4節: 標準スコアラーの個別テスト

ToolCallCorrectness、Correctnessを個別にテストします。

```bash
make test-standard
```

### 5.4.5節: カスタムスコアラーの個別テスト

ルールベース、Guidelines、make_judgeの各スコアラーをテストします。

```bash
make test-custom
```

### 5.4.6節: スコアラーの登録

スコアラーをMLflowに登録してバージョン管理します。

```bash
make register
```

### 5.5-5.6節: 自動評価の実行（メインスクリプト）

評価データセットを定義し、mlflow.genai.evaluate()で自動評価を実行します。

```bash
make eval
```

### 5.7節: [応用] 会話シミュレーション

ConversationSimulatorによる複数ターンの会話シミュレーションと評価を実行します。

```bash
make sim
```

### 一括実行

vibe-check → eval を順番に実行します。

```bash
make eval-all
```

## コマンド一覧

| コマンド | 説明 | 対応セクション |
|---------|------|--------------|
| `make install` | 依存関係をインストール | - |
| `make ingest` | ドキュメントをベクトルストアに取り込み | - |
| `make vibe-check` | テスト質問を実行してトレースを確認 | 5.2 |
| `make test-standard` | 標準スコアラーの個別テスト | 5.4.4 |
| `make test-custom` | カスタムスコアラーの個別テスト | 5.4.5 |
| `make register` | スコアラーをMLflowに登録 | 5.4.6 |
| `make eval` | 自動評価の実行（メインスクリプト） | 5.5-5.6 |
| `make sim` | 会話シミュレーション | 5.7 |
| `make eval-all` | vibe-check → eval を順番に実行 | - |
| `make cli` | CLIエージェントを起動 | - |
| `make clean` | 生成されたファイルを削除 | - |

## ファイル構成

```
ch5/
├── agents/                        # ch4からコピー (変更なし)
│   ├── __init__.py
│   ├── thread.py                  # スレッド・メッセージ管理
│   └── langgraph/
│       ├── __init__.py
│       ├── agent.py               # LangGraphエージェント本体
│       └── tools/
│           ├── __init__.py
│           ├── doc_search.py      # Milvusベクトル検索
│           ├── web_search.py      # Exa Web検索
│           └── open_url.py        # URL コンテンツ取得
├── cli/                           # ch4からコピー (変更なし)
│   └── main.py                    # CLIインターフェース
├── scripts/
│   └── web_ingest.py              # ドキュメント取り込みスクリプト
├── evaluation/                    # ★ 第5章の新規コード
│   ├── __init__.py
│   ├── scorers.py                 # 全スコアラーの定義を集約
│   ├── 01_vibe_check.py           # 5.2: テスト質問の実行と確認
│   ├── 02_standard_scorers.py     # 5.4.4: 標準評価指標の個別テスト
│   ├── 03_custom_scorers.py       # 5.4.5: カスタム評価指標の実装
│   ├── 04_register_scorers.py     # 5.4.6: 評価指標の登録
│   ├── 05_run_evaluation.py       # 5.5-5.6: データセット定義と自動評価の実行
│   └── 06_conversation_sim.py     # 5.7: [応用] 会話シミュレーション
├── Makefile
├── pyproject.toml
├── README.md
├── .env.template
└── .gitignore
```
