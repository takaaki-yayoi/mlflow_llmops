# 第6章 Prompt Registry - プロンプトエンジニアリングの体系化 サンプルコード

第6章「Prompt Registry - プロンプトエンジニアリングの体系化」のサンプルコードです。MLflow Prompt Registryを使ったプロンプトのバージョン管理、評価、最適化の一連のワークフローを実装します。

## 概要

MLflow Prompt Registryの各機能をデモする独立したスクリプト群です。

- **プロンプトの登録**: テンプレート変数を含むプロンプトの登録と埋め込み（6.2.1節）
- **バージョン更新**: プロンプトの改良と不変性の確認（6.2.2節）
- **エイリアス管理**: production/latestエイリアスとロールバック（6.2.3節）
- **タグとメタデータ**: プロンプト・バージョンへのタグ付与（6.2.4節）
- **モデルパラメータ**: プロンプトと共にモデル設定を保存（6.2.6節）
- **システムプロンプト登録**: QAエージェントのシステムプロンプトを登録し第7章と連携（6.4節）
- **構造化出力**: Pydanticモデルによる出力形式の定義（6.2.7節）
- **オフライン評価**: make_judgeとmlflow.genai.evaluate()による評価（6.3.1節）
- **プロンプト最適化**: GepaPromptOptimizerによる自動最適化（6.3.4節）

## セットアップ

### 前提条件

- Python 3.10以上
- uv（パッケージマネージャー）
- OpenAI APIキー（06〜08で必要）

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
| `OPENAI_API_KEY` | LLM呼び出し（構造化出力・評価・最適化） | 06〜08で必要 |

### MLflow Tracking Serverの起動

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

## 実行方法

原稿のセクション順に実行してください。01〜05はOPENAI_API_KEY不要です。

### 6.2.1節: プロンプトの登録

summarization-promptをPrompt Registryに登録し、変数の埋め込みを確認します。

```bash
make register
```

### 6.2.2節: バージョン更新と不変性

プロンプトを改良してバージョン2を登録し、不変性を確認します。

```bash
make version
```

### 6.2.3節: エイリアス管理

productionエイリアスの設定、ロールバック、復元を実行します。

```bash
make alias
```

### 6.2.4節: タグとメタデータ

プロンプトとバージョンにタグを付与します。

```bash
make tags
```

### 6.2.6節: モデルパラメータの保存

モデル名やパラメータをプロンプトと共に保存します。

```bash
make model-config
```

### 6.4節: QAエージェントのシステムプロンプト登録（第7章連携）

第3-4章のQAエージェントで使用するシステムプロンプトをPrompt Registryに登録します。第7章のAgent Serverと連携するため、`@production`エイリアスも設定します。

```bash
make system-prompt
```

### 6.2.7節: 構造化出力（OPENAI_API_KEY必要）

Pydanticモデルで出力形式を定義し、OpenAI APIで構造化出力を取得します。

```bash
make structured
```

### 6.3.1節: プロンプトのオフライン評価（OPENAI_API_KEY必要）

make_judgeでカスタム評価関数を定義し、評価データセットで評価します。

```bash
make eval
```

### 6.3.4節: プロンプト自動最適化（OPENAI_API_KEY必要、コスト注意）

GepaPromptOptimizerでプロンプトを自動最適化します。LLMを多数回呼び出すため、コストに注意してください。

```bash
make optimize
```

### 一括実行

01〜07と09を順番に実行します（08は除外）。

```bash
make all
```

## コマンド一覧

| コマンド | 説明 | 対応セクション |
|---------|------|--------------|
| `make install` | 依存関係をインストール | - |
| `make register` | プロンプトの登録 | 6.2.1 |
| `make version` | バージョン更新と不変性 | 6.2.2 |
| `make alias` | エイリアス管理 | 6.2.3 |
| `make tags` | タグとメタデータ | 6.2.4 |
| `make model-config` | モデルパラメータの保存 | 6.2.6 |
| `make system-prompt` | QAエージェントのシステムプロンプト登録（第7章連携） | 6.4 |
| `make structured` | 構造化出力（OPENAI_API_KEY必要） | 6.2.7 |
| `make eval` | プロンプトのオフライン評価（OPENAI_API_KEY必要） | 6.3.1 |
| `make optimize` | プロンプト自動最適化（OPENAI_API_KEY必要） | 6.3.4 |
| `make all` | 01→07, 09を順番に実行 | - |
| `make clean` | MLflowデータを削除 | - |

## ファイル構成

```
ch6/
├── prompts/                          # ★ 第6章の新規コード
│   ├── __init__.py
│   ├── 01_register_prompt.py         # 6.2.1: プロンプトの登録
│   ├── 02_version_update.py          # 6.2.2: バージョン更新と不変性
│   ├── 03_alias_management.py        # 6.2.3: エイリアス管理
│   ├── 04_tags_metadata.py           # 6.2.4: タグとメタデータ
│   ├── 05_model_config.py            # 6.2.6: モデルパラメータの保存
│   ├── 06_structured_output.py       # 6.2.7: 構造化出力
│   ├── 07_evaluate_prompt.py         # 6.3.1: プロンプトのオフライン評価
│   ├── 08_optimize_prompt.py         # 6.3.4: プロンプト自動最適化
│   └── 09_register_system_prompt.py  # 6.4: QAエージェントのシステムプロンプト登録
├── data/
│   ├── __init__.py
│   └── eval_dataset.py               # 評価用データセット (07, 08で共有)
├── Makefile
├── pyproject.toml
├── README.md
├── .env.template
└── .gitignore
```
