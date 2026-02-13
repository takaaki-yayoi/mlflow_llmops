# 第6章 Prompt Registry - プロンプトエンジニアリングの体系化 サンプルコード

第6章「Prompt Registry - プロンプトエンジニアリングの体系化」のサンプルコードです。MLflow Prompt Registryを使ったプロンプトのバージョン管理、評価、最適化の一連のワークフローを実装します。

## 概要

第5章の評価で改善点が見つかったQAエージェントのプロンプトを、Prompt Registryで管理しながら改善していくストーリー駆動型のデモです。

1. **プロンプトの登録** → 初期プロンプトをレジストリに登録
2. **バージョン更新** → 改善版プロンプトをv2として登録
3. **エイリアス管理** → development/productionで環境別に管理
4. **評価** → v1 vs v2の品質を定量比較
5. **最適化** → MetaPrompt/GEPAによる自動改善
6. **デプロイ** → staging→production昇格、ロールバック
7. **応用機能** → モデルパラメータ、構造化出力

## セットアップ

### 前提条件

- Python 3.10以上
- uv（パッケージマネージャー）
- OpenAI APIキー（04以降で必要）

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
| `OPENAI_API_KEY` | LLM呼び出し（評価・最適化・構造化出力） | 04以降で必要 |

### MLflow Tracking Serverの起動

```bash
uv run mlflow server --host 0.0.0.0 --port 5000
```

## 実行方法

### 基本フロー（01〜04を順番に実行）

#### 6.2節: プロンプトの登録

QAエージェントのシステムプロンプトをPrompt Registryに登録します。

```bash
make register
```

#### 6.2節: バージョン更新

改善版プロンプトをv2として登録し、不変性を確認します。

```bash
make version
```

#### 6.2節: エイリアスによるライフサイクル管理

development/productionエイリアスの設定と@latestの確認を行います。

```bash
make alias
```

#### 6.3節: プロンプトの評価（OPENAI_API_KEY必要）

v1とv2のプロンプトをそれぞれ評価し、改善効果を定量比較します。

```bash
make eval
```

### 応用（OPENAI_API_KEY必要）

#### 6.5節: MetaPromptによる構造改善

MetaPromptOptimizerでプロンプトの構造を自動改善します。

```bash
make optimize-meta
```

#### 6.5節: GEPAによる反復最適化（コスト注意）

GepaPromptOptimizerでプロンプトを反復的に最適化します。LLMを多数回呼び出すため、コストに注意してください。

```bash
make optimize-gepa
```

#### 6.6節: 段階的デプロイとロールバック

staging→production昇格、ロールバック、タグによるガバナンスをデモします。

```bash
make deploy
```

#### 6.7節: モデルパラメータの紐付け

プロンプトと共にモデル名やパラメータを保存します。

```bash
make model-config
```

#### 6.7節: 構造化出力（OPENAI_API_KEY必要）

Pydanticモデルで出力形式を定義し、OpenAI APIで構造化出力を取得します。

```bash
make structured
```

### 一括実行

基本フロー（01〜04）+ deploy + model-config + structuredを順番に実行します（optimize-meta/optimize-gepaは除外）。

```bash
make all
```

## コマンド一覧

| コマンド | 説明 | 対応セクション |
|---------|------|--------------|
| `make install` | 依存関係をインストール | - |
| `make register` | プロンプトの登録 | 6.2 |
| `make version` | バージョン更新（改善版プロンプト） | 6.2 |
| `make alias` | エイリアスによるライフサイクル管理 | 6.2 |
| `make eval` | v1 vs v2の評価比較（OPENAI_API_KEY必要） | 6.3 |
| `make optimize-meta` | MetaPromptによる構造改善（OPENAI_API_KEY必要） | 6.5 |
| `make optimize-gepa` | GEPAによる反復最適化（OPENAI_API_KEY必要） | 6.5 |
| `make deploy` | 段階的デプロイとロールバック | 6.6 |
| `make model-config` | モデルパラメータの紐付け | 6.7 |
| `make structured` | 構造化出力（OPENAI_API_KEY必要） | 6.7 |
| `make all` | 基本フロー + deploy + model-config + structured | - |
| `make clean` | MLflowデータを削除 | - |

## ファイル構成

```
ch6/
├── prompts/
│   ├── __init__.py
│   ├── 01_register_prompt.py         # 6.2: プロンプトの登録
│   ├── 02_version_update.py          # 6.2: バージョン更新(改善版プロンプト)
│   ├── 03_alias_management.py        # 6.2: エイリアスによるライフサイクル管理
│   ├── 04_evaluate_prompt.py         # 6.3: 改善したプロンプトで評価を実行
│   ├── 05_optimize_metaprompt.py     # 6.5: MetaPromptによる構造改善
│   ├── 06_optimize_gepa.py           # 6.5: GEPAによる反復最適化
│   ├── 07_deploy_lifecycle.py        # 6.6: 段階的デプロイとロールバック
│   ├── 08_model_config.py            # 6.7: モデルパラメータの紐付け
│   └── 09_structured_output.py       # 6.7: 構造化出力
├── data/
│   ├── __init__.py
│   └── eval_dataset.py               # 評価用データセット (04, 05, 06で共有)
├── Makefile
├── pyproject.toml
├── README.md
├── .env.template
└── .gitignore
```
