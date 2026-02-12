# 第9章 チュートリアル サンプルノートブック

第9章「チュートリアル」のサンプルノートブックです。MLflowを活用したLLMアプリケーション開発の実践的なチュートリアルを3つのノートブックで提供します。

## ノートブック一覧

| ノートブック | テーマ | 学習内容 |
|------------|--------|----------|
| `9.1.ipynb` | 文書情報抽出モデルの構築 | Prompt Registry、Models from Code、カスタムPyFuncモデル、モデルエイリアス、サービング |
| `9.2.ipynb` | エージェント型RAGシステムの構築 | ベクトルDB(Chroma)、LangGraphワークフロー、動的ルーティング・再試行、MLflow Evaluation |
| `9.3.ipynb` | スーパーバイザー型マルチエージェント | マルチエージェント設計パターン、ResponsesAgent、Safety/Guidelines/カスタムスコアラー/Agent-as-a-Judge評価 |

## 前提条件

- Python 3.11以上
- OpenAI APIキー

## 実行方法

### Databricksで実行する場合

各 `.ipynb` ファイルをDatabricksワークスペースにインポートしてそのまま実行できます。

1. Databricksワークスペースで「インポート」を選択
2. `.ipynb` ファイルをアップロード
3. クラスターをアタッチして上から順にセルを実行

**注意**: `mlflow.set_tracking_uri()` の行はDatabricksでは不要です（自動設定されます）。

### ローカル環境(Jupyter)で実行する場合

```bash
# Jupyter Notebookを起動
pip install notebook
jupyter notebook
```

各ノートブック冒頭の `%pip install` セルで依存パッケージがインストールされます。`YOUR_API_KEY` を実際のOpenAI APIキーに置き換えてから実行してください。

## 各ノートブックの詳細

### 9.1 文書情報抽出モデルの構築

日本語のビジネス文書から構造化された情報（会社名、契約日、金額等）をJSON形式で抽出するモデルを構築します。

- **Prompt Registry**: プロンプトのバージョン管理と再利用
- **Models from Code**: `%%writefile` + `set_model()` でコードをMLflowモデルとして登録
- **カスタムPyFuncモデル**: `PythonModel` を継承した柔軟なモデル実装
- **モデルエイリアス**: `champion` エイリアスによるバージョン管理
- **サービング**: `mlflow models serve` によるREST API公開

### 9.2 エージェント型RAGシステムの構築

LangGraphで動的な判断と再試行ロジックを持つエージェント型RAGを構築します。

- **ベクトルDB構築**: Chromaを使ったドキュメントの埋め込みと検索
- **5ノードのワークフロー**: router → retrieve → check → rewrite → answer
- **条件分岐**: 質問内容に応じた検索要否の判定、検索結果の品質チェック
- **MLflow Tracing**: 各ノードの処理を自動記録
- **MLflow Evaluation**: Correctness、RetrievalSufficiencyによる自動評価

### 9.3 スーパーバイザー型マルチエージェントシステム

複数の専門エージェントが協調して技術レポートを作成するマルチエージェントシステムを構築します。

- **4つの専門エージェント**: リサーチ、構成、ライティング、レビュー
- **スーパーバイザー**: 状態に基づく次エージェントの動的決定
- **ResponsesAgent**: OpenAI互換の標準インターフェースでラッピング
- **多様な評価手法**:
  - Safety: 安全性の自動チェック
  - Guidelines: 自然言語による評価基準の定義
  - カスタムスコアラー: エージェント網羅性の評価
  - Agent-as-a-Judge: エージェント間協調性の総合評価
