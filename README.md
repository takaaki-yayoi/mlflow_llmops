# MLflow LLMOps 書籍サンプルコード

MLflow を活用した LLM アプリケーションの開発・運用（LLMOps）を学ぶための書籍サンプルコードリポジトリです。

## 各章の概要

| 章 | テーマ | 内容 |
|----|--------|------|
| [ch3](ch3/) | LLMアプリケーションの構築 | LangGraphを使用したRAG対応QAエージェントの構築 |
| [ch4](ch4/) | 可観測性の確保 | MLflow Tracingによるトレーシングと可視化 |
| [ch5](ch5/) | 評価の仕組み | MLflow GenAIの評価機能による品質の体系的評価 |
| [ch7](ch7/) | サービングとデプロイメント | Agent Server、AI Gateway、本番デプロイ |
| [ch8](ch8/) | 監視と運用 | トレーシング、コスト管理、フィードバック、OpenTelemetry連携 |
| [ch9](ch9/) | チュートリアル | 文書情報抽出、エージェント型RAG、マルチエージェントの実践ノートブック |

## 前提条件

- Python 3.10以上（ch8, ch9は3.11以上）
- [uv](https://docs.astral.sh/uv/)（パッケージマネージャー）
- OpenAI APIキー
- Exa APIキー（Web検索機能を使用する場合）

## 環境設定

### 初回セットアップ

1. ルートの共通テンプレートから各章フォルダに `.env` をコピーします。

```bash
cp .env.template ch3/.env
```

2. コピーした `.env` ファイルを編集し、APIキーを設定します。

```bash
# .env を編集
OPENAI_API_KEY=your-api-key-here
EXA_API_KEY=your-exa-api-key-here
```

3. 各章のディレクトリに移動し、依存関係をインストールします。

```bash
cd ch3
make install
```

### 章を通して作業する場合

前の章の `.env` をそのままコピーすると、APIキーを再設定する手間が省けます。

```bash
cp ch3/.env ch4/.env
cp ch4/.env ch5/.env
```

各章のフォルダには章固有の `.env.template` も用意されていますので、そちらを使用することもできます。

```bash
cp ch4/.env.template ch4/.env
```

### MLflow Tracking Server（第4章以降）

第4章以降では MLflow Tracking Server が必要です。各章の作業前に別ターミナルで起動してください。

```bash
cd chX
uv run mlflow server --host 0.0.0.0 --port 5000
```
