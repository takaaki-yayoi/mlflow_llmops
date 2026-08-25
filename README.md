# MLflowで実践するLLMOps――生成AIアプリケーションの実験管理と品質保証 サンプルコード

[![CI](https://github.com/takaaki-yayoi/mlflow_llmops/actions/workflows/ci.yml/badge.svg)](https://github.com/takaaki-yayoi/mlflow_llmops/actions/workflows/ci.yml)

MLflow を活用した LLM アプリケーションの開発・運用 (LLMOps) を学ぶための書籍「MLflowで実践するLLMOps――生成AIアプリケーションの実験管理と品質保証」のサンプルコードリポジトリです。

**本文中のコードと差分がある場合、本リポジトリを優先してください。**

本リポジトリのコードは書籍本文のコードとは完全に一致しません。書籍本文のコードは紙面の都合で抜粋・簡略化されています。動作する完全なコードは本リポジトリを正としてください。MLflow のバージョンアップに追従するため、リポジトリ側は継続的に更新されます。

# リポジトリ全体のドキュメント

| ドキュメント | 内容 |
| --- | --- |
| [docs/external-refs.md](docs/external-refs.md) | 本書の脚注 / 注釈で参照されている MLflow 公式 URL の要約集。本書の流れを止めずに該当ページの要点を把握できます。 |
| [docs/why/](docs/why/README.md) | 章別「なぜこの機能が嬉しいのか」サマリ。各章を読む前後に、現場のどんな課題に効くのかを補強します。 |

# 章別の対応表 / 差分ノート (CHAPTER_NOTES.md)

リポジトリと本書本文で挙動や出力が異なる箇所、本書中で言及されていてもリポジトリには未収録の機能、応用編の意図的な未収録などについては、各章の `CHAPTER_NOTES.md` で詳細を解説しています。

| 章 | 対応表 / 差分ノート | なぜ嬉しいのか |
| --- | --- | --- |
| 第3章 | [ch3/CHAPTER_NOTES.md](ch3/CHAPTER_NOTES.md) | [docs/why/ch3.md](docs/why/ch3.md) |
| 第4章 | [ch4/CHAPTER_NOTES.md](ch4/CHAPTER_NOTES.md) | [docs/why/ch4.md](docs/why/ch4.md) |
| 第5章 | [ch5/CHAPTER_NOTES.md](ch5/CHAPTER_NOTES.md) | [docs/why/ch5.md](docs/why/ch5.md) |
| 第6章 | [ch6/CHAPTER_NOTES.md](ch6/CHAPTER_NOTES.md) | [docs/why/ch6.md](docs/why/ch6.md) |
| 第7章 | [ch7/CHAPTER_NOTES.md](ch7/CHAPTER_NOTES.md) | [docs/why/ch7.md](docs/why/ch7.md) |
| 第8章 | [ch8/CHAPTER_NOTES.md](ch8/CHAPTER_NOTES.md) | [docs/why/ch8.md](docs/why/ch8.md) |
| 第9章 | [ch9/CHAPTER_NOTES.md](ch9/CHAPTER_NOTES.md) | [docs/why/ch9.md](docs/why/ch9.md) |

本書記載と挙動が合わない箇所、上記ドキュメントに未記載の差分などを発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ、随時 `CHAPTER_NOTES.md` を更新します。「なぜ嬉しいのか」への補足や反論は `why-discussion` ラベルでフィードバックを歓迎します。

## 各章の概要

| 章 | テーマ | 内容 |
|----|--------|------|
| [ch3](ch3/) | MLflowのインストールと初期設定 | LangGraphを使用したRAG対応QAエージェントのセットアップと実行 |
| [ch4](ch4/) | 可観測性の確保 | MLflow Tracingによるトレーシングと可視化 |
| [ch5](ch5/) | 改善サイクルを加速する | MLflow GenAIの評価機能による品質の体系的評価 |
| [ch6](ch6/) | プロンプトエンジニアリング | Prompt Registryによるバージョン管理、評価、自動最適化 |
| [ch7](ch7/) | 本番環境で動かす | Agent Server、AI Gateway、本番デプロイ |
| [ch8](ch8/) | 監視と運用 | トレーシング、コスト管理、フィードバック、OpenTelemetry連携 |
| [ch9](ch9/) | 実践ケーススタディ | 文書情報抽出、エージェント型RAG、マルチエージェントの実践ノートブック |

## 前提条件

- Python 3.10以上（ch8, ch9は3.11以上）
- [uv](https://docs.astral.sh/uv/)（パッケージマネージャー）
- OpenAI APIキー（チャット用モデルを他プロバイダーに変更した場合も、ドキュメント検索の埋め込みモデルで使用します。詳細は [ch3/CHAPTER_NOTES.md](ch3/CHAPTER_NOTES.md) の3.4節の項を参照）
- Exa APIキー（Web検索機能を使用する場合）

## 環境設定

### 初回セットアップ

1. リポジトリルートで `.env` を作成し、APIキーを設定します（初回のみ）。

```bash
cp .env.template .env
```

`.env` を編集して API キーを入力してください。

```
OPENAI_API_KEY=your-api-key-here
EXA_API_KEY=your-exa-api-key-here
```

2. 作業する章のディレクトリにコピーします。

```bash
cp .env ch3/.env
```

3. 各章のディレクトリに移動し、依存関係をインストールします。

```bash
cd ch3
make install
```

### 別の章に進むとき

ルートの `.env` を再度コピーするだけで済みます。APIキーの再設定は不要です。

```bash
cp .env ch4/.env
cp .env ch5/.env
```

各章のフォルダには章固有の `.env.template` も用意されていますので、そちらを使用することもできます。

### MLflow Tracking Server（第4章以降）

第4章以降では MLflow Tracking Server が必要です。各章の作業前に別ターミナルで起動してください。

```bash
cd chX
uv run mlflow server --host 0.0.0.0 --port 5000
```
