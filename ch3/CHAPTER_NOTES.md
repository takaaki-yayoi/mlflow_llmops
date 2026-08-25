# 第3章 リスト ↔ リポジトリ対応表 と 既知の挙動差分

本書 第3章「MLflowのインストールと初期設定」の各リスト (コード片) と、本リポジトリの実装の対応表です。本書は紙面の都合で周辺コードを省略しており、ここではリポジトリと本書の差分のうち、**読者が混乱しやすい点・実行結果が本書と異なる点**に絞って解説します。

# 章の位置づけ

本章のサンプルコード ch3 は **MLflow Tracing を有効化していないベースライン実装**です。第4章 (ch4) で `agents/langgraph/agent.py` に数行追加するとそのままトレーシング対応になります。ベースが共通なので、本章の差分メモは ch4 にも概ね当てはまります。

# 3.2 サンプルプロジェクトのセットアップ

## 3.2.1 サンプルエージェントの概要 (図3.1)

- **対応ファイル**: `agents/langgraph/agent.py` (エージェント本体)、`agents/langgraph/tools/` (3 つのツール)、`agents/thread.py` (スレッド管理)
- **差分**:
  * 本書は図3.1 で処理フローを示すのみで、エージェント本体のコードは掲載していません。リポジトリでは `StateGraph` の組み立て、`ToolNode` の登録、`MemorySaver` によるチェックポイント機構までを `agent.py` 1 ファイルにまとめています。
  * システムプロンプトは `agent.py` 冒頭の `SYSTEM_PROMPT` 定数です。「ツールから取得した情報を提供する際は必ずURLを含む引用を記載」のような運用上の指示を含みます。
  * `open_url` は本文の説明どおり、URL をシステムのデフォルトブラウザで開きます (`open` / `start` / `xdg-open`)。ページの内容をエージェントに取り込むツールではありません。

## 3.2.2 リポジトリのクローン (プロジェクト構成、表3.1)

- **対応ファイル**: `Makefile`
- **差分**: 表3.1 の `make clean` は `data/milvus.db` とキャッシュを削除します。`make ingest` をやり直す場合は先に `make clean` を実行してください。

# 3.3 サービスの設定とエージェントのテスト実行

## 3.3.1 APIキーの取得

- **差分**:
  * 本書では Exa API キーの設定は「オプション」としています。リポジトリでは `EXA_API_KEY` が未設定でも起動できるように、`ENABLE_WEB_SEARCH=false` を設定すると `web_search` ツールがエージェントから除外されます (`agents/langgraph/tools/__init__.py`)。`EXA_API_KEY` 未設定のまま `ENABLE_WEB_SEARCH=true` にしていると、Web 検索が呼ばれた時点でエラーになります。
  * `OPENAI_API_KEY` はチャット用モデルと埋め込みモデルの両方で使用します。3.4 節でチャット用モデルを他プロバイダーに変更しても、埋め込みモデルは OpenAI のままなので `OPENAI_API_KEY` は引き続き必要です (詳細は 3.4 節の項を参照)。

## 3.3.2 環境変数の設定 (リスト3.1)

- **対応ファイル**: `.env.template`
- **差分**: リポジトリの `.env.template` はリスト3.1 と同じ項目を含みます。`ENABLE_DOC_SEARCH=false` を設定すると `doc_search` ツールも除外できます。

## 3.3.3 ドキュメントデータの取り込み

- **対応ファイル**: `scripts/web_ingest.py`
- **実行**: `make ingest`
- **差分**:
  * ベクトルデータベースは本文どおり **Milvus Lite** (組み込みモード) を使用し、`data/milvus.db` がローカルに作成されます。外部の Milvus サーバは不要です。
  * 埋め込みモデルは `OpenAIEmbeddings` で、`.env` の `EMBEDDING_MODEL` (デフォルト `text-embedding-3-small`) を取り込み側 (`web_ingest.py`) と検索側 (`doc_search.py`) の両方が参照します。取り込み時と検索時で同じモデルを使う必要があるため、`EMBEDDING_MODEL` を変更した場合は `make clean` → `make ingest` を再実行してください。
  * 取得ページ数・チャンク数は MLflow ドキュメントの更新に伴って変動するため、本文の実行例 (293 ページ、1648 チャンク) とは一致しません。

## 3.3.4 エージェントの実行 (図3.3)

- **対応ファイル**: `cli/main.py`
- **実行**: `make cli`
- **差分**: 本書は `/quit` と `/new` のみ紹介しています。リポジトリの CLI は対話ループ・スレッド管理を含む実装で、起動時に有効なツール数とツール名を表示します。

# 3.4 応用 (1)：OpenAI以外のLLMを使用する場合 (リスト3.2〜3.4)

- **対応ファイル**: `agents/langgraph/agent.py` の `_build_graph()` 内 `ChatOpenAI(...)` の 1 行
- **差分・注意**:
  * 本書 3.4 節の手順 (リスト3.2〜3.4) で変更されるのは **チャット用モデルのみ**です。
  * ドキュメント検索用の **埋め込みモデルは以下 2 ファイルで `OpenAIEmbeddings` を使用したまま**のため、3.4 節の手順だけでは `OPENAI_API_KEY` が引き続き必要です。本書 3.3.1 (p.54) の「OpenAIの代わりに…3.4節で設定方法を説明しています」という記述は埋め込みモデルには当てはまりません (正誤表に掲載)。
    - `scripts/web_ingest.py`
    - `agents/langgraph/tools/doc_search.py`
  * 埋め込みモデルも変更する場合は、上記 2 ファイルの `OpenAIEmbeddings` を対象プロバイダーの Embeddings クラスに置き換えてください。取り込み時と検索時で同じモデルを使う必要があるため、変更後は `make clean` → `make ingest` を再実行してください。
  * Anthropic は埋め込みモデルを提供しておらず、公式ドキュメントでは [Voyage AI](https://docs.claude.com/en/docs/build-with-claude/embeddings) を案内しています。LangChain からは `langchain-voyageai` パッケージの `VoyageAIEmbeddings` を利用できます。

```python
# 例: 埋め込みモデルを Voyage AI に変更する場合 (web_ingest.py / doc_search.py の両方)
# uv add langchain-voyageai
# .env に VOYAGE_API_KEY=... を追加

# 変更前
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))

# 変更後
from langchain_voyageai import VoyageAIEmbeddings
embeddings = VoyageAIEmbeddings(model=os.environ.get("EMBEDDING_MODEL", "voyage-3.5"))
```

  * Google Gemini (`langchain-google-genai` の `GoogleGenerativeAIEmbeddings`)、Azure OpenAI (`langchain-openai` の `AzureOpenAIEmbeddings`)、Amazon Bedrock (`langchain-aws` の `BedrockEmbeddings`) も同様に置き換え可能です。モデル名は各プロバイダーの最新ドキュメントで確認してください。

# 第4章との差分の見方

ch3 と ch4 の差分は `agents/langgraph/agent.py` に追加された以下 4 行 (import 1 行 + 設定 3 行) のみです。

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")
mlflow.langchain.autolog()
```

第3章を動かしたあと、第4章ではこの 4 行を加えるだけでトレーシングが有効になります。詳細は [ch4/CHAPTER_NOTES.md](../ch4/CHAPTER_NOTES.md) を参照してください。

# 全体的な注意事項

- 本書本文のコード片は「読んで理解するため」の抜粋であり、実行可能な完全版は本リポジトリにあります。
- ツールの有効・無効は `.env` の `ENABLE_DOC_SEARCH` / `ENABLE_WEB_SEARCH` で制御できます。
- 本ドキュメントに未記載の挙動差分や実装上の不整合を発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ随時更新します。
