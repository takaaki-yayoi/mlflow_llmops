# 第3章 リスト ↔ リポジトリ対応表 と 既知の挙動差分

本書 第3章「LLMアプリケーションの構築」の各リスト (コード片) と、本リポジトリの実装の対応表です。本書は紙面の都合で動作環境設定や周辺コードを省略しており、ここではリポジトリと本書の差分のうち、**読者が混乱しやすい点・実行結果が本書と異なる点**に絞って解説します。

# 3.1 章の位置づけ

本章のサンプルコード ch3 は **MLflow Tracing を有効化していないベースライン実装**です。第4章 (ch4) で 3 行追加するとそのままトレーシング対応になります。ch3 と ch4 の差分は `agents/langgraph/agent.py` の冒頭 3 行のみであり、ベースが共通なので、本章の差分メモは ch4 にも概ね当てはまります。

# 3.2 LLMアプリケーションの全体像 (LangGraph によるエージェント)

## エージェント本体

- **対応ファイル**: `agents/langgraph/agent.py`
- **クラス**: `LangGraphAgent`
- **差分**:
  * 本書はエージェントの構築フローを段階的に解説しているため、コード片は途中状態のものが含まれます。
  * リポジトリは最終形のみを掲載しており、`StateGraph` の組み立て、`ToolNode` の登録、`MemorySaver` によるチェックポイント機構までを 1 ファイルにまとめています。
  * 環境変数 `ENABLE_WEB_SEARCH=false` を設定すると web_search ツールを除外できます。本書には記載されていない運用フックです。

## システムプロンプト

- **対応箇所**: `agents/langgraph/agent.py` 冒頭の `SYSTEM_PROMPT` 定数
- **差分**: 本書では引用箇所のみ抜粋。リポジトリでは「ツールから取得した情報を提供する際は必ずURLを含む引用を記載」のような運用上の指示を全文記載しています。

# 3.3 ツールの実装

## doc_search (ベクトル検索)

- **対応ファイル**: `agents/langgraph/tools/doc_search.py`
- **差分**:
  * Milvus は **milvus-lite** (組み込みモード) を使用。`data/milvus.db` がローカルに作成されます。本書での Milvus サーバ起動手順は不要です。
  * Embedding は OpenAI の `text-embedding-3-small`。ローカル運用しやすさを優先しています。

## web_search (Exa API)

- **対応ファイル**: `agents/langgraph/tools/web_search.py`
- **差分**:
  * `EXA_API_KEY` が未設定または `ENABLE_WEB_SEARCH=false` の場合、本ツールはエージェントから除外されます (`agent.py` の `[t for t in [...] if t is not None]`)。
  * 本書では Exa API キー必須の前提で書かれていますが、リポジトリでは無くても動くフォールバックを用意しています。

## open_url (URL コンテンツ取得)

- **対応ファイル**: `agents/langgraph/tools/open_url.py`
- **差分**: HTTP ステータスやエンコーディングのエラーハンドリングが本書より厚めに書かれています。挙動は同等。

# 3.4 ドキュメントの取り込み

- **対応ファイル**: `scripts/web_ingest.py`
- **実行**: `make ingest`
- **差分**:
  * 本書では Scrapy のスパイダー定義を抜粋。リポジトリでは MLflow ドキュメントをクロールする実行可能な完全版を提供しています。
  * `data/milvus.db` が出力です。再実行する場合は `make clean` で削除してから `make ingest` を実行してください。

# 3.5 CLI でのエージェント起動

- **対応ファイル**: `cli/main.py`
- **実行**: `make cli`
- **差分**: 本書はインターフェースの説明のみ。リポジトリでは対話ループ・スレッド管理・終了コマンドを含む実装になっています。

# 第4章との差分の見方

ch3 と ch4 の差分は `agents/langgraph/agent.py` の以下 3 行のみです。

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")
mlflow.langchain.autolog()
```

第3章を写経して動かしたあと、第4章ではこの 3 行を加えるだけでトレーシングが有効になります。詳細は [ch4/CHAPTER_NOTES.md](../ch4/CHAPTER_NOTES.md) を参照してください。

# 全体的な注意事項

- 本書本文のコード片は「読んで理解するため」の抜粋であり、実行可能な完全版は本リポジトリにあります。
- ツールの選択 (3 つすべて使うか、doc_search のみにするか等) は `agent.py` の `self.tools` リストを編集して制御できます。
- 本ドキュメントに未記載の挙動差分や実装上の不整合を発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ随時更新します。
