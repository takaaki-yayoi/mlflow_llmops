# 第4章 リスト ↔ リポジトリ対応表 と 既知の挙動差分

本書 第4章「可観測性の確保 - トレーシングと評価」の各リスト (コード片) と、本リポジトリの実装の対応表です。

# 4.1 章の位置づけ

第4章は第3章 (ch3) で構築した QA エージェントに **MLflow Tracing を追加するだけ**の章です。本リポジトリの ch4 は ch3 の完全コピーで、`agents/langgraph/agent.py` の冒頭に以下 3 行が追加されているだけが実装上の差分です。

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow QAエージェント")
mlflow.langchain.autolog()
```

ch3 と共通の差分メモは [ch3/CHAPTER_NOTES.md](../ch3/CHAPTER_NOTES.md) を参照してください。本ドキュメントでは **ch3 との差分・トレーシング固有の注意点** に絞って解説します。

# 4.2 MLflow Tracking Server の起動

- **対応コマンド**: `uv run mlflow server --host 0.0.0.0 --port 5000`
- **差分**:
  * 本書は手順を順序立てて説明していますが、リポジトリでは README に直接コマンドが書かれており、別ターミナルで起動した状態で `make cli` を実行する流れになっています。
  * MLflow UI は `http://localhost:5000` でアクセスできます。

# 4.3 自動トレーシングの有効化

## `mlflow.langchain.autolog()`

- **対応箇所**: `agents/langgraph/agent.py` の冒頭 4 行 (上記)
- **差分**:
  * 本書は説明用に途中状態のコード片を含むことがありますが、リポジトリでは agent.py モジュール読み込み時に呼び出される最終形のみを掲載しています。
  * **モジュール読み込み時に Tracking Server に接続する**実装になっているため、Tracking Server を起動していない状態で `make cli` すると初回呼び出し時に MLflow が接続を試みて警告を出します (動作はします)。本書の手順通り Tracking Server を先に起動してください。
  * `mlflow.set_experiment()` の名前 ("MLflow QAエージェント") は本書記載と同じです。MLflow UI の Experiments で確認できます。

# 4.4 トレースの確認方法

- **対応箇所**: 実行後、MLflow UI の Traces タブで確認 (本書通り)
- **差分**: なし。本書の説明手順がそのまま使えます。

# 4.5 トレースに含まれる情報

- **対応**: `mlflow.langchain.autolog()` により、LangGraph の各ノード呼び出し・LLM 呼び出し・ツール呼び出しが自動的にトレースされます。
- **差分**: 本書では UI スクリーンショットで説明されている入出力の内容について、リポジトリでは特別な追加実装は不要です。`autolog()` のデフォルト挙動で本書の例と同等のトレースが記録されます。

# 4.6 トレースのカスタマイズ (任意)

- **対応**: 本リポジトリでは autolog のみを使用しており、`@mlflow.trace` デコレータでの手動計装は行っていません。
- **試したい場合**: 本書のリスト (autolog では捕捉されない処理に手動でトレースを付ける例) のコードを、`agents/langgraph/tools/*.py` の任意の関数にデコレータとして追加することで動作確認できます。

# トラブルシューティング

| 症状 | 原因 | 対処 |
| --- | --- | --- |
| MLflow UI にトレースが出ない | Tracking Server が未起動、または別のポートで起動している | `uv run mlflow server --host 0.0.0.0 --port 5000` で起動 |
| `Connection refused` の警告が出るが応答は返る | Tracking Server 未起動。トレース記録は失敗するが、エージェントの動作自体は継続する | Tracking Server を起動する |
| 既存の experiment と重複している | `set_experiment("MLflow QAエージェント")` で同名の experiment を再利用 | 既存の experiment は維持されます。問題ありません |

# 全体的な注意事項

- 本章は ch3 とのコード差分が極めて小さい (3 行のみ) ため、本書とリポジトリの間で混乱が起きにくい章です。
- ただし「Tracking Server を起動していないと autolog の効果が見えない」という点が、ch3 から ch4 に進んだ読者がつまずきやすいポイントです。本書の手順に厳密に従えば問題ありません。
- 本ドキュメントに未記載の挙動差分や実装上の不整合を発見された場合は、GitHub Issues で `errata` ラベルを付けて報告いただければ随時更新します。
