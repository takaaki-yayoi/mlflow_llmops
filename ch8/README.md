# 第8章 監視と運用 - サンプルノートブック

第8章で解説している監視・運用機能の実践的なサンプルコードです。

## ファイル構成

| ファイル | 説明 |
|---------|------|
| `chapter8_samples.py` | サンプルノートブック (Databricks notebook形式) |

## 前提条件

- Databricks Runtime 15.4 LTS ML以降
- MLflow 3.x (Databricks Runtime MLに含まれる)
- Unity Catalog対応ワークスペース
- OpenAI APIキー (コスト分析以外の機能で必要)

## 使用方法

1. Databricksワークスペースにノートブックをインポート
2. 冒頭の設定変数を自環境に合わせて変更:
   ```python
   CATALOG = "my_catalog"
   SCHEMA = "my_schema"
   TRACE_TABLE = "archived_traces"
   SERVICE_NAME = "chapter8-demo"
   ```
3. 各セルを順番に実行

## ノートブック構成

| セクション | 本文参照 | 内容 |
|-----------|---------|------|
| 1. セットアップ | - | 環境設定とエクスペリメント作成 |
| 2.1 効果的なスパン設計 | 8.1.2 | RAGパイプラインのスパン設計例 |
| 2.2 メタデータ追加 | 8.1.6 | トレースへのタグ・属性追加 |
| 2.3 Delta Tableアーカイブ | 8.1.5 | enable_databricks_trace_archival |
| 3.1 トークン使用量確認 | 8.2.2 | trace.info.token_usage |
| 3.2 コスト計算 | 8.2.3 | calculate_cost関数 |
| 3.3 SQLコスト分析 | 8.2.4 | 日次/モデル別/ユーザー別/異常検出 |
| 4.1 フィードバック記録 | 8.3.3 | mlflow.log_feedback |
| 4.2 スコアラー登録 | 8.3.4 | Safety, Guidelines |
| 4.3 スコアラー確認 | 8.3.4 | list_scorers |
| 5. クリーンアップ | - | スコアラー削除、アーカイブ停止 |

## SQLコスト分析クエリ

セクション3.3では以下のSQLクエリを提供しています:

| クエリ | 説明 |
|-------|------|
| 日次コストサマリー | 過去30日間の日次コスト集計 |
| モデル別コスト内訳 | モデルごとのコスト割合 |
| ユーザー別コストTop 10 | コスト上位ユーザー |
| 時間帯別コスト分析 | ピーク時間帯の特定 |
| コスト異常検出 | 平均+3標準偏差での異常検出 |

**注意**: これらのクエリはDelta Tableへのアーカイブが有効になっている必要があります。

## 注意事項

- **料金情報**: 記載の料金は2024年12月時点の参考値です。最新の公式料金を確認してください
- **スコアラー**: 実際のLLM呼び出しが必要な機能です
- **アーカイブ**: 約15分間隔でトレースが同期されます

## 関連ドキュメント

- [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/)
- [Production Monitoring](https://docs.databricks.com/aws/ja/mlflow3/genai/eval-monitor/production-monitoring)
- [Tracing FAQ](https://docs.databricks.com/gcp/en/mlflow3/genai/tracing/faq)
