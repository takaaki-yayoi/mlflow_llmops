# Databricks notebook source
# MAGIC %md
# MAGIC # 文書情報抽出モデルの構築 - MLflow Tutorial
# MAGIC
# MAGIC ## 概要
# MAGIC このノートブックでは、MLflowを使用して日本語のビジネス文書から情報を抽出するAIモデルを構築し、管理する方法を学びます。
# MAGIC
# MAGIC ### 学習内容
# MAGIC 1. MLflow Prompt Registryの使い方
# MAGIC 2. Models from Codeパターンの実装
# MAGIC 3. カスタムPyFuncモデルの作成
# MAGIC 4. モデルのバージョン管理とエイリアス設定
# MAGIC 5. モデルのサービング

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ1: 環境セットアップ
# MAGIC
# MAGIC ### 必要なライブラリのインストール
# MAGIC - `mlflow[genai]`: MLflowの生成AI機能（Prompt Registryなど）
# MAGIC - `openai`: OpenAI APIクライアント
# MAGIC - `pandas`: データフレーム操作用
# MAGIC
# MAGIC **注意**: インストール後、Pythonランタイムを再起動して変更を反映させます。

# COMMAND ----------

# MAGIC %pip install -U mlflow[genai] openai pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ2: 認証情報の設定
# MAGIC
# MAGIC OpenAI APIを使用するため、APIキーを環境変数に設定します。
# MAGIC
# MAGIC **重要**: 
# MAGIC - `YOUR_API_KEY`を実際のAPIキーに置き換えてください
# MAGIC - 本番環境では、Databricks Secretsを使用することを推奨します
# MAGIC   ```python
# MAGIC   os.environ["OPENAI_API_KEY"] = dbutils.secrets.get(scope="my-scope", key="openai-api-key")
# MAGIC   ```

# COMMAND ----------

import os

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ3: MLflow実験の設定とプロンプト登録
# MAGIC
# MAGIC ### MLflow Trackingとは？
# MAGIC 機械学習実験の記録・管理システムです。パラメータ、メトリクス、モデルを一元管理できます。
# MAGIC
# MAGIC ### Prompt Registryとは？
# MAGIC プロンプトをバージョン管理し、複数のモデルで再利用できる仕組みです。
# MAGIC プロンプトを変更した際の影響追跡や、A/Bテストが容易になります。

# COMMAND ----------

import mlflow

# MLflow Trackingの出力先を設定（ローカルサーバーの例）
# Databricks環境では自動的にワークスペースのMLflowを使用します
mlflow.set_tracking_uri("http://localhost:5000")

# 実験名を設定 - 関連する実行をグループ化します
mlflow.set_experiment("document_extraction_minimal")

# プロンプトの内容を文字列として用意
# このプロンプトは、LLMに対して構造化されたJSON形式で情報を抽出するよう指示します
template_text = """
あなたは日本語のビジネス文書から情報を抽出する補助者です。
入力として任意のテキストを与えます。
以下のJSON形式に従って、必ず有効なJSONだけを返してください。

- company_name: 会社名（不明な場合はnull）
- contract_start_date: 契約開始日（YYYY-MM-DD形式。不明な場合はnull）
- contract_end_date: 契約終了日（YYYY-MM-DD形式。不明な場合はnull）
- monthly_fee_jpy: 月額料金（数値。不明な場合はnull）
- plan_name: プラン名（不明な場合はnull）

制約:
- 回答はJSONオブジェクト1つだけを返してください。
- 余計な文章やコメント、日本語の説明は一切書かないでください。
- nullを使う場合は、小文字のnullを使ってください。

入力テキスト:
{{text}}

出力形式の例:
{
  "company_name": "株式会社サンプル",
  "contract_start_date": "2025-01-01",
  "contract_end_date": "2025-12-31",
  "monthly_fee_jpy": 120000,
  "plan_name": "プレミアム"
}
"""

# プロンプトをMLflow Prompt Registryに登録
# これにより、プロンプトのバージョン管理と再利用が可能になります
mlflow.genai.register_prompt(
    name="document-extraction-system",
    template=template_text,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ4: カスタムモデルクラスの定義
# MAGIC
# MAGIC ### Models from Codeパターン
# MAGIC コードを直接MLflowモデルとして登録する手法です。以下のメリットがあります：
# MAGIC - モデルとコードが一体化し、再現性が向上
# MAGIC - 依存関係が明示的に管理される
# MAGIC - デプロイが容易
# MAGIC
# MAGIC ### このセルの動作
# MAGIC `%%writefile`マジックコマンドで、セルの内容を外部ファイルとして保存します。
# MAGIC
# MAGIC ### 主要コンポーネントの説明
# MAGIC
# MAGIC #### 1. DocumentExtractionModelクラス
# MAGIC - MLflowの`PythonModel`を継承したカスタムモデル
# MAGIC - `load_context()`: モデルロード時の初期化処理
# MAGIC - `predict()`: 推論処理のメインロジック
# MAGIC
# MAGIC #### 2. トレーシング機能
# MAGIC - `@mlflow.trace()`: 各関数の実行を記録し、デバッグや性能分析に活用
# MAGIC - `SpanType.TOOL`, `SpanType.LLM`, `SpanType.CHAIN`で処理の種類を分類
# MAGIC
# MAGIC #### 3. エラー対策
# MAGIC - OpenAIのJSON mode使用時、systemメッセージに"json"を含めることでBAD_REQUESTエラーを回避

# COMMAND ----------

# MAGIC %%writefile ./document_extraction_model.py
# MAGIC import json
# MAGIC from typing import Any, Dict, Optional
# MAGIC
# MAGIC import pandas as pd
# MAGIC import mlflow
# MAGIC from mlflow.pyfunc import PythonModel
# MAGIC from mlflow.models import set_model
# MAGIC from mlflow.entities import SpanType
# MAGIC
# MAGIC from openai import OpenAI
# MAGIC
# MAGIC # デフォルト設定
# MAGIC DEFAULT_PROMPT_URI = "prompts:/document-extraction-system/1"
# MAGIC DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
# MAGIC
# MAGIC # OpenAI APIの呼び出しを自動的にMLflowに記録
# MAGIC mlflow.openai.autolog()
# MAGIC
# MAGIC def _load_prompt(prompt_uri: str):
# MAGIC     """
# MAGIC     Prompt Registryからプロンプトをロードする関数
# MAGIC     URI形式: prompts://<プロンプト名>/<バージョン>
# MAGIC     """
# MAGIC     return mlflow.genai.load_prompt(prompt_uri)
# MAGIC
# MAGIC @mlflow.trace(span_type=SpanType.TOOL)
# MAGIC def _render_prompt(prompt, text: str) -> str:
# MAGIC     """
# MAGIC     プロンプトテンプレートに実際のテキストを埋め込む関数
# MAGIC     {{text}}プレースホルダーを入力テキストで置換します
# MAGIC     """
# MAGIC     try:
# MAGIC         return prompt.format(text=text)
# MAGIC     except Exception:
# MAGIC         # フォールバック: format()が使えない場合は文字列置換
# MAGIC         tmpl = getattr(prompt, "template", None)
# MAGIC         if isinstance(tmpl, str):
# MAGIC             return tmpl.replace("{{text}}", text)
# MAGIC         return str(prompt).replace("{{text}}", text)
# MAGIC
# MAGIC @mlflow.trace(span_type=SpanType.LLM)
# MAGIC def _call_llm_return_json(*, client, prompt_text: str, model: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
# MAGIC     """
# MAGIC     OpenAI APIを呼び出し、JSON形式で結果を返す関数
# MAGIC     
# MAGIC     Args:
# MAGIC         client: OpenAIクライアント
# MAGIC         prompt_text: 生成済みのプロンプト全文
# MAGIC         model: 使用するLLMモデル名
# MAGIC         max_tokens: 最大トークン数
# MAGIC         temperature: 生成のランダム性（0=決定的、1=創造的）
# MAGIC     
# MAGIC     Returns:
# MAGIC         抽出された情報を含む辞書
# MAGIC     """
# MAGIC     res = client.chat.completions.create(
# MAGIC         model=model,
# MAGIC         messages=[
# MAGIC             {"role": "system", "content": "Return ONLY valid JSON (json_object)."},
# MAGIC             {"role": "user", "content": prompt_text},
# MAGIC         ],
# MAGIC         temperature=temperature,
# MAGIC         max_completion_tokens=max_tokens,
# MAGIC         response_format={"type": "json_object"},  # JSON形式を強制
# MAGIC     )
# MAGIC     content = res.choices[0].message.content
# MAGIC     return json.loads(content)
# MAGIC
# MAGIC
# MAGIC class DocumentExtractionModel(PythonModel):
# MAGIC     """
# MAGIC     ビジネス文書から情報を抽出するカスタムMLflowモデル
# MAGIC     
# MAGIC     入力形式: pandas.DataFrame
# MAGIC         - 必須カラム: 'text' (抽出対象のテキスト)
# MAGIC         - オプションカラム: 'model' (使用するLLMモデル名)
# MAGIC     
# MAGIC     出力形式: pandas.DataFrame
# MAGIC         - 抽出されたJSON項目が各カラムとして返される
# MAGIC     """
# MAGIC     
# MAGIC     def load_context(self, context):
# MAGIC         """
# MAGIC         モデルロード時に1回だけ実行される初期化メソッド
# MAGIC         設定の読み込み、クライアントの初期化、プロンプトのロードを行います
# MAGIC         """
# MAGIC         # model_configから設定を取得
# MAGIC         self.cfg = getattr(context, "model_config", {}) or {}
# MAGIC         
# MAGIC         # OpenAIクライアントを初期化（環境変数からAPIキーを取得）
# MAGIC         self.client = OpenAI()
# MAGIC         
# MAGIC         # 設定値を取得（デフォルト値を指定）
# MAGIC         self.prompt_uri = self.cfg.get("prompt_uri", DEFAULT_PROMPT_URI)
# MAGIC         self.default_model = self.cfg.get("default_model", DEFAULT_LLM_MODEL)
# MAGIC         self.max_tokens = int(self.cfg.get("max_tokens", 1024))
# MAGIC         self.temperature = float(self.cfg.get("temperature", 0.0))
# MAGIC         
# MAGIC         # Prompt Registryからプロンプトをロード
# MAGIC         self.prompt = _load_prompt(self.prompt_uri)
# MAGIC     
# MAGIC     @mlflow.trace(span_type=SpanType.CHAIN)
# MAGIC     def predict(self, context, model_input, params=None):
# MAGIC         """
# MAGIC         推論メソッド - 入力テキストから情報を抽出します
# MAGIC         
# MAGIC         処理フロー:
# MAGIC         1. 入力の検証とDataFrame化
# MAGIC         2. 各行に対してループ処理
# MAGIC         3. プロンプトのレンダリング
# MAGIC         4. LLM呼び出し
# MAGIC         5. 結果の収集とDataFrame化
# MAGIC         """
# MAGIC         # 入力がDataFrameでない場合は変換
# MAGIC         if not isinstance(model_input, pd.DataFrame):
# MAGIC             model_input = pd.DataFrame(model_input)
# MAGIC         
# MAGIC         # 'text'カラムの存在を確認
# MAGIC         if "text" not in model_input.columns:
# MAGIC             raise ValueError("Input must contain column 'text'.")
# MAGIC         
# MAGIC         rows = []
# MAGIC         # 各行を処理
# MAGIC         for _, row in model_input.iterrows():
# MAGIC             text = str(row.get("text", ""))
# MAGIC             model = str(row.get("model", self.default_model))
# MAGIC             
# MAGIC             # プロンプトに実際のテキストを埋め込み
# MAGIC             prompt_text = _render_prompt(self.prompt, text)
# MAGIC             
# MAGIC             # LLMを呼び出して情報抽出
# MAGIC             extracted = _call_llm_return_json(
# MAGIC                 client=self.client,
# MAGIC                 prompt_text=prompt_text,
# MAGIC                 model=model,
# MAGIC                 max_tokens=self.max_tokens,
# MAGIC                 temperature=self.temperature,
# MAGIC             )
# MAGIC             rows.append(extracted)
# MAGIC         
# MAGIC         return pd.DataFrame(rows)
# MAGIC
# MAGIC
# MAGIC # ★Models from Codeの重要なポイント★
# MAGIC # set_model()を呼び出して、このファイル全体をMLflowモデルとして認識させます
# MAGIC app = DocumentExtractionModel()
# MAGIC set_model(app)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ5: モデルの記録と登録
# MAGIC
# MAGIC ### このステップで行うこと
# MAGIC 1. **入力例の作成**: モデルの入力スキーマを自動推論するためのサンプルデータ
# MAGIC 2. **MLflow実行の開始**: 実験の記録を開始
# MAGIC 3. **モデルのログ記録**: モデルコード、依存関係、設定をMLflowに保存
# MAGIC 4. **モデルレジストリへの登録**: 本番環境へのデプロイ準備
# MAGIC
# MAGIC ### 重要な設定項目
# MAGIC - `python_model`: モデルコードのファイルパス
# MAGIC - `pip_requirements`: 依存パッケージ（デプロイ時に自動インストール）
# MAGIC - `model_config`: モデルに渡す設定（load_context()で参照）
# MAGIC - `registered_model_name`: モデルレジストリでの名前

# COMMAND ----------

import pandas as pd
import mlflow

# 入力例の作成
# これはモデルの入力スキーマ推論とテストに使用されます
input_example = pd.DataFrame({
    "text": ["契約者は株式会社サンプルで、契約期間は2025年1月1日から同じ年の末までです。サービスプランはプレミアムをご契約いただいたので、月額120,000円になります。"],
    "model": ["gpt-3.5-turbo"],
})

# MLflow実行を開始（実験の1回の実行を表す）
with mlflow.start_run(run_name="doc-extraction-model-from-code"):
    # PyFuncモデルとして記録
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",  # モデルの保存先（実行内のパス）
        python_model="./document_extraction_model.py",  # モデルコードのファイル
        registered_model_name="document-extraction-model",  # レジストリでの名前
        input_example=input_example,  # 入力スキーマ推論用
        pip_requirements=[  # 依存パッケージリスト
            "mlflow[genai]>=2.9.0",
            "openai",
            "pandas",
        ],
        model_config={  # モデルに渡す設定（context.model_configで参照可能）
            "prompt_uri": "prompts:/document-extraction-system/1",
            "default_model": "gpt-3.5-turbo",
            "max_tokens": 1024,
            "temperature": 0.0,  # 決定的な出力のため0に設定
        },
    )

    print("Model URI:", model_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ6: モデルのロードとテスト
# MAGIC
# MAGIC 記録したモデルを実際にロードし、推論が正しく動作するかテストします。
# MAGIC
# MAGIC ### model_uriの形式
# MAGIC - `runs:/<run_id>/model`: 特定の実行からロード
# MAGIC - `models:/<model_name>/<version>`: モデルレジストリからロード
# MAGIC - `models:/<model_name>@<alias>`: エイリアス経由でロード

# COMMAND ----------

# モデルをロード
loaded = mlflow.pyfunc.load_model(model_info.model_uri)

# テスト推論を実行
result = loaded.predict(input_example)
print("推論結果:")
print(result)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ7: モデルエイリアスの設定
# MAGIC
# MAGIC ### モデルエイリアスとは？
# MAGIC モデルバージョンに人間が理解しやすい名前を付ける機能です。
# MAGIC
# MAGIC ### 一般的なエイリアス名
# MAGIC - `champion`: 本番環境で使用中のベストモデル
# MAGIC - `challenger`: 評価中の候補モデル
# MAGIC - `staging`: ステージング環境用
# MAGIC
# MAGIC ### メリット
# MAGIC - バージョン番号を直接指定せずにモデルを参照できる
# MAGIC - モデル切り替えがエイリアスの付け替えだけで完結
# MAGIC - A/Bテストやカナリアリリースが容易

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

# バージョン1に"champion"エイリアスを付与
# これにより models:/document-extraction-model@champion でアクセス可能
client.set_registered_model_alias(
    name="document-extraction-model",
    alias="champion",
    version="1"
)

print("エイリアス 'champion' をバージョン1に設定しました")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ステップ8: モデルのサービング（デプロイ）
# MAGIC
# MAGIC ### MLflow Models Serve
# MAGIC モデルをREST APIエンドポイントとして公開する機能です。
# MAGIC
# MAGIC ### コマンドの説明
# MAGIC - `-m models:/document-extraction-model@champion`: エイリアス経由でモデルを指定
# MAGIC - `-p 5000`: ポート番号
# MAGIC - `--host 0.0.0.0`: すべてのネットワークインターフェースでリッスン
# MAGIC
# MAGIC ### 使用方法
# MAGIC サービング開始後、以下のようにAPIを呼び出せます:
# MAGIC ```bash
# MAGIC curl -X POST http://localhost:5000/invocations \
# MAGIC   -H 'Content-Type: application/json' \
# MAGIC   -d '{
# MAGIC     "dataframe_split": {
# MAGIC       "columns": ["text", "model"],
# MAGIC       "data": [["契約書のテキスト...", "gpt-3.5-turbo"]]
# MAGIC     }
# MAGIC   }'
# MAGIC ```
# MAGIC
# MAGIC **注意**: 本番環境では、Databricks Model ServingやKubernetes等を使用することを推奨します。

# COMMAND ----------

# MAGIC %%sh
# MAGIC
# MAGIC mlflow models serve \
# MAGIC   -m models:/document-extraction-model@champion \
# MAGIC   -p 5000 \
# MAGIC   --host 0.0.0.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## まとめ
# MAGIC
# MAGIC ### このノートブックで学んだこと
# MAGIC
# MAGIC 1. **Prompt Registry**: プロンプトのバージョン管理と再利用
# MAGIC 2. **Models from Code**: コードベースでのモデル定義と登録
# MAGIC 3. **カスタムPyFuncモデル**: 柔軟なモデル実装パターン
# MAGIC 4. **MLflowトレーシング**: LLM呼び出しの可観測性向上
# MAGIC 5. **モデルエイリアス**: バージョン管理のベストプラクティス
# MAGIC 6. **モデルサービング**: REST APIとしてのデプロイ
# MAGIC
# MAGIC ### 次のステップ
# MAGIC
# MAGIC - 異なるLLMモデル（GPT-4、Claude等）でのテスト
# MAGIC - プロンプトの改良とバージョン比較
# MAGIC - 本番環境へのデプロイ
# MAGIC - モニタリングとA/Bテストの実装
# MAGIC
# MAGIC ### 参考リソース
# MAGIC
# MAGIC - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
# MAGIC - [MLflow Prompt Engineering](https://mlflow.org/docs/latest/llms/prompt-engineering/index.html)
# MAGIC - [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

# COMMAND ----------

