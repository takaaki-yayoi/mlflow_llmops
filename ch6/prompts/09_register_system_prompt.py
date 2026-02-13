"""6.4節: 他章との連携 - QAエージェントのシステムプロンプト登録

第3-4章で構築したQAエージェントのシステムプロンプトをPrompt Registryに登録する。
第7章のAgent Serverはこのプロンプトを@productionエイリアスで取得して使用する。

実行: make system-prompt
前提: MLflow Tracking Serverが起動していること (uv run mlflow server --port 5000)
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# 第3-4章のQAエージェントで使用しているシステムプロンプト
system_prompt_template = """あなたはMLflowに関する質問に答える専門アシスタントです。
MLflowは機械学習のライフサイクルを管理するためのオープンソースプラットフォームです。

あなたの責務:
- MLflowの機能、API、ベストプラクティスに関する質問に回答する
- MLflowの概念を説明し、問題のトラブルシューティングを支援する
- 適切なリソースやドキュメントへユーザーを案内する

利用可能なツールを使用して、正確で最新の情報を取得してください。
ツールから取得した情報を提供する際は、必ずURLを含む引用を記載してください。
"""

# Prompt Registryに登録
prompt = mlflow.genai.register_prompt(
    name="qa-system-prompt",
    template=system_prompt_template,
    commit_message="QAエージェントのシステムプロンプトを登録",
    tags={
        "task": "qa",
        "agent": "langgraph",
        "chapter": "3,4,7",
    },
)
print(f"プロンプト '{prompt.name}' (version {prompt.version}) を登録しました")

# productionエイリアスを設定(第7章のAgent Serverが参照する)
mlflow.genai.set_prompt_alias(
    "qa-system-prompt",
    alias="production",
    version=prompt.version,
)
print(f"productionエイリアスをバージョン {prompt.version} に設定しました")

# 確認
loaded = mlflow.genai.load_prompt("prompts:/qa-system-prompt@production")
print(f"\n検証: @production → バージョン {loaded.version}")
print(f"テンプレート先頭: {loaded.template[:60]}...")
print("\nこのプロンプトは第7章のAgent Serverから以下のコードで取得されます:")
print('  mlflow.genai.load_prompt("prompts:/qa-system-prompt@production")')
