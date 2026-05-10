"""6.2節: プロンプトの登録

QAエージェントのシステムプロンプトをPrompt Registryに登録する。
第4章でコード内にハードコードされていたプロンプトを、レジストリに移行する最初のステップ。

実行: make register
前提: MLflow Tracking Serverが起動していること (uv run mlflow server --port 5000)
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# 本書 リスト6.1
# 第4章のQAエージェントで使用していたシステムプロンプト
initial_template = """
あなたはMLflowに関する質問に答える専門アシスタントです。
ユーザーの質問に対して、必要に応じてドキュメント検索やWeb検索を使用して、
正確で詳細な回答を提供してください。

回答の際は以下の点に注意してください：
- 公式ドキュメントに基づいた正確な情報を提供する
- 必要に応じてコード例を含める
- 情報源を明記する
"""

# プロンプトの登録
prompt = mlflow.genai.register_prompt(
    name="qa-agent-system-prompt",  # プロンプト名
    template=initial_template,  # テンプレート
    commit_message="QAエージェントの初期プロンプト",  # コミットメッセージ
    tags={
        "author": "alice@example.com",
        "task": "qa",
        "language": "ja",
    },
)

print(f"プロンプト '{prompt.name}' (version {prompt.version})")
