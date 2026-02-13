"""6.2.1節: プロンプトの登録

summarization-promptをPrompt Registryに登録し、変数の埋め込みを確認する。

実行: make register
前提: MLflow Tracking Serverが起動していること (uv run mlflow server --port 5000)
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# テンプレート文字列は二重中括弧で変数を定義する
initial_template = """
提供されたコンテンツを{{ num_sentences }}文で要約してください。 文章: {{ sentences }}
"""

# プロンプトの登録
prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=initial_template,
    commit_message="初期プロンプト",
    tags={
        "author": "alice@example.com",
        "task": "要約",
        "language": "ja",
    },
)
print(f"プロンプト '{prompt.name}' (version {prompt.version}) を登録しました")

# 変数の埋め込みテスト
formatted = prompt.format(
    sentences="MLflowは機械学習のライフサイクルを管理するオープンソースプラットフォームです。",
    num_sentences=1,
)
print(f"\nフォーマット結果:\n{formatted}")
