"""6.7節: モデルパラメータの紐付け

プロンプトと共にモデル名やパラメータを保存し、再現性を高める。

実行: make model-config
前提: MLflow Tracking Serverが起動していること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

prompt = mlflow.genai.register_prompt(
    name="qa-agent-system-prompt",
    template="(モデルパラメータ付きバージョン)",
    model_config={
        "model_name": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 500,
    },
    commit_message="モデルパラメータを追加",
)

# プロンプトとモデルパラメータをロード
loaded = mlflow.genai.load_prompt(f"prompts:/qa-agent-system-prompt/{prompt.version}")
print(f"プロンプト: {loaded.name} (version {loaded.version})")
print(f"モデル: {loaded.model_config['model_name']}")
print(f"Temperature: {loaded.model_config['temperature']}")
print(f"Max tokens: {loaded.model_config['max_tokens']}")
print("\nアプリケーション側でload_prompt()後にmodel_configを参照して使用します。")
