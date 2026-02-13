"""6.2.6節: モデルパラメータの保存

プロンプトと共にモデル名やパラメータを保存し、再現性を高める。

実行: make model-config
前提: MLflow Tracking Serverが起動していること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

model_config = {
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 0.9,
}

prompt = mlflow.genai.register_prompt(
    name="qa-prompt",
    template="以下の質問に答えて下さい: {{question}}",
    model_config=model_config,
    commit_message="モデルパラメータ付きで登録",
)

# プロンプトとモデルパラメーターをロード
loaded = mlflow.genai.load_prompt("qa-prompt")
print(f"プロンプト: {loaded.name} (version {loaded.version})")
print(f"モデル: {loaded.model_config['model_name']}")
print(f"Temperature: {loaded.model_config['temperature']}")
print(f"Max tokens: {loaded.model_config['max_tokens']}")
