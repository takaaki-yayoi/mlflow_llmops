"""6.2.3節: エイリアス管理とライフサイクル

エイリアスを使ってプロンプトバージョンを環境ごとに管理する。
ロールバックのデモも含む。

実行: make alias
前提: 02_version_update.pyを実行済みであること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# バージョン2をproductionエイリアスとして設定
mlflow.genai.set_prompt_alias(
    "summarization-prompt",
    alias="production",
    version=2,
)
print("productionエイリアスをバージョン2に設定しました")

# エイリアスを使用してプロンプトをロード
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
print(f"@production → バージョン {prompt.version}")

# @latestで最新バージョンをロード
latest = mlflow.genai.load_prompt("prompts:/summarization-prompt@latest")
print(f"@latest → バージョン {latest.version}")

# ロールバック: productionをバージョン1に戻す
mlflow.genai.set_prompt_alias(
    "summarization-prompt",
    alias="production",
    version=1,
)
rollback = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
print(f"\nロールバック後: @production → バージョン {rollback.version}")

# 元に戻す(バージョン2をproductionに)
mlflow.genai.set_prompt_alias(
    "summarization-prompt",
    alias="production",
    version=2,
)
print("productionエイリアスをバージョン2に復元しました")
