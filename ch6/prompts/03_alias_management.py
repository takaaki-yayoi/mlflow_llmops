"""6.2節: エイリアスによるライフサイクル管理

エイリアスを使ってプロンプトバージョンを環境ごとに管理する。
ロールバックのデモも含む。

実行: make alias
前提: 02_version_update.pyを実行済みであること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# 開発環境: 最新の実験的バージョン
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="development",
    version=2,
)
print("developmentエイリアスをバージョン2に設定しました")

# 本番環境: 安定したバージョン
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="production",
    version=1,
)
print("productionエイリアスをバージョン1に設定しました")

# エイリアスを使ってプロンプトをロード
dev_prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@development")
prod_prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(f"\n@development → バージョン {dev_prompt.version}")
print(f"@production  → バージョン {prod_prompt.version}")

# @latestで最新バージョンをロード
latest = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@latest")
print(f"@latest      → バージョン {latest.version}")

# ロールバックのデモ: productionをバージョン2に昇格
mlflow.genai.set_prompt_alias(
    "qa-agent-system-prompt",
    alias="production",
    version=2,
)
prod_after = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(f"\n昇格後: @production → バージョン {prod_after.version}")
print("コード変更なし・再デプロイ不要でプロンプトを切り替えられます。")
