"""6.2節: エイリアスによるライフサイクル管理

エイリアスを使ってプロンプトバージョンを環境ごとに管理する。
ロールバックのデモも含む。

実行: make alias
前提: 02_version_update.pyを実行済みであること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# 本書 リスト6.3
# バージョン2をproductionエイリアスとして設定
mlflow.genai.set_prompt_alias(
    name="qa-agent-system-prompt",
    alias="production",
    version=2,
)

# `@`記号とともにエイリアスを使用してプロンプトをロード
prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(prompt.version)  # => 2

# --- 以下、本書の補足説明を実演 ---
# 開発環境用エイリアスも設定し、エイリアスの使い分けを確認する
mlflow.genai.set_prompt_alias(
    name="qa-agent-system-prompt",
    alias="development",
    version=2,
)

dev_prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@development")
prod_prompt = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(f"\n@development → バージョン {dev_prompt.version}")
print(f"@production  → バージョン {prod_prompt.version}")

# 予約済みエイリアス @latest で常に最新バージョンをロード
latest = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@latest")
print(f"@latest      → バージョン {latest.version}")

# ロールバックのデモ (本書 リスト6.9): productionを安定していたバージョン1に戻す
mlflow.genai.set_prompt_alias(
    name="qa-agent-system-prompt",
    alias="production",
    version=1,  # 安定していた前のバージョンに戻す
)
prod_after = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt@production")
print(f"\nロールバック後: @production → バージョン {prod_after.version}")
print("コード変更なし・再デプロイ不要でプロンプトを切り替えられます。")
