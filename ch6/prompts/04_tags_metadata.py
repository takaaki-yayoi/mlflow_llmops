"""6.2.4節: タグとメタデータ

プロンプトとバージョンにタグを付与し、検索や分類を容易にする。

実行: make tags
前提: 02_version_update.pyを実行済みであること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# --- プロンプトレベルのタグ ---
mlflow.genai.set_prompt_tag("summarization-prompt", "language", "ja")
print("プロンプトレベルのタグ 'language=ja' を設定しました")

# --- バージョンレベルのタグ ---
mlflow.genai.set_prompt_version_tag("summarization-prompt", 1, "author", "alice")
print("バージョン1にタグ 'author=alice' を設定しました")

# タグの取得
prompt_v1 = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
print(f"\nバージョン1のタグ: {prompt_v1.tags}")
