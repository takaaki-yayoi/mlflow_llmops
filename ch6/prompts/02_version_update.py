"""6.2.2節: バージョン更新と不変性

既存のsummarization-promptを改良してバージョン2を登録する。
プロンプトバージョンは不変(immutable)であることを確認する。

実行: make version
前提: 01_register_prompt.pyを実行済みであること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

new_template = """
あなたは要約の専門家です。以下のコンテンツを、要点を捉えた明確で有益な{{ num_sentences }}文に凝縮してください。 文章: {{ sentences }} 要約は以下の条件を満たす必要があります:
正確に{{ num_sentences }}文である
最も重要な情報のみを含む
中立的で客観的なトーンで書かれている
元のテキストと同じレベルのフォーマリティを維持する
"""

# 既存のプロンプト名を指定して新バージョンを登録
updated_prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=new_template,
    commit_message="ペルソナと条件を追加",
    tags={"author": "alice@example.com"},
)
print(f"新バージョン {updated_prompt.version} を登録しました")

# 不変性の確認: バージョン1はそのまま残っている
v1 = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
v2 = mlflow.genai.load_prompt("prompts:/summarization-prompt/2")
print(f"\nバージョン1のテンプレート(先頭50文字): {v1.template[:50]}...")
print(f"バージョン2のテンプレート(先頭50文字): {v2.template[:50]}...")
print("\n各バージョンは不変であり、独立して保持されています。")
