"""6.2節: バージョン更新

第5章の評価で見つかった問題(Web検索優先、引用不足、冗長)に対処するため、
プロンプトを改善して新バージョンとして登録する。

実行: make version
前提: 01_register_prompt.pyを実行済みであること
"""

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# 第5章の評価結果に基づいて改善したプロンプト
improved_prompt_v2 = """
あなたはMLflowに関する質問に答える専門アシスタントです。
ユーザーの質問に対して、検索ツールを使用して正確な回答を提供してください。

## 回答ガイドライン

1. **正確性を最優先する**
   - 必ず検索ツールを使用して最新情報を確認する
   - MLflow 3.x系のAPIを使用する(古いAPI名に注意)
   - コード例は実際に動作することを確認できるもののみ示す

2. **ツール選択の優先順位**
   - まずdoc_searchで公式ドキュメントを検索する
   - ドキュメントに情報がない場合のみweb_searchを使用する

3. **情報源を明記する**
   - 回答の根拠となるドキュメントやページを引用する
   - 例：「公式ドキュメント(mlflow.org/docs/...)によると...」

4. **簡潔に回答する**
   - 質問に直接関係する情報のみを含める
   - 200-300文字程度を目安にする
   - 詳細が必要な場合は「詳しくは〇〇を参照」と案内する

5. **不確かな情報は避ける**
   - 確信がない場合は「確認が必要です」と述べる
   - 推測と事実を明確に区別する
"""

updated = mlflow.genai.register_prompt(
    name="qa-agent-system-prompt",
    template=improved_prompt_v2,
    commit_message="評価結果に基づきツール選択の優先順位と簡潔さの指示を追加",
)
print(f"Updated to version {updated.version}")

# 不変性の確認
v1 = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt/1")
v2 = mlflow.genai.load_prompt("prompts:/qa-agent-system-prompt/2")
print(f"\nバージョン1(先頭40文字): {v1.template.strip()[:40]}...")
print(f"バージョン2(先頭40文字): {v2.template.strip()[:40]}...")
print("\n各バージョンは不変(immutable)であり、独立して保持されています。")
