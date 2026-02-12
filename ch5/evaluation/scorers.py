"""第5章で使用する全評価指標(スコアラー)の定義。

このモジュールには、以下のスコアラーが含まれます:
- 標準スコアラー: Correctness, ToolCallCorrectness
- ルールベースカスタム: contains_code_block
- Guidelinesベース: has_reference_link, appropriate_katakana
- make_judgeベース: katakana_judge (応用)
"""

import re
from typing import Literal

from mlflow.genai.scorers import scorer, Correctness, Guidelines, ToolCallCorrectness
from mlflow.genai.judges import make_judge

# --- 標準スコアラー ---
correctness = Correctness()
tool_call_correctness = ToolCallCorrectness()


# --- ルールベースのカスタムスコアラー (5.4.6) ---
@scorer
def contains_code_block(outputs: str) -> bool:
    """回答にコードブロックが含まれているかを検出する。

    predict_fnが文字列を返す場合、outputsは文字列として渡される。
    """
    text = outputs if isinstance(outputs, str) else str(outputs)
    return bool(re.search(r"```[\s\S]+?```", text))


# --- Guidelinesベースのスコアラー (5.4.6) ---
has_reference_link = Guidelines(
    name="has_reference_link",
    guidelines=[
        "回答の中で、公式ドキュメントのURLやAPIリファレンスへのリンクが提示されている。",
        "リンクの内容はユーザの質問に関連した適切なものである。",
    ],
)

appropriate_katakana = Guidelines(
    name="appropriate_katakana",
    guidelines=(
        "技術用語のカタカナ表記が適切に使用されている。"
        "一般的に日本語で定着している技術用語(例: プロンプト、エージェント)は許容するが、"
        "実験や指標のような日本語が自然な場合はカタカナに変換しない。"
        "また英語のまま使用するのが自然な用語(例: MLflow、Python、API)は"
        "カタカナに変換しないこと。"
    ),
)


# --- make_judgeベースのスコアラー (5.4.6 補足) ---
katakana_judge = make_judge(
    name="katakana_usage",
    instructions="""あなたは技術文書の用語レビュアーです。
以下の回答におけるカタカナ用語の使用が適切かを判定してください。

## 判定基準
- 英語のまま使うのが自然な固有名詞(MLflow, Python, LangChain等)が
不自然にカタカナ化されていないか
- 日本語として定着した用語(プロンプト、エージェント等)は許容する

## 良い例
「MLflowのトレーシング機能を使ってデプロイします」
→ MLflow(固有名詞)は英語のまま、デプロイ(定着語)はカタカナで適切

## 悪い例
「エムエルフローで実験管理を行います」
→ MLflowをカタカナ化しており不適切

「MLflowにメトリックをログします」
→ 「ログします」は日本語で「記録します」と記載した方が自然

## 回答
{{ outputs }}
""",
    feedback_value_type=Literal["yes", "no", "maybe"],
    model="openai:/gpt-4o-mini",
)
