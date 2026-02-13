"""6.3.1節: プロンプトのオフライン評価

Prompt Registryに登録済みのプロンプトを評価データセットで評価する。
@scorerデコレータでカスタム評価関数を定義し、mlflow.genai.evaluate()で実行する。

実行: make eval
前提: 01_register_prompt.pyを実行済み、OPENAI_API_KEYが設定されていること
"""

import mlflow
import openai
from dotenv import load_dotenv
from mlflow.genai.scorers import scorer

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("プロンプト評価")

# 評価データのインポート
from data.eval_dataset import EVAL_DATA


# プロンプト評価用の予測関数
def predict_fn(sentences: str) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
    completion = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt.format(sentences=sentences, num_sentences=1),
            }
        ],
    )
    return completion.choices[0].message.content


# カスタムスコアラー: LLMで要約の類似度を判定
@scorer
def answer_similarity(inputs, outputs, expectations):
    expected_summary = expectations["summary"]
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "提供された出力が期待される回答と意味的に類似しているか評価してください。\n\n"
                    f"出力: {outputs}\n\n"
                    f"期待される回答: {expected_summary}\n\n"
                    "類似している場合は 'yes'、そうでない場合は 'no' のみを返してください。"
                ),
            }
        ],
    )
    judgment = response.choices[0].message.content.strip().lower()
    return judgment == "yes"


# 評価実行
print("プロンプト バージョン1 の評価を実行中...")
results = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=predict_fn,
    scorers=[answer_similarity],
)

print(f"\n評価結果:")
print(results.metrics)
print("\nMLflow UI (http://localhost:5000) で詳細を確認してください。")
