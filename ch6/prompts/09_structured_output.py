"""6.7節: 構造化出力(Structured Output)の利用

response_formatパラメータで期待される出力形式を定義し、
OpenAI APIで構造化出力を取得する。

実行: make structured
前提: OPENAI_API_KEYが設定されていること
"""

from typing import List

import mlflow
import openai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")


class QAResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]


prompt = mlflow.genai.register_prompt(
    name="qa-agent-structured",
    template="質問に回答してください: {{ question }}",
    response_format=QAResponse,
    commit_message="構造化出力を追加",
)
print(f"プロンプト '{prompt.name}' (version {prompt.version}) を登録しました")

loaded = mlflow.genai.load_prompt("prompts:/qa-agent-structured@latest")

response = openai.OpenAI().beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": loaded.format(question="MLflowとは何ですか?"),
        }
    ],
    response_format=QAResponse,
)

result = response.choices[0].message.parsed
print(f"\n構造化出力の結果:")
print(f"  回答: {result.answer}")
print(f"  確信度: {result.confidence}")
print(f"  ソース: {result.sources}")
