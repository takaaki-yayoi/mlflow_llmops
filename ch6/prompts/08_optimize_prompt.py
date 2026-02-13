"""6.3.4節: プロンプト自動最適化

GepaPromptOptimizerを使ってプロンプトを自動最適化する。
最適化されたプロンプトは新バージョンとしてレジストリに自動登録される。

注意: このスクリプトはLLMを多数回呼び出すため、実行コストが高く時間がかかる。
      max_metric_callsで呼び出し回数を制限すること。
      max_metric_calls=10、学習データ3件の場合、完了まで数分程度。

実行: make optimize
前提: 01_register_prompt.pyを実行済み、OPENAI_API_KEYが設定されていること
"""

import time

import mlflow
import openai
from dotenv import load_dotenv
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import scorer

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("プロンプト最適化")

# 評価データのインポート
from data.eval_dataset import EVAL_DATA


# 予測関数
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


# カスタムスコアラー: LLMで要約の品質を0〜1で数値評価
@scorer
def answer_similarity(inputs, outputs, expectations):
    expected_summary = expectations["summary"]
    response = openai.OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    "以下の「出力」が「期待される要約」とどの程度一致しているか、"
                    "0.0〜1.0のスコアで評価してください。\n\n"
                    "評価基準:\n"
                    "- 主要な事実が正確にカバーされているか\n"
                    "- 不要な情報が含まれていないか\n"
                    "- 表現が簡潔で明確か\n\n"
                    f"出力: {outputs}\n\n"
                    f"期待される要約: {expected_summary}\n\n"
                    "スコアのみを数値で返してください（例: 0.7）。"
                ),
            }
        ],
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0


# プロンプトURIを取得
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")

# 最適化を実行
print("プロンプト最適化を開始します...")
print(f"対象プロンプト: {prompt.name} (version {prompt.version})")
print(f"学習データ: {len(EVAL_DATA)}件")
print("(LLMを多数回呼び出すため、完了まで数分かかる場合があります)\n")

start_time = time.time()

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=EVAL_DATA,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-4o",
        max_metric_calls=10,  # コスト・時間制限のため小さい値に設定
    ),
    scorers=[answer_similarity],
)

elapsed = int(time.time() - start_time)
m, s = divmod(elapsed, 60)

# 結果の表示
print(f"\n完了しました。(所要時間: {m}分{s}秒)")
if result.initial_eval_score is not None:
    print(f"初期スコア: {result.initial_eval_score:.3f}")
if result.final_eval_score is not None:
    print(f"最終スコア: {result.final_eval_score:.3f}")

optimized_prompt = result.optimized_prompts[0]
print(f"\n最適化されたプロンプト:")
print(f"テンプレート: {optimized_prompt.template[:200]}...")
print("\n最適化されたプロンプトは新しいバージョンとしてレジストリに自動登録されました。")
print("MLflow UI (http://localhost:5000) のPromptsタブで確認してください。")
