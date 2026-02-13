"""評価用データセット

6.3節と6.5節で使用するQAエージェントの評価データ。
MLflowに関する質問と期待される回答のペア。
"""

EVAL_DATA = [
    {
        "inputs": {
            "question": "MLflow Tracingとは何ですか?",
        },
        "expectations": {
            "expected_answer": "MLflow TracingはLLMアプリケーションの実行フローを可視化する機能です。各ステップの入出力、レイテンシ、トークン使用量を記録し、デバッグや性能分析に活用できます。",
        },
    },
    {
        "inputs": {
            "question": "MLflowでプロンプトをバージョン管理する方法を教えてください。",
        },
        "expectations": {
            "expected_answer": "MLflow Prompt Registryを使ってプロンプトをバージョン管理できます。mlflow.genai.register_prompt()で登録し、エイリアス(@production, @latestなど)で環境ごとに使い分けられます。",
        },
    },
    {
        "inputs": {
            "question": "MLflow Evaluateの主な機能は何ですか?",
        },
        "expectations": {
            "expected_answer": "mlflow.genai.evaluate()はLLMアプリケーションの品質を定量評価する機能です。組み込みスコアラー(Correctness, Safety等)やカスタムスコアラーを使って、正確性・安全性・関連性などを自動評価できます。",
        },
    },
    {
        "inputs": {
            "question": "MLflowのautolog機能について説明してください。",
        },
        "expectations": {
            "expected_answer": "autologはLangChain、OpenAIなどのフレームワークと統合し、コード変更なしで自動的にトレースを記録する機能です。mlflow.langchain.autolog()のように有効化します。",
        },
    },
    {
        "inputs": {
            "question": "MLflow Model Registryとは何ですか?",
        },
        "expectations": {
            "expected_answer": "Model Registryはモデルのバージョン管理とライフサイクル管理を行う機能です。モデルの登録、エイリアス(champion/challengerなど)による管理、ステージ遷移ができます。",
        },
    },
]
