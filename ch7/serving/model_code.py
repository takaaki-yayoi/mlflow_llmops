"""models-from-code用のモデル定義ファイル（7.2節）。

MLflow v3のLangChain統合では、モデルオブジェクトを直接渡すのではなく、
このようなコードファイルで定義してmlflow.models.set_model()で登録します。

参考: https://mlflow.org/docs/latest/ml/model/models-from-code/
"""

import mlflow

from agents.langgraph.agent import LangGraphAgent

agent = LangGraphAgent()
mlflow.models.set_model(agent.executor)
