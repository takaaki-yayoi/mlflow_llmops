"""@mlflow.test (MLflow 3.14以降) を使うための pytest 設定。

@mlflow.test は MLflow の pytest プラグインを有効にしないと動きません。
プラグインは自動では読み込まれないため、ここで明示的に有効にしています。
pytest -p mlflow.pytest.plugin と指定するのと同じ意味です。
"""
from dotenv import load_dotenv

load_dotenv()

pytest_plugins = ["mlflow.pytest.plugin"]
