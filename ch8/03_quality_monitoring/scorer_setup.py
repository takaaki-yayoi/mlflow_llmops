"""
第8章 8.3.4-8.3.5: スコアラーの登録と開始

MLflow 3のスコアラーを使用した本番環境での継続的品質監視の設定例です。
開発時の評価と本番環境の監視で同じスコアラーを使用できます。
"""

import mlflow
from mlflow.genai.scorers import (
    Safety,
    Correctness,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalSufficiency,
    Guidelines,
    ScorerSamplingConfig,
    scorer,
)


def setup_builtin_scorers(
    experiment_name: str,
    safety_sample_rate: float = 1.0,
    quality_sample_rate: float = 0.5,
) -> dict:
    """
    組み込みスコアラーを登録・開始
    
    注意: この関数はDatabricks環境でのみ動作します。
    Safety, RetrievalGroundednessはDatabricks専用スコアラーです。
    
    Args:
        experiment_name: MLflowエクスペリメント名
        safety_sample_rate: 安全性スコアラーのサンプリング率
        quality_sample_rate: 品質スコアラーのサンプリング率
    
    Returns:
        登録されたスコアラーの辞書
    """
    mlflow.set_experiment(experiment_name)
    
    scorers = {}
    
    # 安全性スコアラー (高優先度: 100%サンプリング推奨)
    safety_scorer = Safety().register(name="safety_check")
    safety_scorer = safety_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=safety_sample_rate)
    )
    scorers["safety"] = safety_scorer
    print(f"Registered: safety_check (sample_rate={safety_sample_rate})")
    
    # クエリ関連性スコアラー
    relevance_scorer = RelevanceToQuery().register(name="relevance_check")
    relevance_scorer = relevance_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=quality_sample_rate)
    )
    scorers["relevance"] = relevance_scorer
    print(f"Registered: relevance_check (sample_rate={quality_sample_rate})")
    
    # RAG根拠性スコアラー (RAGアプリケーション向け)
    groundedness_scorer = RetrievalGroundedness().register(name="groundedness_check")
    groundedness_scorer = groundedness_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=quality_sample_rate)
    )
    scorers["groundedness"] = groundedness_scorer
    print(f"Registered: groundedness_check (sample_rate={quality_sample_rate})")
    
    return scorers


def setup_custom_guidelines_scorer(
    experiment_name: str,
    guidelines: str,
    scorer_name: str = "custom_guidelines",
    sample_rate: float = 0.5,
) -> object:
    """
    カスタムガイドラインスコアラーを登録・開始
    
    注意: Databricks環境でのみ動作します。
    
    Args:
        experiment_name: MLflowエクスペリメント名
        guidelines: ガイドライン文字列 (自然言語でpass/fail条件を記述)
        scorer_name: スコアラー名
        sample_rate: サンプリング率
    
    Returns:
        登録されたスコアラー
    """
    mlflow.set_experiment(experiment_name)
    
    # Guidelinesスコアラーを作成
    # 注意: guidelinesパラメータは文字列(リストではない)
    guidelines_scorer = Guidelines(
        name=scorer_name,
        guidelines=guidelines,
    ).register(name=scorer_name)
    
    guidelines_scorer = guidelines_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=sample_rate)
    )
    
    print(f"Registered: {scorer_name} (sample_rate={sample_rate})")
    print(f"  Guidelines: {guidelines[:50]}..." if len(guidelines) > 50 else f"  Guidelines: {guidelines}")
    
    return guidelines_scorer


def setup_custom_code_scorer(
    experiment_name: str,
    sample_rate: float = 0.5,
) -> object:
    """
    カスタムコードベーススコアラーを登録・開始
    
    注意: 
    - Databricks環境でのみ動作します
    - カスタムスコアラーでは関数シグネチャに型ヒントを使用しないでください
    - インポートは関数内で行ってください(シリアライズ対応)
    
    Args:
        experiment_name: MLflowエクスペリメント名
        sample_rate: サンプリング率
    
    Returns:
        登録されたスコアラー
    """
    mlflow.set_experiment(experiment_name)
    
    # カスタムスコアラーを定義
    # 注意: 関数内でインポートを行うこと (シリアライズ対応)
    @scorer(aggregations=["mean", "min", "max"])
    def response_length(outputs):
        """レスポンスの文字数を測定"""
        response = outputs.get("response", "")
        return len(str(response))
    
    @scorer
    def contains_apology(outputs):
        """謝罪表現が含まれているかチェック"""
        response = str(outputs.get("response", "")).lower()
        apology_words = ["sorry", "apologize", "申し訳", "すみません", "ごめん"]
        return any(word in response for word in apology_words)
    
    # スコアラーを登録・開始
    length_scorer = response_length.register(name="response_length")
    length_scorer = length_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=sample_rate)
    )
    print(f"Registered: response_length (sample_rate={sample_rate})")
    
    apology_scorer = contains_apology.register(name="contains_apology")
    apology_scorer = apology_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=sample_rate)
    )
    print(f"Registered: contains_apology (sample_rate={sample_rate})")
    
    return {
        "response_length": length_scorer,
        "contains_apology": apology_scorer,
    }


def setup_quality_scorers(
    experiment_name: str,
    safety_sample_rate: float = 1.0,
    quality_sample_rate: float = 0.5,
    custom_guidelines: str = None,
) -> dict:
    """
    本番環境向けの品質スコアラーをまとめてセットアップ
    
    注意: この関数はDatabricks環境でのみ動作します。
    Safety, RetrievalRelevanceはDatabricks専用スコアラーです。
    
    Args:
        experiment_name: MLflowエクスペリメント名
        safety_sample_rate: 安全性スコアラーのサンプリング率
        quality_sample_rate: 品質スコアラーのサンプリング率
        custom_guidelines: カスタムガイドライン文字列 (オプション)
    
    Returns:
        登録されたスコアラーの辞書
    """
    all_scorers = {}
    
    # 組み込みスコアラー
    builtin = setup_builtin_scorers(
        experiment_name=experiment_name,
        safety_sample_rate=safety_sample_rate,
        quality_sample_rate=quality_sample_rate,
    )
    all_scorers.update(builtin)
    
    # カスタムガイドライン
    if custom_guidelines:
        guidelines = setup_custom_guidelines_scorer(
            experiment_name=experiment_name,
            guidelines=custom_guidelines,
            scorer_name="custom_guidelines",
            sample_rate=quality_sample_rate,
        )
        all_scorers["custom_guidelines"] = guidelines
    
    # カスタムコードスコアラー
    custom = setup_custom_code_scorer(
        experiment_name=experiment_name,
        sample_rate=quality_sample_rate,
    )
    all_scorers.update(custom)
    
    print(f"\nTotal scorers registered: {len(all_scorers)}")
    return all_scorers


# 開発時の評価での使用例
def evaluate_with_scorers(eval_data: list[dict]) -> object:
    """
    開発時にスコアラーを使用して評価
    
    本番環境で使用するスコアラーと同じものを開発時の評価にも使用できます。
    """
    from mlflow.genai.scorers import Safety, Guidelines
    
    results = mlflow.genai.evaluate(
        data=eval_data,
        scorers=[
            Safety(),
            Guidelines(
                name="professional_tone",
                guidelines="回答は専門的で丁寧なトーンである必要があります。"
            ),
        ]
    )
    
    return results


# 使用例
if __name__ == "__main__":
    mlflow.set_tracking_uri("databricks")
    
    # 本番環境向けスコアラーのセットアップ
    # 注意: Databricks環境でのみ動作します
    scorers = setup_quality_scorers(
        experiment_name="/production/customer-support-bot",
        safety_sample_rate=1.0,  # 安全性は100%チェック
        quality_sample_rate=0.5,  # その他は50%サンプリング
        custom_guidelines="回答は専門的で丁寧なトーンである必要があります。機密情報を含めてはいけません。",
    )
    
    print("\nSetup complete!")
    print(f"Registered scorers: {list(scorers.keys())}")
