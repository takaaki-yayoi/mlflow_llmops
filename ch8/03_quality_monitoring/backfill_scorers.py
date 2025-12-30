"""
第8章 8.3.5: 過去トレースへのバックフィル

新しいスコアラーを追加した場合や、サンプリングで評価されなかった
トレースに対して、過去のトレースを評価できます。

注意:
- この機能はDatabricks環境でのみ利用可能です
- databricks-agentsパッケージが必要です: pip install databricks-agents
"""

from datetime import datetime, timedelta
from typing import Optional, Union
import mlflow
from mlflow.genai.scorers import Safety, Guidelines, ScorerSamplingConfig


def backfill_with_scorer_names(
    experiment_id: str,
    scorer_names: list[str],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> str:
    """
    スコアラー名を指定してバックフィル(現在の設定のサンプリング率を使用)
    
    Args:
        experiment_id: エクスペリメントID
        scorer_names: バックフィルするスコアラー名のリスト
        start_time: 開始時刻 (指定しない場合は全期間)
        end_time: 終了時刻 (指定しない場合は現在まで)
    
    Returns:
        バックフィルジョブID
    """
    from databricks.agents.scorers import backfill_scorers
    
    kwargs = {
        "experiment_id": experiment_id,
        "scorers": scorer_names,  # 文字列のリスト
    }
    
    if start_time:
        kwargs["start_time"] = start_time
    if end_time:
        kwargs["end_time"] = end_time
    
    job_id = backfill_scorers(**kwargs)
    
    print(f"Backfill job started: {job_id}")
    print(f"  Scorers: {scorer_names}")
    print(f"  Time range: {start_time or 'beginning'} to {end_time or 'now'}")
    
    return job_id


def backfill_with_custom_sample_rates(
    experiment_id: str,
    scorer_configs: list[dict],
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> str:
    """
    カスタムサンプリング率を指定してバックフィル
    
    Args:
        experiment_id: エクスペリメントID
        scorer_configs: スコアラー設定のリスト
            [{"scorer": scorer_object, "sample_rate": 0.8}, ...]
        start_time: 開始時刻
        end_time: 終了時刻
    
    Returns:
        バックフィルジョブID
    """
    from databricks.agents.scorers import backfill_scorers, BackfillScorerConfig
    
    # BackfillScorerConfigオブジェクトを作成
    configs = [
        BackfillScorerConfig(
            scorer=c["scorer"],
            sample_rate=c.get("sample_rate", 1.0),
        )
        for c in scorer_configs
    ]
    
    kwargs = {
        "experiment_id": experiment_id,
        "scorers": configs,
    }
    
    if start_time:
        kwargs["start_time"] = start_time
    if end_time:
        kwargs["end_time"] = end_time
    
    job_id = backfill_scorers(**kwargs)
    
    print(f"Backfill job started: {job_id}")
    for c in scorer_configs:
        print(f"  {c['scorer'].name}: sample_rate={c.get('sample_rate', 1.0)}")
    
    return job_id


def backfill_last_week(experiment_id: str, scorer_names: list[str]) -> str:
    """
    過去1週間のトレースにバックフィル
    
    Args:
        experiment_id: エクスペリメントID
        scorer_names: スコアラー名のリスト
    
    Returns:
        バックフィルジョブID
    """
    one_week_ago = datetime.now() - timedelta(days=7)
    
    return backfill_with_scorer_names(
        experiment_id=experiment_id,
        scorer_names=scorer_names,
        start_time=one_week_ago,
    )


def backfill_date_range(
    experiment_id: str,
    scorer_names: list[str],
    start_date: str,
    end_date: str,
) -> str:
    """
    指定した日付範囲のトレースにバックフィル
    
    Args:
        experiment_id: エクスペリメントID
        scorer_names: スコアラー名のリスト
        start_date: 開始日 (YYYY-MM-DD形式)
        end_date: 終了日 (YYYY-MM-DD形式)
    
    Returns:
        バックフィルジョブID
    """
    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_time = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # 終了日を含む
    
    return backfill_with_scorer_names(
        experiment_id=experiment_id,
        scorer_names=scorer_names,
        start_time=start_time,
        end_time=end_time,
    )


def backfill_new_scorer(
    experiment_id: str,
    new_scorer,
    sample_rate: float = 0.5,
    days_back: int = 30,
) -> str:
    """
    新しく追加したスコアラーを過去のトレースに適用
    
    Args:
        experiment_id: エクスペリメントID
        new_scorer: 新しく登録したスコアラー
        sample_rate: バックフィルのサンプリング率
        days_back: 何日前までさかのぼるか
    
    Returns:
        バックフィルジョブID
    """
    from databricks.agents.scorers import backfill_scorers, BackfillScorerConfig
    
    start_time = datetime.now() - timedelta(days=days_back)
    
    job_id = backfill_scorers(
        experiment_id=experiment_id,
        scorers=[
            BackfillScorerConfig(scorer=new_scorer, sample_rate=sample_rate)
        ],
        start_time=start_time,
    )
    
    print(f"Backfill job started for new scorer: {job_id}")
    print(f"  Scorer: {new_scorer.name}")
    print(f"  Sample rate: {sample_rate}")
    print(f"  Days back: {days_back}")
    
    return job_id


# 使用例
def example_backfill_workflow():
    """
    バックフィルワークフローの例
    """
    from databricks.agents.scorers import backfill_scorers, BackfillScorerConfig
    
    experiment_id = "YOUR_EXPERIMENT_ID"
    
    # ========================================
    # シナリオ1: 既存のスコアラー名でバックフィル
    # ========================================
    print("=== Scenario 1: Backfill with existing scorers ===")
    
    # 現在登録されているスコアラーの設定でバックフィル
    job_id = backfill_scorers(
        experiment_id=experiment_id,
        scorers=["safety_check", "relevance_check"],  # スコアラー名
    )
    print(f"Job ID: {job_id}")
    
    # ========================================
    # シナリオ2: カスタムサンプリング率でバックフィル
    # ========================================
    print("\n=== Scenario 2: Backfill with custom sample rates ===")
    
    # スコアラーを取得または作成
    safety_scorer = Safety().register(name="safety_check")
    safety_scorer = safety_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=0.5)
    )
    
    # バックフィル時は高いサンプリング率を使用
    job_id = backfill_scorers(
        experiment_id=experiment_id,
        scorers=[
            BackfillScorerConfig(scorer=safety_scorer, sample_rate=0.9),
        ],
        start_time=datetime(2024, 6, 1),
        end_time=datetime(2024, 6, 30),
    )
    print(f"Job ID: {job_id}")
    
    # ========================================
    # シナリオ3: 新しいスコアラーを過去データに適用
    # ========================================
    print("\n=== Scenario 3: Apply new scorer to historical data ===")
    
    # 新しいガイドラインスコアラーを作成・登録
    new_guidelines = Guidelines(
        name="response_quality",
        guidelines=[
            "回答は具体的で実用的である必要があります。",
            "専門用語を使う場合は説明を加える必要があります。",
        ]
    ).register(name="response_quality")
    
    new_guidelines = new_guidelines.start(
        sampling_config=ScorerSamplingConfig(sample_rate=0.3)
    )
    
    # 過去30日分のデータにバックフィル
    one_month_ago = datetime.now() - timedelta(days=30)
    
    job_id = backfill_scorers(
        experiment_id=experiment_id,
        scorers=[
            BackfillScorerConfig(scorer=new_guidelines, sample_rate=0.5),
        ],
        start_time=one_month_ago,
    )
    print(f"Job ID: {job_id}")


if __name__ == "__main__":
    # 注意: このコードはDatabricks環境でのみ実行可能です
    # example_backfill_workflow()
    print("This module provides backfill utilities for Databricks environment.")
    print("See example_backfill_workflow() for usage examples.")
