"""
第8章 8.3.5: スコアラーのライフサイクル管理

スコアラーの状態遷移と管理方法を示します。
スコアラーはイミュータブルで、各操作は新しいインスタンスを返します。

注意: 
- Scorer.register(), start(), stop(), update() はDatabricks環境でのみ動作します
- OSS MLflow環境ではスコアラーの登録・スケジューリングは利用できません
- databricks-agentsパッケージが必要です: pip install databricks-agents
"""

import mlflow
from mlflow.genai.scorers import (
    Safety,
    Guidelines,
    ScorerSamplingConfig,
    list_scorers,
    get_scorer,
    delete_scorer,
)


def demonstrate_scorer_lifecycle():
    """
    スコアラーのライフサイクルを実演
    
    状態遷移:
        Unregistered → Registered → Active → Stopped → Deleted
    """
    mlflow.set_experiment("/test/scorer-lifecycle")
    
    # ========================================
    # 1. Unregistered → Registered
    # ========================================
    print("=== Step 1: Register ===")
    
    # スコアラーを定義 (まだUnregistered)
    safety = Safety()
    print(f"Created Safety scorer (unregistered)")
    
    # エクスペリメントに登録 (Registered状態に)
    registered_scorer = safety.register(name="demo_safety_scorer")
    print(f"Registered: {registered_scorer.name}")
    print(f"  Sample rate: {registered_scorer.sample_rate}")  # 0 (未開始)
    
    # ========================================
    # 2. Registered → Active
    # ========================================
    print("\n=== Step 2: Start ===")
    
    # 監視を開始 (Active状態に)
    active_scorer = registered_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=0.5)
    )
    print(f"Started: {active_scorer.name}")
    print(f"  Sample rate: {active_scorer.sample_rate}")  # 0.5
    
    # 注意: 元のスコアラーは変更されない (イミュータブル)
    print(f"  Original sample rate: {registered_scorer.sample_rate}")  # 0
    
    # ========================================
    # 3. Update (Active → Active with different config)
    # ========================================
    print("\n=== Step 3: Update ===")
    
    # サンプリング率を更新
    updated_scorer = active_scorer.update(
        sampling_config=ScorerSamplingConfig(sample_rate=0.8)
    )
    print(f"Updated: {updated_scorer.name}")
    print(f"  New sample rate: {updated_scorer.sample_rate}")  # 0.8
    print(f"  Original sample rate: {active_scorer.sample_rate}")  # 0.5 (変更なし)
    
    # ========================================
    # 4. Active → Stopped
    # ========================================
    print("\n=== Step 4: Stop ===")
    
    # 監視を停止 (登録は維持)
    stopped_scorer = updated_scorer.stop()
    print(f"Stopped: {stopped_scorer.name}")
    print(f"  Sample rate: {stopped_scorer.sample_rate}")  # 0
    
    # ========================================
    # 5. Stopped → Active (再開)
    # ========================================
    print("\n=== Step 5: Restart ===")
    
    # 監視を再開
    restarted_scorer = stopped_scorer.start(
        sampling_config=ScorerSamplingConfig(sample_rate=0.3)
    )
    print(f"Restarted: {restarted_scorer.name}")
    print(f"  Sample rate: {restarted_scorer.sample_rate}")  # 0.3
    
    # ========================================
    # 6. Delete
    # ========================================
    print("\n=== Step 6: Delete ===")
    
    # まず停止してから削除
    final_scorer = restarted_scorer.stop()
    delete_scorer(name="demo_safety_scorer")
    print(f"Deleted: demo_safety_scorer")
    
    return "Lifecycle demonstration complete"


def list_and_manage_scorers(experiment_name: str):
    """
    登録済みスコアラーの一覧取得と管理
    
    Args:
        experiment_name: MLflowエクスペリメント名
    """
    mlflow.set_experiment(experiment_name)
    
    # 登録済みスコアラーの一覧
    print("=== Registered Scorers ===")
    scorers = list_scorers()
    
    if not scorers:
        print("No scorers registered")
        return
    
    for s in scorers:
        status = "Active" if s.sample_rate > 0 else "Stopped"
        print(f"  {s.name}: {status} (sample_rate={s.sample_rate})")
    
    return scorers


def get_and_update_scorer(scorer_name: str, new_sample_rate: float):
    """
    既存のスコアラーを取得して更新
    
    Args:
        scorer_name: スコアラー名
        new_sample_rate: 新しいサンプリング率
    
    Returns:
        更新されたスコアラー
    """
    # スコアラーを取得
    scorer = get_scorer(name=scorer_name)
    print(f"Retrieved: {scorer.name}")
    print(f"  Current sample rate: {scorer.sample_rate}")
    
    # 更新
    updated = scorer.update(
        sampling_config=ScorerSamplingConfig(sample_rate=new_sample_rate)
    )
    print(f"  New sample rate: {updated.sample_rate}")
    
    return updated


def stop_all_scorers(experiment_name: str):
    """
    すべてのスコアラーを停止
    
    Args:
        experiment_name: MLflowエクスペリメント名
    """
    mlflow.set_experiment(experiment_name)
    
    scorers = list_scorers()
    stopped_count = 0
    
    for s in scorers:
        if s.sample_rate > 0:
            s.stop()
            stopped_count += 1
            print(f"Stopped: {s.name}")
    
    print(f"Total stopped: {stopped_count}")


def delete_all_scorers(experiment_name: str, confirm: bool = False):
    """
    すべてのスコアラーを削除
    
    Args:
        experiment_name: MLflowエクスペリメント名
        confirm: True の場合のみ実行
    """
    if not confirm:
        print("WARNING: This will delete all scorers. Set confirm=True to proceed.")
        return
    
    mlflow.set_experiment(experiment_name)
    
    scorers = list_scorers()
    deleted_count = 0
    
    for s in scorers:
        # まず停止
        if s.sample_rate > 0:
            s.stop()
        # 削除
        delete_scorer(name=s.name)
        deleted_count += 1
        print(f"Deleted: {s.name}")
    
    print(f"Total deleted: {deleted_count}")


# サンプリング設定のベストプラクティス
RECOMMENDED_SAMPLE_RATES = {
    # 安全性は最優先: 100%
    "safety": 1.0,
    
    # 正確性は重要: 50-100%
    "correctness": 0.5,
    
    # RAG品質: 30-50%
    "groundedness": 0.5,
    "relevance": 0.5,
    "retrieval_sufficiency": 0.3,
    
    # カスタムガイドライン: 30-50%
    "guidelines": 0.5,
    
    # 複雑/高コストなカスタムスコアラー: 5-20%
    "complex_custom": 0.1,
}


def setup_scorers_with_best_practices(experiment_name: str) -> dict:
    """
    ベストプラクティスに基づいたスコアラーセットアップ
    """
    mlflow.set_experiment(experiment_name)
    
    scorers = {}
    
    # 安全性 (100%)
    safety = Safety().register(name="safety")
    safety = safety.start(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0)
    )
    scorers["safety"] = safety
    
    # ガイドライン (50%)
    guidelines = Guidelines(
        name="tone",
        guidelines=["回答は専門的で丁寧なトーンである必要があります。"]
    ).register(name="tone_check")
    guidelines = guidelines.start(
        sampling_config=ScorerSamplingConfig(sample_rate=0.5)
    )
    scorers["tone_check"] = guidelines
    
    print("Scorers setup with best practices:")
    for name, s in scorers.items():
        print(f"  {name}: sample_rate={s.sample_rate}")
    
    return scorers


# 使用例
if __name__ == "__main__":
    mlflow.set_tracking_uri("databricks")
    
    # ライフサイクルのデモ
    # demonstrate_scorer_lifecycle()
    
    # スコアラー一覧の取得
    # list_and_manage_scorers("/production/my-app")
    
    # ベストプラクティスに基づくセットアップ
    # setup_scorers_with_best_practices("/production/my-app")
    
    pass
