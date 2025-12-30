"""
第8章 8.3.3: ユーザーフィードバックの収集

MLflowの log_feedback() を使用してユーザーフィードバックを
トレースに紐付けて記録します。
"""

from typing import Optional, Union, Any
from enum import Enum
import mlflow
from mlflow.entities.assessment import AssessmentSource, AssessmentSourceType


class FeedbackType(Enum):
    """フィードバックの種類"""
    THUMBS = "thumbs"  # サムズアップ/ダウン
    RATING = "rating"  # 数値評価 (1-5)
    TEXT = "text"  # テキストコメント
    ISSUE = "issue"  # 問題報告


def log_thumbs_feedback(
    trace_id: str,
    is_positive: bool,
    user_id: str,
    comment: Optional[str] = None,
) -> None:
    """
    サムズアップ/ダウンフィードバックを記録
    
    Args:
        trace_id: トレースID
        is_positive: True=サムズアップ, False=サムズダウン
        user_id: フィードバックを提供したユーザーID
        comment: 追加コメント (オプション)
    """
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_thumbs",
        value=is_positive,
        rationale=comment or ("ユーザーが役立ったと評価" if is_positive else "ユーザーが役立たなかったと評価"),
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        )
    )


def log_rating_feedback(
    trace_id: str,
    rating: int,
    user_id: str,
    rationale: Optional[str] = None,
) -> None:
    """
    評価スコア(1-5)フィードバックを記録
    
    Args:
        trace_id: トレースID
        rating: 評価スコア (1-5)
        user_id: フィードバックを提供したユーザーID
        rationale: 評価理由 (オプション)
    """
    if not 1 <= rating <= 5:
        raise ValueError("Rating must be between 1 and 5")
    
    mlflow.log_feedback(
        trace_id=trace_id,
        name="user_rating",
        value=rating,
        rationale=rationale or f"ユーザーが{rating}/5で評価",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        )
    )


def log_text_feedback(
    trace_id: str,
    comment: str,
    user_id: str,
    category: Optional[str] = None,
) -> None:
    """
    テキストコメントフィードバックを記録
    
    Args:
        trace_id: トレースID
        comment: フィードバックコメント
        user_id: フィードバックを提供したユーザーID
        category: フィードバックカテゴリ (オプション)
    """
    name = f"user_comment_{category}" if category else "user_comment"
    
    mlflow.log_feedback(
        trace_id=trace_id,
        name=name,
        value=comment,
        rationale="ユーザーからのテキストフィードバック",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        )
    )


def log_issue_report(
    trace_id: str,
    issue_type: str,
    description: str,
    user_id: str,
    severity: str = "medium",
) -> None:
    """
    問題報告を記録
    
    Args:
        trace_id: トレースID
        issue_type: 問題の種類 (例: "incorrect", "inappropriate", "slow", "security")
        description: 問題の説明
        user_id: 報告したユーザーID
        severity: 重要度 (low, medium, high, critical)
    """
    mlflow.log_feedback(
        trace_id=trace_id,
        name=f"issue_report_{issue_type}",
        value=False,  # 問題があるのでFalse
        rationale=f"[{severity.upper()}] {description}",
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        ),
        metadata={
            "issue_type": issue_type,
            "severity": severity,
            "description": description,
        }
    )


def log_user_correction(
    trace_id: str,
    expected_response: str,
    user_id: str,
    correction_type: str = "factual",
) -> None:
    """
    ユーザーによる修正(正しい回答)を記録
    
    Args:
        trace_id: トレースID
        expected_response: ユーザーが提供した正しい回答
        user_id: 修正を提供したユーザーID
        correction_type: 修正の種類 (factual, format, tone, etc.)
    """
    # 期待値として記録
    mlflow.log_expectation(
        trace_id=trace_id,
        name=f"user_correction_{correction_type}",
        value=expected_response,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=user_id,
        )
    )


class FeedbackCollector:
    """
    フィードバック収集を管理するクラス
    
    使用例:
        collector = FeedbackCollector(user_id="user-123")
        
        # トレース実行後
        collector.set_trace(trace_id)
        
        # フィードバック収集
        collector.thumbs_up()
        collector.rate(4)
        collector.comment("とても役立ちました")
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self._current_trace_id: Optional[str] = None
    
    def set_trace(self, trace_id: str) -> "FeedbackCollector":
        """フィードバック対象のトレースを設定"""
        self._current_trace_id = trace_id
        return self
    
    def _ensure_trace(self) -> str:
        """トレースIDが設定されていることを確認"""
        if self._current_trace_id is None:
            # 最後のアクティブトレースを使用
            self._current_trace_id = mlflow.get_last_active_trace_id()
            if self._current_trace_id is None:
                raise ValueError("No trace ID set. Call set_trace() first or run a traced function.")
        return self._current_trace_id
    
    def thumbs_up(self, comment: str = None) -> None:
        """サムズアップを記録"""
        log_thumbs_feedback(
            trace_id=self._ensure_trace(),
            is_positive=True,
            user_id=self.user_id,
            comment=comment,
        )
    
    def thumbs_down(self, comment: str = None) -> None:
        """サムズダウンを記録"""
        log_thumbs_feedback(
            trace_id=self._ensure_trace(),
            is_positive=False,
            user_id=self.user_id,
            comment=comment,
        )
    
    def rate(self, rating: int, rationale: str = None) -> None:
        """評価スコアを記録"""
        log_rating_feedback(
            trace_id=self._ensure_trace(),
            rating=rating,
            user_id=self.user_id,
            rationale=rationale,
        )
    
    def comment(self, text: str, category: str = None) -> None:
        """テキストコメントを記録"""
        log_text_feedback(
            trace_id=self._ensure_trace(),
            comment=text,
            user_id=self.user_id,
            category=category,
        )
    
    def report_issue(
        self,
        issue_type: str,
        description: str,
        severity: str = "medium",
    ) -> None:
        """問題を報告"""
        log_issue_report(
            trace_id=self._ensure_trace(),
            issue_type=issue_type,
            description=description,
            user_id=self.user_id,
            severity=severity,
        )
    
    def correct(self, expected_response: str, correction_type: str = "factual") -> None:
        """正しい回答を提供"""
        log_user_correction(
            trace_id=self._ensure_trace(),
            expected_response=expected_response,
            user_id=self.user_id,
            correction_type=correction_type,
        )


# 使用例
if __name__ == "__main__":
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/test/feedback-example")
    
    # トレースを生成 (ダミー)
    @mlflow.trace
    def sample_response(query: str) -> str:
        return f"Answer to: {query}"
    
    response = sample_response("What is MLflow?")
    trace_id = mlflow.get_last_active_trace_id()
    
    print(f"Trace ID: {trace_id}")
    
    # フィードバックを記録
    collector = FeedbackCollector(user_id="user-123")
    collector.set_trace(trace_id)
    
    # サムズアップ
    collector.thumbs_up("とても分かりやすい説明でした")
    
    # 評価スコア
    collector.rate(4, "概ね良いが、もう少し詳しく説明してほしい")
    
    # テキストコメント
    collector.comment("具体例があるとさらに良い", category="suggestion")
    
    print("Feedback logged successfully!")
