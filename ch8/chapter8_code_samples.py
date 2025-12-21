# 第8章 監視と運用 - コードサンプル集
# MLflowで実践するLLMOps

"""
このファイルには、第8章で参照されている完全なコードサンプルが含まれています。
本文では抜粋のみを掲載し、完全な実装はこちらを参照してください。
"""

# =============================================================================
# 8.1 本番環境でのトレースベース監視
# =============================================================================

# -----------------------------------------------------------------------------
# 8.1.3 本番トレーシングの基本設定
# -----------------------------------------------------------------------------

import mlflow
import os
import atexit
import signal
from datetime import datetime

def setup_production_tracing(
    service_name: str,
    environment: str = "production",
    tracking_uri: str = "databricks"
):
    """本番環境向けトレーシング設定"""
    
    # 非同期ログ記録を有効化（本番環境では必須）
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "true"
    
    # サービス名の設定（トレースのグループ化に使用）
    os.environ["OTEL_SERVICE_NAME"] = service_name
    
    # 環境タグの設定
    os.environ["MLFLOW_TRACE_ENVIRONMENT"] = environment
    
    # MLflow接続設定
    mlflow.set_tracking_uri(tracking_uri)
    
    # 本番用Experimentの作成または取得
    experiment_name = f"/{environment}/{service_name}/{datetime.now().strftime('%Y-%m')}"
    mlflow.set_experiment(experiment_name)
    
    # グレースフルシャットダウンの設定
    def graceful_shutdown(signum=None, frame=None):
        print("Flushing pending traces...")
        mlflow.flush_trace_async_logging()
        print("Trace flushing complete.")
    
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    atexit.register(graceful_shutdown)
    
    # 自動トレーシングの有効化
    mlflow.openai.autolog()
    
    print(f"MLflow Tracing initialized for experiment: {experiment_name}")


# -----------------------------------------------------------------------------
# 8.1.5 トレースへのメタデータ追加
# -----------------------------------------------------------------------------

from typing import Optional
import uuid

class TracingContext:
    """トレーシングコンテキストを管理するクラス"""
    
    def __init__(
        self,
        user_id: str,
        session_id: str,
        request_id: Optional[str] = None,
        environment: str = "production"
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id or str(uuid.uuid4())
        self.environment = environment
    
    def apply_to_trace(self):
        """現在のトレースにコンテキスト情報を適用"""
        mlflow.update_current_trace(tags={
            # 標準タグ
            "mlflow.trace.user": self.user_id,
            "mlflow.trace.session": self.session_id,
            "mlflow.trace.request_id": self.request_id,
            
            # カスタムタグ
            "environment": self.environment,
            "service.version": os.getenv("SERVICE_VERSION", "unknown"),
            "deployment.region": os.getenv("DEPLOYMENT_REGION", "unknown"),
        })


# -----------------------------------------------------------------------------
# 8.1.6 サンプリング戦略
# -----------------------------------------------------------------------------

import random
import time
import threading
from functools import wraps
from collections import deque
from datetime import timedelta
from typing import Callable, Set

# 確率的サンプリング
def probabilistic_trace(sample_rate: float = 0.1):
    """確率的サンプリングデコレータ
    
    Args:
        sample_rate: サンプリング率（0.0-1.0）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            should_trace = random.random() < sample_rate
            
            if should_trace:
                with mlflow.start_span(name=func.__name__) as span:
                    span.set_attributes({"sampling.rate": sample_rate})
                    result = func(*args, **kwargs)
                    return result
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# 条件付きサンプリング
class ConditionalSampler:
    """条件付きサンプリングを管理するクラス"""
    
    def __init__(
        self,
        base_sample_rate: float = 0.1,
        error_sample_rate: float = 1.0,
        slow_request_threshold_ms: float = 5000,
        priority_users: Optional[Set[str]] = None
    ):
        self.base_sample_rate = base_sample_rate
        self.error_sample_rate = error_sample_rate
        self.slow_request_threshold_ms = slow_request_threshold_ms
        self.priority_users = priority_users or set()
    
    def should_trace(
        self,
        user_id: Optional[str] = None,
        is_error: bool = False,
        latency_ms: Optional[float] = None
    ) -> bool:
        """トレースすべきかどうかを判定"""
        
        # エラーは常にトレース
        if is_error and random.random() < self.error_sample_rate:
            return True
        
        # 優先ユーザーは常にトレース
        if user_id and user_id in self.priority_users:
            return True
        
        # 遅いリクエストは常にトレース
        if latency_ms and latency_ms > self.slow_request_threshold_ms:
            return True
        
        # それ以外は確率的サンプリング
        return random.random() < self.base_sample_rate


# 適応型サンプリング
class AdaptiveSampler:
    """負荷に応じてサンプリング率を調整するサンプラー"""
    
    def __init__(
        self,
        target_traces_per_minute: int = 100,
        min_sample_rate: float = 0.01,
        max_sample_rate: float = 1.0,
        adjustment_interval_seconds: int = 60
    ):
        self.target_traces_per_minute = target_traces_per_minute
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.adjustment_interval = adjustment_interval_seconds
        
        self.current_sample_rate = max_sample_rate
        self.request_counts = deque(maxlen=60)
        self.trace_counts = deque(maxlen=60)
        
        self._lock = threading.Lock()
        self._start_adjustment_thread()
    
    def _start_adjustment_thread(self):
        """サンプリング率調整スレッドを開始"""
        def adjust_loop():
            while True:
                time.sleep(self.adjustment_interval)
                self._adjust_sample_rate()
        
        thread = threading.Thread(target=adjust_loop, daemon=True)
        thread.start()
    
    def _adjust_sample_rate(self):
        """サンプリング率を調整"""
        with self._lock:
            if len(self.trace_counts) == 0:
                return
            
            current_traces_per_minute = sum(self.trace_counts)
            current_requests_per_minute = sum(self.request_counts)
            
            if current_requests_per_minute == 0:
                return
            
            ideal_rate = self.target_traces_per_minute / current_requests_per_minute
            new_rate = (self.current_sample_rate + ideal_rate) / 2
            
            self.current_sample_rate = max(
                self.min_sample_rate,
                min(self.max_sample_rate, new_rate)
            )
    
    def record_request(self, was_traced: bool):
        """リクエストを記録"""
        with self._lock:
            current_second = int(time.time()) % 60
            
            if len(self.request_counts) <= current_second:
                self.request_counts.append(0)
                self.trace_counts.append(0)
            
            self.request_counts[-1] += 1
            if was_traced:
                self.trace_counts[-1] += 1
    
    def should_trace(self) -> bool:
        """トレースすべきかどうかを判定"""
        return random.random() < self.current_sample_rate


# =============================================================================
# 8.2 トークン使用量とコストの可視化
# =============================================================================

# -----------------------------------------------------------------------------
# 8.2.3 コスト計算の実装
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class ModelPricing:
    """モデルの料金情報"""
    model_name: str
    input_price_per_1k: float  # USD per 1K tokens
    output_price_per_1k: float
    cached_input_price_per_1k: Optional[float] = None
    effective_date: Optional[datetime] = None
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0
    ) -> Dict[str, float]:
        """コストを計算"""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        
        cached_cost = 0
        if cached_input_tokens and self.cached_input_price_per_1k:
            cached_cost = (cached_input_tokens / 1000) * self.cached_input_price_per_1k
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cached_input_cost": cached_cost,
            "total_cost": input_cost + output_cost + cached_cost
        }


class CostCalculator:
    """LLMコスト計算クラス"""
    
    def __init__(self, pricing_config_path: Optional[str] = None):
        self.pricing_models: Dict[str, ModelPricing] = {}
        self._load_default_pricing()
        
        if pricing_config_path:
            self._load_pricing_config(pricing_config_path)
    
    def _load_default_pricing(self):
        """デフォルトの料金設定をロード（2024年12月時点）"""
        default_pricing = [
            # OpenAI Models
            ModelPricing("gpt-4o", 0.0025, 0.01, 0.00125),
            ModelPricing("gpt-4o-mini", 0.00015, 0.0006, 0.000075),
            ModelPricing("gpt-4-turbo", 0.01, 0.03),
            ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015),
            ModelPricing("o1-preview", 0.015, 0.06),
            ModelPricing("o1-mini", 0.003, 0.012),
            
            # Anthropic Models
            ModelPricing("claude-3-5-sonnet-20241022", 0.003, 0.015, 0.0003),
            ModelPricing("claude-3-5-haiku-20241022", 0.0008, 0.004, 0.00008),
            ModelPricing("claude-3-opus-20240229", 0.015, 0.075, 0.0015),
            
            # Google Models
            ModelPricing("gemini-1.5-pro", 0.00125, 0.005),
            ModelPricing("gemini-1.5-flash", 0.000075, 0.0003),
        ]
        
        for pricing in default_pricing:
            self.pricing_models[pricing.model_name] = pricing
    
    def _load_pricing_config(self, config_path: str):
        """外部設定ファイルから料金をロード"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for model_config in config.get("models", []):
            pricing = ModelPricing(
                model_name=model_config["name"],
                input_price_per_1k=model_config["input_price_per_1k"],
                output_price_per_1k=model_config["output_price_per_1k"],
                cached_input_price_per_1k=model_config.get("cached_input_price_per_1k"),
                effective_date=datetime.fromisoformat(model_config["effective_date"])
                    if model_config.get("effective_date") else None
            )
            self.pricing_models[pricing.model_name] = pricing
    
    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0
    ) -> Dict[str, float]:
        """指定モデルのコストを計算"""
        normalized_name = self._normalize_model_name(model_name)
        
        if normalized_name not in self.pricing_models:
            raise ValueError(f"Unknown model: {model_name}")
        
        pricing = self.pricing_models[normalized_name]
        return pricing.calculate_cost(input_tokens, output_tokens, cached_input_tokens)
    
    def _normalize_model_name(self, model_name: str) -> str:
        """モデル名を正規化"""
        aliases = {
            "gpt-4o-2024-11-20": "gpt-4o",
            "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
        }
        return aliases.get(model_name, model_name)
    
    def get_available_models(self) -> list:
        """利用可能なモデル一覧を取得"""
        return list(self.pricing_models.keys())


# コスト追跡ミドルウェア
class CostTrackingMiddleware:
    """コスト追跡ミドルウェア"""
    
    def __init__(self, calculator: CostCalculator):
        self.calculator = calculator
    
    def track_cost(self, model_name: str):
        """コスト追跡デコレータ"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                trace_id = mlflow.get_last_active_trace_id()
                if not trace_id:
                    return result
                
                trace = mlflow.get_trace(trace_id=trace_id)
                token_usage = trace.info.token_usage
                
                if token_usage:
                    cost = self.calculator.calculate_cost(
                        model_name=model_name,
                        input_tokens=token_usage.get("input_tokens", 0),
                        output_tokens=token_usage.get("output_tokens", 0)
                    )
                    
                    mlflow.update_current_trace(tags={
                        "cost.input_usd": str(cost["input_cost"]),
                        "cost.output_usd": str(cost["output_cost"]),
                        "cost.total_usd": str(cost["total_cost"]),
                        "cost.model": model_name
                    })
                
                return result
            return wrapper
        return decorator


# -----------------------------------------------------------------------------
# 8.2.4 コストレポート生成
# -----------------------------------------------------------------------------

import pandas as pd

@dataclass
class CostReport:
    """コストレポート"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_requests: int
    model_breakdown: Dict[str, float]
    daily_costs: List[Dict]
    top_users: List[Dict]
    anomalies: List[Dict]


class CostReporter:
    """コストレポート生成クラス"""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
    
    def generate_weekly_report(self) -> CostReport:
        """週次コストレポートを生成"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        traces = mlflow.search_traces(
            experiment_names=[self.experiment_name],
            filter_string=f"timestamp >= {int(start_date.timestamp() * 1000)}",
            max_results=10000
        )
        
        df = self._traces_to_dataframe(traces)
        
        if df.empty:
            return self._empty_report(start_date, end_date)
        
        total_cost = df['cost_total'].sum()
        total_requests = len(df)
        model_breakdown = df.groupby('model')['cost_total'].sum().to_dict()
        
        df['date'] = df['timestamp'].dt.date
        daily_costs = df.groupby('date').agg({
            'cost_total': 'sum',
            'request_id': 'count'
        }).reset_index().to_dict('records')
        
        top_users = df.groupby('user_id')['cost_total'].sum().nlargest(10).reset_index().to_dict('records')
        anomalies = self._detect_anomalies(df)
        
        return CostReport(
            period_start=start_date,
            period_end=end_date,
            total_cost=total_cost,
            total_requests=total_requests,
            model_breakdown=model_breakdown,
            daily_costs=daily_costs,
            top_users=top_users,
            anomalies=anomalies
        )
    
    def _traces_to_dataframe(self, traces) -> pd.DataFrame:
        """トレースをDataFrameに変換"""
        records = []
        for trace in traces:
            tags = trace.info.tags or {}
            records.append({
                'request_id': trace.info.request_id,
                'timestamp': pd.to_datetime(trace.info.timestamp_ms, unit='ms'),
                'user_id': tags.get('mlflow.trace.user'),
                'model': tags.get('cost.model'),
                'cost_total': float(tags.get('cost.total_usd', 0)),
                'input_tokens': trace.info.token_usage.get('input_tokens', 0) if trace.info.token_usage else 0,
                'output_tokens': trace.info.token_usage.get('output_tokens', 0) if trace.info.token_usage else 0,
            })
        return pd.DataFrame(records)
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """コスト異常を検出"""
        anomalies = []
        
        mean_cost = df['cost_total'].mean()
        std_cost = df['cost_total'].std()
        threshold = mean_cost + 3 * std_cost
        
        high_cost_requests = df[df['cost_total'] > threshold]
        for _, row in high_cost_requests.iterrows():
            anomalies.append({
                'type': 'high_cost_request',
                'request_id': row['request_id'],
                'cost': row['cost_total'],
                'threshold': threshold,
                'timestamp': row['timestamp'].isoformat()
            })
        
        return anomalies
    
    def _empty_report(self, start_date: datetime, end_date: datetime) -> CostReport:
        return CostReport(
            period_start=start_date,
            period_end=end_date,
            total_cost=0,
            total_requests=0,
            model_breakdown={},
            daily_costs=[],
            top_users=[],
            anomalies=[]
        )


# =============================================================================
# 8.3 品質メトリクスのリアルタイム追跡
# =============================================================================

# -----------------------------------------------------------------------------
# 8.3.2 パフォーマンス監視
# -----------------------------------------------------------------------------

import statistics

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    error_rate: float
    timeout_rate: float


class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self, window_size_seconds: int = 300):
        self.window_size = timedelta(seconds=window_size_seconds)
        self.metrics_buffer: List[Dict] = []
    
    def record_request(
        self,
        latency_ms: float,
        is_error: bool = False,
        is_timeout: bool = False
    ):
        """リクエストメトリクスを記録"""
        self.metrics_buffer.append({
            'timestamp': datetime.now(),
            'latency_ms': latency_ms,
            'is_error': is_error,
            'is_timeout': is_timeout
        })
        
        cutoff = datetime.now() - self.window_size
        self.metrics_buffer = [
            m for m in self.metrics_buffer 
            if m['timestamp'] > cutoff
        ]
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """現在のメトリクスを取得"""
        if not self.metrics_buffer:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0)
        
        latencies = [m['latency_ms'] for m in self.metrics_buffer]
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        
        return PerformanceMetrics(
            latency_p50_ms=latencies_sorted[int(n * 0.50)],
            latency_p95_ms=latencies_sorted[int(n * 0.95)],
            latency_p99_ms=latencies_sorted[int(n * 0.99)] if n > 100 else latencies_sorted[-1],
            throughput_rps=n / self.window_size.total_seconds(),
            error_rate=sum(1 for m in self.metrics_buffer if m['is_error']) / n,
            timeout_rate=sum(1 for m in self.metrics_buffer if m['is_timeout']) / n
        )


# -----------------------------------------------------------------------------
# 8.3.3 ユーザーフィードバック収集
# -----------------------------------------------------------------------------

from enum import Enum
from typing import Any

class FeedbackType(Enum):
    """フィードバックの種類"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    TEXT_COMMENT = "text_comment"
    CORRECTION = "correction"
    REPORT = "report"


@dataclass
class UserFeedback:
    """ユーザーフィードバック"""
    feedback_id: str
    trace_id: str
    user_id: str
    feedback_type: FeedbackType
    value: Any
    comment: Optional[str] = None
    timestamp: datetime = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.feedback_id is None:
            self.feedback_id = str(uuid.uuid4())


class FeedbackCollector:
    """フィードバック収集クラス"""
    
    def __init__(self):
        self.feedback_buffer: List[UserFeedback] = []
    
    def record_thumbs_feedback(
        self,
        trace_id: str,
        user_id: str,
        is_positive: bool,
        comment: Optional[str] = None
    ) -> UserFeedback:
        """サムズアップ/ダウンフィードバックを記録"""
        feedback_type = FeedbackType.THUMBS_UP if is_positive else FeedbackType.THUMBS_DOWN
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            trace_id=trace_id,
            user_id=user_id,
            feedback_type=feedback_type,
            value=is_positive,
            comment=comment
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def record_rating(
        self,
        trace_id: str,
        user_id: str,
        rating: int,
        comment: Optional[str] = None
    ) -> UserFeedback:
        """評価スコア（1-5）を記録"""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            trace_id=trace_id,
            user_id=user_id,
            feedback_type=FeedbackType.RATING,
            value=rating,
            comment=comment
        )
        
        self._store_feedback(feedback)
        return feedback
    
    def _store_feedback(self, feedback: UserFeedback):
        """フィードバックを保存"""
        self.feedback_buffer.append(feedback)
        
        try:
            mlflow.log_feedback(
                trace_id=feedback.trace_id,
                name=feedback.feedback_type.value,
                value=feedback.value,
                comment=feedback.comment
            )
        except Exception as e:
            print(f"Failed to log feedback to MLflow: {e}")
    
    def get_feedback_stats(self, hours: int = 24) -> Dict:
        """フィードバック統計を取得"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [f for f in self.feedback_buffer if f.timestamp > cutoff]
        
        thumbs_up = sum(1 for f in recent if f.feedback_type == FeedbackType.THUMBS_UP)
        thumbs_down = sum(1 for f in recent if f.feedback_type == FeedbackType.THUMBS_DOWN)
        ratings = [f.value for f in recent if f.feedback_type == FeedbackType.RATING]
        
        return {
            'total_feedback': len(recent),
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down,
            'positive_rate': thumbs_up / (thumbs_up + thumbs_down) if (thumbs_up + thumbs_down) > 0 else None,
            'average_rating': statistics.mean(ratings) if ratings else None,
        }


# -----------------------------------------------------------------------------
# 8.3.5 継続的評価パイプライン
# -----------------------------------------------------------------------------

import schedule
from mlflow.genai import evaluate, create_dataset
from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety


class ContinuousEvaluationPipeline:
    """継続的評価パイプライン"""
    
    def __init__(
        self,
        experiment_name: str,
        scorers: List,
        sample_rate: float = 0.1,
        evaluation_interval_minutes: int = 60,
        alert_thresholds: Optional[Dict] = None
    ):
        self.experiment_name = experiment_name
        self.scorers = scorers
        self.sample_rate = sample_rate
        self.evaluation_interval = evaluation_interval_minutes
        self.alert_thresholds = alert_thresholds or {
            'relevance': 0.7,
            'safety': 0.9,
            'groundedness': 0.7
        }
        
        self.evaluation_history: List[Dict] = []
        self.alert_handlers: List[Callable] = []
    
    def add_alert_handler(self, handler: Callable):
        """アラートハンドラを追加"""
        self.alert_handlers.append(handler)
    
    def run_evaluation(self) -> Dict:
        """評価を実行"""
        print(f"[{datetime.now()}] Starting evaluation...")
        
        since = datetime.now() - timedelta(minutes=self.evaluation_interval)
        traces = mlflow.search_traces(
            experiment_names=[self.experiment_name],
            filter_string=f"timestamp >= {int(since.timestamp() * 1000)}",
            max_results=1000
        )
        
        if not traces:
            print("No traces found for evaluation")
            return {}
        
        sample_size = max(1, int(len(traces) * self.sample_rate))
        sampled_traces = random.sample(list(traces), sample_size)
        
        dataset = create_dataset(f"continuous-eval-{datetime.now().strftime('%Y%m%d%H%M')}")
        dataset.insert(sampled_traces)
        
        results = evaluate(data=dataset, scorers=self.scorers)
        
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'traces_evaluated': sample_size,
            'total_traces': len(traces),
            'metrics': results.metrics
        }
        self.evaluation_history.append(evaluation_result)
        
        self._check_thresholds(results.metrics)
        
        print(f"Evaluation complete. Metrics: {results.metrics}")
        return evaluation_result
    
    def _check_thresholds(self, metrics: Dict):
        """閾値をチェックしてアラートを発火"""
        for metric_name, threshold in self.alert_thresholds.items():
            if metric_name in metrics and metrics[metric_name] < threshold:
                alert = {
                    'type': 'quality_threshold_violation',
                    'metric': metric_name,
                    'value': metrics[metric_name],
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat()
                }
                
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        print(f"Alert handler failed: {e}")
    
    def start_scheduler(self):
        """定期実行スケジューラを開始"""
        schedule.every(self.evaluation_interval).minutes.do(self.run_evaluation)
        
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)
        
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
        print(f"Scheduler started. Running every {self.evaluation_interval} minutes.")


# =============================================================================
# 8.4 アラート設定とインシデント対応
# =============================================================================

# -----------------------------------------------------------------------------
# 8.4.1 アラート戦略の設計
# -----------------------------------------------------------------------------

class AlertSeverity(Enum):
    """アラートの重要度"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertCategory(Enum):
    """アラートのカテゴリ"""
    AVAILABILITY = "availability"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    COST = "cost"
    SECURITY = "security"


@dataclass
class Alert:
    """アラート"""
    alert_id: str
    category: AlertCategory
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    metadata: Optional[Dict] = None
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'category': self.category.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class AlertRule:
    """アラートルール"""
    
    def __init__(
        self,
        name: str,
        category: AlertCategory,
        metric_name: str,
        condition: Callable[[float], bool],
        threshold: float,
        severity: AlertSeverity,
        title_template: str,
        description_template: str,
        cooldown_minutes: int = 15
    ):
        self.name = name
        self.category = category
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold
        self.severity = severity
        self.title_template = title_template
        self.description_template = description_template
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_fired: Optional[datetime] = None
    
    def evaluate(self, current_value: float) -> Optional[Alert]:
        """ルールを評価してアラートを生成"""
        if not self.condition(current_value):
            return None
        
        if self.last_fired and (datetime.now() - self.last_fired) < self.cooldown:
            return None
        
        self.last_fired = datetime.now()
        
        return Alert(
            alert_id=f"{self.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            category=self.category,
            severity=self.severity,
            title=self.title_template.format(value=current_value, threshold=self.threshold),
            description=self.description_template.format(value=current_value, threshold=self.threshold),
            metric_name=self.metric_name,
            current_value=current_value,
            threshold=self.threshold,
            timestamp=datetime.now()
        )


# -----------------------------------------------------------------------------
# 8.4.3 アラート通知システム
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod
import requests

class AlertNotifier(ABC):
    """アラート通知の基底クラス"""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        pass


class SlackNotifier(AlertNotifier):
    """Slack通知"""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, alert: Alert) -> bool:
        severity_emoji = {
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.INFO: "ℹ️"
        }
        
        severity_color = {
            AlertSeverity.CRITICAL: "#FF0000",
            AlertSeverity.WARNING: "#FFA500",
            AlertSeverity.INFO: "#0000FF"
        }
        
        payload = {
            "channel": self.channel,
            "attachments": [{
                "color": severity_color[alert.severity],
                "blocks": [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{severity_emoji[alert.severity]} {alert.title}"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*カテゴリ:*\n{alert.category.value}"},
                            {"type": "mrkdwn", "text": f"*重要度:*\n{alert.severity.value}"},
                            {"type": "mrkdwn", "text": f"*メトリクス:*\n{alert.metric_name}"},
                            {"type": "mrkdwn", "text": f"*現在値:*\n{alert.current_value:.4f}"},
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": alert.description}
                    },
                ]
            }]
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Slack notification failed: {e}")
            return False


class AlertManager:
    """アラート管理クラス"""
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.notifiers: Dict[AlertSeverity, List[AlertNotifier]] = {
            AlertSeverity.CRITICAL: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.INFO: []
        }
        self.alert_history: List[Alert] = []
    
    def add_rule(self, rule: AlertRule):
        """ルールを追加"""
        self.rules.append(rule)
    
    def add_notifier(
        self,
        notifier: AlertNotifier,
        severities: List[AlertSeverity] = None
    ):
        """通知先を追加"""
        if severities is None:
            severities = list(AlertSeverity)
        
        for severity in severities:
            self.notifiers[severity].append(notifier)
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """メトリクスを評価してアラートを生成"""
        alerts = []
        
        for rule in self.rules:
            if rule.metric_name in metrics:
                alert = rule.evaluate(metrics[rule.metric_name])
                if alert:
                    alerts.append(alert)
                    self.alert_history.append(alert)
                    self._send_notifications(alert)
        
        return alerts
    
    def _send_notifications(self, alert: Alert):
        """通知を送信"""
        notifiers = self.notifiers.get(alert.severity, [])
        
        for notifier in notifiers:
            try:
                success = notifier.send(alert)
                if not success:
                    print(f"Notification failed for {type(notifier).__name__}")
            except Exception as e:
                print(f"Notification error: {e}")


# =============================================================================
# 8.5 OpenTelemetryとの統合
# =============================================================================

# -----------------------------------------------------------------------------
# 8.5.2 OTLP設定
# -----------------------------------------------------------------------------

class OTelConfiguration:
    """OpenTelemetry設定のベストプラクティス"""
    
    @staticmethod
    def configure_production(
        service_name: str,
        service_version: str,
        environment: str,
        collector_endpoint: str,
        sample_rate: float = 1.0,
        enable_metrics: bool = True,
        enable_logs: bool = False
    ):
        """本番環境向けOTel設定"""
        
        os.environ["OTEL_SERVICE_NAME"] = service_name
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = collector_endpoint
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
        
        resource_attrs = [
            f"service.name={service_name}",
            f"service.version={service_version}",
            f"deployment.environment={environment}",
            f"service.namespace=llm-applications",
        ]
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)
        
        if sample_rate < 1.0:
            os.environ["OTEL_TRACES_SAMPLER"] = "parentbased_traceidratio"
            os.environ["OTEL_TRACES_SAMPLER_ARG"] = str(sample_rate)
        
        if enable_metrics:
            os.environ["OTEL_METRICS_EXPORTER"] = "otlp"
        else:
            os.environ["OTEL_METRICS_EXPORTER"] = "none"
        
        if enable_logs:
            os.environ["OTEL_LOGS_EXPORTER"] = "otlp"
        else:
            os.environ["OTEL_LOGS_EXPORTER"] = "none"
        
        os.environ["OTEL_BSP_SCHEDULE_DELAY"] = "5000"
        os.environ["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] = "512"
        os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "2048"
        os.environ["OTEL_EXPORTER_OTLP_TIMEOUT"] = "10000"
    
    @staticmethod
    def configure_development():
        """開発環境向けOTel設定"""
        os.environ["OTEL_SERVICE_NAME"] = "llm-app-dev"
        os.environ["OTEL_TRACES_EXPORTER"] = "console"
        os.environ["OTEL_METRICS_EXPORTER"] = "none"
        os.environ["OTEL_LOGS_EXPORTER"] = "none"


# -----------------------------------------------------------------------------
# Prometheusメトリクス
# -----------------------------------------------------------------------------

from prometheus_client import Counter, Histogram, Gauge, start_http_server

LLM_REQUEST_COUNT = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['model', 'status', 'endpoint']
)

LLM_LATENCY = Histogram(
    'llm_request_duration_seconds',
    'LLM request latency in seconds',
    ['model', 'endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

LLM_TOKENS = Counter(
    'llm_tokens_total',
    'Total tokens used',
    ['model', 'token_type']
)

LLM_COST = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['model']
)


class PrometheusMetricsCollector:
    """Prometheusメトリクス収集クラス"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")
    
    def record_request(
        self,
        model: str,
        endpoint: str,
        latency_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        status: str = "success"
    ):
        """リクエストメトリクスを記録"""
        LLM_REQUEST_COUNT.labels(model=model, status=status, endpoint=endpoint).inc()
        LLM_LATENCY.labels(model=model, endpoint=endpoint).observe(latency_seconds)
        LLM_TOKENS.labels(model=model, token_type="input").inc(input_tokens)
        LLM_TOKENS.labels(model=model, token_type="output").inc(output_tokens)
        LLM_COST.labels(model=model).inc(cost_usd)
