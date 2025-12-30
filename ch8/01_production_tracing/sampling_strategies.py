"""
第8章 8.1.6: サンプリング戦略

本番環境でのトレースサンプリング戦略の実装例です。
確率的サンプリング、条件付きサンプリング、適応型サンプリングを提供します。
"""

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import deque
import threading


class SamplingStrategy(ABC):
    """サンプリング戦略の基底クラス"""
    
    @abstractmethod
    def should_sample(self, context: dict) -> bool:
        """
        トレースをサンプリングするかどうかを判定
        
        Args:
            context: リクエストのコンテキスト情報
        
        Returns:
            True: サンプリングする, False: サンプリングしない
        """
        pass


class ProbabilisticSampler(SamplingStrategy):
    """
    確率的サンプリング
    
    最もシンプルなサンプリング方式で、一定の確率でトレースを記録します。
    """
    
    def __init__(self, sample_rate: float = 0.1):
        """
        Args:
            sample_rate: サンプリング率 (0.0 - 1.0)
        """
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        self.sample_rate = sample_rate
    
    def should_sample(self, context: dict = None) -> bool:
        return random.random() < self.sample_rate


@dataclass
class ConditionalSamplerConfig:
    """条件付きサンプラーの設定"""
    # 常にサンプリングする条件
    always_sample_errors: bool = True
    always_sample_vip_users: bool = True
    vip_user_ids: set = field(default_factory=set)
    
    # レイテンシベースのサンプリング
    latency_threshold_ms: float = 5000  # 5秒
    always_sample_slow_requests: bool = True
    
    # デフォルトのサンプリング率
    default_sample_rate: float = 0.1


class ConditionalSampler(SamplingStrategy):
    """
    条件付きサンプリング
    
    特定の条件に基づいてサンプリングを決定します。
    重要なリクエストは必ずサンプリングし、それ以外は確率的にサンプリングします。
    """
    
    def __init__(self, config: ConditionalSamplerConfig = None):
        self.config = config or ConditionalSamplerConfig()
        self._probabilistic = ProbabilisticSampler(self.config.default_sample_rate)
    
    def should_sample(self, context: dict) -> bool:
        """
        条件に基づいてサンプリングを判定
        
        Args:
            context: {
                "is_error": bool,
                "user_id": str,
                "latency_ms": float,  # 事後判定の場合
            }
        """
        # エラーは常にサンプリング
        if self.config.always_sample_errors and context.get("is_error", False):
            return True
        
        # VIPユーザーは常にサンプリング
        if self.config.always_sample_vip_users:
            user_id = context.get("user_id", "")
            if user_id in self.config.vip_user_ids:
                return True
        
        # 高レイテンシは常にサンプリング (事後判定)
        if self.config.always_sample_slow_requests:
            latency = context.get("latency_ms", 0)
            if latency > self.config.latency_threshold_ms:
                return True
        
        # それ以外は確率的サンプリング
        return self._probabilistic.should_sample(context)


class AdaptiveSampler(SamplingStrategy):
    """
    適応型サンプリング
    
    システムの負荷に応じてサンプリング率を動的に調整します。
    目標トレース数/分を維持するようにサンプリング率を自動調整します。
    """
    
    def __init__(
        self,
        target_traces_per_minute: int = 100,
        min_sample_rate: float = 0.01,
        max_sample_rate: float = 1.0,
        adjustment_interval_seconds: float = 60.0,
    ):
        """
        Args:
            target_traces_per_minute: 目標トレース数/分
            min_sample_rate: 最小サンプリング率
            max_sample_rate: 最大サンプリング率
            adjustment_interval_seconds: 調整間隔(秒)
        """
        self.target_traces_per_minute = target_traces_per_minute
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.adjustment_interval = adjustment_interval_seconds
        
        self._current_rate = 0.5  # 初期サンプリング率
        self._request_timestamps: deque = deque()
        self._sampled_timestamps: deque = deque()
        self._last_adjustment = time.time()
        self._lock = threading.Lock()
    
    @property
    def current_sample_rate(self) -> float:
        return self._current_rate
    
    def should_sample(self, context: dict = None) -> bool:
        current_time = time.time()
        
        with self._lock:
            # リクエストを記録
            self._request_timestamps.append(current_time)
            
            # 古いタイムスタンプを削除
            cutoff = current_time - 60  # 過去1分間
            while self._request_timestamps and self._request_timestamps[0] < cutoff:
                self._request_timestamps.popleft()
            while self._sampled_timestamps and self._sampled_timestamps[0] < cutoff:
                self._sampled_timestamps.popleft()
            
            # サンプリング判定
            should_sample = random.random() < self._current_rate
            
            if should_sample:
                self._sampled_timestamps.append(current_time)
            
            # 定期的にサンプリング率を調整
            if current_time - self._last_adjustment >= self.adjustment_interval:
                self._adjust_sample_rate()
                self._last_adjustment = current_time
        
        return should_sample
    
    def _adjust_sample_rate(self) -> None:
        """サンプリング率を調整"""
        request_count = len(self._request_timestamps)
        sampled_count = len(self._sampled_timestamps)
        
        if request_count == 0:
            return
        
        # 現在のトレース数/分
        current_traces_per_minute = sampled_count
        
        # 目標との差分に基づいて調整
        if current_traces_per_minute < self.target_traces_per_minute:
            # トレース数が少ない → サンプリング率を上げる
            adjustment_factor = min(1.5, self.target_traces_per_minute / max(current_traces_per_minute, 1))
            self._current_rate = min(self.max_sample_rate, self._current_rate * adjustment_factor)
        elif current_traces_per_minute > self.target_traces_per_minute * 1.2:
            # トレース数が多い → サンプリング率を下げる
            adjustment_factor = self.target_traces_per_minute / current_traces_per_minute
            self._current_rate = max(self.min_sample_rate, self._current_rate * adjustment_factor)
    
    def get_stats(self) -> dict:
        """現在の統計情報を取得"""
        with self._lock:
            return {
                "current_sample_rate": self._current_rate,
                "requests_per_minute": len(self._request_timestamps),
                "traces_per_minute": len(self._sampled_timestamps),
                "target_traces_per_minute": self.target_traces_per_minute,
            }


class SamplingManager:
    """
    サンプリング管理クラス
    
    複数のサンプリング戦略を組み合わせて使用できます。
    """
    
    def __init__(self, strategy: SamplingStrategy):
        self.strategy = strategy
        self._sampled_count = 0
        self._total_count = 0
    
    def should_trace(self, context: dict = None) -> bool:
        """
        トレースすべきかどうかを判定
        
        Args:
            context: リクエストのコンテキスト情報
        
        Returns:
            True: トレースする, False: トレースしない
        """
        self._total_count += 1
        should_sample = self.strategy.should_sample(context or {})
        
        if should_sample:
            self._sampled_count += 1
        
        return should_sample
    
    def get_stats(self) -> dict:
        """サンプリング統計を取得"""
        actual_rate = self._sampled_count / self._total_count if self._total_count > 0 else 0
        return {
            "total_requests": self._total_count,
            "sampled_requests": self._sampled_count,
            "actual_sample_rate": actual_rate,
        }


# 使用例
if __name__ == "__main__":
    # 1. 確率的サンプリング
    print("=== Probabilistic Sampling ===")
    prob_sampler = SamplingManager(ProbabilisticSampler(sample_rate=0.1))
    
    for _ in range(1000):
        prob_sampler.should_trace({})
    
    print(f"Stats: {prob_sampler.get_stats()}")
    
    # 2. 条件付きサンプリング
    print("\n=== Conditional Sampling ===")
    config = ConditionalSamplerConfig(
        vip_user_ids={"vip-user-1", "vip-user-2"},
        latency_threshold_ms=5000,
        default_sample_rate=0.1,
    )
    cond_sampler = SamplingManager(ConditionalSampler(config))
    
    # エラーは必ずサンプリング
    assert cond_sampler.should_trace({"is_error": True}) == True
    
    # VIPユーザーは必ずサンプリング
    assert cond_sampler.should_trace({"user_id": "vip-user-1"}) == True
    
    # 高レイテンシは必ずサンプリング
    assert cond_sampler.should_trace({"latency_ms": 6000}) == True
    
    print(f"Stats: {cond_sampler.get_stats()}")
    
    # 3. 適応型サンプリング
    print("\n=== Adaptive Sampling ===")
    adaptive_sampler = AdaptiveSampler(
        target_traces_per_minute=50,
        min_sample_rate=0.05,
        max_sample_rate=1.0,
    )
    manager = SamplingManager(adaptive_sampler)
    
    # シミュレーション
    for i in range(500):
        manager.should_trace({})
        if i % 100 == 0:
            stats = adaptive_sampler.get_stats()
            print(f"  Iteration {i}: rate={stats['current_sample_rate']:.3f}")
    
    print(f"Final stats: {manager.get_stats()}")
    print(f"Adaptive stats: {adaptive_sampler.get_stats()}")
