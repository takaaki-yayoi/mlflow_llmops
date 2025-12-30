"""
第8章 8.2.3: コスト計算の実装

MLflowはトークン使用量の追跡を提供しますが、
コスト計算はユーザーが実装する必要があります。

注意: 料金は2024年12月時点の参考値です。
最新の料金は各プロバイダーの公式サイトで確認してください。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import mlflow


@dataclass
class ModelPricing:
    """モデルの料金情報"""
    input_price_per_1k: float  # 入力1000トークンあたりの価格 (USD)
    output_price_per_1k: float  # 出力1000トークンあたりの価格 (USD)
    cached_input_price_per_1k: Optional[float] = None  # キャッシュ入力の価格


# デフォルトの料金設定 (2024年12月時点の参考値)
DEFAULT_PRICING = {
    # OpenAI Models
    "gpt-4o": ModelPricing(input_price_per_1k=0.0025, output_price_per_1k=0.01),
    "gpt-4o-mini": ModelPricing(input_price_per_1k=0.00015, output_price_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(input_price_per_1k=0.01, output_price_per_1k=0.03),
    "gpt-3.5-turbo": ModelPricing(input_price_per_1k=0.0005, output_price_per_1k=0.0015),
    "o1-preview": ModelPricing(input_price_per_1k=0.015, output_price_per_1k=0.06),
    "o1-mini": ModelPricing(input_price_per_1k=0.003, output_price_per_1k=0.012),
    
    # Anthropic Models
    "claude-3-5-sonnet": ModelPricing(input_price_per_1k=0.003, output_price_per_1k=0.015),
    "claude-3-5-haiku": ModelPricing(input_price_per_1k=0.0008, output_price_per_1k=0.004),
    "claude-3-opus": ModelPricing(input_price_per_1k=0.015, output_price_per_1k=0.075),
    
    # エイリアス (モデル名の正規化)
    "claude-3-5-sonnet-20241022": ModelPricing(input_price_per_1k=0.003, output_price_per_1k=0.015),
    "claude-3-5-haiku-20241022": ModelPricing(input_price_per_1k=0.0008, output_price_per_1k=0.004),
}


class CostCalculator:
    """
    LLM APIコストの計算クラス
    
    使用例:
        calculator = CostCalculator()
        cost = calculator.calculate("gpt-4o", input_tokens=1000, output_tokens=500)
        print(f"Cost: ${cost:.6f}")
    """
    
    def __init__(self, pricing: dict[str, ModelPricing] = None, pricing_file: str = None):
        """
        Args:
            pricing: モデル名 → 料金情報のマッピング
            pricing_file: 料金設定JSONファイルのパス
        """
        self.pricing = pricing or DEFAULT_PRICING.copy()
        
        if pricing_file:
            self._load_pricing_file(pricing_file)
    
    def _load_pricing_file(self, filepath: str) -> None:
        """外部ファイルから料金設定を読み込む"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pricing file not found: {filepath}")
        
        with open(path) as f:
            data = json.load(f)
        
        for model_name, prices in data.items():
            self.pricing[model_name] = ModelPricing(
                input_price_per_1k=prices["input"],
                output_price_per_1k=prices["output"],
                cached_input_price_per_1k=prices.get("cached_input"),
            )
    
    def _normalize_model_name(self, model: str) -> str:
        """
        モデル名を正規化
        
        例: "gpt-4o-2024-08-06" → "gpt-4o"
        """
        # 完全一致を優先
        if model in self.pricing:
            return model
        
        # プレフィックスマッチング
        for known_model in self.pricing.keys():
            if model.startswith(known_model):
                return known_model
        
        # バージョンサフィックスを除去
        parts = model.rsplit("-", 1)
        if len(parts) > 1 and parts[1].replace("-", "").isdigit():
            base_model = parts[0]
            if base_model in self.pricing:
                return base_model
        
        return model
    
    def calculate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """
        コストを計算
        
        Args:
            model: モデル名
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
            cached_input_tokens: キャッシュされた入力トークン数
        
        Returns:
            コスト (USD)
        """
        normalized_model = self._normalize_model_name(model)
        
        if normalized_model not in self.pricing:
            print(f"Warning: Unknown model '{model}' (normalized: '{normalized_model}')")
            return 0.0
        
        pricing = self.pricing[normalized_model]
        
        # 通常の入力トークンコスト
        input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
        
        # キャッシュ入力トークンコスト
        if cached_input_tokens > 0 and pricing.cached_input_price_per_1k is not None:
            cached_cost = (cached_input_tokens / 1000) * pricing.cached_input_price_per_1k
        else:
            cached_cost = 0.0
        
        # 出力トークンコスト
        output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
        
        return input_cost + cached_cost + output_cost
    
    def calculate_from_trace(self, trace_id: str, model: str = None) -> dict:
        """
        トレースからコストを計算
        
        Args:
            trace_id: トレースID
            model: モデル名 (指定しない場合はトレースから取得を試みる)
        
        Returns:
            コスト情報の辞書
        """
        trace = mlflow.get_trace(trace_id=trace_id)
        token_usage = trace.info.token_usage or {}
        
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        
        if model is None:
            # トレースからモデル名を取得する試み
            # (実装はアプリケーション依存)
            model = "unknown"
        
        cost = self.calculate(model, input_tokens, output_tokens)
        
        return {
            "trace_id": trace_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd": cost,
        }
    
    def add_cost_tags_to_trace(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        現在のトレースにコストタグを追加
        
        Args:
            model: モデル名
            input_tokens: 入力トークン数
            output_tokens: 出力トークン数
        
        Returns:
            計算されたコスト (USD)
        """
        cost = self.calculate(model, input_tokens, output_tokens)
        
        mlflow.update_current_trace(tags={
            "cost.model": model,
            "cost.input_tokens": str(input_tokens),
            "cost.output_tokens": str(output_tokens),
            "cost.total_usd": f"{cost:.8f}",
        })
        
        return cost


class CostTracker:
    """
    コストの累積追跡クラス
    
    使用例:
        tracker = CostTracker()
        
        # API呼び出しごとにコストを記録
        tracker.record("gpt-4o", input_tokens=1000, output_tokens=500)
        
        # 統計を取得
        print(tracker.get_summary())
    """
    
    def __init__(self, calculator: CostCalculator = None):
        self.calculator = calculator or CostCalculator()
        self._records: list[dict] = []
    
    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict = None,
    ) -> float:
        """
        コストを記録
        
        Returns:
            計算されたコスト (USD)
        """
        cost = self.calculator.calculate(model, input_tokens, output_tokens)
        
        record = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            **(metadata or {}),
        }
        self._records.append(record)
        
        return cost
    
    def get_summary(self) -> dict:
        """コストサマリーを取得"""
        if not self._records:
            return {
                "total_cost_usd": 0,
                "total_requests": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "by_model": {},
            }
        
        total_cost = sum(r["cost_usd"] for r in self._records)
        total_input = sum(r["input_tokens"] for r in self._records)
        total_output = sum(r["output_tokens"] for r in self._records)
        
        # モデル別集計
        by_model = {}
        for record in self._records:
            model = record["model"]
            if model not in by_model:
                by_model[model] = {
                    "requests": 0,
                    "cost_usd": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            by_model[model]["requests"] += 1
            by_model[model]["cost_usd"] += record["cost_usd"]
            by_model[model]["input_tokens"] += record["input_tokens"]
            by_model[model]["output_tokens"] += record["output_tokens"]
        
        return {
            "total_cost_usd": total_cost,
            "total_requests": len(self._records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_cost_per_request": total_cost / len(self._records),
            "by_model": by_model,
        }
    
    def reset(self) -> None:
        """記録をリセット"""
        self._records = []


# 使用例
if __name__ == "__main__":
    # 基本的な使用
    calculator = CostCalculator()
    
    # 単純なコスト計算
    cost = calculator.calculate("gpt-4o", input_tokens=1000, output_tokens=500)
    print(f"GPT-4o cost: ${cost:.6f}")
    
    cost = calculator.calculate("claude-3-5-sonnet", input_tokens=1000, output_tokens=500)
    print(f"Claude 3.5 Sonnet cost: ${cost:.6f}")
    
    # モデル名の正規化
    cost = calculator.calculate("gpt-4o-2024-08-06", input_tokens=1000, output_tokens=500)
    print(f"GPT-4o (dated) cost: ${cost:.6f}")
    
    # コストトラッカーの使用
    print("\n=== Cost Tracker ===")
    tracker = CostTracker()
    
    # 複数のAPI呼び出しをシミュレート
    tracker.record("gpt-4o", input_tokens=500, output_tokens=200)
    tracker.record("gpt-4o", input_tokens=800, output_tokens=400)
    tracker.record("gpt-4o-mini", input_tokens=1000, output_tokens=500)
    tracker.record("claude-3-5-sonnet", input_tokens=600, output_tokens=300)
    
    summary = tracker.get_summary()
    print(f"Total cost: ${summary['total_cost_usd']:.6f}")
    print(f"Total requests: {summary['total_requests']}")
    print(f"Avg cost/request: ${summary['avg_cost_per_request']:.6f}")
    print(f"By model: {json.dumps(summary['by_model'], indent=2)}")
