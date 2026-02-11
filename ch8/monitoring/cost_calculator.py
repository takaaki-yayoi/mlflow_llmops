"""
コスト計算ユーティリティ

トークン使用量からAPI費用を計算します。
料金は頻繁に更新されるため、最新の公式料金を確認してください。

参考:
  - OpenAI: https://openai.com/api/pricing/
  - Anthropic: https://www.anthropic.com/pricing
"""

# 料金テーブル (per 1K tokens, USD)
# 最終更新: 2025年1月
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku-20241022": {"input": 0.0008, "output": 0.004},
}

# モデル名のエイリアスマッピング
MODEL_ALIASES: dict[str, str] = {
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-08-06": "gpt-4o",
    "claude-3-5-sonnet": "claude-sonnet-4-20250514",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3.5-sonnet": "claude-sonnet-4-20250514",
    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
}


def resolve_model_name(model: str) -> str:
    """モデル名のエイリアスを解決"""
    return MODEL_ALIASES.get(model, model)


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """
    トークン使用量からAPIコスト(USD)を計算。

    Args:
        model: モデル名
        input_tokens: 入力トークン数
        output_tokens: 出力トークン数
        cached_input_tokens: キャッシュされた入力トークン数(通常、入力料金の50%)

    Returns:
        コスト(USD)
    """
    resolved = resolve_model_name(model)
    pricing = MODEL_PRICING.get(resolved)
    if pricing is None:
        return 0.0

    # キャッシュ入力トークンは通常50%割引
    regular_input = input_tokens - cached_input_tokens
    cached_cost = (cached_input_tokens / 1000) * pricing["input"] * 0.5
    regular_cost = (regular_input / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]

    return regular_cost + cached_cost + output_cost


def format_cost_report(
    model: str, input_tokens: int, output_tokens: int, cost: float
) -> str:
    """コストレポートの文字列を生成"""
    return (
        f"Model: {model}\n"
        f"  Input:  {input_tokens:>8,} tokens\n"
        f"  Output: {output_tokens:>8,} tokens\n"
        f"  Cost:   ${cost:.6f}"
    )
