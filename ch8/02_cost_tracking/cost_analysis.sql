-- 第8章 8.2.4: コスト分析SQLクエリ
-- 
-- 注意: 以下のクエリは、カスタムコスト計算ロジックによって
-- トレースにタグ(cost.total_usd, cost.model等)が追加されている前提です。
-- MLflowが自動的にコストを計算するわけではありません。

-- ============================================================
-- テーブル名: my_catalog.my_schema.archived_traces
-- (enable_databricks_trace_archival() で設定したテーブル)
-- ============================================================

-- ------------------------------------------------------------
-- 1. 日次コストサマリー
-- ------------------------------------------------------------
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request,
    SUM(CAST(tags['cost.input_tokens'] AS INT)) as total_input_tokens,
    SUM(CAST(tags['cost.output_tokens'] AS INT)) as total_output_tokens
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
  AND tags['cost.total_usd'] IS NOT NULL
GROUP BY DATE(timestamp)
ORDER BY date DESC;


-- ------------------------------------------------------------
-- 2. モデル別コスト内訳
-- ------------------------------------------------------------
SELECT 
    tags['cost.model'] as model,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request,
    ROUND(
        SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) * 100.0 / 
        SUM(SUM(CAST(tags['cost.total_usd'] AS DOUBLE))) OVER (), 
        2
    ) as cost_percentage
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND tags['cost.model'] IS NOT NULL
GROUP BY tags['cost.model']
ORDER BY total_cost_usd DESC;


-- ------------------------------------------------------------
-- 3. ユーザー別コストTop 10
-- ------------------------------------------------------------
SELECT 
    tags['mlflow.trace.user'] as user_id,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request,
    SUM(CAST(tags['cost.input_tokens'] AS INT)) as total_input_tokens,
    SUM(CAST(tags['cost.output_tokens'] AS INT)) as total_output_tokens
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND tags['mlflow.trace.user'] IS NOT NULL
  AND tags['cost.total_usd'] IS NOT NULL
GROUP BY tags['mlflow.trace.user']
ORDER BY total_cost_usd DESC
LIMIT 10;


-- ------------------------------------------------------------
-- 4. 時間帯別コスト分布
-- ------------------------------------------------------------
SELECT 
    HOUR(timestamp) as hour_of_day,
    COUNT(*) as request_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost_usd,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND tags['cost.total_usd'] IS NOT NULL
GROUP BY HOUR(timestamp)
ORDER BY hour_of_day;


-- ------------------------------------------------------------
-- 5. コスト異常検出 (過去平均の2倍を超えるリクエスト)
-- ------------------------------------------------------------
WITH avg_cost AS (
    SELECT AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost
    FROM my_catalog.my_schema.archived_traces
    WHERE timestamp >= CURRENT_DATE - INTERVAL 30 DAYS
      AND tags['cost.total_usd'] IS NOT NULL
)
SELECT 
    t.trace_id,
    t.timestamp,
    tags['mlflow.trace.user'] as user_id,
    tags['cost.model'] as model,
    CAST(tags['cost.total_usd'] AS DOUBLE) as cost_usd,
    a.avg_cost as avg_cost_30d,
    CAST(tags['cost.total_usd'] AS DOUBLE) / a.avg_cost as cost_ratio
FROM my_catalog.my_schema.archived_traces t
CROSS JOIN avg_cost a
WHERE t.timestamp >= CURRENT_DATE - INTERVAL 1 DAY
  AND tags['cost.total_usd'] IS NOT NULL
  AND CAST(tags['cost.total_usd'] AS DOUBLE) > a.avg_cost * 2
ORDER BY cost_usd DESC
LIMIT 50;


-- ------------------------------------------------------------
-- 6. 週次コスト推移 (前週比)
-- ------------------------------------------------------------
WITH weekly_costs AS (
    SELECT 
        DATE_TRUNC('week', timestamp) as week_start,
        SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_cost
    FROM my_catalog.my_schema.archived_traces
    WHERE timestamp >= CURRENT_DATE - INTERVAL 12 WEEKS
      AND tags['cost.total_usd'] IS NOT NULL
    GROUP BY DATE_TRUNC('week', timestamp)
)
SELECT 
    week_start,
    total_cost,
    LAG(total_cost) OVER (ORDER BY week_start) as prev_week_cost,
    ROUND(
        (total_cost - LAG(total_cost) OVER (ORDER BY week_start)) / 
        NULLIF(LAG(total_cost) OVER (ORDER BY week_start), 0) * 100,
        2
    ) as week_over_week_change_pct
FROM weekly_costs
ORDER BY week_start DESC;


-- ------------------------------------------------------------
-- 7. セッション別コスト (マルチターン会話の総コスト)
-- ------------------------------------------------------------
SELECT 
    tags['mlflow.trace.session'] as session_id,
    tags['mlflow.trace.user'] as user_id,
    COUNT(*) as turn_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as total_session_cost,
    MIN(timestamp) as session_start,
    MAX(timestamp) as session_end,
    TIMESTAMPDIFF(MINUTE, MIN(timestamp), MAX(timestamp)) as session_duration_min
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 1 DAY
  AND tags['mlflow.trace.session'] IS NOT NULL
  AND tags['cost.total_usd'] IS NOT NULL
GROUP BY tags['mlflow.trace.session'], tags['mlflow.trace.user']
HAVING COUNT(*) > 1
ORDER BY total_session_cost DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 8. エラー時のコスト損失
-- ------------------------------------------------------------
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as error_count,
    SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as wasted_cost_usd,
    tags['cost.model'] as model
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND status = 'ERROR'
  AND tags['cost.total_usd'] IS NOT NULL
GROUP BY DATE(timestamp), tags['cost.model']
ORDER BY date DESC, wasted_cost_usd DESC;


-- ------------------------------------------------------------
-- 9. 予算消化率 (月間予算: $1000と仮定)
-- ------------------------------------------------------------
WITH monthly_budget AS (
    SELECT 1000.0 as budget_usd  -- 月間予算を設定
),
current_month_cost AS (
    SELECT SUM(CAST(tags['cost.total_usd'] AS DOUBLE)) as cost_usd
    FROM my_catalog.my_schema.archived_traces
    WHERE timestamp >= DATE_TRUNC('month', CURRENT_DATE)
      AND tags['cost.total_usd'] IS NOT NULL
)
SELECT 
    b.budget_usd as monthly_budget,
    c.cost_usd as current_month_spent,
    b.budget_usd - c.cost_usd as remaining_budget,
    ROUND(c.cost_usd / b.budget_usd * 100, 2) as budget_used_pct,
    DAY(CURRENT_DATE) as days_elapsed,
    DAY(LAST_DAY(CURRENT_DATE)) as days_in_month,
    ROUND(c.cost_usd / DAY(CURRENT_DATE), 2) as avg_daily_cost,
    ROUND(c.cost_usd / DAY(CURRENT_DATE) * DAY(LAST_DAY(CURRENT_DATE)), 2) as projected_month_cost
FROM monthly_budget b
CROSS JOIN current_month_cost c;


-- ------------------------------------------------------------
-- 10. トークン効率分析 (入力/出力比率)
-- ------------------------------------------------------------
SELECT 
    tags['cost.model'] as model,
    COUNT(*) as request_count,
    AVG(CAST(tags['cost.input_tokens'] AS INT)) as avg_input_tokens,
    AVG(CAST(tags['cost.output_tokens'] AS INT)) as avg_output_tokens,
    ROUND(
        AVG(CAST(tags['cost.output_tokens'] AS DOUBLE)) / 
        NULLIF(AVG(CAST(tags['cost.input_tokens'] AS DOUBLE)), 0),
        3
    ) as output_input_ratio,
    AVG(CAST(tags['cost.total_usd'] AS DOUBLE)) as avg_cost_per_request
FROM my_catalog.my_schema.archived_traces
WHERE timestamp >= CURRENT_DATE - INTERVAL 7 DAYS
  AND tags['cost.model'] IS NOT NULL
  AND tags['cost.input_tokens'] IS NOT NULL
  AND tags['cost.output_tokens'] IS NOT NULL
GROUP BY tags['cost.model']
ORDER BY avg_cost_per_request DESC;
