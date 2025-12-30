"""
第8章 8.4.3: アラート通知ハンドラー

Slack、PagerDuty、Emailなどへの通知実装例です。
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import urllib.request
import urllib.parse


class AlertSeverity(Enum):
    """アラートの重要度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """アラート情報"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = None
    dashboard_url: Optional[str] = None
    additional_info: Optional[dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class NotificationHandler(ABC):
    """通知ハンドラーの基底クラス"""
    
    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        アラートを送信
        
        Args:
            alert: アラート情報
        
        Returns:
            送信成功: True, 失敗: False
        """
        pass


class SlackNotificationHandler(NotificationHandler):
    """
    Slack通知ハンドラー
    
    使用方法:
        handler = SlackNotificationHandler(webhook_url="https://hooks.slack.com/...")
        handler.send(alert)
    """
    
    # 重要度別の色
    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#36a64f",      # 緑
        AlertSeverity.WARNING: "#ff9800",   # オレンジ
        AlertSeverity.CRITICAL: "#ff0000",  # 赤
    }
    
    def __init__(self, webhook_url: str, channel: str = None):
        """
        Args:
            webhook_url: Slack Webhook URL
            channel: チャンネル名 (オプション、Webhook設定が優先)
        """
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, alert: Alert) -> bool:
        payload = self._build_payload(alert)
        
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            print(f"Slack notification failed: {e}")
            return False
    
    def _build_payload(self, alert: Alert) -> dict:
        """Slackペイロードを構築"""
        color = self.SEVERITY_COLORS.get(alert.severity, "#808080")
        
        fields = [
            {
                "title": "Metric",
                "value": alert.metric_name,
                "short": True,
            },
            {
                "title": "Value",
                "value": f"{alert.metric_value:.4f}",
                "short": True,
            },
            {
                "title": "Threshold",
                "value": f"{alert.threshold:.4f}",
                "short": True,
            },
            {
                "title": "Severity",
                "value": alert.severity.value.upper(),
                "short": True,
            },
        ]
        
        payload = {
            "attachments": [
                {
                    "color": color,
                    "title": f"🚨 {alert.title}",
                    "text": alert.message,
                    "fields": fields,
                    "footer": f"Alert ID: {alert.id}",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ]
        }
        
        if alert.dashboard_url:
            payload["attachments"][0]["actions"] = [
                {
                    "type": "button",
                    "text": "View Dashboard",
                    "url": alert.dashboard_url,
                }
            ]
        
        if self.channel:
            payload["channel"] = self.channel
        
        return payload


class PagerDutyNotificationHandler(NotificationHandler):
    """
    PagerDuty通知ハンドラー
    
    Criticalアラートでオンコール担当者に即座に通知します。
    """
    
    EVENTS_API_URL = "https://events.pagerduty.com/v2/enqueue"
    
    def __init__(self, routing_key: str, service_name: str = "LLM App"):
        """
        Args:
            routing_key: PagerDuty Integration Key
            service_name: サービス名
        """
        self.routing_key = routing_key
        self.service_name = service_name
    
    def send(self, alert: Alert) -> bool:
        # Criticalアラートのみトリガー
        if alert.severity != AlertSeverity.CRITICAL:
            return self._send_change_event(alert)
        
        return self._send_trigger_event(alert)
    
    def _send_trigger_event(self, alert: Alert) -> bool:
        """インシデントをトリガー"""
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": alert.id,  # 重複排除キー
            "payload": {
                "summary": f"[{self.service_name}] {alert.title}",
                "source": self.service_name,
                "severity": "critical",
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": {
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                },
            },
        }
        
        if alert.dashboard_url:
            payload["links"] = [{"href": alert.dashboard_url, "text": "Dashboard"}]
        
        return self._send_event(payload)
    
    def _send_change_event(self, alert: Alert) -> bool:
        """変更イベントを送信 (非Critical)"""
        payload = {
            "routing_key": self.routing_key,
            "event_action": "change",
            "payload": {
                "summary": f"[{self.service_name}] {alert.title}",
                "source": self.service_name,
                "timestamp": alert.timestamp.isoformat(),
                "custom_details": {
                    "severity": alert.severity.value,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                },
            },
        }
        return self._send_event(payload)
    
    def _send_event(self, payload: dict) -> bool:
        """PagerDuty APIにイベントを送信"""
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.EVENTS_API_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status in (200, 201, 202)
        except Exception as e:
            print(f"PagerDuty notification failed: {e}")
            return False
    
    def resolve(self, alert_id: str) -> bool:
        """インシデントを解決"""
        payload = {
            "routing_key": self.routing_key,
            "event_action": "resolve",
            "dedup_key": alert_id,
        }
        return self._send_event(payload)


class MultiChannelNotifier:
    """
    複数チャネルへの通知を管理
    
    使用例:
        notifier = MultiChannelNotifier()
        notifier.add_handler("slack", SlackNotificationHandler(...))
        notifier.add_handler("pagerduty", PagerDutyNotificationHandler(...))
        
        # 重要度に応じた通知先を設定
        notifier.set_severity_routing({
            AlertSeverity.CRITICAL: ["pagerduty", "slack"],
            AlertSeverity.WARNING: ["slack"],
            AlertSeverity.INFO: ["slack"],
        })
        
        notifier.notify(alert)
    """
    
    def __init__(self):
        self.handlers: dict[str, NotificationHandler] = {}
        self.severity_routing: dict[AlertSeverity, list[str]] = {
            AlertSeverity.CRITICAL: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.INFO: [],
        }
    
    def add_handler(self, name: str, handler: NotificationHandler) -> None:
        """通知ハンドラーを追加"""
        self.handlers[name] = handler
    
    def set_severity_routing(self, routing: dict[AlertSeverity, list[str]]) -> None:
        """重要度別の通知先を設定"""
        self.severity_routing = routing
    
    def notify(self, alert: Alert) -> dict[str, bool]:
        """
        アラートを送信
        
        Returns:
            ハンドラー名 → 成功/失敗 のマッピング
        """
        results = {}
        
        # 重要度に応じた通知先を取得
        handler_names = self.severity_routing.get(alert.severity, [])
        
        for name in handler_names:
            handler = self.handlers.get(name)
            if handler:
                results[name] = handler.send(alert)
            else:
                print(f"Warning: Handler '{name}' not found")
                results[name] = False
        
        return results


# 使用例
if __name__ == "__main__":
    # アラートの作成
    alert = Alert(
        id="alert-001",
        title="High Error Rate Detected",
        message="Error rate exceeded threshold for the last 5 minutes",
        severity=AlertSeverity.CRITICAL,
        metric_name="error_rate",
        metric_value=0.08,
        threshold=0.05,
        dashboard_url="https://your-dashboard.com/llm-monitoring",
    )
    
    print(f"Alert created: {alert.title}")
    print(f"  Severity: {alert.severity.value}")
    print(f"  Metric: {alert.metric_name} = {alert.metric_value} (threshold: {alert.threshold})")
    
    # 実際の使用時はWebhook URLを設定
    # slack_handler = SlackNotificationHandler(
    #     webhook_url=os.environ.get("SLACK_WEBHOOK_URL")
    # )
    # slack_handler.send(alert)
