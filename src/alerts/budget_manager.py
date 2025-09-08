"""
Budget Management & Alerting System - Smart cost monitoring and notifications
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

import pandas as pd
from slack_sdk.webhook import WebhookClient
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"


@dataclass
class BudgetAlert:
    """Budget alert data structure"""
    id: str
    budget_id: str
    severity: AlertSeverity
    message: str
    current_spend: float
    budget_limit: float
    percentage_used: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


@dataclass
class Budget:
    """Budget configuration"""
    id: str
    name: str
    monthly_limit: float
    alert_thresholds: List[float]  # [80, 90, 100] for 80%, 90%, 100%
    notification_channels: List[NotificationChannel]
    services: List[str] = None  # Specific services to monitor
    tags: Dict[str, str] = None
    created_at: datetime = None
    updated_at: datetime = None


class BudgetManager:
    """
    Advanced budget management with intelligent alerting
    """
    
    def __init__(self, config_file: str = "budget_config.json"):
        self.config_file = config_file
        self.budgets: Dict[str, Budget] = {}
        self.alert_history: List[BudgetAlert] = []
        self.notification_config = {}
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load budget configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Load budgets
            for budget_data in config.get('budgets', []):
                budget = Budget(
                    id=budget_data['id'],
                    name=budget_data['name'],
                    monthly_limit=budget_data['monthly_limit'],
                    alert_thresholds=budget_data['alert_thresholds'],
                    notification_channels=[
                        NotificationChannel(ch) for ch in budget_data['notification_channels']
                    ],
                    services=budget_data.get('services'),
                    tags=budget_data.get('tags', {}),
                    created_at=datetime.fromisoformat(budget_data.get('created_at', datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(budget_data.get('updated_at', datetime.now().isoformat()))
                )
                self.budgets[budget.id] = budget
            
            # Load notification configuration
            self.notification_config = config.get('notifications', {})
            
            logger.info(f"Loaded {len(self.budgets)} budgets")
            
        except FileNotFoundError:
            logger.info("No budget configuration found, creating default")
            self._create_default_config()
        except Exception as e:
            logger.error(f"Failed to load budget configuration: {e}")
            raise
    
    def _create_default_config(self):
        """Create default budget configuration"""
        default_budget = Budget(
            id="default",
            name="Default Budget",
            monthly_limit=10000.0,
            alert_thresholds=[80.0, 90.0, 100.0],
            notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.budgets["default"] = default_budget
        self._save_config()
    
    def _save_config(self):
        """Save budget configuration to file"""
        config = {
            'budgets': [
                {
                    'id': budget.id,
                    'name': budget.name,
                    'monthly_limit': budget.monthly_limit,
                    'alert_thresholds': budget.alert_thresholds,
                    'notification_channels': [ch.value for ch in budget.notification_channels],
                    'services': budget.services,
                    'tags': budget.tags,
                    'created_at': budget.created_at.isoformat(),
                    'updated_at': budget.updated_at.isoformat()
                }
                for budget in self.budgets.values()
            ],
            'notifications': self.notification_config
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def set_budget(
        self,
        budget_id: str,
        name: str,
        monthly_limit: float,
        alert_thresholds: List[float] = None,
        notification_channels: List[NotificationChannel] = None,
        services: List[str] = None,
        tags: Dict[str, str] = None
    ) -> Budget:
        """Set or update a budget"""
        if alert_thresholds is None:
            alert_thresholds = [80.0, 90.0, 100.0]
        
        if notification_channels is None:
            notification_channels = [NotificationChannel.EMAIL]
        
        budget = Budget(
            id=budget_id,
            name=name,
            monthly_limit=monthly_limit,
            alert_thresholds=alert_thresholds,
            notification_channels=notification_channels,
            services=services,
            tags=tags,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.budgets[budget_id] = budget
        self._save_config()
        
        logger.info(f"Budget '{name}' set with limit ${monthly_limit}")
        return budget
    
    async def check_budget(self, budget_id: str = None) -> List[BudgetAlert]:
        """Check budget status and generate alerts"""
        alerts = []
        
        budgets_to_check = [self.budgets[budget_id]] if budget_id else self.budgets.values()
        
        for budget in budgets_to_check:
            try:
                # Get current spending
                current_spend = await self._get_current_spending(budget)
                percentage_used = (current_spend / budget.monthly_limit) * 100
                
                # Check for threshold breaches
                for threshold in budget.alert_thresholds:
                    if percentage_used >= threshold:
                        # Check if we already sent an alert for this threshold
                        if not self._alert_already_sent(budget.id, threshold, current_spend):
                            alert = await self._create_alert(
                                budget, threshold, current_spend, percentage_used
                            )
                            alerts.append(alert)
                            
                            # Send notifications
                            await self._send_notifications(alert)
                            
                            # Store alert
                            self.alert_history.append(alert)
            
            except Exception as e:
                logger.error(f"Failed to check budget {budget.id}: {e}")
        
        return alerts
    
    async def _get_current_spending(self, budget: Budget) -> float:
        """Get current spending for a budget"""
        # This would integrate with your cost data collection
        # For now, return a mock value
        from .aws_collector import AWSCostCollector
        
        collector = AWSCostCollector()
        current_costs = await collector.get_current_month_costs()
        
        if budget.services:
            # Filter by specific services
            total_cost = sum(
                cost for service, cost in current_costs.items() 
                if service in budget.services
            )
        else:
            # Use total cost
            total_cost = sum(current_costs.values())
        
        return total_cost
    
    def _alert_already_sent(
        self, 
        budget_id: str, 
        threshold: float, 
        current_spend: float
    ) -> bool:
        """Check if alert for this threshold was already sent recently"""
        recent_time = datetime.now() - timedelta(hours=24)  # 24-hour cooldown
        
        for alert in self.alert_history:
            if (alert.budget_id == budget_id and 
                alert.percentage_used >= threshold and 
                alert.timestamp > recent_time):
                return True
        
        return False
    
    async def _create_alert(
        self, 
        budget: Budget, 
        threshold: float, 
        current_spend: float, 
        percentage_used: float
    ) -> BudgetAlert:
        """Create a budget alert"""
        # Determine severity
        if percentage_used >= 100:
            severity = AlertSeverity.CRITICAL
        elif percentage_used >= 90:
            severity = AlertSeverity.HIGH
        elif percentage_used >= 80:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        # Create message
        message = self._generate_alert_message(budget, threshold, current_spend, percentage_used)
        
        alert = BudgetAlert(
            id=f"{budget.id}_{threshold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            budget_id=budget.id,
            severity=severity,
            message=message,
            current_spend=current_spend,
            budget_limit=budget.monthly_limit,
            percentage_used=percentage_used,
            timestamp=datetime.now(),
            metadata={
                'threshold': threshold,
                'budget_name': budget.name,
                'services': budget.services
            }
        )
        
        return alert
    
    def _generate_alert_message(
        self, 
        budget: Budget, 
        threshold: float, 
        current_spend: float, 
        percentage_used: float
    ) -> str:
        """Generate alert message"""
        days_remaining = (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1) - datetime.now()
        
        message = f"""
🚨 Budget Alert: {budget.name}

Current Spending: ${current_spend:,.2f}
Budget Limit: ${budget.monthly_limit:,.2f}
Percentage Used: {percentage_used:.1f}%
Threshold: {threshold}%

Days Remaining in Month: {days_remaining.days}

Projected Monthly Cost: ${(current_spend / datetime.now().day) * 30:,.2f}

This alert was triggered because spending has reached {threshold}% of your monthly budget.
        """.strip()
        
        if percentage_used >= 100:
            message += "\n\n⚠️ CRITICAL: Budget limit exceeded!"
        elif percentage_used >= 90:
            message += "\n\n⚠️ WARNING: Approaching budget limit!"
        
        return message
    
    async def _send_notifications(self, alert: BudgetAlert):
        """Send notifications through configured channels"""
        for channel in self.budgets[alert.budget_id].notification_channels:
            try:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_alert(alert)
                elif channel == NotificationChannel.SLACK:
                    await self._send_slack_alert(alert)
                elif channel == NotificationChannel.TEAMS:
                    await self._send_teams_alert(alert)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_alert(alert)
            
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")
    
    async def _send_email_alert(self, alert: BudgetAlert):
        """Send email alert"""
        if 'email' not in self.notification_config:
            logger.warning("Email configuration not found")
            return
        
        config = self.notification_config['email']
        
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = ', '.join(config['to_emails'])
        msg['Subject'] = f"Budget Alert: {alert.metadata['budget_name']}"
        
        msg.attach(MIMEText(alert.message, 'plain'))
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
        
        logger.info(f"Email alert sent for budget {alert.budget_id}")
    
    async def _send_slack_alert(self, alert: BudgetAlert):
        """Send Slack alert"""
        if 'slack' not in self.notification_config:
            logger.warning("Slack configuration not found")
            return
        
        webhook_url = self.notification_config['slack']['webhook_url']
        webhook = WebhookClient(webhook_url)
        
        # Create Slack message
        color = {
            AlertSeverity.LOW: "#36a64f",
            AlertSeverity.MEDIUM: "#ffaa00",
            AlertSeverity.HIGH: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000"
        }[alert.severity]
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 Budget Alert: {alert.metadata['budget_name']}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Current Spending:*\n${alert.current_spend:,.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Budget Limit:*\n${alert.budget_limit:,.2f}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Percentage Used:*\n{alert.percentage_used:.1f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Severity:*\n{alert.severity.value.upper()}"
                    }
                ]
            }
        ]
        
        webhook.send(blocks=blocks)
        logger.info(f"Slack alert sent for budget {alert.budget_id}")
    
    async def _send_teams_alert(self, alert: BudgetAlert):
        """Send Microsoft Teams alert"""
        if 'teams' not in self.notification_config:
            logger.warning("Teams configuration not found")
            return
        
        # Teams webhook implementation
        webhook_url = self.notification_config['teams']['webhook_url']
        
        message = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000" if alert.severity == AlertSeverity.CRITICAL else "FF6600",
            "summary": f"Budget Alert: {alert.metadata['budget_name']}",
            "sections": [{
                "activityTitle": f"🚨 Budget Alert: {alert.metadata['budget_name']}",
                "activitySubtitle": f"Spending: ${alert.current_spend:,.2f} / ${alert.budget_limit:,.2f} ({alert.percentage_used:.1f}%)",
                "facts": [
                    {"name": "Current Spending", "value": f"${alert.current_spend:,.2f}"},
                    {"name": "Budget Limit", "value": f"${alert.budget_limit:,.2f}"},
                    {"name": "Percentage Used", "value": f"{alert.percentage_used:.1f}%"},
                    {"name": "Severity", "value": alert.severity.value.upper()}
                ]
            }]
        }
        
        import requests
        requests.post(webhook_url, json=message)
        logger.info(f"Teams alert sent for budget {alert.budget_id}")
    
    async def _send_webhook_alert(self, alert: BudgetAlert):
        """Send webhook alert"""
        if 'webhook' not in self.notification_config:
            logger.warning("Webhook configuration not found")
            return
        
        webhook_url = self.notification_config['webhook']['url']
        
        payload = {
            'alert_id': alert.id,
            'budget_id': alert.budget_id,
            'severity': alert.severity.value,
            'message': alert.message,
            'current_spend': alert.current_spend,
            'budget_limit': alert.budget_limit,
            'percentage_used': alert.percentage_used,
            'timestamp': alert.timestamp.isoformat(),
            'metadata': alert.metadata
        }
        
        import requests
        requests.post(webhook_url, json=payload)
        logger.info(f"Webhook alert sent for budget {alert.budget_id}")
    
    async def get_budget_status(self, budget_id: str = None) -> Dict[str, Any]:
        """Get current budget status"""
        if budget_id and budget_id not in self.budgets:
            raise ValueError(f"Budget {budget_id} not found")
        
        budgets_to_check = [self.budgets[budget_id]] if budget_id else self.budgets.values()
        
        status = {}
        
        for budget in budgets_to_check:
            try:
                current_spend = await self._get_current_spending(budget)
                percentage_used = (current_spend / budget.monthly_limit) * 100
                
                # Get recent alerts
                recent_alerts = [
                    alert for alert in self.alert_history
                    if alert.budget_id == budget.id and 
                    alert.timestamp > datetime.now() - timedelta(days=7)
                ]
                
                status[budget.id] = {
                    'name': budget.name,
                    'monthly_limit': budget.monthly_limit,
                    'current_spend': current_spend,
                    'percentage_used': percentage_used,
                    'remaining_budget': budget.monthly_limit - current_spend,
                    'days_remaining': (datetime.now().replace(day=1) + timedelta(days=32)).replace(day=1) - datetime.now(),
                    'projected_monthly_cost': (current_spend / datetime.now().day) * 30,
                    'recent_alerts': len(recent_alerts),
                    'status': self._get_budget_status_level(percentage_used),
                    'last_updated': datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"Failed to get status for budget {budget.id}: {e}")
                status[budget.id] = {'error': str(e)}
        
        return status
    
    def _get_budget_status_level(self, percentage_used: float) -> str:
        """Get budget status level"""
        if percentage_used >= 100:
            return "CRITICAL"
        elif percentage_used >= 90:
            return "WARNING"
        elif percentage_used >= 80:
            return "CAUTION"
        else:
            return "HEALTHY"
    
    async def get_alert_history(self, budget_id: str = None, days: int = 30) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_date and
            (budget_id is None or alert.budget_id == budget_id)
        ]
        
        return [
            {
                'id': alert.id,
                'budget_id': alert.budget_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'current_spend': alert.current_spend,
                'budget_limit': alert.budget_limit,
                'percentage_used': alert.percentage_used,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            for alert in alerts
        ]
    
    def configure_notifications(self, channel: NotificationChannel, config: Dict[str, Any]):
        """Configure notification channels"""
        self.notification_config[channel.value] = config
        self._save_config()
        logger.info(f"Configured {channel.value} notifications")
