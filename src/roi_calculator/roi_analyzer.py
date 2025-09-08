"""
ROI Calculator & Analyzer - Calculate return on investment for infrastructure changes
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of infrastructure changes"""
    MIGRATION = "migration"
    OPTIMIZATION = "optimization"
    SCALING = "scaling"
    NEW_SERVICE = "new_service"
    DEPRECATION = "deprecation"


class RiskLevel(Enum):
    """Risk levels for infrastructure changes"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CostBreakdown:
    """Cost breakdown for infrastructure"""
    compute: float = 0.0
    storage: float = 0.0
    network: float = 0.0
    database: float = 0.0
    monitoring: float = 0.0
    security: float = 0.0
    other: float = 0.0
    
    @property
    def total(self) -> float:
        return sum([
            self.compute, self.storage, self.network,
            self.database, self.monitoring, self.security, self.other
        ])


@dataclass
class InfrastructureChange:
    """Infrastructure change definition"""
    id: str
    name: str
    change_type: ChangeType
    description: str
    current_cost: CostBreakdown
    new_cost: CostBreakdown
    migration_cost: float = 0.0
    risk_level: RiskLevel = RiskLevel.MEDIUM
    implementation_time_days: int = 30
    expected_benefits: List[str] = None
    dependencies: List[str] = None


@dataclass
class ROIAnalysis:
    """ROI analysis results"""
    change_id: str
    change_name: str
    total_savings: float
    migration_cost: float
    payback_period_months: float
    roi_percentage: float
    net_present_value: float
    internal_rate_of_return: float
    risk_adjusted_roi: float
    break_even_date: datetime
    confidence_score: float
    recommendations: List[str]
    cost_breakdown: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]


class ROIAnalyzer:
    """
    Advanced ROI calculator for infrastructure changes
    """
    
    def __init__(self, discount_rate: float = 0.08):
        self.discount_rate = discount_rate  # 8% default discount rate
        self.analysis_history: List[ROIAnalysis] = []
        
    async def calculate_roi(
        self,
        current_cost: float,
        new_cost: float,
        migration_cost: float = 0.0,
        timeframe_months: int = 12,
        change_type: ChangeType = ChangeType.OPTIMIZATION,
        risk_level: RiskLevel = RiskLevel.MEDIUM
    ) -> ROIAnalysis:
        """
        Calculate ROI for infrastructure changes
        """
        logger.info(f"Calculating ROI for {change_type.value} change")
        
        # Basic calculations
        monthly_savings = current_cost - new_cost
        total_savings = monthly_savings * timeframe_months
        
        # Payback period
        if monthly_savings > 0:
            payback_period_months = migration_cost / monthly_savings
        else:
            payback_period_months = float('inf')
        
        # ROI percentage
        if migration_cost > 0:
            roi_percentage = ((total_savings - migration_cost) / migration_cost) * 100
        else:
            roi_percentage = (total_savings / current_cost) * 100 if current_cost > 0 else 0
        
        # Net Present Value (NPV)
        npv = self._calculate_npv(monthly_savings, migration_cost, timeframe_months)
        
        # Internal Rate of Return (IRR)
        irr = self._calculate_irr(monthly_savings, migration_cost, timeframe_months)
        
        # Risk-adjusted ROI
        risk_adjusted_roi = self._calculate_risk_adjusted_roi(
            roi_percentage, risk_level, change_type
        )
        
        # Break-even date
        break_even_date = self._calculate_break_even_date(payback_period_months)
        
        # Confidence score
        confidence_score = self._calculate_confidence_score(
            change_type, risk_level, timeframe_months, monthly_savings
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            roi_percentage, payback_period_months, risk_level, change_type
        )
        
        # Cost breakdown analysis
        cost_breakdown = self._analyze_cost_breakdown(current_cost, new_cost, migration_cost)
        
        # Sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            current_cost, new_cost, migration_cost, timeframe_months
        )
        
        analysis = ROIAnalysis(
            change_id=f"change_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            change_name=f"{change_type.value.title()} Change",
            total_savings=total_savings,
            migration_cost=migration_cost,
            payback_period_months=payback_period_months,
            roi_percentage=roi_percentage,
            net_present_value=npv,
            internal_rate_of_return=irr,
            risk_adjusted_roi=risk_adjusted_roi,
            break_even_date=break_even_date,
            confidence_score=confidence_score,
            recommendations=recommendations,
            cost_breakdown=cost_breakdown,
            sensitivity_analysis=sensitivity_analysis
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    async def analyze_infrastructure_change(self, change: InfrastructureChange) -> ROIAnalysis:
        """
        Analyze ROI for a detailed infrastructure change
        """
        logger.info(f"Analyzing infrastructure change: {change.name}")
        
        # Calculate total costs
        current_total = change.current_cost.total
        new_total = change.new_cost.total
        
        # Calculate ROI
        analysis = await self.calculate_roi(
            current_cost=current_total,
            new_cost=new_total,
            migration_cost=change.migration_cost,
            timeframe_months=12,
            change_type=change.change_type,
            risk_level=change.risk_level
        )
        
        # Update with change-specific details
        analysis.change_id = change.id
        analysis.change_name = change.name
        
        # Add detailed cost breakdown
        analysis.cost_breakdown.update({
            'current_breakdown': {
                'compute': change.current_cost.compute,
                'storage': change.current_cost.storage,
                'network': change.current_cost.network,
                'database': change.current_cost.database,
                'monitoring': change.current_cost.monitoring,
                'security': change.current_cost.security,
                'other': change.current_cost.other
            },
            'new_breakdown': {
                'compute': change.new_cost.compute,
                'storage': change.new_cost.storage,
                'network': change.new_cost.network,
                'database': change.new_cost.database,
                'monitoring': change.new_cost.monitoring,
                'security': change.new_cost.security,
                'other': change.new_cost.other
            },
            'cost_differences': {
                'compute': change.new_cost.compute - change.current_cost.compute,
                'storage': change.new_cost.storage - change.current_cost.storage,
                'network': change.new_cost.network - change.current_cost.network,
                'database': change.new_cost.database - change.current_cost.database,
                'monitoring': change.new_cost.monitoring - change.current_cost.monitoring,
                'security': change.new_cost.security - change.current_cost.security,
                'other': change.new_cost.other - change.current_cost.other
            }
        })
        
        return analysis
    
    def _calculate_npv(self, monthly_savings: float, migration_cost: float, months: int) -> float:
        """Calculate Net Present Value"""
        npv = -migration_cost  # Initial investment (negative)
        
        for month in range(1, months + 1):
            discount_factor = 1 / ((1 + self.discount_rate / 12) ** month)
            npv += monthly_savings * discount_factor
        
        return npv
    
    def _calculate_irr(self, monthly_savings: float, migration_cost: float, months: int) -> float:
        """Calculate Internal Rate of Return"""
        # Use numpy's financial functions for IRR calculation
        cash_flows = [-migration_cost] + [monthly_savings] * months
        
        # Simple IRR approximation
        if migration_cost == 0:
            return float('inf') if monthly_savings > 0 else 0
        
        # Use iterative method to find IRR
        for rate in np.arange(0.01, 2.0, 0.01):
            npv = -migration_cost
            for month in range(1, months + 1):
                discount_factor = 1 / ((1 + rate / 12) ** month)
                npv += monthly_savings * discount_factor
            
            if abs(npv) < 0.01:  # Close to zero
                return rate * 100  # Return as percentage
        
        return 0.0  # No IRR found
    
    def _calculate_risk_adjusted_roi(
        self, 
        roi_percentage: float, 
        risk_level: RiskLevel, 
        change_type: ChangeType
    ) -> float:
        """Calculate risk-adjusted ROI"""
        # Risk multipliers
        risk_multipliers = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.9,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.5
        }
        
        # Change type multipliers
        change_multipliers = {
            ChangeType.OPTIMIZATION: 1.0,
            ChangeType.MIGRATION: 0.8,
            ChangeType.SCALING: 0.9,
            ChangeType.NEW_SERVICE: 0.7,
            ChangeType.DEPRECATION: 0.6
        }
        
        risk_multiplier = risk_multipliers[risk_level]
        change_multiplier = change_multipliers[change_type]
        
        return roi_percentage * risk_multiplier * change_multiplier
    
    def _calculate_break_even_date(self, payback_period_months: float) -> datetime:
        """Calculate break-even date"""
        if payback_period_months == float('inf'):
            return datetime.max
        
        return datetime.now() + timedelta(days=payback_period_months * 30)
    
    def _calculate_confidence_score(
        self,
        change_type: ChangeType,
        risk_level: RiskLevel,
        timeframe_months: int,
        monthly_savings: float
    ) -> float:
        """Calculate confidence score for the ROI analysis"""
        score = 1.0
        
        # Risk level impact
        risk_scores = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.4
        }
        score *= risk_scores[risk_level]
        
        # Change type impact
        change_scores = {
            ChangeType.OPTIMIZATION: 0.9,
            ChangeType.SCALING: 0.8,
            ChangeType.MIGRATION: 0.7,
            ChangeType.NEW_SERVICE: 0.6,
            ChangeType.DEPRECATION: 0.5
        }
        score *= change_scores[change_type]
        
        # Timeframe impact (longer timeframes are less certain)
        if timeframe_months > 24:
            score *= 0.8
        elif timeframe_months > 12:
            score *= 0.9
        
        # Savings magnitude impact
        if abs(monthly_savings) < 100:  # Small savings are less certain
            score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(
        self,
        roi_percentage: float,
        payback_period_months: float,
        risk_level: RiskLevel,
        change_type: ChangeType
    ) -> List[str]:
        """Generate recommendations based on ROI analysis"""
        recommendations = []
        
        # ROI-based recommendations
        if roi_percentage > 100:
            recommendations.append("✅ Excellent ROI - Strongly recommend proceeding")
        elif roi_percentage > 50:
            recommendations.append("✅ Good ROI - Recommend proceeding")
        elif roi_percentage > 0:
            recommendations.append("⚠️ Positive ROI but consider alternatives")
        else:
            recommendations.append("❌ Negative ROI - Do not recommend")
        
        # Payback period recommendations
        if payback_period_months < 6:
            recommendations.append("💰 Quick payback period - Low risk investment")
        elif payback_period_months < 12:
            recommendations.append("⏱️ Reasonable payback period")
        elif payback_period_months < 24:
            recommendations.append("⏳ Long payback period - Consider risk factors")
        else:
            recommendations.append("🚫 Very long payback period - High risk")
        
        # Risk-based recommendations
        if risk_level == RiskLevel.HIGH or risk_level == RiskLevel.CRITICAL:
            recommendations.append("⚠️ High risk change - Implement in phases")
            recommendations.append("📊 Monitor closely during implementation")
        
        # Change type specific recommendations
        if change_type == ChangeType.MIGRATION:
            recommendations.append("🔄 Plan for migration downtime")
            recommendations.append("🧪 Test thoroughly in staging environment")
        elif change_type == ChangeType.NEW_SERVICE:
            recommendations.append("📚 Ensure team training on new service")
            recommendations.append("🔧 Plan for integration complexity")
        
        return recommendations
    
    def _analyze_cost_breakdown(
        self, 
        current_cost: float, 
        new_cost: float, 
        migration_cost: float
    ) -> Dict[str, Any]:
        """Analyze cost breakdown and savings"""
        monthly_savings = current_cost - new_cost
        annual_savings = monthly_savings * 12
        
        return {
            'current_monthly_cost': current_cost,
            'new_monthly_cost': new_cost,
            'monthly_savings': monthly_savings,
            'annual_savings': annual_savings,
            'migration_cost': migration_cost,
            'savings_percentage': (monthly_savings / current_cost * 100) if current_cost > 0 else 0,
            'cost_reduction_areas': self._identify_cost_reduction_areas(current_cost, new_cost)
        }
    
    def _identify_cost_reduction_areas(self, current_cost: float, new_cost: float) -> List[str]:
        """Identify areas of cost reduction"""
        areas = []
        
        # This would be more sophisticated in a real implementation
        # For now, provide generic recommendations
        if new_cost < current_cost:
            areas.append("Overall infrastructure optimization")
            areas.append("Resource right-sizing")
            areas.append("Cost-effective service selection")
        
        return areas
    
    def _perform_sensitivity_analysis(
        self,
        current_cost: float,
        new_cost: float,
        migration_cost: float,
        timeframe_months: int
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on key variables"""
        base_savings = current_cost - new_cost
        
        # Test different scenarios
        scenarios = {
            'optimistic': {
                'savings_multiplier': 1.2,
                'migration_cost_multiplier': 0.8
            },
            'realistic': {
                'savings_multiplier': 1.0,
                'migration_cost_multiplier': 1.0
            },
            'pessimistic': {
                'savings_multiplier': 0.8,
                'migration_cost_multiplier': 1.2
            }
        }
        
        sensitivity_results = {}
        
        for scenario_name, multipliers in scenarios.items():
            adjusted_savings = base_savings * multipliers['savings_multiplier']
            adjusted_migration_cost = migration_cost * multipliers['migration_cost_multiplier']
            
            if adjusted_savings > 0:
                payback_period = adjusted_migration_cost / adjusted_savings
                roi = ((adjusted_savings * timeframe_months - adjusted_migration_cost) / 
                      adjusted_migration_cost * 100) if adjusted_migration_cost > 0 else 0
            else:
                payback_period = float('inf')
                roi = 0
            
            sensitivity_results[scenario_name] = {
                'monthly_savings': adjusted_savings,
                'migration_cost': adjusted_migration_cost,
                'payback_period_months': payback_period,
                'roi_percentage': roi
            }
        
        return sensitivity_results
    
    async def compare_changes(self, changes: List[InfrastructureChange]) -> Dict[str, Any]:
        """Compare multiple infrastructure changes"""
        logger.info(f"Comparing {len(changes)} infrastructure changes")
        
        analyses = []
        for change in changes:
            analysis = await self.analyze_infrastructure_change(change)
            analyses.append(analysis)
        
        # Sort by risk-adjusted ROI
        analyses.sort(key=lambda x: x.risk_adjusted_roi, reverse=True)
        
        comparison = {
            'total_changes': len(changes),
            'ranked_changes': [
                {
                    'rank': i + 1,
                    'change_id': analysis.change_id,
                    'change_name': analysis.change_name,
                    'roi_percentage': analysis.roi_percentage,
                    'risk_adjusted_roi': analysis.risk_adjusted_roi,
                    'payback_period_months': analysis.payback_period_months,
                    'confidence_score': analysis.confidence_score
                }
                for i, analysis in enumerate(analyses)
            ],
            'best_change': {
                'change_id': analyses[0].change_id,
                'change_name': analyses[0].change_name,
                'roi_percentage': analyses[0].roi_percentage,
                'risk_adjusted_roi': analyses[0].risk_adjusted_roi
            } if analyses else None,
            'summary': {
                'average_roi': np.mean([a.roi_percentage for a in analyses]),
                'average_payback_period': np.mean([a.payback_period_months for a in analyses]),
                'total_potential_savings': sum([a.total_savings for a in analyses]),
                'total_migration_cost': sum([a.migration_cost for a in analyses])
            }
        }
        
        return comparison
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent ROI analysis history"""
        recent_analyses = self.analysis_history[-limit:]
        
        return [
            {
                'change_id': analysis.change_id,
                'change_name': analysis.change_name,
                'roi_percentage': analysis.roi_percentage,
                'payback_period_months': analysis.payback_period_months,
                'confidence_score': analysis.confidence_score,
                'timestamp': datetime.now().isoformat()
            }
            for analysis in recent_analyses
        ]
