#!/usr/bin/env python3
"""
Basic Usage Example - Infrastructure Cost Predictor
"""

import asyncio
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_models.cost_predictor import CostPredictor
from data_collection.aws_collector import AWSCostCollector
from alerts.budget_manager import BudgetManager, NotificationChannel
from roi_calculator.roi_analyzer import ROIAnalyzer, InfrastructureChange, ChangeType, RiskLevel, CostBreakdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function"""
    print("🚀 Infrastructure Cost Predictor - Basic Usage Example")
    print("=" * 60)
    
    # 1. Cost Prediction Example
    print("\n📊 1. Cost Prediction Example")
    print("-" * 30)
    
    cost_predictor = CostPredictor()
    
    # Load sample data (in real usage, this would be from AWS)
    sample_data = generate_sample_data()
    cost_predictor.training_data = sample_data
    cost_predictor.feature_columns = [col for col in sample_data.columns if col != 'total_cost']
    
    # Train models
    print("Training ML models...")
    await cost_predictor.train()
    
    # Make predictions
    prediction = await cost_predictor.predict(days=30)
    print(f"Predicted cost for next 30 days: ${prediction['total_cost']:,.2f}")
    print(f"Daily average: ${prediction['daily_average']:,.2f}")
    print(f"Confidence: {prediction['confidence']:.1%}")
    
    # 2. AWS Cost Collection Example
    print("\n☁️ 2. AWS Cost Collection Example")
    print("-" * 35)
    
    try:
        aws_collector = AWSCostCollector()
        
        # Get current month costs
        current_costs = await aws_collector.get_current_month_costs()
        print("Current month costs by service:")
        for service, cost in current_costs.items():
            print(f"  {service}: ${cost:,.2f}")
        
        # Get cost trends
        trends = await aws_collector.get_cost_trends(days=30)
        print(f"\nCost trends (last 30 days):")
        print(f"  Total cost: ${trends['total_cost']:,.2f}")
        print(f"  Growth rate: {trends['growth_rate_percent']:.1f}%")
        print(f"  Average daily: ${trends['average_daily_cost']:,.2f}")
        
    except Exception as e:
        print(f"⚠️ AWS collection failed (expected in demo): {e}")
    
    # 3. Budget Management Example
    print("\n💰 3. Budget Management Example")
    print("-" * 32)
    
    budget_manager = BudgetManager()
    
    # Set up a budget
    budget = await budget_manager.set_budget(
        budget_id="demo_budget",
        name="Demo Infrastructure Budget",
        monthly_limit=10000.0,
        alert_thresholds=[80.0, 90.0, 100.0],
        notification_channels=[NotificationChannel.EMAIL]
    )
    
    print(f"Created budget: {budget.name}")
    print(f"Monthly limit: ${budget.monthly_limit:,.2f}")
    
    # Check budget status
    status = await budget_manager.get_budget_status("demo_budget")
    print(f"Current status: {status['demo_budget']['status']}")
    print(f"Percentage used: {status['demo_budget']['percentage_used']:.1f}%")
    
    # 4. ROI Analysis Example
    print("\n📈 4. ROI Analysis Example")
    print("-" * 28)
    
    roi_analyzer = ROIAnalyzer()
    
    # Create an infrastructure change scenario
    current_cost = CostBreakdown(
        compute=5000.0,
        storage=1000.0,
        network=500.0,
        database=1500.0,
        monitoring=200.0,
        security=300.0
    )
    
    new_cost = CostBreakdown(
        compute=3500.0,  # Reduced compute costs
        storage=800.0,   # Optimized storage
        network=400.0,   # Better networking
        database=1200.0, # Optimized database
        monitoring=200.0,
        security=300.0
    )
    
    change = InfrastructureChange(
        id="ec2_optimization",
        name="EC2 Instance Optimization",
        change_type=ChangeType.OPTIMIZATION,
        description="Right-size EC2 instances and optimize storage",
        current_cost=current_cost,
        new_cost=new_cost,
        migration_cost=500.0,
        risk_level=RiskLevel.LOW,
        implementation_time_days=14,
        expected_benefits=[
            "Reduced compute costs",
            "Better resource utilization",
            "Improved performance"
        ]
    )
    
    # Analyze ROI
    roi_analysis = await roi_analyzer.analyze_infrastructure_change(change)
    
    print(f"ROI Analysis for: {roi_analysis.change_name}")
    print(f"  ROI Percentage: {roi_analysis.roi_percentage:.1f}%")
    print(f"  Payback Period: {roi_analysis.payback_period_months:.1f} months")
    print(f"  Total Savings: ${roi_analysis.total_savings:,.2f}")
    print(f"  Migration Cost: ${roi_analysis.migration_cost:,.2f}")
    print(f"  Confidence Score: {roi_analysis.confidence_score:.1%}")
    
    print("\nRecommendations:")
    for rec in roi_analysis.recommendations:
        print(f"  {rec}")
    
    # 5. Cost Anomaly Detection Example
    print("\n🚨 5. Cost Anomaly Detection Example")
    print("-" * 38)
    
    try:
        anomalies = await aws_collector.get_cost_anomalies(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            threshold=2.0
        )
        
        if anomalies:
            print(f"Found {len(anomalies)} cost anomalies:")
            for anomaly in anomalies[:3]:  # Show first 3
                print(f"  {anomaly['date']}: {anomaly['service']} - ${anomaly['cost']:,.2f} (Z-score: {anomaly['z_score']:.1f})")
        else:
            print("No significant cost anomalies detected")
            
    except Exception as e:
        print(f"⚠️ Anomaly detection failed (expected in demo): {e}")
    
    print("\n✅ Example completed successfully!")
    print("\nTo use this in production:")
    print("1. Configure AWS credentials")
    print("2. Set up notification channels (Slack, email)")
    print("3. Customize budget thresholds")
    print("4. Train models with your historical data")


def generate_sample_data():
    """Generate sample cost data for demonstration"""
    import pandas as pd
    import numpy as np
    
    # Generate 365 days of sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create realistic cost patterns with seasonality
    base_cost = 8000
    seasonal_pattern = 1000 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    trend = np.arange(len(dates)) * 5  # Slight upward trend
    noise = np.random.normal(0, 200, len(dates))
    
    total_cost = base_cost + seasonal_pattern + trend + noise
    
    # Create service breakdown
    ec2_cost = total_cost * 0.6 + np.random.normal(0, 100, len(dates))
    s3_cost = total_cost * 0.15 + np.random.normal(0, 50, len(dates))
    rds_cost = total_cost * 0.2 + np.random.normal(0, 75, len(dates))
    other_cost = total_cost * 0.05 + np.random.normal(0, 25, len(dates))
    
    data = pd.DataFrame({
        'date': dates,
        'total_cost': total_cost,
        'ec2_cost': ec2_cost,
        's3_cost': s3_cost,
        'rds_cost': rds_cost,
        'other_cost': other_cost,
        'year': dates.year,
        'month': dates.month,
        'day_of_week': dates.dayofweek,
        'day_of_year': dates.dayofyear,
        'is_weekend': (dates.dayofweek >= 5).astype(int)
    })
    
    return data


if __name__ == "__main__":
    asyncio.run(main())
