"""
AWS Cost Data Collection - Gather cost and usage data from AWS
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class AWSCostCollector:
    """
    Collect cost and usage data from AWS Cost Explorer and other services
    """
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.cost_explorer = None
        self.ce_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AWS clients"""
        try:
            # Cost Explorer client
            self.ce_client = boto3.client('ce', region_name=self.region)
            
            # Other AWS services for detailed cost data
            self.ec2_client = boto3.client('ec2', region_name=self.region)
            self.s3_client = boto3.client('s3', region_name=self.region)
            self.rds_client = boto3.client('rds', region_name=self.region)
            self.lambda_client = boto3.client('lambda', region_name=self.region)
            
            logger.info("AWS clients initialized successfully")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    async def get_cost_and_usage(
        self, 
        start_date: datetime, 
        end_date: datetime,
        granularity: str = 'DAILY',
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Get cost and usage data from AWS Cost Explorer
        """
        if metrics is None:
            metrics = ['BlendedCost', 'UsageQuantity']
        
        logger.info(f"Fetching cost data from {start_date} to {end_date}")
        
        try:
            # Prepare request parameters
            params = {
                'TimePeriod': {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                'Granularity': granularity,
                'Metrics': metrics,
                'GroupBy': [
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    }
                ]
            }
            
            # Get cost data
            response = self.ce_client.get_cost_and_usage(**params)
            
            # Process response
            cost_data = self._process_cost_response(response)
            
            logger.info(f"Retrieved {len(cost_data)} cost records")
            return cost_data
            
        except ClientError as e:
            logger.error(f"AWS API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get cost data: {e}")
            raise
    
    def _process_cost_response(self, response: Dict) -> pd.DataFrame:
        """Process AWS Cost Explorer response"""
        records = []
        
        for result in response.get('ResultsByTime', []):
            time_period = result['TimePeriod']
            date = datetime.strptime(time_period['Start'], '%Y-%m-%d')
            
            # Process each service group
            for group in result.get('Groups', []):
                service = group['Keys'][0] if group['Keys'] else 'Unknown'
                
                # Extract metrics
                metrics = group['Metrics']
                cost = float(metrics.get('BlendedCost', {}).get('Amount', 0))
                usage = float(metrics.get('UsageQuantity', {}).get('Amount', 0))
                
                records.append({
                    'date': date,
                    'service': service,
                    'cost': cost,
                    'usage': usage,
                    'currency': metrics.get('BlendedCost', {}).get('Unit', 'USD')
                })
        
        return pd.DataFrame(records)
    
    async def get_detailed_cost_breakdown(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Get detailed cost breakdown by service and resource
        """
        logger.info("Fetching detailed cost breakdown")
        
        breakdown = {}
        
        # Get overall cost data
        breakdown['overall'] = await self.get_cost_and_usage(start_date, end_date)
        
        # Get EC2 costs
        breakdown['ec2'] = await self._get_ec2_costs(start_date, end_date)
        
        # Get S3 costs
        breakdown['s3'] = await self._get_s3_costs(start_date, end_date)
        
        # Get RDS costs
        breakdown['rds'] = await self._get_rds_costs(start_date, end_date)
        
        # Get Lambda costs
        breakdown['lambda'] = await self._get_lambda_costs(start_date, end_date)
        
        return breakdown
    
    async def _get_ec2_costs(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get EC2-specific cost data"""
        try:
            params = {
                'TimePeriod': {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                'Granularity': 'DAILY',
                'Metrics': ['BlendedCost'],
                'GroupBy': [
                    {
                        'Type': 'DIMENSION',
                        'Key': 'INSTANCE_TYPE'
                    }
                ],
                'Filter': {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Elastic Compute Cloud - Compute']
                    }
                }
            }
            
            response = self.ce_client.get_cost_and_usage(**params)
            return self._process_cost_response(response)
            
        except Exception as e:
            logger.error(f"Failed to get EC2 costs: {e}")
            return pd.DataFrame()
    
    async def _get_s3_costs(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get S3-specific cost data"""
        try:
            params = {
                'TimePeriod': {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                'Granularity': 'DAILY',
                'Metrics': ['BlendedCost'],
                'GroupBy': [
                    {
                        'Type': 'DIMENSION',
                        'Key': 'USAGE_TYPE'
                    }
                ],
                'Filter': {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Simple Storage Service']
                    }
                }
            }
            
            response = self.ce_client.get_cost_and_usage(**params)
            return self._process_cost_response(response)
            
        except Exception as e:
            logger.error(f"Failed to get S3 costs: {e}")
            return pd.DataFrame()
    
    async def _get_rds_costs(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get RDS-specific cost data"""
        try:
            params = {
                'TimePeriod': {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                'Granularity': 'DAILY',
                'Metrics': ['BlendedCost'],
                'GroupBy': [
                    {
                        'Type': 'DIMENSION',
                        'Key': 'DATABASE_ENGINE'
                    }
                ],
                'Filter': {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['Amazon Relational Database Service']
                    }
                }
            }
            
            response = self.ce_client.get_cost_and_usage(**params)
            return self._process_cost_response(response)
            
        except Exception as e:
            logger.error(f"Failed to get RDS costs: {e}")
            return pd.DataFrame()
    
    async def _get_lambda_costs(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get Lambda-specific cost data"""
        try:
            params = {
                'TimePeriod': {
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                'Granularity': 'DAILY',
                'Metrics': ['BlendedCost'],
                'Filter': {
                    'Dimensions': {
                        'Key': 'SERVICE',
                        'Values': ['AWS Lambda']
                    }
                }
            }
            
            response = self.ce_client.get_cost_and_usage(**params)
            return self._process_cost_response(response)
            
        except Exception as e:
            logger.error(f"Failed to get Lambda costs: {e}")
            return pd.DataFrame()
    
    async def get_current_month_costs(self) -> Dict[str, float]:
        """Get current month's costs by service"""
        today = datetime.now()
        start_of_month = today.replace(day=1)
        
        cost_data = await self.get_cost_and_usage(start_of_month, today)
        
        # Aggregate by service
        service_costs = cost_data.groupby('service')['cost'].sum().to_dict()
        
        return service_costs
    
    async def get_cost_anomalies(
        self, 
        start_date: datetime, 
        end_date: datetime,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect cost anomalies using statistical methods
        """
        logger.info("Detecting cost anomalies")
        
        cost_data = await self.get_cost_and_usage(start_date, end_date)
        
        if cost_data.empty:
            return []
        
        anomalies = []
        
        # Group by service and detect anomalies
        for service in cost_data['service'].unique():
            service_data = cost_data[cost_data['service'] == service]['cost']
            
            if len(service_data) < 7:  # Need at least a week of data
                continue
            
            # Calculate rolling statistics
            rolling_mean = service_data.rolling(window=7, min_periods=3).mean()
            rolling_std = service_data.rolling(window=7, min_periods=3).std()
            
            # Detect anomalies (costs > threshold * standard deviations from mean)
            z_scores = (service_data - rolling_mean) / rolling_std
            anomaly_mask = z_scores > threshold
            
            for idx, is_anomaly in anomaly_mask.items():
                if is_anomaly:
                    anomalies.append({
                        'date': cost_data.iloc[idx]['date'],
                        'service': service,
                        'cost': cost_data.iloc[idx]['cost'],
                        'expected_cost': rolling_mean.iloc[idx],
                        'z_score': z_scores.iloc[idx],
                        'severity': 'high' if z_scores.iloc[idx] > 3 else 'medium'
                    })
        
        return anomalies
    
    async def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trends and insights"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        cost_data = await self.get_cost_and_usage(start_date, end_date)
        
        if cost_data.empty:
            return {}
        
        # Calculate trends
        daily_costs = cost_data.groupby('date')['cost'].sum()
        
        # Linear trend
        x = np.arange(len(daily_costs))
        y = daily_costs.values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Growth rate
        first_cost = daily_costs.iloc[0]
        last_cost = daily_costs.iloc[-1]
        growth_rate = ((last_cost - first_cost) / first_cost) * 100 if first_cost > 0 else 0
        
        # Top services by cost
        service_costs = cost_data.groupby('service')['cost'].sum().sort_values(ascending=False)
        top_services = service_costs.head(5).to_dict()
        
        return {
            'total_cost': daily_costs.sum(),
            'average_daily_cost': daily_costs.mean(),
            'growth_rate_percent': growth_rate,
            'trend_slope': slope,
            'top_services': top_services,
            'cost_variance': daily_costs.var(),
            'peak_cost': daily_costs.max(),
            'lowest_cost': daily_costs.min()
        }
    
    async def export_cost_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        format: str = 'csv',
        filename: str = None
    ) -> str:
        """
        Export cost data to file
        """
        if filename is None:
            filename = f"aws_costs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        cost_data = await self.get_detailed_cost_breakdown(start_date, end_date)
        
        if format.lower() == 'csv':
            # Combine all data
            all_data = []
            for service, data in cost_data.items():
                if not data.empty:
                    data['service_type'] = service
                    all_data.append(data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                filepath = f"{filename}.csv"
                combined_data.to_csv(filepath, index=False)
                logger.info(f"Cost data exported to {filepath}")
                return filepath
        
        elif format.lower() == 'json':
            # Convert to JSON format
            json_data = {}
            for service, data in cost_data.items():
                if not data.empty:
                    json_data[service] = data.to_dict('records')
            
            filepath = f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            logger.info(f"Cost data exported to {filepath}")
            return filepath
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def get_cost_forecast_data(self, days: int = 90) -> pd.DataFrame:
        """
        Get historical cost data for ML model training
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily cost data
        cost_data = await self.get_cost_and_usage(start_date, end_date)
        
        if cost_data.empty:
            return pd.DataFrame()
        
        # Aggregate by date
        daily_costs = cost_data.groupby('date').agg({
            'cost': 'sum',
            'usage': 'sum'
        }).reset_index()
        
        # Add service breakdown
        service_breakdown = cost_data.pivot_table(
            index='date', 
            columns='service', 
            values='cost', 
            fill_value=0
        )
        
        # Combine data
        forecast_data = daily_costs.merge(service_breakdown, on='date', how='left')
        forecast_data = forecast_data.fillna(0)
        
        # Rename columns for ML model
        forecast_data = forecast_data.rename(columns={'cost': 'total_cost'})
        
        return forecast_data


# Utility functions
async def collect_aws_costs_for_training(days: int = 365) -> pd.DataFrame:
    """
    Collect AWS cost data for ML model training
    """
    collector = AWSCostCollector()
    return await collector.get_cost_forecast_data(days)


async def get_current_aws_costs() -> Dict[str, float]:
    """
    Get current month's AWS costs
    """
    collector = AWSCostCollector()
    return await collector.get_current_month_costs()


async def detect_aws_cost_anomalies(days: int = 30) -> List[Dict[str, Any]]:
    """
    Detect AWS cost anomalies
    """
    collector = AWSCostCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return await collector.get_cost_anomalies(start_date, end_date)
