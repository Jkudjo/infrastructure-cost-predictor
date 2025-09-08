# 🚀 Infrastructure Cost Predictor & Optimizer

[![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg?style=for-the-badge&logo=terraform&logoColor=white)](https://www.terraform.io/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive platform for predicting, monitoring, and optimizing cloud infrastructure costs using machine learning and advanced analytics.

## 🎯 Features

### 📊 **Cost Forecasting Models**
- **Resource Usage Prediction**: ML-based forecasting of compute, storage, and network usage
- **Seasonal Pattern Detection**: Identify recurring cost patterns and anomalies
- **Growth Trend Analysis**: Predict future costs based on historical growth
- **Multi-Cloud Support**: AWS, Azure, GCP cost prediction

### 💰 **Budget Management & Alerts**
- **Smart Budget Alerts**: Proactive notifications before budget thresholds
- **Cost Anomaly Detection**: Identify unexpected cost spikes
- **Resource Optimization**: Automated recommendations for cost reduction
- **ROI Calculation**: Measure return on investment for infrastructure changes

### 📈 **Advanced Analytics**
- **Cost Breakdown Analysis**: Detailed cost attribution by service, region, team
- **Trend Visualization**: Interactive dashboards and reports
- **What-If Scenarios**: Model cost impact of infrastructure changes
- **Compliance Reporting**: Generate cost reports for stakeholders

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cost Prediction Platform                  │
├─────────────────────────────────────────────────────────────┤
│  Data Collection  │  ML Models  │  Alerts  │  Dashboard    │
│  ┌─────────────┐  │ ┌─────────┐ │ ┌──────┐ │ ┌───────────┐ │
│  │ AWS Cost    │  │ │ LSTM    │ │ │ Slack│ │ │ React UI  │ │
│  │ Explorer    │  │ │ Models  │ │ │ Email│ │ │ Charts    │ │
│  │ Azure       │  │ │ ARIMA   │ │ │ Teams│ │ │ Reports   │ │
│  │ GCP         │  │ │ Prophet │ │ │ Pager│ │ │ Alerts    │ │
│  └─────────────┘  │ └─────────┘ │ └──────┘ │ └───────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    ROI Calculator & Optimization            │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- AWS CLI configured
- Docker (optional)
- Node.js 16+ (for dashboard)

### Installation

```bash
# Clone the repository
git clone https://github.com/jkudjo/infrastructure-cost-predictor.git
cd infrastructure-cost-predictor

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies (for dashboard)
cd dashboard
npm install
cd ..

# Run the application
python src/main.py
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the dashboard
open http://localhost:3000
```

## 📖 Usage

### Basic Cost Prediction

```python
from src.ml_models.cost_predictor import CostPredictor

# Initialize predictor
predictor = CostPredictor()

# Load historical data
predictor.load_data('aws_cost_data.csv')

# Train model
predictor.train()

# Predict next month's costs
prediction = predictor.predict(days=30)
print(f"Predicted cost: ${prediction.total_cost:.2f}")
```

### Budget Alerts

```python
from src.alerts.budget_manager import BudgetManager

# Set up budget alerts
budget_manager = BudgetManager()
budget_manager.set_budget(
    monthly_limit=10000,
    alert_thresholds=[80, 90, 100],
    notification_channels=['slack', 'email']
)

# Check current spending
budget_manager.check_budget()
```

### ROI Calculation

```python
from src.roi_calculator.roi_analyzer import ROIAnalyzer

# Analyze infrastructure change ROI
roi_analyzer = ROIAnalyzer()
roi = roi_analyzer.calculate_roi(
    current_cost=5000,
    new_cost=3500,
    migration_cost=2000,
    timeframe_months=12
)

print(f"ROI: {roi.percentage:.1f}%")
print(f"Payback period: {roi.payback_months} months")
```

## 🔧 Configuration

### Environment Variables

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-west-2

# Database
DATABASE_URL=postgresql://user:pass@localhost/costdb

# Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
EMAIL_SMTP_SERVER=smtp.gmail.com
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_password

# Dashboard
DASHBOARD_PORT=3000
API_PORT=8000
```

### Configuration File

```yaml
# config.yaml
data_sources:
  aws:
    enabled: true
    regions: ["us-west-2", "us-east-1"]
    services: ["ec2", "s3", "rds", "lambda"]
  
  azure:
    enabled: false
    subscription_id: "your-subscription-id"
  
  gcp:
    enabled: false
    project_id: "your-project-id"

alerts:
  budget_thresholds: [80, 90, 100]
  notification_channels: ["slack", "email"]
  check_frequency: "daily"

ml_models:
  prediction_horizon_days: 30
  retrain_frequency: "weekly"
  models: ["lstm", "arima", "prophet"]
```

## 📊 Dashboard

The web dashboard provides:

- **Cost Overview**: Current spending and trends
- **Predictions**: Future cost forecasts
- **Alerts**: Budget and anomaly notifications
- **Optimization**: Cost reduction recommendations
- **Reports**: Detailed cost analysis and ROI calculations

Access the dashboard at: `http://localhost:3000`

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=src tests/
```

## 📚 Examples

See the `examples/` directory for:

- **Basic Cost Prediction**: Simple forecasting example
- **Multi-Cloud Setup**: AWS + Azure cost monitoring
- **Custom Alerts**: Advanced alerting configurations
- **ROI Analysis**: Infrastructure change evaluation

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Philip Kumah Junior** - *DevOps Engineer*

- GitHub: [@jkudjo](https://github.com/jkudjo)
- LinkedIn: [Philip Kumah Junior](https://www.linkedin.com/in/philip-kumah-junior/)
- Portfolio: [VertaForge](https://vertaforge.com/contact)

## 🙏 Acknowledgments

- AWS Cost Explorer API
- Scikit-learn and TensorFlow for ML models
- React and Chart.js for the dashboard
- The open-source community for inspiration

---

⭐ **If you found this project helpful, please give it a star!**
