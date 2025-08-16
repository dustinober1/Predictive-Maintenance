# Insight - Predictive Maintenance Dashboard

A comprehensive real-time predictive maintenance system that monitors industrial equipment health and predicts failures before they happen. Built as a portfolio project demonstrating full-stack development with machine learning integration.

## 🏗️ Architecture

- **Backend**: FastAPI with WebSocket support for real-time communication
- **ML Models**: 
  - Autoencoder neural network for anomaly detection
  - LSTM/Linear regression for Remaining Useful Life (RUL) prediction
- **Data Streaming**: Custom message broker simulating real-time sensor data
- **Frontend**: React with TypeScript and responsive dashboard components
- **Deployment**: Docker containers with CI/CD pipeline

## ✨ Features

### Real-Time Monitoring
- Live sensor data streaming from multiple engines
- Real-time anomaly detection with threshold-based alerts
- WebSocket connections for instant dashboard updates

### Machine Learning
- **Anomaly Detection**: Autoencoder model trained on healthy engine data
- **RUL Prediction**: Regression model predicting remaining useful life
- Automated model training and evaluation pipelines

### Dashboard Interface
- Interactive fleet overview with engine status grid
- Real-time metrics cards showing system health
- Alert management system with severity levels
- Responsive design optimized for operations centers

### DevOps & Deployment
- Containerized microservices architecture
- GitHub Actions CI/CD pipeline
- Automated testing and code quality checks
- Production-ready deployment configuration

## 📊 Dataset

Uses NASA Turbofan Engine Degradation Simulation Data Set with:
- 100 engines with complete run-to-failure data
- 21 sensor measurements per engine cycle
- 3 operational settings (altitude, throttle, etc.)
- Realistic degradation patterns and failure modes

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)
```bash
git clone <repo-url>
cd Predictive-Maintenance
docker-compose up -d
```

### Option 2: Manual Setup
```bash
# Backend
cd backend
pip install -r requirements.txt
python app/simple_server.py

# Frontend (in new terminal)
cd frontend
npm install
npm run dev
```

### Access Points
- **Dashboard**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📁 Project Structure

```
Predictive-Maintenance/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI application
│   │   ├── simple_server.py # Alternative HTTP server
│   │   └── data_streaming.py # Real-time data simulation
│   ├── models/             # ML model implementations
│   │   ├── autoencoder.py  # Anomaly detection model
│   │   ├── lstm_rul.py     # LSTM RUL prediction
│   │   └── simple_rul.py   # Linear regression RUL
│   └── requirements.txt    # Python dependencies
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   ├── EngineGrid.tsx
│   │   │   └── AlertsList.tsx
│   │   └── App.tsx         # Main application
│   └── package.json        # Node.js dependencies
├── data/                   # Dataset and processed data
│   ├── raw/               # Original NASA dataset
│   └── processed/         # Preprocessed training data
├── models/                # Trained model artifacts
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Utility scripts
├── .github/workflows/     # CI/CD pipeline
├── docker-compose.yml     # Container orchestration
└── README.md
```

## 🤖 Machine Learning Models

### Anomaly Detection (Autoencoder)
- **Architecture**: Input → Hidden(32) → Output reconstruction
- **Training**: Only on healthy engine data (64% accuracy on test set)
- **Detection**: Reconstruction error above 95th percentile threshold
- **Features**: 24 sensor and operational features

### RUL Prediction (Linear Regression)
- **Features**: Sensor readings + engineered features (cycle patterns)
- **Target**: Remaining useful life in cycles
- **Performance**: RMSE ~164 cycles, suitable for trend analysis
- **Deployment**: Real-time inference via REST API

## 🔄 Data Flow

1. **Data Simulation**: Historical NASA data replayed at configurable speed
2. **Stream Processing**: Message broker distributes sensor readings
3. **ML Inference**: Real-time anomaly detection and RUL prediction
4. **API Layer**: FastAPI serves processed data and predictions
5. **Frontend**: React dashboard displays real-time visualizations
6. **WebSocket**: Bi-directional communication for live updates

## 🐳 Deployment

### Development
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production
The GitHub Actions pipeline automatically:
1. Runs tests and code quality checks
2. Builds Docker images
3. Pushes to container registry
4. Deploys to cloud platform
5. Performs health checks

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test

# Integration tests
python test_api.py
```

## 📈 Performance Metrics

### Model Performance
- **Anomaly Detection**: 64% accuracy, 63% precision, 14% recall
- **RUL Prediction**: RMSE 164 cycles, R² -2.77 (needs improvement)
- **Inference Speed**: <50ms per prediction

### System Performance
- **Data Throughput**: 1000+ sensor readings/minute
- **Dashboard Latency**: <100ms real-time updates
- **Memory Usage**: <512MB per service
- **Startup Time**: <30 seconds full system

## 🔮 Future Enhancements

### Machine Learning
- [ ] Implement advanced LSTM/GRU architectures
- [ ] Add ensemble models for improved accuracy
- [ ] Implement online learning for model adaptation
- [ ] Add explainable AI features

### System Features
- [ ] Multi-tenant support for different fleets
- [ ] Advanced alerting with notification channels
- [ ] Historical trend analysis and reporting
- [ ] Maintenance scheduling optimization

### Technical Improvements
- [ ] Kubernetes deployment manifests
- [ ] Advanced monitoring with Prometheus/Grafana
- [ ] Load balancing and auto-scaling
- [ ] Database sharding for scalability

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NASA for the Turbofan Engine Degradation Simulation Dataset
- The open-source community for the amazing tools and libraries
- Industrial IoT community for domain expertise and best practices