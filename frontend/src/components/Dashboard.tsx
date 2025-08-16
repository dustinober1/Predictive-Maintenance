import React from 'react'
import MetricCard from './MetricCard'
import EngineGrid from './EngineGrid'
import AlertsList from './AlertsList'
import './Dashboard.css'

interface DashboardProps {
  data: {
    summary: {
      total_engines: number
      active_engines: number
      critical_alerts: number
      avg_predicted_rul: number
      last_update: number
    }
    recent_data: {
      sensor_readings: number
      alerts: number
      predictions: number
    }
  } | null
}

const Dashboard: React.FC<DashboardProps> = ({ data }) => {
  if (!data) {
    return (
      <div className="dashboard">
        <div className="loading-state">
          <h2>Loading Dashboard...</h2>
        </div>
      </div>
    )
  }

  const { summary, recent_data } = data

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Fleet Overview</h1>
        <p className="last-update">
          Last updated: {new Date(summary.last_update * 1000).toLocaleTimeString()}
        </p>
      </div>

      <div className="metrics-grid">
        <MetricCard
          title="Total Engines"
          value={summary.total_engines}
          unit=""
          status="neutral"
          icon="⚙️"
        />
        <MetricCard
          title="Active Engines"
          value={summary.active_engines}
          unit=""
          status={summary.active_engines > 0 ? "good" : "warning"}
          icon="🟢"
        />
        <MetricCard
          title="Critical Alerts"
          value={summary.critical_alerts}
          unit=""
          status={summary.critical_alerts === 0 ? "good" : "critical"}
          icon="🚨"
        />
        <MetricCard
          title="Avg Predicted RUL"
          value={Math.round(summary.avg_predicted_rul)}
          unit="cycles"
          status={summary.avg_predicted_rul > 100 ? "good" : summary.avg_predicted_rul > 50 ? "warning" : "critical"}
          icon="⏱️"
        />
      </div>

      <div className="dashboard-content">
        <div className="dashboard-section">
          <h2>Engine Fleet Status</h2>
          <EngineGrid />
        </div>

        <div className="dashboard-section">
          <h2>Recent Activity</h2>
          <div className="activity-metrics">
            <div className="activity-item">
              <span className="activity-label">Sensor Readings</span>
              <span className="activity-value">{recent_data.sensor_readings}</span>
            </div>
            <div className="activity-item">
              <span className="activity-label">Alerts Generated</span>
              <span className="activity-value">{recent_data.alerts}</span>
            </div>
            <div className="activity-item">
              <span className="activity-label">Predictions Made</span>
              <span className="activity-value">{recent_data.predictions}</span>
            </div>
          </div>
        </div>

        <div className="dashboard-section">
          <h2>Latest Alerts</h2>
          <AlertsList />
        </div>
      </div>
    </div>
  )
}

export default Dashboard