import React from 'react'
import './MetricCard.css'

interface MetricCardProps {
  title: string
  value: number
  unit: string
  status: 'good' | 'warning' | 'critical' | 'neutral'
  icon: string
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, unit, status, icon }) => {
  return (
    <div className={`metric-card ${status}`}>
      <div className="metric-header">
        <span className="metric-icon">{icon}</span>
        <span className="metric-title">{title}</span>
      </div>
      
      <div className="metric-value">
        <span className="value">{value.toLocaleString()}</span>
        {unit && <span className="unit">{unit}</span>}
      </div>
      
      <div className={`metric-status ${status}`}>
        <div className="status-indicator"></div>
        <span className="status-text">
          {status === 'good' && 'Healthy'}
          {status === 'warning' && 'Warning'}
          {status === 'critical' && 'Critical'}
          {status === 'neutral' && 'Normal'}
        </span>
      </div>
    </div>
  )
}

export default MetricCard