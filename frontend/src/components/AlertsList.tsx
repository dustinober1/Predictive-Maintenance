import React, { useState, useEffect } from 'react'
import './AlertsList.css'

interface Alert {
  id: number
  type: 'anomaly' | 'maintenance' | 'critical'
  engine: string
  message: string
  timestamp: string
  severity: 'low' | 'medium' | 'high'
}

const AlertsList: React.FC = () => {
  const [alerts] = useState<Alert[]>([
    {
      id: 1,
      type: 'critical',
      engine: 'Engine D4',
      message: 'Critical temperature threshold exceeded',
      timestamp: '2024-01-16 14:32:15',
      severity: 'high'
    },
    {
      id: 2,
      type: 'anomaly',
      engine: 'Engine B2',
      message: 'Vibration pattern anomaly detected',
      timestamp: '2024-01-16 14:28:42',
      severity: 'medium'
    },
    {
      id: 3,
      type: 'maintenance',
      engine: 'Engine F6',
      message: 'Scheduled maintenance due in 5 days',
      timestamp: '2024-01-16 14:15:33',
      severity: 'low'
    },
    {
      id: 4,
      type: 'anomaly',
      engine: 'Engine C3',
      message: 'Pressure sensor reading outside normal range',
      timestamp: '2024-01-16 13:58:21',
      severity: 'medium'
    },
    {
      id: 5,
      type: 'maintenance',
      engine: 'Engine A1',
      message: 'Oil change recommended',
      timestamp: '2024-01-16 13:45:18',
      severity: 'low'
    }
  ])

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'critical': return '🚨'
      case 'anomaly': return '⚠️'
      case 'maintenance': return '🔧'
      default: return 'ℹ️'
    }
  }

  const getSeverityClass = (severity: string) => {
    switch (severity) {
      case 'high': return 'severity-high'
      case 'medium': return 'severity-medium'
      case 'low': return 'severity-low'
      default: return 'severity-low'
    }
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    
    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="alerts-list">
      {alerts.length === 0 ? (
        <div className="no-alerts">
          <span className="no-alerts-icon">✅</span>
          <p>No recent alerts</p>
        </div>
      ) : (
        <div className="alerts-container">
          {alerts.map((alert) => (
            <div key={alert.id} className={`alert-item ${getSeverityClass(alert.severity)}`}>
              <div className="alert-icon">
                {getAlertIcon(alert.type)}
              </div>
              
              <div className="alert-content">
                <div className="alert-header">
                  <span className="alert-engine">{alert.engine}</span>
                  <span className="alert-timestamp">
                    {formatTimestamp(alert.timestamp)}
                  </span>
                </div>
                
                <p className="alert-message">{alert.message}</p>
                
                <div className="alert-footer">
                  <span className={`alert-type type-${alert.type}`}>
                    {alert.type.charAt(0).toUpperCase() + alert.type.slice(1)}
                  </span>
                  <span className={`alert-severity ${getSeverityClass(alert.severity)}`}>
                    {alert.severity.charAt(0).toUpperCase() + alert.severity.slice(1)}
                  </span>
                </div>
              </div>
              
              <div className="alert-actions">
                <button className="action-btn view">View</button>
                <button className="action-btn dismiss">Dismiss</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default AlertsList