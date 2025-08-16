import React, { useState, useEffect } from 'react'
import './EngineGrid.css'

interface Engine {
  id: number
  name: string
  status: 'healthy' | 'warning' | 'critical'
  rul: number
  lastMaintenance: string
  efficiency: number
}

const EngineGrid: React.FC = () => {
  const [engines] = useState<Engine[]>([
    { id: 1, name: 'Engine A1', status: 'healthy', rul: 245, lastMaintenance: '2024-01-15', efficiency: 95 },
    { id: 2, name: 'Engine B2', status: 'warning', rul: 67, lastMaintenance: '2024-01-10', efficiency: 82 },
    { id: 3, name: 'Engine C3', status: 'healthy', rul: 189, lastMaintenance: '2024-01-20', efficiency: 91 },
    { id: 4, name: 'Engine D4', status: 'critical', rul: 23, lastMaintenance: '2024-01-05', efficiency: 68 },
    { id: 5, name: 'Engine E5', status: 'healthy', rul: 312, lastMaintenance: '2024-01-18', efficiency: 97 },
    { id: 6, name: 'Engine F6', status: 'warning', rul: 89, lastMaintenance: '2024-01-12', efficiency: 79 },
  ])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#2ecc71'
      case 'warning': return '#f39c12'
      case 'critical': return '#e74c3c'
      default: return '#95a5a6'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return '✅'
      case 'warning': return '⚠️'
      case 'critical': return '🚨'
      default: return '❓'
    }
  }

  return (
    <div className="engine-grid">
      {engines.map((engine) => (
        <div key={engine.id} className={`engine-card ${engine.status}`}>
          <div className="engine-header">
            <h3 className="engine-name">{engine.name}</h3>
            <span className="engine-status-icon">
              {getStatusIcon(engine.status)}
            </span>
          </div>
          
          <div className="engine-metrics">
            <div className="metric">
              <span className="metric-label">RUL</span>
              <span className="metric-value">
                {engine.rul} 
                <span className="metric-unit">cycles</span>
              </span>
            </div>
            
            <div className="metric">
              <span className="metric-label">Efficiency</span>
              <span className="metric-value">
                {engine.efficiency}
                <span className="metric-unit">%</span>
              </span>
            </div>
          </div>
          
          <div className="rul-progress">
            <div className="progress-label">
              <span>Remaining Life</span>
              <span>{engine.rul} cycles</span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ 
                  width: `${Math.min(100, (engine.rul / 300) * 100)}%`,
                  backgroundColor: getStatusColor(engine.status)
                }}
              />
            </div>
          </div>
          
          <div className="engine-footer">
            <small className="last-maintenance">
              Last maintenance: {engine.lastMaintenance}
            </small>
          </div>
        </div>
      ))}
    </div>
  )
}

export default EngineGrid