import React, { useState } from 'react'
import './Sidebar.css'

const Sidebar: React.FC = () => {
  const [activeItem, setActiveItem] = useState('dashboard')

  const menuItems = [
    { id: 'dashboard', label: 'Dashboard', icon: '📊' },
    { id: 'engines', label: 'Engines', icon: '⚙️' },
    { id: 'alerts', label: 'Alerts', icon: '🚨' },
    { id: 'predictions', label: 'Predictions', icon: '📈' },
    { id: 'analytics', label: 'Analytics', icon: '📉' },
    { id: 'maintenance', label: 'Maintenance', icon: '🔧' },
    { id: 'reports', label: 'Reports', icon: '📋' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
  ]

  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        <ul className="nav-list">
          {menuItems.map((item) => (
            <li key={item.id} className="nav-item">
              <button
                className={`nav-link ${activeItem === item.id ? 'active' : ''}`}
                onClick={() => setActiveItem(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span className="nav-label">{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="sidebar-footer">
        <div className="system-status">
          <div className="status-item">
            <span className="status-label">System Health</span>
            <span className="status-value good">Good</span>
          </div>
          <div className="status-item">
            <span className="status-label">API Status</span>
            <span className="status-value online">Online</span>
          </div>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar