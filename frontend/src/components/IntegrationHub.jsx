import React, { useState, useEffect, useCallback } from 'react'
import { LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const API_BASE = 'http://localhost:8010/api'

// Integration type definitions
const INTEGRATION_TYPES = {
  iot: {
    name: 'IoT Devices',
    icon: '📡',
    description: 'EEG headsets, wearables, sensors',
    color: '#4CAF50'
  },
  whatsapp: {
    name: 'WhatsApp',
    icon: '💬',
    description: 'Alerts, reports, notifications',
    color: '#25D366'
  },
  emotive: {
    name: 'Emotive',
    icon: '🧠',
    description: 'EPOC X, EPOC Flex, Insight, MN8',
    color: '#9C27B0'
  },
  fpga: {
    name: 'FPGA',
    icon: '⚡',
    description: 'Hardware acceleration',
    color: '#FF5722'
  },
  pcb: {
    name: 'PCB',
    icon: '🔌',
    description: 'Custom EEG acquisition',
    color: '#795548'
  },
  mobile: {
    name: 'Mobile',
    icon: '📱',
    description: 'iOS, Android, React Native',
    color: '#2196F3'
  }
}

// Simulation configs
const EMOTIVE_DEVICES = ['EPOC X', 'EPOC Flex', 'Insight', 'MN8']
const FPGA_BOARDS = ['PYNQ-Z2', 'DE10-Nano', 'Zynq UltraScale+', 'Cyclone V']
const PCB_PROTOCOLS = ['UART', 'SPI', 'I2C', 'USB', 'Bluetooth']
const MOBILE_PLATFORMS = ['iOS', 'Android', 'React Native', 'Flutter']

function IntegrationHub() {
  const [activeIntegration, setActiveIntegration] = useState('iot')
  const [integrationStatus, setIntegrationStatus] = useState({})
  const [simulationRunning, setSimulationRunning] = useState(false)
  const [simulationData, setSimulationData] = useState([])
  const [logs, setLogs] = useState([])
  const [configForm, setConfigForm] = useState({})
  const [devices, setDevices] = useState([])
  const [selectedDevice, setSelectedDevice] = useState(null)
  const [eegData, setEegData] = useState([])
  const [metricsData, setMetricsData] = useState([])

  // Add log entry
  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev.slice(-49), { timestamp, message, type }])
  }, [])

  // Fetch integration status
  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/integration/status`)
      const data = await response.json()
      setIntegrationStatus(data)
    } catch (error) {
      // Use simulated status
      setIntegrationStatus({
        iot: { connected: 2, available: 5 },
        whatsapp: { connected: true, messages_sent: 127 },
        emotive: { connected: 1, streaming: false },
        fpga: { connected: true, accelerators: 3 },
        pcb: { connected: 1, calibrated: true },
        mobile: { users: 45, active: 12 }
      })
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 10000)
    return () => clearInterval(interval)
  }, [])

  // Simulate EEG data streaming
  const simulateEEGData = useCallback(() => {
    const t = Date.now() / 1000
    const channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2']

    const newSample = channels.map((ch, i) => {
      const alpha = 15 * Math.sin(2 * Math.PI * 10 * t + i * 0.5)
      const beta = 8 * Math.sin(2 * Math.PI * 20 * t + i * 0.3)
      const theta = 10 * Math.sin(2 * Math.PI * 6 * t + i * 0.7)
      const noise = 3 * (Math.random() - 0.5)
      return { channel: ch, value: alpha + beta + theta + noise }
    })

    setEegData(prev => {
      const updated = [...prev, { time: t % 10, ...Object.fromEntries(newSample.map(s => [s.channel, s.value])) }]
      return updated.slice(-100)
    })
  }, [])

  // Simulate performance metrics
  const simulateMetrics = useCallback(() => {
    setMetricsData([
      { metric: 'Engagement', value: 0.6 + Math.random() * 0.3 },
      { metric: 'Focus', value: 0.5 + Math.random() * 0.4 },
      { metric: 'Stress', value: 0.3 + Math.random() * 0.3 },
      { metric: 'Relaxation', value: 0.4 + Math.random() * 0.4 },
      { metric: 'Interest', value: 0.5 + Math.random() * 0.3 },
      { metric: 'Excitement', value: 0.4 + Math.random() * 0.4 }
    ])
  }, [])

  // Start simulation
  const startSimulation = () => {
    setSimulationRunning(true)
    addLog(`Started ${INTEGRATION_TYPES[activeIntegration].name} simulation`, 'success')

    // Simulate different integrations
    const interval = setInterval(() => {
      if (activeIntegration === 'emotive' || activeIntegration === 'pcb' || activeIntegration === 'iot') {
        simulateEEGData()
        simulateMetrics()
      }

      // Add random simulation events
      const events = [
        'Data packet received',
        'Signal quality: Good',
        'Processing complete',
        'Buffer flushed',
        'Sync successful'
      ]
      if (Math.random() > 0.7) {
        addLog(events[Math.floor(Math.random() * events.length)])
      }
    }, 50)

    return () => clearInterval(interval)
  }

  // Stop simulation
  const stopSimulation = () => {
    setSimulationRunning(false)
    addLog('Simulation stopped', 'warning')
  }

  // Handle device discovery
  const discoverDevices = () => {
    addLog('Discovering devices...', 'info')

    setTimeout(() => {
      const mockDevices = {
        iot: [
          { id: 'EEG-001', name: 'NeuroAI EEG Headset', status: 'available', battery: 85 },
          { id: 'HR-002', name: 'Heart Rate Monitor', status: 'connected', battery: 72 },
          { id: 'SLEEP-003', name: 'Sleep Tracker', status: 'available', battery: 90 }
        ],
        emotive: [
          { id: 'EPOC-X-001', name: 'EPOC X (14 channels)', status: 'available', battery: 78 },
          { id: 'INSIGHT-001', name: 'Insight (5 channels)', status: 'connected', battery: 65 }
        ],
        fpga: [
          { id: 'PYNQ-001', name: 'PYNQ-Z2 Board', status: 'ready', temp: 42 },
          { id: 'DE10-001', name: 'DE10-Nano', status: 'available', temp: 38 }
        ],
        pcb: [
          { id: 'PCB-EEG-001', name: '8-Channel EEG Board', status: 'connected', sampling: 256 },
          { id: 'PCB-EEG-002', name: '16-Channel EEG Board', status: 'available', sampling: 512 }
        ],
        mobile: [
          { id: 'ios-001', name: 'iPhone 15 Pro', platform: 'iOS', version: '2.1.0' },
          { id: 'android-001', name: 'Pixel 8', platform: 'Android', version: '2.1.0' }
        ]
      }

      setDevices(mockDevices[activeIntegration] || [])
      addLog(`Found ${(mockDevices[activeIntegration] || []).length} devices`, 'success')
    }, 1000)
  }

  // Connect to device
  const connectDevice = (deviceId) => {
    addLog(`Connecting to ${deviceId}...`, 'info')

    setTimeout(() => {
      setDevices(prev => prev.map(d =>
        d.id === deviceId ? { ...d, status: 'connected' } : d
      ))
      setSelectedDevice(deviceId)
      addLog(`Connected to ${deviceId}`, 'success')
    }, 500)
  }

  // Disconnect device
  const disconnectDevice = (deviceId) => {
    addLog(`Disconnecting ${deviceId}...`, 'info')

    setTimeout(() => {
      setDevices(prev => prev.map(d =>
        d.id === deviceId ? { ...d, status: 'available' } : d
      ))
      if (selectedDevice === deviceId) setSelectedDevice(null)
      addLog(`Disconnected ${deviceId}`, 'warning')
    }, 300)
  }

  // Send test notification (WhatsApp/Mobile)
  const sendTestNotification = () => {
    addLog('Sending test notification...', 'info')

    setTimeout(() => {
      addLog('Notification sent successfully', 'success')
      setSimulationData(prev => [...prev, {
        time: new Date().toLocaleTimeString(),
        type: 'notification',
        status: 'delivered'
      }])
    }, 500)
  }

  // Run FPGA processing
  const runFPGAProcessing = () => {
    addLog('Running FPGA signal processing...', 'info')

    setTimeout(() => {
      addLog('FFT accelerator: 10μs', 'success')
      addLog('FIR filter: 8μs', 'success')
      addLog('CNN inference: 1.2ms', 'success')
      addLog('Total processing: 1.218ms', 'success')
    }, 800)
  }

  // Calibrate PCB
  const calibratePCB = () => {
    addLog('Starting PCB calibration...', 'info')

    setTimeout(() => {
      addLog('Channel offsets measured', 'info')
      addLog('Gain calibration complete', 'info')
      addLog('Noise floor: 0.5μV RMS', 'success')
      addLog('Calibration complete', 'success')
    }, 1500)
  }

  // Render integration-specific content
  const renderIntegrationContent = () => {
    switch (activeIntegration) {
      case 'iot':
        return renderIoTContent()
      case 'whatsapp':
        return renderWhatsAppContent()
      case 'emotive':
        return renderEmotiveContent()
      case 'fpga':
        return renderFPGAContent()
      case 'pcb':
        return renderPCBContent()
      case 'mobile':
        return renderMobileContent()
      default:
        return null
    }
  }

  // IoT Content
  const renderIoTContent = () => (
    <div className="integration-content">
      <div className="config-section">
        <h4>IoT Device Configuration</h4>
        <div className="config-grid">
          <div className="config-field">
            <label>Device Type</label>
            <select value={configForm.deviceType || ''} onChange={e => setConfigForm({...configForm, deviceType: e.target.value})}>
              <option value="">Select Type</option>
              <option value="eeg_headset">EEG Headset</option>
              <option value="heart_rate">Heart Rate Monitor</option>
              <option value="sleep_tracker">Sleep Tracker</option>
              <option value="activity_tracker">Activity Tracker</option>
            </select>
          </div>
          <div className="config-field">
            <label>Protocol</label>
            <select value={configForm.protocol || ''} onChange={e => setConfigForm({...configForm, protocol: e.target.value})}>
              <option value="">Select Protocol</option>
              <option value="bluetooth">Bluetooth LE</option>
              <option value="wifi">WiFi</option>
              <option value="zigbee">Zigbee</option>
            </select>
          </div>
          <div className="config-field">
            <label>Sampling Rate (Hz)</label>
            <input type="number" placeholder="256" value={configForm.samplingRate || ''} onChange={e => setConfigForm({...configForm, samplingRate: e.target.value})} />
          </div>
        </div>
      </div>
    </div>
  )

  // WhatsApp Content
  const renderWhatsAppContent = () => (
    <div className="integration-content">
      <div className="config-section">
        <h4>WhatsApp Business API Configuration</h4>
        <div className="config-grid">
          <div className="config-field">
            <label>Phone Number ID</label>
            <input type="text" placeholder="Enter Phone Number ID" value={configForm.phoneNumberId || ''} onChange={e => setConfigForm({...configForm, phoneNumberId: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Business Account ID</label>
            <input type="text" placeholder="Enter Business Account ID" value={configForm.businessAccountId || ''} onChange={e => setConfigForm({...configForm, businessAccountId: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Access Token</label>
            <input type="password" placeholder="Enter Access Token" value={configForm.accessToken || ''} onChange={e => setConfigForm({...configForm, accessToken: e.target.value})} />
          </div>
          <div className="config-field">
            <label>API Version</label>
            <input type="text" placeholder="v18.0" value={configForm.apiVersion || 'v18.0'} onChange={e => setConfigForm({...configForm, apiVersion: e.target.value})} />
          </div>
        </div>

        <div className="test-section">
          <h4>Test Notification</h4>
          <div className="config-grid">
            <div className="config-field">
              <label>Recipient Phone</label>
              <input type="text" placeholder="+1234567890" value={configForm.testPhone || ''} onChange={e => setConfigForm({...configForm, testPhone: e.target.value})} />
            </div>
            <div className="config-field">
              <label>Message Template</label>
              <select value={configForm.template || ''} onChange={e => setConfigForm({...configForm, template: e.target.value})}>
                <option value="">Select Template</option>
                <option value="diagnostic_alert">Diagnostic Alert</option>
                <option value="appointment_reminder">Appointment Reminder</option>
                <option value="eeg_report">EEG Report</option>
                <option value="critical_alert">Critical Alert</option>
              </select>
            </div>
          </div>
          <button className="btn-primary" onClick={sendTestNotification}>Send Test Message</button>
        </div>
      </div>
    </div>
  )

  // Emotive Content
  const renderEmotiveContent = () => (
    <div className="integration-content">
      <div className="config-section">
        <h4>Emotiv Cortex API Configuration</h4>
        <div className="config-grid">
          <div className="config-field">
            <label>Client ID</label>
            <input type="text" placeholder="Enter Cortex Client ID" value={configForm.clientId || ''} onChange={e => setConfigForm({...configForm, clientId: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Client Secret</label>
            <input type="password" placeholder="Enter Client Secret" value={configForm.clientSecret || ''} onChange={e => setConfigForm({...configForm, clientSecret: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Device Type</label>
            <select value={configForm.emotiveDevice || ''} onChange={e => setConfigForm({...configForm, emotiveDevice: e.target.value})}>
              <option value="">Select Device</option>
              {EMOTIVE_DEVICES.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <div className="config-field">
            <label>License ID (Optional)</label>
            <input type="text" placeholder="For raw EEG access" value={configForm.licenseId || ''} onChange={e => setConfigForm({...configForm, licenseId: e.target.value})} />
          </div>
        </div>
      </div>

      {simulationRunning && eegData.length > 0 && (
        <div className="visualization-section">
          <h4>Real-time EEG Stream</h4>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={eegData.slice(-50)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" tick={{ fill: '#888' }} />
                <YAxis domain={[-50, 50]} tick={{ fill: '#888' }} />
                <Tooltip contentStyle={{ background: '#1e1e1e', border: '1px solid #333' }} />
                <Legend />
                <Line type="monotone" dataKey="AF3" stroke="#FF6384" dot={false} strokeWidth={1} />
                <Line type="monotone" dataKey="F7" stroke="#36A2EB" dot={false} strokeWidth={1} />
                <Line type="monotone" dataKey="F3" stroke="#FFCE56" dot={false} strokeWidth={1} />
                <Line type="monotone" dataKey="FC5" stroke="#4BC0C0" dot={false} strokeWidth={1} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h4>Performance Metrics</h4>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={250}>
              <RadarChart data={metricsData}>
                <PolarGrid stroke="#333" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: '#888' }} />
                <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#888' }} />
                <Radar name="Metrics" dataKey="value" stroke="#9C27B0" fill="#9C27B0" fillOpacity={0.5} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )

  // FPGA Content
  const renderFPGAContent = () => (
    <div className="integration-content">
      <div className="config-section">
        <h4>FPGA Configuration</h4>
        <div className="config-grid">
          <div className="config-field">
            <label>Board Type</label>
            <select value={configForm.fpgaBoard || ''} onChange={e => setConfigForm({...configForm, fpgaBoard: e.target.value})}>
              <option value="">Select Board</option>
              {FPGA_BOARDS.map(b => <option key={b} value={b}>{b}</option>)}
            </select>
          </div>
          <div className="config-field">
            <label>Interface</label>
            <select value={configForm.fpgaInterface || ''} onChange={e => setConfigForm({...configForm, fpgaInterface: e.target.value})}>
              <option value="">Select Interface</option>
              <option value="pcie">PCIe</option>
              <option value="usb">USB</option>
              <option value="ethernet">Ethernet</option>
              <option value="uart">UART</option>
            </select>
          </div>
          <div className="config-field">
            <label>Clock Frequency (MHz)</label>
            <input type="number" placeholder="100" value={configForm.clockFreq || ''} onChange={e => setConfigForm({...configForm, clockFreq: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Bitstream Path</label>
            <input type="text" placeholder="/path/to/bitstream.bit" value={configForm.bitstreamPath || ''} onChange={e => setConfigForm({...configForm, bitstreamPath: e.target.value})} />
          </div>
        </div>

        <div className="accelerators-section">
          <h4>Available Accelerators</h4>
          <div className="accelerator-grid">
            {['FFT', 'DWT', 'FIR Filter', 'IIR Filter', 'Conv NN', 'LSTM', 'Matrix Mul'].map(accel => (
              <div key={accel} className="accelerator-card">
                <span className="accel-icon">⚡</span>
                <span className="accel-name">{accel}</span>
                <span className="accel-status">Ready</span>
              </div>
            ))}
          </div>
          <button className="btn-primary" onClick={runFPGAProcessing}>Run Processing Test</button>
        </div>
      </div>
    </div>
  )

  // PCB Content
  const renderPCBContent = () => (
    <div className="integration-content">
      <div className="config-section">
        <h4>PCB Configuration</h4>
        <div className="config-grid">
          <div className="config-field">
            <label>Protocol</label>
            <select value={configForm.pcbProtocol || ''} onChange={e => setConfigForm({...configForm, pcbProtocol: e.target.value})}>
              <option value="">Select Protocol</option>
              {PCB_PROTOCOLS.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="config-field">
            <label>Serial Port</label>
            <input type="text" placeholder="/dev/ttyUSB0" value={configForm.serialPort || ''} onChange={e => setConfigForm({...configForm, serialPort: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Baud Rate</label>
            <select value={configForm.baudRate || ''} onChange={e => setConfigForm({...configForm, baudRate: e.target.value})}>
              <option value="">Select Rate</option>
              <option value="9600">9600</option>
              <option value="115200">115200</option>
              <option value="921600">921600</option>
              <option value="2000000">2000000</option>
            </select>
          </div>
          <div className="config-field">
            <label>Channels</label>
            <select value={configForm.pcbChannels || ''} onChange={e => setConfigForm({...configForm, pcbChannels: e.target.value})}>
              <option value="">Select Channels</option>
              <option value="8">8 Channels</option>
              <option value="16">16 Channels</option>
              <option value="32">32 Channels</option>
            </select>
          </div>
        </div>

        <div className="adc-config">
          <h4>ADC Configuration</h4>
          <div className="config-grid">
            <div className="config-field">
              <label>Resolution</label>
              <select value={configForm.adcResolution || ''} onChange={e => setConfigForm({...configForm, adcResolution: e.target.value})}>
                <option value="">Select Resolution</option>
                <option value="12">12-bit</option>
                <option value="16">16-bit</option>
                <option value="24">24-bit</option>
                <option value="32">32-bit</option>
              </select>
            </div>
            <div className="config-field">
              <label>Sampling Rate (Hz)</label>
              <input type="number" placeholder="256" value={configForm.adcSamplingRate || ''} onChange={e => setConfigForm({...configForm, adcSamplingRate: e.target.value})} />
            </div>
            <div className="config-field">
              <label>PGA Gain</label>
              <select value={configForm.adcGain || ''} onChange={e => setConfigForm({...configForm, adcGain: e.target.value})}>
                <option value="">Select Gain</option>
                <option value="1">1x</option>
                <option value="2">2x</option>
                <option value="4">4x</option>
                <option value="8">8x</option>
                <option value="12">12x</option>
                <option value="24">24x</option>
              </select>
            </div>
            <div className="config-field">
              <label>Reference Voltage (V)</label>
              <input type="number" step="0.1" placeholder="2.5" value={configForm.adcVref || ''} onChange={e => setConfigForm({...configForm, adcVref: e.target.value})} />
            </div>
          </div>
          <button className="btn-secondary" onClick={calibratePCB}>Run Calibration</button>
        </div>
      </div>

      {simulationRunning && eegData.length > 0 && (
        <div className="visualization-section">
          <h4>PCB Data Stream</h4>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={eegData.slice(-50)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="time" tick={{ fill: '#888' }} />
                <YAxis domain={[-50, 50]} tick={{ fill: '#888' }} />
                <Tooltip contentStyle={{ background: '#1e1e1e', border: '1px solid #333' }} />
                <Legend />
                <Line type="monotone" dataKey="AF3" stroke="#FF6384" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="F7" stroke="#36A2EB" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="F3" stroke="#FFCE56" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="FC5" stroke="#4BC0C0" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="T7" stroke="#9966FF" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="P7" stroke="#FF9F40" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="O1" stroke="#FF6384" dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="O2" stroke="#C9CBCF" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )

  // Mobile Content
  const renderMobileContent = () => (
    <div className="integration-content">
      <div className="config-section">
        <h4>Mobile Push Configuration</h4>
        <div className="config-grid">
          <div className="config-field">
            <label>Push Provider</label>
            <select value={configForm.pushProvider || ''} onChange={e => setConfigForm({...configForm, pushProvider: e.target.value})}>
              <option value="">Select Provider</option>
              <option value="apns">APNS (iOS)</option>
              <option value="fcm">FCM (Android)</option>
              <option value="onesignal">OneSignal</option>
            </select>
          </div>
          <div className="config-field">
            <label>API Key</label>
            <input type="password" placeholder="Enter API Key" value={configForm.pushApiKey || ''} onChange={e => setConfigForm({...configForm, pushApiKey: e.target.value})} />
          </div>
          <div className="config-field">
            <label>App ID</label>
            <input type="text" placeholder="com.neuroai.app" value={configForm.appId || ''} onChange={e => setConfigForm({...configForm, appId: e.target.value})} />
          </div>
          <div className="config-field">
            <label>Platform</label>
            <select value={configForm.mobilePlatform || ''} onChange={e => setConfigForm({...configForm, mobilePlatform: e.target.value})}>
              <option value="">Select Platform</option>
              {MOBILE_PLATFORMS.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
        </div>

        <div className="test-section">
          <h4>Test Push Notification</h4>
          <div className="config-grid">
            <div className="config-field">
              <label>Title</label>
              <input type="text" placeholder="Notification Title" value={configForm.notifTitle || ''} onChange={e => setConfigForm({...configForm, notifTitle: e.target.value})} />
            </div>
            <div className="config-field">
              <label>Body</label>
              <input type="text" placeholder="Notification Body" value={configForm.notifBody || ''} onChange={e => setConfigForm({...configForm, notifBody: e.target.value})} />
            </div>
          </div>
          <button className="btn-primary" onClick={sendTestNotification}>Send Test Push</button>
        </div>
      </div>
    </div>
  )

  return (
    <div className="integration-hub">
      <div className="hub-header">
        <h2>Integration Hub</h2>
        <p>Connect and test IoT, WhatsApp, Emotive, FPGA, PCB, and Mobile integrations</p>
      </div>

      {/* Integration Type Selector */}
      <div className="integration-selector">
        {Object.entries(INTEGRATION_TYPES).map(([key, info]) => (
          <button
            key={key}
            className={`integration-btn ${activeIntegration === key ? 'active' : ''}`}
            onClick={() => {
              setActiveIntegration(key)
              setSimulationRunning(false)
              setEegData([])
              setDevices([])
            }}
            style={{ '--accent-color': info.color }}
          >
            <span className="integration-icon">{info.icon}</span>
            <span className="integration-name">{info.name}</span>
          </button>
        ))}
      </div>

      {/* Main Content Area */}
      <div className="hub-content">
        {/* Left Panel - Configuration */}
        <div className="config-panel">
          <div className="panel-header">
            <span className="panel-icon">{INTEGRATION_TYPES[activeIntegration].icon}</span>
            <h3>{INTEGRATION_TYPES[activeIntegration].name}</h3>
          </div>
          <p className="panel-description">{INTEGRATION_TYPES[activeIntegration].description}</p>

          {renderIntegrationContent()}

          {/* Action Buttons */}
          <div className="action-buttons">
            <button className="btn-discover" onClick={discoverDevices}>
              🔍 Discover Devices
            </button>
            {!simulationRunning ? (
              <button className="btn-start" onClick={startSimulation}>
                ▶️ Start Simulation
              </button>
            ) : (
              <button className="btn-stop" onClick={stopSimulation}>
                ⏹️ Stop Simulation
              </button>
            )}
          </div>
        </div>

        {/* Right Panel - Devices & Logs */}
        <div className="monitor-panel">
          {/* Devices List */}
          <div className="devices-section">
            <h4>Discovered Devices</h4>
            <div className="devices-list">
              {devices.length === 0 ? (
                <p className="no-devices">No devices discovered. Click "Discover Devices" to scan.</p>
              ) : (
                devices.map(device => (
                  <div key={device.id} className={`device-card ${device.status}`}>
                    <div className="device-info">
                      <span className="device-name">{device.name}</span>
                      <span className="device-id">{device.id}</span>
                      {device.battery !== undefined && (
                        <span className="device-battery">🔋 {device.battery}%</span>
                      )}
                      {device.temp !== undefined && (
                        <span className="device-temp">🌡️ {device.temp}°C</span>
                      )}
                    </div>
                    <div className="device-actions">
                      <span className={`status-badge ${device.status}`}>{device.status}</span>
                      {device.status === 'available' || device.status === 'ready' ? (
                        <button className="btn-sm btn-connect" onClick={() => connectDevice(device.id)}>
                          Connect
                        </button>
                      ) : (
                        <button className="btn-sm btn-disconnect" onClick={() => disconnectDevice(device.id)}>
                          Disconnect
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Activity Logs */}
          <div className="logs-section">
            <h4>Activity Log</h4>
            <div className="logs-container">
              {logs.length === 0 ? (
                <p className="no-logs">No activity yet.</p>
              ) : (
                logs.map((log, idx) => (
                  <div key={idx} className={`log-entry ${log.type}`}>
                    <span className="log-time">{log.timestamp}</span>
                    <span className="log-message">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Status Bar */}
      <div className="status-bar">
        <div className="status-item">
          <span className="status-label">IoT:</span>
          <span className="status-value">{integrationStatus.iot?.connected || 0} connected</span>
        </div>
        <div className="status-item">
          <span className="status-label">WhatsApp:</span>
          <span className="status-value">{integrationStatus.whatsapp?.messages_sent || 0} sent</span>
        </div>
        <div className="status-item">
          <span className="status-label">Emotive:</span>
          <span className="status-value">{integrationStatus.emotive?.connected || 0} devices</span>
        </div>
        <div className="status-item">
          <span className="status-label">FPGA:</span>
          <span className="status-value">{integrationStatus.fpga?.accelerators || 0} accelerators</span>
        </div>
        <div className="status-item">
          <span className="status-label">Mobile:</span>
          <span className="status-value">{integrationStatus.mobile?.active || 0} active</span>
        </div>
      </div>
    </div>
  )
}

export default IntegrationHub
