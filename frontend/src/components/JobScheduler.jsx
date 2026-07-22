import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function JobScheduler() {
  const [jobs, setJobs] = useState([])
  const [queuedJobs, setQueuedJobs] = useState([])
  const [runningJobs, setRunningJobs] = useState([])
  const [schedulerStatus, setSchedulerStatus] = useState(null)
  const [activeView, setActiveView] = useState('dashboard')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedJob, setSelectedJob] = useState(null)

  // Job creation form
  const [jobForm, setJobForm] = useState({
    name: '',
    description: '',
    priority: 'normal',
    schedule_type: 'immediate',
    scheduled_time: '',
    pipeline_config: {
      pipeline_type: 'full_pipeline',
      data_modality: 'eeg',
      model_type: 'cnn',
      task_type: 'classification',
      epochs: 100,
      batch_size: 32
    },
    resources: {
      cpu_cores: 2,
      gpu_count: 0,
      memory_gb: 4
    },
    tags: []
  })

  useEffect(() => {
    fetchAll()
    const interval = setInterval(fetchAll, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchAll = async () => {
    await Promise.all([
      fetchSchedulerStatus(),
      fetchJobs(),
      fetchQueuedJobs(),
      fetchRunningJobs()
    ])
  }

  const fetchSchedulerStatus = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/jobs/status`)
      setSchedulerStatus(response.data)
    } catch (err) {
      setSchedulerStatus({
        available: true,
        is_running: false,
        statistics: {
          total_jobs_submitted: 0,
          total_jobs_completed: 0,
          total_jobs_failed: 0,
          queued_jobs: 0,
          running_jobs: 0
        },
        resources: {
          cpu: { total: 8, available: 8 },
          gpu: { total: 1, available: 1 },
          memory_gb: { total: 32, available: 32 }
        }
      })
    }
  }

  const fetchJobs = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/jobs/list/all`)
      setJobs(response.data.jobs || [])
    } catch (err) {
      console.error('Error fetching jobs:', err)
    }
  }

  const fetchQueuedJobs = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/jobs/list/queued`)
      setQueuedJobs(response.data.jobs || [])
    } catch (err) {
      console.error('Error fetching queued jobs:', err)
    }
  }

  const fetchRunningJobs = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/jobs/list/running`)
      setRunningJobs(response.data.jobs || [])
    } catch (err) {
      console.error('Error fetching running jobs:', err)
    }
  }

  const startScheduler = async () => {
    try {
      await axios.post(`${API_URL}/api/jobs/start-scheduler`)
      await fetchSchedulerStatus()
    } catch (err) {
      console.error('Error starting scheduler:', err)
    }
  }

  const stopScheduler = async () => {
    try {
      await axios.post(`${API_URL}/api/jobs/stop-scheduler`)
      await fetchSchedulerStatus()
    } catch (err) {
      console.error('Error stopping scheduler:', err)
    }
  }

  const createJob = async () => {
    setIsLoading(true)
    try {
      const response = await axios.post(`${API_URL}/api/jobs/create`, jobForm)
      if (response.data.success) {
        await fetchAll()
        setActiveView('dashboard')
        resetForm()
      }
    } catch (err) {
      console.error('Error creating job:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const cancelJob = async (jobId) => {
    try {
      await axios.delete(`${API_URL}/api/jobs/${jobId}`)
      await fetchAll()
    } catch (err) {
      console.error('Error cancelling job:', err)
    }
  }

  const pauseJob = async (jobId) => {
    try {
      await axios.post(`${API_URL}/api/jobs/${jobId}/pause`)
      await fetchAll()
    } catch (err) {
      console.error('Error pausing job:', err)
    }
  }

  const resumeJob = async (jobId) => {
    try {
      await axios.post(`${API_URL}/api/jobs/${jobId}/resume`)
      await fetchAll()
    } catch (err) {
      console.error('Error resuming job:', err)
    }
  }

  const updatePriority = async (jobId, priority) => {
    try {
      await axios.put(`${API_URL}/api/jobs/${jobId}/priority?priority=${priority}`)
      await fetchAll()
    } catch (err) {
      console.error('Error updating priority:', err)
    }
  }

  const resetForm = () => {
    setJobForm({
      name: '',
      description: '',
      priority: 'normal',
      schedule_type: 'immediate',
      scheduled_time: '',
      pipeline_config: {
        pipeline_type: 'full_pipeline',
        data_modality: 'eeg',
        model_type: 'cnn',
        task_type: 'classification',
        epochs: 100,
        batch_size: 32
      },
      resources: {
        cpu_cores: 2,
        gpu_count: 0,
        memory_gb: 4
      },
      tags: []
    })
  }

  const getStatusColor = (status) => {
    const colors = {
      pending: '#ff9800',
      queued: '#1e88e5',
      running: '#7c4dff',
      completed: '#4caf50',
      failed: '#f44336',
      cancelled: '#9e9e9e',
      paused: '#00bcd4'
    }
    return colors[status] || '#9e9e9e'
  }

  const renderDashboard = () => {
    const stats = schedulerStatus?.statistics || {}
    const resources = schedulerStatus?.resources || {}

    const jobStatusData = [
      { name: 'Completed', value: stats.total_jobs_completed || 0, color: '#4caf50' },
      { name: 'Failed', value: stats.total_jobs_failed || 0, color: '#f44336' },
      { name: 'Queued', value: stats.queued_jobs || 0, color: '#1e88e5' },
      { name: 'Running', value: stats.running_jobs || 0, color: '#7c4dff' }
    ].filter(d => d.value > 0)

    const resourceData = [
      { name: 'CPU', used: (resources.cpu?.total || 8) - (resources.cpu?.available || 8), total: resources.cpu?.total || 8 },
      { name: 'GPU', used: (resources.gpu?.total || 1) - (resources.gpu?.available || 1), total: resources.gpu?.total || 1 },
      { name: 'Memory', used: (resources.memory_gb?.total || 32) - (resources.memory_gb?.available || 32), total: resources.memory_gb?.total || 32 }
    ]

    return (
      <div className="scheduler-dashboard">
        <div className="dashboard-header">
          <h2>Job Scheduler Dashboard</h2>
          <div className="scheduler-controls">
            <span className={`scheduler-status ${schedulerStatus?.is_running ? 'running' : 'stopped'}`}>
              {schedulerStatus?.is_running ? 'Scheduler Running' : 'Scheduler Stopped'}
            </span>
            {schedulerStatus?.is_running ? (
              <button className="btn btn-danger" onClick={stopScheduler}>Stop Scheduler</button>
            ) : (
              <button className="btn btn-success" onClick={startScheduler}>Start Scheduler</button>
            )}
            <button className="btn btn-primary" onClick={() => setActiveView('create')}>
              + Schedule Job
            </button>
          </div>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Total Jobs</div>
            <div className="metric-value">{stats.total_jobs_submitted || 0}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Completed</div>
            <div className="metric-value" style={{ color: '#4caf50' }}>{stats.total_jobs_completed || 0}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Failed</div>
            <div className="metric-value" style={{ color: '#f44336' }}>{stats.total_jobs_failed || 0}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Running</div>
            <div className="metric-value" style={{ color: '#7c4dff' }}>{stats.running_jobs || 0}</div>
          </div>
        </div>

        <div className="charts-grid">
          <div className="chart-card">
            <div className="chart-title">Job Status Distribution</div>
            {jobStatusData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={jobStatusData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {jobStatusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-chart">No jobs to display</div>
            )}
          </div>

          <div className="chart-card">
            <div className="chart-title">Resource Usage</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={resourceData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis type="category" dataKey="name" stroke="#94a3b8" />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155' }} />
                <Bar dataKey="used" name="Used" fill="#1e88e5" stackId="a" />
                <Bar dataKey="total" name="Total" fill="#334155" stackId="b" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="job-section">
          <h3>Running Jobs ({runningJobs.length})</h3>
          <div className="job-list">
            {runningJobs.length === 0 ? (
              <div className="empty-state small">No running jobs</div>
            ) : (
              runningJobs.map((job, i) => (
                <div key={i} className="job-card running">
                  <div className="job-header">
                    <span className="job-name">{job.name}</span>
                    <span className="job-priority" style={{ background: getPriorityColor(job.priority) }}>
                      P{job.priority}
                    </span>
                  </div>
                  <div className="job-progress">
                    <div className="progress-bar">
                      <div className="progress-fill running" style={{ width: '60%' }} />
                    </div>
                  </div>
                  <div className="job-actions">
                    <button className="btn btn-sm btn-warning" onClick={() => pauseJob(job.job_id)}>Pause</button>
                    <button className="btn btn-sm btn-danger" onClick={() => cancelJob(job.job_id)}>Cancel</button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="job-section">
          <h3>Queued Jobs ({queuedJobs.length})</h3>
          <div className="job-list">
            {queuedJobs.length === 0 ? (
              <div className="empty-state small">No queued jobs</div>
            ) : (
              queuedJobs.map((job, i) => (
                <div key={i} className="job-card queued">
                  <div className="job-header">
                    <span className="job-name">{job.name}</span>
                    <span className="job-priority" style={{ background: getPriorityColor(job.priority) }}>
                      P{job.priority}
                    </span>
                  </div>
                  <div className="job-info">
                    <span>Type: {job.pipeline_config?.pipeline_type}</span>
                    <span>Model: {job.pipeline_config?.model_type}</span>
                  </div>
                  <div className="job-actions">
                    <select
                      className="priority-select"
                      value={job.priority}
                      onChange={(e) => updatePriority(job.job_id, getPriorityName(parseInt(e.target.value)))}
                    >
                      <option value="1">Critical</option>
                      <option value="2">High</option>
                      <option value="3">Normal</option>
                      <option value="4">Low</option>
                      <option value="5">Background</option>
                    </select>
                    <button className="btn btn-sm btn-danger" onClick={() => cancelJob(job.job_id)}>Cancel</button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="job-section">
          <h3>All Jobs</h3>
          <div className="job-table">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Priority</th>
                  <th>Type</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job, i) => (
                  <tr key={i}>
                    <td>{job.name}</td>
                    <td>
                      <span className="status-badge" style={{ background: getStatusColor(job.status) }}>
                        {job.status}
                      </span>
                    </td>
                    <td>P{job.priority}</td>
                    <td>{job.pipeline_config?.pipeline_type}</td>
                    <td>{new Date(job.created_at).toLocaleString()}</td>
                    <td>
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => {
                          setSelectedJob(job)
                          setActiveView('detail')
                        }}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  const getPriorityColor = (priority) => {
    const colors = { 1: '#f44336', 2: '#ff9800', 3: '#1e88e5', 4: '#4caf50', 5: '#9e9e9e' }
    return colors[priority] || '#9e9e9e'
  }

  const getPriorityName = (priority) => {
    const names = { 1: 'critical', 2: 'high', 3: 'normal', 4: 'low', 5: 'background' }
    return names[priority] || 'normal'
  }

  const renderCreateJob = () => (
    <div className="job-create">
      <div className="job-create-header">
        <button className="btn btn-secondary" onClick={() => setActiveView('dashboard')}>
          ← Back
        </button>
        <h2>Schedule New Job</h2>
      </div>

      <div className="job-form">
        <div className="form-section">
          <h3>Job Information</h3>
          <div className="form-group">
            <label>Job Name</label>
            <input
              type="text"
              value={jobForm.name}
              onChange={(e) => setJobForm({ ...jobForm, name: e.target.value })}
              placeholder="Enter job name"
            />
          </div>
          <div className="form-group">
            <label>Description</label>
            <textarea
              value={jobForm.description}
              onChange={(e) => setJobForm({ ...jobForm, description: e.target.value })}
              placeholder="Enter description"
            />
          </div>
        </div>

        <div className="form-section">
          <h3>Scheduling</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Priority</label>
              <select
                value={jobForm.priority}
                onChange={(e) => setJobForm({ ...jobForm, priority: e.target.value })}
              >
                <option value="critical">Critical (P1)</option>
                <option value="high">High (P2)</option>
                <option value="normal">Normal (P3)</option>
                <option value="low">Low (P4)</option>
                <option value="background">Background (P5)</option>
              </select>
            </div>
            <div className="form-group">
              <label>Schedule Type</label>
              <select
                value={jobForm.schedule_type}
                onChange={(e) => setJobForm({ ...jobForm, schedule_type: e.target.value })}
              >
                <option value="immediate">Immediate</option>
                <option value="once">Scheduled Once</option>
                <option value="recurring">Recurring</option>
              </select>
            </div>
          </div>
          {jobForm.schedule_type !== 'immediate' && (
            <div className="form-group">
              <label>Scheduled Time</label>
              <input
                type="datetime-local"
                value={jobForm.scheduled_time}
                onChange={(e) => setJobForm({ ...jobForm, scheduled_time: e.target.value })}
              />
            </div>
          )}
        </div>

        <div className="form-section">
          <h3>Pipeline Configuration</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Pipeline Type</label>
              <select
                value={jobForm.pipeline_config.pipeline_type}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  pipeline_config: { ...jobForm.pipeline_config, pipeline_type: e.target.value }
                })}
              >
                <option value="full_pipeline">Full Pipeline</option>
                <option value="training">Training</option>
                <option value="inference">Inference</option>
                <option value="hyperparameter_tuning">Hyperparameter Tuning</option>
              </select>
            </div>
            <div className="form-group">
              <label>Data Modality</label>
              <select
                value={jobForm.pipeline_config.data_modality}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  pipeline_config: { ...jobForm.pipeline_config, data_modality: e.target.value }
                })}
              >
                <option value="eeg">EEG</option>
                <option value="mri">MRI</option>
                <option value="ct">CT</option>
                <option value="image">Image</option>
              </select>
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Model Type</label>
              <select
                value={jobForm.pipeline_config.model_type}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  pipeline_config: { ...jobForm.pipeline_config, model_type: e.target.value }
                })}
              >
                <option value="cnn">CNN</option>
                <option value="lstm">LSTM</option>
                <option value="transformer">Transformer</option>
                <option value="resnet">ResNet</option>
                <option value="yolo">YOLO</option>
                <option value="unet">U-Net</option>
                <option value="gan">GAN</option>
                <option value="vae">VAE</option>
              </select>
            </div>
            <div className="form-group">
              <label>Task Type</label>
              <select
                value={jobForm.pipeline_config.task_type}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  pipeline_config: { ...jobForm.pipeline_config, task_type: e.target.value }
                })}
              >
                <option value="classification">Classification</option>
                <option value="segmentation">Segmentation</option>
                <option value="detection">Detection</option>
                <option value="generation">Generation</option>
              </select>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>Resource Requirements</h3>
          <div className="form-row">
            <div className="form-group">
              <label>CPU Cores: {jobForm.resources.cpu_cores}</label>
              <input
                type="range"
                min="1"
                max="8"
                value={jobForm.resources.cpu_cores}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  resources: { ...jobForm.resources, cpu_cores: parseInt(e.target.value) }
                })}
              />
            </div>
            <div className="form-group">
              <label>GPU Count: {jobForm.resources.gpu_count}</label>
              <input
                type="range"
                min="0"
                max="4"
                value={jobForm.resources.gpu_count}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  resources: { ...jobForm.resources, gpu_count: parseInt(e.target.value) }
                })}
              />
            </div>
            <div className="form-group">
              <label>Memory (GB): {jobForm.resources.memory_gb}</label>
              <input
                type="range"
                min="2"
                max="32"
                step="2"
                value={jobForm.resources.memory_gb}
                onChange={(e) => setJobForm({
                  ...jobForm,
                  resources: { ...jobForm.resources, memory_gb: parseInt(e.target.value) }
                })}
              />
            </div>
          </div>
        </div>

        <div className="form-actions">
          <button className="btn btn-secondary" onClick={resetForm}>Reset</button>
          <button
            className="btn btn-primary"
            onClick={createJob}
            disabled={isLoading || !jobForm.name}
          >
            {isLoading ? 'Scheduling...' : 'Schedule Job'}
          </button>
        </div>
      </div>
    </div>
  )

  const renderJobDetail = () => {
    if (!selectedJob) return null

    return (
      <div className="job-detail">
        <div className="job-detail-header">
          <button className="btn btn-secondary" onClick={() => setActiveView('dashboard')}>
            ← Back
          </button>
          <h2>{selectedJob.name}</h2>
          <span className="status-badge" style={{ background: getStatusColor(selectedJob.status) }}>
            {selectedJob.status}
          </span>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Status</div>
            <div className="metric-value">{selectedJob.status}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Priority</div>
            <div className="metric-value">P{selectedJob.priority}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Run Count</div>
            <div className="metric-value">{selectedJob.run_count || 0}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Retry Count</div>
            <div className="metric-value">{selectedJob.retry_count || 0}</div>
          </div>
        </div>

        <div className="chart-card">
          <div className="chart-title">Job Configuration</div>
          <div className="config-detail">
            <div className="config-row">
              <span className="config-label">Pipeline Type:</span>
              <span className="config-value">{selectedJob.pipeline_config?.pipeline_type}</span>
            </div>
            <div className="config-row">
              <span className="config-label">Data Modality:</span>
              <span className="config-value">{selectedJob.pipeline_config?.data_modality}</span>
            </div>
            <div className="config-row">
              <span className="config-label">Model Type:</span>
              <span className="config-value">{selectedJob.pipeline_config?.model_type}</span>
            </div>
            <div className="config-row">
              <span className="config-label">Task Type:</span>
              <span className="config-value">{selectedJob.pipeline_config?.task_type}</span>
            </div>
          </div>
        </div>

        {selectedJob.result && (
          <div className="chart-card">
            <div className="chart-title">Results</div>
            <pre className="result-json">
              {JSON.stringify(selectedJob.result, null, 2)}
            </pre>
          </div>
        )}

        {selectedJob.error && (
          <div className="alert alert-danger">
            <span className="alert-icon">!</span>
            <div className="alert-content">
              <div className="alert-title">Error</div>
              <div className="alert-message">{selectedJob.error}</div>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="job-scheduler">
      {activeView === 'dashboard' && renderDashboard()}
      {activeView === 'create' && renderCreateJob()}
      {activeView === 'detail' && renderJobDetail()}
    </div>
  )
}

export default JobScheduler
