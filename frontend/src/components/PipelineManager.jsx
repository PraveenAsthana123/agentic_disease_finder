import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const COLORS = ['#1e88e5', '#7c4dff', '#4caf50', '#ff9800', '#f44336', '#00bcd4']

function PipelineManager() {
  // Pipeline state
  const [pipelines, setPipelines] = useState([])
  const [activeView, setActiveView] = useState('list') // list, create, detail
  const [selectedPipeline, setSelectedPipeline] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  // Pipeline configuration
  const [availableModels, setAvailableModels] = useState([])
  const [preprocessingOptions, setPreprocessingOptions] = useState({})

  // Create pipeline form
  const [pipelineForm, setPipelineForm] = useState({
    name: '',
    description: '',
    pipeline_type: 'full_pipeline',
    data_modality: 'eeg',
    model_type: 'cnn',
    task_type: 'classification',
    train_split: 0.7,
    val_split: 0.15,
    test_split: 0.15,
    epochs: 100,
    batch_size: 32,
    learning_rate: 0.001,
    preprocessing: {
      filters: ['bandpass'],
      noise_removal: ['wavelet_denoising'],
      transformation: { '1d_to_2d': 'spectrogram' }
    }
  })

  // Fetch pipelines and options
  useEffect(() => {
    fetchPipelines()
    fetchModels()
    fetchPreprocessing()
  }, [])

  const fetchPipelines = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/pipeline/list/all`)
      setPipelines(response.data.pipelines || [])
    } catch (err) {
      console.error('Error fetching pipelines:', err)
    }
  }

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/pipeline/models`)
      setAvailableModels(response.data.models || [])
    } catch (err) {
      setAvailableModels(getMockModels())
    }
  }

  const fetchPreprocessing = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/pipeline/preprocessing`)
      setPreprocessingOptions(response.data.options || {})
    } catch (err) {
      setPreprocessingOptions(getMockPreprocessing())
    }
  }

  const createPipeline = async () => {
    setIsLoading(true)
    try {
      const response = await axios.post(`${API_URL}/api/pipeline/create`, pipelineForm)
      if (response.data.success) {
        await fetchPipelines()
        setActiveView('list')
        resetForm()
      }
    } catch (err) {
      console.error('Error creating pipeline:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const runPipeline = async (pipelineId) => {
    setIsLoading(true)
    try {
      const response = await axios.post(`${API_URL}/api/pipeline/run/${pipelineId}`)
      if (response.data.success) {
        setSelectedPipeline(response.data.result)
        setActiveView('detail')
        await fetchPipelines()
      }
    } catch (err) {
      console.error('Error running pipeline:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const cancelPipeline = async (pipelineId) => {
    try {
      await axios.delete(`${API_URL}/api/pipeline/${pipelineId}`)
      await fetchPipelines()
    } catch (err) {
      console.error('Error cancelling pipeline:', err)
    }
  }

  const resetForm = () => {
    setPipelineForm({
      name: '',
      description: '',
      pipeline_type: 'full_pipeline',
      data_modality: 'eeg',
      model_type: 'cnn',
      task_type: 'classification',
      train_split: 0.7,
      val_split: 0.15,
      test_split: 0.15,
      epochs: 100,
      batch_size: 32,
      learning_rate: 0.001,
      preprocessing: {
        filters: ['bandpass'],
        noise_removal: ['wavelet_denoising'],
        transformation: { '1d_to_2d': 'spectrogram' }
      }
    })
  }

  // Mock data
  const getMockModels = () => [
    { category: 'Machine Learning', models: [
      { id: 'random_forest', name: 'Random Forest', tasks: ['classification', 'regression'] },
      { id: 'xgboost', name: 'XGBoost', tasks: ['classification', 'regression'] },
      { id: 'svm', name: 'SVM', tasks: ['classification'] }
    ]},
    { category: 'Deep Learning', models: [
      { id: 'cnn', name: 'CNN', tasks: ['classification', 'segmentation'] },
      { id: 'lstm', name: 'LSTM', tasks: ['classification', 'regression'] },
      { id: 'transformer', name: 'Transformer', tasks: ['classification'] }
    ]},
    { category: 'Computer Vision', models: [
      { id: 'yolo', name: 'YOLO', tasks: ['detection'] },
      { id: 'unet', name: 'U-Net', tasks: ['segmentation'] },
      { id: 'resnet', name: 'ResNet', tasks: ['classification'] }
    ]},
    { category: 'Generative', models: [
      { id: 'gan', name: 'GAN', tasks: ['generation'] },
      { id: 'vae', name: 'VAE', tasks: ['generation', 'anomaly_detection'] }
    ]}
  ]

  const getMockPreprocessing = () => ({
    filters: [
      { id: 'bandpass', name: 'Bandpass Filter' },
      { id: 'lowpass', name: 'Lowpass Filter' },
      { id: 'highpass', name: 'Highpass Filter' },
      { id: 'notch', name: 'Notch Filter' }
    ],
    noise_removal: [
      { id: 'wavelet_denoising', name: 'Wavelet Denoising' },
      { id: 'ica', name: 'ICA' },
      { id: 'pca_denoising', name: 'PCA Denoising' }
    ],
    transformations: {
      '1d_to_2d': [
        { id: 'spectrogram', name: 'Spectrogram' },
        { id: 'scalogram', name: 'Scalogram' },
        { id: 'recurrence_plot', name: 'Recurrence Plot' }
      ]
    }
  })

  // Render pipeline list
  const renderPipelineList = () => (
    <div className="pipeline-list">
      <div className="pipeline-list-header">
        <h2>Pipeline Manager</h2>
        <button className="btn btn-primary" onClick={() => setActiveView('create')}>
          + Create Pipeline
        </button>
      </div>

      <div className="pipeline-grid">
        {pipelines.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">📊</div>
            <div className="empty-title">No Pipelines</div>
            <div className="empty-description">Create your first pipeline to get started</div>
          </div>
        ) : (
          pipelines.map((p, i) => (
            <div key={i} className="pipeline-card">
              <div className="pipeline-card-header">
                <span className="pipeline-name">{p.config?.name || 'Unnamed'}</span>
                <span className={`pipeline-status ${p.result?.status || 'pending'}`}>
                  {p.result?.status || 'pending'}
                </span>
              </div>
              <div className="pipeline-card-body">
                <div className="pipeline-info">
                  <span>Type: {p.config?.pipeline_type}</span>
                  <span>Modality: {p.config?.data_modality}</span>
                  <span>Model: {p.config?.model_type}</span>
                </div>
                {p.result?.progress !== undefined && (
                  <div className="pipeline-progress">
                    <div className="progress-bar">
                      <div className="progress-fill" style={{ width: `${p.result.progress}%` }} />
                    </div>
                    <span>{p.result.progress?.toFixed(1)}%</span>
                  </div>
                )}
              </div>
              <div className="pipeline-card-actions">
                <button
                  className="btn btn-sm btn-primary"
                  onClick={() => runPipeline(p.config?.pipeline_id)}
                  disabled={p.result?.status === 'running'}
                >
                  Run
                </button>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => {
                    setSelectedPipeline(p)
                    setActiveView('detail')
                  }}
                >
                  View
                </button>
                <button
                  className="btn btn-sm btn-danger"
                  onClick={() => cancelPipeline(p.config?.pipeline_id)}
                >
                  Cancel
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )

  // Render create pipeline form
  const renderCreatePipeline = () => (
    <div className="pipeline-create">
      <div className="pipeline-create-header">
        <button className="btn btn-secondary" onClick={() => setActiveView('list')}>
          ← Back
        </button>
        <h2>Create New Pipeline</h2>
      </div>

      <div className="pipeline-form">
        <div className="form-section">
          <h3>Basic Information</h3>
          <div className="form-group">
            <label>Pipeline Name</label>
            <input
              type="text"
              value={pipelineForm.name}
              onChange={(e) => setPipelineForm({ ...pipelineForm, name: e.target.value })}
              placeholder="Enter pipeline name"
            />
          </div>
          <div className="form-group">
            <label>Description</label>
            <textarea
              value={pipelineForm.description}
              onChange={(e) => setPipelineForm({ ...pipelineForm, description: e.target.value })}
              placeholder="Enter description"
            />
          </div>
        </div>

        <div className="form-section">
          <h3>Pipeline Configuration</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Pipeline Type</label>
              <select
                value={pipelineForm.pipeline_type}
                onChange={(e) => setPipelineForm({ ...pipelineForm, pipeline_type: e.target.value })}
              >
                <option value="full_pipeline">Full Pipeline</option>
                <option value="data">Data Pipeline</option>
                <option value="training">Training Pipeline</option>
                <option value="inference">Inference Pipeline</option>
                <option value="hyperparameter_tuning">Hyperparameter Tuning</option>
                <option value="drift_detection">Drift Detection</option>
              </select>
            </div>
            <div className="form-group">
              <label>Data Modality</label>
              <select
                value={pipelineForm.data_modality}
                onChange={(e) => setPipelineForm({ ...pipelineForm, data_modality: e.target.value })}
              >
                <option value="eeg">EEG</option>
                <option value="mri">MRI</option>
                <option value="ct">CT Scan</option>
                <option value="image">Image</option>
                <option value="tabular">Tabular</option>
                <option value="multimodal">Multimodal</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label>Task Type</label>
              <select
                value={pipelineForm.task_type}
                onChange={(e) => setPipelineForm({ ...pipelineForm, task_type: e.target.value })}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
                <option value="segmentation">Segmentation</option>
                <option value="detection">Detection</option>
                <option value="generation">Generation</option>
                <option value="anomaly_detection">Anomaly Detection</option>
              </select>
            </div>
            <div className="form-group">
              <label>Model Type</label>
              <select
                value={pipelineForm.model_type}
                onChange={(e) => setPipelineForm({ ...pipelineForm, model_type: e.target.value })}
              >
                {(availableModels.length > 0 ? availableModels : getMockModels()).map(category => (
                  <optgroup key={category.category} label={category.category}>
                    {category.models.map(model => (
                      <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>Data Split Configuration</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Train Split: {(pipelineForm.train_split * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0.5"
                max="0.9"
                step="0.05"
                value={pipelineForm.train_split}
                onChange={(e) => setPipelineForm({ ...pipelineForm, train_split: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label>Validation Split: {(pipelineForm.val_split * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0.05"
                max="0.3"
                step="0.05"
                value={pipelineForm.val_split}
                onChange={(e) => setPipelineForm({ ...pipelineForm, val_split: parseFloat(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label>Test Split: {(pipelineForm.test_split * 100).toFixed(0)}%</label>
              <input
                type="range"
                min="0.05"
                max="0.3"
                step="0.05"
                value={pipelineForm.test_split}
                onChange={(e) => setPipelineForm({ ...pipelineForm, test_split: parseFloat(e.target.value) })}
              />
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>Training Configuration</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Epochs: {pipelineForm.epochs}</label>
              <input
                type="range"
                min="10"
                max="500"
                step="10"
                value={pipelineForm.epochs}
                onChange={(e) => setPipelineForm({ ...pipelineForm, epochs: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label>Batch Size: {pipelineForm.batch_size}</label>
              <input
                type="range"
                min="8"
                max="128"
                step="8"
                value={pipelineForm.batch_size}
                onChange={(e) => setPipelineForm({ ...pipelineForm, batch_size: parseInt(e.target.value) })}
              />
            </div>
            <div className="form-group">
              <label>Learning Rate: {pipelineForm.learning_rate}</label>
              <select
                value={pipelineForm.learning_rate}
                onChange={(e) => setPipelineForm({ ...pipelineForm, learning_rate: parseFloat(e.target.value) })}
              >
                <option value="0.1">0.1</option>
                <option value="0.01">0.01</option>
                <option value="0.001">0.001</option>
                <option value="0.0001">0.0001</option>
                <option value="0.00001">0.00001</option>
              </select>
            </div>
          </div>
        </div>

        <div className="form-section">
          <h3>Preprocessing</h3>
          <div className="form-row">
            <div className="form-group">
              <label>Filters</label>
              <div className="checkbox-grid">
                {(preprocessingOptions.filters || getMockPreprocessing().filters).map(filter => (
                  <label key={filter.id} className="checkbox-item">
                    <input
                      type="checkbox"
                      checked={pipelineForm.preprocessing.filters?.includes(filter.id)}
                      onChange={(e) => {
                        const filters = e.target.checked
                          ? [...(pipelineForm.preprocessing.filters || []), filter.id]
                          : (pipelineForm.preprocessing.filters || []).filter(f => f !== filter.id)
                        setPipelineForm({
                          ...pipelineForm,
                          preprocessing: { ...pipelineForm.preprocessing, filters }
                        })
                      }}
                    />
                    {filter.name}
                  </label>
                ))}
              </div>
            </div>
            <div className="form-group">
              <label>Noise Removal</label>
              <div className="checkbox-grid">
                {(preprocessingOptions.noise_removal || getMockPreprocessing().noise_removal).map(method => (
                  <label key={method.id} className="checkbox-item">
                    <input
                      type="checkbox"
                      checked={pipelineForm.preprocessing.noise_removal?.includes(method.id)}
                      onChange={(e) => {
                        const noise_removal = e.target.checked
                          ? [...(pipelineForm.preprocessing.noise_removal || []), method.id]
                          : (pipelineForm.preprocessing.noise_removal || []).filter(f => f !== method.id)
                        setPipelineForm({
                          ...pipelineForm,
                          preprocessing: { ...pipelineForm.preprocessing, noise_removal }
                        })
                      }}
                    />
                    {method.name}
                  </label>
                ))}
              </div>
            </div>
          </div>
          <div className="form-group">
            <label>1D to 2D Transformation</label>
            <select
              value={pipelineForm.preprocessing.transformation?.['1d_to_2d'] || ''}
              onChange={(e) => setPipelineForm({
                ...pipelineForm,
                preprocessing: {
                  ...pipelineForm.preprocessing,
                  transformation: { '1d_to_2d': e.target.value }
                }
              })}
            >
              <option value="">None</option>
              {(preprocessingOptions.transformations?.['1d_to_2d'] || getMockPreprocessing().transformations['1d_to_2d']).map(t => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="form-actions">
          <button className="btn btn-secondary" onClick={resetForm}>Reset</button>
          <button
            className="btn btn-primary"
            onClick={createPipeline}
            disabled={isLoading || !pipelineForm.name}
          >
            {isLoading ? 'Creating...' : 'Create Pipeline'}
          </button>
        </div>
      </div>
    </div>
  )

  // Render pipeline detail
  const renderPipelineDetail = () => {
    const config = selectedPipeline?.config || {}
    const result = selectedPipeline?.result || {}
    const metrics = result.metrics || {}

    const stageData = (result.stages_completed || []).map((stage, i) => ({
      name: stage.replace(/_/g, ' '),
      completed: true,
      index: i
    }))

    return (
      <div className="pipeline-detail">
        <div className="pipeline-detail-header">
          <button className="btn btn-secondary" onClick={() => setActiveView('list')}>
            ← Back
          </button>
          <h2>{config.name || 'Pipeline Details'}</h2>
          <span className={`pipeline-status ${result.status || 'pending'}`}>
            {result.status || 'pending'}
          </span>
        </div>

        <div className="metrics-grid">
          <div className="metric-card">
            <div className="metric-label">Status</div>
            <div className="metric-value">{result.status || 'N/A'}</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Progress</div>
            <div className="metric-value">{(result.progress || 0).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Duration</div>
            <div className="metric-value">{(result.duration_seconds || 0).toFixed(1)}s</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Stages Completed</div>
            <div className="metric-value">{(result.stages_completed || []).length}</div>
          </div>
        </div>

        {Object.keys(metrics).length > 0 && (
          <div className="chart-card">
            <div className="chart-title">Pipeline Metrics</div>
            <div className="metrics-detail">
              {Object.entries(metrics).map(([key, value]) => (
                <div key={key} className="metric-item">
                  <span className="metric-key">{key.replace(/_/g, ' ')}</span>
                  <span className="metric-val">
                    {typeof value === 'number' ? value.toFixed(4) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {stageData.length > 0 && (
          <div className="chart-card">
            <div className="chart-title">Completed Stages</div>
            <div className="stage-list">
              {stageData.map((stage, i) => (
                <div key={i} className="stage-item completed">
                  <span className="stage-icon">✓</span>
                  <span className="stage-name">{stage.name}</span>
                </div>
              ))}
              {result.current_stage && result.status === 'running' && (
                <div className="stage-item running">
                  <span className="stage-icon spinning">◌</span>
                  <span className="stage-name">{result.current_stage.replace(/_/g, ' ')}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {result.errors && result.errors.length > 0 && (
          <div className="alert alert-danger">
            <span className="alert-icon">!</span>
            <div className="alert-content">
              <div className="alert-title">Errors</div>
              {result.errors.map((err, i) => (
                <div key={i} className="alert-message">{err}</div>
              ))}
            </div>
          </div>
        )}

        {result.logs && result.logs.length > 0 && (
          <div className="chart-card">
            <div className="chart-title">Logs</div>
            <div className="log-container">
              {result.logs.map((log, i) => (
                <div key={i} className="log-entry">{log}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="pipeline-manager">
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <div className="loading-text">Processing...</div>
        </div>
      )}

      {activeView === 'list' && renderPipelineList()}
      {activeView === 'create' && renderCreatePipeline()}
      {activeView === 'detail' && renderPipelineDetail()}
    </div>
  )
}

export default PipelineManager
