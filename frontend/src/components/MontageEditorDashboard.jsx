import React, { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'

const ELECTRODES_10_20 = [
  'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
  'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2'
]

function Card({ title, children, span }) {
  return (
    <div style={{
      background: '#fff', borderRadius: 12, padding: 20,
      boxShadow: '0 1px 3px rgba(0,0,0,.08)',
      gridColumn: span ? `span ${span}` : undefined
    }}>
      {title && <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#334155', fontWeight: 600 }}>{title}</h3>}
      {children}
    </div>
  )
}

function KPI({ label, value, color, sub }) {
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: color || '#1e293b' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function TypeBadge({ type }) {
  const colors = {
    bipolar: '#3b82f6',
    referential: '#8b5cf6',
    average: '#06b6d4',
    laplacian: '#f97316',
    custom: '#22c55e'
  }
  const c = colors[type] || '#94a3b8'
  return (
    <span style={{
      display: 'inline-block', padding: '3px 10px', borderRadius: 12,
      background: c + '22', color: c, fontSize: 12, fontWeight: 600, border: `1px solid ${c}44`
    }}>{type}</span>
  )
}

export default function MontageEditorDashboard() {
  const [presets, setPresets] = useState([])
  const [customMontages, setCustomMontages] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tab, setTab] = useState('presets')

  // Custom montage form state
  const [formName, setFormName] = useState('')
  const [formType, setFormType] = useState('bipolar')
  const [formDesc, setFormDesc] = useState('')
  const [formChannels, setFormChannels] = useState([{ label: '', anode: 'Fp1', cathode: 'F3' }])
  const [saving, setSaving] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const [p, c] = await Promise.all([
        axios.get(`${API_URL}/api/montage-editor/presets`),
        axios.get(`${API_URL}/api/montage-editor/montages`)
      ])
      setPresets(p.data.presets || [])
      setCustomMontages(c.data.montages || [])
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const addChannel = () => {
    setFormChannels([...formChannels, { label: '', anode: 'Fp1', cathode: 'F3' }])
  }

  const removeChannel = (idx) => {
    setFormChannels(formChannels.filter((_, i) => i !== idx))
  }

  const updateChannel = (idx, field, value) => {
    const updated = [...formChannels]
    updated[idx] = { ...updated[idx], [field]: value }
    if (field === 'anode' || field === 'cathode') {
      updated[idx].label = `${updated[idx].anode}-${updated[idx].cathode}`
    }
    setFormChannels(updated)
  }

  const saveMontage = async () => {
    if (!formName.trim()) return
    setSaving(true)
    try {
      const channels = formChannels.map(ch => ({
        label: ch.label || `${ch.anode}-${ch.cathode}`,
        anode: ch.anode,
        cathode: ch.cathode
      }))
      await axios.post(`${API_URL}/api/montage-editor/montages`, {
        name: formName, type: formType, description: formDesc, channels
      })
      setFormName('')
      setFormDesc('')
      setFormChannels([{ label: '', anode: 'Fp1', cathode: 'F3' }])
      await load()
    } catch (e) {
      setError(e.message)
    } finally {
      setSaving(false)
    }
  }

  const deleteMontage = async (id) => {
    try {
      await axios.delete(`${API_URL}/api/montage-editor/montages/${id}`)
      await load()
    } catch (e) {
      setError(e.message)
    }
  }

  const totalChannels = presets.reduce((s, p) => s + (p.channels?.length || 0), 0)

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#64748b' }}>Loading Montage Editor...</div>
  if (error) return <div style={{ padding: 40, textAlign: 'center', color: '#ef4444' }}>Error: {error} <button onClick={load}>Retry</button></div>

  const tabs = [
    { id: 'presets', label: 'Preset Montages' },
    { id: 'create', label: 'Create Custom' },
    { id: 'saved', label: 'Saved Montages' },
    { id: 'electrodes', label: '10-20 Electrode Map' }
  ]

  return (
    <div style={{ padding: 24, maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ margin: 0, fontSize: 22, color: '#1e293b' }}>Montage Editor Dashboard</h2>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#64748b' }}>
          Configure and manage EEG montage derivations for clinical review
        </p>
      </div>

      {/* KPI Row */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24
      }}>
        <Card>
          <KPI label="Standard Presets" value={presets.length} color="#3b82f6" />
        </Card>
        <Card>
          <KPI label="Custom Montages" value={customMontages.length} color="#22c55e" />
        </Card>
        <Card>
          <KPI label="Total Preset Channels" value={totalChannels} color="#8b5cf6" />
        </Card>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t.id ? '#3b82f6' : '#f1f5f9',
            color: tab === t.id ? '#fff' : '#475569',
            fontWeight: 600, fontSize: 13, transition: 'all .15s'
          }}>{t.label}</button>
        ))}
      </div>

      {/* Presets Tab */}
      {tab === 'presets' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380, 1fr))', gap: 16 }}>
          {presets.map((preset, i) => (
            <Card key={i} title={preset.name}>
              <div style={{ marginBottom: 10 }}>
                <TypeBadge type={preset.type} />
                <span style={{ marginLeft: 10, fontSize: 12, color: '#64748b' }}>{preset.description}</span>
              </div>
              <div style={{ fontSize: 12, color: '#334155', lineHeight: 1.8 }}>
                {(preset.channels || []).map((ch, j) => (
                  <span key={j} style={{
                    display: 'inline-block', padding: '2px 8px', margin: '2px 4px 2px 0',
                    background: '#f1f5f9', borderRadius: 6, fontFamily: 'monospace', fontSize: 11
                  }}>{ch.label}</span>
                ))}
              </div>
              <div style={{ marginTop: 8, fontSize: 11, color: '#94a3b8' }}>
                {preset.channels?.length || 0} channels
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Create Custom Tab */}
      {tab === 'create' && (
        <Card title="Create Custom Montage" span={2}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 16 }}>
            <div>
              <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 4 }}>Montage Name</label>
              <input value={formName} onChange={e => setFormName(e.target.value)}
                placeholder="e.g., My Custom Bipolar"
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
            </div>
            <div>
              <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 4 }}>Type</label>
              <select value={formType} onChange={e => setFormType(e.target.value)}
                style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }}>
                <option value="bipolar">Bipolar</option>
                <option value="referential">Referential</option>
                <option value="average">Average</option>
                <option value="laplacian">Laplacian</option>
                <option value="custom">Custom</option>
              </select>
            </div>
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 12, color: '#64748b', display: 'block', marginBottom: 4 }}>Description</label>
            <input value={formDesc} onChange={e => setFormDesc(e.target.value)}
              placeholder="Brief description of this montage"
              style={{ width: '100%', padding: '8px 12px', borderRadius: 8, border: '1px solid #e2e8f0', fontSize: 13 }} />
          </div>

          <h4 style={{ fontSize: 13, color: '#334155', margin: '0 0 10px' }}>Channel Derivations</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 12 }}>#</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 12 }}>Anode (+)</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 12 }}>Cathode (-)</th>
                <th style={{ textAlign: 'left', padding: '6px 8px', color: '#64748b', fontSize: 12 }}>Label</th>
                <th style={{ padding: '6px 8px' }}></th>
              </tr>
            </thead>
            <tbody>
              {formChannels.map((ch, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #f1f5f9' }}>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{i + 1}</td>
                  <td style={{ padding: '6px 8px' }}>
                    <select value={ch.anode} onChange={e => updateChannel(i, 'anode', e.target.value)}
                      style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 12 }}>
                      {ELECTRODES_10_20.map(el => <option key={el} value={el}>{el}</option>)}
                    </select>
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    <select value={ch.cathode} onChange={e => updateChannel(i, 'cathode', e.target.value)}
                      style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid #e2e8f0', fontSize: 12 }}>
                      {ELECTRODES_10_20.map(el => <option key={el} value={el}>{el}</option>)}
                    </select>
                  </td>
                  <td style={{ padding: '6px 8px', fontFamily: 'monospace', fontSize: 12, color: '#334155' }}>
                    {ch.label || `${ch.anode}-${ch.cathode}`}
                  </td>
                  <td style={{ padding: '6px 8px' }}>
                    {formChannels.length > 1 && (
                      <button onClick={() => removeChannel(i)} style={{
                        background: '#fee2e2', border: 'none', borderRadius: 6, padding: '4px 8px',
                        color: '#ef4444', cursor: 'pointer', fontSize: 11
                      }}>Remove</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 12, display: 'flex', gap: 10 }}>
            <button onClick={addChannel} style={{
              padding: '8px 16px', borderRadius: 8, border: '1px solid #e2e8f0',
              background: '#f8fafc', cursor: 'pointer', fontSize: 12, fontWeight: 600, color: '#475569'
            }}>+ Add Channel</button>
            <button onClick={saveMontage} disabled={saving || !formName.trim()} style={{
              padding: '8px 20px', borderRadius: 8, border: 'none',
              background: formName.trim() ? '#3b82f6' : '#94a3b8', color: '#fff',
              cursor: formName.trim() ? 'pointer' : 'not-allowed', fontSize: 12, fontWeight: 600
            }}>{saving ? 'Saving...' : 'Save Montage'}</button>
          </div>
        </Card>
      )}

      {/* Saved Montages Tab */}
      {tab === 'saved' && (
        <Card title="Saved Custom Montages" span={2}>
          {customMontages.length === 0 ? (
            <div style={{ padding: 30, textAlign: 'center', color: '#94a3b8' }}>
              No custom montages saved yet. Use the Create tab to add one.
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid #e2e8f0' }}>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', fontSize: 12 }}>Name</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', fontSize: 12 }}>Type</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', fontSize: 12 }}>Description</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', fontSize: 12 }}>Channels</th>
                  <th style={{ textAlign: 'left', padding: '8px', color: '#64748b', fontSize: 12 }}>Created</th>
                  <th style={{ padding: '8px' }}></th>
                </tr>
              </thead>
              <tbody>
                {customMontages.map(m => (
                  <tr key={m.id} style={{ borderBottom: '1px solid #f1f5f9' }}>
                    <td style={{ padding: '8px', fontWeight: 600 }}>{m.name}</td>
                    <td style={{ padding: '8px' }}><TypeBadge type={m.type} /></td>
                    <td style={{ padding: '8px', color: '#64748b' }}>{m.description}</td>
                    <td style={{ padding: '8px' }}>
                      {(m.channels || []).map((ch, j) => (
                        <span key={j} style={{
                          display: 'inline-block', padding: '1px 6px', margin: '1px 3px 1px 0',
                          background: '#f1f5f9', borderRadius: 4, fontFamily: 'monospace', fontSize: 11
                        }}>{ch.label || `${ch.anode}-${ch.cathode}`}</span>
                      ))}
                    </td>
                    <td style={{ padding: '8px', color: '#94a3b8', fontSize: 11 }}>
                      {m.created_at ? new Date(m.created_at).toLocaleDateString() : '-'}
                    </td>
                    <td style={{ padding: '8px' }}>
                      <button onClick={() => deleteMontage(m.id)} style={{
                        background: '#fee2e2', border: 'none', borderRadius: 6, padding: '4px 10px',
                        color: '#ef4444', cursor: 'pointer', fontSize: 11, fontWeight: 600
                      }}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      )}

      {/* 10-20 Electrode Map Tab */}
      {tab === 'electrodes' && (
        <Card title="International 10-20 Electrode System" span={2}>
          <p style={{ fontSize: 12, color: '#64748b', margin: '0 0 16px' }}>
            Standard electrode positions used in clinical EEG montage derivations
          </p>
          <div style={{
            display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 10, maxWidth: 500, margin: '0 auto'
          }}>
            {ELECTRODES_10_20.map(el => {
              const regionColor =
                el.startsWith('Fp') ? '#ef4444' :
                el.startsWith('F') ? '#f97316' :
                el.startsWith('C') ? '#eab308' :
                el.startsWith('T') ? '#22c55e' :
                el.startsWith('P') ? '#3b82f6' :
                el.startsWith('O') ? '#8b5cf6' :
                el.startsWith('A') ? '#94a3b8' : '#64748b'
              return (
                <div key={el} style={{
                  background: regionColor + '18', border: `2px solid ${regionColor}44`,
                  borderRadius: 10, padding: '12px 6px', textAlign: 'center',
                  fontFamily: 'monospace', fontWeight: 700, fontSize: 13, color: regionColor
                }}>{el}</div>
              )
            })}
          </div>
          <div style={{ marginTop: 20, display: 'flex', gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
            {[
              ['Fp - Frontopolar', '#ef4444'],
              ['F - Frontal', '#f97316'],
              ['C - Central', '#eab308'],
              ['T - Temporal', '#22c55e'],
              ['P - Parietal', '#3b82f6'],
              ['O - Occipital', '#8b5cf6'],
              ['A - Auricular', '#94a3b8']
            ].map(([label, color]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#64748b' }}>
                <div style={{ width: 12, height: 12, borderRadius: 3, background: color + '44' }} />
                {label}
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}
