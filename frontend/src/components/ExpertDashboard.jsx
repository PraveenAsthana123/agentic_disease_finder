import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  PieChart, Pie, Cell, Legend
} from 'recharts'

const API_URL = '/api'
const BAND_COLORS = { delta: '#1e88e5', theta: '#7c4dff', alpha: '#4caf50', beta: '#ff9800', gamma: '#f44336' }
const PIE_COLORS = ['#4caf50', '#f44336', '#ff9800', '#1e88e5']

export default function ExpertDashboard() {
  const [panel, setPanel] = useState('montage')
  const [montage, setMontage] = useState(null)
  const [localization, setLocalization] = useState(null)
  const [falseAlarm, setFalseAlarm] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const [m, l, f] = await Promise.all([
          axios.get(`${API_URL}/eeg-viz/montage-comparison`),
          axios.get(`${API_URL}/eeg-viz/localization`),
          axios.get(`${API_URL}/eeg-viz/false-alarm`)
        ])
        setMontage(m.data)
        setLocalization(l.data)
        setFalseAlarm(f.data)
      } catch (err) {
        setError(err.message || 'Failed to load expert dashboards')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{ padding: 32, textAlign: 'center' }}>Loading expert dashboards...</div>
  if (error) return <div style={{ padding: 32, color: '#f44336' }}>Error: {error}</div>

  const panels = [
    { id: 'montage', label: '🔀 Montage Comparison' },
    { id: 'localization', label: '📍 Localization' },
    { id: 'falsealarm', label: '🚨 False Alarm Review' }
  ]

  const card = (title, children) => (
    <div style={{ background: '#fff', borderRadius: 10, padding: 18, boxShadow: '0 1px 6px rgba(0,0,0,0.08)', marginBottom: 16 }}>
      <h4 style={{ margin: '0 0 10px', color: '#1e293b', fontSize: 14 }}>{title}</h4>
      {children}
    </div>
  )

  const kpi = (label, value, sub) => (
    <div style={{ background: '#f8fafc', borderRadius: 8, padding: '12px 16px', textAlign: 'center', minWidth: 110 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: '#0f172a' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{sub}</div>}
    </div>
  )

  /* ─── Montage Comparison ────────────────────────────────── */
  const renderMontage = () => {
    if (!montage?.available) return <div style={{ padding: 16, color: '#94a3b8' }}>No monopolar EDF available for montage comparison.</div>

    const montageKeys = Object.keys(montage.montages || {})
    const bands = Object.keys(BAND_COLORS)

    // Bar chart: amplitude comparison across montages
    const ampData = montageKeys.map(m => ({
      name: m.replace(/_/g, ' '),
      amplitude: montage.montages[m].mean_amplitude_uv,
      channels: montage.montages[m].n_channels
    }))

    // Radar chart: band power per montage
    const radarData = bands.map(b => {
      const row = { band: b.charAt(0).toUpperCase() + b.slice(1) }
      montageKeys.forEach(m => { row[m] = montage.montages[m].band_power?.[b] || 0 })
      return row
    })

    // Band power delta vs referential
    const deltas = montage.band_power_delta_vs_referential || {}
    const deltaData = Object.entries(deltas).flatMap(([m, bds]) =>
      Object.entries(bds).map(([b, v]) => ({ montage: m.replace(/_/g, ' '), band: b, delta: v }))
    )

    return (
      <>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
          {kpi('File', montage.file)}
          {kpi('Sample Rate', `${montage.sfreq} Hz`)}
          {kpi('Duration', `${montage.seconds}s`)}
          {montageKeys.map(m => kpi(m.replace(/_/g, ' '), `${montage.montages[m].n_channels} ch`, `${montage.montages[m].mean_amplitude_uv} µV`))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {card('Mean Amplitude by Montage (µV)',
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={ampData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="amplitude" fill="#1e88e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {card('Band Power by Montage (Radar)',
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="band" fontSize={11} />
                <PolarRadiusAxis fontSize={10} />
                {montageKeys.map((m, i) => (
                  <Radar key={m} name={m.replace(/_/g, ' ')} dataKey={m}
                    stroke={Object.values(BAND_COLORS)[i]} fill={Object.values(BAND_COLORS)[i]} fillOpacity={0.15} />
                ))}
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </RadarChart>
            </ResponsiveContainer>
          )}

          {card('Band Power Delta vs Referential',
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={deltaData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="band" fontSize={11} />
                <YAxis fontSize={11} />
                <Tooltip />
                <Bar dataKey="delta" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {card('Montage Details',
            <div style={{ fontSize: 12 }}>
              {montageKeys.map(m => (
                <div key={m} style={{ marginBottom: 10, padding: 8, background: '#f1f5f9', borderRadius: 6 }}>
                  <strong style={{ color: '#1e293b' }}>{m.replace(/_/g, ' ')}</strong>
                  <div style={{ color: '#475569', marginTop: 2 }}>{montage.montages[m].description}</div>
                  {montage.montages[m].example_derivations && (
                    <div style={{ color: '#64748b', fontFamily: 'monospace', fontSize: 11, marginTop: 4 }}>
                      e.g. {montage.montages[m].example_derivations.join(', ')}
                    </div>
                  )}
                </div>
              ))}
              <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 8 }}>{montage.note}</div>
            </div>
          )}
        </div>
      </>
    )
  }

  /* ─── Localization ──────────────────────────────────────── */
  const renderLocalization = () => {
    if (!localization?.available) return <div style={{ padding: 16, color: '#94a3b8' }}>No annotated seizure EDF available for localization.</div>

    const focus = localization.localized_focus || {}
    const top = localization.top_focus_channels || []
    const allRanked = localization.all_channels_ranked || []

    // Bar chart: top channels ictal increase
    const topBar = top.map(c => ({ channel: c.channel, increase: c.ictal_increase_x }))

    // Pie chart: region distribution among top channels
    const regionCounts = {}
    top.forEach(c => { regionCounts[c.region] = (regionCounts[c.region] || 0) + 1 })
    const regionPie = Object.entries(regionCounts).map(([r, v]) => ({ name: r, value: v }))

    // Hemisphere distribution
    const hemCounts = {}
    allRanked.forEach(c => { hemCounts[c.hemisphere] = (hemCounts[c.hemisphere] || 0) + 1 })
    const hemPie = Object.entries(hemCounts).map(([h, v]) => ({ name: h, value: v }))

    return (
      <>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
          {kpi('Focus', focus.summary || 'N/A')}
          {kpi('Peak Increase', `${focus.peak_increase_x || '—'}×`)}
          {kpi('File', localization.file)}
          {kpi('Seizure', `${localization.seizure_window?.start_s}–${localization.seizure_window?.end_s}s`)}
          {kpi('Channels Ranked', allRanked.length)}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {card('Top Focus Channels — Ictal Power Increase (×)',
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={topBar} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" fontSize={11} />
                <YAxis dataKey="channel" type="category" fontSize={11} width={70} />
                <Tooltip />
                <Bar dataKey="increase" fill="#f44336" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}

          {card('Focus Region Distribution',
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={regionPie} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name} (${value})`}>
                  {regionPie.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}

          {card('Hemisphere Distribution (All Channels)',
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={hemPie} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name} (${value})`}>
                  {hemPie.map((_, i) => <Cell key={i} fill={['#1e88e5', '#4caf50', '#ff9800', '#7c4dff'][i % 4]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          )}

          {card('Channel Ranking Table',
            <div style={{ maxHeight: 240, overflowY: 'auto' }}>
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: '#f1f5f9' }}>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>#</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Channel</th>
                    <th style={{ padding: '6px 8px', textAlign: 'right' }}>Increase ×</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Region</th>
                    <th style={{ padding: '6px 8px', textAlign: 'left' }}>Hemisphere</th>
                  </tr>
                </thead>
                <tbody>
                  {allRanked.slice(0, 20).map((c, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #f1f5f9', background: i < 3 ? '#fef2f2' : 'transparent' }}>
                      <td style={{ padding: '5px 8px', fontFamily: 'monospace' }}>{i + 1}</td>
                      <td style={{ padding: '5px 8px', fontWeight: i < 3 ? 600 : 400 }}>{c.channel}</td>
                      <td style={{ padding: '5px 8px', textAlign: 'right', fontFamily: 'monospace' }}>{c.ictal_increase_x}×</td>
                      <td style={{ padding: '5px 8px' }}>{c.region}</td>
                      <td style={{ padding: '5px 8px' }}>{c.hemisphere}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 8, padding: '0 4px' }}>
          {localization.method}<br />{localization.note}
        </div>
      </>
    )
  }

  /* ─── False Alarm Review ────────────────────────────────── */
  const renderFalseAlarm = () => {
    if (!falseAlarm?.available) return <div style={{ padding: 16, color: '#94a3b8' }}>No annotated seizure EDF available for false-alarm review.</div>

    const sens = falseAlarm.sensitivity
    const faRate = falseAlarm.false_alarms_per_hour
    const verdict = falseAlarm.verdict || ''
    const verdictColor = verdict.includes('acceptable') ? '#16a34a' : '#dc2626'
    const faWindows = falseAlarm.false_alarm_windows || []

    // Pie: TP vs FP windows
    const detPie = [
      { name: 'True Positive', value: falseAlarm.true_positive_windows || 0 },
      { name: 'False Alarm', value: falseAlarm.false_alarms || 0 }
    ]

    // Bar: FA windows timeline (first 20)
    const faBar = faWindows.map(w => ({ time: `${w.time_s}s`, window: w.window }))

    return (
      <>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
          {kpi('Sensitivity', sens != null ? `${(sens * 100).toFixed(0)}%` : '—')}
          {kpi('FA / Hour', faRate != null ? faRate.toFixed(1) : '—', faRate != null && faRate <= 6 ? '≤ 6 target' : '> 6 ⚠')}
          {kpi('Seizures', `${falseAlarm.seizures_detected}/${falseAlarm.n_seizures_annotated}`)}
          {kpi('TP Windows', falseAlarm.true_positive_windows)}
          {kpi('False Alarms', falseAlarm.false_alarms)}
          {kpi('Duration', `${falseAlarm.recording_hours} hr`)}
        </div>

        <div style={{ background: verdictColor === '#16a34a' ? '#f0fdf4' : '#fef2f2', border: `1px solid ${verdictColor}33`, borderRadius: 8, padding: '10px 14px', marginBottom: 16, fontSize: 13, color: verdictColor, fontWeight: 600 }}>
          Verdict: {verdict}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          {card('Detection Breakdown (TP vs FA)',
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie data={detPie} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                  <Cell fill="#4caf50" />
                  <Cell fill="#f44336" />
                </Pie>
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          )}

          {card('False Alarm Windows (time in recording)',
            faBar.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={faBar}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" fontSize={10} angle={-45} textAnchor="end" height={50} />
                  <YAxis fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="window" fill="#ff9800" radius={[4, 4, 0, 0]} name="Window #" />
                </BarChart>
              </ResponsiveContainer>
            ) : <div style={{ padding: 16, color: '#94a3b8', textAlign: 'center' }}>No false alarm windows to display.</div>
          )}

          {card('Detector Configuration',
            <div style={{ fontSize: 12 }}>
              <div style={{ padding: 8, background: '#f1f5f9', borderRadius: 6, marginBottom: 8 }}>
                <strong>Method:</strong> {falseAlarm.detector?.method}
              </div>
              <div style={{ padding: 8, background: '#f1f5f9', borderRadius: 6 }}>
                <strong>Threshold k:</strong> {falseAlarm.detector?.threshold_k}
              </div>
            </div>
          )}

          {card('Clinical Context',
            <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6 }}>
              <div style={{ marginBottom: 8 }}>
                <strong>Sensitivity</strong> measures how many real seizures the detector catches.
                <strong> False-alarm rate</strong> measures spurious detections per hour — clinical target is &lt; 0.15/hr for wearable alerts, but simple power detectors are typically 1–10/hr.
              </div>
              <div style={{ color: '#94a3b8', fontSize: 11 }}>{falseAlarm.note}</div>
            </div>
          )}
        </div>
      </>
    )
  }

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <h2 style={{ margin: '0 0 6px', color: '#0f172a' }}>Expert Dashboards</h2>
      <p style={{ margin: '0 0 16px', fontSize: 13, color: '#64748b' }}>
        Clinical expert tools — montage comparison, seizure localization, and false-alarm analysis from real EDF recordings.
      </p>

      {/* Panel selector */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {panels.map(p => (
          <button
            key={p.id}
            onClick={() => setPanel(p.id)}
            style={{
              padding: '8px 16px', borderRadius: 8, fontSize: 13, fontWeight: 600,
              border: panel === p.id ? '2px solid #3b82f6' : '1px solid #e2e8f0',
              background: panel === p.id ? '#eff6ff' : '#fff',
              color: panel === p.id ? '#1e40af' : '#475569',
              cursor: 'pointer'
            }}
          >
            {p.label}
          </button>
        ))}
      </div>

      {panel === 'montage' && renderMontage()}
      {panel === 'localization' && renderLocalization()}
      {panel === 'falsealarm' && renderFalseAlarm()}
    </div>
  )
}
