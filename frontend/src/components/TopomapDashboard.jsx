import React, { useState, useEffect } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, Cell
} from 'recharts'

const API_URL = (typeof window !== 'undefined' && window._env_?.REACT_APP_API_URL) || 'http://localhost:8010'
const BAND_COLORS = {
  delta: '#1e88e5',
  theta: '#7c4dff',
  alpha: '#4caf50',
  beta: '#ff9800',
  gamma: '#f44336',
}

export default function TopomapDashboard() {
  const [overview, setOverview] = useState(null)
  const [electrodes, setElectrodes] = useState(null)
  const [asymmetry, setAsymmetry] = useState(null)
  const [defs, setDefs] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeBand, setActiveBand] = useState('alpha')

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [ov, el, asym, df] = await Promise.all([
          axios.get(`${API_URL}/api/topomap/overview?seconds=30`),
          axios.get(`${API_URL}/api/topomap/electrodes`),
          axios.get(`${API_URL}/api/topomap/asymmetry?seconds=30`),
          axios.get(`${API_URL}/api/topomap/definitions`)
        ])
        setOverview(ov.data)
        setElectrodes(el.data)
        setAsymmetry(asym.data)
        setDefs(df.data)
      } catch (err) {
        setError(err.message || 'Failed to load topographic data')
      }
      setLoading(false)
    }
    load()
  }, [])

  if (loading) return <div style={{padding: 32, textAlign: 'center'}}>Loading topographic maps...</div>
  if (error) return <div style={{padding: 32, color: '#f44336'}}>Error: {error}</div>
  if (!overview?.available) return <div style={{padding: 32}}>No EDF data available for topographic mapping.</div>

  const bandKeys = Object.keys(BAND_COLORS)

  // Bar data: per-channel power for active band
  const barData = overview.electrodes.map(e => ({
    channel: e.channel,
    power: e[activeBand] || 0,
  })).sort((a, b) => b.power - a.power)

  // Radar data: band summary means
  const radarData = bandKeys.map(b => ({
    band: b.charAt(0).toUpperCase() + b.slice(1),
    power: overview.band_summary?.[b]?.mean || 0,
  }))

  // Head map SVG data
  const headPoints = overview.electrodes.map(e => ({
    ...e,
    svgX: 150 + e.x * 100,
    svgY: 150 - e.y * 100,
    power: e[activeBand] || 0,
  }))

  // Asymmetry bar data
  const asymData = (asymmetry?.pairs || []).map(p => ({
    region: p.region.charAt(0).toUpperCase() + p.region.slice(1),
    asymmetry: p.asymmetry,
    left: p.left_alpha,
    right: p.right_alpha,
  }))

  // Power-to-color for head map
  const maxPower = Math.max(...headPoints.map(p => p.power), 0.001)
  const getColor = (val) => {
    const ratio = val / maxPower
    const r = Math.round(30 + ratio * 225)
    const g = Math.round(136 - ratio * 80)
    const b = Math.round(229 - ratio * 180)
    return `rgb(${r},${g},${b})`
  }

  return (
    <div style={{padding: '16px 24px'}}>
      <h2 style={{margin: '0 0 4px', fontSize: 20}}>Topographic Power Maps (Nilearn / MNE)</h2>
      <p style={{color: '#666', margin: '0 0 16px', fontSize: 13}}>
        Real EEG band power from {overview.n_channels_mapped} electrodes mapped to 10-20 positions
        &nbsp;|&nbsp;{overview.duration_seconds}s segment&nbsp;|&nbsp;{overview.sfreq} Hz
      </p>

      {/* Band selector */}
      <div style={{display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap'}}>
        {bandKeys.map(b => (
          <button key={b} onClick={() => setActiveBand(b)}
            style={{
              padding: '6px 16px', borderRadius: 6, fontSize: 13, cursor: 'pointer',
              border: activeBand === b ? `2px solid ${BAND_COLORS[b]}` : '1px solid #ddd',
              background: activeBand === b ? BAND_COLORS[b] + '18' : '#fff',
              color: activeBand === b ? BAND_COLORS[b] : '#555', fontWeight: activeBand === b ? 700 : 400,
            }}>
            {b.charAt(0).toUpperCase() + b.slice(1)}
          </button>
        ))}
      </div>

      <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16}}>

        {/* Head map (SVG) */}
        <div style={{background: '#fff', borderRadius: 10, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,.08)'}}>
          <h3 style={{margin: '0 0 8px', fontSize: 15}}>Head Map — {activeBand.charAt(0).toUpperCase() + activeBand.slice(1)} Power</h3>
          <svg viewBox="0 0 300 300" style={{width: '100%', maxWidth: 320, margin: '0 auto', display: 'block'}}>
            {/* Head outline */}
            <ellipse cx="150" cy="150" rx="120" ry="130" fill="none" stroke="#ccc" strokeWidth="2" />
            {/* Nose */}
            <polygon points="150,10 140,35 160,35" fill="none" stroke="#ccc" strokeWidth="1.5" />
            {/* Ears */}
            <ellipse cx="25" cy="150" rx="8" ry="20" fill="none" stroke="#ccc" strokeWidth="1.5" />
            <ellipse cx="275" cy="150" rx="8" ry="20" fill="none" stroke="#ccc" strokeWidth="1.5" />
            {/* Electrodes */}
            {headPoints.map((p, i) => (
              <g key={i}>
                <circle cx={p.svgX} cy={p.svgY} r={14} fill={getColor(p.power)} opacity={0.75}
                  stroke="#fff" strokeWidth={1.5} />
                <text x={p.svgX} y={p.svgY + 1} textAnchor="middle" dominantBaseline="middle"
                  fontSize={8} fill="#fff" fontWeight="bold">{p.channel}</text>
              </g>
            ))}
            {/* Color bar label */}
            <text x="280" y="290" fontSize="8" fill="#999" textAnchor="end">
              max: {maxPower.toFixed(3)}
            </text>
          </svg>
        </div>

        {/* Band summary radar */}
        <div style={{background: '#fff', borderRadius: 10, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,.08)'}}>
          <h3 style={{margin: '0 0 8px', fontSize: 15}}>Band Power Distribution</h3>
          <ResponsiveContainer width="100%" height={260}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="band" tick={{fontSize: 12}} />
              <PolarRadiusAxis tick={{fontSize: 10}} />
              <Radar dataKey="power" stroke="#1e88e5" fill="#1e88e5" fillOpacity={0.3} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Per-channel bar chart */}
        <div style={{background: '#fff', borderRadius: 10, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,.08)'}}>
          <h3 style={{margin: '0 0 8px', fontSize: 15}}>Per-Channel {activeBand} Power</h3>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="channel" tick={{fontSize: 10}} />
              <YAxis tick={{fontSize: 10}} />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <Bar dataKey="power" fill={BAND_COLORS[activeBand]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Alpha asymmetry */}
        <div style={{background: '#fff', borderRadius: 10, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,.08)'}}>
          <h3 style={{margin: '0 0 8px', fontSize: 15}}>Alpha Asymmetry (Davidson 1998)</h3>
          {asymData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={asymData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tick={{fontSize: 10}} />
                <YAxis type="category" dataKey="region" tick={{fontSize: 11}} width={75} />
                <Tooltip formatter={(v) => v.toFixed(4)} />
                <Bar dataKey="asymmetry" fill="#7c4dff" radius={[0, 4, 4, 0]}>
                  {asymData.map((entry, i) => (
                    <Cell key={i} fill={entry.asymmetry >= 0 ? '#4caf50' : '#f44336'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p style={{color: '#999', fontSize: 13}}>Insufficient paired electrodes for asymmetry analysis.</p>
          )}
        </div>
      </div>

      {/* Band definitions table */}
      {defs?.bands && (
        <div style={{marginTop: 16, background: '#fff', borderRadius: 10, padding: 16, boxShadow: '0 1px 4px rgba(0,0,0,.08)'}}>
          <h3 style={{margin: '0 0 8px', fontSize: 15}}>Band Definitions & Clinical Relevance</h3>
          <table style={{width: '100%', borderCollapse: 'collapse', fontSize: 13}}>
            <thead>
              <tr style={{borderBottom: '2px solid #e0e0e0'}}>
                <th style={{padding: '8px 12px', textAlign: 'left'}}>Band</th>
                <th style={{padding: '8px 12px', textAlign: 'left'}}>Range</th>
                <th style={{padding: '8px 12px', textAlign: 'left'}}>Role</th>
                <th style={{padding: '8px 12px', textAlign: 'left'}}>Clinical Relevance</th>
              </tr>
            </thead>
            <tbody>
              {defs.bands.map((b, i) => (
                <tr key={i} style={{borderBottom: '1px solid #f0f0f0'}}>
                  <td style={{padding: '8px 12px', fontWeight: 600, color: Object.values(BAND_COLORS)[i]}}>
                    {b.name}
                  </td>
                  <td style={{padding: '8px 12px'}}>{b.range}</td>
                  <td style={{padding: '8px 12px'}}>{b.role}</td>
                  <td style={{padding: '8px 12px', color: '#666'}}>{b.clinical}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Tools attribution */}
      <div style={{marginTop: 12, fontSize: 12, color: '#999', textAlign: 'center'}}>
        Nilearn (Abraham et al., 2014) + MNE-Python (Gramfort et al., 2013)
        &nbsp;|&nbsp;10-20 system (Jasper, 1958)
        {asymmetry?.reference && <span>&nbsp;|&nbsp;{asymmetry.reference}</span>}
      </div>
    </div>
  )
}
