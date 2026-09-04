'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// SCA Atlas color palette
const COLOR  = '#1a237e';  // deep indigo — neurology/ataxia
const LIGHT  = '#e8eaf6';  // indigo tint
const COLOR2 = '#880e4f';  // deep pink — FXN/Friedreich
const COLOR3 = '#1b5e20';  // green — treatable (acetazolamide EA2, omaveloxolone)
const COLOR4 = '#b71c1c';  // red — cardiac/severe
const COLOR5 = '#e65100';  // orange — visual loss SCA7
const COLOR6 = '#006064';  // teal — RFC1/CANVAS

const GENE_COLORS = {
  FXN:     '#880e4f',  // FRDA — most common AR ataxia (deep pink)
  ATXN1:   '#1a237e',  // SCA1 — olivopontocerebellar (indigo)
  ATXN2:   '#4a148c',  // SCA2 — slow saccades / ALS (purple)
  ATXN3:   '#1565c0',  // SCA3/MJD — most common SCA (blue)
  CACNA1A: '#1b5e20',  // SCA6/EA2 — acetazolamide (green)
  ATXN7:   '#e65100',  // SCA7 — visual loss (orange)
  TBP:     '#f57f17',  // SCA17 — Huntington DDx (amber)
  RFC1:    '#006064',  // CANVAS — vestibular + cough (teal)
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'info', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

export default function ScaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sca-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/sca-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sca-atlas/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-primary" /></div>;
  if (err) return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  return (
    <div className="container-fluid py-3" style={{ background: LIGHT, minHeight: '100vh' }}>
      <div className="container">
        {/* Header */}
        <div className="card mb-4 shadow-sm border-0" style={{ background: COLOR, color: '#fff' }}>
          <div className="card-body py-3">
            <h3 className="mb-0 fw-bold">🧬 {ov?.title}</h3>
            <div className="opacity-75 small mt-1">{ov?.subtitle}</div>
            <div className="opacity-60 small">320 patients · 8 genes · seeds {ov?.seeds_used} · Hereditary Ataxia Reference</div>
          </div>
        </div>

        {/* Critical Alerts */}
        <AlertBox type="info" title="Key Diagnostic Triggers">
          <strong>ATXN2 SLOW SACCADES:</strong> pathognomonic — test ATXN2 first when slow saccades seen ·
          <strong> ATXN7 VISUAL LOSS + ATAXIA:</strong> only polyQ SCA with retinal dystrophy ·
          <strong> TBP: Huntington-like with normal HTT</strong> → test SCA17/TBP ·
          <strong> RFC1/CANVAS:</strong> add to ALL late-onset ataxia panels (carrier 1:80)
        </AlertBox>
        <AlertBox type="success" title="Treatable Conditions">
          <strong>FRDA/FXN:</strong> Omaveloxolone (Skyclarys FDA 2023) — first disease-modifying Rx ·
          <strong> EA2/CACNA1A:</strong> Acetazolamide HIGHLY EFFECTIVE for episodic attacks · Friedreich HCM: annual echo mandatory
        </AlertBox>

        {/* Tabs */}
        <ul className="nav nav-tabs mb-3">
          {TABS.map(t => (
            <li className="nav-item" key={t}>
              <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`} onClick={() => setTab(t)}>{t}</button>
            </li>
          ))}
        </ul>

        {/* Overview Tab */}
        {tab === 'Overview' && ov && (
          <>
            <div className="row mb-3">
              <KPI label="Total Patients" value={ov.total_patients} />
              <KPI label="Genes" value={ov.genes?.length} />
              <KPI label="Avg Onset (y)" value={ov.avg_onset_y} />
              <KPI label="Avg Dx Delay (y)" value={ov.avg_dx_delay_y} color={COLOR4} />
              <KPI label="Cardiac (FXN)" value={`${ov.cardiac_pct}%`} color={COLOR4} />
              <KPI label="Visual Loss (ATXN7)" value={`${ov.visual_loss_pct}%`} color={COLOR5} />
              <KPI label="Vestibular (RFC1)" value={`${ov.vestibular_pct}%`} color={COLOR6} />
              <KPI label="Chronic Cough (RFC1)" value={`${ov.chronic_cough_pct}%`} color={COLOR6} />
            </div>

            <div className="row">
              {/* Severity */}
              <div className="col-md-4 mb-3">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Severity Distribution</h6>
                    {Object.entries(ov.severity_distribution || {}).map(([s, c]) => (
                      <BarRow key={s} label={s} pct={Math.round(100*c/ov.total_patients)}
                        color={s==='Severe'?COLOR4:s==='Moderate'?'#f57f17':COLOR3} note={`n=${c}`} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Ataxia Groups */}
              <div className="col-md-4 mb-3">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Ataxia Groups</h6>
                    {Object.entries(ov.ataxia_groups || {}).map(([g, c]) => (
                      <BarRow key={g} label={g} pct={Math.round(100*c/ov.total_patients)}
                        color={g.includes('Recessive')?COLOR2:COLOR} note={`n=${c}`} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Per-Gene Summary */}
              <div className="col-md-4 mb-3">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Patients per Gene</h6>
                    {Object.entries(ov.gene_summary || {}).map(([gene, n]) => (
                      <BarRow key={gene} label={gene} pct={Math.round(100*n/ov.total_patients)}
                        color={GENE_COLORS[gene]||COLOR} note={`n=${n}`} />
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Key Facts */}
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Key Clinical Facts</h6>
                <ul className="list-unstyled mb-0">
                  {(ov.key_facts || []).map((f, i) => (
                    <li key={i} className="mb-1 small">🧬 {f}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Critical Distinctions */}
            <div className="card shadow-sm mb-3">
              <div className="card-body">
                <h6 className="fw-bold mb-2" style={{ color: COLOR4 }}>Critical DDx Distinctions</h6>
                {Object.entries(ov.critical_distinctions || {}).map(([k, v]) => (
                  <div key={k} className="mb-2 small">
                    <span className="fw-bold">{k}:</span> {v}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* Gene Table Tab */}
        {tab === 'Gene Table' && bd && (
          <div className="table-responsive">
            <table className="table table-sm table-hover small shadow-sm">
              <thead style={{ background: COLOR, color: '#fff' }}>
                <tr>
                  <th>Gene</th><th>Subtype</th><th>Inheritance</th><th>Locus</th>
                  <th>AA</th><th>OMIM</th><th>Repeat</th><th>Onset (y)</th>
                  <th>Dx Delay (y)</th><th>Pts</th>
                </tr>
              </thead>
              <tbody>
                {(bd.breakdown || []).map(g => (
                  <tr key={g.gene}>
                    <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene]||COLOR }}>{g.gene}</span></td>
                    <td style={{ maxWidth: 200, whiteSpace: 'normal' }}>{g.subtype}</td>
                    <td><span className={`badge ${g.inheritance?.startsWith('X')?'bg-warning text-dark':g.inheritance?.startsWith('AR')?'bg-secondary':'bg-primary'}`}>
                      {g.inheritance?.split('(')[0].trim()}
                    </span></td>
                    <td>{g.locus}</td>
                    <td>{g.aa}</td>
                    <td><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer" className="text-decoration-none">{g.omim_disease}</a></td>
                    <td className="small text-muted">{g.repeat_type?.split(' ')[0]}</td>
                    <td>{g.avg_onset_y}</td>
                    <td className={g.avg_dx_delay_y > 5 ? 'text-danger fw-bold' : ''}>{g.avg_dx_delay_y}</td>
                    <td>{g.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Clinical Atlas Tab */}
        {tab === 'Clinical Atlas' && bd && (
          <div className="row">
            {(bd.breakdown || []).map(g => (
              <div className="col-md-6 col-lg-4 mb-3" key={g.gene}>
                <div className="card h-100 shadow-sm" style={{ borderLeft: `4px solid ${GENE_COLORS[g.gene]||COLOR}` }}>
                  <div className="card-body">
                    <div className="d-flex justify-content-between align-items-start mb-2">
                      <h6 className="fw-bold mb-0" style={{ color: GENE_COLORS[g.gene]||COLOR }}>{g.gene}</h6>
                      <span className="badge" style={{ background: GENE_COLORS[g.gene]||COLOR, fontSize: '0.65rem' }}>
                        {g.n_patients} pts
                      </span>
                    </div>
                    <div className="small text-muted mb-2">{g.subtype}</div>
                    <div className="small mb-1"><strong>Locus:</strong> {g.locus} · {g.aa}</div>
                    <div className="small mb-1"><strong>Inheritance:</strong> {g.inheritance?.split('.')[0]}</div>
                    <div className="small mb-1"><strong>Repeat:</strong> {g.repeat_type || 'N/A'}</div>
                    <div className="small mb-1"><strong>Avg onset:</strong> {g.avg_onset_y}y · <strong>Dx delay:</strong> {g.avg_dx_delay_y}y</div>
                    {g.cardiac_pct > 0 && <div className="small text-danger mb-1">❤️ Cardiac disease: {g.cardiac_pct}%</div>}
                    {g.visual_loss_pct > 0 && <div className="small mb-1" style={{color:COLOR5}}>👁️ Visual loss: {g.visual_loss_pct}%</div>}
                    {g.vestibular_pct > 0 && <div className="small mb-1" style={{color:COLOR6}}>🫀 Vestibular areflexia: {g.vestibular_pct}%</div>}
                    {g.cough_pct > 0 && <div className="small mb-1" style={{color:COLOR6}}>🌬️ Chronic cough: {g.cough_pct}%</div>}
                    <div className="mt-2">
                      {Object.entries(g.severity_distribution||{}).map(([s, n]) => (
                        <span key={s} className="badge me-1" style={{
                          background: s==='Severe'?COLOR4:s==='Moderate'?'#f57f17':COLOR3,
                          fontSize: '0.65rem'
                        }}>{s}: {n}</span>
                      ))}
                    </div>
                    {g.top_treatments?.length > 0 && (
                      <div className="mt-2 small text-muted">
                        <strong>Tx:</strong> {g.top_treatments.slice(0,1).map(t => t.tx).join('; ')}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Definitions Tab */}
        {tab === 'Definitions' && defs && (
          <div>
            {(defs.definitions || []).map((d, i) => (
              <div className="card mb-3 shadow-sm" key={i}>
                <div className="card-body">
                  <h6 className="fw-bold mb-2" style={{ color: COLOR }}>{d.term}</h6>
                  <p className="mb-0 small" style={{ whiteSpace: 'pre-wrap' }}>{d.definition}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
