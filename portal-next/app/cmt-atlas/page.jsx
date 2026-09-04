'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// CMT Atlas color palette — neuromuscular / hereditary neuropathy
const COLOR  = '#1b5e20';  // dark green — neuromuscular
const LIGHT  = '#e8f5e9';  // green tint
const COLOR2 = '#1a237e';  // indigo — demyelinating
const COLOR3 = '#e65100';  // orange — axonal
const COLOR4 = '#b71c1c';  // red — severe/AR
const COLOR5 = '#4a148c';  // purple — X-linked
const COLOR6 = '#37474f';  // blue-grey — intermediate

const GENE_COLORS = {
  PMP22:  '#1a237e',  // CMT1A/HNPP — most common demyelinating (indigo)
  MPZ:    '#283593',  // CMT1B — P0 protein (blue)
  GJB1:   '#6a1b9a',  // CMTX1 — X-linked connexin 32 (purple)
  MFN2:   '#e65100',  // CMT2A — most common axonal (orange)
  SH3TC2: '#b71c1c',  // CMT4C — severe AR (red)
  GDAP1:  '#c62828',  // CMT4A — vocal cord paresis (deep red)
  HSPB1:  '#2e7d32',  // CMT2F — HSP27/motor (green)
  NEFL:   '#00695c',  // CMT2E/1F — NF-L biomarker (teal)
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

export default function CmtAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cmt-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/cmt-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cmt-atlas/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border text-success" /></div>;
  if (err) return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  return (
    <div className="container-fluid py-3" style={{ background: LIGHT, minHeight: '100vh' }}>
      <div className="container">
        {/* Header */}
        <div className="card mb-4 shadow-sm border-0" style={{ background: COLOR, color: '#fff' }}>
          <div className="card-body py-3">
            <h3 className="mb-0 fw-bold">🦵 {ov?.title}</h3>
            <div className="opacity-75 small mt-1">{ov?.subtitle}</div>
            <div className="opacity-60 small">320 patients · 8 genes · seeds {ov?.seeds_used} · Hereditary Neuropathy Reference</div>
          </div>
        </div>

        {/* Critical Alerts */}
        <AlertBox type="info" title="Key Diagnostic Triggers">
          <strong>MLPA/aCGH FIRST</strong> for CMT1 (PMP22 copy number) ·
          <strong> NCV &lt;38 m/s = demyelinating (CMT1);</strong> NCV &gt;38 m/s with low CMAP = axonal (CMT2) ·
          <strong> CMTX1: intermediate NCVs (25-45 m/s)</strong> — X-linked; no male-to-male transmission
        </AlertBox>
        <AlertBox type="warning" title="Critical Pitfalls">
          <strong>ASCORBIC ACID: NEGATIVE</strong> — three large RCTs failed for CMT1A; NOT recommended ·
          <strong> GDAP1 vocal cord paresis:</strong> ENT + respiratory review mandatory ·
          <strong> SH3TC2 scoliosis:</strong> early orthopedic referral (Cobb &gt;40° → surgery)
        </AlertBox>
        <AlertBox type="success" title="Treatment Mainstays (No Disease-Modifying Rx Approved 2026)">
          <strong>AFO/ankle-foot orthoses:</strong> mainstay for foot drop ·
          <strong> Physiotherapy:</strong> evidence-based for strength + function ·
          <strong> PXT3003 Phase 3 ongoing</strong> for CMT1A (baclofen+naltrexone+sorbitol)
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
              <KPI label="Pes Cavus (%)" value={`${ov.pes_cavus_pct}%`} color={COLOR2} />
              <KPI label="Demyelinating (%)" value={`${ov.demyelinating_pct}%`} color={COLOR2} />
              <KPI label="Axonal (%)" value={`${ov.axonal_pct}%`} color={COLOR3} />
              <KPI label="AR CMT (%)" value={`${ov.ar_cmt_pct}%`} color={COLOR4} />
            </div>

            <div className="row">
              {/* Severity */}
              <div className="col-md-4 mb-3">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Severity Distribution</h6>
                    {Object.entries(ov.severity_distribution || {}).map(([s, c]) => (
                      <BarRow key={s} label={s} pct={Math.round(100*c/ov.total_patients)}
                        color={s==='Severe'?COLOR4:s==='Moderate'?'#f57f17':COLOR} note={`n=${c}`} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Neuropathy Groups */}
              <div className="col-md-4 mb-3">
                <div className="card h-100 shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>CMT Groups</h6>
                    {Object.entries(ov.neuropathy_groups || {}).map(([g, c]) => (
                      <BarRow key={g} label={g} pct={Math.round(100*c/ov.total_patients)}
                        color={g.includes('AR')?COLOR4:g.includes('X-Linked')?COLOR5:g.includes('Axonal')?COLOR3:COLOR2}
                        note={`n=${c}`} />
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

            {/* Clinical Flags Summary */}
            <div className="row mb-3">
              <div className="col-md-6">
                <div className="card shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Feature Summary</h6>
                    <BarRow label="Pes Cavus (all genes)" pct={ov.pes_cavus_pct} color={COLOR2} note={`n=${ov.pes_cavus_n}`} />
                    <BarRow label="Scoliosis (SH3TC2 dominant)" pct={ov.scoliosis_pct} color={COLOR4} note={`n=${ov.scoliosis_n}`} />
                    <BarRow label="Vocal Cord Paresis (GDAP1)" pct={ov.vocal_cord_pct} color={COLOR4} note={`n=${ov.vocal_cord_n}`} />
                    <BarRow label="Optic Atrophy (MFN2)" pct={ov.optic_atrophy_pct} color={COLOR3} note={`n=${ov.optic_atrophy_n}`} />
                    <BarRow label="CNS WM Lesions (GJB1 males)" pct={ov.cns_lesion_pct} color={COLOR5} note={`n=${ov.cns_lesion_n}`} />
                  </div>
                </div>
              </div>
              <div className="col-md-6">
                <div className="card shadow-sm">
                  <div className="card-body">
                    <h6 className="fw-bold mb-3" style={{ color: COLOR }}>NCV Type Distribution</h6>
                    <BarRow label="Demyelinating (NCV <38 m/s)" pct={ov.demyelinating_pct} color={COLOR2} note={`n=${ov.demyelinating_n}`} />
                    <BarRow label="Axonal (NCV >38 m/s)" pct={ov.axonal_pct} color={COLOR3} note={`n=${ov.axonal_n}`} />
                    <BarRow label="Intermediate / Variable" pct={Math.round(100-(ov.demyelinating_pct||0)-(ov.axonal_pct||0))} color={COLOR6} />
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
                    <li key={i} className="mb-1 small">🦵 {f}</li>
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
                  <th>AA</th><th>OMIM</th><th>NVC Range</th><th>Type</th>
                  <th>Onset (y)</th><th>Dx Delay (y)</th><th>Pts</th>
                </tr>
              </thead>
              <tbody>
                {(bd.breakdown || []).map(g => (
                  <tr key={g.gene}>
                    <td><span className="fw-bold" style={{ color: GENE_COLORS[g.gene]||COLOR }}>{g.gene}</span></td>
                    <td style={{ maxWidth: 180, whiteSpace: 'normal' }}>{g.subtype}</td>
                    <td>
                      <span className={`badge ${
                        g.inheritance?.startsWith('X') ? 'bg-warning text-dark' :
                        g.inheritance?.startsWith('AR') || g.inheritance?.startsWith('Autosomal Recessive') ? 'bg-danger' :
                        'bg-primary'
                      }`}>
                        {g.inheritance?.split('.')[0].trim().substring(0, 20)}
                      </span>
                    </td>
                    <td>{g.locus}</td>
                    <td>{g.aa}</td>
                    <td><a href={`https://omim.org/entry/${g.omim_disease}`} target="_blank" rel="noreferrer" className="text-decoration-none">{g.omim_disease}</a></td>
                    <td className="small text-muted">{g.nvc_range}</td>
                    <td>
                      <span className={`badge ${
                        g.neuropathy_type === 'demyelinating' ? 'bg-primary' :
                        g.neuropathy_type === 'axonal' ? 'bg-warning text-dark' :
                        'bg-secondary'
                      }`}>{g.neuropathy_type}</span>
                    </td>
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
                    <div className="small mb-1"><strong>NCV:</strong> {g.nvc_range} ({g.neuropathy_type})</div>
                    <div className="small mb-1"><strong>Avg onset:</strong> {g.avg_onset_y}y · <strong>Dx delay:</strong> {g.avg_dx_delay_y}y</div>
                    {g.pes_cavus_pct > 0 && <div className="small mb-1" style={{color:COLOR2}}>🦴 Pes cavus: {g.pes_cavus_pct}%</div>}
                    {g.scoliosis_pct > 0 && <div className="small text-danger mb-1">⚕️ Scoliosis: {g.scoliosis_pct}%</div>}
                    {g.vocal_cord_pct > 0 && <div className="small text-danger mb-1">🎙️ Vocal cord paresis: {g.vocal_cord_pct}%</div>}
                    {g.optic_atrophy_pct > 0 && <div className="small mb-1" style={{color:COLOR3}}>👁️ Optic atrophy: {g.optic_atrophy_pct}%</div>}
                    {g.cns_lesion_pct > 0 && <div className="small mb-1" style={{color:COLOR5}}>🧠 CNS WM lesions: {g.cns_lesion_pct}%</div>}
                    <div className="mt-2">
                      {Object.entries(g.severity_distribution||{}).map(([s, n]) => (
                        <span key={s} className="badge me-1" style={{
                          background: s==='Severe'?COLOR4:s==='Moderate'?'#f57f17':COLOR,
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
