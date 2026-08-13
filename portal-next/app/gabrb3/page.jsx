'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — GABRB3 / GABA-A
const ACCENT2 = '#b71c1c';   // dark red — LTG CI / danger / BDZ warning
const ACCENT3 = '#1b5e20';   // dark green — KD / ACTH success / precision
const ACCENT4 = '#4a148c';   // purple — NREM tonic / dominant-negative

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function PctBar({ label, pct, color = ACCENT }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-header fw-bold" style={{ backgroundColor: '#eef2fb', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function TabBtn({ label, active, onClick }) {
  return (
    <button
      className={`btn btn-sm me-1 mb-1 ${active ? 'btn-primary' : 'btn-outline-secondary'}`}
      style={active ? { backgroundColor: ACCENT, borderColor: ACCENT } : {}}
      onClick={onClick}
    >{label}</button>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const k = data.kpis || {};

  return (
    <>
      <div className="alert alert-primary mb-4" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        <strong>🧬 {data.syndrome}</strong><br />
        Gene: <strong>{data.gene}</strong> · {data.inheritance}<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      {/* Critical Alerts */}
      {(data.critical_alerts || []).map((a, i) => (
        <div key={i} className={`alert alert-${a.color} py-2 mb-2`} style={{ fontSize: 13 }}>
          <strong>⚠️ {a.severity}:</strong> {a.alert}<br />
          <span className="text-muted small">→ {a.action}</span>
        </div>
      ))}

      {/* KPI tiles */}
      <div className="row g-2 mb-4 mt-2">
        {[
          ['Patients', k.total_patients, ACCENT],
          ['ACTH Given', `${k.acth_pct ?? '—'}%`, '#7b1fa2'],
          ['VGB Given', `${k.vgb_pct ?? '—'}%`, '#0277bd'],
          ['On KD', `${k.kd_pct ?? '—'}%`, ACCENT3],
          ['LGS Evolved', `${k.lgs_pct ?? '—'}%`, ACCENT4],
          ['Drug-Resistant', `${k.drug_resistant_pct ?? '—'}%`, ACCENT2],
          ['BDZ ↓ Efficacy', `${k.bdz_reduced_pct ?? '—'}%`, '#e65100'],
          ['Spasm-Free', `${k.spasm_free_pct ?? '—'}%`, '#2e7d32'],
          ['ERG Done', k.erg_done, '#5d4037'],
          ['Etiology Classes', k.etiology_classes, ACCENT],
          ['Seizure Types', k.seizure_types, ACCENT],
          ['Treatments', k.treatments, '#006064'],
        ].map(([label, val, color]) => (
          <KPI key={label} label={label} value={val ?? '—'} color={color} />
        ))}
      </div>

      {/* Pathway Summary */}
      <SectionCard title="🗺️ Clinical Management Pathway" borderColor={ACCENT3}>
        <p className="small mb-0" style={{ lineHeight: 1.7 }}>{data.pathway_summary}</p>
      </SectionCard>

      {/* EEG Hallmarks */}
      {data.eeg_hallmarks && (
        <SectionCard title="🧠 EEG Hallmarks — GABRB3-DEE28" borderColor={ACCENT4}>
          <ul className="mb-0 small">
            {data.eeg_hallmarks.map((h, i) => <li key={i}>{h}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* Standards */}
      <SectionCard title="📋 Evidence Standards" borderColor="#546e7a">
        <ul className="mb-0 small">
          {(data.standards || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function EtiologyTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [sel, setSel] = useState(null);

  return (
    <>
      <SectionCard title="📊 Etiology Distribution — 5 Classes (N=41)" borderColor={ACCENT}>
        {(data.etiology_distribution || []).map((e, i) => (
          <div key={i} className="mb-3">
            <div
              className="d-flex justify-content-between align-items-center mb-1 cursor-pointer"
              onClick={() => setSel(sel === i ? null : i)}
              style={{ cursor: 'pointer' }}
            >
              <span className="small fw-semibold">{e.etiology}</span>
              <span className="badge" style={{ backgroundColor: ACCENT }}>{e.n} / {e.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 12 }}>
              <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: ACCENT }} />
            </div>
            {sel === i && (
              <div className="border rounded p-2 mt-1 bg-light small">
                <strong>Mechanism:</strong> {e.mechanism_short}<br />
                <strong>EEG:</strong> {e.eeg_signature_short}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Summary Table (N=41)" borderColor={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Age(mo)</th><th>Onset(mo)</th><th>Etiology</th>
                <th>ACTH</th><th>ACTH Resp</th><th>VGB</th><th>KD</th>
                <th>LGS</th><th>BDZ↓</th><th>Spasm-Free</th><th>POLG</th><th>Meds</th><th>Control</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map(p => (
                <tr key={p.id}>
                  <td><span className="badge" style={{ backgroundColor: ACCENT }}>{p.id}</span></td>
                  <td>{p.age_months}</td>
                  <td>{p.onset_months}</td>
                  <td><span className="text-truncate d-inline-block" style={{ maxWidth: 80 }} title={p.etiology_category}>{p.etiology_category.split('-').slice(0,2).join('-')}</span></td>
                  <td><span className={`badge ${p.acth_given ? 'bg-success' : 'bg-secondary'}`}>{p.acth_given ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.acth_response === 'complete' ? 'bg-success' : p.acth_response === 'partial' ? 'bg-warning text-dark' : p.acth_response === 'none' ? 'bg-danger' : 'bg-secondary'}`}>{p.acth_response ?? '—'}</span></td>
                  <td><span className={`badge ${p.vgb_given ? 'bg-info text-dark' : 'bg-secondary'}`}>{p.vgb_given ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.on_kd ? 'bg-success' : 'bg-secondary'}`}>{p.on_kd ? `${p.kd_ketosis_mmol}mmol` : 'N'}</span></td>
                  <td><span className={`badge ${p.lgs_evolved ? 'bg-danger' : 'bg-success'}`}>{p.lgs_evolved ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.bdz_reduced_efficacy ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.bdz_reduced_efficacy ? '↓BDZ' : '—'}</span></td>
                  <td><span className={`badge ${p.spasm_free ? 'bg-success' : 'bg-danger'}`}>{p.spasm_free ? 'Y' : 'N'}</span></td>
                  <td><span className={`badge ${p.polg_tested === 'Y' ? 'bg-success' : 'bg-warning text-dark'}`}>{p.polg_tested}</span></td>
                  <td><span className="text-muted" style={{ fontSize: 10 }}>{p.current_meds}</span></td>
                  <td><span className={`badge ${p.seizure_control === 'drug-resistant' ? 'bg-danger' : p.seizure_control === 'partially-controlled' ? 'bg-warning text-dark' : 'bg-success'}`}>{p.seizure_control}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📈 ACTH Response Summary" borderColor={ACCENT3}>
        <div className="row text-center">
          {[
            ['Complete Response', data.summary?.acth_complete_pct, 'bg-success'],
            ['Partial Response', data.summary?.acth_partial_pct, 'bg-warning text-dark'],
            ['No Response', data.summary?.acth_none_pct, 'bg-danger'],
          ].map(([label, val, cls]) => (
            <div key={label} className="col-4">
              <div className={`badge ${cls} w-100 py-2 fs-6`}>{val ?? '—'}%</div>
              <div className="small text-muted mt-1">{label}</div>
            </div>
          ))}
        </div>
        {data.summary?.vpa_without_polg > 0 && (
          <Alert text={`⚠️ ${data.summary.vpa_without_polg} patient(s) received VPA WITHOUT documented POLG testing — immediate action required.`} variant="danger" />
        )}
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [selSeizure, setSelSeizure] = useState(null);
  const [selTrigger, setSelTrigger] = useState(null);

  return (
    <>
      <SectionCard title="⚡ Seizure Type Prevalence" borderColor={ACCENT}>
        {(data.seizure_types || []).map((s, i) => (
          <div key={i} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{s.type}</span>
              <span className="text-muted">{s.pct}%</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: ACCENT }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 Seizure Type Detail (click to expand)" borderColor={ACCENT4}>
        {(data.seizure_detail || []).map((s, i) => (
          <div key={i} className="mb-3 border rounded p-2">
            <div
              className="d-flex justify-content-between align-items-center"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelSeizure(selSeizure === i ? null : i)}
            >
              <span className="fw-semibold small">{s.type} — {s.prevalence_pct}%</span>
              <span className="text-muted small">{selSeizure === i ? '▲' : '▼'}</span>
            </div>
            {selSeizure === i && (
              <div className="mt-2 small">
                <div className="mb-2"><strong>Semiology:</strong> {s.semiology}</div>
                <div className="mb-2"><strong>EEG Pattern:</strong> {s.eeg_pattern}</div>
                <div className="p-2 rounded" style={{ backgroundColor: '#fff8e1' }}>
                  <strong>⚡ Clinical Tip:</strong> {s.clinical_tip}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🎯 Trigger Prevalence" borderColor={ACCENT2}>
        {(data.triggers || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={
            t.pct >= 80 ? ACCENT2 : t.pct >= 60 ? '#e65100' : t.pct >= 40 ? '#f57c00' : '#558b2f'
          } />
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Trigger Detail — Management Protocols (click to expand)" borderColor="#6a1b9a">
        {(data.trigger_detail || []).map((t, i) => (
          <div key={i} className="mb-2 border rounded p-2">
            <div
              className="d-flex justify-content-between"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelTrigger(selTrigger === i ? null : i)}
            >
              <span className="fw-semibold small">{t.trigger} — {t.prevalence_pct}%</span>
              <span className="text-muted small">{selTrigger === i ? '▲' : '▼'}</span>
            </div>
            {selTrigger === i && (
              <div className="mt-2 small">
                <div className="mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
                <div className="p-2 rounded" style={{ backgroundColor: '#f1f8e9' }}>
                  <strong>Management:</strong> {t.management}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ────────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [selTx, setSelTx] = useState(null);
  const [selCI, setSelCI] = useState(null);

  return (
    <>
      <Alert
        text="⚠️ GABRB3-DEE SAFETY: LTG AVOID (myoclonic component) · CBZ/OXC/PHT AVOID (generalised seizure aggravation) · BDZ: use higher rescue dose (0.3-0.4 mg/kg) · POLG test before VPA · VGB ERG q6M"
        variant="danger"
      />

      <SectionCard title="💊 Treatment Ladder (click for full detail)" borderColor={ACCENT3}>
        {(data.treatment_detail || []).map((t, i) => (
          <div key={i} className="mb-3 border rounded p-3"
            style={{ borderLeft: `4px solid ${t.evidence.startsWith('Level A') ? ACCENT3 : t.evidence.startsWith('Level B') ? '#1565c0' : '#6d4c41'}` }}
          >
            <div
              className="d-flex justify-content-between align-items-start"
              style={{ cursor: 'pointer' }}
              onClick={() => setSelTx(selTx === i ? null : i)}
            >
              <div>
                <span className="fw-bold">{t.name}</span>
                <span className="badge ms-2" style={{ backgroundColor: t.evidence.startsWith('Level A') ? ACCENT3 : t.evidence.startsWith('Level B') ? '#1565c0' : '#6d4c41' }}>
                  {t.evidence.split('—')[0].trim()}
                </span>
              </div>
              <span className="text-muted small">{selTx === i ? '▲' : '▼'}</span>
            </div>
            <div className="small text-muted mt-1">{t.status}</div>
            {selTx === i && (
              <div className="mt-3 small">
                <div className="mb-2 p-2 rounded" style={{ backgroundColor: '#e8f5e9' }}>
                  <strong>Dose:</strong> {t.dose}
                </div>
                <div className="mb-2"><strong>Mechanism:</strong> {t.moa}</div>
                <div className="mb-2 p-2 rounded" style={{ backgroundColor: '#e3f2fd' }}>
                  <strong>Efficacy:</strong> {t.efficacy}
                </div>
                <div className="mb-2 p-2 rounded" style={{ backgroundColor: '#fff3e0' }}>
                  <strong>Safety:</strong> {t.safety}
                </div>
                <div className="p-2 rounded" style={{ backgroundColor: '#f3e5f5' }}>
                  <strong>Monitoring:</strong> {t.monitoring}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications — GABRB3-DEE28" borderColor={ACCENT2}>
        {(data.contraindications || []).map((c, i) => (
          <div key={i} className="mb-2 border border-danger rounded p-2"
            style={{ cursor: 'pointer' }}
            onClick={() => setSelCI(selCI === i ? null : i)}
          >
            <div className="d-flex justify-content-between">
              <span className="fw-bold text-danger small">{c.drug}</span>
              <span className={`badge ${c.severity.startsWith('ABSOLUTE') ? 'bg-danger' : c.severity.startsWith('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                {c.severity.split('—')[0].trim()}
              </span>
            </div>
            {selCI === i && (
              <div className="mt-2 small">
                <div className="mb-1"><strong>Mechanism:</strong> {c.mechanism}</div>
                <div className="p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                  <strong>Action:</strong> {c.action}
                </div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔍 Monitoring Protocol" borderColor="#0277bd">
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
            <thead className="table-primary">
              <tr><th>Item</th><th>Frequency</th><th>Threshold / Action</th></tr>
            </thead>
            <tbody>
              {(data.monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td><strong>{m.item}</strong><br /><span className="text-muted">{m.rationale}</span></td>
                  <td><span className="badge bg-info text-dark">{m.frequency}</span></td>
                  <td className="text-danger small">{m.threshold}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🗂️ Lifecycle Management Windows" borderColor={ACCENT}>
        {(data.lifecycle || []).map((lc, i) => (
          <div key={i} className="mb-3 border-start border-3 ps-3" style={{ borderColor: ACCENT }}>
            <div className="fw-bold small">{lc.window}</div>
            <div className="text-muted small">{lc.focus}</div>
            <div className="small mt-1"><strong>Interventions:</strong> {lc.interventions}</div>
            <div className="small"><strong>Goals:</strong> {lc.goals}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [selDef, setSelDef] = useState(null);

  return (
    <>
      <SectionCard title={`📖 Key Concepts — ${data.total_definitions} Definitions`} borderColor={ACCENT}>
        {(data.definitions || []).map((d, i) => (
          <div key={i} className="mb-2 border rounded p-2"
            style={{ cursor: 'pointer', borderLeft: `3px solid ${ACCENT}` }}
            onClick={() => setSelDef(selDef === i ? null : i)}
          >
            <div className="fw-semibold small" style={{ color: ACCENT }}>{d.term}</div>
            {selDef === i && (
              <div className="mt-1 small text-muted">{d.definition}</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="⚡ Thresholds — Action Triggers" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
            <thead className="table-danger">
              <tr><th>Threshold / Rule</th><th>Value</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td>{t.threshold}</td>
                  <td><span className="badge bg-danger">{t.value}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📚 Evidence Standards" borderColor="#546e7a">
        <ul className="mb-0 small">
          {(data.standards || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="📄 Key References" borderColor={ACCENT3}>
        <ol className="mb-0 small">
          {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
        </ol>
      </SectionCard>
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function GABRB3Page() {
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('Overview');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/gabrb3/overview`).then(r => r.json()),
      fetch(`${API}/api/gabrb3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gabrb3/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!overview) return <div className="text-muted p-3">Loading GABRB3 / DEE28 dashboard…</div>;

  return (
    <div className="container-fluid py-3">
      <h3 style={{ color: ACCENT }}>🧬 GABRB3 Epilepsy — DEE28 / West Syndrome → Lennox-Gastaut</h3>
      <p className="text-muted small">
        {overview.gene} · {overview.inheritance} · {overview.total} patients · Dashboard #{overview.dashboard_number}
      </p>

      {/* Tab buttons */}
      <div className="mb-3">
        {TABS.map(t => (
          <TabBtn key={t} label={t} active={tab === t} onClick={() => setTab(t)} />
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Etiology' && <EtiologyTab data={breakdown} />}
      {tab === 'Seizure Types & Triggers' && <SeizureTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={defs} />}

      <div className="text-muted small mt-4 border-top pt-2">
        GABRB3 (15q12) · GABA-A β3 Subunit · DEE28 West Syndrome → LGS ·
        BDZ reduced efficacy · ACTH + VGB → KD + CLB + RUF → FFA · N={overview.total}
      </div>
    </div>
  );
}
