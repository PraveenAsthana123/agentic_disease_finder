'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT = '#6f42c1'; // purple — genetic/imprinting theme

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f5f0ff', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function AngelmanDashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [patientSearch, setPatientSearch] = useState('');
  const [patientSort, setPatientSort] = useState({ key: 'id', dir: 1 });

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/angelman/overview`).then(r => r.json()),
      fetch(`${API}/api/angelman/breakdown`).then(r => r.json()),
      fetch(`${API}/api/angelman/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading Angelman Syndrome Dashboard…</div>;
  if (!overview || !breakdown) return <div className="container py-5 text-center text-danger">Failed to load data.</div>;

  const kpis = overview.kpis || {};
  const alerts = overview.clinical_alerts || [];
  const etiologies = overview.etiologies || [];
  const seizurePrevalence = overview.seizure_type_prevalence || {};
  const triggerRates = overview.trigger_seizure_rates || {};

  // Tab 1 — Overview
  const OverviewTab = () => (
    <div>
      <div className="alert alert-info mb-3" style={{ fontSize: 13 }}>
        <strong>🧬 Angelman Syndrome:</strong> UBE3A haploinsufficiency (15q11.2-q13.3, maternal imprinting) →
        severe epileptic encephalopathy with characteristic high-amplitude delta EEG.
        Key EEG: <strong>Triphasic high-amplitude delta + alpha-frequency bursts (5-10 Hz)</strong>.
        First-line: <strong>Clonazepam + LEV</strong>. Caution: <strong>PHT, CBZ, VGB</strong>.
      </div>

      {alerts.map((a, i) => <Alert key={i} text={a} variant={a.includes('ABSOLUTE') || a.includes('CONTRAINDICATED') || a.includes('AVOID') ? 'danger' : 'warning'} />)}

      <div className="row g-2 mb-4">
        <KPI label="Patients" value={overview.n_patients} color={ACCENT} />
        <KPI label="Seizure-Free %" value={`${kpis.seizure_free_pct ?? '—'}%`} color="#198754" />
        <KPI label="Avg Onset Age" value={`${kpis.avg_onset_age_months ?? '—'}m`} color="#6f42c1" />
        <KPI label="KD Responders" value={`${kpis.kd_responder_pct ?? '—'}%`} color="#0dcaf0" />
        <KPI label="PHT Avoided" value={`${kpis.pht_avoided_pct ?? '—'}%`} color="#dc3545" />
        <KPI label="PPR Present" value={`${kpis.ppr_present_pct ?? '—'}%`} color="#fd7e14" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution (N=41)">
            {etiologies.map((e, i) => (
              <PctBar key={i} label={`${e.category || e.etiology}`} pct={e.pct} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Type Prevalence">
            {Object.entries(seizurePrevalence).map(([k, v]) => (
              <PctBar key={k} label={k} pct={v} color="#6f42c1" />
            ))}
          </SectionCard>
          <SectionCard title="Key Clinical Information" borderColor="#198754">
            <div className="small">
              <div><strong>Gene:</strong> {overview.key_gene}</div>
              <div><strong>EEG Hallmark:</strong> {overview.eeg_hallmark}</div>
              <div><strong>Biomarker:</strong> {overview.key_biomarker}</div>
              <div className="text-danger fw-bold mt-1"><strong>Key AHA:</strong> {overview.key_aha}</div>
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Top Triggers — Seizure Rate by Trigger" borderColor="#dc3545">
        {Object.entries(triggerRates).slice(0, 6).map(([k, v]) => (
          <PctBar key={k} label={k} pct={v} color="#dc3545" />
        ))}
      </SectionCard>

      <SectionCard title="Lifecycle Windows" borderColor="#0dcaf0">
        {(overview.lifecycle_windows || []).map((w, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom">
            <span className="fw-bold">{i + 1}. {w.window}</span>
            <span className="badge bg-secondary ms-2">{w.phase}</span>
            <div className="small text-muted mt-1">{w.description ? w.description.slice(0, 200) + '…' : ''}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );

  // Tab 2 — Patients & Etiology
  const PatientsTab = () => {
    const pts = breakdown.patients || [];
    const etCat = breakdown.etiology_catalog || [];

    const filtered = pts.filter(p =>
      !patientSearch ||
      (p.id || '').toLowerCase().includes(patientSearch.toLowerCase()) ||
      (p.etiology || '').toLowerCase().includes(patientSearch.toLowerCase()) ||
      (p.seizure_control || '').toLowerCase().includes(patientSearch.toLowerCase()) ||
      (p.current_treatment || '').toLowerCase().includes(patientSearch.toLowerCase())
    );
    const sorted = [...filtered].sort((a, b) => {
      const av = a[patientSort.key] ?? '';
      const bv = b[patientSort.key] ?? '';
      return patientSort.dir * (av < bv ? -1 : av > bv ? 1 : 0);
    });
    const toggleSort = key => setPatientSort(s => ({ key, dir: s.key === key ? -s.dir : 1 }));

    return (
      <div>
        {etCat.map((e, i) => (
          <SectionCard key={i} title={`Etiology: ${e.etiology} (N=${e.n}, ${e.pct}%)`} borderColor={ACCENT}>
            <div className="mb-2"><strong>Mechanism:</strong> <span className="small text-muted">{(e.mechanism || '').slice(0, 300)}…</span></div>
            <div className="mb-2"><strong>EEG Correlate:</strong> <span className="small text-muted">{(e.eeg_correlate || '').slice(0, 250)}…</span></div>
            <div className="mb-2"><strong>MRI:</strong> <span className="small text-muted">{(e.mri_finding || '').slice(0, 200)}…</span></div>
            <div><strong>Clinical Note:</strong> <span className="small text-muted">{(e.clinical_note || '').slice(0, 250)}…</span></div>
          </SectionCard>
        ))}

        <div className="mb-2">
          <input
            className="form-control form-control-sm"
            placeholder="Search patients (ID, etiology, control, treatment)…"
            value={patientSearch}
            onChange={e => setPatientSearch(e.target.value)}
          />
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
            <thead className="table-primary">
              <tr>
                {['id','age','sex','onset_age_months','etiology','seizure_types','disease_phase','current_treatment','seizure_control','ppr_present','kd_on'].map(k => (
                  <th key={k} style={{ cursor: 'pointer' }} onClick={() => toggleSort(k)}>
                    {k.replace(/_/g,' ')} {patientSort.key === k ? (patientSort.dir > 0 ? '▲' : '▼') : ''}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, 50).map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.age}y</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_age_months}m</td>
                  <td><span style={{ fontSize: 11 }}>{(p.etiology || '').replace(/-/g,' ')}</span></td>
                  <td><span style={{ fontSize: 11 }}>{p.seizure_types}</span></td>
                  <td>{p.disease_phase}</td>
                  <td>{p.current_treatment}</td>
                  <td>
                    <span className={`badge ${p.seizure_control === 'seizure-free' ? 'bg-success' : p.seizure_control === 'partial-response' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                      {p.seizure_control}
                    </span>
                  </td>
                  <td>{p.ppr_present === 'Y' ? <span className="badge bg-warning text-dark">PPR</span> : '—'}</td>
                  <td>{p.kd_on === 'Y' ? <span className="badge bg-info text-dark">KD</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="text-muted small">Showing {Math.min(sorted.length, 50)} of {sorted.length} patients</div>
      </div>
    );
  };

  // Tab 3 — Seizure Types & Triggers
  const SeizureTab = () => {
    const seizures = breakdown.seizure_types || [];
    const triggers = breakdown.triggers || [];
    const lifecycle = breakdown.lifecycle_windows || [];

    return (
      <div>
        <SectionCard title="Seizure Types in Angelman Syndrome" borderColor="#6f42c1">
          {seizures.map((s, i) => (
            <div key={i} className="mb-3 pb-3 border-bottom">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <strong>{s.type}</strong>
                <span className="badge" style={{ backgroundColor: '#6f42c1' }}>{s.prevalence_pct}%</span>
              </div>
              <div className="progress mb-2" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.prevalence_pct}%`, backgroundColor: '#6f42c1' }} />
              </div>
              <div className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg_correlate}</div>
              <div className="small"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
            </div>
          ))}
        </SectionCard>

        <SectionCard title="Trigger Analysis — Seizure Rate by Trigger" borderColor="#dc3545">
          {triggers.map((t, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <strong className="small">{t.trigger}</strong>
                <span className="badge bg-danger">{t.seizure_rate_pct}%</span>
              </div>
              <div className="progress mb-1" style={{ height: 6 }}>
                <div className="progress-bar bg-danger" style={{ width: `${t.seizure_rate_pct}%` }} />
              </div>
              <div className="small text-muted">{t.note}</div>
            </div>
          ))}
        </SectionCard>

        <SectionCard title="6-Window Patient Lifecycle" borderColor="#0dcaf0">
          {lifecycle.map((w, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold">{i + 1}. {w.window}</div>
              <div className="badge bg-secondary mb-1">{w.phase}</div>
              <div className="small text-muted mb-1">{(w.description || '').slice(0, 350)}…</div>
              <div className="small"><strong>Key Actions:</strong> {w.key_actions}</div>
            </div>
          ))}
        </SectionCard>
      </div>
    );
  };

  // Tab 4 — Treatments
  const TreatmentsTab = () => {
    const treatments = breakdown.treatments || [];
    const aedMonitoring = breakdown.aed_monitoring || [];

    return (
      <div>
        <div className="alert alert-warning mb-3" style={{ fontSize: 13 }}>
          <strong>⚠️ ANGELMAN SYNDROME AED CAUTIONS:</strong> PHT is CONTRAINDICATED (worsens myoclonus).
          CBZ/OXC are RELATIVE CONTRAINDICATION (worsens myoclonus/absence in ~25%).
          VGB worsens myoclonus + visual field loss. First-line: <strong>CLN + LEV</strong>.
          KD effective for refractory seizures.
        </div>

        {treatments.map((t, i) => {
          const isCI = (t.evidence_level || '').includes('AVOID') || (t.evidence_level || '').includes('CI') || (t.name || '').includes('PHT') || (t.name || '').includes('VGB');
          const isExperimental = (t.evidence_level || '').includes('experimental') || (t.evidence_level || '').includes('Phase');
          return (
            <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${isCI ? '#dc3545' : isExperimental ? '#0dcaf0' : ACCENT}` }}>
              <div className="card-header d-flex justify-content-between align-items-center"
                style={{ backgroundColor: isCI ? '#fff5f5' : isExperimental ? '#f0faff' : '#f5f0ff' }}>
                <strong style={{ color: isCI ? '#dc3545' : isExperimental ? '#0dcaf0' : ACCENT }}>
                  {isCI ? '🚫 ' : isExperimental ? '🔬 ' : '💊 '}{t.name}
                </strong>
                <span className={`badge ${isCI ? 'bg-danger' : isExperimental ? 'bg-info text-dark' : 'bg-primary'}`}>{t.evidence_level}</span>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <div className="mb-2"><strong>Dose:</strong> <span className="small">{t.dose}</span></div>
                    <div className="mb-2"><strong>Mechanism:</strong> <span className="small text-muted">{(t.moa || '').slice(0, 250)}</span></div>
                  </div>
                  <div className="col-md-6">
                    <div className="mb-2"><strong>Efficacy:</strong> <span className="small">{t.efficacy}</span></div>
                    <div className="mb-2"><strong>Safety:</strong> <span className="small text-muted">{t.safety}</span></div>
                    <div className="mb-2"><strong>Monitoring:</strong> <span className="small">{t.monitoring}</span></div>
                  </div>
                </div>
                {t.clinical_alert && (
                  <div className={`alert ${isCI ? 'alert-danger' : 'alert-warning'} py-2 mb-0 mt-2`} style={{ fontSize: 12 }}>
                    {t.clinical_alert}
                  </div>
                )}
              </div>
            </div>
          );
        })}

        <SectionCard title="AED Monitoring Protocol" borderColor="#198754">
          {aedMonitoring.map((m, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold">{m.item || m.aed}</div>
              <div className="small"><strong>Frequency:</strong> {m.frequency}</div>
              <div className="small"><strong>Target:</strong> <span className="text-success">{m.target}</span></div>
              <div className="small text-muted">{m.rationale}</div>
            </div>
          ))}
        </SectionCard>
      </div>
    );
  };

  // Tab 5 — Definitions
  const DefinitionsTab = () => {
    const concepts = definitions?.concepts || [];
    const contraindications = definitions?.absolute_contraindications || [];
    const thresholds = definitions?.thresholds || [];
    const refs = definitions?.references || [];

    return (
      <div>
        <SectionCard title="Absolute & Relative Contraindications" borderColor="#dc3545">
          {contraindications.map((c, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold text-danger">🚫 {c.drug}</div>
              {c.scope && <div className="small"><strong>Scope:</strong> {c.scope}</div>}
              <div className="small text-muted"><strong>Mechanism:</strong> {(c.mechanism || '').slice(0, 300)}…</div>
              {c.consequence && <div className="small"><strong>Consequence:</strong> {c.consequence}</div>}
              {c.action && <div className="small"><strong>Action:</strong> {c.action}</div>}
            </div>
          ))}
        </SectionCard>

        <SectionCard title="Clinical Thresholds" borderColor="#fd7e14">
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
              <thead className="table-warning">
                <tr><th>Threshold</th><th>Category</th><th>Action</th></tr>
              </thead>
              <tbody>
                {thresholds.map((t, i) => (
                  <tr key={i}>
                    <td><strong>{t.name}</strong></td>
                    <td><span className="badge bg-secondary">{t.category}</span></td>
                    <td className="small">{t.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>

        <SectionCard title="Key Concepts & Definitions">
          {concepts.map((c, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold" style={{ color: ACCENT }}>{c.term}</div>
              <div className="small text-muted">{c.definition}</div>
            </div>
          ))}
        </SectionCard>

        <SectionCard title="Evidence Standards" borderColor="#6f42c1">
          {(breakdown.standards || []).map((s, i) => (
            <div key={i} className="mb-2 pb-2 border-bottom">
              <div className="fw-bold small">{s.name}</div>
              <div className="small text-muted"><span className="badge bg-secondary me-1">{s.domain}</span>{s.relevance}</div>
            </div>
          ))}
        </SectionCard>

        <SectionCard title="References" borderColor="#6c757d">
          {refs.map((r, i) => (
            <div key={i} className="mb-2 pb-2 border-bottom small">
              <div><strong>{r.authors}</strong> ({r.year})</div>
              <div className="text-muted fst-italic">{r.title}</div>
              <div><em>{r.journal}</em> {r.vol ? `${r.vol}: ${r.pages}` : ''} {r.pmid ? `· PMID ${r.pmid}` : ''}</div>
              {r.note && <div className="text-success">{r.note}</div>}
            </div>
          ))}
        </SectionCard>
      </div>
    );
  };

  const TabContent = [OverviewTab, PatientsTab, SeizureTab, TreatmentsTab, DefinitionsTab][tab];

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3">
        <span style={{ fontSize: 32, marginRight: 12 }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>Angelman Syndrome (AS)</h4>
          <div className="text-muted small">UBE3A haploinsufficiency · 15q11.2-q13.3 maternal imprinting · Triphasic high-amplitude delta EEG · N=41</div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      <TabContent />
    </div>
  );
}
