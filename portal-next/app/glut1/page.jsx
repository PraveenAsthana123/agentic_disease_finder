'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT = '#0d6efd'; // blue — metabolic/genetic theme

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eff6ff', color: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

export default function GLUT1Dashboard() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [patientSearch, setPatientSearch] = useState('');
  const [patientSort, setPatientSort] = useState({ key: 'id', dir: 1 });

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/glut1/overview`).then(r => r.json()),
      fetch(`${API}/api/glut1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/glut1/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center text-muted">Loading GLUT1-DS Dashboard…</div>;
  if (!overview || !breakdown) return <div className="container py-5 text-center text-danger">Failed to load data.</div>;

  const etDist = overview.etiology_distribution || {};
  const kpis = overview.kpis || {};
  const alerts = overview.clinical_alerts || [];
  const triggerRates = overview.trigger_seizure_rates || {};
  const seizurePrevalence = overview.seizure_type_prevalence || {};

  // Tab 1 — Overview
  const OverviewTab = () => (
    <div>
      <div className="alert alert-info mb-3" style={{ fontSize: 13 }}>
        <strong>🔬 GLUT1 Deficiency Syndrome (De Vivo Disease):</strong> SLC2A1 haploinsufficiency → reduced GLUT1 transporter density at BBB → chronic cerebral energy insufficiency.
        Key biomarker: <strong>CSF:serum glucose ratio &lt;0.40</strong> (after 4h fast).
        First-line: <strong>Ketogenic Diet — Level A (disease-modifying)</strong>. Absolute CI: <strong>Valproate</strong>.
      </div>

      {alerts.map((a, i) => <Alert key={i} text={a} variant={a.includes('ABSOLUTE') || a.includes('DANGEROUS') ? 'danger' : 'warning'} />)}

      <div className="row g-2 mb-4">
        <KPI label="Patients" value={overview.n_patients} color={ACCENT} />
        <KPI label="Seizure-Free %" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="Avg Onset Age" value={`${kpis.avg_onset_age_months}m`} color="#6f42c1" />
        <KPI label="Avg CSF Ratio" value={kpis.avg_csf_serum_ratio} color="#dc3545" />
        <KPI label="Avg BHB (KD)" value={`${kpis.avg_bhb_mmol_l} mmol/L`} color="#0dcaf0" />
        <KPI label="PED Present" value={`${kpis.ped_present_pct}%`} color="#fd7e14" />
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution (N=41)">
            {Object.entries(etDist).map(([k, v]) => (
              <PctBar key={k} label={k.replace(/-/g, ' ')} pct={v.pct} />
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
              <div><strong>Gene:</strong> {overview.gene}</div>
              <div><strong>Mechanism:</strong> {overview.mechanism}</div>
              <div><strong>Biomarker:</strong> {overview.key_biomarker}</div>
              <div><strong>First-Line Rx:</strong> {overview.first_line_treatment}</div>
              <div className="text-danger fw-bold"><strong>ABSOLUTE CI:</strong> {overview.absolute_contraindication}</div>
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Top Triggers — Seizure Rate by Trigger">
        {Object.entries(triggerRates).slice(0, 6).map(([k, v]) => (
          <PctBar key={k} label={k} pct={v} color="#dc3545" />
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
      p.id.toLowerCase().includes(patientSearch.toLowerCase()) ||
      p.etiology.toLowerCase().includes(patientSearch.toLowerCase()) ||
      p.seizure_control.toLowerCase().includes(patientSearch.toLowerCase()) ||
      p.current_treatment.toLowerCase().includes(patientSearch.toLowerCase())
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
          <SectionCard key={i} title={`Etiology: ${e.etiology} (N=${e.n}, ${e.pct}%)`} borderColor="#0d6efd">
            <div className="mb-2"><strong>Mechanism:</strong> <span className="small text-muted">{e.mechanism.slice(0, 300)}…</span></div>
            <div className="mb-2"><strong>EEG Correlate:</strong> <span className="small text-muted">{e.eeg_correlate.slice(0, 250)}…</span></div>
            <div className="mb-2"><strong>MRI:</strong> <span className="small text-muted">{e.mri_finding.slice(0, 200)}…</span></div>
            <div><strong>Clinical Note:</strong> <span className="small text-muted">{e.clinical_note.slice(0, 250)}…</span></div>
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
                {['id','age','sex','onset_age_months','etiology','seizure_types','disease_phase','current_treatment','seizure_control','csf_serum_ratio','bhb_mmol_l','ped_present'].map(k => (
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
                  <td><span style={{ fontSize: 11 }}>{p.etiology.replace(/-/g,' ')}</span></td>
                  <td><span style={{ fontSize: 11 }}>{p.seizure_types}</span></td>
                  <td>{p.disease_phase}</td>
                  <td>{p.current_treatment}</td>
                  <td>
                    <span className={`badge ${p.seizure_control === 'seizure-free' ? 'bg-success' : p.seizure_control === 'partial-response' ? 'bg-warning text-dark' : 'bg-danger'}`}>
                      {p.seizure_control}
                    </span>
                  </td>
                  <td className={p.csf_serum_ratio < 0.35 ? 'text-danger fw-bold' : 'text-warning'}>{p.csf_serum_ratio}</td>
                  <td className={p.bhb_mmol_l >= 2 && p.bhb_mmol_l <= 4 ? 'text-success' : 'text-warning'}>{p.bhb_mmol_l}</td>
                  <td>{p.ped_present === 'Y' ? <span className="badge bg-warning text-dark">PED</span> : '—'}</td>
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
        <SectionCard title="Seizure Types in GLUT1-DS" borderColor="#6f42c1">
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
              <div className="small text-muted mb-1">{w.description.slice(0, 350)}…</div>
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
    const monitoring = breakdown.monitoring_protocol || [];

    return (
      <div>
        <div className="alert alert-danger mb-3" style={{ fontSize: 13 }}>
          <strong>⚠️ DISEASE-MODIFYING THERAPY:</strong> Ketogenic Diet is FIRST-LINE and DISEASE-MODIFYING in GLUT1-DS — it corrects the cerebral energy deficit, not just seizures.
          Delays in KD initiation permanently reduce cognitive outcomes.
          <strong> VPA is ABSOLUTE CONTRAINDICATION (inhibits GLUT1 transporter).</strong>
        </div>

        {treatments.map((t, i) => {
          const isCI = t.evidence_level.includes('ABSOLUTE');
          return (
            <div key={i} className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${isCI ? '#dc3545' : '#0d6efd'}` }}>
              <div className="card-header d-flex justify-content-between align-items-center" style={{ backgroundColor: isCI ? '#fff5f5' : '#eff6ff' }}>
                <strong style={{ color: isCI ? '#dc3545' : '#0d6efd' }}>
                  {isCI ? '🚫 ' : '💊 '}{t.name}
                </strong>
                <span className={`badge ${isCI ? 'bg-danger' : 'bg-primary'}`}>{t.evidence_level}</span>
              </div>
              <div className="card-body">
                <div className="row">
                  <div className="col-md-6">
                    <div className="mb-2"><strong>Dose:</strong> <span className="small">{t.dose}</span></div>
                    <div className="mb-2"><strong>Mechanism:</strong> <span className="small text-muted">{t.moa.slice(0, 250)}…</span></div>
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

        <SectionCard title="Monitoring Protocol" borderColor="#198754">
          {monitoring.map((m, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold">{m.item}</div>
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
        <SectionCard title="Absolute Contraindications" borderColor="#dc3545">
          {contraindications.map((c, i) => (
            <div key={i} className="mb-3 pb-2 border-bottom">
              <div className="fw-bold text-danger">🚫 {c.drug}</div>
              <div className="small"><strong>Scope:</strong> {c.scope}</div>
              <div className="small text-muted"><strong>Mechanism:</strong> {c.mechanism.slice(0, 250)}…</div>
              <div className="small"><strong>Consequence:</strong> {c.consequence}</div>
              <div className="small"><strong>Action:</strong> {c.action}</div>
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
              <div><em>{r.journal}</em> {r.vol}: {r.pages} · PMID {r.pmid}</div>
              <div className="text-success">{r.note}</div>
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
        <span style={{ fontSize: 32, marginRight: 12 }}>🔬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>GLUT1 Deficiency Syndrome (De Vivo Disease)</h4>
          <div className="text-muted small">SLC2A1 haploinsufficiency · BBB glucose transport defect · KD first-line disease-modifying therapy · N=41</div>
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
