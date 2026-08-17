'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const ACCENT = '#1a5f7a';
const ACCENT_LIGHT = '#e0f4fb';

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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3" style={{ borderLeft: `4px solid ${ACCENT}` }}>
        <strong>ATP1A3 (19q13.2) — Na+/K+-ATPase α3 · Neuron-Specific Electrogenic Pump:</strong>{' '}
        ATP1A3 LOF → 50% reduction in neuronal Na+/K+ pump → Na+ accumulation → Vm failure →{' '}
        <strong>episodic hemiplegic attacks (AHC) + epilepsy</strong>.{' '}
        <em>AHC PATHOGNOMONIC: all attacks resolve with sleep — no other epilepsy gene has this feature.</em>{' '}
        <strong>FLUNARIZINE = first-line AHC attack prevention (NOT a Na-channel blocker).</strong>{' '}
        <span className="text-danger fw-bold">ABSOLUTE CI: CBZ/OXC (bilateral hemiplegic crisis) · Tiagabine (NCSE). HIGH RISK: D2-antagonists (worsens dystonia — never metoclopramide). POLG before VPA. VPPP females.</span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={ACCENT} />
        <KPI label="Have Epilepsy" value={`${kpis.epilepsy_pct}%`} color="#0d6efd" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="On Flunarizine" value={`${kpis.on_flunarizine_pct}%`} color={ACCENT} />
        <KPI label="On LEV" value={`${kpis.on_lev_pct}%`} color="#198754" />
        <KPI label="On VPA" value={`${kpis.on_vpa_pct}%`} color="#6f42c1" />
        <KPI label="On KD" value={`${kpis.on_kd_pct}%`} color="#ffc107" />
        <KPI label="CAPOS (E818K)" value={kpis.capos_n} color="#e83e8c" />
        <KPI label="Optic Atrophy" value={kpis.optic_atrophy_n} color="#fd7e14" />
        <KPI label="SNHL (CAPOS)" value={kpis.snhl_n} color="#20c997" />
        <KPI label="Status Epilepticus" value={kpis.status_epilepticus_n} color="#dc3545" />
        <KPI label="SUDEP High Risk" value={kpis.sudep_high_risk_n} color="#dc3545" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: ACCENT }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.etiology}</span>
                    <span className="badge" style={{ background: ACCENT }}>{e.n} ({e.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.pct / maxEtio * 100)}%`, background: ACCENT }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: ACCENT }}>
              Treatments (Evidence Level)
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="d-flex justify-content-between align-items-center mb-2 border-bottom pb-1">
                  <span className="small">{t.drug}</span>
                  <span className={`badge ${t.level === 'A' ? 'bg-success' : t.level === 'B' ? 'bg-primary' : 'bg-secondary'}`}>
                    Level {t.level}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
              ⚠️ Key Thresholds
            </div>
            <div className="card-body p-2">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Metric</th><th>Threshold</th></tr></thead>
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td>{t.metric}</td>
                      <td><span className="badge bg-warning text-dark">{t.threshold}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
              🔬 Lifecycle Windows
            </div>
            <div className="card-body p-2">
              {lifecycle.map((l, i) => (
                <div key={i} className="mb-2 border-bottom pb-1">
                  <div className="fw-semibold small" style={{ color: ACCENT }}>{l.window}</div>
                  <div className="text-muted" style={{ fontSize: '0.78rem' }}>{l.headline}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold py-2" style={{ background: ACCENT_LIGHT }}>
          🩺 Monitoring (top 8)
        </div>
        <div className="card-body p-2">
          <div className="row">
            {monitoring.map((m, i) => (
              <div key={i} className="col-md-6 small mb-1">
                <span className="fw-semibold">{m.item}</span>{' '}
                <span className="text-muted">— {m.frequency}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const etiologies = data.etiology_distribution || [];
  const patients = data.patient_sample || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Etiology Classes (5) + Patient Sample (n=15 of 40)</h6>
      {etiologies.map((e, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header py-2 fw-semibold text-white" style={{ background: ACCENT }}>
            {e.etiology} — {e.n} patients ({e.pct}%)
          </div>
          <div className="card-body small">
            <div className="row g-2">
              <div className="col-md-4"><strong>Mechanism:</strong> {e.mechanism}</div>
              <div className="col-md-3"><strong>Key Variants:</strong> {e.typical_variants}</div>
              <div className="col-md-3"><strong>EEG:</strong> {e.eeg_signature}</div>
              <div className="col-md-2"><strong>Phenotype:</strong> {e.phenotype}</div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-2" style={{ color: ACCENT }}>Patient Sample (first 15)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-striped small">
          <thead className="table-dark">
            <tr>
              <th>ID</th><th>Category</th><th>Sex</th><th>Age</th><th>Onset(M)</th>
              <th>Epilepsy</th><th>DRE</th><th>Flunarizine</th><th>LEV</th><th>VPA</th>
              <th>KD</th><th>POLG</th><th>SE</th><th>Fever↑</th><th>SUDEP↑</th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p, i) => (
              <tr key={i}>
                <td>{p.id}</td>
                <td><span className="badge bg-secondary small">{p.category.replace('ATP1A3-','').substring(0,12)}</span></td>
                <td>{p.sex}</td>
                <td>{p.age}y</td>
                <td>{p.onset_age_months}M</td>
                <td>{p.has_epilepsy ? '✅' : '—'}</td>
                <td>{p.drug_resistant ? <span className="text-danger fw-bold">DRE</span> : '—'}</td>
                <td>{p.on_flunarizine ? '✅' : '—'}</td>
                <td>{p.on_lev ? '✅' : '—'}</td>
                <td>{p.on_vpa ? '✅' : '—'}</td>
                <td>{p.on_kd ? '✅' : '—'}</td>
                <td>{p.polg_tested}</td>
                <td>{p.has_status_epilepticus ? <span className="text-danger">SE</span> : '—'}</td>
                <td>{p.fever_triggered ? '🌡️' : '—'}</td>
                <td>{p.sudep_high_risk ? <span className="text-danger">⚠️</span> : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_detail || [];
  const triggers = data.trigger_detail || [];
  const maxSz = Math.max(...seizures.map(s => s.prevalence_pct || 0), 1);

  return (
    <div>
      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Seizure Types (5)</h6>
      {seizures.map((s, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{ background: ACCENT_LIGHT }}>
            <span className="fw-semibold" style={{ color: ACCENT }}>{s.type}</span>
            <span className="badge" style={{ background: ACCENT }}>{s.prevalence_pct}%</span>
          </div>
          <div className="card-body small">
            <div className="mb-1">
              <div className="progress mb-2" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.prevalence_pct / maxSz * 100}%`, background: ACCENT }} />
              </div>
            </div>
            <div className="row g-2">
              <div className="col-md-4"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="col-md-4"><strong>EEG:</strong> {s.eeg_pattern}</div>
              <div className="col-md-4 text-info-emphasis"><strong>💡 Clinical Tip:</strong> {s.clinical_tip}</div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-3" style={{ color: ACCENT }}>Attack / Seizure Triggers (8)</h6>
      <div className="row g-3">
        {triggers.map((t, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 d-flex justify-content-between" style={{ background: ACCENT_LIGHT }}>
                <span className="fw-semibold small" style={{ color: ACCENT }}>{t.trigger}</span>
                <span className="badge bg-warning text-dark">{t.prevalence_pct}%</span>
              </div>
              <div className="card-body small">
                <div className="mb-1"><strong>Mechanism:</strong> {t.mechanism}</div>
                <div className="text-success"><strong>Management:</strong> {t.management}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatment_detail || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Treatments (7)</h6>
      {treatments.map((t, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header py-2 d-flex justify-content-between align-items-center text-white" style={{ background: ACCENT }}>
            <span className="fw-semibold small">{t.drug}</span>
            <span className={`badge ${t.level === 'A' ? 'bg-success' : t.level === 'B' ? 'bg-primary' : 'bg-secondary'}`}>
              Level {t.level}
            </span>
          </div>
          <div className="card-body small">
            <div className="row g-2">
              <div className="col-md-3"><strong>MOA:</strong> {t.moa}</div>
              <div className="col-md-2"><strong>Dose:</strong> {t.dose}</div>
              <div className="col-md-2"><strong>Efficacy:</strong> {t.efficacy}</div>
              <div className="col-md-2"><strong>Safety:</strong> {t.safety}</div>
              <div className="col-md-3 text-primary"><strong>ATP1A3 Note:</strong> {t.atp1a3_note}</div>
            </div>
          </div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-3 text-danger">🚫 Contraindications (5)</h6>
      {contraindications.map((c, i) => (
        <div key={i} className={`alert py-2 mb-2 ${c.risk === 'ABSOLUTE CI' ? 'alert-danger' : 'alert-warning'}`}>
          <strong>{c.drug}</strong>{' '}
          <span className={`badge ${c.risk === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'} ms-1`}>{c.risk}</span>
          <div className="small mt-1 text-muted">{c.reason}</div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const references = data.references || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Key Concepts (15)</h6>
      <div className="row g-2 mb-4">
        {concepts.map((c, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header py-1 fw-semibold small" style={{ background: ACCENT_LIGHT, color: ACCENT }}>
                {c.term}
              </div>
              <div className="card-body py-2 small">{c.definition}</div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Clinical Thresholds (12)</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered small">
          <thead className="table-dark">
            <tr><th>Metric</th><th>Threshold</th><th>Note</th></tr>
          </thead>
          <tbody>
            {thresholds.map((t, i) => (
              <tr key={i}>
                <td>{t.metric}</td>
                <td><span className="badge bg-warning text-dark">{t.threshold}</span></td>
                <td className="text-muted">{t.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Standards &amp; Guidelines (12)</h6>
      <div className="row g-2 mb-4">
        {standards.map((s, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm">
              <div className="card-body py-2 small">
                <span className="badge me-1" style={{ background: ACCENT }}>{s.code}</span>
                <strong>{s.title}</strong>
                <div className="text-muted">{s.relevance}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-semibold mb-3" style={{ color: ACCENT }}>Key References (6)</h6>
      {references.map((r, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-body py-2 small">
            <span className="badge me-1 bg-secondary">{r.id}</span>
            <strong>{r.citation}</strong>
            <div className="text-muted mt-1">Key finding: {r.key_finding}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default function ATP1A3Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/atp1a3/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setErr('Failed to load overview'));
  }, []);

  useEffect(() => {
    if (tab >= 1 && tab <= 3 && !breakdown) {
      fetch(`${API}/api/atp1a3/breakdown`)
        .then(r => r.json()).then(setBreakdown)
        .catch(() => setErr('Failed to load breakdown'));
    }
    if (tab === 4 && !definitions) {
      fetch(`${API}/api/atp1a3/definitions`)
        .then(r => r.json()).then(setDefinitions)
        .catch(() => setErr('Failed to load definitions'));
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-1">
        <span style={{ fontSize: '1.5rem' }}>🧬</span>
        <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
          ATP1A3 Epilepsy
        </h4>
        <span className="badge bg-secondary ms-1">19q13.2</span>
        <span className="badge bg-info text-dark ms-1">Na+/K+-ATPase α3</span>
        <span className="badge bg-primary ms-1">AHC / CAPOS / RDP / DEE</span>
        <span className="badge bg-danger ms-1">AD de novo</span>
      </div>
      <p className="text-muted small mb-3">
        Na+/K+-ATPase alpha-3 subunit · Neuron-specific electrogenic pump · 40-patient cohort ·
        D801N (60% AHC) / E815K (20% severe DEE) / E818K (CAPOS) · Flunarizine first-line ·
        CBZ/OXC ABSOLUTE CI · Sleep resolves attacks (pathognomonic)
      </p>
      {err && <div className="alert alert-danger py-2 small">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: ACCENT, color: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
