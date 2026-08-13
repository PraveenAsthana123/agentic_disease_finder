'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT = '#6c3483'; // deep purple — Ohtahara/STXBP1 theme
const ACCENT2 = '#1a5276'; // navy accent for SNARE/Munc18-1

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#f5eef8', color: borderColor }}>
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const k = data.kpis || {};
  const colors = { ok: ACCENT, warn: '#e67e22', danger: '#e74c3c', info: ACCENT2 };

  return (
    <>
      <div className="alert alert-primary mb-4" style={{ borderLeft: `5px solid ${ACCENT}`, fontSize: 14 }}>
        <strong>🧬 {data.syndrome}</strong> — Gene: <strong>{data.gene}</strong> ({data.inheritance})<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      <div className="alert alert-danger mb-4" style={{ fontSize: 13 }}>
        <strong>⚡ Clinical AHA — STXBP1-DEE:</strong> {data.key_aha}
      </div>

      <div className="row g-2 mb-4">
        <KPI label="N Patients" value={k.seizure_free_pct !== undefined ? data.n_patients : '—'} color={ACCENT} />
        <KPI label="Seizure-Free" value={`${k.seizure_free_pct ?? '—'}%`} color={colors.ok} />
        <KPI label="Drug-Resistant" value={`${k.drug_resistant_pct ?? '—'}%`} color={colors.danger} />
        <KPI label="KD On" value={`${k.kd_on_pct ?? '—'}%`} color={ACCENT2} />
        <KPI label="VGB (SHARE)" value={`${k.vgb_on_pct ?? '—'}%`} color={colors.warn} />
        <KPI label="POLG Tested" value={`${k.polg_tested_pct ?? '—'}%`} color={colors.info} />
        <KPI label="Avg Onset" value={`${k.avg_onset_age_hours ?? '—'}h`} color={ACCENT} />
        <KPI label="Theta EEG" value={`${k.theta_hypersynchrony_pct ?? '—'}%`} color={ACCENT2} />
      </div>

      <SectionCard title="⚠️ Clinical Alerts">
        {(data.clinical_alerts || []).map((a, i) => (
          <Alert key={i} text={a}
            variant={a.includes('🚫') || a.includes('MANDATORY') ? 'danger' : a.includes('⚡') || a.includes('EEG') ? 'primary' : 'warning'} />
        ))}
      </SectionCard>

      <SectionCard title="🧬 Etiology Distribution (N=41)" borderColor={ACCENT2}>
        {(data.etiologies || []).map((e, i) => (
          <PctBar key={i} label={e.etiology} pct={e.pct} color={i === 0 ? ACCENT : i === 1 ? ACCENT2 : '#27ae60'} />
        ))}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="⚡ Seizure Type Prevalence">
            {Object.entries(data.seizure_type_prevalence || {}).map(([k2, v], i) => (
              <PctBar key={i} label={k2} pct={v} color={ACCENT} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⚡ Seizure Trigger Rates" borderColor={ACCENT2}>
            {Object.entries(data.trigger_seizure_rates || {}).map(([k2, v], i) => (
              <PctBar key={i} label={k2} pct={v} color={ACCENT2} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🕐 Lifecycle Windows" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Window</th><th>Age</th><th>Focus</th><th>Key Action</th></tr>
            </thead>
            <tbody>
              {(data.lifecycle_windows || []).map((w, i) => (
                <tr key={i}>
                  <td className="fw-bold">{w.window}</td>
                  <td className="text-nowrap">{w.age_range}</td>
                  <td>{w.focus}</td>
                  <td><strong>{w.key_action}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [search, setSearch] = useState('');
  const [catFilter, setCatFilter] = useState('All');
  if (!data) return <div className="text-muted">Loading…</div>;

  const cats = ['All', ...new Set((data.patients || []).map(p => p.category))];
  const filtered = (data.patients || []).filter(p => {
    const matchCat = catFilter === 'All' || p.category === catFilter;
    const matchSearch = !search || JSON.stringify(p).toLowerCase().includes(search.toLowerCase());
    return matchCat && matchSearch;
  });

  return (
    <>
      <div className="row mb-3">
        <div className="col-md-6">
          <input className="form-control form-control-sm" placeholder="Search patients…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <div className="col-md-6">
          <select className="form-select form-select-sm" value={catFilter} onChange={e => setCatFilter(e.target.value)}>
            {cats.map(c => <option key={c}>{c}</option>)}
          </select>
        </div>
      </div>

      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped table-bordered small">
          <thead style={{ backgroundColor: '#f5eef8', position: 'sticky', top: 0 }}>
            <tr>
              <th>ID</th><th>Age(M)</th><th>Sex</th><th>Onset(h)</th>
              <th>Category</th><th>Phase</th><th>Treatment</th>
              <th>Control</th><th>KD</th><th>BHB</th><th>VGB</th>
              <th>SHARE</th><th>PB(µg/mL)</th><th>POLG</th><th>Theta</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}
                style={{ backgroundColor: p.seizure_control === 'drug-resistant' ? '#fdf2f8' : p.seizure_control === 'seizure-free' ? '#eafaf1' : 'inherit' }}>
                <td className="fw-bold text-nowrap">{p.id}</td>
                <td>{p.age_months}</td>
                <td>{p.sex}</td>
                <td>{p.onset_age_hours}</td>
                <td style={{ fontSize: 11 }}>{p.category}</td>
                <td style={{ fontSize: 11 }}>{p.disease_phase}</td>
                <td style={{ fontSize: 11 }}>{p.current_treatment}</td>
                <td>
                  <span className={`badge ${p.seizure_control === 'drug-resistant' ? 'bg-danger' : p.seizure_control === 'seizure-free' ? 'bg-success' : 'bg-warning text-dark'}`}>
                    {p.seizure_control}
                  </span>
                </td>
                <td>{p.kd_on}</td>
                <td>{p.bhb_mmoll ?? '—'}</td>
                <td>{p.vgb_on}</td>
                <td>{p.share_rems_enrolled}</td>
                <td>{p.pb_level_ugml ?? '—'}</td>
                <td>
                  <span className={`badge ${p.polg_tested === 'Y' ? 'bg-success' : 'bg-warning text-dark'}`}>
                    {p.polg_tested}
                  </span>
                </td>
                <td>
                  <span className={`badge ${p.eeg_theta_hypersynchrony === 'Y' ? 'bg-primary' : 'bg-secondary'}`}>
                    {p.eeg_theta_hypersynchrony}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SectionCard title="🧬 Etiology Catalog — Detailed">
        {(data.etiology_catalog || []).map((e, i) => (
          <div key={i} className="mb-4 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{e.etiology} ({e.pct}%, N={e.n})</div>
            <div className="small mb-2"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="small mb-2"><strong>EEG Correlate:</strong> {e.eeg_correlate}</div>
            <div className="small mb-2"><strong>MRI Finding:</strong> {e.mri_finding}</div>
            <div className="small alert alert-light py-1 mb-0"><strong>Clinical Note:</strong> {e.clinical_note}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <>
      <SectionCard title="⚡ Seizure Types — EEG Correlates & Clinical Tips">
        {(data.seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 p-3 border rounded" style={{ borderLeft: `4px solid ${ACCENT}` }}>
            <div className="d-flex align-items-center mb-2">
              <span className="fw-bold me-2" style={{ color: ACCENT }}>{s.type}</span>
              <span className="badge bg-primary ms-auto">{s.prevalence_pct}%</span>
            </div>
            <div className="small mb-2"><strong>EEG Correlate:</strong> {s.eeg_correlate}</div>
            <div className="small alert alert-info py-1 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🌡️ Seizure Triggers" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Trigger</th><th>Prevalence</th><th>Management Note</th></tr>
            </thead>
            <tbody>
              {(data.triggers || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.trigger}</td>
                  <td><div className="d-flex align-items-center gap-2">
                    <div className="progress flex-grow-1" style={{ height: 8 }}>
                      <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: ACCENT2 }} />
                    </div>
                    <span className="text-nowrap">{t.prevalence_pct}%</span>
                  </div></td>
                  <td>{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <>
      <SectionCard title="🚫 Absolute Contraindications">
        {(data.treatments || []).filter(t => t.evidence && t.evidence.includes('CI')).map((t, i) => (
          <div key={i} className="alert alert-danger small mb-2">
            <strong>{t.drug}</strong> — {t.safety}
          </div>
        ))}
        {(data.absolute_contraindications || data.treatments?.filter(t => t.severity) || []).map((c, i) => (
          <div key={`ci-${i}`} className="alert alert-danger small mb-2">
            <strong>{c.drug}</strong> ({c.severity})<br />{c.reason}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="💊 Treatment Catalog" borderColor={ACCENT2}>
        {(data.treatments || []).filter(t => !t.evidence?.includes('ABSOLUTE CI')).map((t, i) => (
          <div key={i} className="mb-4 p-3 border rounded" style={{ borderLeft: `4px solid ${i < 4 ? ACCENT : '#27ae60'}` }}>
            <div className="d-flex align-items-start mb-2">
              <span className="fw-bold me-2" style={{ color: ACCENT }}>{t.drug}</span>
              <span className="badge ms-auto" style={{ backgroundColor: ACCENT, fontSize: 11 }}>{t.evidence}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1 text-danger"><strong>Safety:</strong> {t.safety}</div>
            <div className="small alert alert-light py-1 mb-0"><strong>Monitoring:</strong> {t.monitoring}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔬 AED Monitoring Protocol">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Item</th><th>Target</th><th>Frequency</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(data.aed_monitoring || []).map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold">{m.item}</td>
                  <td>{m.target}</td>
                  <td>{m.frequency}</td>
                  <td>{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📋 Clinical Standards">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Standard</th><th>Title</th><th>Relevance</th></tr>
            </thead>
            <tbody>
              {(data.standards || []).map((s, i) => (
                <tr key={i}>
                  <td className="fw-bold text-nowrap">{s.std}</td>
                  <td>{s.title}</td>
                  <td>{s.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  return (
    <>
      <SectionCard title="📚 Key Concepts">
        {(data.concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded" style={{ borderLeft: `3px solid ${ACCENT}` }}>
            <div className="fw-bold mb-1" style={{ color: ACCENT }}>{c.term}</div>
            <div className="small">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Absolute Contraindications" borderColor="#e74c3c">
        {(data.absolute_contraindications || []).map((c, i) => (
          <div key={i} className="alert alert-danger mb-2">
            <div className="fw-bold">{c.drug}</div>
            <div className="small"><strong>Severity:</strong> {c.severity}</div>
            <div className="small">{c.reason}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Threshold</th><th>Action Required</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📖 Key References" borderColor={ACCENT2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: '#f5eef8' }}>
              <tr><th>Reference</th><th>Title</th><th>Clinical Note</th></tr>
            </thead>
            <tbody>
              {(data.references || []).map((r, i) => (
                <tr key={i}>
                  <td className="fw-bold text-nowrap">{r.ref}</td>
                  <td><em>{r.title}</em></td>
                  <td>{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function STXBP1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/stxbp1/overview`).then(r => r.json()),
      fetch(`${API}/api/stxbp1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/stxbp1/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-3">Error: {error}</div>;

  const tabData = [overview, breakdown, breakdown, breakdown, definitions];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 style={{ color: ACCENT }}>
          🧠 STXBP1 Encephalopathy (STXBP1-DEE / EIEE4)
        </h4>
        <p className="text-muted small mb-2">
          STXBP1 (9q34.11) · Munc18-1 Synaptic Vesicle Fusion Regulator · 41-patient DEE cohort ·
          Asynchronous BS → IS/West → Theta Hypersynchrony · ACTH+VGB (UKISS) · KD Level B ·
          POLG exclusion mandatory · VGB SHARE REMS (USA)
        </p>
        <div className="d-flex flex-wrap">
          {TABS.map((t, i) => <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
        </div>
      </div>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizureTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
