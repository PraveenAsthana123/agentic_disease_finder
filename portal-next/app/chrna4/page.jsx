'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#6d4c41'; // warm brown — nicotinic/cholinergic/tobacco receptor family
const DANGER = '#b71c1c'; // dark red — HLA-B*15:02 / SJS-TEN / CI warning
const SUCCESS = '#2e7d32'; // dark green — seizure freedom
const WARN = '#e65100';   // deep orange — nicotine paradox / monitoring

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, max = 100, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card shadow-sm mb-3">
      <div className="card-header fw-semibold text-white py-2" style={{ background: borderColor }}>
        {title}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const etiologies = data.etiology_distribution || {};
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const ciSummary = data.contraindications_summary || [];
  const seizureSummary = data.seizure_summary || [];
  const etioEntries = Object.entries(etiologies).map(([k, v]) => ({ category: k, count: v }));
  const maxEtio = Math.max(...etioEntries.map(e => e.count), 1);

  return (
    <div>
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#efebe9' }}>
        <strong>🧬 CHRNA4 (20q13.33) — ADNFLE — First Epilepsy Channelopathy (Steinlein 1995):</strong>{' '}
        CHRNA4 encodes the <strong>nAChR α4 subunit</strong> of the neuronal nicotinic acetylcholine receptor (α4)₂(β2)₃.{' '}
        GOF mutations (predominantly M2 domain: S284L, 776insL) → <strong>delayed desensitisation</strong> → excessive cholinergic activation during NREM sleep → <strong>hypermotor frontal lobe seizures from NREM stage 2–3</strong>.{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          ⚠️ HLA-B*15:02 MANDATORY before CBZ/OXC in SE Asian ancestry (SJS/TEN risk — fatal).{' '}
          ⚠️ TGB ABSOLUTE CI (NCSE). ⚠️ High-dose acute nicotine HIGH RISK (GOF receptor activation).{' '}
          ✅ CBZ-XR first-line (70–80% seizure freedom). ✅ Video-polysomnography gold standard.{' '}
          🚬 Nicotine paradox: low-dose patch investigational (desensitisation); high-dose triggers seizures.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} />
        <KPI label="Seizure-Free" value={`${data.seizure_free_pct}%`} color={SUCCESS} />
        <KPI label="Drug-Resistant" value={`${data.dre_pct}%`} color={DANGER} />
        <KPI label="Hypermotor Seizures" value={`${data.hypermotor_count}`} color={COLOR} />
        <KPI label="Misdiag. Parasomnia" value={`${data.misdiagnosed_parasomnia_count}`} color={WARN} />
        <KPI label="Locus" value="20q13.33" color={COLOR} />
      </div>

      {/* Key thresholds alert */}
      <div className="alert alert-warning py-2 small mb-3">
        <strong>⚡ Key Thresholds:</strong>{' '}
        CBZ trough 8–12 mg/L (nocturnal coverage) · HLA-B*15:02 test BEFORE CBZ in SE Asian ·
        Na⁺ q3M (CBZ SIADH) · Neutrophils q3M (CBZ FBC) ·
        Nicotine patch 7–14 mg (investigational, NOT high-dose NRT) · VPSG at diagnosis
      </div>

      {/* Etiology distribution */}
      <SectionCard title="📊 Etiology Distribution (40 patients)">
        {etioEntries.map(e => (
          <div key={e.category} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span className="text-truncate" style={{ maxWidth: '75%' }}>{e.category.replace(/-/g, ' ')}</span>
              <span className="text-muted">{e.count} pts</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${Math.round(e.count / maxEtio * 100)}%`, backgroundColor: COLOR }} />
            </div>
          </div>
        ))}
      </SectionCard>

      {/* Seizure summary */}
      <SectionCard title="⚡ Seizure Type Prevalence">
        <div className="row">
          {seizureSummary.map(s => (
            <div key={s.type} className="col-12 col-md-6 mb-1">
              <Bar label={s.type} value={s.frequency_pct} max={100} color={COLOR} />
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Treatments summary */}
      <SectionCard title="💊 Treatment Summary">
        <div className="row g-2">
          {treatments.map(t => (
            <div key={t.drug} className="col-12 col-md-6">
              <div className="d-flex align-items-start gap-2 border rounded p-2 small h-100">
                <span className="badge rounded-pill text-white" style={{ background: COLOR, whiteSpace: 'nowrap' }}>{t.level}</span>
                <div className="fw-semibold">{t.drug}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Contraindications summary */}
      <SectionCard title="🚫 Contraindications" borderColor={DANGER}>
        <div className="row g-2">
          {ciSummary.map(c => (
            <div key={c.drug} className="col-12 col-md-6">
              <div className="d-flex align-items-center gap-2 border border-danger rounded p-2 small">
                <span className="badge bg-danger text-wrap text-start" style={{ minWidth: 80 }}>{c.risk}</span>
                <span className="fw-semibold">{c.drug}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="🩺 Monitoring Protocol">
        <div className="row g-2">
          {monitoring.map(m => (
            <div key={m.item} className="col-12 col-md-6">
              <div className="border rounded p-2 small h-100">
                <div className="fw-semibold">{m.item}</div>
                <div className="text-muted">{m.frequency}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Lifecycle */}
      <SectionCard title="🗓 Lifecycle Stages">
        {lifecycle.map((l, i) => (
          <div key={i} className="mb-2 border rounded p-2">
            <div className="fw-semibold small" style={{ color: COLOR }}>{l.stage}</div>
            <div className="small text-muted">{l.key_action}</div>
          </div>
        ))}
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="📏 Key Thresholds">
        <div className="row g-2">
          {thresholds.map((t, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="border rounded p-2 small">
                <span className="fw-semibold">{t.threshold}: </span>
                <span className="text-primary">{t.action}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const catalog = data.etiology_catalog || data.etiologies || [];
  const patients = data.patients || [];
  const [filter, setFilter] = useState('all');
  const filtered = filter === 'all' ? patients : patients.filter(p => p.etiology.includes(filter));

  return (
    <div>
      <SectionCard title="🧬 Etiology Catalog">
        {catalog.map((e, i) => (
          <div key={i} className="mb-3 border rounded p-3">
            <div className="d-flex justify-content-between align-items-start flex-wrap gap-2 mb-1">
              <strong style={{ color: COLOR }}>{e.category.replace(/-/g, ' ')}</strong>
              <div className="d-flex gap-2">
                <span className="badge" style={{ background: COLOR }}>{e.pct}%</span>
                <span className="badge bg-secondary">{Math.round(40 * e.pct / 100)} pts</span>
              </div>
            </div>
            <div className="small text-muted mb-1">{e.mechanism}</div>
            {e.eeg && <div className="small"><strong>EEG/VPSG:</strong> {e.eeg}</div>}
            {e.onset_months && <div className="small"><strong>Onset:</strong> {e.onset_months}</div>}
            {e.severity && <div className="small"><strong>Severity:</strong> {e.severity}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Cohort (40 patients)">
        <div className="mb-2">
          <select className="form-select form-select-sm" style={{ maxWidth: 340 }} value={filter} onChange={e => setFilter(e.target.value)}>
            <option value="all">All etiologies</option>
            {catalog.map(e => <option key={e.category} value={e.category}>{e.category.replace(/-/g, ' ')}</option>)}
          </select>
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Onset (yr)</th><th>Age (yr)</th><th>Sex</th>
                <th>Etiology</th><th>DRE</th><th>SF</th><th>VPSG</th><th>Family Hx</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 40).map(p => (
                <tr key={p.patient_id}>
                  <td className="fw-semibold">{p.patient_id}</td>
                  <td>{p.age_onset_years}y</td>
                  <td>{p.current_age_years}y</td>
                  <td>{p.sex}</td>
                  <td className="small text-truncate" style={{ maxWidth: 160 }}>{p.etiology.replace(/-/g, ' ')}</td>
                  <td><span className={`badge ${p.dre ? 'bg-danger' : 'bg-success'}`}>{p.dre ? 'Yes' : 'No'}</span></td>
                  <td><span className={`badge ${p.seizure_free ? 'bg-success' : 'bg-secondary'}`}>{p.seizure_free ? 'Yes' : 'No'}</span></td>
                  <td><span className={`badge ${p.vpsg_confirmed ? 'bg-primary' : 'bg-secondary'}`}>{p.vpsg_confirmed ? 'Yes' : 'No'}</span></td>
                  <td><span className={`badge ${p.family_history_adnfle ? 'bg-info' : 'bg-secondary'}`}>{p.family_history_adnfle ? 'Yes' : 'No'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizureTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_catalog || data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <SectionCard title="⚡ Seizure Types (ADNFLE Spectrum)">
        {seizures.map((s, i) => (
          <div key={i} className="mb-3 border rounded p-3">
            <div className="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-1">
              <strong style={{ color: COLOR }}>{s.type}</strong>
              <span className="badge" style={{ background: COLOR }}>{s.frequency_pct}% prevalence</span>
            </div>
            <div className="small mb-1"><strong>EEG/VPSG:</strong> {s.eeg_correlate}</div>
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            {s.clinical_tip && (
              <div className="alert alert-info py-1 px-2 mb-0 small">
                💡 <strong>Clinical tip:</strong> {s.clinical_tip}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔥 Seizure Triggers">
        <div className="row g-2">
          {triggers.map(t => (
            <div key={t.trigger} className="col-12 col-md-6">
              <div className="border rounded p-2 h-100">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <strong className="small">{t.trigger}</strong>
                  <span className="badge" style={{ background: WARN }}>{t.pct}%</span>
                </div>
                <div className="small text-muted">{t.note}</div>
                <div className="progress mt-1" style={{ height: 6 }}>
                  <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: WARN }} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatment_catalog || data.treatments || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <SectionCard title="💊 Treatment Catalog">
        {treatments.map((t, i) => (
          <div key={i} className="mb-3 border rounded p-3">
            <div className="d-flex justify-content-between align-items-start flex-wrap gap-2 mb-2">
              <strong style={{ color: COLOR }}>{t.drug}</strong>
              <span className="badge" style={{ background: COLOR }}>{t.level}</span>
            </div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.chrna4_note && (
              <div className="alert alert-secondary py-1 px-2 mb-0 small mt-1">
                🧬 <strong>CHRNA4-specific:</strong> {t.chrna4_note}
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Contraindications" borderColor={DANGER}>
        {contraindications.map((c, i) => (
          <div key={i} className="mb-2 border border-danger rounded p-2">
            <div className="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-1">
              <strong className="text-danger">{c.drug}</strong>
              <span className="badge bg-danger">{c.risk}</span>
            </div>
            <div className="small">{c.mechanism}</div>
            {c.alternative && <div className="small text-success mt-1"><strong>Alternative:</strong> {c.alternative}</div>}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const defs = data.definitions || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const references = data.references || [];
  const [open, setOpen] = useState(null);

  return (
    <div>
      <SectionCard title="📖 Key Concepts (15 Definitions)">
        {defs.map((d, i) => (
          <div key={i} className="mb-1 border rounded overflow-hidden">
            <button
              className="btn btn-link text-start w-100 text-decoration-none py-2 px-3"
              style={{ color: COLOR }}
              onClick={() => setOpen(open === i ? null : i)}
            >
              <strong>{d.term}</strong>
            </button>
            {open === i && (
              <div className="px-3 pb-3 small text-muted border-top">{d.definition}</div>
            )}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Thresholds">
        <div className="row g-2">
          {thresholds.map((t, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="border rounded p-2 small">
                <div className="fw-semibold">{t.threshold}</div>
                <div className="text-primary">{t.action}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="📋 Standards">
        <ul className="mb-0 small">
          {standards.map((s, i) => (
            <li key={i}><strong>{s.name}</strong>: {s.applies}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References">
        <ol className="mb-0 small">
          {references.map((r, i) => (
            <li key={i}><strong>{r.author} ({r.year})</strong> <em>{r.journal}</em>. {r.title} PMID: {r.pmid}</li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

export default function CHRNA4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/chrna4/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/chrna4/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
      }
    }
    if (tab === 4) {
      if (!definitions) {
        fetch(`${API}/api/chrna4/definitions`)
          .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
      }
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 CHRNA4 Epilepsy
          </h4>
          <div className="small text-muted">
            ADNFLE (Autosomal Dominant Nocturnal Frontal Lobe Epilepsy) · nAChR α4 GOF · First Epilepsy Channelopathy (1995) · 20q13.33
          </div>
        </div>
        <span className="badge ms-auto" style={{ background: COLOR }}>20q13.33 · OMIM *118504 · #600513</span>
      </div>

      {error && <div className="alert alert-danger small">Error: {error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <SeizureTab data={breakdown} />}
      {tab === 3 && <TreatmentTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
