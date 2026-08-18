'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#4a148c'; // deep purple — TBC1D24 RAB35 synaptic / DOORS syndrome
const DANGER = '#b71c1c'; // dark red — CI / VGB deaf-blind warning
const SUCCESS = '#1565c0'; // dark blue — cochlear implant / seizure freedom
const WARN = '#e65100';   // deep orange — 2-OGA elevation / monitoring warning

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

function Bar({ label, value, max, color = COLOR }) {
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
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#f3e5f5' }}>
        <strong>🧬 TBC1D24 (16p13.3) — DOORS Syndrome / FHEIG / DEE16 — Rab-GTPase-Activating Protein:</strong>{' '}
        TBC1D24 encodes a dual-domain protein: <strong>TBC domain</strong> (RAB35 Rab-GAP → synaptic vesicle recycling)
        + <strong>TLDc domain</strong> (NAD+-binding → oxidative stress resistance).{' '}
        Biallelic AR LOF → <strong>DOORS</strong> (Deafness + Onychodystrophy + Osteodystrophy + cognitive Retardation + Seizures)
        or <strong>FHEIG</strong> (myoclonic epilepsy) or <strong>DEE16</strong> (infantile encephalopathy).{' '}
        Pathognomonic biomarker: <strong>elevated urinary 2-oxoglutarate (2-OGA)</strong>.{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          ⚠️ VGB ABSOLUTE AVOID maintenance in DOORS: VFD (blindness) + profound SNHL = near-sensory isolation.{' '}
          ⚠️ POLG1 MANDATORY before VPA: 2-OGA mimics mitochondrial disease (Alpers fatal hepatotoxicity).{' '}
          ⚠️ TGB ABSOLUTE CI (NCSE). PHT HIGH RISK (neuropathy). CBZ/OXC HIGH RISK in FHEIG myoclonic.{' '}
          ✅ LEV preferred (SV2A + POLG-safe). KD best DRE option (seizure + oxidative stress). CI before 24M.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={data.total_patients} />
        <KPI label="Seizure-Free" value={`${data.seizure_free_pct}%`} color={SUCCESS} />
        <KPI label="Drug-Resistant" value={`${data.dre_pct}%`} color={DANGER} />
        <KPI label="SNHL / Deafness" value={`${data.snhl_pct}%`} color={WARN} />
        <KPI label="Syndromes" value="DOORS/FHEIG/DEE16" color={COLOR} />
        <KPI label="Locus" value="16p13.3" color={COLOR} />
      </div>

      {/* Key thresholds alert */}
      <div className="alert alert-warning py-2 small mb-3">
        <strong>⚡ Key Thresholds:</strong>{' '}
        VPA trough 50–100 µg/mL · KD BHB 2–4 mmol/L · Fever prophylaxis ≥38.0°C (FHEIG low threshold) ·
        Cochlear implant <strong>before 24 months</strong> · VPPP from age 12y · VGB ERG q3M (SHARE/REMS)
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
              <Bar label={s.type} value={s.prevalence_pct} max={100} color={COLOR} />
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
                <div>
                  <div className="fw-semibold">{t.drug}</div>
                  <div className="text-muted">{t.role}</div>
                </div>
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
                <span className="badge bg-danger text-wrap text-start" style={{ minWidth: 80 }}>{c.level}</span>
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
            <div className="small text-muted">{l.focus}</div>
          </div>
        ))}
      </SectionCard>

      {/* Clinical Pearls */}
      {data.clinical_pearls && (
        <SectionCard title="💡 Clinical Pearls">
          <ul className="mb-0 small">
            {data.clinical_pearls.map((p, i) => <li key={i}>{p}</li>)}
          </ul>
        </SectionCard>
      )}

      {/* Thresholds */}
      <SectionCard title="📏 Key Thresholds">
        <div className="row g-2">
          {thresholds.map(t => (
            <div key={t.name} className="col-12 col-md-6">
              <div className="border rounded p-2 small">
                <span className="fw-semibold">{t.name}: </span>
                <span className="text-primary fw-bold">{t.value}</span>
                {t.note && <div className="text-muted">{t.note}</div>}
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
  const catalog = data.etiology_catalog || [];
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
                <span className="badge bg-secondary">{e.n || Math.round(40 * e.pct / 100)} pts</span>
              </div>
            </div>
            <div className="small text-muted mb-1">{e.mechanism}</div>
            {e.eeg_correlate && <div className="small"><strong>EEG:</strong> {e.eeg_correlate}</div>}
            {e.typical_age_onset && <div className="small"><strong>Onset:</strong> {e.typical_age_onset}</div>}
            {e.drug_resistance && <div className="small"><strong>DRE:</strong> {e.drug_resistance}</div>}
            {e.snhl && <div className="small"><strong>SNHL:</strong> {e.snhl}</div>}
            {e['2oga'] && <div className="small"><strong>2-OGA:</strong> {e['2oga']}</div>}
            {e.nail_bones && <div className="small"><strong>Nail/Bone:</strong> {e.nail_bones}</div>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="👥 Patient Cohort (40 patients)">
        <div className="mb-2">
          <select className="form-select form-select-sm" style={{ maxWidth: 320 }} value={filter} onChange={e => setFilter(e.target.value)}>
            <option value="all">All etiologies</option>
            {catalog.map(e => <option key={e.category} value={e.category}>{e.category.replace(/-/g, ' ')}</option>)}
          </select>
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Age Onset</th><th>Age Now</th><th>Sex</th>
                <th>Etiology</th><th>DRE</th><th>SF</th><th>SNHL</th><th>2-OGA</th>
              </tr>
            </thead>
            <tbody>
              {filtered.slice(0, 40).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.age_onset_months}m</td>
                  <td>{p.current_age_years}y</td>
                  <td>{p.sex}</td>
                  <td className="small">{p.etiology.replace(/-/g, ' ')}</td>
                  <td><span className={`badge ${p.drug_resistant === 'Yes' ? 'bg-danger' : 'bg-success'}`}>{p.drug_resistant}</span></td>
                  <td><span className={`badge ${p.seizure_free === 'Yes' ? 'bg-success' : 'bg-secondary'}`}>{p.seizure_free}</span></td>
                  <td className="small">{p.snhl}</td>
                  <td className="small">{p.two_oga}</td>
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
  const seizures = data.seizure_catalog || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <SectionCard title="⚡ Seizure Types">
        {seizures.map((s, i) => (
          <div key={i} className="mb-3 border rounded p-3">
            <div className="d-flex justify-content-between align-items-center flex-wrap gap-2 mb-1">
              <strong style={{ color: COLOR }}>{s.type}</strong>
              <span className="badge" style={{ background: COLOR }}>{s.pct}% prevalence</span>
            </div>
            <div className="small mb-1"><strong>EEG:</strong> {s.eeg}</div>
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
  const treatments = data.treatment_catalog || [];
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
            <div className="small mb-1"><strong>Role:</strong> {t.role}</div>
            <div className="small mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.tbc1d24_note && (
              <div className="alert alert-secondary py-1 px-2 mb-0 small mt-1">
                🧬 <strong>TBC1D24-specific:</strong> {t.tbc1d24_note}
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
              <span className="badge bg-danger">{c.level}</span>
            </div>
            <div className="small">{c.reason}</div>
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
                <div className="fw-semibold">{t.name}</div>
                <div className="text-primary fw-bold">{t.value}</div>
                {t.note && <div className="text-muted">{t.note}</div>}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="📋 Standards">
        <ul className="mb-0 small">
          {standards.map((s, i) => (
            <li key={i}><strong>{s.name}</strong>: {s.scope}</li>
          ))}
        </ul>
      </SectionCard>

      <SectionCard title="📚 References">
        <ol className="mb-0 small">
          {references.map((r, i) => (
            <li key={i}><strong>{r.author} ({r.year})</strong> <em>{r.journal}</em>. {r.title}</li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

export default function TBC1D24Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/tbc1d24/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) {
        fetch(`${API}/api/tbc1d24/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
      }
    }
    if (tab === 4) {
      if (!definitions) {
        fetch(`${API}/api/tbc1d24/definitions`)
          .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
      }
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 TBC1D24 Epilepsy
          </h4>
          <div className="small text-muted">
            DOORS Syndrome / FHEIG / DEE16 · Rab-GAP (RAB35) + TLDc (Oxidative Stress) · AR-LOF · 16p13.3
          </div>
        </div>
        <span className="badge ms-auto" style={{ background: COLOR }}>16p13.3 · OMIM 220500/605021/615338</span>
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
