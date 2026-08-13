'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT = '#1a5276'; // deep blue — neonatal/channelopathy theme
const ACCENT2 = '#0e6655'; // teal accent for M-current/Kv7

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
      <div className="card-header fw-bold" style={{ backgroundColor: '#eaf4fb', color: borderColor }}>
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
        <strong>⚡ {data.syndrome}</strong> — Gene: <strong>{data.gene}</strong> ({data.inheritance})<br />
        <strong>EEG hallmark:</strong> {data.eeg_hallmark}<br />
        <strong>Key biomarker:</strong> {data.key_biomarker}
      </div>

      <div className="alert alert-danger mb-4" style={{ fontSize: 13 }}>
        <strong>⚠️ KEY ALERTS:</strong> {data.key_aha}
      </div>

      <div className="row mb-3">
        <KPI label="N (Cohort)" value={data.n_patients} color={ACCENT} />
        <KPI label="Seizure-free %" value={`${k.seizure_free_pct}%`} color={colors.info} />
        <KPI label="Drug-resistant %" value={`${k.drug_resistant_pct}%`} color={colors.danger} />
        <KPI label="CBZ/OXC on %" value={`${k.cbz_oxc_on_pct}%`} color={ACCENT} />
        <KPI label="KD on %" value={`${k.kd_on_pct}%`} color={colors.warn} />
        <KPI label="Avg onset (hrs)" value={`${k.avg_onset_age_hours}h`} color={colors.danger} />
        <KPI label="HLA-B*15:02 tested" value={`${k.hla_b1502_tested_pct}%`} color={colors.warn} />
        <KPI label="GOF/dom-neg %" value={`${k.gof_dominant_negative_pct}%`} color={colors.danger} />
        <KPI label="Hyponatraemia %" value={`${k.hyponatraemia_pct}%`} color={colors.warn} />
      </div>

      <SectionCard title="Clinical Alerts" borderColor={colors.danger}>
        {(data.clinical_alerts || []).map((a, i) => <Alert key={i} text={a} variant={a.includes('FIRST') || a.includes('MANDATORY') ? 'success' : a.includes('WITHDRAWN') || a.includes('POLG') ? 'danger' : 'warning'} />)}
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution" borderColor={ACCENT}>
            {(data.etiologies || []).map((e, i) => (
              <PctBar key={i} label={e.etiology} pct={e.pct} color={i === 0 ? colors.danger : i === 1 ? colors.warn : ACCENT} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Seizure Type Prevalence" borderColor={ACCENT2}>
            {Object.entries(data.seizure_type_prevalence || {}).map(([k, v], i) => (
              <PctBar key={i} label={k} pct={v} color={i === 0 ? colors.danger : ACCENT2} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Seizure Triggers (% patients affected)" borderColor="#7d3c98">
        <div className="row">
          {Object.entries(data.trigger_seizure_rates || {}).map(([k, v], i) => (
            <div key={i} className="col-md-6">
              <PctBar label={k} pct={v} color="#7d3c98" />
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Lifecycle Windows" borderColor={ACCENT2}>
        {(data.lifecycle_windows || []).map((w, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f0faf8', border: '1px solid #d5eae6' }}>
            <div className="fw-bold text-primary">{w.window}</div>
            <div className="small text-muted mb-1"><strong>Key events:</strong> {w.key_events}</div>
            <div className="small mb-1"><strong>Management:</strong> {w.management_focus}</div>
            <div className="small text-danger"><strong>Red flags:</strong> {w.red_flags}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Etiology ─────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('All');

  const cats = ['All', ...Array.from(new Set((data.patients || []).map(p => p.category)))];
  const visible = (data.patients || []).filter(p =>
    (filter === 'All' || p.category === filter) &&
    (p.id?.toLowerCase().includes(search.toLowerCase()) || p.etiology?.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <>
      <SectionCard title="Etiology Catalog (5 classes)" borderColor={ACCENT}>
        {(data.etiology_catalog || []).map((e, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f4f8fb', border: '1px solid #d0e4f5' }}>
            <div className="fw-bold" style={{ color: ACCENT }}>{e.etiology} <span className="badge bg-primary ms-2">{e.pct}% (n={e.n})</span></div>
            <div className="small mt-1"><strong>Mechanism:</strong> {e.mechanism}</div>
            <div className="small mt-1 text-muted"><strong>EEG:</strong> {e.eeg_correlate}</div>
            <div className="small mt-1 text-info"><strong>MRI:</strong> {e.mri_finding}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`Patient Records (N=${data.patients?.length || 0})`} borderColor={ACCENT2}>
        <div className="row mb-2">
          <div className="col-md-6">
            <input className="form-control form-control-sm" placeholder="Search ID or etiology…" value={search} onChange={e => setSearch(e.target.value)} />
          </div>
          <div className="col-md-6">
            <select className="form-select form-select-sm" value={filter} onChange={e => setFilter(e.target.value)}>
              {cats.map(c => <option key={c}>{c}</option>)}
            </select>
          </div>
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
            <thead>
              <tr>
                <th>ID</th><th>Age (y)</th><th>Sex</th><th>Onset (hrs)</th><th>Category</th>
                <th>Treatment</th><th>Control</th><th>Phase</th><th>CBZ (µg/mL)</th>
                <th>Na+</th><th>KD</th><th>BHB</th><th>HLA</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.age_years}</td>
                  <td>{p.sex}</td>
                  <td className={p.onset_age_hours <= 24 ? 'text-danger fw-bold' : ''}>{p.onset_age_hours}h</td>
                  <td><span className="badge" style={{ backgroundColor: ACCENT, fontSize: 10 }}>{p.category?.replace('De-novo-KCNQ2-','').replace('-','\n')}</span></td>
                  <td>{p.current_treatment}</td>
                  <td className={p.seizure_control === 'drug-resistant' ? 'text-danger fw-bold' : p.seizure_control === 'seizure-free' ? 'text-success' : ''}>
                    {p.seizure_control}
                  </td>
                  <td>{p.disease_phase}</td>
                  <td className={p.cbz_level_ugml < 4 ? 'text-danger' : p.cbz_level_ugml > 12 ? 'text-warning' : 'text-success'}>
                    {p.cbz_level_ugml ?? '—'}
                  </td>
                  <td className={p.na_level_mmoll < 133 ? 'text-danger fw-bold' : ''}>{p.na_level_mmoll}</td>
                  <td>{p.kd_on}</td>
                  <td>{p.bhb_mmol_l ?? '—'}</td>
                  <td className={p.hla_b1502_tested === 'N' ? 'text-danger' : 'text-success'}>{p.hla_b1502_tested}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Seizure Types & Triggers ─────────────────────────────────────────────
function SeizureTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <SectionCard title="Seizure Types (4) — EEG Correlates & Clinical Tips" borderColor={ACCENT2}>
        {(data.seizure_types || []).map((s, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: '#f0faf8', border: '1px solid #c8e6c9' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold" style={{ color: ACCENT2 }}>{s.type}</span>
              <span className="badge bg-success">{s.prevalence_pct}% of patients</span>
            </div>
            <div className="small mb-1">{s.description}</div>
            <div className="small text-muted mt-1"><strong>EEG:</strong> {s.eeg_correlate}</div>
            <div className="alert alert-info py-1 mt-2 mb-0" style={{ fontSize: 12 }}><strong>Clinical tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8) — Mechanisms & Actions" borderColor="#7d3c98">
        {(data.triggers || []).map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{ background: '#f9f0ff', border: '1px solid #e1bee7' }}>
            <div className="d-flex justify-content-between mb-1">
              <span className="fw-bold" style={{ color: '#7d3c98' }}>{t.trigger}</span>
              <span className="badge bg-warning text-dark">{t.frequency_pct}%</span>
            </div>
            <div className="small">{t.mechanism}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments ───────────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;

  const levelColor = (lvl) => {
    if (lvl?.includes('WITHDRAWN')) return '#e74c3c';
    if (lvl?.includes('Level A')) return '#1a5276';
    if (lvl?.includes('Level B')) return '#1a5276';
    if (lvl?.includes('Level C')) return '#7d6608';
    return '#555';
  };

  return (
    <>
      <div className="alert alert-success mb-3" style={{ fontSize: 13 }}>
        <strong>KCNQ2-Specific Treatment Principle:</strong> CBZ/OXC are FIRST-LINE — unique dual mechanism:
        Na-channel block + M-current (Kv7.2/Kv7.3) enhancement. PB alone is insufficient.
        HLA-B*15:02 screening MANDATORY before CBZ/OXC in South/SE Asian patients (CPIC Level A).
      </div>

      {(data.treatments || []).map((t, i) => (
        <div key={i} className={`card mb-3 shadow-sm ${t.level?.includes('WITHDRAWN') ? 'border-danger' : ''}`}
          style={{ borderLeft: `4px solid ${levelColor(t.level)}` }}>
          <div className="card-header d-flex justify-content-between align-items-center py-2"
            style={{ backgroundColor: t.level?.includes('WITHDRAWN') ? '#fdecea' : '#eaf4fb' }}>
            <span className="fw-bold">{t.name}</span>
            <span className={`badge ${t.level?.includes('WITHDRAWN') ? 'bg-danger' : t.level?.includes('Level A') ? 'bg-primary' : t.level?.includes('Level B') ? 'bg-primary' : 'bg-secondary'}`}
              style={{ fontSize: 11 }}>{t.level}</span>
          </div>
          <div className="card-body" style={{ fontSize: 13 }}>
            <div className="mb-1"><strong>Indication:</strong> {t.indication}</div>
            <div className="mb-1"><strong>Dose:</strong> <code>{t.dose}</code></div>
            <div className="mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="mb-1"><strong>Safety:</strong> {t.safety}</div>
            <div className="mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.kcnq2_specific_note && (
              <div className={`alert py-1 mt-2 mb-0 ${t.level?.includes('WITHDRAWN') ? 'alert-danger' : t.kcnq2_specific_note?.includes('KCNQ2-SPECIFIC') ? 'alert-success' : 'alert-info'}`}
                style={{ fontSize: 12 }}>
                <strong>KCNQ2 Note:</strong> {t.kcnq2_specific_note}
              </div>
            )}
          </div>
        </div>
      ))}

      <SectionCard title="Absolute Contraindications (4)" borderColor="#e74c3c">
        {(data.aed_monitoring ? [] : []).concat([])}
        {/* Rendered from definitions tab */}
        <div className="text-muted small">See Definitions tab for absolute contraindications detail.</div>
      </SectionCard>

      <SectionCard title="AED Monitoring Protocol (5 items)" borderColor={ACCENT}>
        {(data.aed_monitoring || []).map((m, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f4f8fb', border: '1px solid #d0e4f5' }}>
            <div className="fw-bold small">{m.item}</div>
            <div className="small text-muted"><strong>Schedule:</strong> {m.schedule}</div>
            <div className="small"><strong>Target:</strong> <code>{m.target}</code></div>
            <div className="small text-info"><strong>Rationale:</strong> {m.rationale}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <>
      <SectionCard title="Absolute Contraindications (4)" borderColor="#e74c3c">
        {(data.absolute_contraindications || []).map((c, i) => (
          <div key={i} className="mb-3 p-3 rounded border-danger" style={{ background: '#fdecea', border: '1px solid #f5c6cb' }}>
            <div className="fw-bold text-danger">{c.drug}</div>
            <div className="badge bg-danger mb-1">{c.severity}</div>
            <div className="small mt-1">{c.reason}</div>
            <div className="small text-muted mt-1"><strong>Standard:</strong> {c.standard}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Thresholds (8)" borderColor={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
            <thead className="table-primary">
              <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td><code>{t.threshold}</code></td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Key Concepts (14)" borderColor={ACCENT2}>
        {(data.concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f0faf8', border: '1px solid #d5eae6' }}>
            <div className="fw-bold" style={{ color: ACCENT2 }}>{c.term}</div>
            <div className="small mt-1">{c.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Standards & Guidelines (6)" borderColor={ACCENT}>
        {(data.standards || []).map((s, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: '#f4f8fb' }}>
            <div className="fw-bold small">{s.code}</div>
            <div className="small text-muted">{s.title}</div>
            <div className="small">{s.relevance}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References (6)" borderColor="#7d3c98">
        {(data.references || []).map((r, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: '#f9f0ff' }}>
            <div className="fw-bold small" style={{ color: '#7d3c98' }}>{r.citation} — {r.title}</div>
            <div className="small text-muted">{r.journal}</div>
            <div className="small">{r.key_finding}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function KCNQ2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnq2/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnq2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnq2/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="pt" data={breakdown} />,
    <SeizureTab key="sz" data={breakdown} />,
    <TreatmentsTab key="tx" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT}, ${ACCENT2})` }}>
        <h4 className="mb-0 fw-bold">⚡ KCNQ2 Encephalopathy (KCNQ2-DEE)</h4>
        <div className="small opacity-75">Neonatal-onset DEE · Kv7.2 M-current channelopathy · 20q13.33 · CBZ/OXC first-line KCNQ2-specific therapy · XEN496 EPIK trial NCT05374343</div>
      </div>

      <div className="mb-3">
        {TABS.map((t, i) => <TabBtn key={i} label={t} active={tab === i} onClick={() => setTab(i)} />)}
      </div>

      {tabContent[tab]}

      <div className="text-muted text-end" style={{ fontSize: 11 }}>
        KCNQ2-DEE Dashboard · N=41 cohort · Sources: Pisano 2015 · Numis 2014 · Millichap 2017 · CPIC-HLA-B-CBZ-2023 · ILAE-2022 · NICE-NG217
      </div>
    </div>
  );
}
