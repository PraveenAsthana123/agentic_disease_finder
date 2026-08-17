'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#006064'; // dark cyan — L-type Cav1.2 cardiac-neuronal identity (distinct from CACNA1A/E/G/H/I, HCN1/2)
const DANGER = '#b71c1c'; // dark red — CI / cardiac danger
const SUCCESS = '#1b5e20'; // dark green — seizure freedom
const WARN = '#e65100';   // deep orange — LQTS / cardiac warning

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
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const ciSummary = data.contraindications_summary || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5, background: '#e0f2f1' }}>
        <strong>🧬 CACNA1C (12p13.33) — Cav1.2 L-type HVA Ca²⁺ Channel — Timothy Syndrome / LQTS8 / DEE — GOF de novo:</strong>{' '}
        CACNA1C encodes Cav1.2 (α1C), the dominant <strong>L-type</strong> (dihydropyridine-sensitive) HVA Ca²⁺ channel in cardiac muscle and neurons.{' '}
        GOF G406R (TS1) / G402S (TS2) → impaired voltage-dependent inactivation (VDI) → <strong>enlarged window current</strong> → prolonged Ca²⁺ influx{' '}
        → cardiac (<strong>LQTS8</strong>; QTc&gt;500 ms; 2:1 AV block risk) + neuronal (DEE) + autism + syndactyly.{' '}
        <strong>Cav1 subfamily:</strong> Cav1.1/CACNA1S (skeletal) · Cav1.2/CACNA1C (cardiac-neuronal/TS) · Cav1.3/CACNA1D (cochlear/SANDD) · Cav1.4/CACNA1F (retinal/CSNB2).{' '}
        Splawski 2004 Cell 119:19 (TS1/G406R); Jacobs 2006 (verapamil precision).{' '}
        <span style={{ color: DANGER }} className="fw-bold">
          PRECISION: Verapamil Level B (L-type blocker; dual cardiac+seizure benefit).{' '}
          ABSOLUTE CI: Class Ia/III antiarrhythmics (fatal QT) · TGB (NCSE+QT) · VPA+POLG1 (Alpers).{' '}
          HIGH RISK: CBZ/OXC/PHT (reduce verapamil via CYP3A4) · VGB long-term (VFD).{' '}
          MANDATORY: ECG+Holter every clinic; POLG1 before VPA; VPPP females.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color={SUCCESS} />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color={DANGER} />
        <KPI label="LQTS Present" value={kpis.lqts_present_n} color={WARN} />
        <KPI label="Verapamil Tried" value={kpis.verapamil_tried_n} color={COLOR} />
        <KPI label="ACTH Received" value={kpis.acth_received_n} color={COLOR} />
        <KPI label="West/IS (spasms)" value={kpis.spasms_n} color={WARN} />
        <KPI label="KD On" value={kpis.kd_on_n} color={COLOR} />
        <KPI label="Autism/ASD" value={kpis.autism_asd_n} color={COLOR} />
        <KPI label="Syndactyly (TS1)" value={kpis.syndactyly_n} color={WARN} />
        <KPI label="Cardiac Device" value={kpis.cardiac_device_n} color={DANGER} />
        <KPI label="Avg Age (yrs)" value={kpis.avg_age_years} />
      </div>

      <div className="row g-3">
        {/* Etiology */}
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution (GOF Spectrum)">
            {etiologies.map((e, i) => (
              <Bar key={i} label={e.category.replace(/-/g, ' ')} value={e.pct} max={maxEtio} />
            ))}
          </SectionCard>
        </div>

        {/* Seizure Summary */}
        <div className="col-md-6">
          <SectionCard title="Seizure Types (% of cohort)">
            {data.seizure_summary?.map((st, i) => (
              <Bar key={i} label={st.type.replace(/-/g, ' ')} value={st.pct} max={100} color={WARN} />
            ))}
          </SectionCard>
        </div>

        {/* Contraindications */}
        <div className="col-md-6">
          <SectionCard title="Contraindications (CACNA1C-Specific)" borderColor={DANGER}>
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Drug/Class</th><th>Risk</th></tr></thead>
              <tbody>
                {ciSummary.map((ci, i) => (
                  <tr key={i}>
                    <td className="small">{ci.drug}</td>
                    <td><span className="badge" style={{ background: DANGER, fontSize: '0.7rem' }}>{ci.risk}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>
        </div>

        {/* Treatments */}
        <div className="col-md-6">
          <SectionCard title="Treatments (Evidence Levels)">
            <table className="table table-sm table-hover mb-0">
              <thead><tr><th>Drug</th><th>Level</th></tr></thead>
              <tbody>
                {treatments.map((t, i) => (
                  <tr key={i}>
                    <td className="small">{t.drug}</td>
                    <td><span className="badge" style={{ background: COLOR, fontSize: '0.7rem' }}>{t.level}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </SectionCard>
        </div>

        {/* Monitoring */}
        <div className="col-md-6">
          <SectionCard title="Monitoring Priorities (Top 6)">
            <ul className="list-unstyled mb-0 small">
              {monitoring.map((m, i) => (
                <li key={i} className="mb-1">
                  <span className="fw-semibold" style={{ color: COLOR }}>{m.item?.slice(0, 55)}</span>
                  <br /><span className="text-muted">{m.frequency?.slice(0, 60)}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>

        {/* Lifecycle */}
        <div className="col-md-6">
          <SectionCard title="Lifecycle Windows">
            <ul className="list-unstyled mb-0 small">
              {lifecycle.map((lc, i) => (
                <li key={i} className="mb-1">
                  <span className="fw-semibold" style={{ color: COLOR }}>{lc.window}</span>
                  <br /><span className="text-muted">{lc.key}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>

        {/* Thresholds */}
        <div className="col-12">
          <SectionCard title="Clinical Thresholds">
            <div className="row row-cols-2 row-cols-md-3 g-2">
              {thresholds.map((th, i) => (
                <div key={i} className="col">
                  <div className="border rounded p-2 h-100">
                    <div className="fw-semibold small" style={{ color: COLOR }}>{th.param}</div>
                    <div className="small text-muted">{th.action_threshold}</div>
                  </div>
                </div>
              ))}
            </div>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

function BreakdownTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      {/* Etiology Catalog */}
      <SectionCard title="Etiology Catalog — 5-Class GOF Spectrum">
        <div className="row g-3">
          {(data.etiologies || []).map((e, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="fw-bold mb-1" style={{ color: COLOR }}>{e.category.replace(/-/g, ' ')} — {e.pct}%</div>
                <div className="small"><strong>Mechanism:</strong> {e.mechanism}</div>
                <div className="small mt-1"><strong>Phenotype:</strong> {e.phenotype}</div>
                <div className="small mt-1"><strong>EEG:</strong> {e.eeg_pattern}</div>
                <div className="small mt-1 text-muted"><em>{e.reference}</em></div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Patients Table */}
      <SectionCard title="Patient Cohort (40 Patients — Synthetic)">
        <div className="table-responsive">
          <table className="table table-sm table-hover table-striped mb-0" style={{ fontSize: '0.78rem' }}>
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Etiology</th><th>Age</th><th>Variant</th>
                <th>QTc (ms)</th><th>Spasms</th><th>LQTS</th><th>Verapamil</th>
                <th>KD</th><th>Autism</th><th>Syndactyly</th><th>SeizFree</th><th>DRE</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td style={{ maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology?.replace(/-/g, ' ')}</td>
                  <td>{p.current_age}y</td>
                  <td>{p.variant}</td>
                  <td style={{ color: p.qtc_ms > 500 ? DANGER : p.qtc_ms > 460 ? WARN : 'inherit', fontWeight: p.qtc_ms > 500 ? 'bold' : 'normal' }}>{p.qtc_ms}</td>
                  <td>{p.spasms_present ? '✓' : '—'}</td>
                  <td>{p.lqts_present ? '✓' : '—'}</td>
                  <td>{p.verapamil_tried ? '✓' : '—'}</td>
                  <td>{p.kd_on ? '✓' : '—'}</td>
                  <td>{p.autism_asd ? '✓' : '—'}</td>
                  <td>{p.syndactyly ? '✓' : '—'}</td>
                  <td style={{ color: p.seizure_free ? SUCCESS : 'inherit' }}>{p.seizure_free ? '✓' : '—'}</td>
                  <td style={{ color: p.drug_resistant ? DANGER : 'inherit' }}>{p.drug_resistant ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizuresTriggerTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      {/* Seizure Types */}
      <SectionCard title="Seizure Types — 5-Type Profile">
        <div className="row g-3">
          {(data.seizure_types || []).map((st, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ color: COLOR }}>{st.type.replace(/-/g, ' ')}</span>
                  <span className="badge" style={{ background: WARN }}>{st.pct}%</span>
                </div>
                <div className="small"><strong>Onset:</strong> {st.onset_age}</div>
                <div className="small mt-1"><strong>EEG:</strong> {st.eeg}</div>
                <div className="small mt-1"><strong>Semiology:</strong> {st.semiology}</div>
                <div className="small mt-1 text-muted" style={{ color: COLOR }}><em>Tip: {st.clinical_tip}</em></div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Triggers */}
      <SectionCard title="Seizure / Cardiac Triggers — 8 Triggers">
        <div className="row g-3">
          {(data.triggers || []).map((tr, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="d-flex justify-content-between align-items-center mb-1">
                  <span className="fw-bold" style={{ color: WARN }}>{tr.trigger}</span>
                  <span className="badge" style={{ background: COLOR }}>{tr.pct}%</span>
                </div>
                <div className="small"><strong>Mechanism:</strong> {tr.mechanism}</div>
                <div className="small mt-1 text-muted"><em>Management: {tr.management}</em></div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const [expanded, setExpanded] = useState(null);

  return (
    <div>
      {/* Contraindications */}
      <SectionCard title="Contraindications — 5 CACNA1C-Specific CIs" borderColor={DANGER}>
        <div className="row g-3 mb-2">
          {(data.contraindications || []).map((ci, i) => (
            <div key={i} className="col-md-6">
              <div className="border border-danger rounded p-3 h-100">
                <div className="fw-bold mb-1" style={{ color: DANGER }}>{ci.drug.replace(/-/g, ' ')}</div>
                <span className="badge mb-2" style={{ background: DANGER }}>{ci.risk}</span>
                <div className="small"><strong>Mechanism:</strong> {ci.mechanism}</div>
                <div className="small mt-1" style={{ color: COLOR }}><em>CACNA1C note: {ci.cacna1c_note}</em></div>
                <div className="small mt-1 text-muted"><strong>Alternative:</strong> {ci.alternative}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Treatments — 8 Agents (Evidence + CACNA1C-Specific Notes)">
        {(data.treatments || []).map((t, i) => (
          <div key={i} className="border rounded mb-2">
            <button
              className="btn btn-link w-100 text-start fw-semibold py-2 px-3 d-flex justify-content-between"
              style={{ color: COLOR, textDecoration: 'none' }}
              onClick={() => setExpanded(expanded === i ? null : i)}
            >
              <span>{t.drug?.split(' (')[0]}</span>
              <span className="badge" style={{ background: COLOR, fontSize: '0.7rem' }}>{t.level?.split(' —')[0]}</span>
            </button>
            {expanded === i && (
              <div className="px-3 pb-3 small">
                <div><strong>Level of evidence:</strong> {t.level}</div>
                <div className="mt-1"><strong>MOA:</strong> {t.moa}</div>
                <div className="mt-1"><strong>Dose:</strong> {t.dose}</div>
                <div className="mt-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                <div className="mt-1"><strong>Safety:</strong> {t.safety}</div>
                <div className="mt-1"><strong>Monitoring:</strong> {t.monitoring}</div>
                <div className="mt-1 p-2 rounded" style={{ background: '#e0f2f1', color: COLOR }}><strong>CACNA1C note:</strong> {t.cacna1c_note}</div>
              </div>
            )}
          </div>
        ))}
      </SectionCard>

      {/* Monitoring */}
      <SectionCard title="Monitoring — 14 Items (CACNA1C + Cardiac)">
        <div className="row g-3">
          {(data.monitoring || []).map((m, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-2 h-100">
                <div className="fw-semibold small" style={{ color: COLOR }}>{m.item}</div>
                <div className="small text-muted">{m.frequency}</div>
                {m.threshold && <div className="small mt-1" style={{ color: WARN }}><strong>Threshold:</strong> {m.threshold}</div>}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const gs = data.gene_summary || {};

  return (
    <div>
      {/* Gene Summary Card */}
      <SectionCard title="CACNA1C — Gene Summary">
        <div className="row g-2 small">
          {Object.entries(gs).map(([k, v]) => (
            <div key={k} className="col-md-6">
              <span className="fw-semibold text-capitalize" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}: </span>
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Key Concepts */}
      <SectionCard title="Key Concepts — 15 Definitions">
        <div className="row g-3">
          {(data.definitions || []).map((d, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-3 h-100">
                <div className="fw-bold mb-1" style={{ color: COLOR }}>{d.term}</div>
                <div className="small text-muted">{d.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="Clinical Thresholds — 12">
        <div className="row g-2">
          {(data.thresholds || []).map((th, i) => (
            <div key={i} className="col-md-6">
              <div className="border rounded p-2 h-100">
                <div className="fw-semibold small" style={{ color: WARN }}>{th.param} ({th.unit})</div>
                <div className="small text-muted">{th.action_threshold}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Standards + References */}
      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="Evidence Standards — 12">
            <ul className="list-unstyled mb-0 small">
              {(data.standards || []).map((s, i) => (
                <li key={i} className="mb-1">
                  <span className="fw-semibold" style={{ color: COLOR }}>{s.code}:</span>{' '}
                  <span className="text-muted">{s.title}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="References — 6">
            <ul className="list-unstyled mb-0 small">
              {(data.references || []).map((r, i) => (
                <li key={i} className="mb-2">
                  <span className="fw-semibold" style={{ color: COLOR }}>[{r.key}]</span>{' '}
                  <span className="text-muted">{r.citation}</span>
                </li>
              ))}
            </ul>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

export default function CACNA1CPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cacna1c/overview`).then(r => r.json()),
      fetch(`${API}/api/cacna1c/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cacna1c/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="d-flex justify-content-center align-items-center" style={{ minHeight: 300 }}><div className="spinner-border" style={{ color: COLOR }} /></div>;
  if (error) return <div className="alert alert-danger m-3">{error}</div>;

  const tabData = [overview, breakdown, breakdown, breakdown, definitions];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="rounded p-3 mb-3 text-white" style={{ background: COLOR }}>
        <h4 className="mb-0 fw-bold">🧬 CACNA1C Epilepsy — Timothy Syndrome / LQTS8 / DEE</h4>
        <div className="small opacity-75 mt-1">
          Cav1.2 L-type HVA Ca²⁺ Channel · GOF G406R (TS1) / G402S (TS2) · VDI Impairment · 12p13.33
          · OMIM #601005 · Verapamil Precision Level B · Splawski 2004 Cell · 40-Patient Cohort
        </div>
        <div className="small opacity-75">
          Cav1 Subfamily: Cav1.1/CACNA1S (skeletal) · <strong>Cav1.2/CACNA1C (cardiac-neuronal/TS)</strong> · Cav1.3/CACNA1D (cochlear/SANDD) · Cav1.4/CACNA1F (retinal/CSNB2)
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab Content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <BreakdownTab data={breakdown} />}
      {tab === 2 && <SeizuresTriggerTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
