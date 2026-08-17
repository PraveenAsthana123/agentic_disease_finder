'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#1565c0'; // deep blue — TC-dominant HCN2/Ih pacemaker identity (distinct from HCN1 #1a237e deep indigo)
const DANGER = '#c62828'; // dark red — CI / danger
const SUCCESS = '#2e7d32'; // dark green — ETX / success
const WARN = '#e65100';   // deep orange — LTG caution / fever warning

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
      <div className="alert alert-info py-2 small mb-3 border" style={{ borderColor: COLOR, borderLeftWidth: 5 }}>
        <strong>⚡ HCN2 (19p13.3) — Ih Pacemaker Channel — TC-Dominant — GEFS+ / Febrile Seizures / CAE:</strong>{' '}
        HCN2 generates the Ih 'funny' pacemaker current in <strong>thalamic relay (TC) neurons</strong>.{' '}
        LOF → reduced TC Ih → prolonged TC hyperpolarisation → larger Cav3.1/Cav3.2 LTCS → <strong>3-Hz SWD → absence / GEFS+</strong>.{' '}
        Ludwig 2003: HCN2 KO mice → spontaneous absence + sinus dysrhythmia.{' '}
        <strong>HCN2 vs HCN1:</strong> MILDER (GGE not DEE24) · TC-dominant (not hippocampus) · LOF primary (not dual GOF/LOF).{' '}
        <span className="text-danger fw-bold">
          ABSOLUTE CI: CBZ / OXC / PHT (GGE aggravation) · TGB (NCSE) · VPA+POLG1 (Alpers).{' '}
          LTG HIGH RISK in LOF (Poolos 2002 — Ih blocker → absence worsening). POLG1 before VPA mandatory.
        </span>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color={SUCCESS} />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color={DANGER} />
        <KPI label="On ETX" value={kpis.on_etx_n} />
        <KPI label="Febrile Seizures" value={kpis.febrile_seizures_n} color={WARN} />
        <KPI label="Absence EEG" value={kpis.absence_n} />
        <KPI label="GTCS Present" value={kpis.gtcs_n} color={DANGER} />
        <KPI label="HV-SWD Positive" value={kpis.hv_swd_n} color={WARN} />
        <KPI label="Catamenial" value={kpis.catamenial_n} color="#6a1a9a" />
        <KPI label="Avg Age (yr)" value={kpis.avg_age_years} color="#37474f" />
      </div>

      {/* Etiology + CI */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="Etiology Distribution (n=40)">
            {etiologies.map(e => (
              <Bar key={e.category} label={e.category.replace('LOF-', '').replace('-', ' ')} value={e.pct} max={maxEtio} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Contraindications (Absolute / High Risk)" borderColor={DANGER}>
            {ciSummary.map(ci => (
              <div key={ci.drug} className="mb-2 small border-start border-danger ps-2">
                <span className="fw-semibold text-danger">{ci.drug.substring(0, 40)}</span>
                <div className="text-muted">{ci.risk}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Seizure types + Treatments */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="Seizure Type Prevalence">
            {(data.seizure_summary || []).map(s => (
              <Bar key={s.type} label={s.type.split(' (')[0]} value={s.pct} max={100} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Treatment Summary">
            {treatments.slice(0, 6).map(t => (
              <div key={t.drug} className="mb-1 small d-flex justify-content-between">
                <span className="text-truncate me-2">{t.drug}</span>
                <span className="badge text-bg-light border">{t.level}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Lifecycle + Thresholds */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="Lifecycle Windows">
            {lifecycle.map((lc, i) => (
              <div key={i} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR }}>
                <span className="fw-semibold">{lc.window}</span>
                <div className="text-muted">{lc.key}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Key Thresholds">
            {thresholds.map((t, i) => (
              <div key={i} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR }}>
                <span className="fw-semibold">{t.name}</span>
                <div className="text-muted">{t.value}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      {/* Monitoring summary */}
      <SectionCard title="Monitoring (first 6 items)">
        <div className="row g-2">
          {monitoring.map((m, i) => (
            <div key={i} className="col-md-6 small border-start ps-2" style={{ borderColor: COLOR }}>
              <span className="fw-semibold">{m.item}</span>
              <div className="text-muted">{m.frequency}</div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading patients…</div>;
  const patients = data.patients || [];
  const etiologies = data.etiologies || [];

  return (
    <div>
      <SectionCard title="Etiology Catalog (5 classes)">
        {etiologies.map(e => (
          <div key={e.category} className="mb-3 border rounded p-2">
            <div className="fw-semibold" style={{ color: COLOR }}>
              {e.category.replace(/-/g, ' ')} — {e.pct}%
            </div>
            <div className="small text-muted mb-1">{e.etiology}</div>
            <div className="small">{e.mechanism}</div>
            <div className="small mt-1">
              <span className="badge text-bg-secondary me-1">Onset: {e.onset_age_years}Y</span>
              <span className="text-muted">{e.outcome}</span>
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (n=40)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr>
                <th>ID</th><th>Age</th><th>Onset</th><th>Gender</th>
                <th>Syndrome</th><th>Etiology</th><th>Sz-Free</th>
                <th>ETX</th><th>FS</th><th>Absence</th><th>GTCS</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.patient_id}>
                  <td className="fw-semibold">{p.patient_id}</td>
                  <td>{p.current_age}Y</td>
                  <td>{p.onset_age}Y</td>
                  <td>{p.gender}</td>
                  <td className="text-truncate" style={{ maxWidth: 140 }}>{p.syndrome}</td>
                  <td className="text-truncate" style={{ maxWidth: 120 }}>{p.etiology?.replace(/-/g, ' ')}</td>
                  <td>{p.seizure_free ? <span className="text-success fw-bold">✓</span> : <span className="text-danger">✗</span>}</td>
                  <td>{p.etx_on ? '✓' : '—'}</td>
                  <td>{p.febrile_seizures ? '✓' : '—'}</td>
                  <td>{p.absence_present ? '✓' : '—'}</td>
                  <td>{p.gtcs_present ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading seizures…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];
  const maxTrig = Math.max(...triggers.map(t => t.pct || 0), 1);

  return (
    <div>
      <SectionCard title="Seizure Types (5)">
        {seizures.map(s => (
          <div key={s.type} className="mb-3 border rounded p-2">
            <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="fw-semibold" style={{ color: COLOR }}>{s.type}</span>
              <span className="badge text-bg-primary">{s.pct}%</span>
            </div>
            <div className="mb-1">
              <div className="progress" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.pct}%`, backgroundColor: COLOR }} />
              </div>
            </div>
            <div className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg_pattern}</div>
            <div className="small mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="small text-info"><strong>Clinical tip:</strong> {s.clinical_tip}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Seizure Triggers (8)">
        <div className="row g-2 mb-3">
          {triggers.map(t => (
            <div key={t.trigger} className="col-md-6">
              <Bar label={t.trigger} value={t.pct} max={maxTrig} />
            </div>
          ))}
        </div>
        {triggers.map(t => (
          <div key={t.trigger} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR }}>
            <span className="fw-semibold">{t.trigger} ({t.pct}%)</span>
            <div className="text-muted">{t.mechanism}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatments…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.contraindications || [];

  const levelColor = (level) => {
    if (level.includes('Level A')) return '#1b5e20';
    if (level.includes('Level B')) return '#1565c0';
    if (level.includes('CAUTION') || level.includes('HIGH RISK')) return '#b71c1c';
    return '#37474f';
  };

  return (
    <div>
      <div className="alert small py-2" style={{ backgroundColor: '#fff3e0', borderLeft: `5px solid ${WARN}` }}>
        <strong>⚠️ LTG CAUTION in HCN2-LOF:</strong> Lamotrigine is an Ih blocker (Poolos 2002 Nat Neurosci) —
        in LOF HCN2, LTG further reduces already-low Ih → TC stays hyperpolarised longer → worsened 3-Hz SWD →
        absence AGGRAVATION. <strong>Always obtain EEG 4-8 weeks after LTG initiation in any HCN2-GGE patient.</strong>
        Prefer ETX/VPA/LEV. LTG CONTRAINDICATED in confirmed LOF + absence phenotype.
      </div>

      <SectionCard title="Treatments (8)">
        {treatments.map(t => (
          <div key={t.drug} className="mb-3 border rounded p-2">
            <div className="d-flex justify-content-between align-items-start mb-1">
              <span className="fw-semibold" style={{ color: levelColor(t.level) }}>{t.drug}</span>
              <span className="badge ms-2" style={{ backgroundColor: levelColor(t.level), fontSize: 11 }}>
                {t.level.split(' —')[0].substring(0, 18)}
              </span>
            </div>
            <div className="small text-muted mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="small mb-1"><strong>MOA:</strong> {t.moa}</div>
            <div className="small text-success mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="small text-danger mb-1"><strong>Safety:</strong> {t.safety}</div>
            <div className="small text-muted mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            <div className="small p-1 rounded" style={{ background: '#e3f2fd' }}>
              <strong>HCN2-specific:</strong> {t.hcn2_note}
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Contraindications (5)" borderColor={DANGER}>
        {contraindications.map(ci => (
          <div key={ci.drug} className="mb-3 border-start border-danger ps-2">
            <div className="fw-semibold text-danger">{ci.drug}</div>
            <div className="small text-muted mb-1"><strong>Risk:</strong> {ci.risk}</div>
            <div className="small mb-1"><strong>Mechanism:</strong> {ci.mechanism}</div>
            <div className="small text-success"><strong>Alternative:</strong> {ci.alternative}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  const defs = data.definitions || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const refs = data.references || [];
  const gs = data.gene_summary || {};

  return (
    <div>
      <SectionCard title="Gene Summary — HCN2 (19p13.3)">
        <div className="row g-2 small">
          {Object.entries(gs).map(([k, v]) => (
            <div key={k} className="col-md-6">
              <div className="border-start ps-2 mb-1" style={{ borderColor: COLOR }}>
                <span className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}:</span>{' '}
                <span className="text-muted">{v}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key Concepts (15 Definitions)">
        {defs.map(d => (
          <div key={d.term} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR }}>
            <span className="fw-semibold" style={{ color: COLOR }}>{d.term}</span>
            <div className="text-muted">{d.definition}</div>
          </div>
        ))}
      </SectionCard>

      <div className="row g-3">
        <div className="col-md-6">
          <SectionCard title="Clinical Thresholds (12)">
            {thresholds.map((t, i) => (
              <div key={i} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR }}>
                <span className="fw-semibold">{t.name}</span>
                <div className="text-muted">{t.value}</div>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Standards (12)">
            {standards.map((s, i) => (
              <div key={i} className="mb-2 small border-start ps-2" style={{ borderColor: COLOR }}>
                <span className="fw-semibold">{s.name}</span>
                <div className="text-muted">{s.description}</div>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="References (6)">
        <ol className="mb-0 small">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

export default function HCN2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState({});
  const [error, setError] = useState(null);

  const fetchData = async (endpoint, setter, key) => {
    if (loading[key]) return;
    setLoading(l => ({ ...l, [key]: true }));
    try {
      const res = await fetch(`${API}${endpoint}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setter(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(l => ({ ...l, [key]: false }));
    }
  };

  useEffect(() => { fetchData('/api/hcn2/overview', setOverview, 'overview'); }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2 || tab === 3) {
      if (!breakdown) fetchData('/api/hcn2/breakdown', setBreakdown, 'breakdown');
    }
    if (tab === 4) {
      if (!definitions) fetchData('/api/hcn2/definitions', setDefinitions, 'definitions');
    }
  }, [tab]);

  const tabContent = [
    <OverviewTab key="ov" data={overview} />,
    <PatientsTab key="pt" data={breakdown} />,
    <SeizuresTab key="sz" data={breakdown} />,
    <TreatmentsTab key="tx" data={breakdown} />,
    <DefinitionsTab key="df" data={definitions} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            ⚡ HCN2 Epilepsy
          </h4>
          <div className="text-muted small">
            GEFS+ / Febrile Seizures / CAE / GGE · Ih Pacemaker Channel · TC-Dominant · LOF · 19p13.3 · n=40
          </div>
        </div>
        <div className="ms-auto">
          <span className="badge me-1" style={{ background: COLOR }}>HCN2 LOF</span>
          <span className="badge me-1" style={{ background: '#1b5e20' }}>ETX Level A</span>
          <span className="badge me-1" style={{ background: DANGER }}>LTG HIGH RISK-LOF</span>
          <span className="badge" style={{ background: WARN }}>Fever CI</span>
        </div>
      </div>

      {error && (
        <div className="alert alert-warning py-2 small">
          API error: {error}. Check backend at {API}.
        </div>
      )}

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { backgroundColor: COLOR, borderColor: COLOR, color: '#fff' } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {Object.values(loading).some(Boolean) && (
        <div className="text-center py-2 text-muted small">Loading…</div>
      )}

      {tabContent[tab]}
    </div>
  );
}
