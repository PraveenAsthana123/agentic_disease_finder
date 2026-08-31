'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure & Dystonia', 'Treatments', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — NBIA/iron accumulation
const LIGHT = '#e8eaf6';

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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const cis = data.contraindications_summary || [];
  const ddx = data.ddx_table || [];
  const highlights = data.clinical_highlights || [];
  const tier = data.tier_summary || {};
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>PANK2 (20p13) — 570aa mitochondrial pantothenate kinase · OMIM 606157/234200 · PKAN/NBIA1:</strong>{' '}
        Most common NBIA (~35-50%). AR biallelic. CoA deficiency + cysteine-iron Fenton reaction → GP + SN iron.{' '}
        <strong className="text-danger">Eye-of-the-tiger sign (bilateral GP T2: central hyperintense + hypointense rim) = PATHOGNOMONIC.</strong>{' '}
        Classic PKAN (onset &lt;6yr, rapid dystonia) vs Atypical PKAN (~13yr, speech, OCD/psychiatric).{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          PHT/VGB AVOID. POLG mandatory before VPA. GPi-DBS Level B.
          Deferiprone: no functional benefit (TIRCON 2019). Fosmetpantotenate (PANK2 bypass) investigational.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Classic PKAN" value={kpis.n_classic} color={COLOR} />
        <KPI label="Atypical PKAN" value={kpis.n_atypical} color="#3949ab" />
        <KPI label="Eye-of-Tiger" value={`${kpis.eye_of_tiger_pct}%`} color="#dc3545" />
        <KPI label="Has Seizures" value={`${kpis.has_seizures_pct}%`} color="#e65100" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Severe Dystonia" value={`${kpis.dystonia_severe_pct}%`} color="#dc3545" />
        <KPI label="Lost Ambulation" value={`${kpis.ambulation_lost_pct}%`} color="#dc3545" />
        <KPI label="Pigmentary Retina" value={`${kpis.retinal_pct}%`} color="#6f42c1" />
        <KPI label="Acanthocytes" value={`${kpis.acanthocytes_pct}%`} color="#6f42c1" />
        <KPI label="GPi-DBS" value={`${kpis.dbs_pct}%`} color="#0d6efd" />
        <KPI label="Psychiatric (Atyp)" value={`${kpis.psychiatric_pct}%`} color="#6f42c1" />
        <KPI label="OCD" value={`${kpis.ocd_pct}%`} color="#6f42c1" />
        <KPI label="Cognitive Decline" value={`${kpis.cognitive_decline_pct}%`} color={COLOR} />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#0d6efd" />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="Mean Onset (yr)" value={kpis.mean_onset_yr} color={COLOR} />
        <KPI label="Classic Onset (yr)" value={kpis.classic_mean_onset_yr} color="#dc3545" />
        <KPI label="Atypical Onset (yr)" value={kpis.atypical_mean_onset_yr} color="#3949ab" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Etiology Distribution (4 Categories)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <Bar key={i} label={`${e.etiology} (n=${e.n})`} value={e.pct} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Highlights
            </div>
            <div className="card-body p-2">
              <ul className="small mb-0 ps-3">
                {highlights.map((h, i) => <li key={i} className="mb-1">{h}</li>)}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small text-danger" style={{ background: '#fff3cd' }}>
              Contraindications
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Drug</th><th>Reason</th></tr></thead>
                <tbody>
                  {cis.map((c, i) => (
                    <tr key={i}>
                      <td className="fw-bold text-danger text-nowrap">{c.drug}</td>
                      <td>{c.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Monitoring Schedule
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Item</th><th>Frequency</th></tr></thead>
                <tbody>
                  {monitoring.map((m, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{m.item}</td>
                      <td className="text-muted">{m.frequency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Disease Lifecycle (Classic → Atypical timelines)
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Stage</th><th>Clinical Features</th></tr></thead>
                <tbody>
                  {lifecycle.map((l, i) => (
                    <tr key={i} style={i >= 4 ? { background: '#fce4ec' } : {}}>
                      <td className="fw-semibold text-nowrap">{l.stage}</td>
                      <td>{l.features}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              DDx — PKAN vs Other NBIA + Dystonia Syndromes
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead>
                  <tr><th>Condition</th><th>Distinguishing from PKAN</th><th>Shared</th></tr>
                </thead>
                <tbody>
                  {ddx.map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold text-nowrap">{d.condition}</td>
                      <td>{d.distinguishing}</td>
                      <td className="text-muted">{d.shared}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Thresholds
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead><tr><th>Parameter</th><th>Threshold</th><th>Significance</th></tr></thead>
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.parameter}</td>
                      <td><code>{t.threshold}</code></td>
                      <td className="text-muted">{t.significance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Tier Summary
        </div>
        <div className="card-body p-2">
          <table className="table table-sm mb-0 small">
            <tbody>
              {Object.entries(tier).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-capitalize">{k.replace(/_/g, ' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const etio = data.etiology_breakdown || [];
  const pts = data.per_patient || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Etiology Breakdown — 4 Categories (n=40, seed-517)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr>
                <th>Etiology</th><th>n</th><th>%</th>
                <th>Classic %</th><th>Eye-of-Tiger</th>
                <th>Seizures</th><th>Retinal</th>
                <th>DBS</th><th>Mean Onset (yr)</th>
              </tr>
            </thead>
            <tbody>
              {etio.map((e, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{e.etiology}</td>
                  <td>{e.n}</td>
                  <td>{e.pct}%</td>
                  <td>{e.classic_pct}%</td>
                  <td>{e.eye_of_tiger_pct}%</td>
                  <td>{e.has_seizures_pct}%</td>
                  <td>{e.retinal_pct}%</td>
                  <td>{e.dbs_pct}%</td>
                  <td>{e.mean_onset_yr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Per-Patient Summary (n=40)
        </div>
        <div className="card-body p-0">
          <div style={{ maxHeight: 420, overflowY: 'auto' }}>
            <table className="table table-sm mb-0 small">
              <thead>
                <tr>
                  <th>ID</th><th>Form</th><th>Etiology</th><th>Onset(yr)</th>
                  <th>EoT</th><th>Dystonia</th><th>Amb.Lost</th>
                  <th>Seizures</th><th>DR</th><th>AEDs</th>
                  <th>Retinal</th><th>Acanth.</th><th>DBS</th>
                  <th>Psych</th><th>OCD</th><th>POLG</th><th>SF</th>
                </tr>
              </thead>
              <tbody>
                {pts.map((p, i) => (
                  <tr key={i} style={p.eye_of_tiger ? { background: '#e8eaf6' } : {}}>
                    <td className="fw-semibold">{p.id}</td>
                    <td>
                      <span className="badge" style={{ background: p.form === 'Classic' ? '#dc3545' : '#3949ab' }}>
                        {p.form}
                      </span>
                    </td>
                    <td>{p.etiology}</td>
                    <td>{p.onset_yr}</td>
                    <td>{p.eye_of_tiger ? '🔴' : '–'}</td>
                    <td>{p.dystonia_severity}</td>
                    <td>{p.ambulation_lost ? '⚠️' : '–'}</td>
                    <td>{p.has_seizures ? '✓' : '–'}</td>
                    <td>{p.drug_resistant ? '🔴' : '–'}</td>
                    <td>{p.n_aeds}</td>
                    <td>{p.retinal ? '⚠️' : '–'}</td>
                    <td>{p.acanthocytes ? '✓' : '–'}</td>
                    <td>{p.dbs ? '✓' : '–'}</td>
                    <td>{p.psychiatric ? '✓' : '–'}</td>
                    <td>{p.ocd ? '✓' : '–'}</td>
                    <td>{p.polg_tested ? '✓' : '–'}</td>
                    <td>{p.seizure_free ? '✓' : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function SeizureDystoniaTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const sz = data.seizure_breakdown || [];
  const lc = data.lifecycle || [];
  const thr = data.thresholds || [];
  const refs = data.references || [];
  const stds = data.standards || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Seizure Type Breakdown (40-50% of PKAN patients, n=40)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr><th>Type</th><th>n</th><th>%</th><th>Drug-Resistant (%)</th></tr>
            </thead>
            <tbody>
              {sz.map((s, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{s.type}</td>
                  <td>{s.n}</td>
                  <td>{s.pct}%</td>
                  <td>{s.drug_resistant_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: '#e8eaf6' }}>
          Clinical Lifecycle — Classic &amp; Atypical PKAN
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Stage</th><th>Clinical Features</th></tr></thead>
            <tbody>
              {lc.map((l, i) => (
                <tr key={i} style={i >= 4 ? { background: '#fce4ec' } : {}}>
                  <td className="fw-semibold text-nowrap">{l.stage}</td>
                  <td>{l.features}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Standards &amp; References
        </div>
        <div className="card-body p-2 small">
          <p className="fw-semibold mb-1">Clinical Standards:</p>
          <ul className="ps-3 mb-2">
            {stds.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
          <p className="fw-semibold mb-1">Key References:</p>
          <ol className="ps-3 mb-0">
            {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const tx = data.treatment_summary || [];
  const cis = data.contraindications || [];
  const mon = data.monitoring || [];

  const levelColor = (lv) => ({
    'A': '#198754', 'B': '#0d6efd', 'C': '#fd7e14',
    'INV': '#6f42c1'
  }[lv] || '#6c757d');

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          AED &amp; Symptomatic Treatment Summary
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr><th>Drug</th><th>Level</th><th>Tried %</th><th>Responder %</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {tx.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.drug}</td>
                  <td>
                    <span className="badge" style={{ background: levelColor(t.level) }}>
                      {t.level}
                    </span>
                  </td>
                  <td>{t.tried_pct}%</td>
                  <td>{t.responder_pct}%</td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small text-danger" style={{ background: '#fff3cd' }}>
          Contraindications — ABSOLUTE CI &amp; AVOID
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Drug</th><th>Reason</th></tr></thead>
            <tbody>
              {cis.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger text-nowrap">{c.drug}</td>
                  <td>{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Full Monitoring Schedule
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr></thead>
            <tbody>
              {mon.map((m, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{m.item}</td>
                  <td className="text-nowrap">{m.frequency}</td>
                  <td className="text-muted">{m.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions...</div>;
  const defs = data.definitions || [];
  const refs = data.references || [];
  const stds = data.standards || [];
  const cis = data.contraindications || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Key Concepts — {defs.length} Definitions
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th style={{ width: '28%' }}>Term</th><th>Definition</th></tr></thead>
            <tbody>
              {defs.map((d, i) => (
                <tr key={i}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{d.term}</td>
                  <td>{d.def}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small text-danger" style={{ background: '#fff3cd' }}>
          Contraindications Summary
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead><tr><th>Drug</th><th>Reason</th></tr></thead>
            <tbody>
              {cis.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold text-danger text-nowrap">{c.drug}</td>
                  <td>{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Standards &amp; References
        </div>
        <div className="card-body p-2 small">
          <p className="fw-semibold mb-1">Clinical Standards ({stds.length}):</p>
          <ul className="ps-3 mb-2">
            {stds.map((s, i) => <li key={i}>{s}</li>)}
          </ul>
          <p className="fw-semibold mb-1">References ({refs.length}):</p>
          <ol className="ps-3 mb-0">
            {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}

export default function PANK2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/pank2/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Backend offline — start: bash scripts/restart_backend.sh'));
    fetch(`${API}/api/pank2/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (breakdown) {
      fetch(`${API}/api/pank2/definitions`)
        .then(r => r.json())
        .then(d => {
          setBreakdown(prev => prev ? {
            ...prev,
            definitions: d.definitions || [],
            references: d.references || [],
            standards: d.standards || [],
            contraindications: d.contraindications || [],
          } : prev);
        })
        .catch(() => {});
    }
  }, [breakdown]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${COLOR}`, paddingLeft: 12 }}>
        <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
          PANK2 PKAN — Pantothenate Kinase-Associated Neurodegeneration (NBIA1)
        </h4>
        <div className="text-muted small">
          PANK2 (Pantothenate Kinase 2) · 20p13 · OMIM 606157/234200 · Autosomal Recessive ·
          40-patient cohort (seed-517) · Most common NBIA (~35-50%) ·
          Classic PKAN (&lt;6yr, dystonia, retinopathy, acanthocytes) vs Atypical PKAN (~13yr, speech, OCD) ·
          Eye-of-tiger GP PATHOGNOMONIC · PHT/VGB AVOID · POLG mandatory · GPi-DBS Level B ·
          Deferiprone: no functional benefit (TIRCON) · Fosmetpantotenate (PANK2 bypass) investigational
        </div>
      </div>

      {error && <div className="alert alert-danger small py-2">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <EtiologyTab data={breakdown} />}
      {tab === 2 && <SeizureDystoniaTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={breakdown} />}
    </div>
  );
}
