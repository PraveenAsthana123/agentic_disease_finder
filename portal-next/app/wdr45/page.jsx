'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure & Phase 2', 'Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — NBIA/neurodegeneration
const LIGHT = '#f3e5f5';

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
        <strong>WDR45 (Xp11.23) — 7-bladed beta-propeller autophagy scaffold · OMIM 300526/300643 · BPAN/NBIA5:</strong>{' '}
        De novo X-linked dominant (90% females). Biphasic:{' '}
        <strong>Phase 1</strong> = childhood static encephalopathy (ID + epilepsy) →{' '}
        <strong className="text-danger">Phase 2</strong> = SUDDEN adolescent-onset rapidly progressive parkinsonism + dementia.{' '}
        <strong className="text-danger">MRI SWI: SN + GP iron hypointensity + T1 halo sign = PATHOGNOMONIC.</strong>{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          PHT/CBZ/OXC AVOID. LEV/VPA/CLB Level B. POLG mandatory before VPA.
          Deferiprone investigational (NBIA Research Institute trials).
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Female (X-dom)" value={kpis.n_female} color={COLOR} />
        <KPI label="Male (mosaic)" value={kpis.n_male} color="#6f42c1" />
        <KPI label="Focal Seizures" value={`${kpis.focal_pct}%`} color="#e65100" />
        <KPI label="Absence" value={`${kpis.absence_pct}%`} color="#6f42c1" />
        <KPI label="GTCS" value={`${kpis.gtcs_pct}%`} color="#6f42c1" />
        <KPI label="Myoclonic" value={`${kpis.myoclonic_pct}%`} color="#6f42c1" />
        <KPI label="Infantile Spasms" value={`${kpis.infantile_spasms_pct}%`} color="#dc3545" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Absent Speech" value={`${kpis.absent_speech_pct}%`} color="#dc3545" />
        <KPI label="Profound ID" value={`${kpis.profound_id_pct}%`} color={COLOR} />
        <KPI label="CC Thin/Absent" value={`${kpis.corpus_callosum_thin_pct}%`} color={COLOR} />
        <KPI label="SN Iron MRI" value={`${kpis.sn_iron_pct}%`} color="#dc3545" />
        <KPI label="T1 Halo Sign" value={`${kpis.halo_sign_pct}%`} color="#dc3545" />
        <KPI label="Autism Features" value={`${kpis.autism_pct}%`} color="#6f42c1" />
        <KPI label="Phase 2 (Neuro)" value={`${kpis.phase2_pct}%`} color="#dc3545" />
        <KPI label="Parkinsonism" value={`${kpis.parkinsonism_pct}%`} color="#dc3545" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#0d6efd" />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="Mean Onset (mo)" value={kpis.mean_onset_months} color={COLOR} />
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
              Biphasic Lifecycle (Phase 1 → Phase 2)
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
              DDx — BPAN vs Other NBIA + Angelman-Like Syndromes
            </div>
            <div className="card-body p-0">
              <table className="table table-sm mb-0 small">
                <thead>
                  <tr><th>Condition</th><th>Distinguishing from BPAN</th><th>Shared</th></tr>
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
  const sz = data.seizure_breakdown || [];
  const pts = data.per_patient || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Etiology Breakdown — 4 Categories (n=40, seed-515)
        </div>
        <div className="card-body p-0">
          <table className="table table-sm mb-0 small">
            <thead>
              <tr>
                <th>Etiology</th><th>n</th><th>%</th>
                <th>Drug-Resistant</th><th>Absent Speech</th>
                <th>Phase 2</th><th>SN Iron</th>
                <th>Mean Onset (mo)</th><th>Mean AEDs</th>
              </tr>
            </thead>
            <tbody>
              {etio.map((e, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{e.etiology}</td>
                  <td>{e.n}</td>
                  <td>{e.pct}%</td>
                  <td>{e.drug_resistant_pct}%</td>
                  <td>{e.absent_speech_pct}%</td>
                  <td>{e.phase2_pct}%</td>
                  <td>{e.sn_iron_pct}%</td>
                  <td>{e.mean_onset_mo}</td>
                  <td>{e.mean_aeds}</td>
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
                  <th>ID</th><th>Sex</th><th>Etiology</th><th>Onset(mo)</th>
                  <th>Seizure Types</th><th>DR</th><th>AEDs</th>
                  <th>Speech</th><th>ID Level</th><th>CC Thin</th>
                  <th>SN Iron</th><th>Halo</th><th>Ph2</th>
                  <th>Park.</th><th>Autism</th><th>POLG</th><th>SF</th>
                </tr>
              </thead>
              <tbody>
                {pts.map((p, i) => (
                  <tr key={i} style={p.sn_iron ? { background: '#fce4ec' } : {}}>
                    <td className="fw-semibold">{p.id}</td>
                    <td>{p.sex}</td>
                    <td>{p.etiology}</td>
                    <td>{p.onset_mo}</td>
                    <td>{p.seizure_types}</td>
                    <td>{p.drug_resistant ? '🔴' : '🟢'}</td>
                    <td>{p.n_aeds}</td>
                    <td>{p.absent_speech ? 'Absent' : 'Partial'}</td>
                    <td>{p.id_level}</td>
                    <td>{p.corpus_callosum_thin ? '✓' : '–'}</td>
                    <td>{p.sn_iron ? '⚠️' : '–'}</td>
                    <td>{p.halo_sign ? '⚠️' : '–'}</td>
                    <td>{p.phase2 ? '⚠️' : '–'}</td>
                    <td>{p.parkinsonism ? '⚠️' : '–'}</td>
                    <td>{p.autism ? '✓' : '–'}</td>
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

function SeizureTab({ data }) {
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
          Seizure Type Breakdown — BPAN Phase 1 (n=40)
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
        <div className="card-header fw-bold small" style={{ background: '#fce4ec' }}>
          Phase 2 Neurodegeneration — Features &amp; Timeline
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
          AED &amp; Treatment Summary
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

export default function WDR45Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${API}/api/wdr45/overview`)
      .then(r => r.json()).then(setOverview)
      .catch(() => setError('Backend offline — start: bash scripts/restart_backend.sh'));
    fetch(`${API}/api/wdr45/breakdown`)
      .then(r => r.json()).then(setBreakdown)
      .catch(() => {});
  }, []);

  const definitions = breakdown
    ? {
        definitions: breakdown.definitions || [],
        references: breakdown.references || [],
        standards: breakdown.standards || [],
        contraindications: breakdown.contraindications || [],
      }
    : null;

  useEffect(() => {
    if (breakdown) {
      fetch(`${API}/api/wdr45/definitions`)
        .then(r => r.json())
        .then(d => {
          setBreakdown(prev => prev ? { ...prev, definitions: d.definitions || [], references: d.references || [], standards: d.standards || [], contraindications: d.contraindications || [] } : prev);
        })
        .catch(() => {});
    }
  }, [breakdown]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderLeft: `6px solid ${COLOR}`, paddingLeft: 12 }}>
        <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
          WDR45 BPAN — Beta-propeller protein-associated neurodegeneration (NBIA5)
        </h4>
        <div className="text-muted small">
          WDR45 (WIPI4) · Xp11.23 · OMIM 300526/300643 · X-linked Dominant De Novo ·
          40-patient cohort (seed-515) · Phase 1: static encephalopathy + epilepsy ·
          Phase 2: SUDDEN parkinsonism + dementia · MRI SWI: SN+GP iron PATHOGNOMONIC ·
          PHT/CBZ/OXC AVOID · POLG mandatory · Deferiprone investigational
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
      {tab === 2 && <SeizureTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={breakdown} />}
    </div>
  );
}
