'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Breakdown', 'Etiology Detail', 'Treatments', 'Definitions'];
const COLOR = '#b71c1c';   // deep red — PLCB1 Ohtahara severity
const LIGHT = '#ffebee';

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
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const kpis = data.kpis || {};
  const etiologies = data.etiology_distribution || [];
  const treatments = data.treatments_summary || [];
  const monitoring = data.monitoring_summary || [];
  const lifecycle = data.lifecycle || [];
  const thresholds = data.thresholds || [];
  const cis = data.contraindications_summary || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-danger py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
        <strong>PLCB1 (20p12.3) — Phospholipase C beta-1 · OMIM 607120/614563 · DEE12/EIEE12:</strong>{' '}
        PLCβ1 generates IP3 + DAG from PIP2 downstream of Gαq/11 (mGluR1/5, M1-AChR).{' '}
        LOF → <strong>mGluR-LTD failure + PKC-NMDAR gating loss</strong> → cortical excitatory runaway.{' '}
        Biallelic null → <strong>Ohtahara syndrome (burst-suppression, neonatal)</strong>;{' '}
        hypomorphic/de-novo → <strong>West syndrome (infantile spasms, hypsarrhythmia)</strong>;{' '}
        somatic mosaic → <strong>FCD IIb (surgically curable)</strong>.{' '}
        <span className="text-danger fw-bold">
          ABSOLUTE CI: PHT/CBZ/OXC worsen burst-suppression.
          POLG mandatory before VPA. ACTH+VGB (UKISS) Level A for IS.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Ohtahara" value={`${kpis.ohtahara_pct}%`} color="#dc3545" />
        <KPI label="West Syndrome" value={`${kpis.west_syndrome_pct}%`} color="#e65100" />
        <KPI label="Burst-Suppression" value={`${kpis.burst_suppression_pct}%`} color="#dc3545" />
        <KPI label="Hypsarrhythmia" value={`${kpis.hypsarrhythmia_pct}%`} color="#fd7e14" />
        <KPI label="EEG Abnormal" value={`${kpis.eeg_abnormal_pct}%`} color="#6f42c1" />
        <KPI label="Profound ID" value={`${kpis.profound_id_pct}%`} color={COLOR} />
        <KPI label="Any ID" value={`${kpis.any_id_pct}%`} color="#6f42c1" />
        <KPI label="ACTH+VGB" value={`${kpis.acth_vgb_pct}%`} color="#0d6efd" />
        <KPI label="KD Tried" value={`${kpis.kd_tried_pct}%`} color="#198754" />
        <KPI label="MRI Done" value={`${kpis.mri_done_pct}%`} color="#0d6efd" />
        <KPI label="FCD on MRI" value={`${kpis.fcd_on_mri_pct}%`} color="#fd7e14" />
        <KPI label="Surgery Done" value={`${kpis.surgery_done_pct}%`} color="#198754" />
        <KPI label="Seizure-Free Post-Surg" value={`${kpis.seizure_free_post_surg_pct}%`} color="#198754" />
        <KPI label="Year-1 Mortality" value={`${kpis.yr1_mortality_pct}%`} color="#dc3545" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#6c757d" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Etiology Distribution (5 Categories)
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
              Treatment Lines (8 options)
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="mb-2 pb-1 border-bottom small">
                  <div className="fw-semibold">{t.drug}</div>
                  <div className="text-muted" style={{ fontSize: '0.78rem' }}>{t.level?.split('—')[0]?.trim()}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small text-danger" style={{ background: LIGHT }}>
              Contraindications
            </div>
            <div className="card-body">
              {cis.map((ci, i) => (
                <div key={i} className="mb-1 small text-danger">⚠ {ci}</div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Monitoring Schedule
            </div>
            <div className="card-body">
              {monitoring.map((m, i) => (
                <div key={i} className="mb-2 small">
                  <span className="badge me-1 text-bg-secondary">{m.timepoint}</span>
                  {m.action}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Lifecycle
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0 small">
                <thead><tr><th>Stage</th><th>Events</th><th>Key Action</th></tr></thead>
                <tbody>
                  {lifecycle.map((lc, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ whiteSpace: 'nowrap' }}>{lc.stage}</td>
                      <td>{lc.events}</td>
                      <td>{lc.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Key Thresholds
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0 small">
                <thead><tr><th>Parameter</th><th>Threshold</th><th>Rationale</th></tr></thead>
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.parameter}</td>
                      <td>{t.threshold}</td>
                      <td>{t.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function BreakdownTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const bycat = data.by_category || [];
  const summary = data.summary || {};

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Cohort Summary — {data.cohort_size} patients (seed {data.cohort_seed})
        </div>
        <div className="card-body">
          <div className="row g-2">
            {Object.entries(summary).map(([k, v]) => (
              <div key={k} className="col-6 col-md-4 col-lg-3 mb-2">
                <div className="border rounded p-2 text-center small">
                  <div className="fw-bold fs-6" style={{ color: COLOR }}>{v}%</div>
                  <div className="text-muted" style={{ fontSize: '0.75rem' }}>
                    {k.replace(/_pct$/, '').replace(/_/g, ' ')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Per-Etiology Breakdown
        </div>
        <div className="card-body p-0">
          <table className="table table-sm table-bordered mb-0 small">
            <thead className="table-light">
              <tr>
                <th>Category</th><th>n</th><th>Ohtahara%</th><th>West%</th>
                <th>Burst-Supp%</th><th>Profound ID%</th>
                <th>ACTH+VGB%</th><th>KD%</th><th>Surgery%</th><th>Yr1 Mort%</th>
              </tr>
            </thead>
            <tbody>
              {bycat.map((b, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{b.category}</td>
                  <td>{b.n}</td>
                  <td>{b.ohtahara_pct}</td>
                  <td>{b.west_pct}</td>
                  <td>{b.burst_suppression_pct}</td>
                  <td>{b.profound_id_pct}</td>
                  <td>{b.acth_vgb_pct}</td>
                  <td>{b.kd_pct}</td>
                  <td>{b.surgery_pct}</td>
                  <td>{b.yr1_mortality_pct}</td>
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
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const details = data.etiology_details || [];
  const thresholds = data.thresholds || [];
  return (
    <div>
      {details.map((e, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>
            {e.category}
          </div>
          <div className="card-body small">
            <div className="mb-1"><span className="fw-semibold">Typical variant:</span> {e.typical_variant}</div>
            <div className="mb-1"><span className="fw-semibold">Inheritance:</span> {e.inheritance}</div>
            <div className="mb-1"><span className="fw-semibold">Functional deficit:</span> {e.functional_deficit}</div>
            <div className="text-muted">{e.description}</div>
          </div>
        </div>
      ))}
      <div className="card shadow-sm mt-2">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>All Thresholds</div>
        <div className="card-body p-0">
          <table className="table table-sm table-striped mb-0 small">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Rationale</th></tr></thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{t.parameter}</td>
                  <td>{t.threshold}</td>
                  <td>{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];
  return (
    <div>
      <h6 className="fw-bold mb-3">Treatments ({treatments.length})</h6>
      {treatments.map((t, i) => (
        <div key={i} className={`card shadow-sm mb-3 ${t.contraindication_flag ? 'border-warning' : ''}`}>
          <div className="card-header fw-bold small"
            style={{ background: t.contraindication_flag ? '#fff3cd' : LIGHT }}>
            {t.drug}
            {t.contraindication_flag && <span className="ms-2 badge bg-warning text-dark">POLG CHECK</span>}
          </div>
          <div className="card-body small">
            <div className="mb-1"><span className="fw-semibold">Level:</span> {t.level}</div>
            <div className="mb-1"><span className="fw-semibold">Dose:</span> {t.dose}</div>
            <div className="mb-1"><span className="fw-semibold">Mechanism:</span> {t.mechanism}</div>
            {t.note && <div className="text-muted fst-italic">{t.note}</div>}
          </div>
        </div>
      ))}
      <h6 className="fw-bold mt-4 mb-3 text-danger">Contraindications ({cis.length})</h6>
      {cis.map((c, i) => (
        <div key={i} className="card shadow-sm mb-3 border-danger">
          <div className="card-header fw-bold small text-danger" style={{ background: '#fff5f5' }}>
            ⚠ {c.drug}
          </div>
          <div className="card-body small">
            <div className="mb-1"><span className="fw-semibold">Reason:</span> {c.reason}</div>
            <div className="mb-1"><span className="fw-semibold text-danger">Risk:</span> {c.risk}</div>
            <div><span className="fw-semibold">Level:</span> {c.level}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const defs = data.definitions || [];
  const ddx = data.key_ddx || [];
  const workup = data.mandatory_workup || [];
  const facts = data.five_key_facts || [];
  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small text-danger" style={{ background: LIGHT }}>
          5 Key Facts — PLCB1 DEE12
        </div>
        <ul className="list-group list-group-flush small">
          {facts.map((f, i) => (
            <li key={i} className="list-group-item">{f}</li>
          ))}
        </ul>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          DDx — Neonatal DEE / Ohtahara Mimics
        </div>
        <ul className="list-group list-group-flush small">
          {ddx.map((d, i) => (
            <li key={i} className="list-group-item">{d}</li>
          ))}
        </ul>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Mandatory Workup
        </div>
        <ul className="list-group list-group-flush small">
          {workup.map((w, i) => (
            <li key={i} className="list-group-item">{w}</li>
          ))}
        </ul>
      </div>

      <h6 className="fw-bold mb-3">Core Definitions ({defs.length})</h6>
      {defs.map((d, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>
            {d.term}
          </div>
          <div className="card-body small text-muted">{d.definition}</div>
        </div>
      ))}
    </div>
  );
}

export default function PLCB1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/plcb1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
    fetch(`${API}/api/plcb1/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/plcb1/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div
          className="rounded-circle d-flex align-items-center justify-content-center text-white fw-bold"
          style={{ width: 52, height: 52, background: COLOR, fontSize: 18 }}
        >
          P1
        </div>
        <div>
          <h4 className="mb-0 fw-bold">PLCB1 — DEE12 / EIEE12</h4>
          <div className="text-muted small">
            Phospholipase C beta-1 · 20p12.3 · OMIM 607120/614563 ·
            Gαq/IP3/DAG Pathway · Ohtahara→West→FCD · AR + de novo + Somatic Mosaic
          </div>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger py-2 small">
          API error: {error}
        </div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              onClick={() => setTab(i)}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <BreakdownTab data={breakdown} />}
      {tab === 2 && <EtiologyTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
