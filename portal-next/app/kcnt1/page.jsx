'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Breakdown', 'Etiology Detail', 'Treatments', 'Definitions'];
const COLOR = '#b71c1c';   // dark red — malignant migrating severity
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
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
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
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>KCNT1 (9q34.3) — KNa1.1 / SLACK / SLO2.2 · OMIM 608042/614959 · DEE14/MMPSI:</strong>{' '}
        GoF mutations lower Na+ activation threshold → constitutive K+ efflux → paradoxical excitation.{' '}
        <strong>No metabolic biomarker</strong> (plasma amino acids NORMAL — unlike SLC25A22).{' '}
        Migrating ictal EEG pattern is the hallmark of MMPSI.{' '}
        <strong>Quinidine is GoF-SPECIFIC — QTc monitoring mandatory.</strong>{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          ABSOLUTE CI: CBZ/OXC/PHT/LTG worsen MMPSI (Na-channel blockers pro-convulsant in GoF).
          POLG mandatory before VPA. GoF functional confirmation required before quinidine.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="MMPSI" value={`${kpis.mmpsi_pct}%`} color="#dc3545" />
        <KPI label="ADNFLE" value={`${kpis.adnfle_pct}%`} color="#e65100" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Quinidine Tried" value={`${kpis.quinidine_tried_pct}%`} color="#6f42c1" />
        <KPI label="Quinidine Responded" value={`${kpis.quinidine_responded_pct}%`} color="#198754" />
        <KPI label="Migrating EEG" value={`${kpis.migrating_eeg_pct}%`} color={COLOR} />
        <KPI label="Profound ID" value={`${kpis.profound_id_pct}%`} color={COLOR} />
        <KPI label="Any ID" value={`${kpis.any_id_pct}%`} color="#6f42c1" />
        <KPI label="ACTH+VGB" value={`${kpis.acth_vgb_pct}%`} color="#0d6efd" />
        <KPI label="KD Tried" value={`${kpis.kd_pct}%`} color="#198754" />
        <KPI label="Mean AEDs Failed" value={kpis.mean_aeds_failed} color={COLOR} />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
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
              Treatment Lines (7 options)
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="mb-2 pb-1 border-bottom small">
                  <div className="fw-semibold">{t.drug}</div>
                  <div className="text-muted" style={{ fontSize: '0.78rem' }}>{t.level?.substring(0, 90)}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ color: COLOR, background: LIGHT }}>
              Contraindications
            </div>
            <div className="card-body">
              {cis.map((ci, i) => (
                <div key={i} className="mb-1 small fw-semibold" style={{ color: COLOR }}>&#9888; {ci}</div>
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
                  {lifecycle.map((l, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ color: COLOR }}>{l.stage}</td>
                      <td>{l.events}</td>
                      <td>{l.key_action}</td>
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
              Clinical Thresholds
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0 small">
                <thead><tr><th>Metric</th><th>Normal</th><th>Alert</th><th>Critical / Action</th></tr></thead>
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.metric}</td>
                      <td className="text-success">{t.normal}</td>
                      <td className="text-warning">{t.alert_value}</td>
                      <td className="text-danger small">{t.critical_value}</td>
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
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown...</div>;
  const cats = data.by_category || [];
  const summary = data.summary || {};
  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>40-patient cohort (seed 509) across 4 etiologic categories.</strong>{' '}
        Drug-resistant rate: <strong>{summary.drug_resistant_pct}%</strong>.{' '}
        Quinidine tried: <strong>{summary.quinidine_tried_pct}%</strong>{' '}
        of whom responded: <strong>{summary.quinidine_responded_pct}%</strong>.{' '}
        Migrating EEG: <strong>{summary.migrating_eeg_pct}%</strong>.{' '}
        Mean AEDs failed: <strong>{summary.mean_aeds_failed}</strong>.
      </div>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>Category</th><th>N</th><th>MMPSI%</th><th>ADNFLE%</th>
              <th>Ohtahara%</th><th>Burst-Sup%</th><th>Migr EEG%</th><th>Drug-Res%</th>
              <th>Mean AEDs</th><th>Quinidine%</th><th>Responded%</th><th>Profound ID%</th><th>Seizure-Free%</th>
            </tr>
          </thead>
          <tbody>
            {cats.map((c, i) => (
              <tr key={i}>
                <td className="fw-semibold" style={{ color: COLOR }}>{c.category}</td>
                <td>{c.n}</td>
                <td>{c.mmpsi_pct}%</td>
                <td>{c.adnfle_pct}%</td>
                <td>{c.ohtahara_like_pct}%</td>
                <td>{c.burst_suppression_pct}%</td>
                <td>{c.migrating_eeg_pct}%</td>
                <td>{c.drug_resistant_pct}%</td>
                <td>{c.mean_aeds_failed}</td>
                <td>{c.quinidine_tried_pct}%</td>
                <td>{c.quinidine_responded_pct}%</td>
                <td>{c.profound_id_pct}%</td>
                <td>{c.seizure_free_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const details = data.etiology_details || [];
  return (
    <div>
      {details.map((d, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header fw-bold small" style={{ background: LIGHT, color: COLOR }}>
            {d.category}
          </div>
          <div className="card-body small">
            <div className="mb-1"><strong>Typical Variant:</strong> {d.typical_variant}</div>
            <div className="mb-1"><strong>Inheritance:</strong> {d.inheritance}</div>
            <div className="mb-1"><strong>Functional Deficit:</strong> {d.functional_deficit}</div>
            <div className="text-muted">{d.description}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const treatments = data.treatments || [];
  const cis = data.contraindications || [];
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Treatment Lines</h6>
      {treatments.map((t, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>
            {i + 1}. {t.drug}
          </div>
          <div className="card-body small text-muted">{t.level}</div>
        </div>
      ))}
      <h6 className="fw-bold mt-4 mb-3" style={{ color: '#dc3545' }}>Contraindications</h6>
      {cis.map((c, i) => (
        <div key={i} className="card shadow-sm mb-2 border-danger">
          <div className="card-header fw-bold small text-danger">&#9888; {c.drug}</div>
          <div className="card-body small text-muted">{c.reason}</div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const defs = data.definitions || [];
  const ddx = data.key_ddx || [];
  const workup = data.mandatory_workup || [];
  const facts = data.five_key_facts || [];
  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: '#fff3e0' }}>
          5 Key Facts — KCNT1 DEE14/MMPSI
        </div>
        <div className="card-body">
          <ol className="small mb-0">
            {facts.map((f, i) => <li key={i} className="mb-2">{f}</li>)}
          </ol>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Mandatory Workup
        </div>
        <div className="card-body">
          <ul className="small mb-0">
            {workup.map((w, i) => <li key={i} className="mb-1">{w}</li>)}
          </ul>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-bold small" style={{ background: '#e8f5e9' }}>
          Key DDx
        </div>
        <div className="card-body">
          <ul className="small mb-0">
            {ddx.map((d, i) => <li key={i} className="mb-1">{d}</li>)}
          </ul>
        </div>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Glossary (15 terms)</h6>
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

export default function KcnT1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/kcnt1/overview`).then(r => r.json()),
      fetch(`${API}/api/kcnt1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/kcnt1/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setErr(e.message));
  }, []);

  const tabContent = () => {
    if (tab === 0) return <OverviewTab data={overview} />;
    if (tab === 1) return <BreakdownTab data={breakdown} />;
    if (tab === 2) return <EtiologyTab data={breakdown} />;
    if (tab === 3) return <TreatmentsTab data={breakdown} />;
    if (tab === 4) return <DefinitionsTab data={definitions} />;
    return null;
  };

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          &#x1f9ec; KCNT1 — DEE14 / MMPSI / ADNFLE2
        </h4>
        <div className="text-muted small">
          KNa1.1 / SLACK / SLO2.2 · 1237 aa · S1-S6 + RCK1 + RCK2 · Na+-Activated K+ Channel ·
          Chr 9q34.3 · OMIM 608042/614959/615005 · GoF de novo + AD familial + AR biallelic · 40-patient cohort (seed 509)
        </div>
        <div className="text-muted small mt-1">
          <strong>Hallmark:</strong> Migrating ictal EEG pattern (MMPSI) · No metabolic biomarker (amino acids NORMAL) ·
          Quinidine GoF-specific (QTc mandatory) · CBZ/OXC/PHT/LTG ABSOLUTE CI · POLG mandatory before VPA
        </div>
      </div>

      {err && <div className="alert alert-danger small">Error: {err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tabContent()}
    </div>
  );
}
