'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Breakdown', 'Etiology Detail', 'Treatments', 'Definitions'];
const COLOR = '#311b92';   // deep indigo — X-linked severe NHE6/Christianson
const LIGHT = '#ede7f6';

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
        <strong>SLC9A6 (Xq26.3) — NHE6 Na+/H+ Exchanger 6 · OMIM 300243 · MRXSCH Christianson Syndrome:</strong>{' '}
        NHE6 LOF → hyperacidic endosomes → TrkB/TrkC recycling failure → BDNF/NT-3 signalling defect.{' '}
        <strong>Angelman-like phenotype (males only)</strong> — profound ID + absent speech + epilepsy + cerebellar ataxia.{' '}
        Key DDx: <strong>progressive cerebellar atrophy on MRI</strong> (absent in Angelman) + normal UBE3A methylation.{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          No precision therapy (2026). POLG mandatory before VPA. VGB REMS mandatory. XCI-HUMARA for symptomatic carrier females.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Infantile Spasms" value={`${kpis.infantile_spasms_pct}%`} color="#dc3545" />
        <KPI label="Myoclonic" value={`${kpis.myoclonic_pct}%`} color="#e65100" />
        <KPI label="Absence" value={`${kpis.absence_pct}%`} color="#6f42c1" />
        <KPI label="Focal Seizures" value={`${kpis.focal_pct}%`} color="#6f42c1" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Cerebellar Atrophy" value={`${kpis.cerebellar_atrophy_pct}%`} color={COLOR} />
        <KPI label="Progressive Ataxia" value={`${kpis.progressive_ataxia_pct}%`} color={COLOR} />
        <KPI label="Absent Speech" value={`${kpis.absent_speech_pct}%`} color="#dc3545" />
        <KPI label="Profound ID" value={`${kpis.profound_id_pct}%`} color={COLOR} />
        <KPI label="Any ID" value={`${kpis.any_id_pct}%`} color="#6f42c1" />
        <KPI label="Happy Affect" value={`${kpis.happy_affect_pct}%`} color="#198754" />
        <KPI label="Hyperkinesia" value={`${kpis.hyperkinesia_pct}%`} color="#6f42c1" />
        <KPI label="Microcephaly" value={`${kpis.microcephaly_pct}%`} color={COLOR} />
        <KPI label="ACTH Given" value={`${kpis.acth_given_pct}%`} color="#0d6efd" />
        <KPI label="KD Tried" value={`${kpis.kd_tried_pct}%`} color="#198754" />
        <KPI label="Angelman Misdx" value={`${kpis.angelman_misdiagnosed_pct}%`} color="#dc3545" />
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
              Treatment Lines
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
        <strong>40-patient cohort (seed 511) across 4 etiologic categories.</strong>{' '}
        Drug-resistant: <strong>{summary.drug_resistant_pct}%</strong>.{' '}
        Cerebellar atrophy: <strong>{summary.cerebellar_atrophy_pct}%</strong>.{' '}
        Progressive ataxia: <strong>{summary.progressive_ataxia_pct}%</strong>.{' '}
        Absent speech: <strong>{summary.absent_speech_pct}%</strong>.{' '}
        Angelman misdiagnosed: <strong>{summary.angelman_misdiagnosed_pct}%</strong>.{' '}
        Mean AEDs failed: <strong>{summary.mean_aeds_failed}</strong>.
      </div>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>Category</th><th>N</th><th>IS%</th><th>Myoclonic%</th>
              <th>Drug-Res%</th><th>Cerebellar%</th><th>Ataxia%</th><th>Absent Speech%</th>
              <th>Profound ID%</th><th>Happy Affect%</th><th>Hyperkin%</th>
              <th>Angelman Misdx%</th><th>Mean AEDs</th><th>Seizure-Free%</th>
            </tr>
          </thead>
          <tbody>
            {cats.map((c, i) => (
              <tr key={i}>
                <td className="fw-semibold" style={{ color: COLOR }}>{c.category}</td>
                <td>{c.n}</td>
                <td>{c.infantile_spasms_pct}%</td>
                <td>{c.myoclonic_pct}%</td>
                <td>{c.drug_resistant_pct}%</td>
                <td>{c.cerebellar_atrophy_pct}%</td>
                <td>{c.progressive_ataxia_pct}%</td>
                <td>{c.absent_speech_pct}%</td>
                <td>{c.profound_id_pct}%</td>
                <td>{c.happy_affect_pct}%</td>
                <td>{c.hyperkinesia_pct}%</td>
                <td>{c.angelman_misdiagnosed_pct}%</td>
                <td>{c.mean_aeds_failed}</td>
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
  const standards = data.standards || [];
  const facts = data.five_key_facts || [];
  return (
    <div>
      <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Gene Summary
        </div>
        <div className="card-body small">
          <div><strong>Gene:</strong> {data.gene} ({data.chromosome}) — {data.protein}</div>
          <div><strong>OMIM Gene:</strong> #{data.omim_gene} &nbsp;|&nbsp; <strong>OMIM Disease:</strong> #{data.omim_disease}</div>
          <div><strong>Disease:</strong> {data.disease_name}</div>
          <div><strong>Inheritance:</strong> {data.inheritance}</div>
        </div>
      </div>

      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>5 Key Facts</h6>
      {facts.map((f, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>Fact {i + 1}</div>
          <div className="card-body small text-muted">{f}</div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-2" style={{ color: COLOR }}>Key DDx</h6>
      {ddx.map((d, i) => (
        <div key={i} className="alert py-2 small mb-2" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
          {d}
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-2" style={{ color: COLOR }}>Mandatory Workup</h6>
      <ul className="list-group list-group-flush mb-4">
        {workup.map((w, i) => (
          <li key={i} className="list-group-item small">{w}</li>
        ))}
      </ul>

      <h6 className="fw-bold mb-2" style={{ color: COLOR }}>Definitions ({defs.length})</h6>
      {defs.map((d, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>{d.term}</div>
          <div className="card-body small text-muted">{d.definition}</div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-2" style={{ color: COLOR }}>Standards</h6>
      <ul className="list-group list-group-flush">
        {standards.map((s, i) => (
          <li key={i} className="list-group-item small">{s}</li>
        ))}
      </ul>
    </div>
  );
}

export default function SLC9A6Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/slc9a6/overview`).then(r => r.json()),
          fetch(`${API}/api/slc9a6/breakdown`).then(r => r.json()),
          fetch(`${API}/api/slc9a6/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      } catch (e) {
        setError(e.message);
      }
    };
    fetchAll();
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-2">
        <span style={{ fontSize: '2rem' }}>🧬</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            SLC9A6 MRXSCH — Christianson Syndrome
          </h4>
          <div className="text-muted small">
            NHE6 Na+/H+ Exchanger 6 · 701 aa · Xq26.3 · OMIM 300243 · X-linked · 40-patient cohort seed-511
          </div>
        </div>
      </div>

      {error && (
        <div className="alert alert-danger small">API error: {error}</div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${activeTab === i ? ' active fw-bold' : ''}`}
              style={activeTab === i ? { color: COLOR } : {}}
              onClick={() => setActiveTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <BreakdownTab data={breakdown} />}
      {activeTab === 2 && <EtiologyTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
