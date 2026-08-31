'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure & Breathing', 'Treatments', 'Definitions'];
const COLOR = '#7b2d8b';   // purple — TCF4 bHLH transcription factor
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
        <strong>TCF4 (18q21.2) — bHLH E-protein 667 aa · OMIM 602272/610954 · Pitt-Hopkins Syndrome:</strong>{' '}
        De novo AD TCF4 haploinsufficiency or dominant-negative (bHLH missense R576W/A597T/L600P).{' '}
        <strong className="text-danger">Episodic hyperventilation + apnoea — PATHOGNOMONIC (absent in ALL other Angelman-like syndromes).</strong>{' '}
        Beaked nasal bridge + Cupid-bow lip + deep-set eyes + wide mouth + widely-spaced teeth.{' '}
        <span className="fw-bold" style={{ color: COLOR }}>
          CBZ/OXC HIGH CAUTION. LEV/VPA/CLB Level B. POLG mandatory before VPA. Acetazolamide (Level C) for breathing episodes.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Breathing Episodes" value={`${kpis.breathing_episodes_pct}%`} color="#dc3545" />
        <KPI label="Focal Seizures" value={`${kpis.focal_pct}%`} color="#e65100" />
        <KPI label="Tonic-Clonic" value={`${kpis.tonic_clonic_pct}%`} color="#6f42c1" />
        <KPI label="Myoclonic" value={`${kpis.myoclonic_pct}%`} color="#6f42c1" />
        <KPI label="Absence" value={`${kpis.absence_pct}%`} color="#6f42c1" />
        <KPI label="Infantile Spasms" value={`${kpis.infantile_spasms_pct}%`} color="#dc3545" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="Absent Speech" value={`${kpis.absent_speech_pct}%`} color="#dc3545" />
        <KPI label="Profound ID" value={`${kpis.profound_id_pct}%`} color={COLOR} />
        <KPI label="Corpus Callosum ✗" value={`${kpis.corpus_callosum_absent_pct}%`} color={COLOR} />
        <KPI label="Beaked Nose" value={`${kpis.beaked_nose_pct}%`} color={COLOR} />
        <KPI label="Happy Affect" value={`${kpis.happy_affect_pct}%`} color="#198754" />
        <KPI label="Hand Stereotypies" value={`${kpis.hand_stereotypies_pct}%`} color="#6f42c1" />
        <KPI label="KD Tried" value={`${kpis.kd_tried_pct}%`} color="#198754" />
        <KPI label="POLG Tested" value={`${kpis.polg_tested_pct}%`} color="#0d6efd" />
        <KPI label="Acetazolamide" value={`${kpis.acetazolamide_pct}%`} color="#0d6efd" />
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
              Treatment Tier Summary
            </div>
            <div className="card-body small">
              <div className="mb-1"><span className="badge bg-success me-1">Level A</span>{tier.level_a}</div>
              <div className="mb-1"><span className="badge bg-primary me-1">Level B</span>{tier.level_b}</div>
              <div className="mb-1"><span className="badge bg-secondary me-1">Level C</span>{tier.level_c}</div>
              <div className="mb-1"><span className="badge bg-warning text-dark me-1">High Caution</span>{tier.high_caution}</div>
              <div className="mb-1"><span className="badge bg-danger me-1">Avoid</span>{tier.absolute_avoid}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ color: '#dc3545', background: LIGHT }}>
              ⚠ Contraindications
            </div>
            <div className="card-body">
              {cis.map((ci, i) => (
                <div key={i} className="mb-1 small fw-semibold text-danger">&#9888; {ci}</div>
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
            <div className="card-header fw-bold small" style={{ color: '#dc3545', background: LIGHT }}>
              DDx Table — Key Distinguishing Features from Angelman-Like Syndromes
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0 small">
                <thead><tr><th>Syndrome</th><th>Key DDx vs Pitt-Hopkins</th></tr></thead>
                <tbody>
                  {ddx.map((d, i) => (
                    <tr key={i}>
                      <td className="fw-semibold" style={{ color: COLOR }}>{d.syndrome}</td>
                      <td>{d.key_ddx}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
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

      <div className="row g-3 mb-4">
        <div className="col-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Highlights
            </div>
            <div className="card-body">
              <ul className="list-group list-group-flush">
                {highlights.map((h, i) => (
                  <li key={i} className="list-group-item small">{h}</li>
                ))}
              </ul>
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

function PatientsEtiologyTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const cats = data.by_category || [];
  const summary = data.summary || {};
  const details = data.etiology_details || [];
  return (
    <div>
      <div className="alert py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}`, background: LIGHT }}>
        <strong>40-patient cohort (seed 513) across 4 etiologic categories.</strong>{' '}
        Breathing episodes: <strong>{summary.breathing_episodes_pct}%</strong>.{' '}
        Drug-resistant: <strong>{summary.drug_resistant_pct}%</strong>.{' '}
        Absent speech: <strong>{summary.absent_speech_pct}%</strong>.{' '}
        Corpus callosum absent: <strong>{summary.corpus_callosum_absent_pct}%</strong>.{' '}
        Beaked nose: <strong>{summary.beaked_nose_pct}%</strong>.{' '}
        Mean AEDs failed: <strong>{summary.mean_aeds_failed}</strong>.
      </div>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>Category</th><th>N</th><th>Breathing%</th><th>Focal%</th>
              <th>TC%</th><th>Myocl%</th><th>IS%</th><th>Drug-Res%</th>
              <th>Absent Speech%</th><th>Profound ID%</th><th>CC Absent%</th>
              <th>Beaked Nose%</th><th>Happy%</th><th>Mean AEDs</th><th>Sz-Free%</th>
            </tr>
          </thead>
          <tbody>
            {cats.map((c, i) => (
              <tr key={i}>
                <td className="fw-semibold" style={{ color: COLOR }}>{c.category}</td>
                <td>{c.n}</td>
                <td className="text-danger fw-bold">{c.breathing_episodes_pct}%</td>
                <td>{c.focal_pct}%</td>
                <td>{c.tonic_clonic_pct}%</td>
                <td>{c.myoclonic_pct}%</td>
                <td>{c.infantile_spasms_pct}%</td>
                <td>{c.drug_resistant_pct}%</td>
                <td>{c.absent_speech_pct}%</td>
                <td>{c.profound_id_pct}%</td>
                <td>{c.corpus_callosum_absent_pct}%</td>
                <td>{c.beaked_nose_pct}%</td>
                <td>{c.happy_affect_pct}%</td>
                <td>{c.mean_aeds_failed}</td>
                <td>{c.seizure_free_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Etiology Detail</h6>
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

function SeizureBreathingTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const summary = data.summary || {};
  const thresholds = data.thresholds || [];
  const monitoring = data.monitoring || [];
  return (
    <div>
      <div className="alert alert-danger py-2 small mb-3" style={{ borderLeft: `6px solid #dc3545` }}>
        <strong>PATHOGNOMONIC: Episodic Hyperventilation + Apnoea</strong><br />
        Breathing episodes are PRESENT in ~{summary.breathing_episodes_pct}% of this cohort. They are central (not obstructive),
        triggered by excitement/emotional arousal (NOT fever), EEG is NORMAL during episodes (not ictal).
        Video-EEG confirmation is MANDATORY before treating as seizures.
        Acetazolamide (Level C) is the specific treatment for severe/frequent episodes.
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small text-danger" style={{ background: '#fff3f3' }}>
              Breathing Episode Profile (Cohort-Wide)
            </div>
            <div className="card-body">
              <Bar label="Breathing Episodes" value={summary.breathing_episodes_pct || 0} max={100} color="#dc3545" />
              <Bar label="Focal Seizures" value={summary.focal_pct || 0} max={100} />
              <Bar label="Tonic-Clonic" value={summary.tonic_clonic_pct || 0} max={100} />
              <Bar label="Myoclonic" value={summary.myoclonic_pct || 0} max={100} />
              <Bar label="Absence" value={summary.absence_pct || 0} max={100} />
              <Bar label="Infantile Spasms" value={summary.infantile_spasms_pct || 0} max={100} color="#dc3545" />
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Drug Resistance & Treatment Metrics
            </div>
            <div className="card-body">
              <Bar label="Drug-Resistant" value={summary.drug_resistant_pct || 0} max={100} color="#dc3545" />
              <Bar label="Absent Speech" value={summary.absent_speech_pct || 0} max={100} />
              <Bar label="Profound ID" value={summary.profound_id_pct || 0} max={100} />
              <Bar label="Corpus Callosum Absent" value={summary.corpus_callosum_absent_pct || 0} max={100} />
              <Bar label="KD Tried" value={summary.kd_tried_pct || 0} max={100} color="#198754" />
              <Bar label="Acetazolamide Used" value={summary.acetazolamide_pct || 0} max={100} color="#0d6efd" />
              <Bar label="Seizure-Free" value={summary.seizure_free_pct || 0} max={100} color="#198754" />
            </div>
          </div>
        </div>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Thresholds</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-bordered small">
          <thead className="table-dark">
            <tr><th>Metric</th><th>Normal</th><th>Alert Value</th><th>Critical / Action</th></tr>
          </thead>
          <tbody>
            {thresholds.map((t, i) => (
              <tr key={i}>
                <td className="fw-semibold" style={{ color: COLOR }}>{t.metric}</td>
                <td className="text-success small">{t.normal}</td>
                <td className="text-warning small">{t.alert_value}</td>
                <td className="text-danger small">{t.critical_value}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Monitoring Schedule</h6>
      {monitoring.map((m, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header fw-bold small" style={{ background: LIGHT }}>
            {m.timepoint}
          </div>
          <div className="card-body small text-muted">{m.action}</div>
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
  const workup = data.mandatory_steps || [];
  const standards = data.standards || [];
  const facts = data.key_facts || [];
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

      <h6 className="fw-bold mt-4 mb-2" style={{ color: COLOR }}>Mandatory Workup Steps</h6>
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

export default function TCF4Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/tcf4/overview`).then(r => r.json()),
          fetch(`${API}/api/tcf4/breakdown`).then(r => r.json()),
          fetch(`${API}/api/tcf4/definitions`).then(r => r.json()),
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
        <span style={{ fontSize: '2rem' }}>&#x1f9ec;</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            TCF4 — Pitt-Hopkins Syndrome
          </h4>
          <div className="text-muted small">
            bHLH E-protein Transcription Factor 4 · 667 aa · 18q21.2 · OMIM 602272/610954 · De novo AD · 40-patient cohort seed-513
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
      {activeTab === 1 && <PatientsEtiologyTab data={breakdown} />}
      {activeTab === 2 && <SeizureBreathingTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
