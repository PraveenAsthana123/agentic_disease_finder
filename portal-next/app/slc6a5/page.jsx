'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Events', 'Event Types & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#2e7d32';    // forest green — distinct from GLRA1 teal (#00695c) and GLRB (#00515a)
const LIGHT = '#e8f5e9';

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
      <div className="alert alert-success py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
        <strong>SLC6A5 (11p15.1) — GlyT2 / Glycine Transporter 2 · Hyperekplexia Type 3 (HYPER3):</strong>{' '}
        GlyT2 is the <strong>presynaptic glycine reuptake transporter</strong> — the sole mechanism for replenishing
        synaptic vesicle glycine stores after release. SLC6A5 LOF →{' '}
        <strong>vesicle-glycine-depletion</strong> during sustained inhibitory bursts → brainstem/spinal disinhibition →
        hyperekplexia. <strong>Second most common</strong> genetic hyperekplexia gene (~15%).{' '}
        <strong>PURELY AR</strong> (no dominant-negative mechanism — unlike GLRA1/GLRB).{' '}
        <strong>First-line: Clonazepam + Forward-Flexion Manoeuvre.</strong>{' '}
        <span className="text-danger fw-bold">
          5-gene panel simultaneously (GLRA1+GLRB+SLC6A5+GPHN+ARHGEF9) — clinically indistinguishable.
          Discharge without forward-flexion training prohibited.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Apnoeic Events" value={`${kpis.apnoeic_events_pct}%`} color="#dc3545" />
        <KPI label="Rigid-Baby" value={`${kpis.rigid_baby_pct}%`} color="#dc3545" />
        <KPI label="Startle Falls" value={`${kpis.startle_falls_pct}%`} color="#fd7e14" />
        <KPI label="Epileptic Sz" value={`${kpis.epileptic_seizures_pct}%`} color="#6610f2" />
        <KPI label="Intellect Disab" value={`${kpis.intellectual_disability_pct}%`} color="#6f42c1" />
        <KPI label="On CLZ" value={`${kpis.on_clonazepam_pct}%`} color={COLOR} />
        <KPI label="Glycine Supp" value={`${kpis.glycine_supplementation_pct}%`} color="#20c997" />
        <KPI label="Manoeuvre Trained" value={`${kpis.forward_flexion_trained_pct}%`} color="#198754" />
        <KPI label="Nose-Tap +" value={`${kpis.nose_tap_positive_pct}%`} color={COLOR} />
        <KPI label="Metabolic Screen" value={`${kpis.metabolic_screened_pct}%`} color="#0d6efd" />
        <KPI label="Gene Panel Done" value={`${kpis.gene_panel_tested_pct}%`} color="#198754" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Etiology Distribution (5 Classes)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <Bar key={i} label={`${e.etiology} (${e.n})`} value={e.pct} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Treatment Evidence Levels
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="mb-2 pb-1 border-bottom small">
                  <span className="fw-bold">{t.drug}</span>
                  <span className="ms-2 text-muted">{t.level}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: LIGHT }}>
              Clinical Thresholds
            </div>
            <div className="card-body">
              {thresholds.map((t, i) => (
                <div key={i} className="mb-2 small border-bottom pb-1">
                  <span className="fw-semibold">{t.parameter}:</span>{' '}
                  <span className="text-primary">{t.threshold}</span>{' '}
                  <span className="text-muted">— {t.action}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small text-danger" style={{ background: '#fff5f5' }}>
              Contraindications (NEVER DO)
            </div>
            <div className="card-body">
              {cis.map((c, i) => (
                <div key={i} className="mb-1 small text-danger">⛔ {c}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Monitoring Schedule
        </div>
        <div className="card-body">
          <div className="table-responsive">
            <table className="table table-sm table-hover small mb-0">
              <thead><tr><th>Timepoint</th><th>Action</th></tr></thead>
              <tbody>
                {monitoring.map((m, i) => (
                  <tr key={i}><td className="fw-semibold">{m.timepoint}</td><td>{m.action}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Disease Lifecycle Windows
        </div>
        <div className="card-body d-flex flex-wrap gap-2">
          {lifecycle.map((l, i) => (
            <div key={i} className="card border p-2 small" style={{ minWidth: 180, borderLeft: `3px solid ${COLOR}` }}>
              <div className="fw-bold">{l.phase}</div>
              <div className="text-muted">{l.summary}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading patients…</div>;
  const patients = data.patient_sample || [];
  const summary = data.summary || {};

  return (
    <div>
      <div className="row g-3 mb-3">
        {[
          ['Apnoeic Events', `${summary.apnoeic_pct}%`, '#dc3545'],
          ['Rigid Baby', `${summary.rigid_baby_pct}%`, '#dc3545'],
          ['Epileptic Sz', `${summary.epileptic_seizures_pct}%`, '#6610f2'],
          ['Intellect Disab', `${summary.intellectual_disability_pct}%`, '#6f42c1'],
          ['Manoeuvre Trained', `${summary.forward_flexion_trained_pct}%`, '#198754'],
          ['Metabolic Screen', `${summary.metabolic_screened_pct}%`, '#0d6efd'],
          ['Video-EEG Done', `${summary.video_eeg_done_pct}%`, '#0dcaf0'],
          ['Gene Panel Done', `${summary.gene_panel_tested_pct}%`, '#198754'],
        ].map(([label, value, color], i) => (
          <KPI key={i} label={label} value={value} color={color} />
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-bold small" style={{ background: LIGHT }}>
          Patient Cohort Sample (first 15 of 40)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover small mb-0">
            <thead>
              <tr>
                <th>ID</th><th>Sex</th><th>Age</th><th>Onset</th><th>Category</th>
                <th>Apnoea</th><th>Rigid</th><th>Epileptic</th><th>ID</th>
                <th>CLZ</th><th>Gly-Supp</th><th>Trained</th><th>Panel</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age}y</td>
                  <td>{p.onset_age}y</td>
                  <td><span className="badge" style={{ background: COLOR, fontSize: '0.65rem' }}>
                    {p.category.replace('SLC6A5-', '').replace('Phenocopy-SLC6A5-Negative', 'Phenocopy')}
                  </span></td>
                  <td>{p.apnoeic_events ? '✓' : '—'}</td>
                  <td>{p.rigid_baby ? '✓' : '—'}</td>
                  <td>{p.epileptic_seizures ? '⚠' : '—'}</td>
                  <td>{p.intellectual_disability ? '⚠' : '—'}</td>
                  <td>{p.on_clonazepam ? '✓' : '—'}</td>
                  <td>{p.glycine_supplementation ? '🧪' : '—'}</td>
                  <td>{p.forward_flexion_trained ? '✓' : '❌'}</td>
                  <td>{p.gene_panel_tested ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function EventsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const events = data.event_detail || [];
  const triggers = data.trigger_detail || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Event Types (5)</h6>
      <div className="row g-3 mb-4">
        {events.map((e, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100" style={{ borderLeft: `3px solid ${COLOR}` }}>
              <div className="card-body p-3">
                <div className="fw-bold small mb-1">{e.event}</div>
                <div className="text-muted small mb-2">{e.description}</div>
                <div className="small mb-1"><span className="fw-semibold">Frequency:</span> {e.frequency_pct}%</div>
                <div className="small mb-1"><span className="fw-semibold">EEG:</span>{' '}
                  <span className="text-success">{e.eeg}</span></div>
                <div className="small text-primary">{e.management}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Triggers (7)</h6>
      <div className="row g-2">
        {triggers.map((t, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100 p-2 small">
              <div className="fw-bold">{t.trigger}</div>
              <div className="text-muted">{t.description}</div>
              <div className="text-primary mt-1">{t.management}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatment_detail || [];
  const cis = data.contraindications || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Treatment Lines (8)</h6>
      {treatments.map((t, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header py-2" style={{ background: LIGHT }}>
            <span className="fw-bold">{t.drug}</span>
            <span className="ms-2 badge" style={{ background: COLOR }}>{t.level}</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><span className="fw-semibold">Mechanism:</span> {t.mechanism}</div>
            <div className="mb-1"><span className="fw-semibold">Dose:</span> {t.dose}</div>
            <div className="mb-1"><span className="fw-semibold">CI:</span> <span className="text-danger">{t.ci}</span></div>
            <div><span className="fw-semibold">Evidence:</span> <span className="text-primary">{t.evidence}</span></div>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3 text-danger">Absolute Contraindications (5)</h6>
      {cis.map((c, i) => (
        <div key={i} className="alert alert-danger py-2 small mb-2">
          <span className="fw-bold text-danger">⛔ {c.drug}</span>
          <span className="ms-2 badge bg-danger">{c.level}</span>
          <div className="mt-1 text-dark">{c.reason}</div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const concepts = data.concepts || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const refs = data.references || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Core Concepts (15)</h6>
      {concepts.map((c, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header py-1 small fw-bold" style={{ background: LIGHT }}>{c.concept}</div>
          <div className="card-body py-2 small text-muted">{c.explanation}</div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: COLOR }}>Clinical Thresholds</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-hover small mb-0">
          <thead><tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr></thead>
          <tbody>
            {thresholds.map((t, i) => (
              <tr key={i}>
                <td className="fw-semibold">{t.parameter}</td>
                <td className="text-primary">{t.threshold}</td>
                <td className="text-muted">{t.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Clinical Standards</h6>
      <ul className="small mb-4">
        {standards.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
      </ul>

      <h6 className="fw-bold mb-3" style={{ color: COLOR }}>Key References</h6>
      <ol className="small">
        {refs.map((r, i) => <li key={i} className="mb-1 text-muted">{r}</li>)}
      </ol>
    </div>
  );
}

export default function SLC6A5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/slc6a5/overview`).then(r => r.json()),
      fetch(`${API}/api/slc6a5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/slc6a5/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 SLC6A5 Hyperekplexia Type 3 (HYPER3)
          </h4>
          <div className="text-muted small">
            GlyT2 / Glycine Transporter 2 / 11p15.1 · OMIM Gene 604159 · OMIM Disease 614618 ·
            799aa · 12 TM · 3Na⁺+Cl⁻+Gly stoichiometry · AR only · 2nd most common (~15%) ·
            Presynaptic vesicle-glycine-depletion mechanism · 40-patient cohort seed-499
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger py-2 small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <EventsTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
