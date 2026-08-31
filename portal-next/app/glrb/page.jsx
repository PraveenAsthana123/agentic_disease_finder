'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Events', 'Event Types & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#00515a';
const LIGHT = '#e0f4f7';

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
      <div className="alert alert-info py-2 small mb-3" style={{ borderLeft: `4px solid ${COLOR}` }}>
        <strong>GLRB (4q32.1) — Glycine Receptor β1 · Hyperekplexia Type 1B:</strong>{' '}
        GLRB is the obligate <strong>structural partner of GLRA1 in the adult GlyR pentamer (α1₂β₃)</strong>.{' '}
        GLRB LOF → homomeric α1 formation (reduced conductance) + loss of gephyrin-mediated GlyR anchoring{' '}
        → <strong>dual postsynaptic glycinergic deficit</strong> → hyperekplexia.{' '}
        Accounts for <strong>~5% of genetic hyperekplexia</strong>.{' '}
        <strong>First-line: Clonazepam + Forward-Flexion Manoeuvre.</strong>{' '}
        <span className="text-danger fw-bold">
          DIAGNOSIS: 5-gene panel simultaneously (GLRB+GLRA1+SLC6A5+GPHN+ARHGEF9) —
          never single-gene only. Discharge without forward-flexion training prohibited.
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
        <KPI label="Manoeuvre Trained" value={`${kpis.forward_flexion_trained_pct}%`} color="#198754" />
        <KPI label="Nose-Tap +" value={`${kpis.nose_tap_positive_pct}%`} color={COLOR} />
        <KPI label="Metabolic Screened" value={`${kpis.metabolic_screened_pct}%`} color="#0d6efd" />
        <KPI label="Video-EEG Done" value={`${kpis.video_eeg_done_pct}%`} color="#0dcaf0" />
        <KPI label="Gene Panel Tested" value={`${kpis.gene_panel_tested_pct}%`} color="#198754" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.etiology}</span>
                    <span className="badge" style={{ background: COLOR }}>{e.n} ({e.pct}%)</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.pct / maxEtio * 100)}%`, background: COLOR }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
              Treatment Lines
            </div>
            <div className="card-body">
              {treatments.map((t, i) => (
                <div key={i} className="d-flex justify-content-between align-items-start mb-2 small">
                  <span className="fw-semibold">{t.drug}</span>
                  <span className="badge ms-2" style={{ background: COLOR, whiteSpace: 'normal', textAlign: 'right' }}>{t.level}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
              Clinical Thresholds
            </div>
            <div className="card-body">
              <table className="table table-sm small mb-0">
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.label}</td>
                      <td className="text-end"><span className="badge" style={{ background: COLOR }}>{t.value} {t.unit}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#b71c1c' }}>
              ⚠ Contraindications / Safety Requirements
            </div>
            <div className="card-body">
              {cis.map((c, i) => (
                <div key={i} className="alert alert-danger py-1 px-2 small mb-1">{c}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          Monitoring Schedule (Key Items)
        </div>
        <div className="card-body">
          <table className="table table-sm small mb-0">
            <thead><tr><th>Item</th><th>Frequency</th></tr></thead>
            <tbody>
              {monitoring.map((m, i) => (
                <tr key={i}><td className="fw-semibold">{m.item}</td><td>{m.frequency}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          Lifecycle Management Windows
        </div>
        <div className="card-body">
          {lifecycle.map((lw, i) => (
            <div key={i} className="mb-2 p-2 rounded small" style={{ background: i % 2 === 0 ? LIGHT : '#fff' }}>
              <strong style={{ color: COLOR }}>{lw.window}:</strong>{' '}{lw.headline}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const etios = data.etiology_distribution || [];
  const patients = data.patient_sample || [];
  const summary = data.summary || {};

  return (
    <div>
      <div className="row g-3 mb-4">
        {etios.map((e, i) => (
          <div key={i} className="col-md-6 col-lg-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold text-white py-2 small" style={{ background: COLOR }}>
                {e.etiology} — {e.n} pts ({e.pct}%)
              </div>
              <div className="card-body small">
                <p className="mb-2" style={{ whiteSpace: 'pre-line' }}><strong>Mechanism:</strong> {e.mechanism?.slice(0, 280)}…</p>
                <p className="mb-1"><strong>Typical Variants:</strong> {e.typical_variants}</p>
                <p className="mb-1"><strong>EEG:</strong> {e.eeg_signature}</p>
                <p className="mb-0"><strong>Phenotype:</strong> {e.phenotype}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        {[
          { label: 'Apnoeic Events', val: `${summary.apnoeic_pct}%`, color: '#dc3545' },
          { label: 'Rigid-Baby', val: `${summary.rigid_baby_pct}%`, color: '#fd7e14' },
          { label: 'Epileptic Seizures', val: `${summary.epileptic_seizures_pct}%`, color: '#6610f2' },
          { label: 'Intellect Disab', val: `${summary.intellectual_disability_pct}%`, color: '#6f42c1' },
          { label: 'Manoeuvre Trained', val: `${summary.forward_flexion_trained_pct}%`, color: '#198754' },
          { label: 'Gene Panel Tested', val: `${summary.gene_panel_tested_pct}%`, color: COLOR },
        ].map((s, i) => (
          <div key={i} className="col-6 col-md-4 col-lg-2">
            <div className="card text-center shadow-sm">
              <div className="card-body py-2">
                <div className="fw-bold fs-5" style={{ color: s.color }}>{s.val}</div>
                <div className="text-muted small">{s.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          Patient Sample (first 15 of 40)
        </div>
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-striped small mb-0">
              <thead>
                <tr style={{ background: LIGHT }}>
                  <th>ID</th><th>Sex</th><th>Age</th><th>Onset</th><th>Category</th>
                  <th>Apnoea</th><th>Rigid</th><th>Falls</th><th>EpilSz</th><th>ID</th>
                  <th>CLZ</th><th>Trained</th><th>Nose+</th><th>Panel</th>
                </tr>
              </thead>
              <tbody>
                {patients.map((p, i) => (
                  <tr key={i}>
                    <td>{p.id}</td>
                    <td>{p.sex}</td>
                    <td>{p.age}y</td>
                    <td>{p.onset_age}y</td>
                    <td><small>{p.category}</small></td>
                    <td>{p.apnoeic_events ? '✓' : '–'}</td>
                    <td>{p.rigid_baby ? '✓' : '–'}</td>
                    <td>{p.startle_falls ? '✓' : '–'}</td>
                    <td>{p.epileptic_seizures ? <span className="text-danger">✓</span> : '–'}</td>
                    <td>{p.intellectual_disability ? <span className="text-warning">✓</span> : '–'}</td>
                    <td>{p.on_clonazepam ? '✓' : '–'}</td>
                    <td>{p.forward_flexion_trained ? <span className="text-success">✓</span> : <span className="text-danger">✗</span>}</td>
                    <td>{p.nose_tap_positive ? '✓' : '–'}</td>
                    <td>{p.gene_panel_tested ? <span className="text-success">✓</span> : <span className="text-warning">!</span>}</td>
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

function EventsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const events = data.event_detail || [];
  const triggers = data.trigger_detail || [];
  const maxTrig = Math.max(...triggers.map(t => t.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-warning py-2 small mb-3">
        <strong>⚡ Critical Distinction:</strong> GLRB hyperekplexia events are <strong>NON-EPILEPTIC</strong> —
        normal EEG during events. Forward-flexion manoeuvre terminates apnoea; AEDs do not.
        GLRB phenotype generally milder than GLRA1 Arg271 but management principles identical.
        5-gene panel (GLRB+GLRA1+SLC6A5+GPHN+ARHGEF9) mandatory — cannot distinguish clinically.
      </div>

      <div className="row g-3 mb-4">
        {events.map((ev, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold text-white py-1 small" style={{ background: COLOR }}>
                {ev.type} — {ev.prevalence_pct}%
              </div>
              <div className="card-body small">
                <p className="mb-2"><strong>Semiology:</strong> {ev.semiology?.slice(0, 250)}…</p>
                <p className="mb-2"><strong>EEG:</strong> <em>{ev.eeg_pattern?.slice(0, 180)}…</em></p>
                <div className="alert alert-info py-1 px-2 mb-0 small"><strong>Clinical tip:</strong> {ev.clinical_tip?.slice(0, 200)}…</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
          Hyperekplexia Triggers (% of cohort)
        </div>
        <div className="card-body">
          {triggers.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="d-flex justify-content-between small fw-semibold mb-1">
                <span>{t.trigger}</span><span className="text-muted">{t.pct}%</span>
              </div>
              <div className="progress mb-1" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${Math.round(t.pct / maxTrig * 100)}%`, background: COLOR }} />
              </div>
              <small className="text-muted">{t.note}</small>
            </div>
          ))}
        </div>
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
      <div className="alert alert-success py-2 small mb-3">
        <strong>Treatment Principle:</strong> Clonazepam (GABAergic compensation) + Forward-Flexion Manoeuvre (acute apnoea).
        GLRB events are NOT epileptic — do not use phenytoin or carbamazepine.
        GLRB patients typically respond at lower CLZ doses than GLRA1 Arg271.
        5-gene panel mandatory — phenotypic overlap prevents single-gene diagnosis.
      </div>

      <div className="row g-3 mb-4">
        {treatments.map((t, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold text-white py-1 small" style={{ background: COLOR }}>
                {t.drug} — {t.level}
              </div>
              <div className="card-body small">
                <p className="mb-1"><strong>MOA:</strong> {t.moa?.slice(0, 180)}…</p>
                <p className="mb-1"><strong>Dose:</strong> {t.dose}</p>
                <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
                <p className="mb-1"><strong>Safety:</strong> {t.safety?.slice(0, 120)}…</p>
                <div className="alert alert-info py-1 px-2 mb-0 small">
                  <strong>GLRB Note:</strong> {t.glrb_note?.slice(0, 200)}…
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: '#b71c1c' }}>
          ⚠ Contraindications &amp; Safety Requirements
        </div>
        <div className="card-body">
          {cis.map((ci, i) => (
            <div key={i} className="alert alert-danger py-2 px-3 mb-2 small">
              <div className="fw-bold">{ci.drug} — <span className="text-danger">{ci.risk}</span></div>
              <div>{ci.reason}</div>
            </div>
          ))}
        </div>
      </div>
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
      <div className="row g-3 mb-4">
        {concepts.map((c, i) => (
          <div key={i} className="col-md-6 col-lg-4">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-semibold text-white py-1 small" style={{ background: COLOR }}>{c.term}</div>
              <div className="card-body small">{c.definition}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>Clinical Thresholds</div>
            <div className="card-body">
              <table className="table table-sm small mb-0">
                <tbody>
                  {thresholds.map((t, i) => (
                    <tr key={i}>
                      <td className="fw-semibold">{t.label}</td>
                      <td className="text-end"><span className="badge" style={{ background: COLOR }}>{t.value} {t.unit}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>Clinical Standards</div>
            <div className="card-body">
              {standards.map((s, i) => (
                <div key={i} className="small mb-2 pb-2 border-bottom">{s}</div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>Key References</div>
        <div className="card-body">
          {refs.map((r, i) => (
            <div key={i} className="small mb-1 pb-1 border-bottom text-muted">{r}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function GLRBPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/glrb/overview`).then(r => r.json()).then(setOverview).catch(() => {});
  }, []);

  useEffect(() => {
    if (tab === 'Patients & Events' && !breakdown)
      fetch(`${API}/api/glrb/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 'Event Types & Triggers' && !breakdown)
      fetch(`${API}/api/glrb/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 'Treatments' && !breakdown)
      fetch(`${API}/api/glrb/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    if (tab === 'Definitions' && !definitions)
      fetch(`${API}/api/glrb/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
  }, [tab]);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h2 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 GLRB Hyperekplexia
          </h2>
          <div className="text-muted small">
            Hyperekplexia Type 1B (HYPER1B) · OMIM Gene 138492 · 4q32.1 ·
            Glycine Receptor β1 · α1₂β₃ Structural Partner · ~5% of Genetic Hyperekplexia ·
            Dual Deficit: Homomeric α1 + Gephyrin-Loss · 40-patient cohort seed-497
          </div>
        </div>
        <span className="badge ms-auto" style={{ background: COLOR, fontSize: '0.9rem' }}>
          NON-EPILEPTIC
        </span>
      </div>

      <div className="d-flex gap-2 mb-4 flex-wrap">
        {TABS.map(t => (
          <button
            key={t}
            className={`btn btn-sm ${tab === t ? 'text-white' : 'btn-outline-secondary'}`}
            style={tab === t ? { background: COLOR, borderColor: COLOR } : {}}
            onClick={() => setTab(t)}
          >{t}</button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Events' && <PatientsTab data={breakdown} />}
      {tab === 'Event Types & Triggers' && <EventsTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
