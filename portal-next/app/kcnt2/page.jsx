'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiology', 'Seizure Types & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a5276';   // deep navy — KNa1.2 / KCNT2 sibling-channel identity
const ACCENT2 = '#7b241c';   // dark crimson — contraindications / danger alerts
const ACCENT3 = '#1e6823';   // dark green — KD / effective treatments

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

function PctBar({ label, pct, color = ACCENT }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 8 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function AlertBadge({ text, variant = 'danger' }) {
  const bg = variant === 'warning' ? '#fff3cd' : variant === 'info' ? '#cff4fc' : '#f8d7da';
  const border = variant === 'warning' ? '#ffc107' : variant === 'info' ? '#0dcaf0' : '#dc3545';
  return (
    <div className="p-2 mb-2 rounded small" style={{ backgroundColor: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

// ─── TAB: Overview ───────────────────────────────────────────────────────────
function OverviewTab({ ov }) {
  if (!ov) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  const k = ov.kpis || {};
  return (
    <div>
      {/* Header */}
      <div className="card mb-4 shadow-sm" style={{ borderLeft: `5px solid ${ACCENT}` }}>
        <div className="card-body">
          <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>{ov.syndrome}</h4>
          <div className="row small text-muted g-1 mt-1">
            <div className="col-auto"><span className="badge bg-secondary me-1">Gene</span>{ov.gene}</div>
            <div className="col-auto"><span className="badge bg-secondary me-1">Locus</span>{ov.chromosome}</div>
            <div className="col-auto"><span className="badge bg-secondary me-1">Protein</span>{ov.protein}</div>
            <div className="col-auto"><span className="badge bg-secondary me-1">Inheritance</span>{ov.inheritance}</div>
            <div className="col-auto"><span className="badge bg-secondary me-1">OMIM DEE57</span>{ov.omim_dee57}</div>
          </div>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Patients" value={ov.n_patients} color={ACCENT} />
        <KPI label="GOF Severe (DEE57)" value={`${k.gof_severe_pct}%`} color={ACCENT2} />
        <KPI label="GOF Moderate" value={`${k.gof_moderate_pct}%`} color="#e67e22" />
        <KPI label="DRE" value={`${k.dre_pct}%`} color={ACCENT2} />
        <KPI label="Seizure Free" value={`${k.seizure_free_pct}%`} color={ACCENT3} />
        <KPI label="KD On" value={`${k.kd_on_pct}%`} color={ACCENT3} />
        <KPI label="ACTH Response" value={`${k.acth_responded_pct}%`} color={ACCENT} />
        <KPI label="IS Presentation" value={`${k.is_initial_presentation_pct}%`} color="#8e44ad" />
        <KPI label="Hypsarrhythmia" value={`${k.hypsarrhythmia_pct}%`} color="#c0392b" />
      </div>

      {/* Clinical Alerts */}
      <h6 className="fw-bold mb-2" style={{ color: ACCENT2 }}>⚠️ Clinical Alerts — KCNT2 Specific</h6>
      <div className="mb-4">
        {(ov.clinical_alerts || []).map((a, i) => {
          const variant = a.startsWith('🚨') ? 'danger' : a.startsWith('⚡') ? 'info' : 'warning';
          return <AlertBadge key={i} text={a} variant={variant} />;
        })}
      </div>

      {/* EEG & Biomarker */}
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>EEG Hallmark</div>
            <div className="card-body small">{ov.eeg_hallmark}</div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold small" style={{ background: ACCENT3, color: '#fff' }}>Key Biomarker</div>
            <div className="card-body small">{ov.key_biomarker}</div>
          </div>
        </div>
      </div>

      {/* Key AHA */}
      <div className="alert alert-primary border-0 shadow-sm">
        <strong>Key Clinical Insight:</strong> {ov.key_aha}
      </div>

      {/* Lifecycle Windows */}
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Lifecycle Windows</h6>
      <div className="row g-3">
        {(ov.lifecycle_windows || []).map((w, i) => (
          <div key={i} className="col-md-6 col-lg-4">
            <div className="card h-100 shadow-sm">
              <div className="card-header small fw-bold" style={{ background: ACCENT, color: '#fff' }}>
                {w.window} <span className="fw-normal opacity-75">({w.age_range})</span>
              </div>
              <div className="card-body small">
                <div className="mb-1"><strong>Focus:</strong> {w.focus}</div>
                <div className="text-muted">{w.key_action}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── TAB: Patients & Etiology ─────────────────────────────────────────────────
function PatientsTab({ ov }) {
  if (!ov) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Etiology Breakdown (n={ov.n_patients})</h6>
      {(ov.etiologies || []).map((e, i) => (
        <div key={i} className="mb-3">
          <PctBar label={e.etiology} pct={e.pct} color={ACCENT} />
          <div className="text-muted small ms-1">n={e.n} patients</div>
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>KCNT2 vs KCNT1 — Key Distinctions</h6>
      <table className="table table-bordered table-sm small">
        <thead style={{ background: ACCENT, color: '#fff' }}>
          <tr><th>Feature</th><th>KCNT1 (KNa1.1 / Slack)</th><th>KCNT2 (KNa1.2 / Slick)</th></tr>
        </thead>
        <tbody>
          <tr><td>Locus</td><td>9q34.3</td><td>1q31.3</td></tr>
          <tr><td>Primary cell type</td><td>PV+ interneurons (dominant)</td><td>Excitatory pyramidal neurons</td></tr>
          <tr><td>EEG hallmark</td><td>Migrating focal ictal discharges (EIMFS)</td><td>Hypsarrhythmia / West Syndrome</td></tr>
          <tr><td>OMIM</td><td>614959 (EIMFS) / 615005 (NFLE)</td><td>617771 (DEE57)</td></tr>
          <tr><td>Quinidine</td><td>Negative RCT (Numis 2020) — NOT recommended</td><td>No evidence at all — ABSOLUTE CI</td></tr>
          <tr><td>DRE rate</td><td>~40% (EIMFS)</td><td>~58% (DEE57)</td></tr>
          <tr><td>Single-channel conductance</td><td>~200 pS</td><td>~170 pS</td></tr>
          <tr><td>Heteromers</td><td>KCNT1 homodimer dominant</td><td>Can form KCNT1/KCNT2 heteromers (~183 pS)</td></tr>
        </tbody>
      </table>

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Key Recurrent Variants</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-dark">
            <tr><th>Variant</th><th>Domain</th><th>Phenotype</th><th>N reported</th></tr>
          </thead>
          <tbody>
            {[
              { v: 'p.Ile209Phe', d: 'S4-S5 linker (TM)', p: 'Severe DEE57 + IS; DRE', n: 12 },
              { v: 'p.Gly459Asp', d: 'C-term RCK1 linker', p: 'West syndrome; some voluntary motor', n: 7 },
              { v: 'p.Ala934Val', d: 'RCK2 domain', p: 'Later focal epilepsy; milder', n: 4 },
              { v: 'c.2042+1G>A', d: 'Splice site intron 14', p: 'Variable; partial exon skipping', n: 3 },
              { v: 'p.Arg474His', d: 'RCK1 domain (Na-binding)', p: 'IS + focal; intermediate', n: 3 }
            ].map((r, i) => (
              <tr key={i}>
                <td><code>{r.v}</code></td>
                <td>{r.d}</td>
                <td>{r.p}</td>
                <td className="text-center">{r.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── TAB: Seizure Types & Triggers ───────────────────────────────────────────
function SeizuresTab({ ov }) {
  if (!ov) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  const sz = ov.seizure_type_prevalence || {};
  const tr = ov.trigger_seizure_rates || {};
  const szColors = ['#1a5276', '#117a65', '#7d6608', '#6e2c00', '#6c3483'];
  const trColors = ['#c0392b', '#e67e22', '#f1c40f', '#27ae60', '#2980b9', '#8e44ad', '#16a085', '#2c3e50'];

  return (
    <div>
      <div className="row g-4">
        <div className="col-md-6">
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Seizure Type Prevalence</h6>
          {Object.entries(sz).map(([type, pct], i) => (
            <div key={i} className="mb-3">
              <PctBar label={type} pct={pct} color={szColors[i % szColors.length]} />
            </div>
          ))}
          <div className="alert alert-warning small mt-3 p-2">
            <strong>EEG:</strong> IS phase → hypsarrhythmia (73%). Post-IS → multifocal independent spike-wave. Serial EEG at 3M, 6M, 12M mandatory to track evolution.
          </div>
        </div>
        <div className="col-md-6">
          <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Seizure Trigger Rates</h6>
          {Object.entries(tr).map(([trigger, pct], i) => (
            <div key={i} className="mb-3">
              <PctBar label={trigger} pct={pct} color={trColors[i % trColors.length]} />
            </div>
          ))}
        </div>
      </div>

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>EEG Evolution — KCNT2</h6>
      <div className="table-responsive">
        <table className="table table-bordered table-sm small">
          <thead style={{ background: ACCENT, color: '#fff' }}>
            <tr><th>Phase</th><th>Age</th><th>EEG Pattern</th><th>Clinical Correlate</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>IS Phase</td><td>1–8M</td>
              <td><strong>Hypsarrhythmia / modified hypsarrhythmia</strong></td>
              <td>Epileptic spasms clusters; developmental arrest</td>
            </tr>
            <tr>
              <td>Post-IS Transition</td><td>8–24M</td>
              <td>Multifocal independent spike-wave (MISW)</td>
              <td>Focal motor seizures; FBTCS; variable improvement</td>
            </tr>
            <tr>
              <td>DRE Phase</td><td>2–6Y</td>
              <td>Multifocal SWD ± generalised bursts</td>
              <td>Drug-resistant focal epilepsy; tonic seizures</td>
            </tr>
            <tr>
              <td>Late Childhood</td><td>6–12Y</td>
              <td>Focal SWD (temporal/frontal predominant)</td>
              <td>Nocturnal GTCS; myoclonic in some; SUDEP risk</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── TAB: Treatments ─────────────────────────────────────────────────────────
function TreatmentsTab({ bk }) {
  if (!bk) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  return (
    <div>
      {/* Treatments */}
      <h6 className="fw-bold mb-3" style={{ color: ACCENT3 }}>Treatments</h6>
      {(bk.treatments || []).map((t, i) => (
        <div key={i} className="card mb-3 shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ background: ACCENT3, color: '#fff' }}>
            <span className="fw-bold">{t.drug}</span>
            <span className="badge bg-light text-dark">{t.level}</span>
          </div>
          <div className="card-body small">
            <div className="row g-2">
              <div className="col-md-6">
                <div className="mb-1"><strong>Dose:</strong> {t.dose}</div>
                <div className="mb-1"><strong>MOA:</strong> {t.moa}</div>
                <div className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
              </div>
              <div className="col-md-6">
                <div className="mb-1"><strong>Safety:</strong> {t.safety}</div>
                <div className="mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
                <div className="mt-1 p-2 rounded" style={{ background: '#e8f8f5' }}>
                  <strong style={{ color: ACCENT3 }}>KCNT2 note:</strong> {t.kcnt2_note}
                </div>
              </div>
            </div>
            <div className="mt-2 text-muted border-top pt-2">
              <strong>Evidence:</strong> {t.evidence_basis}
            </div>
          </div>
        </div>
      ))}

      {/* Contraindications */}
      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT2 }}>Contraindications</h6>
      {(bk.contraindications || []).map((c, i) => (
        <div key={i} className="card mb-3 border-danger shadow-sm">
          <div className="card-header d-flex justify-content-between"
            style={{ background: ACCENT2, color: '#fff' }}>
            <span className="fw-bold">{c.drug}</span>
            <span className="badge bg-warning text-dark">{c.level}</span>
          </div>
          <div className="card-body small">{c.reason}</div>
        </div>
      ))}

      {/* Monitoring */}
      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Monitoring Protocol</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-dark">
            <tr><th>Item</th><th>Timing</th><th>Why</th></tr>
          </thead>
          <tbody>
            {(bk.monitoring || []).map((m, i) => (
              <tr key={i}>
                <td className="fw-bold">{m.item}</td>
                <td>{m.timing}</td>
                <td className="text-muted">{m.why}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── TAB: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ df }) {
  if (!df) return <div className="text-center py-5"><div className="spinner-border" /></div>;
  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: ACCENT }}>Key Concepts (15)</h6>
      <div className="accordion" id="defAccordion">
        {(df.concepts || []).map((c, i) => (
          <div key={i} className="accordion-item border mb-2 shadow-sm rounded">
            <h2 className="accordion-header">
              <button className="accordion-button collapsed rounded fw-bold small"
                type="button" data-bs-toggle="collapse"
                data-bs-target={`#def${i}`}
                style={{ background: i % 2 === 0 ? '#eaf2ff' : '#f0fff4' }}>
                {c.term}
              </button>
            </h2>
            <div id={`def${i}`} className="accordion-collapse collapse">
              <div className="accordion-body small">{c.definition}</div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Clinical Thresholds (12)</h6>
      <div className="table-responsive">
        <table className="table table-bordered table-sm small">
          <thead style={{ background: ACCENT, color: '#fff' }}>
            <tr><th>Threshold</th><th>Value</th><th>Clinical Note</th></tr>
          </thead>
          <tbody>
            {(df.thresholds || []).map((t, i) => (
              <tr key={i}>
                <td className="fw-bold">{t.threshold}</td>
                <td><code>{t.value}</code></td>
                <td className="text-muted">{t.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mt-4 mb-3" style={{ color: ACCENT }}>Standards & References</h6>
      <div className="row g-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: ACCENT, color: '#fff' }}>
              Standards (12)
            </div>
            <ul className="list-group list-group-flush small">
              {(df.standards || []).map((s, i) => (
                <li key={i} className="list-group-item py-1">{s}</li>
              ))}
            </ul>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold small" style={{ background: ACCENT3, color: '#fff' }}>
              References (6)
            </div>
            <ul className="list-group list-group-flush small">
              {(df.references || []).map((r, i) => (
                <li key={i} className="list-group-item py-1">{r}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── PAGE ROOT ────────────────────────────────────────────────────────────────
export default function KCNT2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/kcnt2/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2) {
      if (!overview) fetch(`${API}/api/kcnt2/overview`).then(r => r.json()).then(setOverview).catch(() => {});
    }
    if (tab === 3) {
      fetch(`${API}/api/kcnt2/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 4) {
      fetch(`${API}/api/kcnt2/definitions`).then(r => r.json()).then(setDefinitions).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-4">
      {/* Page Header */}
      <div className="d-flex align-items-center mb-3">
        <div className="me-3 rounded-circle d-flex align-items-center justify-content-center"
          style={{ width: 52, height: 52, background: ACCENT, color: '#fff', fontSize: 22, flexShrink: 0 }}>
          ⚡
        </div>
        <div>
          <h3 className="fw-bold mb-0" style={{ color: ACCENT }}>
            KCNT2 Epilepsy — DEE57 / West Syndrome
          </h3>
          <div className="text-muted small">
            KNa1.2 / Slick / Slo2.1 · Sodium-Activated K⁺ Channel · 1q31.3 · AD de novo &gt;90%
            &nbsp;·&nbsp; Sibling of KCNT1 — <strong>Quinidine NOT indicated</strong>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}>
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab ov={overview} />}
      {tab === 1 && <PatientsTab ov={overview} />}
      {tab === 2 && <SeizuresTab ov={overview} />}
      {tab === 3 && <TreatmentsTab bk={breakdown} />}
      {tab === 4 && <DefinitionsTab df={definitions} />}

      <div className="text-muted small mt-5 border-top pt-3">
        KCNT2 · DEE57 · KNa1.2 / Slick / Slo2.1 · 1q31.3 · OMIM 617771
        &nbsp;|&nbsp; Ref: Ambrosino 2015 Ann Neurol · Bhatt 2023 Epilepsia · UKISS 2004 Lancet
        &nbsp;|&nbsp; Standards: ILAE 2022 · NICE NG217 · CPIC POLG 2023
      </div>
    </div>
  );
}
