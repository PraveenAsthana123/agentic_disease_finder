'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// Deep teal/cerulean theme — NMDA receptor / ion channel identity
const C = '#0b6e6e';
const CL = '#e0f4f4';

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

function Bar({ label, value, color = C }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const etioDist = data.etiology_distribution || {};
  const etiologies = Object.entries(etioDist).map(([cat, n]) => ({ cat, n }));
  const maxN = Math.max(...etiologies.map(e => e.n), 1);

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3" style={{ borderLeft: `4px solid ${C}` }}>
        <strong>GRIN1 (9q34.3) — GluN1 / NR1 · Obligatory NMDA Receptor Subunit · DEE + Hyperkinetic Movement Disorder:</strong>{' '}
        GRIN1 encodes GluN1 (NR1) — the <strong>OBLIGATORY subunit of ALL NMDA receptors</strong> (2x GluN1 + 2x GluN2/3 per heterotetramer).{' '}
        GluN1 binds glycine/D-serine as the essential co-agonist.{' '}
        LOF variants → <strong>NMDA hypofunction → PV⁺ interneuron failure → cortical disinhibition → DEE + hyperkinetic movement disorder</strong>.{' '}
        GOF variants (lurcher-equivalent p.Asn615Lys) → <strong>constitutive Ca²⁺ influx → excitotoxicity → severe DEE</strong>.{' '}
        pLI ~0.99 (most intolerant gene in genome). AD de novo {'>'}95%. OMIM #616346. Discovery: Lemke 2016.
      </div>

      <div className="alert alert-warning py-2 small mb-3">
        <strong>⚠ GENOTYPE-FIRST: </strong>
        D-serine (LOF precision) WORSENS GOF variants. Memantine (GOF precision) WORSENS LOF.{' '}
        <strong>Functional electrophysiology required to confirm LOF vs GOF before precision therapy.</strong>{' '}
        LEV is safe in BOTH — start LEV while awaiting functional data.
      </div>

      <div className="row g-2 mb-3">
        <KPI label="Cohort" value={data.cohort_size} color={C} />
        <KPI label="Seizure-Free %" value={`${data.seizure_free_pct}%`} color="#198754" />
        <KPI label="Locus" value={data.locus} color="#6f42c1" />
        <KPI label="pLI" value={data.pli} color="#dc3545" />
        <KPI label="Inheritance" value="AD de novo" color="#fd7e14" />
        <KPI label="OMIM" value="#616346" color="#0d6efd" />
      </div>

      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header small fw-bold" style={{ backgroundColor: CL }}>
              🧬 Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map(({ cat, n }) => (
                <div key={cat} className="mb-2">
                  <div className="d-flex justify-content-between small mb-1">
                    <span style={{ fontSize: '0.78rem' }}>{cat.replace(/-/g, ' ')}</span>
                    <span className="text-muted">{n}</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar" style={{ width: `${(n / maxN) * 100}%`, backgroundColor: C }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header small fw-bold" style={{ backgroundColor: CL }}>
              ⚗️ Mechanism + Precision Pharmacology
            </div>
            <div className="card-body small">
              <p>{data.mechanism}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <div className="card shadow-sm border-warning">
            <div className="card-header small fw-bold bg-warning bg-opacity-25">
              💡 Key Clinical Pearl
            </div>
            <div className="card-body small">{data.key_clinical_pearl}</div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm border-danger">
            <div className="card-header small fw-bold bg-danger bg-opacity-10">
              🚫 Key Contraindications
            </div>
            <div className="card-body small">{data.key_contraindication}</div>
          </div>
        </div>
      </div>

      {data.key_references && (
        <div className="card shadow-sm mt-3">
          <div className="card-header small fw-bold" style={{ backgroundColor: CL }}>
            📚 Key References
          </div>
          <ul className="list-group list-group-flush small">
            {data.key_references.map((r, i) => <li key={i} className="list-group-item py-1">{r}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { etiology_catalog = [], patients_sample = [], summary_stats = {} } = data;

  return (
    <div>
      <div className="row g-2 mb-3">
        {Object.entries(summary_stats).map(([k, v]) => (
          <div key={k} className="col-6 col-md-4 col-lg-3">
            <div className="card text-center shadow-sm">
              <div className="card-body py-2">
                <div className="fw-bold" style={{ color: C }}>{v}</div>
                <div className="text-muted small">{k.replace(/_/g, ' ')}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mb-4">
        <h6 className="fw-bold">Etiology Classes</h6>
        {etiology_catalog.map((ec, i) => (
          <div key={i} className="card shadow-sm mb-2">
            <div className="card-header small fw-bold d-flex justify-content-between" style={{ backgroundColor: CL }}>
              <span>{ec.category}</span>
              <span className="badge text-bg-secondary">{ec.pct}%</span>
            </div>
            <div className="card-body small">
              <p className="mb-1"><strong>Etiology:</strong> {ec.etiology}</p>
              <p className="mb-1"><strong>Mechanism:</strong> {ec.mechanism}</p>
              <p className="mb-1"><strong>Variants:</strong> <code>{ec.typical_variants}</code></p>
              <p className="mb-1"><strong>Onset:</strong> {ec.onset_age_years} years</p>
              <p className="mb-0"><strong>Outcome:</strong> {ec.outcome}</p>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold">Patient Sample (first 15)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-light">
            <tr>
              <th>ID</th><th>Name</th><th>Age Onset</th><th>Age Now</th>
              <th>Etiology</th><th>Variant</th><th>AED</th><th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {patients_sample.slice(0, 15).map(p => (
              <tr key={p.id}>
                <td><code>{p.id}</code></td>
                <td>{p.name}</td>
                <td>{p.age_at_onset_years}y</td>
                <td>{p.age_now_years}y</td>
                <td><span className="badge" style={{ backgroundColor: C, fontSize: '0.7rem' }}>
                  {p.etiology_class.split('-').slice(0, 2).join('-')}
                </span></td>
                <td><code style={{ fontSize: '0.7rem' }}>{p.variant}</code></td>
                <td>{p.current_aed}</td>
                <td>{p.outcome}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { seizure_types = [], triggers = [] } = data;

  return (
    <div>
      <h6 className="fw-bold mb-3">Seizure Types (5)</h6>
      {seizure_types.map((s, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header small fw-bold d-flex justify-content-between" style={{ backgroundColor: CL }}>
            <span>{s.type}</span>
            <span className="badge text-bg-primary">{s.freq_pct}%</span>
          </div>
          <div className="card-body small">
            <Bar label="Frequency" value={s.freq_pct} />
            <p className="mb-1"><strong>EEG:</strong> {s.eeg_signature}</p>
            <p className="mb-1"><strong>Semiology:</strong> {s.semiology}</p>
            <p className="mb-0"><strong>Clinical Tip:</strong> {s.clinical_tip}</p>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Triggers (8)</h6>
      {triggers.map((t, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header small fw-bold d-flex justify-content-between" style={{ backgroundColor: '#fff3cd' }}>
            <span>⚡ {t.trigger}</span>
            <span className="badge text-bg-warning text-dark">{t.freq_pct}%</span>
          </div>
          <div className="card-body small">
            <Bar label="Frequency" value={t.freq_pct} color="#ffc107" />
            <p className="mb-1"><strong>Mechanism:</strong> {t.mechanism}</p>
            <p className="mb-0"><strong>Clinical Advice:</strong> {t.clinical_advice}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { treatments = [], contraindications = [], monitoring = [], lifecycle = [] } = data;

  return (
    <div>
      <h6 className="fw-bold mb-3">Treatments (8)</h6>
      {treatments.map((t, i) => (
        <div key={i} className="card shadow-sm mb-3">
          <div className="card-header small fw-bold d-flex justify-content-between" style={{ backgroundColor: CL }}>
            <span>💊 {t.drug}</span>
            <span className="badge text-bg-success">{t.level}</span>
          </div>
          <div className="card-body small">
            <p className="mb-1"><strong>Indication:</strong> {t.indication}</p>
            <p className="mb-1"><strong>Dose:</strong> {t.dose}</p>
            <p className="mb-1"><strong>MOA:</strong> {t.moa}</p>
            <p className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</p>
            <p className="mb-1"><strong>Safety:</strong> {t.safety}</p>
            <p className="mb-1"><strong>Monitoring:</strong> {t.monitoring}</p>
            {t.grin1_note && (
              <p className="mb-0 text-primary"><strong>GRIN1 Note:</strong> {t.grin1_note}</p>
            )}
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Contraindications (5)</h6>
      {contraindications.map((c, i) => (
        <div key={i} className="card shadow-sm mb-2 border-danger">
          <div className="card-header small fw-bold bg-danger bg-opacity-10 d-flex justify-content-between">
            <span>🚫 {c.drug_or_class}</span>
            <span className="badge text-bg-danger">{c.risk}</span>
          </div>
          <div className="card-body small">
            <p className="mb-1"><strong>Mechanism:</strong> {c.mechanism}</p>
            <p className="mb-0"><strong>Action:</strong> {c.clinical_action}</p>
          </div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Monitoring (14 items)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-light">
            <tr><th>Item</th><th>Frequency</th><th>Rationale</th></tr>
          </thead>
          <tbody>
            {monitoring.map((m, i) => (
              <tr key={i}>
                <td className="fw-semibold">{m.item}</td>
                <td>{m.frequency}</td>
                <td>{m.rationale}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3 mt-4">Lifecycle (6 windows)</h6>
      {lifecycle.map((lc, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header small fw-bold" style={{ backgroundColor: CL }}>
            🗓️ {lc.phase}
          </div>
          <div className="card-body small">
            <p className="mb-1 fw-semibold">Clinical Focus: {lc.clinical_focus}</p>
            <ul className="mb-0 ps-3">
              {(lc.key_events || []).map((ev, j) => <li key={j}>{ev}</li>)}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const { definitions = [], thresholds = [], standards = [], references = [] } = data;

  return (
    <div>
      <h6 className="fw-bold mb-3">Key Concepts (15)</h6>
      {definitions.map((d, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header small fw-bold" style={{ backgroundColor: CL }}>
            {d.term}
          </div>
          <div className="card-body small">{d.definition}</div>
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">Clinical Thresholds (12)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-light">
            <tr><th>Metric</th><th>Value / Threshold</th><th>Action</th></tr>
          </thead>
          <tbody>
            {thresholds.map((t, i) => (
              <tr key={i}>
                <td className="fw-semibold">{t.metric}</td>
                <td><code>{t.value}</code></td>
                <td>{t.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3 mt-4">Clinical Standards (12)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-light">
            <tr><th>Code</th><th>Standard</th><th>Relevance</th></tr>
          </thead>
          <tbody>
            {standards.map((s, i) => (
              <tr key={i}>
                <td><code>{s.code}</code></td>
                <td>{s.title}</td>
                <td>{s.relevance}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3 mt-4">References (6)</h6>
      {references.map((r, i) => (
        <div key={i} className="card shadow-sm mb-2">
          <div className="card-header small fw-bold" style={{ backgroundColor: CL }}>
            📖 {r.citation}
          </div>
          <div className="card-body small">{r.key_finding}</div>
        </div>
      ))}
    </div>
  );
}

export default function GRIN1Page() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/grin1/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(() => {});

    Promise.all([
      fetch(`${API}/api/grin1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/grin1/definitions`).then(r => r.json()),
    ])
      .then(([b, d]) => { setBreakdown(b); setDefinitions(d); })
      .catch(() => {});
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: '1.6rem' }}>🧬</span>
        <div>
          <h5 className="mb-0 fw-bold" style={{ color: C }}>
            GRIN1 Epilepsy — GluN1 / NR1 Obligatory NMDA Receptor Subunit
          </h5>
          <small className="text-muted">
            DEE + Hyperkinetic Movement Disorder · 9q34.3 · AD de novo · pLI ~0.99 ·
            D-serine Precision (LOF) · Memantine Precision (GOF) · OMIM #616346
          </small>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active' : ''}`}
              style={tab === t ? { borderBottomColor: C, color: C, fontWeight: 600 } : {}}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Etiologies' && <PatientsTab data={breakdown} />}
      {tab === 'Seizures & Triggers' && <SeizuresTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
