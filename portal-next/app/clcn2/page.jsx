'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// Teal-green theme — chloride channel / ionic homeostasis identity
const C = '#0d7377';
const CL = '#e0f5f5';

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
  const etiologies = data.etiology_catalog || [];

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3" style={{ borderLeft: `4px solid ${C}` }}>
        <strong>CLCN2 (3q26.1) — CLC-2 Voltage-Gated Chloride Channel · GGE Spectrum · GOF Precision:</strong>{' '}
        CLCN2 GOF variants increase CLC-2 Cl⁻/HCO₃⁻ conductance → elevated neuronal [Cl⁻]ᵢ → E_Cl shift toward
        depolarisation → impaired GABAergic inhibition → thalamo-cortical 3-Hz SWD →{' '}
        <strong>JME / CAE / JAE / GTCS-Alone</strong> (GGE spectrum, AD, ~65% penetrance).{' '}
        <em>
          Precision treatment: <strong>Acetazolamide (AZM)</strong> — carbonic anhydrase inhibition
          reduces [HCO₃⁻] → attenuates CLC-2 GOF HCO₃⁻ current component. Level C add-on.
          Biallelic LOF (AR) → leukoencephalopathy (completely different entity; not epilepsy).
        </em>{' '}
        <span className="text-danger fw-bold">
          ABSOLUTE CI: CBZ/OXC/PHT (GGE aggravation) · TGB (NCSE).
          HIGH RISK: AZM + topiramate (additive metabolic acidosis — NEVER combine).
          POLG1 mandatory before VPA. LTG: EEG mandatory pre-prescribing (15–20% myoclonic aggravation).
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Cohort Size" value={data.cohort_size} color={C} />
        <KPI label="Seizure-Free" value={`${data.seizure_freedom_pct}%`} color="#198754" />
        <KPI label="pLI Score" value={data.pli} color={C} />
        <KPI label="Inheritance" value="AD GOF" color="#0d6efd" />
        <KPI label="Penetrance" value="~65%" color="#e83e8c" />
        <KPI label="Locus" value={data.locus} color="#6f42c1" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map((e, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between small fw-semibold mb-1">
                    <span>{e.category.replace('CLCN2-', '').replace(/-/g, ' ')}</span>
                    <span className="badge" style={{ background: C }}>{e.pct}%</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar" style={{ width: `${e.pct}%`, backgroundColor: C }} />
                  </div>
                  <div className="small text-muted mt-1">{e.etiology}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
              Mechanism &amp; Precision Therapy
            </div>
            <div className="card-body small">
              <p><strong>CLC-2 Channel Biology:</strong> {data.mechanism}</p>
              <hr />
              <p className="mb-1"><strong>Gene:</strong> {data.gene} · {data.locus}</p>
              <p className="mb-1"><strong>Protein:</strong> {data.protein}</p>
              <p className="mb-1"><strong>OMIM:</strong> {data.omim}</p>
              <p className="mb-1"><strong>Inheritance:</strong> {data.inheritance}</p>
              <p className="mb-1"><strong>Avg. Onset:</strong> {data.avg_onset_years} years</p>
              <p className="mb-1"><strong>VPPP Active:</strong> {data.vppp_active} patients</p>
              <div className="alert alert-warning py-2 mt-2 small mb-0">
                <strong>⚠ Precision Rule:</strong> Acetazolamide (AZM) + Topiramate = ABSOLUTE COMBINATION CI.
                Both inhibit carbonic anhydrase → severe metabolic acidosis (HCO₃⁻ &lt;15 mEq/L).
                If AZM is chosen → NEVER add topiramate. If topiramate used → NEVER add AZM.
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Lifecycle Windows
        </div>
        <div className="card-body">
          <div className="row g-2">
            {(data.lifecycle || []).map((lc, i) => (
              <div key={i} className="col-md-4 mb-2">
                <div className="border rounded p-2 h-100 small" style={{ borderColor: C }}>
                  <div className="fw-bold mb-1" style={{ color: C }}>{lc.window}</div>
                  <div className="text-muted mb-1">{lc.focus}</div>
                  <ul className="mb-0 ps-3">
                    {(lc.key_actions || []).slice(0, 3).map((a, j) => <li key={j}>{a}</li>)}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const patients = data.patients || [];
  const [filter, setFilter] = useState('All');
  const categories = ['All', ...new Set(patients.map(p => p.etiology.replace('CLCN2-GOF-', '').replace('CLCN2-', '')))];
  const filtered = filter === 'All' ? patients : patients.filter(p =>
    p.etiology.replace('CLCN2-GOF-', '').replace('CLCN2-', '') === filter
  );

  return (
    <div>
      <div className="mb-3 d-flex flex-wrap gap-2">
        {categories.map(cat => (
          <button key={cat}
            className={`btn btn-sm ${filter === cat ? 'text-white' : 'btn-outline-secondary'}`}
            style={filter === cat ? { background: C, borderColor: C } : {}}
            onClick={() => setFilter(cat)}
          >{cat.replace(/-/g, ' ')}</button>
        ))}
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="text-white" style={{ background: C }}>
            <tr>
              <th>ID</th><th>Name</th><th>Sex</th><th>Onset(Y)</th><th>Age</th>
              <th>Etiology</th><th>Variant</th><th>AEDs</th><th>Sz-Free</th><th>VPPP</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}>
                <td>{p.id}</td>
                <td>{p.name}</td>
                <td>{p.sex}</td>
                <td>{p.age_onset}</td>
                <td>{p.age_current}</td>
                <td className="small">{p.etiology.replace('CLCN2-GOF-', '').replace('CLCN2-', '')}</td>
                <td><code className="small">{p.variant}</code></td>
                <td>{(p.aeds || []).join(', ')}</td>
                <td>{p.seizure_free
                  ? <span className="badge bg-success">Yes</span>
                  : <span className="badge bg-secondary">No</span>}</td>
                <td>{p.vppp_active
                  ? <span className="badge bg-warning text-dark">Active</span>
                  : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="row g-3 mt-2">
        {(data.etiology_catalog || []).map((e, i) => (
          <div key={i} className="col-md-6">
            <div className="card h-100 shadow-sm">
              <div className="card-header small fw-semibold text-white py-2" style={{ background: C }}>
                {e.category.replace('CLCN2-', '').replace(/-/g, ' ')} — {e.pct}%
              </div>
              <div className="card-body small">
                <p className="mb-1"><strong>Etiology:</strong> {e.etiology}</p>
                <p className="mb-1"><strong>Variants:</strong> <code>{e.typical_variants}</code></p>
                <p className="mb-1"><strong>Onset:</strong> ~{e.onset_age_years} years</p>
                <p className="mb-1"><strong>Outcome:</strong> {e.outcome}</p>
                <p className="mb-0 text-muted">{e.mechanism.slice(0, 200)}…</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: C }}>Seizure Types</h6>
      <div className="row g-3 mb-4">
        {seizures.map((s, i) => (
          <div key={i} className="col-md-6">
            <div className="card h-100 shadow-sm">
              <div className="card-header text-white py-2 d-flex justify-content-between" style={{ background: C }}>
                <span className="fw-semibold small">{s.type}</span>
                <span className="badge bg-light text-dark">{s.prevalence_pct}%</span>
              </div>
              <div className="card-body small">
                <p className="mb-1"><strong>EEG:</strong> {s.eeg}</p>
                <p className="mb-1"><strong>Semiology:</strong> {s.semiology}</p>
                <div className="alert alert-secondary py-1 px-2 mb-0 small">
                  💡 {s.clinical_tips}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3" style={{ color: C }}>Seizure Triggers</h6>
      <div className="row g-2">
        {triggers.sort((a, b) => b.prevalence_pct - a.prevalence_pct).map((t, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-body small p-3">
                <div className="d-flex align-items-center mb-2">
                  <div className="fw-bold me-2">{t.trigger}</div>
                  <span className="badge ms-auto" style={{ background: C }}>{t.prevalence_pct}%</span>
                </div>
                <div className="progress mb-2" style={{ height: 8 }}>
                  <div className="progress-bar" style={{ width: `${t.prevalence_pct}%`, backgroundColor: C }} />
                </div>
                <p className="text-muted mb-1"><strong>Mechanism:</strong> {t.mechanism.slice(0, 150)}…</p>
                <p className="mb-0"><strong>Advice:</strong> {t.clinical_advice.slice(0, 150)}…</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <h6 className="fw-bold mb-3" style={{ color: C }}>Treatment Protocol</h6>
      <div className="row g-3 mb-4">
        {treatments.map((t, i) => (
          <div key={i} className="col-12">
            <div className="card shadow-sm">
              <div className="card-header d-flex align-items-center py-2" style={{ background: CL }}>
                <span className="fw-bold me-2" style={{ color: C }}>{t.drug}</span>
                <span className={`badge ms-2 ${t.level.includes('A') ? 'bg-success' : t.level.includes('B') ? 'bg-primary' : 'bg-secondary'}`}>
                  {t.level}
                </span>
              </div>
              <div className="card-body small">
                <div className="row g-2">
                  <div className="col-md-3">
                    <strong>Indication:</strong><br />{t.indication}
                  </div>
                  <div className="col-md-3">
                    <strong>Dosing:</strong><br />{t.dose}
                  </div>
                  <div className="col-md-3">
                    <strong>Efficacy:</strong><br />{t.efficacy}
                  </div>
                  <div className="col-md-3">
                    <strong>Safety:</strong><br />{t.safety.slice(0, 120)}…
                  </div>
                </div>
                <div className="alert alert-info py-1 px-2 mt-2 mb-0 small">
                  <strong>CLCN2-specific:</strong> {t.clcn2_note}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3 text-danger">Contraindications</h6>
      <div className="row g-2">
        {contraindications.map((c, i) => (
          <div key={i} className="col-md-6">
            <div className={`card shadow-sm h-100 border-${c.risk.includes('ABSOLUTE') ? 'danger' : 'warning'}`}>
              <div className={`card-header py-2 text-white ${c.risk.includes('ABSOLUTE') ? 'bg-danger' : 'bg-warning text-dark'}`}>
                <strong>{c.drug}</strong>
              </div>
              <div className="card-body small">
                <p className="fw-bold mb-1">{c.risk}</p>
                <p className="text-muted mb-0">{c.mechanism.slice(0, 200)}…</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card shadow-sm mt-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Monitoring Protocol (14 items)
        </div>
        <div className="card-body">
          <div className="row g-2">
            {(data.monitoring || []).map((m, i) => (
              <div key={i} className="col-md-6 small">
                <div className="border rounded p-2" style={{ borderColor: C }}>
                  <div className="fw-bold" style={{ color: C }}>{m.item}</div>
                  <div className="text-muted">{m.timing}</div>
                  <div>{m.rationale}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading…</div>;

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Gene Summary
        </div>
        <div className="card-body small">{data.gene_summary}</div>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: C }}>Key Concepts (15)</h6>
      <div className="row g-2 mb-4">
        {(data.concepts || []).map((c, i) => (
          <div key={i} className="col-md-6">
            <div className="card shadow-sm h-100">
              <div className="card-header py-2 small fw-bold" style={{ color: C, background: CL }}>
                {c.term}
              </div>
              <div className="card-body small">{c.definition}</div>
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3" style={{ color: C }}>Clinical Thresholds</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm small">
          <thead style={{ background: CL }}>
            <tr><th>Threshold</th><th>Value</th></tr>
          </thead>
          <tbody>
            {(data.thresholds || []).map((t, i) => (
              <tr key={i}><td>{t.name}</td><td><code>{t.value}</code></td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3" style={{ color: C }}>Clinical Standards (12)</h6>
      <div className="row g-2 mb-4">
        {(data.standards || []).map((s, i) => (
          <div key={i} className="col-md-6">
            <div className="small border rounded p-2">
              <span className="badge me-1" style={{ background: C }}>{s.code}</span>
              {s.title}
            </div>
          </div>
        ))}
      </div>

      <h6 className="fw-bold mb-3" style={{ color: C }}>References</h6>
      <ol className="small">
        {(data.references || []).map((r, i) => (
          <li key={i}>{r.citation}</li>
        ))}
      </ol>
    </div>
  );
}

export default function CLCN2Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/clcn2/overview`).then(r => r.json()),
      fetch(`${API}/api/clcn2/breakdown`).then(r => r.json()),
      fetch(`${API}/api/clcn2/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  const tabContent = [
    <OverviewTab data={overview} />,
    <PatientsTab data={overview} />,
    <SeizuresTab data={overview} />,
    <TreatmentsTab data={overview} />,
    <DefinitionsTab data={defs} />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3">
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: C }}>
            🧬 CLCN2 Epilepsy — GGE / JME / CAE / GTCS-Alone
          </h4>
          <div className="text-muted small">
            CLC-2 Voltage-Gated Chloride Channel · GOF → GGE Spectrum ·
            Precision: Acetazolamide (CA inhibitor) · 3q26.1 · AD ~65% penetrance
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: C, borderBottomColor: C } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      <div>{tabContent[tab]}</div>
    </div>
  );
}
