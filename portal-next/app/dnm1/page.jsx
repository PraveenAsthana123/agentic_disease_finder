'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];

// Teal/cyan theme — endocytosis vesicle recycling identity
const C = '#0f6b73';
const CL = '#e0f5f6';

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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const etioDist = data.etiology_distribution || {};
  const etiologies = Object.entries(etioDist).map(([cat, n]) => ({ cat, n }));
  const maxN = Math.max(...etiologies.map(e => e.n), 1);

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3" style={{ borderLeft: `4px solid ${C}` }}>
        <strong>DNM1 (9q34.11) — Dynamin-1 GTPase · Synaptic Vesicle Endocytosis · DEE31:</strong>{' '}
        DNM1 encodes Dynamin-1, the rate-limiting GTPase for{' '}
        <strong>clathrin-mediated synaptic vesicle endocytosis (membrane fission)</strong>.{' '}
        LOF → vesicle pool depletion at high-frequency synapses. PV+ interneurons (100–600 Hz) most vulnerable →{' '}
        <strong>cortical disinhibition → early infantile spasms / DEE31</strong>.{' '}
        Dominant negative p.R237W: mutant protomers trap WT dynamin → >80% functional LOF.{' '}
        <em>No precision therapy available (2025) — ACTH + KD (early escalation mandatory).</em>{' '}
        <span className="text-danger fw-bold">
          ABSOLUTE CI: Tiagabine (NCSE) · VPA + POLG1 (Alpers fatal).
          HIGH CAUTION: CBZ/OXC myoclonic aggravation · VGB long-term (VFD).
          POLG1 MANDATORY before VPA. KD early (do not wait for 3 AED failures).
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Cohort Size" value={data.cohort_size} color={C} />
        <KPI label="Spasm-Free" value={`${data.seizure_free_pct}%`} color="#198754" />
        <KPI label="pLI Score" value={data.pli} color={C} />
        <KPI label="Inheritance" value="AD de novo" color="#0d6efd" />
        <KPI label="De Novo" value=">95%" color="#e83e8c" />
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
                    <span>{e.cat}</span>
                    <span className="badge" style={{ background: C }}>{e.n} pts</span>
                  </div>
                  <div className="progress" style={{ height: 10 }}>
                    <div className="progress-bar"
                      style={{ width: `${Math.round(e.n / maxN * 100)}%`, background: C }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
              Gene at a Glance
            </div>
            <div className="card-body small">
              <table className="table table-sm table-borderless mb-0">
                <tbody>
                  <tr><td className="fw-semibold text-nowrap">Gene</td><td>{data.gene}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">Locus</td><td>{data.locus}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">OMIM</td><td>{data.omim}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">Protein</td><td>{data.protein}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">Inheritance</td><td>{data.inheritance}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">pLI</td><td>{data.pli}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">Top Trigger</td><td>{data.top_trigger}</td></tr>
                  <tr><td className="fw-semibold text-nowrap">Top Treatment</td><td>{data.top_treatment}</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Mechanism — DNM1 / Dynamin-1 / Synaptic Vesicle Endocytosis / PV+ Interneuron Vulnerability
        </div>
        <div className="card-body small text-muted">
          {data.mechanism}
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          ⭐ Key Clinical Pearl
        </div>
        <div className="card-body small">
          {data.key_clinical_pearl}
        </div>
      </div>

      <div className="card shadow-sm border-danger mb-3">
        <div className="card-header fw-semibold text-danger py-2">⛔ Contraindication Summary</div>
        <div className="card-body small text-danger">
          {data.key_contraindication}
        </div>
      </div>

      <div className="card shadow-sm mb-3">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Key References
        </div>
        <div className="card-body p-0">
          {(data.key_references || []).map((r, i) => (
            <div key={i} className="border-bottom px-3 py-1 small text-muted">{r}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const etiologies = data.etiology_catalog || [];
  const patients = data.patients_sample || [];
  const stats = data.summary_stats || {};

  return (
    <div>
      <div className="row g-3 mb-4">
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: C }}>{stats.seizure_free}</div>
              <div className="small text-muted">Spasm-Free (of 40)</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4 text-danger">{stats.etiology_classes}</div>
              <div className="small text-muted">Etiology Classes</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: C }}>{stats.seizure_types_count}</div>
              <div className="small text-muted">Seizure Types</div>
            </div>
          </div>
        </div>
        <div className="col-md-3">
          <div className="card text-center shadow-sm">
            <div className="card-body py-2">
              <div className="fw-bold fs-4" style={{ color: C }}>{stats.treatments_count}</div>
              <div className="small text-muted">Treatments</div>
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Etiology Catalog — DNM1 (5 Classes)
        </div>
        <div className="card-body p-0">
          {etiologies.map((e, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-semibold small">{e.etiology || e.category}</span>
                <span className="badge" style={{ background: C }}>{e.pct}%</span>
              </div>
              <div className="small text-muted mb-1">{e.mechanism}</div>
              <div className="small"><span className="fw-semibold">Typical Variants: </span>{e.typical_variants}</div>
              <div className="small text-primary"><span className="fw-semibold">Outcome: </span>{e.outcome}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Patient Sample (n=15 of 40)
        </div>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0 small">
            <thead className="table-light">
              <tr>
                <th>ID</th><th>Name</th><th>Sex</th><th>Age Dx (m)</th><th>Etiology</th>
                <th>Variant</th><th>Spasm-Free</th><th>Treatment</th>
                <th>POLG1</th><th>VPPP</th>
              </tr>
            </thead>
            <tbody>
              {patients.map((p, i) => (
                <tr key={i}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.name}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_at_diagnosis_months}m</td>
                  <td><span className="badge bg-secondary text-wrap" style={{ fontSize: '0.65rem', maxWidth: 120 }}>{p.etiology_class}</span></td>
                  <td><code className="small">{p.variant}</code></td>
                  <td>{p.seizure_free ? <span className="text-success fw-bold">✅</span> : <span className="text-danger">🔴</span>}</td>
                  <td className="small">{p.current_treatment}</td>
                  <td><span className={p.polg1_tested ? 'text-success fw-bold' : 'text-danger fw-bold'}>{p.polg1_tested ? '✅' : '❌'}</span></td>
                  <td><span className={p.vppp_enrolled ? 'text-success' : 'text-muted'}>{p.sex === 'F' ? (p.vppp_enrolled ? '✅' : '⚠️') : '—'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const seizures = data.seizure_types || [];
  const triggers = data.triggers || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Seizure Types — DNM1 DEE31 / West Syndrome / LGS Spectrum
        </div>
        <div className="card-body p-0">
          {seizures.map((s, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fw-bold">{s.type}</span>
                <span className="badge" style={{ background: C }}>{s.pct}%</span>
              </div>
              <div className="progress mb-2" style={{ height: 8 }}>
                <div className="progress-bar" style={{ width: `${s.pct}%`, background: C }} />
              </div>
              <div className="small text-muted mb-1"><strong>Description:</strong> {s.description}</div>
              <div className="small text-muted mb-1"><strong>EEG:</strong> {s.eeg}</div>
              <div className="small text-muted mb-1"><strong>Semiology:</strong> {s.semiology}</div>
              <div className="small alert alert-warning py-1 px-2 mb-0"><strong>Clinical Tip:</strong> {s.clinical_tips}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Seizure Triggers — DNM1 DEE31 Specific
        </div>
        <div className="card-body">
          {triggers.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="d-flex justify-content-between small fw-semibold mb-1">
                <span>{t.trigger}</span>
                <span>{t.pct}%</span>
              </div>
              <div className="progress mb-1" style={{ height: 10 }}>
                <div className="progress-bar" style={{ width: `${t.pct}%`, background: i === 0 ? '#dc3545' : C }} />
              </div>
              <div className="small text-muted">{t.notes}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold text-white py-2" style={{ background: C }}>
          Treatment Arsenal — DNM1 DEE31 / West Syndrome / LGS Spectrum
        </div>
        <div className="card-body p-0">
          {treatments.map((t, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fw-bold">{t.drug}</span>
                <span className="badge text-bg-light text-dark border">{t.level}</span>
              </div>
              <div className="small text-muted mb-1"><strong>Indication:</strong> {t.indication}</div>
              <div className="row g-2 small">
                <div className="col-md-6">
                  <div className="text-muted"><strong>MOA:</strong> {t.moa}</div>
                  <div className="text-muted mt-1"><strong>Dose:</strong> {t.dose}</div>
                  <div className="text-muted mt-1"><strong>Efficacy:</strong> {t.efficacy}</div>
                </div>
                <div className="col-md-6">
                  <div className="text-muted"><strong>Safety:</strong> {t.safety}</div>
                  <div className="text-muted mt-1"><strong>Monitoring:</strong> {t.monitoring}</div>
                </div>
              </div>
              {t.dnm1_note && (
                <div className="small alert alert-primary py-1 px-2 mt-2 mb-0">
                  <strong>DNM1 Note:</strong> {t.dnm1_note}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="card shadow-sm border-danger">
        <div className="card-header fw-semibold text-danger py-2">
          ⛔ Contraindications — DNM1 DEE31 / West Syndrome / LGS
        </div>
        <div className="card-body p-0">
          {contraindications.map((c, i) => (
            <div key={i} className="border-bottom p-3">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="fw-bold text-danger">{c.drug}</span>
                <span className={`badge ${c.level && c.level.includes('ABSOLUTE') ? 'bg-danger' : c.level && c.level.includes('HIGH') ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                  {c.level}
                </span>
              </div>
              <div className="small text-muted mb-1">{c.reason}</div>
              {c.alternative && (
                <div className="small text-success"><strong>Alternative:</strong> {c.alternative}</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  const definitions = data.definitions || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const references = data.references || [];

  return (
    <div>
      <div className="card shadow-sm mb-4">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          15 Core Concepts — DNM1 / Dynamin-1 / SV Endocytosis / DEE31 / West Syndrome
        </div>
        <div className="card-body p-0">
          {definitions.map((d, i) => (
            <div key={i} className="border-bottom px-3 py-2">
              <span className="fw-semibold" style={{ color: C }}>{d.term}: </span>
              <span className="small text-muted">{d.definition}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: CL }}>
              Clinical Thresholds (12)
            </div>
            <div className="card-body p-0">
              {thresholds.map((t, i) => (
                <div key={i} className="border-bottom px-3 py-2 small">
                  <div className="fw-semibold">{t.name}</div>
                  <div className="text-primary">{t.value}</div>
                  <div className="text-muted">{t.action}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: CL }}>
              Clinical Standards (12)
            </div>
            <div className="card-body p-0">
              {standards.map((s, i) => (
                <div key={i} className="border-bottom px-3 py-2 small">
                  <div className="fw-semibold" style={{ color: C }}>{s.name}</div>
                  <div className="text-muted">{s.relevance}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card shadow-sm">
        <div className="card-header fw-semibold py-2" style={{ background: CL }}>
          Key References (6)
        </div>
        <div className="card-body p-0">
          {references.map((r, i) => (
            <div key={i} className="border-bottom px-3 py-2 small text-muted">
              <span className="badge me-2" style={{ background: C }}>{r.pmid}</span>
              {r.citation}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function DNM1Page() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/dnm1/overview`)
      .then(r => r.json())
      .then(setOverview)
      .catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (activeTab >= 1 && activeTab <= 3 && !breakdown) {
      Promise.all([
        fetch(`${API}/api/dnm1/breakdown`).then(r => r.json()),
      ]).then(([bd]) => setBreakdown(bd))
        .catch(e => setError(e.message));
    }
    if (activeTab === 4 && !definitions) {
      fetch(`${API}/api/dnm1/definitions`)
        .then(r => r.json())
        .then(setDefinitions)
        .catch(e => setError(e.message));
    }
  }, [activeTab]);

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="mb-1" style={{ color: C }}>
          🧬 DNM1 Epilepsy — DEE31 / Dynamin-1 GTPase / Synaptic Vesicle Endocytosis / PV+ Interneuron
        </h4>
        <p className="text-muted small mb-0">
          DNM1 (9q34.11) · Dynamin-1 870aa ~96kDa Large GTPase · Clathrin-Mediated SV Endocytosis Fission ·
          AD de novo &gt;95% · OMIM DEE31 #617107 · pLI ~0.98 ·
          ACTH + VGB Level-A (UKISS) · KD early escalation · 40-patient cohort · EuroEPINOMICS 2014
        </p>
      </div>

      {error && (
        <div className="alert alert-danger small py-2">
          Backend error: {error}
        </div>
      )}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <PatientsTab data={breakdown} />}
      {activeTab === 2 && <SeizuresTab data={breakdown} />}
      {activeTab === 3 && <TreatmentsTab data={breakdown} />}
      {activeTab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
