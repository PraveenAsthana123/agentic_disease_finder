'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Etiologies', 'Seizures & Triggers', 'Treatments', 'Definitions'];
const COLOR = '#00897b'; // teal — distinguishes Cav3.1 TC-neuron identity from CACNA1H amber

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
  const ciSummary = data.contraindications_summary || [];
  const maxEtio = Math.max(...etiologies.map(e => e.pct || 0), 1);

  return (
    <div>
      <div className="alert alert-info py-2 small mb-3 border" style={{ borderColor: COLOR }}>
        <strong>⚡ CACNA1G (17q21.33) — Cav3.1 T-type Ca²⁺ Channel — GGE / CAE / JME:</strong>{' '}
        <strong>Kim 2001 Nature:</strong> CACNA1G KO mice lack TC-neuron LTCS and are fully protected from absence seizures.
        Cav3.1 is the PRIMARY T-type channel driving thalamic relay (TC) low-threshold Ca²⁺ spikes.{' '}
        GOF variants enlarge the <strong>window current</strong> → enhanced 3-Hz burst-pause → SWD.{' '}
        <strong>PRECISION:</strong> Ethosuximide (ETX, Level A) — direct Cav3.1 block; HV abolition = ETX adequacy marker.{' '}
        <span className="text-danger fw-bold">
          ABSOLUTE CI: CBZ / OXC / PHT (GGE aggravation → status) · Tiagabine (NCSE). VPA females: VPPP mandatory. POLG1 screen before VPA.
        </span>
      </div>

      <div className="row g-2 mb-4">
        <KPI label="Total Patients" value={kpis.n_patients} color={COLOR} />
        <KPI label="Seizure-Free" value={`${kpis.seizure_free_pct}%`} color="#198754" />
        <KPI label="Drug-Resistant" value={`${kpis.drug_resistant_pct}%`} color="#dc3545" />
        <KPI label="On ETX" value={kpis.on_etx_n} color={COLOR} />
        <KPI label="HV-SWD Abolished" value={kpis.hv_swd_abolished_n} color="#198754" />
        <KPI label="Photo-Sensitive" value={kpis.photosensitive_n} color="#fd7e14" />
        <KPI label="GTCS Risk" value={kpis.gtcs_n} color="#dc3545" />
        <KPI label="Catamenial" value={kpis.catamenial_n} color="#6f42c1" />
        <KPI label="SUDEP High Risk" value={kpis.sudep_high_risk_n} color="#dc3545" />
        <KPI label="Avg Age (yr)" value={kpis.avg_age_years} color="#343a40" />
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: COLOR }}>
              Etiology Distribution (n=40)
            </div>
            <div className="card-body">
              {etiologies.map(e => (
                <Bar key={e.category} label={e.category.replace('GOF-', '')} value={e.pct} max={maxEtio} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold text-white py-2" style={{ background: '#c0392b' }}>
              Contraindications (Absolute / High Risk)
            </div>
            <div className="card-body">
              {ciSummary.map(ci => (
                <div key={ci.drug} className="mb-2 small border-start border-danger ps-2">
                  <span className="fw-semibold text-danger">{ci.drug.split(' (')[0]}</span>
                  <div className="text-muted">{ci.risk}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold py-2" style={{ background: '#1a5276', color: '#fff' }}>
              Key Monitoring Items
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0 small">
                <thead><tr><th>Item</th><th>Frequency</th></tr></thead>
                <tbody>
                  {monitoring.map(m => (
                    <tr key={m.item}>
                      <td>{m.item}</td>
                      <td className="text-muted">{m.frequency}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-semibold py-2" style={{ background: '#117a65', color: '#fff' }}>
              Treatment Ladder
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0 small">
                <thead><tr><th>Drug</th><th>Level</th><th>Indication</th></tr></thead>
                <tbody>
                  {treatments.map(t => (
                    <tr key={t.drug}>
                      <td className="fw-semibold">{t.drug.split(' (')[0].split(' /')[0]}</td>
                      <td><span className={`badge ${t.level.startsWith('Level A') ? 'bg-success' : t.level.startsWith('Level B') ? 'bg-warning text-dark' : 'bg-secondary'}`}>{t.level.split('—')[0].trim()}</span></td>
                      <td className="text-muted">{t.indication.split(';')[0]}</td>
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
            <div className="card-header fw-semibold py-2" style={{ background: '#6c3483', color: '#fff' }}>
              Lifecycle Windows
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0 small">
                <thead><tr><th>Window</th><th>Phase</th><th>Key Risk</th></tr></thead>
                <tbody>
                  {lifecycle.map(lc => (
                    <tr key={lc.window}>
                      <td className="fw-semibold">{lc.window}</td>
                      <td>{lc.phase}</td>
                      <td className="text-danger">{lc.key_risk}</td>
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
          <div className="card shadow-sm border-warning">
            <div className="card-header fw-semibold py-2 bg-warning text-dark">
              Thresholds &amp; Action Triggers
            </div>
            <div className="card-body p-0">
              <table className="table table-sm table-hover mb-0 small">
                <thead><tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr></thead>
                <tbody>
                  {thresholds.map(t => (
                    <tr key={t.name}>
                      <td className="fw-semibold">{t.name}</td>
                      <td><code>{t.value}</code></td>
                      <td className="text-muted">{t.action}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {data.clinical_alert && (
        <div className="alert alert-info py-2 small mt-2">
          <strong>📋 Clinical Summary:</strong> {data.clinical_alert}
        </div>
      )}
      {data.gene_family_note && (
        <div className="alert alert-secondary py-2 small mt-2">
          <strong>🔗 Gene Family Note:</strong> {data.gene_family_note}
        </div>
      )}
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading patients…</div>;
  const etiologies = data.etiologies || [];
  const patients = data.patients || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3">Etiology Catalog — CACNA1G GGE (5 classes)</h6>
      {etiologies.map(e => (
        <div key={e.category} className="card mb-3 shadow-sm border-start border-4" style={{ borderColor: COLOR }}>
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{ background: '#e8f5e9' }}>
            <span className="fw-semibold">{e.category.replace('GOF-', '')}</span>
            <span className="badge text-white" style={{ background: COLOR }}>{e.pct}% of cohort</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><strong>Etiology:</strong> {e.etiology}</div>
            <div className="mb-1 text-muted">{e.mechanism}</div>
            <div className="mb-1"><strong>Typical variants:</strong> <code>{e.typical_variants}</code></div>
            <div><strong>Onset:</strong> ~{e.onset_age_years}Y &nbsp;|&nbsp; <strong>Outcome:</strong> {e.outcome}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-3">Patient Cohort (n=40)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-dark">
            <tr>
              <th>#</th><th>Name</th><th>Age/Sex</th><th>Etiology</th>
              <th>Onset</th><th>Seizure Types</th><th>Drug</th>
              <th>SF</th><th>DR</th><th>HV-OK</th><th>SUDEP</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.id}>
                <td>{p.id}</td>
                <td>{p.name}</td>
                <td>{p.age}y/{p.sex}</td>
                <td><span className="badge text-white" style={{ background: COLOR }}>{p.etiology.replace('GOF-','')}</span></td>
                <td>{p.onset_age}y</td>
                <td>{(p.seizure_types || []).join(', ')}</td>
                <td><code>{p.primary_drug}</code></td>
                <td>{p.seizure_free ? '✅' : '❌'}</td>
                <td>{p.drug_resistant ? '🔴' : '—'}</td>
                <td>{p.hv_swd_abolished ? '✅' : p.etx_levels_ok ? '⚠️' : '—'}</td>
                <td>
                  <span className={`badge ${p.sudep_risk === 'high' ? 'bg-danger' : p.sudep_risk === 'moderate' ? 'bg-warning text-dark' : 'bg-success'}`}>
                    {p.sudep_risk}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function SeizuresTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading seizures…</div>;
  const seizureTypes = data.seizure_types || [];
  const triggers = data.triggers || [];
  const freqDist = data.seizure_frequency_distribution || {};
  const etxMetrics = data.etx_precision_metrics || {};

  return (
    <div>
      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold py-2" style={{ background: '#1a5276', color: '#fff' }}>
              Seizure Type Frequencies (n=40 patients)
            </div>
            <div className="card-body">
              {Object.entries(freqDist).map(([type, count]) => (
                <Bar key={type} label={`${type} (${count}pts)`} value={Math.round(count/40*100)} max={100} color={COLOR} />
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm">
            <div className="card-header fw-semibold py-2" style={{ background: '#117a65', color: '#fff' }}>
              ETX Cav3.1 Precision Metrics
            </div>
            <div className="card-body">
              <div className="mb-2 small">
                <div className="d-flex justify-content-between"><span>On ETX (any)</span><strong>{etxMetrics.on_etx} pts</strong></div>
                <div className="d-flex justify-content-between"><span>ETX levels therapeutic (40–100 mg/L)</span><strong>{etxMetrics.etx_levels_therapeutic} pts</strong></div>
                <div className="d-flex justify-content-between"><span>HV-SWD abolished on ETX</span><strong>{etxMetrics.hv_swd_abolished_on_etx} pts</strong></div>
              </div>
              <div className="alert alert-info py-2 small mt-2">
                <strong>Cav3.1 precision note:</strong> {etxMetrics.cacna1g_precision_note}
              </div>
              <div className="alert alert-warning py-2 small mt-1">
                <strong>ETX adequacy marker:</strong> {etxMetrics.etx_adequacy_note}
              </div>
            </div>
          </div>
        </div>
      </div>

      <h6 className="fw-semibold mb-3">Seizure Types — Clinical Detail</h6>
      {seizureTypes.map(s => (
        <div key={s.type} className="card mb-3 shadow-sm">
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{ background: '#f1f8e9' }}>
            <span className="fw-semibold">{s.type}</span>
            <span className="badge bg-primary">{s.pct}% of patients</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><strong>EEG:</strong> {s.eeg}</div>
            <div className="mb-1"><strong>Semiology:</strong> {s.semiology}</div>
            <div className="text-success"><strong>💡 Clinical tip:</strong> {s.clinical_tip}</div>
          </div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-3">Seizure Triggers</h6>
      <div className="row g-2 mb-3">
        {triggers.map(t => (
          <div key={t.trigger} className="col-12">
            <div className="d-flex justify-content-between align-items-center mb-1 small">
              <strong>{t.trigger}</strong><span>{t.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 10 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, background: COLOR }} />
            </div>
            <div className="text-muted small mb-2">{t.note}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading treatments…</div>;
  const treatments = data.treatments || [];
  const contraindications = data.contraindications || [];

  return (
    <div>
      <h6 className="fw-semibold mb-3">Treatment Catalog — CACNA1G GGE</h6>
      {treatments.map(t => (
        <div key={t.drug} className="card mb-3 shadow-sm border-start border-4" style={{ borderColor: '#27ae60' }}>
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{ background: '#eafaf1' }}>
            <span className="fw-semibold">{t.drug}</span>
            <span className={`badge ${t.level.startsWith('Level A') ? 'bg-success' : t.level.startsWith('Level B') ? 'bg-warning text-dark' : 'bg-secondary'}`}>{t.level.split('—')[0].trim()}</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><strong>Indication:</strong> {t.indication}</div>
            <div className="mb-1"><strong>Dose:</strong> {t.dose}</div>
            <div className="mb-1"><strong>Mechanism:</strong> {t.moa}</div>
            <div className="mb-1"><strong>Efficacy:</strong> {t.efficacy}</div>
            <div className="mb-1"><strong>Safety:</strong> {t.safety}</div>
            <div className="mb-1"><strong>Monitoring:</strong> {t.monitoring}</div>
            {t.cacna1g_note && (
              <div className="alert py-1 mb-0 mt-1 small" style={{ background: '#e0f7fa', borderColor: COLOR }}>
                <strong>⚡ CACNA1G note:</strong> {t.cacna1g_note}
              </div>
            )}
          </div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-3 text-danger">Contraindications</h6>
      {contraindications.map(ci => (
        <div key={ci.drug} className="card mb-3 shadow-sm border-start border-4 border-danger">
          <div className="card-header py-2 d-flex justify-content-between align-items-center" style={{ background: '#fdedec' }}>
            <span className="fw-semibold text-danger">{ci.drug}</span>
            <span className="badge bg-danger">{ci.risk.split('—')[0].trim()}</span>
          </div>
          <div className="card-body py-2 small">
            <div className="mb-1"><strong>Mechanism:</strong> {ci.mechanism}</div>
            <div className="text-danger"><strong>Management:</strong> {ci.management}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  const defs = data.definitions || [];
  const thresholds = data.thresholds || [];
  const standards = data.standards || [];
  const geneSum = data.gene_summary || {};

  return (
    <div>
      <div className="card mb-4 shadow-sm" style={{ borderColor: COLOR }}>
        <div className="card-header fw-semibold py-2 text-white" style={{ background: COLOR }}>CACNA1G Gene Summary</div>
        <div className="card-body small">
          <div className="row g-2">
            {Object.entries(geneSum).map(([k, v]) => (
              <div key={k} className="col-md-6">
                <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong>{' '}
                <span>{Array.isArray(v) ? v.join(', ') : String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <h6 className="fw-semibold mb-3">Concepts &amp; Definitions (15)</h6>
      {defs.map(d => (
        <div key={d.term} className="card mb-2 shadow-sm">
          <div className="card-header py-2 fw-semibold small" style={{ background: '#e0f7fa' }}>
            {d.term}
          </div>
          <div className="card-body py-2 small text-muted">{d.definition}</div>
        </div>
      ))}

      <h6 className="fw-semibold mt-4 mb-3">Evidence Standards (12)</h6>
      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead className="table-dark"><tr><th>Code</th><th>Standard</th><th>Relevance</th></tr></thead>
          <tbody>
            {standards.map(s => (
              <tr key={s.code}>
                <td><code>{s.code}</code></td>
                <td>{s.title}</td>
                <td className="text-muted">{s.relevance}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function CACNA1GPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [o, b, d] = await Promise.all([
          fetch(`${API}/api/cacna1g/overview`).then(r => r.json()),
          fetch(`${API}/api/cacna1g/breakdown`).then(r => r.json()),
          fetch(`${API}/api/cacna1g/definitions`).then(r => r.json()),
        ]);
        setOverview(o);
        setBreakdown(b);
        setDefinitions(d);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2 text-muted">Loading CACNA1G dashboard…</p></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3 flex-wrap">
        <h4 className="mb-0 fw-bold">
          ⚡ CACNA1G Epilepsy
        </h4>
        <span className="badge text-white fs-6" style={{ background: COLOR }}>GGE / CAE / JME / GEFS+</span>
        <span className="badge bg-secondary">Cav3.1 T-type Ca²⁺ Channel</span>
        <span className="badge bg-info text-dark">17q21.33 · AD GOF · ETX Precision</span>
      </div>

      <div className="alert alert-light border small py-2 mb-3">
        <strong>CACNA1G (17q21.33)</strong> encodes <strong>Cav3.1</strong>, the dominant T-type (low-voltage-activated) Ca²⁺ channel
        in thalamic relay (TC) neurons. <strong>Kim et al. 2001 Nature:</strong> CACNA1G KO mice completely lack TC-neuron LTCS and
        are fully protected from absence seizures — establishing Cav3.1 as the PRIMARY driver of thalamo-cortical 3-Hz oscillation.
        GOF variants enlarge the Cav3.1 <strong>window current</strong> → enhanced rebound LTCS → stronger 3-Hz SWD → GGE/CAE.
        Precision: <strong>Ethosuximide (Level A)</strong> — primary Cav3.1 TC-current blocker (Coulter 1989); HV abolition is the clinical ETX adequacy marker.
        <span className="text-danger fw-bold ms-2">ABSOLUTE CI: CBZ/OXC/PHT · Tiagabine.</span>
      </div>

      <ul className="nav nav-tabs mb-3 flex-wrap">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <SeizuresTab data={breakdown} />}
      {tab === 3 && <TreatmentsTab data={breakdown} />}
      {tab === 4 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
