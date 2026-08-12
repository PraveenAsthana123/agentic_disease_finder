'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Variants', 'Seizure Types & Monitoring', 'Treatments', 'Definitions'];

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

function Bar({ label, value, max, color = '#3b82f6' }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview…</div>;
  const maxVar = Math.max(...(data.variant_distribution || []).map(v => v.count), 1);
  const maxTand = Math.max(...(data.tand_prevalence || []).map(t => t.count), 1);
  const maxTx = Math.max(...(data.treatment_use || []).map(t => t.n_patients), 1);
  const maxTuber = Math.max(...(data.tuber_histogram || []).map(t => t.count), 1);

  return (
    <div>
      <div className="alert alert-warning py-2 small mb-3">
        <strong>mTOR-driven epilepsy:</strong> Tuberous Sclerosis Complex (TSC) — autosomal dominant, 1:6,000 births.
        TSC1/TSC2 mutations → mTOR hyperactivation → cortical tubers → pharmacoresistant epilepsy in 80–90%.
        Infantile spasms (35–50%) require <em>urgent treatment within 2 weeks</em> for best developmental outcome.
      </div>

      <div className="row mb-4">
        {(data.kpis || []).map(k => <KPI key={k.label} {...k} />)}
      </div>

      <div className="row">
        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">TSC Gene Distribution</div>
            <div className="card-body">
              {(data.variant_distribution || []).map(v => (
                <Bar key={v.gene} label={`${v.gene} (${v.pct}%)`} value={v.count} max={maxVar} />
              ))}
              <div className="text-muted small mt-2">TSC2 generally more severe phenotype</div>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Tuber Count Distribution</div>
            <div className="card-body">
              {(data.tuber_histogram || []).map(t => (
                <Bar key={t.range} label={`${t.range} tubers`} value={t.count} max={maxTuber} color="#ef4444" />
              ))}
              <div className="text-muted small mt-2">Avg: {data.avg_tuber_count} tubers/patient</div>
            </div>
          </div>
        </div>

        <div className="col-md-4 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Seizure Types (Prevalence)</div>
            <div className="card-body">
              {(data.seizure_type_distribution || []).map(s => (
                <Bar key={s.type} label={s.type.split('(')[0].trim()} value={s.prevalence_pct} max={100} color="#8b5cf6" />
              ))}
              <div className="text-muted small mt-2">% of TSC patients affected</div>
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">TAND Prevalence (Neuropsychiatric)</div>
            <div className="card-body">
              {(data.tand_prevalence || []).map(t => (
                <Bar key={t.feature} label={t.feature} value={t.count} max={maxTand} color="#f59e0b" />
              ))}
            </div>
          </div>
        </div>

        <div className="col-md-6 mb-3">
          <div className="card h-100 shadow-sm">
            <div className="card-header fw-bold">Everolimus Trough Distribution (target 5–15 ng/mL)</div>
            <div className="card-body">
              {(data.everolimus_trough_distribution || []).map(e => (
                <Bar key={e.bin} label={e.bin} value={e.count}
                  max={Math.max(...(data.everolimus_trough_distribution || []).map(x => x.count), 1)}
                  color={e.bin.includes('therapeutic)') || e.bin.includes('supra') ? '#ef4444' : '#10b981'} />
              ))}
              <div className="text-muted small mt-2">{data.n_on_everolimus} patients on Everolimus</div>
            </div>
          </div>
        </div>

        <div className="col-md-12 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Current Treatment Use</div>
            <div className="card-body">
              <div className="row">
                {(data.treatment_use || []).map(t => (
                  <div key={t.drug} className="col-6 col-md-3 mb-2">
                    <Bar label={t.drug} value={t.n_patients} max={maxTx} color="#6366f1" />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="text-muted small mt-2">
        <strong>Source:</strong> {data.reference}
      </div>
    </div>
  );
}

function PatientsTab({ data }) {
  const [filter, setFilter] = useState('all');
  const [search, setSearch] = useState('');
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;

  let patients = data.patient_table || [];
  if (filter === 'is') patients = patients.filter(p => p.infantile_spasms === 'Yes');
  if (filter === 'everolimus') patients = patients.filter(p => p.on_everolimus === 'Yes');
  if (filter === 'tsc2') patients = patients.filter(p => p.gene === 'TSC2');
  if (filter === 'responder') patients = patients.filter(p => p.responder === 'Yes');
  if (search) patients = patients.filter(p =>
    p.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    p.variant.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <div className="row mb-3">
        <div className="col-md-8">
          <div className="btn-group flex-wrap" role="group">
            {['all', 'is', 'everolimus', 'tsc2', 'responder'].map(f => (
              <button key={f} className={`btn btn-sm ${filter === f ? 'btn-primary' : 'btn-outline-secondary'}`}
                onClick={() => setFilter(f)}>
                {f === 'all' ? 'All Patients' : f === 'is' ? 'Infantile Spasms' :
                  f === 'everolimus' ? 'On Everolimus' : f === 'tsc2' ? 'TSC2' : 'Responders'}
              </button>
            ))}
          </div>
        </div>
        <div className="col-md-4">
          <input className="form-control form-control-sm" placeholder="Search patient / variant…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>

      <div className="table-responsive mb-4" style={{ maxHeight: 420, overflowY: 'auto' }}>
        <table className="table table-sm table-striped table-hover small">
          <thead className="table-dark sticky-top">
            <tr>
              <th>Patient</th><th>Age</th><th>Gene</th><th>Variant</th>
              <th>Tubers</th><th>IS</th><th>Sz/Mo</th><th>Everolimus</th>
              <th>Trough</th><th>Regimen</th><th>TAND</th><th>Responder</th>
            </tr>
          </thead>
          <tbody>
            {patients.map(p => (
              <tr key={p.patient_id}>
                <td><code>{p.patient_id}</code></td>
                <td>{p.age}</td>
                <td><span className={`badge ${p.gene === 'TSC2' ? 'bg-danger' : 'bg-primary'}`}>{p.gene}</span></td>
                <td><span className="badge bg-secondary small" title={p.variant}>{p.variant.split(' (')[0]}</span></td>
                <td>{p.tuber_count}</td>
                <td>{p.infantile_spasms === 'Yes' ? <span className="badge bg-warning text-dark">Yes</span> : '—'}</td>
                <td>{p.seizures_per_month}</td>
                <td>{p.on_everolimus === 'Yes' ? <span className="badge bg-success">Yes</span> : '—'}</td>
                <td>{p.everolimus_trough ? `${p.everolimus_trough} ng/mL` : '—'}</td>
                <td><small>{p.current_regimen}</small></td>
                <td><small>{p.tand_features}</small></td>
                <td>{p.responder === 'Yes' ? <span className="badge bg-success">Yes</span> : <span className="badge bg-secondary">No</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="row">
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">TSC1/TSC2 Variant Catalog</div>
            <div className="card-body p-0">
              <table className="table table-sm small mb-0">
                <thead className="table-light"><tr><th>Class</th><th>Example</th><th>%</th><th>mTOR Effect</th></tr></thead>
                <tbody>
                  {(data.variant_catalog || []).map(v => (
                    <tr key={v.class}>
                      <td><strong>{v.class}</strong></td>
                      <td><code>{v.example}</code></td>
                      <td>{v.pct}%</td>
                      <td className="text-muted">{v.mtor_effect}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6 mb-3">
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Tuber Type Catalog</div>
            <div className="card-body p-0">
              <table className="table table-sm small mb-0">
                <thead className="table-light"><tr><th>Type</th><th>MRI</th><th>Epileptogenicity</th><th>Surgery</th></tr></thead>
                <tbody>
                  {(data.tuber_type_catalog || []).map(t => (
                    <tr key={t.type}>
                      <td><strong>{t.type}</strong> <span className="badge bg-secondary">{t.prevalence_pct}%</span></td>
                      <td className="text-muted small">{t.mri_features}</td>
                      <td>{t.epileptogenicity.split(' — ')[0]}</td>
                      <td className="text-muted small">{t.surgical_outcome.split(';')[0]}</td>
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

function SeizureTab({ data }) {
  const [open, setOpen] = useState(null);
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;

  return (
    <div>
      <h6 className="fw-bold mb-3">Seizure Types in TSC</h6>
      {(data.seizure_type_catalog || []).map((s, i) => (
        <div key={s.type} className="card mb-2 shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer' }} onClick={() => setOpen(open === i ? null : i)}>
            <span>
              <strong>{s.type}</strong>
              <span className="badge bg-primary ms-2">{s.prevalence_pct}% of TSC</span>
            </span>
            <span>{open === i ? '▲' : '▼'}</span>
          </div>
          {open === i && (
            <div className="card-body small">
              <div className="row">
                <div className="col-md-3"><strong>Onset:</strong> {s.onset}</div>
                <div className="col-md-5"><strong>EEG:</strong> {s.eeg}</div>
                <div className="col-md-4"><strong>Treatment Priority:</strong> {s.treatment_priority}</div>
              </div>
              <div className="alert alert-info py-1 mt-2 small">{s.outcome_note}</div>
            </div>
          )}
        </div>
      ))}

      <h6 className="fw-bold mb-3 mt-4">AED Monitoring Requirements</h6>
      {(data.aed_monitoring || []).map(a => (
        <div key={a.aed} className="card mb-2 shadow-sm">
          <div className="card-body py-2">
            <div className="d-flex justify-content-between mb-1">
              <strong>{a.aed}</strong>
              <span className={`badge ${a.category === 'REQUIRED MONITORING' ? 'bg-danger' :
                a.category === 'DRUG LEVEL MONITORING' ? 'bg-warning text-dark' :
                a.category === 'LFT MONITORING' ? 'bg-warning text-dark' :
                a.category === 'CARDIAC MONITORING' ? 'bg-danger' : 'bg-secondary'}`}>
                {a.category}
              </span>
            </div>
            <div className="small text-muted mb-1"><strong>Risk:</strong> {a.risk}</div>
            <div className="small mb-1"><strong>Monitoring:</strong> {a.monitoring}</div>
            <div className="small"><strong>Evidence:</strong> {a.evidence}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function TreatmentsTab({ data }) {
  const [open, setOpen] = useState(0);
  if (!data) return <div className="text-center py-4 text-muted">Loading breakdown…</div>;

  return (
    <div>
      <div className="alert alert-success py-2 small mb-3">
        <strong>FDA-Approved TSC Therapies:</strong> Everolimus (EXIST-3 2018) — mTOR inhibitor;
        Cannabidiol / Epidiolex (GWPCARE6 2018) — TSC-specific indication.
        Vigabatrin: first-line for infantile spasms (FDA-approved 2009, IS indication).
      </div>
      {(data.treatment_catalog || []).map((t, i) => (
        <div key={t.drug} className="card mb-2 shadow-sm">
          <div className="card-header d-flex justify-content-between align-items-center"
            style={{ cursor: 'pointer' }} onClick={() => setOpen(open === i ? null : i)}>
            <span>
              <strong>{t.drug}</strong>
              <span className={`badge ms-2 ${t.fda_status.includes('FDA-approved') ? 'bg-success' :
                t.fda_status.includes('Level A') ? 'bg-primary' : 'bg-secondary'}`}>
                {t.fda_status.split('(')[0].trim()}
              </span>
              {t.year && <span className="badge bg-light text-dark border ms-1">{t.year}</span>}
            </span>
            <span>{open === i ? '▲' : '▼'}</span>
          </div>
          {open === i && (
            <div className="card-body small">
              <div className="mb-2"><strong>Dose:</strong> {t.dose}</div>
              <div className="mb-2"><strong>Mechanism:</strong> {t.moa}</div>
              <div className="mb-2 text-success"><strong>Efficacy:</strong> {t.efficacy}</div>
              <div className="text-danger"><strong>Safety:</strong> {t.safety}</div>
            </div>
          )}
        </div>
      ))}

      <h6 className="fw-bold mt-4 mb-3">Developmental Trajectory &amp; TAND</h6>
      <div className="table-responsive">
        <table className="table table-sm table-bordered small">
          <thead className="table-dark">
            <tr><th>Age Window</th><th>Expected</th><th>TSC / TAND Pattern</th></tr>
          </thead>
          <tbody>
            {(data.developmental_trajectory || []).map(m => (
              <tr key={m.age_window}>
                <td><strong>{m.age_window}</strong></td>
                <td className="text-muted">{m.expected}</td>
                <td>{m.tsc_pattern}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading definitions…</div>;
  return (
    <div>
      <h6 className="fw-bold mb-3">Core Concepts</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped small">
          <thead className="table-dark"><tr><th style={{ width: '220px' }}>Term</th><th>Definition</th></tr></thead>
          <tbody>
            {(data.concepts || []).map(c => (
              <tr key={c.term}>
                <td><strong>{c.term}</strong></td>
                <td>{c.definition}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3">Standards &amp; Guidelines</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped small">
          <thead className="table-dark"><tr><th>Standard</th><th>Scope</th></tr></thead>
          <tbody>
            {(data.standards || []).map(s => (
              <tr key={s.name}><td><strong>{s.name}</strong></td><td>{s.scope}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-3">Key Thresholds</h6>
      <div className="table-responsive mb-4">
        <table className="table table-sm table-striped small">
          <thead className="table-dark">
            <tr><th>Metric</th><th>Target</th><th>Action Below / Issue</th><th>Action Above</th></tr>
          </thead>
          <tbody>
            {(data.thresholds || []).map(t => (
              <tr key={t.metric}>
                <td><strong>{t.metric}</strong></td>
                <td><span className="badge bg-success">{t.target}</span></td>
                <td className="text-danger small">{t.action_below}</td>
                <td className="text-muted small">{t.action_above}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h6 className="fw-bold mb-2">References</h6>
      <ul className="small text-muted">
        {(data.references || []).map((r, i) => <li key={i}>{r}</li>)}
      </ul>
    </div>
  );
}

export default function TSCDashboard() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/tsc/overview`).then(r => r.json()),
      fetch(`${API}/api/tsc/breakdown`).then(r => r.json()),
      fetch(`${API}/api/tsc/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOverview(o); setBreakdown(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger">Error: {err}</div>;

  return (
    <div>
      <div className="d-flex justify-content-between align-items-center mb-3">
        <div>
          <h3>&#x1f9ec; Tuberous Sclerosis Complex (TSC)</h3>
          <div className="text-muted small">
            mTOR-driven epilepsy · TSC1/TSC2 mutations · Infantile spasms · Everolimus (EXIST-3 2018) · Epidiolex (GWPCARE6 2018)
            {overview && ` · ${overview.total_patients} patients`}
          </div>
        </div>
        <span className="badge bg-warning text-dark fs-6">mTOR</span>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Patients & Variants' && <PatientsTab data={breakdown} />}
      {tab === 'Seizure Types & Monitoring' && <SeizureTab data={breakdown} />}
      {tab === 'Treatments' && <TreatmentsTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={defs} />}
    </div>
  );
}
