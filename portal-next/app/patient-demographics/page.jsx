'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8010';

function kpiCard(label, value, sub, color = '#0b1f3a') {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3" key={label}>
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-3">
          <div className="fw-bold fs-4" style={{ color }}>{value}</div>
          <div className="small text-muted">{label}</div>
          {sub && <div className="text-muted" style={{ fontSize: '0.72rem' }}>{sub}</div>}
        </div>
      </div>
    </div>
  );
}

function bar(label, count, total, color = '#1a4d8f') {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <div className="mb-2" key={label}>
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{count} ({pct}%)</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  );
}

export default function PatientDemographicsDashboard() {
  const [ov, setOv] = useState(null);
  const [bk, setBk] = useState(null);
  const [def, setDef] = useState(null);
  const [tab, setTab] = useState('overview');
  const [sort, setSort] = useState({ col: 'id', dir: 1 });
  const [search, setSearch] = useState('');
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/patient-demographics/overview`).then(r => r.json()),
      fetch(`${API}/api/patient-demographics/breakdown`).then(r => r.json()),
      fetch(`${API}/api/patient-demographics/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBk(b); setDef(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-4">Error: {err}</div>;
  if (!ov) return <div className="text-center py-5"><div className="spinner-border text-primary" /></div>;

  const { kpis, sex_dist, epilepsy_type_dist, insurance_type_dist, blood_type_dist,
    referral_source_dist, education_level_dist, employment_status_dist,
    age_histogram, enrollment_trend, state_distribution, bmi_category_dist } = ov;

  const total = kpis.total_patients;

  const COLORS = ['#1a4d8f','#2e7d32','#f57c00','#6a1b9a','#00838f','#c62828'];

  // Filter + sort per-patient table
  const patients = (bk?.patients || []).filter(p =>
    !search || [p.full_name, p.patient_id, p.epilepsy_type, p.sex, p.insurance_type, p.address_city]
      .some(v => v && String(v).toLowerCase().includes(search.toLowerCase()))
  ).sort((a, b) => {
    const av = a[sort.col] ?? ''; const bv = b[sort.col] ?? '';
    return sort.dir * (av < bv ? -1 : av > bv ? 1 : 0);
  });

  const sortBy = col => setSort(s => ({ col, dir: s.col === col ? -s.dir : 1 }));
  const th = (col, label) => (
    <th style={{ cursor: 'pointer', whiteSpace: 'nowrap' }} onClick={() => sortBy(col)}>
      {label}{sort.col === col ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
    </th>
  );

  const tabs = ['overview', 'distributions', 'per-patient', 'enrollment', 'definitions'];
  const tabLabel = { overview: 'Overview', distributions: 'Distributions', 'per-patient': 'Per Patient', enrollment: 'Enrollment', definitions: 'Definitions' };

  return (
    <div>
      <h3 className="mb-1">👥 Patient Demographics</h3>
      <p className="text-muted small mb-3">
        {total} patients · sex/epilepsy/insurance/age/BMI cohort analysis · enrollment trend · real clinical.db data
      </p>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
        {tabs.map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {tabLabel[t]}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'overview' && (
        <>
          <div className="row mb-4">
            {kpiCard('Total Patients', total, null, '#0b1f3a')}
            {kpiCard('Avg Age', kpis.avg_age + ' yr', null, '#1a4d8f')}
            {kpiCard('Avg BMI', kpis.avg_bmi, null, '#2e7d32')}
            {kpiCard('Interpreter Needed', kpis.interpreter_needed_pct + '%', 'of patients', '#f57c00')}
            {kpiCard('Avg Epilepsy Duration', kpis.avg_years_with_epilepsy + ' yr', 'mean years with dx', '#6a1b9a')}
            {kpiCard('Avg Onset Age', kpis.avg_onset_age + ' yr', 'first diagnosis', '#00838f')}
          </div>

          <div className="row">
            {/* Sex */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Sex</div>
                <div className="card-body">
                  {sex_dist.map((r, i) => bar(r.sex, r.count, total, COLORS[i]))}
                </div>
              </div>
            </div>
            {/* Epilepsy Type */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Epilepsy Type</div>
                <div className="card-body">
                  {epilepsy_type_dist.map((r, i) => bar(r.epilepsy_type, r.count, total, COLORS[i]))}
                </div>
              </div>
            </div>
            {/* Insurance */}
            <div className="col-md-4 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold">Insurance Type</div>
                <div className="card-body">
                  {insurance_type_dist.map((r, i) => bar(r.insurance_type, r.count, total, COLORS[i]))}
                </div>
              </div>
            </div>
          </div>

          <div className="row">
            {/* Age Histogram */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">Age Distribution</div>
                <div className="card-body">
                  {(() => {
                    const mx = Math.max(...age_histogram.map(r => r.count));
                    return age_histogram.map(r => (
                      <div className="d-flex align-items-center mb-2" key={r.age_group}>
                        <span className="small me-2" style={{ width: 80 }}>{r.age_group}</span>
                        <div className="flex-grow-1 me-2">
                          <div style={{ height: 18, background: '#1a4d8f', width: `${mx > 0 ? (r.count / mx) * 100 : 0}%`, borderRadius: 3 }} />
                        </div>
                        <span className="small text-muted">{r.count}</span>
                      </div>
                    ));
                  })()}
                </div>
              </div>
            </div>
            {/* BMI */}
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold">BMI Category</div>
                <div className="card-body">
                  {bmi_category_dist.map((r, i) => bar(r.bmi_category, r.count, total, COLORS[i]))}
                  <div className="text-muted small mt-2">Mean BMI: {kpis.avg_bmi} kg/m²</div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* ── DISTRIBUTIONS TAB ── */}
      {tab === 'distributions' && (
        <div className="row">
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Referral Source</div>
              <div className="card-body">
                {referral_source_dist.map((r, i) => bar(r.referral_source, r.count, total, COLORS[i]))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Blood Type</div>
              <div className="card-body">
                {blood_type_dist.map((r, i) => bar(r.blood_type, r.count, total, COLORS[i]))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Education Level</div>
              <div className="card-body">
                {education_level_dist.map((r, i) => bar(r.education_level, r.count, total, COLORS[i]))}
              </div>
            </div>
          </div>
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">Employment Status</div>
              <div className="card-body">
                {employment_status_dist.map((r, i) => bar(r.employment_status, r.count, total, COLORS[i]))}
              </div>
            </div>
          </div>
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-semibold">State Distribution ({state_distribution.length} states)</div>
              <div className="card-body">
                <div className="row">
                  {state_distribution.map((r, i) => (
                    <div className="col-6 col-md-3 mb-2" key={r.state}>
                      <div className="d-flex justify-content-between small">
                        <span>{r.state}</span>
                        <span className="text-muted">{r.count}</span>
                      </div>
                      <div className="progress" style={{ height: 6 }}>
                        <div className="progress-bar" style={{ width: `${Math.round((r.count / total) * 100)}%`, background: COLORS[i % COLORS.length] }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── PER PATIENT TAB ── */}
      {tab === 'per-patient' && (
        <>
          <div className="mb-3">
            <input className="form-control" placeholder="Search name, ID, city, epilepsy type…"
              value={search} onChange={e => setSearch(e.target.value)} style={{ maxWidth: 340 }} />
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-hover">
              <thead className="table-dark">
                <tr>
                  {th('patient_id', 'Patient ID')}
                  {th('full_name', 'Name')}
                  {th('age', 'Age')}
                  {th('sex', 'Sex')}
                  {th('epilepsy_type', 'Epilepsy Type')}
                  {th('insurance_type', 'Insurance')}
                  {th('bmi', 'BMI')}
                  {th('address_city', 'City')}
                  {th('interpreter_needed', 'Interpreter')}
                  {th('years_with_epilepsy', 'Yrs w/ Epilepsy')}
                  {th('primary_neurologist', 'Neurologist')}
                </tr>
              </thead>
              <tbody>
                {patients.map(p => (
                  <tr key={p.patient_id}>
                    <td className="font-monospace small">{p.patient_id}</td>
                    <td>{p.full_name}</td>
                    <td>{p.age}</td>
                    <td>{p.sex}</td>
                    <td><span className={`badge ${p.epilepsy_type === 'Focal' ? 'bg-primary' : p.epilepsy_type === 'Generalized' ? 'bg-success' : p.epilepsy_type === 'Combined' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.epilepsy_type}</span></td>
                    <td>{p.insurance_type}</td>
                    <td>{p.bmi}</td>
                    <td>{p.address_city}</td>
                    <td>{p.interpreter_needed ? '✓' : ''}</td>
                    <td>{p.years_with_epilepsy}</td>
                    <td className="small">{p.primary_neurologist}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-muted small">{patients.length} patients shown</div>
        </>
      )}

      {/* ── ENROLLMENT TAB ── */}
      {tab === 'enrollment' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">Monthly Enrollment Trend ({enrollment_trend.length} months)</div>
          <div className="card-body">
            {(() => {
              const mx = Math.max(...enrollment_trend.map(r => r.count));
              return enrollment_trend.map(r => (
                <div className="d-flex align-items-center mb-2" key={r.month}>
                  <span className="small me-2 font-monospace" style={{ width: 90 }}>{r.month}</span>
                  <div className="flex-grow-1 me-2">
                    <div style={{ height: 20, background: '#1a4d8f', width: `${mx > 0 ? (r.count / mx) * 100 : 0}%`, borderRadius: 3, minWidth: r.count > 0 ? 6 : 0 }} />
                  </div>
                  <span className="small text-muted">{r.count}</span>
                </div>
              ));
            })()}
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'definitions' && (
        <div className="card shadow-sm">
          <div className="card-header fw-semibold">{def?.title || 'Definitions'}</div>
          <div className="card-body">
            <table className="table table-sm">
              <thead className="table-light">
                <tr><th>Term</th><th>Description</th></tr>
              </thead>
              <tbody>
                {(def?.concepts || []).map(c => (
                  <tr key={c.name}>
                    <td className="fw-semibold text-nowrap" style={{ width: 200 }}>{c.name}</td>
                    <td className="small">{c.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
