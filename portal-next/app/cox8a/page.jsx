'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#00695c';   // dark teal — structural subunit / epilepsy-dominant
const LIGHT = '#e0f2f1';

function KPI({ label, value, color = COLOR }) {
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

function Bar({ label, value, color = COLOR }) {
  const numVal = typeof value === 'string' ? parseInt(value) || 0 : value;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="text-muted">{value}{typeof value === 'number' ? '%' : ''}</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${Math.min(numVal, 100)}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#b71c1c' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
  return (
    <div className="mb-2 p-2 rounded small" style={{ background: bg, borderLeft: `4px solid ${border}` }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Spinner() {
  return <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>;
}

// ── Tab 1: Overview ──────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <Spinner />;
  const kpis = data.kpis || {};
  return (
    <div>
      <SectionCard title="Gene & Disease Identity">
        <div className="row g-2 small">
          {[
            ['Gene', data.gene + ' (alias: ' + (data.alias || 'COX8 · Subunit VIIIa') + ')'],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset', data.onset],
            ['Assembly Pathway', data.assembly_pathway],
            ['Cardinal Feature', data.cardinal_feature],
          ].map(([k, v]) => (
            <div className="col-12 col-md-6" key={k}>
              <span className="text-muted">{k}: </span>
              <span className="fw-semibold">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded small fw-semibold"
             style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          🔬 {data.biochemical_fingerprint}
        </div>
      </SectionCard>

      <SectionCard title="40-Patient Cohort KPIs (seed-633)">
        <div className="row">
          <KPI label="Cohort size"        value={data.cohort_size} />
          <KPI label="Avg lactate (mM)"   value={data.avg_lactate_mM} />
          <KPI label="Avg COX residual"   value={(data.avg_cox_residual_pct || 0) + '%'} />
          <KPI label="Seizures (cardinal)" value={(kpis.seizures_pct || 0) + '%'} color="#b71c1c" />
          <KPI label="Leigh MRI"          value={(kpis.leigh_mri_pct || 0) + '%'} />
          <KPI label="1yr survival"       value={(kpis.survived_1yr_pct || 0) + '%'} />
        </div>
        <div className="row mt-2">
          <KPI label="Hypotonia"          value={(kpis.hypotonia_pct || 0) + '%'} />
          <KPI label="Encephalopathy"     value={(kpis.encephalopathy_pct || 0) + '%'} />
          <KPI label="Feeding difficulty" value={(kpis.feeding_pct || 0) + '%'} />
          <KPI label="Respiratory"        value={(kpis.respiratory_pct || 0) + '%'} />
          <KPI label="HCM"               value="0%" color="#2e7d32" />
          <KPI label="Hepatopathy"       value="0%" color="#2e7d32" />
        </div>
      </SectionCard>

      <SectionCard title="🧠 Seizures — Cardinal / Distinguishing Feature of COX8A (COXPD15)" borderColor="#b71c1c">
        <Alert variant="danger"
          text="🔴 SEIZURES are CARDINAL in COX8A (~85%): Brain expresses ONLY COX8A (no COX8B compensation) → brain maximally affected → refractory epileptic encephalopathy. This distinguishes COX8A from COX6B1 (~45%), COX14 (~45%), COA3 (~38%) within the CIV structural/assembly class." />
        <div className="row g-2 small mt-2">
          {[
            ['COX8A (COXPD15)',  '~85%', 'Structural subunit — brain only COX8A',         COLOR],
            ['COX6B1 (COXPD7)',  '~45%', 'Structural subunit — lower seizure burden',      '#00838f'],
            ['SURF1 (COXPD1)',   '~40%', 'Heme a3/CuB insertion factor — Leigh dominant', '#00695c'],
            ['COX14 (COXPD6)',   '~45%', 'MITRAC MT-CO1 assembly — Leigh dominant',       '#004d40'],
            ['COA3 (COXPD10)',   '~38%', 'MITRAC12 assembly — Leigh dominant',            '#1b5e20'],
            ['COA5 (COXPD11)',   '~38%', 'HCM 88% dominant — seizures secondary',         '#880e4f'],
            ['SCO2 (COXPD2)',    '~30%', 'HCM 100% — cardiac dominant, not epileptic',    '#ad1457'],
            ['COX20 (COXPD8)',   '~28%', 'Ataxia 100% CARDINAL — childhood onset',        '#6a1a4c'],
          ].map(([gene, pct, mech, c]) => (
            <div className="col-12 col-md-6" key={gene}>
              <div className="p-2 rounded" style={{ background: c + '18', borderLeft: `3px solid ${c}` }}>
                <span className="fw-bold" style={{ color: c }}>{gene}</span>
                <span className="ms-2 badge" style={{ background: c, color: '#fff' }}>Seizures {pct}</span>
                <div className="text-muted mt-1">{mech}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="✅ KEY DDx Negatives — Cardinal Clinical Rules">
        <div className="row g-2">
          {(data.key_ddx_negatives || []).map((neg, i) => (
            <div className="col-12 col-md-6" key={i}>
              <Alert variant="success" text={`✅ ${neg}`} />
            </div>
          ))}
        </div>
        <Alert variant="warning"
          text="⚠️ COX8A brain-dominant because COX8B compensates in heart muscle. NO HCM is the bedside rule that distinguishes COX8A from SCO2/COA5/COA6/COX15. WES/WGS mandatory to confirm molecular diagnosis." />
      </SectionCard>

      <SectionCard title="Key Contrasts with Related Diseases">
        <div className="row g-3 small">
          {Object.entries(data.key_contrasts || {}).map(([pair, desc]) => (
            <div className="col-12" key={pair}>
              <Alert variant="info" text={`${pair.replace(/_vs_/g, ' vs ')}: ${desc}`} />
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Features ───────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Genotype Distribution (40-patient cohort, seed-633)">
        {(data.genotype_dist || []).map(g => (
          <Bar key={g.genotype} label={`${g.genotype} (n=${g.n}, ${g.pct}%)`} value={g.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Clinical Feature Prevalence">
        {(data.feature_prev || []).map(f => (
          <div key={f.feature} className="mb-2">
            <div className="d-flex justify-content-between small mb-1">
              <span>{f.feature}</span>
              <span className="text-muted">{f.pct}% (n={f.n}/40)</span>
            </div>
            <div className="progress" style={{ height: 10 }}>
              <div className="progress-bar"
                   style={{ width: `${Math.min(f.pct, 100)}%`,
                            backgroundColor: f.feature.startsWith('NO') ? '#2e7d32'
                                           : f.feature.includes('Seizure') ? '#b71c1c' : COLOR }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.72rem' }}>{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Outcome by Genotype Class">
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee', border: '1px solid #b71c1c' }}>
              <div className="fw-bold text-danger">Null/truncating alleles</div>
              <div>1yr survival: <strong>{data.outcome?.null_allele_1yr_survival_pct}%</strong></div>
              <div className="text-muted">{data.outcome?.note}</div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#fff8e1', border: '1px solid #f57f17' }}>
              <div className="fw-bold text-warning">Missense alleles</div>
              <div>1yr survival: <strong>{data.outcome?.missense_allele_1yr_survival_pct}%</strong></div>
              <div className="text-muted">Slightly milder course (partial residual CIV)</div>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Seizures vs COX Residual Activity">
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee' }}>
              <div className="fw-semibold">COX ≤10% residual</div>
              <div>Seizure rate: <strong>{data.seizure_vs_cox_activity?.seizure_pct_when_cox_at_or_below_10pct}%</strong></div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: LIGHT }}>
              <div className="fw-semibold">COX &gt;10% residual</div>
              <div>Seizure rate: <strong>{data.seizure_vs_cox_activity?.seizure_pct_when_cox_above_10pct}%</strong></div>
            </div>
          </div>
        </div>
        <div className="text-muted small mt-2">{data.seizure_vs_cox_activity?.note}</div>
      </SectionCard>

      <SectionCard title="Per-Patient Table (40 patients)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['ID','Sex','Onset (mo)','Lactate (mM)','COX %','Seizures','Leigh MRI','Respiratory','1yr survival'].map(h => (
                  <th key={h} className="text-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data.patient_table || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_mo}</td>
                  <td>{p.lactate_mM}</td>
                  <td>{p.cox_pct}</td>
                  <td>
                    <span className={`badge ${p.seizures === 'Yes' ? 'bg-danger' : 'bg-success'}`}>
                      {p.seizures}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${p.leigh_mri === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {p.leigh_mri}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${p.respiratory === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>
                      {p.respiratory}
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${p.survived_1yr === 'Yes' ? 'bg-success' : 'bg-danger'}`}>
                      {p.survived_1yr}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: DDx & Treatments ──────────────────────────────────────────────────
function DdxTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="⚠️ Absolute Contraindications & High-Risk Drugs" borderColor="#b71c1c">
        {(data.contraindications || []).map((ci, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{
            background: ci.severity === 'ABSOLUTE CI' ? '#ffebee'
                      : ci.severity === 'CONTRAINDICATED' ? '#fff3e0' : '#fff8e1',
            borderLeft: `4px solid ${ci.severity === 'ABSOLUTE CI' ? '#b71c1c' : '#f57f17'}`
          }}>
            <div className="fw-bold small">
              <span className={`badge me-2 ${ci.severity === 'ABSOLUTE CI' ? 'bg-danger' : 'bg-warning text-dark'}`}>
                {ci.severity}
              </span>
              {ci.drug}
            </div>
            <div className="text-muted small mt-1">{ci.reason}</div>
          </div>
        ))}
        <Alert variant="danger"
          text="🚨 NEVER administer VPA or ketogenic diet in COX8A deficiency. Seizures must be managed with LEV ± clobazam; infantile spasms with ACTH/VGB. Propofol absolutely forbidden — sevoflurane for all anaesthesia." />
      </SectionCard>

      <SectionCard title="Recommended Treatments & Supportive Measures">
        {(data.treatments || []).map((tx, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
            <div className="d-flex justify-content-between small">
              <span className="fw-semibold">{tx.tx}</span>
              <span className="badge" style={{ background: COLOR, color: '#fff' }}>{tx.level}</span>
            </div>
            <div className="text-muted small">{tx.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="DDx Matrix — Isolated COX Deficiency">
        {(data.ddx_matrix || []).map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#f5f5f5', borderLeft: `3px solid #546e7a` }}>
            <div className="fw-bold small text-primary">{d.disease}</div>
            <div className="small mt-1"><span className="text-muted">Shared: </span>{d.shared}</div>
            <div className="small mt-1"><span className="text-muted">Distinguishing: </span>{d.distinguishing}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ───────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      {(data.terms || []).map((t, i) => (
        <div key={i} className="mb-3 p-3 rounded" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
          <div className="fw-bold small" style={{ color: COLOR }}>{t.term}</div>
          <div className="text-muted small mt-1">{t.definition}</div>
        </div>
      ))}
      {(data.clinical_notes || []).map((note, i) => (
        <Alert key={i} variant="info" text={note} />
      ))}
      {(data.references || []).map((ref, i) => (
        <div key={i} className="mb-2 p-2 rounded small" style={{ background: '#f5f5f5' }}>
          <div className="fw-semibold">{ref.citation}</div>
          <div className="text-muted">{ref.note}</div>
        </div>
      ))}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function Cox8aDashboard() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [def,  setDef]  = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    const h = { headers: { 'Cache-Control': 'no-cache' } };
    Promise.all([
      fetch(`${API}/api/cox8a/overview`,    h).then(r => r.json()),
      fetch(`${API}/api/cox8a/breakdown`,   h).then(r => r.json()),
      fetch(`${API}/api/cox8a/definitions`, h).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4 p-3 rounded text-white" style={{ background: COLOR }}>
        <h4 className="mb-1 fw-bold">🧬 COX8A — COXPD15</h4>
        <div className="small opacity-90">
          Progressive Epileptic Encephalopathy / Leigh Syndrome · Complex IV Deficiency · AR Biallelic · 11q13.1
        </div>
        <div className="small opacity-75 mt-1">
          OMIM Gene *123870 · Disease #619062 · Structural Subunit VIIIa (ubiquitous isoform) · Brain-dominant (COX8B heart compensation)
        </div>
      </div>

      {err && <div className="alert alert-danger">{err}</div>}

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === i ? 'active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={ov}  />}
      {tab === 1 && <PatientsTab    data={bd}  />}
      {tab === 2 && <DdxTab         data={bd}  />}
      {tab === 3 && <DefinitionsTab data={def} />}
    </div>
  );
}
