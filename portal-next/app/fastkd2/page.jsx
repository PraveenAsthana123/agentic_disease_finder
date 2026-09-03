'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#1b5e20';   // dark green — mitoribosome / mt-RNA processing / FASTKD2
const LIGHT = '#e8f5e9';

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
            ['Gene', data.gene + ' (alias: ' + (data.alias || 'FASTKD2 — mt-RNA Processing Factor') + ')'],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', '*' + data.omim_gene],
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

      <SectionCard title="40-Patient Cohort KPIs (seed-649)">
        <div className="row">
          <KPI label="Cohort size"            value={data.cohort_size} />
          <KPI label="Avg lactate (mM)"       value={data.avg_lactate_mM} />
          <KPI label="Avg CI residual"        value={(data.avg_ci_residual_pct || 0) + '%'} color="#880e4f" />
          <KPI label="Avg CIV residual"       value={(data.avg_civ_residual_pct || 0) + '%'} color="#880e4f" />
          <KPI label="Leigh MRI"              value={(kpis.leigh_mri_pct || 0) + '%'} color="#f57f17" />
          <KPI label="Encephalopathy"         value="90%" color="#b71c1c" />
        </div>
        <div className="row mt-2">
          <KPI label="Seizures"               value={(kpis.seizures_pct || 0) + '%'} color="#f57f17" />
          <KPI label="Ataxia"                 value={(kpis.ataxia_pct || 0) + '%'} color="#f57f17" />
          <KPI label="Hepatopathy"            value={(kpis.hepato_pct || 0) + '%'} color="#546e7a" />
          <KPI label="2yr survival"           value={(kpis.survived_2yr_pct || 0) + '%'} />
          <KPI label="HCM"                    value="0%" color="#2e7d32" />
          <KPI label="Tubulopathy"            value="0%" color="#2e7d32" />
        </div>
      </SectionCard>

      <SectionCard title="🚨 Critical Biochemical Alert — Combined CI+CIV (NOT Isolated CIV)" borderColor="#b71c1c">
        <Alert variant="danger"
          text="🔴 FASTKD2 produces COMBINED Complex I + Complex IV deficiency (NOT isolated CIV). If enzyme analysis shows both CI and CIV deficient with normal CII, consider: (1) FASTKD2 (AR, 2q33.1 — mt-RNA processing); (2) m.3243A>G MTTL1 (maternal inheritance — check heteroplasmy); (3) POLG (AR — check mtDNA copy number). Combined deficiency rules out isolated CIV diseases (SURF1, PET100, COX14, COA3)." />
        <Alert variant="warning"
          text="⚠️ Italian/European ancestry + early childhood onset (3-18 months) + combined CI+CIV + cerebellar-predominant MRI → suspect FASTKD2 pGlu323Lys (c.967G>A) founder mutation (30% of cohort). Confirm with WES." />
      </SectionCard>

      <SectionCard title="✅ KEY DDx Negatives — Cardinal Clinical Rules">
        <div className="row g-2">
          {(data.key_ddx_negatives || []).map((neg, i) => (
            <div className="col-12 col-md-6" key={i}>
              <Alert variant="success" text={`✅ ${neg}`} />
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Key Contrasts with Related Diseases">
        <div className="row g-3 small">
          {Object.entries(data.key_contrasts || {}).map(([pair, desc]) => (
            <div className="col-12" key={pair}>
              <Alert variant="info" text={`${pair.replace(/_vs_/g, ' vs ').replace(/_/g, ' ')}: ${desc}`} />
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
      <SectionCard title="Genotype Distribution (40-patient cohort, seed-649)">
        {(data.genotype_dist || []).map(g => (
          <Bar key={g.genotype} label={`${g.genotype} (n=${g.n}, ${g.pct}%)`} value={g.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Variant Details">
        {(data.variants || []).map((v, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
            <div className="fw-bold small" style={{ color: COLOR }}>{v.genotype}</div>
            <div className="d-flex gap-2 mt-1 small flex-wrap">
              <span className="badge" style={{ background: COLOR, color: '#fff' }}>{v.pct}% (n={v.n})</span>
              <span className="badge bg-secondary">{v.domain}</span>
              <span className={`badge ${v.severity === 'Severe' ? 'bg-danger' : 'bg-warning text-dark'}`}>{v.severity}</span>
            </div>
            <div className="text-muted mt-1" style={{ fontSize: '0.75rem' }}>{v.note}</div>
          </div>
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
                                           : f.pct > 80 ? '#b71c1c'
                                           : f.pct > 60 ? COLOR : '#546e7a' }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.72rem' }}>{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Biomarker Summary">
        <div className="row g-2 small">
          {Object.entries(data.biomarker_summary || {}).map(([k, v]) => (
            <div className="col-12 col-md-6" key={k}>
              <div className="p-2 rounded" style={{ background: LIGHT, border: `1px solid ${COLOR}` }}>
                <div className="fw-bold" style={{ color: COLOR }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                <div className="text-muted">{v}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Outcome by Allele Class">
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee', border: '1px solid #b71c1c' }}>
              <div className="fw-bold text-danger">Null/truncating alleles (pArg611Ter, splice/null, pGly508Arg)</div>
              <div>2yr survival: <strong>{data.outcome?.null_allele_2yr_survival_pct}%</strong></div>
              <div className="text-muted">{data.outcome?.note}</div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#fff8e1', border: '1px solid #f57f17' }}>
              <div className="fw-bold text-warning">Missense alleles (pGlu323Lys, pLeu199Pro)</div>
              <div>2yr survival: <strong>{data.outcome?.missense_allele_2yr_survival_pct}%</strong></div>
              <div className="text-muted">30-50% residual CI/CIV; slower progression; longer survival possible</div>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Per-Patient Table (40 patients, seed-649)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['ID','Sex','Genotype','Onset (mo)','Lactate (mM)','CI%','CIV%','Leigh MRI','Seizures','Ataxia','Hepato','2yr survival'].map(h => (
                  <th key={h} className="text-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data.patient_table || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{p.id}</td>
                  <td>{p.sex}</td>
                  <td className="small" style={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                      title={p.genotype}>{p.genotype}</td>
                  <td>{p.onset_mo}</td>
                  <td>{p.lactate_mM}</td>
                  <td>{p.ci_pct}</td>
                  <td>{p.civ_pct}</td>
                  <td><span className={`badge ${p.leigh_mri === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.leigh_mri}</span></td>
                  <td><span className={`badge ${p.seizures === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.seizures}</span></td>
                  <td><span className={`badge ${p.ataxia === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.ataxia}</span></td>
                  <td><span className={`badge ${p.hepato === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.hepato}</span></td>
                  <td><span className={`badge ${p.survived_2yr === 'Yes' ? 'bg-success' : 'bg-danger'}`}>{p.survived_2yr}</span></td>
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
      <SectionCard title="⚠️ Absolute Contraindications & High-Risk Agents" borderColor="#b71c1c">
        {(data.contraindications || []).map((ci, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{
            background: ci.severity === 'ABSOLUTE CI' ? '#ffebee' : '#fff3e0',
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
      </SectionCard>

      <SectionCard title="Recommended Treatments (FASTKD2 — Combined OXPHOS Deficiency)">
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

      <SectionCard title="DDx Matrix — FASTKD2 vs Combined OXPHOS Deficiency Mimics">
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
  const entries = Object.entries(data).filter(([k]) => k !== 'references' && k !== 'management_summary');
  return (
    <div>
      {entries.map(([key, val], i) => (
        <div key={i} className="mb-3 p-3 rounded" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
          <div className="fw-bold small mb-1" style={{ color: COLOR }}>
            {key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
          </div>
          <div className="text-muted small">{typeof val === 'string' ? val : JSON.stringify(val)}</div>
        </div>
      ))}
      {data.management_summary && (
        <div className="mt-3 p-3 rounded" style={{ background: '#e8f5e9', borderLeft: `4px solid ${COLOR}` }}>
          <div className="fw-bold small mb-2" style={{ color: COLOR }}>Management Summary</div>
          <div className="text-muted small">{data.management_summary}</div>
        </div>
      )}
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
export default function Fastkd2Dashboard() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [def,  setDef]  = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    const h = { headers: { 'Cache-Control': 'no-cache' } };
    Promise.all([
      fetch(`${API}/api/fastkd2/overview`,    h).then(r => r.json()),
      fetch(`${API}/api/fastkd2/breakdown`,   h).then(r => r.json()),
      fetch(`${API}/api/fastkd2/definitions`, h).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4 p-3 rounded text-white" style={{ background: COLOR }}>
        <h4 className="mb-1 fw-bold">🧬 FASTKD2 — Combined OXPHOS Deficiency (CI + CIV) · mt-RNA Processing Defect</h4>
        <div className="small opacity-90">
          Mitochondrial 16S rRNA Processing · AR Biallelic · 2q33.1 · Early Childhood Onset
        </div>
        <div className="small opacity-75 mt-1">
          OMIM Gene *612322 · 916aa · ~99.4kDa · mt-Matrix · Ghezzi 2008 AmJHumGenet · No isolated CIV — combined CI+CIV
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
