'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — matrix-domain subunit / multi-organ PINDAC
const LIGHT = '#f3e5f5';

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
            ['Gene', data.gene + ' (alias: ' + (data.alias || 'COX4 · Subunit IV · PINDAC gene') + ')'],
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

      <SectionCard title="40-Patient Cohort KPIs (seed-641)">
        <div className="row">
          <KPI label="Cohort size"                value={data.cohort_size} />
          <KPI label="Avg lactate (mM)"           value={data.avg_lactate_mM} />
          <KPI label="Avg COX residual"           value={(data.avg_cox_residual_pct || 0) + '%'} />
          <KPI label="Pancreatic insuff (CARDINAL)" value={(kpis.pancreatic_insuff_pct || 0) + '%'} color="#b71c1c" />
          <KPI label="Dyserythropoietic (CARDINAL)" value={(kpis.dyserythropoietic_pct || 0) + '%'} color="#880e4f" />
          <KPI label="Calvarial hyperostosis"     value={(kpis.calvarial_hyper_pct || 0) + '%'} color="#6a1a4c" />
        </div>
        <div className="row mt-2">
          <KPI label="Seizures (moderate)"        value={(kpis.seizures_pct || 0) + '%'} />
          <KPI label="Leigh MRI (minority)"       value={(kpis.leigh_mri_pct || 0) + '%'} />
          <KPI label="1yr survival"               value={(kpis.survived_1yr_pct || 0) + '%'} />
          <KPI label="Hypotonia"                  value={(kpis.hypotonia_pct || 0) + '%'} />
          <KPI label="Hepatic involvement"        value={(kpis.hepatic_pct || 0) + '%'} />
          <KPI label="HCM"                        value="0%" color="#2e7d32" />
        </div>
      </SectionCard>

      <SectionCard title="🧬 PINDAC Triad — Pathognomonic for COX4I1 Deficiency" borderColor="#b71c1c">
        <Alert variant="danger"
          text="🔴 PINDAC TRIAD is PATHOGNOMONIC for COX4I1 / COXPD12: Exocrine Pancreatic Insufficiency + Dyserythropoietic Anaemia + Calvarial Hyperostosis. This combination is found in NO other COXPD disease. Recognition of any two of the three should prompt immediate COX4I1 molecular testing." />
        <div className="row g-2 small mt-2">
          {[
            ['Exocrine Pancreatic Insufficiency', '~90%', 'Steatorrhoea, ADEK deficiency, malabsorption — PERT mandatory', '#b71c1c'],
            ['Dyserythropoietic Anaemia',         '~88%', 'Dysplastic erythroid precursors on BM — NOT Fanconi anaemia', '#880e4f'],
            ['Calvarial Hyperostosis',             '~77%', 'Skull thickening on CT/X-ray — hallmark radiographic finding', '#6a1a4c'],
          ].map(([feature, pct, mech, c]) => (
            <div className="col-12 col-md-4" key={feature}>
              <div className="p-2 rounded" style={{ background: c + '18', borderLeft: `3px solid ${c}` }}>
                <span className="fw-bold" style={{ color: c }}>{feature}</span>
                <span className="ms-2 badge" style={{ background: c, color: '#fff' }}>{pct}</span>
                <div className="text-muted mt-1">{mech}</div>
              </div>
            </div>
          ))}
        </div>
        <Alert variant="warning"
          text="⚠️ Seizures are MODERATE in COX4I1 (~38%) — NOT the cardinal feature (contrast: COX8A ~85% CARDINAL). Leigh MRI present in only ~20-25% (NOT dominant — contrast SURF1 95%, COX14 80%). WES/WGS mandatory to confirm molecular diagnosis." />
      </SectionCard>

      <SectionCard title="✅ KEY DDx Negatives — Cardinal Clinical Rules">
        <div className="row g-2">
          {(data.key_ddx_negatives || []).map((neg, i) => (
            <div className="col-12 col-md-6" key={i}>
              <Alert variant="success" text={`✅ ${neg}`} />
            </div>
          ))}
        </div>
        <Alert variant="info"
          text="ℹ️ COX4I1 is the ONLY CIV deficiency with exocrine pancreatic failure + bone marrow dyserythropoiesis + calvarial hyperostosis. All other isolated COX deficiencies lack this triad." />
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
      <SectionCard title="Genotype Distribution (40-patient cohort, seed-641)">
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
                                           : f.feature.includes('PATHOGNOMONIC') ? '#b71c1c'
                                           : f.feature.includes('UNIQUE') ? '#880e4f' : COLOR }} />
            </div>
            <div className="text-muted" style={{ fontSize: '0.72rem' }}>{f.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Outcome by Genotype Class">
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee', border: '1px solid #b71c1c' }}>
              <div className="fw-bold text-danger">Null/deletion alleles</div>
              <div>1yr survival: <strong>{data.outcome?.null_allele_1yr_survival_pct}%</strong></div>
              <div className="text-muted">{data.outcome?.note}</div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#fff8e1', border: '1px solid #f57f17' }}>
              <div className="fw-bold text-warning">Missense alleles</div>
              <div>1yr survival: <strong>{data.outcome?.missense_allele_1yr_survival_pct}%</strong></div>
              <div className="text-muted">Milder course (15–35% residual CIV); PERT allows better nutritional management</div>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Exocrine Pancreatic Insufficiency vs COX Residual Activity">
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee' }}>
              <div className="fw-semibold">COX ≤18% residual</div>
              <div>Pancreatic insufficiency: <strong>{data.pancreatic_vs_cox_activity?.pancreatic_pct_when_cox_at_or_below_18pct}%</strong></div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: LIGHT }}>
              <div className="fw-semibold">COX &gt;18% residual</div>
              <div>Pancreatic insufficiency: <strong>{data.pancreatic_vs_cox_activity?.pancreatic_pct_when_cox_above_18pct}%</strong></div>
            </div>
          </div>
        </div>
        <div className="text-muted small mt-2">{data.pancreatic_vs_cox_activity?.note}</div>
      </SectionCard>

      <SectionCard title="Per-Patient Table (40 patients)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['ID','Sex','Onset (mo)','Lactate (mM)','COX %','Pancreatic','Dyseryth','Calvarial','Seizures','1yr survival'].map(h => (
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
                  <td><span className={`badge ${p.pancreatic === 'Yes' ? 'bg-danger' : 'bg-success'}`}>{p.pancreatic}</span></td>
                  <td><span className={`badge ${p.dyseryth === 'Yes' ? 'bg-danger' : 'bg-success'}`}>{p.dyseryth}</span></td>
                  <td><span className={`badge ${p.calvarial === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.calvarial}</span></td>
                  <td><span className={`badge ${p.seizures === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.seizures}</span></td>
                  <td><span className={`badge ${p.survived_1yr === 'Yes' ? 'bg-success' : 'bg-danger'}`}>{p.survived_1yr}</span></td>
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
          text="🚨 NEVER administer VPA in COX4I1 deficiency. Ketogenic diet is CONTRAINDICATED (β-oxidation requires CIV + fat malabsorption from EPI makes KD adherence dangerous). Propofol absolutely forbidden — sevoflurane for all anaesthesia." />
      </SectionCard>

      <SectionCard title="Recommended Treatments (COX4I1-specific + General OXPHOS)">
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

      <SectionCard title="DDx Matrix — Isolated COX Deficiency vs COX4I1">
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
      {(data.glossary || data.terms || []).map((t, i) => (
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
      {data.management_summary && (
        <div className="mt-3 p-3 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <div className="fw-bold small mb-2" style={{ color: COLOR }}>Management Summary</div>
          <div className="text-muted small">{data.management_summary}</div>
        </div>
      )}
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function Cox4i1Dashboard() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [def,  setDef]  = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    const h = { headers: { 'Cache-Control': 'no-cache' } };
    Promise.all([
      fetch(`${API}/api/cox4i1/overview`,    h).then(r => r.json()),
      fetch(`${API}/api/cox4i1/breakdown`,   h).then(r => r.json()),
      fetch(`${API}/api/cox4i1/definitions`, h).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-4 p-3 rounded text-white" style={{ background: COLOR }}>
        <h4 className="mb-1 fw-bold">🧬 COX4I1 — COXPD12 · PINDAC Syndrome</h4>
        <div className="small opacity-90">
          Exocrine Pancreatic Insufficiency + Dyserythropoietic Anaemia + Calvarial Hyperostosis · Complex IV Deficiency · AR Biallelic · 16q22.1
        </div>
        <div className="small opacity-75 mt-1">
          OMIM Gene *123864 · Disease #616501 · Structural Subunit IV (ubiquitous/normoxia) · Matrix-facing · ATP allosteric site
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
