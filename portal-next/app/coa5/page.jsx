'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#880e4f';   // deep crimson — HCM-dominant cardiac phenotype
const LIGHT = '#fce4ec';

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
            ['Gene', data.gene + ' (alias: ' + (data.alias || 'C2orf64 · PET191') + ')'],
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

      <SectionCard title="40-Patient Cohort KPIs (seed-629)">
        <div className="row">
          <KPI label="Cohort size"       value={data.cohort_size} />
          <KPI label="Avg lactate (mM)"  value={data.avg_lactate_mM} />
          <KPI label="Avg COX residual"  value={(data.avg_cox_residual_pct || 0) + '%'} />
          <KPI label="HCM (cardinal)"    value={(kpis.hcm_pct || 0) + '%'}  color="#b71c1c" />
          <KPI label="Leigh MRI"         value={(kpis.leigh_mri_pct || 0) + '%'} />
          <KPI label="1yr survival"      value={(kpis.survived_1yr_pct || 0) + '%'} />
        </div>
        <div className="row mt-2">
          <KPI label="Hypotonia"         value={(kpis.hypotonia_pct || 0) + '%'} />
          <KPI label="Seizures"          value={(kpis.seizures_pct || 0) + '%'} />
          <KPI label="Feeding difficulty" value={(kpis.feeding_pct || 0) + '%'} />
          <KPI label="Respiratory"       value={(kpis.respiratory_pct || 0) + '%'} />
          <KPI label="Hepatopathy"       value="0%"  color="#2e7d32" />
          <KPI label="Tubulopathy"       value="0%"  color="#2e7d32" />
        </div>
      </SectionCard>

      <SectionCard title="❤️ HCM — Cardinal Feature of COA5 (COXPD11)">
        <Alert variant="danger"
          text="🔴 HCM is CARDINAL in COA5 (~88%): Hypertrophic cardiomyopathy reflects severe ATP depletion in cardiomyocytes from isolated CIV deficiency. HCM separates COA5 from COX14/COA3/COX20/COX6B1 (all NO HCM)." />
        <div className="row g-2 small mt-2">
          {[
            ['SCO2 (COXPD2)',  '100%', 'CuA metalation (MT-CO2 copper)', '#b71c1c'],
            ['COA6 (COXPD14)', '90%',  'Copper chaperone twin-CX9C',     '#c62828'],
            ['COA5 (COXPD11)', '~88%', 'MT-CO1 co-translational ← THIS', COLOR],
            ['COX15 (COXPD5)', '78%',  'Heme a synthase step 2',         '#ad1457'],
            ['SURF1 (COXPD1)', '10%',  'Heme a3/CuB insertion',          '#e91e63'],
            ['COX14 (COXPD6)', '0%',   'MT-CO1 MITRAC — NO HCM',        '#2e7d32'],
            ['COA3 (COXPD10)', '0%',   'MT-CO1 MITRAC — NO HCM',        '#2e7d32'],
            ['COX20 (COXPD8)', '0%',   'MT-CO2 MITRAC — NO HCM',        '#2e7d32'],
          ].map(([gene, pct, mech, c]) => (
            <div className="col-12 col-md-6" key={gene}>
              <div className="p-2 rounded" style={{ background: c + '18', borderLeft: `3px solid ${c}` }}>
                <span className="fw-bold" style={{ color: c }}>{gene}</span>
                <span className="ms-2 badge" style={{ background: c, color: '#fff' }}>HCM {pct}</span>
                <div className="text-muted mt-1">{mech}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="⚠️ KEY DDx Negatives — Cardinal Clinical Rules">
        <div className="row g-2">
          {(data.key_ddx_negatives || []).map((neg, i) => (
            <div className="col-12 col-md-6" key={i}>
              <Alert variant="success" text={`✅ ${neg}`} />
            </div>
          ))}
        </div>
        <Alert variant="warning"
          text="⚠️ COA5 vs SCO2: Both HCM-dominant isolated COX — SCO2 uses copper metalation (MT-CO2 CuA); COA5 uses MT-CO1 stabilisation. Phenotype overlap is high; WES mandatory for separation." />
      </SectionCard>

      <SectionCard title="Key Contrasts with Related Diseases">
        <div className="row g-3 small">
          {Object.entries(data.key_contrasts || {}).map(([pair, desc]) => (
            <div className="col-12" key={pair}>
              <Alert variant="info" text={`${pair.replace('_vs_', ' vs ')}: ${desc}`} />
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
      <SectionCard title="Genotype Distribution (40-patient cohort, seed-629)">
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
                            backgroundColor: f.feature.startsWith('NO') ? '#2e7d32' : COLOR }} />
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
              <div className="text-muted">Slightly milder cardiac course</div>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="HCM vs COX Activity">
        <div className="row g-2 small">
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: LIGHT }}>
              <div className="fw-semibold">COX &gt;10% residual</div>
              <div>HCM rate: <strong>{data.hcm_vs_cox_activity?.hcm_pct_when_cox_above_10pct}%</strong></div>
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="p-2 rounded" style={{ background: '#ffebee' }}>
              <div className="fw-semibold">COX ≤10% residual</div>
              <div>HCM rate: <strong>{data.hcm_vs_cox_activity?.hcm_pct_when_cox_at_or_below_10pct}%</strong></div>
            </div>
          </div>
        </div>
        <div className="text-muted small mt-2">{data.hcm_vs_cox_activity?.note}</div>
      </SectionCard>

      <SectionCard title="Per-Patient Table (40 patients)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['ID','Sex','Onset (mo)','Lactate (mM)','COX %','HCM','Leigh MRI','1yr survival'].map(h => (
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
                    <span className={`badge ${p.hcm === 'Yes' ? 'bg-danger' : 'bg-success'}`}>{p.hcm}</span>
                  </td>
                  <td>
                    <span className={`badge ${p.leigh_mri === 'Yes' ? 'bg-warning text-dark' : 'bg-secondary'}`}>{p.leigh_mri}</span>
                  </td>
                  <td>
                    <span className={`badge ${p.survived_1yr === 'Yes' ? 'bg-success' : 'bg-danger'}`}>{p.survived_1yr}</span>
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
      <SectionCard title="Absolute Contraindications (NEVER in COXPD11)">
        {(data.contraindications || []).filter(d => d.severity === 'ABSOLUTE CI').map(d => (
          <Alert key={d.drug} variant="danger"
            text={`🚫 ${d.drug} [${d.severity}]: ${d.reason}`} />
        ))}
        {(data.contraindications || []).filter(d => d.severity !== 'ABSOLUTE CI').map(d => (
          <Alert key={d.drug} variant="warning"
            text={`⚠️ ${d.drug} [${d.severity}]: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="Recommended Treatments">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['Agent', 'Evidence Level', 'Notes'].map(h => <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {(data.treatments || []).map(t => (
                <tr key={t.tx}>
                  <td className="fw-semibold">{t.tx}</td>
                  <td>
                    <span className={`badge ${
                      t.level.includes('MANDATORY') ? 'bg-warning text-dark' :
                      t.level.includes('Level B') ? 'bg-primary' :
                      t.level.includes('Level C') ? 'bg-secondary' : 'bg-info text-dark'
                    }`}>{t.level}</span>
                  </td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Differential Diagnosis Matrix">
        <div className="table-responsive">
          <table className="table table-sm small table-hover">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['Disease', 'Shared Features', 'Distinguishing from COA5'].map(h => (
                  <th key={h}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map(r => (
                <tr key={r.disease}>
                  <td className="fw-semibold small">{r.disease}</td>
                  <td className="text-muted small">{r.shared}</td>
                  <td className="small">{r.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="danger"
          text="🔴 WES/WGS MANDATORY: Isolated COX deficiency with HCM overlaps SCO2, COA6, COX15, COA5. Enzyme assay + ECHO cannot uniquely identify COA5 — sequencing is required." />
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ───────────────────────────────────────────────────────
function DefsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Glossary">
        {(data.glossary || []).map(g => (
          <div key={g.term} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR }}>{g.term}</div>
            <div className="text-muted small">{g.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Notes">
        {(data.clinical_notes || []).map((note, i) => (
          <Alert key={i} variant="info" text={note} />
        ))}
      </SectionCard>

      <SectionCard title="Management Summary">
        <div className="p-2 rounded small" style={{ background: LIGHT }}>
          {data.management_summary}
        </div>
      </SectionCard>

      <SectionCard title="Inheritance">
        <div className="small text-muted">{data.inheritance_detail}</div>
      </SectionCard>

      <SectionCard title="Key References">
        {(data.references || []).map(r => (
          <div key={r.citation} className="mb-3">
            <div className="fw-semibold small">{r.citation}</div>
            <div className="text-muted small">{r.note}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────────
export default function COA5Page() {
  const [tab,      setTab]      = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,     setDefs]     = useState(null);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/coa5/overview`).then(r => r.json()),
      fetch(`${API}/api/coa5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/coa5/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefs(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">API error: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div className="rounded-circle d-flex align-items-center justify-content-center fw-bold text-white"
             style={{ width: 48, height: 48, background: COLOR, fontSize: 16 }}>
          C5
        </div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            COA5 — Cardiomyopathic Complex IV Deficiency (COXPD11)
          </h4>
          <div className="text-muted small">
            COA5 · C2orf64 · PET191 · 168 aa · 2q11.2 · AR · Isolated CIV &lt;15% · HCM-dominant ·
            OMIM Gene <strong>*614657</strong> · Disease <strong>#614932</strong> · 40-patient cohort seed-629
          </div>
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <DdxTab data={breakdown} />}
      {tab === 3 && <DefsTab data={defs} />}
    </div>
  );
}
