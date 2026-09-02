'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#1565c0';   // dark blue — MITRAC MT-CO1 assembly, severe neonatal phenotype
const LIGHT = '#e3f2fd';

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
  return (
    <div>
      <SectionCard title="Gene & Disease Identity">
        <div className="row g-2 small">
          {[
            ['Gene', data.gene + ' (alias: ' + (data.alias || 'CCDC56 · MITRAC12') + ')'],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset', data.onset],
            ['MITRAC Branch', data.mitrac_branch],
            ['MITRAC Step', data.mitrac_step],
          ].map(([k, v]) => (
            <div className="col-12 col-md-6" key={k}>
              <span className="text-muted">{k}: </span>
              <span className="fw-semibold">{v}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 p-2 rounded small fw-semibold"
             style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          🔬 {data.ddx_fingerprint}
        </div>
      </SectionCard>

      <SectionCard title="40-Patient Cohort KPIs (seed-625)">
        <div className="row">
          <KPI label="Cohort size" value={data.cohort_size} />
          <KPI label="Avg lactate (mM)" value={data.avg_lactate_mM} />
          <KPI label="Avg COX residual" value={data.avg_cox_pct + '%'} />
          <KPI label="Leigh MRI" value={data.pct_leigh_mri + '%'} />
          <KPI label="Respiratory" value={data.pct_respiratory + '%'} />
          <KPI label="1yr survival" value={data.pct_survived_1yr + '%'} />
        </div>
        <div className="row mt-2">
          <KPI label="Hypotonia" value={data.pct_hypotonia + '%'} />
          <KPI label="Seizures" value={data.pct_seizures + '%'} />
          <KPI label="Feeding difficulty" value={data.pct_feeding_difficulty + '%'} />
          <KPI label="HCM" value={data.pct_hcm + '%'} color="#2e7d32" />
          <KPI label="Hepatopathy" value={data.pct_hepatopathy + '%'} color="#2e7d32" />
          <KPI label="Renal tubular" value={data.pct_renal_tubular + '%'} color="#2e7d32" />
        </div>
      </SectionCard>

      <SectionCard title="⚠️ KEY DDx Negatives — Cardinal Clinical Rule">
        <div className="row g-2">
          {[
            ['NO HCM (0%)', 'KEY DDx vs SCO2 (100%), COX15 (78%), COA6 (90%) — ECHO NORMAL rules these out'],
            ['NO Hepatopathy (0%)', 'KEY DDx vs SCO1 (100% neonatal hepatic failure) — LFTs NORMAL'],
            ['NO Tubulopathy (0%)', 'KEY DDx vs COX10 (65% Fanconi syndrome) — urine amino acids NORMAL'],
            ['NO Anaemia (0%)', 'KEY DDx vs COX10 (80%) — full blood count NORMAL'],
          ].map(([title, desc]) => (
            <div className="col-12 col-md-6" key={title}>
              <Alert variant="success" text={`✅ ${title}: ${desc}`} />
            </div>
          ))}
        </div>
        <Alert variant="danger"
          text="🔴 CRITICAL: COA3 vs COX14 — BOTH share all 4 cardinal negatives + Leigh MRI + isolated COX deficiency. Biochemistry CANNOT distinguish them. WES/WGS is the ONLY separating test." />
      </SectionCard>

      <SectionCard title="MT-CO1 MITRAC Assembly Pathway (COA3 Acts at Step 3)">
        <div className="row g-2">
          {(data.pathway?.steps || []).map((step, i) => {
            const isCoa3   = i + 1 === data.pathway?.coa3_step;
            const isCox14  = i + 1 === data.pathway?.cox14_step;
            const bg       = isCoa3 ? COLOR : isCox14 ? '#0288d1' : '#f5f5f5';
            const textC    = (isCoa3 || isCox14) ? '#fff' : '#333';
            return (
              <div className="col-12 col-md-6" key={i}>
                <div className="p-2 rounded small" style={{ background: bg, color: textC, border: '1px solid #ddd' }}>
                  <span className="fw-bold me-1">{i + 1}.</span>{step}
                  {isCoa3  && <span className="ms-2 badge" style={{ background: '#fff', color: COLOR }}>← COA3 DEFECT</span>}
                  {isCox14 && <span className="ms-2 badge" style={{ background: '#fff', color: '#0288d1' }}>COX14 step</span>}
                </div>
              </div>
            );
          })}
        </div>
      </SectionCard>

      <SectionCard title="Key Distinction: COA3 vs COX14 — Same MITRAC Branch, Different Proteins">
        <div className="row g-3 small">
          {[
            ['COA3 (COXPD10)', COLOR, [
              '109 aa, single TM, 17q24.2',
              'Joins MITRAC after COX14 (step 3)',
              'OMIM *614775 / #616006',
              'Ultra-rare (<15 published patients)',
              'Isolated COX <5% — identical biochemistry to COX14',
            ]],
            ['COX14 (COXPD6)', '#0288d1', [
              '66 aa, single TM, 12q24.31',
              'First MT-CO1 MITRAC contact (step 2)',
              'OMIM *614478 / #614749',
              'Equally rare (<15 published cases)',
              'Isolated COX <5% — identical biochemistry to COA3',
            ]],
          ].map(([title, c, points]) => (
            <div className="col-12 col-md-6" key={title}>
              <div className="p-3 rounded" style={{ background: c + '18', border: `2px solid ${c}` }}>
                <div className="fw-bold mb-2" style={{ color: c }}>{title}</div>
                {points.map(p => <div key={p} className="mb-1">• {p}</div>)}
              </div>
            </div>
          ))}
        </div>
        <Alert variant="danger"
          text="🔴 WES/WGS MANDATORY: COA3 and COX14 are INDISTINGUISHABLE by clinical phenotype, enzyme assay, BN-PAGE, or immunoblot. Only sequencing separates them." />
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Features ───────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  const cf = data.clinical_features || {};
  return (
    <div>
      <SectionCard title="Genotype Distribution (40-patient cohort, seed-625)">
        {(data.genotype_distribution || []).map(g => (
          <Bar key={g.genotype} label={`${g.genotype} (n=${g.count})`} value={g.pct} />
        ))}
      </SectionCard>

      <SectionCard title="Average COX Residual Activity by Genotype">
        {(data.genotype_avg_cox_pct || []).map(g => (
          <Bar key={g.genotype} label={g.genotype} value={g.avg_cox_pct} />
        ))}
        <div className="text-muted small mt-2">
          Lower COX % = more severe genotype. All values &lt;10% confirm isolated CIV deficiency.
        </div>
      </SectionCard>

      <SectionCard title="Clinical Features (40-patient cohort)">
        <div className="row g-2">
          {Object.entries(cf).map(([k, v]) => (
            <div className="col-6 col-md-4" key={k}>
              <div className="p-2 rounded small text-center"
                   style={{ background: v && v.toString().startsWith('0') ? '#e8f5e9' : LIGHT }}>
                <div className="fw-bold" style={{ color: v && v.toString().startsWith('0') ? '#2e7d32' : COLOR }}>{v}</div>
                <div className="text-muted">{k.replace(/_/g, ' ')}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Representative Patients (first 20 of 40)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['ID','Sex','Onset (mo)','Lactate (mM)','COX %','Genotype','Features','1yr survival'].map(h => (
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
                  <td>{p.lactate}</td>
                  <td>{p.cox_pct}</td>
                  <td className="text-muted">{p.genotype}</td>
                  <td>{p.features}</td>
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
      <SectionCard title="Absolute Contraindicated Drugs (NEVER in COXPD10)">
        {(data.absolute_ci_drugs || []).map(d => (
          <Alert key={d.drug} variant="danger"
            text={`🚫 ${d.drug}: ${d.mechanism}`} />
        ))}
      </SectionCard>

      <SectionCard title="Treatment Ladder">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['Agent','Dose','Level','Notes'].map(h => <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {(data.treatment_ladder || []).map(t => (
                <tr key={t.agent}>
                  <td className="fw-semibold">{t.agent}</td>
                  <td>{t.dose}</td>
                  <td>
                    <span className={`badge ${t.level === 'A' ? 'bg-success' : t.level === 'B' ? 'bg-primary' : 'bg-secondary'}`}>
                      {t.level}
                    </span>
                  </td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Differential Diagnosis — Isolated COX Deficiency COXPD Panel">
        <div className="table-responsive">
          <table className="table table-sm small table-hover">
            <thead>
              <tr style={{ background: LIGHT }}>
                {['Gene (Disease)','Locus','HCM','Hepatopathy','Tubulopathy','Leigh MRI','COX Defect','Key Distinguisher'].map(h => (
                  <th key={h} className="text-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data.ddx_table || []).map(r => {
                const isThis = r.gene.includes('← THIS');
                return (
                  <tr key={r.gene} style={isThis ? { background: LIGHT, fontWeight: 'bold' } : {}}>
                    <td style={{ color: isThis ? COLOR : undefined }}>{r.gene}</td>
                    <td>{r.locus}</td>
                    <td style={{ color: r.hcm !== '0%' ? '#b71c1c' : '#2e7d32' }}>{r.hcm}</td>
                    <td style={{ color: r.hepatopathy !== '0%' ? '#b71c1c' : '#2e7d32' }}>{r.hepatopathy}</td>
                    <td style={{ color: r.tubulopathy !== '0%' ? '#b71c1c' : '#2e7d32' }}>{r.tubulopathy}</td>
                    <td>{r.leigh}</td>
                    <td>{r.cox_defect}</td>
                    <td className="text-muted small">{r.distinguisher}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
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
export default function COA3Page() {
  const [tab,      setTab]     = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,     setDefs]    = useState(null);
  const [error,    setError]   = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/coa3/overview`).then(r => r.json()),
      fetch(`${API}/api/coa3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/coa3/definitions`).then(r => r.json()),
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
             style={{ width: 48, height: 48, background: COLOR, fontSize: 18 }}>
          C3
        </div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            COA3 — MITRAC-Assembly Complex IV Deficiency (COXPD10)
          </h4>
          <div className="text-muted small">
            COA3 · CCDC56 · MITRAC12 · 109 aa · 17q24.2 · AR · Isolated CIV &lt;5% · Leigh-like ·
            OMIM Gene <strong>*614775</strong> · Disease <strong>#616006</strong> · 40-patient cohort seed-625
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
