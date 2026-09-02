'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#4527a0';   // deep purple — rare early-assembly, no cardinal organ
const LIGHT = '#ede7f6';

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
            ['Gene', data.gene + ' (alias: ' + (data.alias || 'C12orf62') + ')'],
            ['Protein', data.protein],
            ['Disease', data.disease],
            ['OMIM Gene', data.omim_gene],
            ['OMIM Disease', data.omim_disease],
            ['Locus', data.chromosome],
            ['Inheritance', data.inheritance],
            ['Onset', data.onset],
            ['Cohort', data.cohort],
          ].map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <span className="fw-semibold">{k}:</span>{' '}
              <span className="text-muted">{v}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Assembly Pathway — MITRAC Early MT-CO1 Co-translational Assembly">
        {data.pathway && (
          <div>
            <p className="small text-muted mb-2">{data.pathway.name}</p>
            <ol className="small ps-3">
              {(data.pathway.steps || []).map((s, i) => (
                <li
                  key={i}
                  className="mb-1"
                  style={{ color: i + 1 === data.pathway.cox14_step ? COLOR : undefined, fontWeight: i + 1 === data.pathway.cox14_step ? 'bold' : 'normal' }}
                >
                  {s} {i + 1 === data.pathway.cox14_step ? '← COX14 acts here' : ''}
                </li>
              ))}
            </ol>
            {data.pathway.footnote && <p className="small text-muted fst-italic mb-0">{data.pathway.footnote}</p>}
          </div>
        )}
      </SectionCard>

      <SectionCard title="Key Clinical Differentiator — NO HCM · NO Hepatopathy · NO Tubulopathy · Leigh-dominant" borderColor="#e65100">
        <Alert variant="warning" text={
          <span>
            <strong>COX14 (COXPD6) three-negative bedside rule:</strong>{' '}
            (1) NO HCM on ECHO — rules out SCO2 (100%), COX15 (78%), COA6 (90%).{' '}
            (2) NO hepatopathy / elevated LFTs — rules out SCO1 (100%).{' '}
            (3) NO renal Fanconi / aminoaciduria — rules out COX10 (65%).{' '}
            All three negative findings + isolated COX deficiency &lt;5% + Leigh MRI = COX14 / COX6B1 / SURF1 priority targets → WES/WGS mandatory.
          </span>
        } />
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={
              k.label.includes('HCM') || k.label.includes('Hepatopathy') || k.label.includes('Tubulopathy')
                ? '#2e7d32' : COLOR
            } />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Clinical Feature Prevalence">
        {(data.feature_bars || []).map(f => (
          <Bar
            key={f.label}
            label={f.label}
            value={f.value}
            color={
              f.label.toLowerCase().includes('hcm') || f.label.toLowerCase().includes('hepatopathy') || f.label.toLowerCase().includes('tubulopathy')
                ? '#2e7d32'
                : f.label.toLowerCase().includes('regression') || f.label.toLowerCase().includes('lactic')
                  ? '#b71c1c'
                  : f.label.toLowerCase().includes('leigh') || f.label.toLowerCase().includes('respiratory')
                    ? '#e65100'
                    : COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Summary">
        <p className="small text-muted mb-0">{data.summary}</p>
      </SectionCard>

      <SectionCard title="Rapid Diagnosis Checklist" borderColor="#6a1b9a">
        <ol className="small ps-3">
          {(data.diagnosis_checklist || []).map((item, i) => (
            <li key={i} className="mb-1 text-muted">{item}</li>
          ))}
        </ol>
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor="#b71c1c">
        {(data.absolute_ci || []).map(ci => (
          <Alert key={ci.drug} variant="danger" text={
            <><strong>{ci.drug}</strong> — {ci.reason}</>
          } />
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 2: Patients & Features ───────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Genotype Distribution (40-patient COX14 cohort, seed-617)">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-dark">
              <tr><th>Genotype</th><th>n</th><th>%</th></tr>
            </thead>
            <tbody>
              {(data.genotype_distribution || []).map(g => (
                <tr key={g.genotype}>
                  <td>{g.genotype}</td>
                  <td>{g.count}</td>
                  <td style={{ color: COLOR }}>{g.pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Avg COX% Residual Activity by Genotype">
        {(data.genotype_avg_cox_pct || []).map(g => (
          <Bar
            key={g.genotype}
            label={g.genotype}
            value={parseFloat(g.avg_cox_pct) * 10}
            color={parseFloat(g.avg_cox_pct) < 4 ? '#b71c1c' : parseFloat(g.avg_cox_pct) < 6 ? '#e65100' : COLOR}
          />
        ))}
        <p className="small text-muted fst-italic mt-1">Bar scaled ×10 for visibility. Values shown are actual % of normal COX activity.</p>
      </SectionCard>

      <SectionCard title="Patient Cohort — First 20 of 40 (seed-617)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th><th>Lactate</th>
                <th>COX%</th><th>Leigh</th><th>Hypotonia</th><th>Seizures</th>
                <th>Resp</th><th>HCM ⚑</th><th>Hepatopathy ⚑</th><th>Tubulopathy ⚑</th><th>5yr</th>
              </tr>
            </thead>
            <tbody>
              {(data.patient_table || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_mo} mo</td>
                  <td style={{ color: parseFloat(p.lactate) >= 8 ? '#b71c1c' : parseFloat(p.lactate) >= 5 ? '#e65100' : '#2e7d32' }}>
                    {p.lactate}
                  </td>
                  <td style={{ color: parseFloat(p.cox_pct) < 4 ? '#b71c1c' : parseFloat(p.cox_pct) < 6 ? '#e65100' : '#2e7d32' }}>
                    {p.cox_pct}%
                  </td>
                  <td style={{ color: p.leigh === 'Yes' ? '#6a1b9a' : '#555' }}>{p.leigh}</td>
                  <td>{p.hypotonia}</td>
                  <td>{p.seizures}</td>
                  <td style={{ color: p.resp === 'Yes' ? '#e65100' : '#555' }}>{p.resp}</td>
                  <td style={{ color: '#2e7d32', fontWeight: 'bold' }}>{p.hcm}</td>
                  <td style={{ color: '#2e7d32', fontWeight: 'bold' }}>{p.hepatopathy}</td>
                  <td style={{ color: '#2e7d32', fontWeight: 'bold' }}>{p.tubulopathy}</td>
                  <td style={{ color: p.survived === 'Yes' ? '#2e7d32' : '#b71c1c' }}>{p.survived}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Treatment Ladder (Level C — supportive / empiric)">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-dark">
              <tr><th>Agent</th><th>Dose</th><th>Level</th><th>Note</th></tr>
            </thead>
            <tbody>
              {(data.treatment_ladder || []).map(t => (
                <tr key={t.agent}>
                  <td className="fw-semibold" style={{ color: COLOR }}>{t.agent}</td>
                  <td>{t.dose}</td>
                  <td><span className="badge" style={{ background: COLOR }}>Level {t.level}</span></td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Absolute Contraindications" borderColor="#b71c1c">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-dark">
              <tr><th>Drug</th><th>Mechanism</th></tr>
            </thead>
            <tbody>
              {(data.absolute_ci_drugs || []).map(d => (
                <tr key={d.drug}>
                  <td className="fw-bold text-danger">{d.drug}</td>
                  <td className="text-muted">{d.mechanism}</td>
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
function DDxTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="COX14 vs Other COX Assembly Factor Diseases — Differential Diagnosis" borderColor="#b71c1c">
        <Alert variant="danger" text={
          <span>
            <strong>All listed diseases share ISOLATED Complex IV deficiency — WES/WGS is MANDATORY.</strong>{' '}
            COX14 three-negative rule: NO HCM + NO hepatopathy + NO tubulopathy + Leigh-dominant encephalopathy + COX &lt;5%.
            DDx priority: COX14 / COX6B1 / SURF1 share this triad — WES distinguishes them.
          </span>
        } />
        <div className="table-responsive mt-2">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>Gene</th><th>Locus</th><th>Disease</th><th>HCM</th>
                <th>Hepatopathy</th><th>Tubulopathy</th><th>Leigh MRI</th><th>COX</th><th>Key Distinguisher</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_table || []).map(row => (
                <tr
                  key={row.gene}
                  style={{
                    fontWeight: row.gene.startsWith('COX14') ? 'bold' : 'normal',
                    background: row.gene.startsWith('COX14') ? LIGHT : undefined,
                  }}
                >
                  <td style={{ color: row.gene.startsWith('COX14') ? COLOR : undefined }}>{row.gene}</td>
                  <td>{row.locus}</td>
                  <td>{row.disease}</td>
                  <td style={{ color: row.hcm.includes('CARDINAL') || row.hcm.includes('90%') || row.hcm.includes('78%') ? '#b71c1c' : '#2e7d32' }}>
                    {row.hcm}
                  </td>
                  <td style={{ color: row.hepatopathy.includes('100%') ? '#b71c1c' : row.hepatopathy.includes('35%') ? '#e65100' : '#2e7d32' }}>
                    {row.hepatopathy}
                  </td>
                  <td style={{ color: row.tubulopathy.includes('65%') || row.tubulopathy.includes('80%') ? '#b71c1c' : '#2e7d32' }}>
                    {row.tubulopathy}
                  </td>
                  <td style={{ color: COLOR }}>{row.leigh}</td>
                  <td>{row.cox_defect}</td>
                  <td className="text-muted" style={{ maxWidth: 220, wordBreak: 'break-word', fontSize: '0.75rem' }}>{row.distinguisher}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="MITRAC Complex — MT-CO1 Early Assembly Pathway Context" borderColor={COLOR}>
        {[
          ['COX14 + COA3 (MITRAC12)', 'Bind nascent MT-CO1 immediately after mitoribosomal translation. COX14 TM helix anchors complex in IMM; COA3 stabilises the co-translational intermediate.'],
          ['YME1L / m-AAA / AFG3L2', 'IMM quality-control proteases. Without COX14 protection, MT-CO1 is exposed to these proteases and rapidly degraded → no COX1 module → no CIV.'],
          ['COX20 (COXPD8)', 'Analogous early assembly factor for MT-CO2 (COX2). COX20 deficiency causes COXPD8; clinically indistinguishable from COX14 (COXPD6) — WES mandatory.'],
          ['COA3 (MITRAC12, COXPD10)', 'COX14 binding partner in the early MITRAC checkpoint. COA3 mutations cause COXPD10 — similar Leigh-like encephalopathy phenotype; WES distinguishes.'],
          ['Downstream assembly', 'After COX14 stabilisation: MT-CO1 haem a + CuB metalation (SURF1/COX10/COX15/COX11 pathway), followed by MT-CO2 CuA metalation (SCO1/SCO2/COA6), then structural subunit addition.'],
        ].map(([k, v]) => (
          <div key={k} className="mb-3 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Anaesthesia Protocol — COX14 / Complex IV Deficiency" borderColor="#b71c1c">
        {[
          ['Induction (AVOID Propofol — ABSOLUTE CI)', 'Sevoflurane inhalational; PRIS risk with <5% residual COX is catastrophic'],
          ['Maintenance', 'Sevoflurane preferred; NEVER propofol for any indication'],
          ['Sedation (ICU / procedural)', 'Dexmedetomidine preferred; NEVER propofol or ketamine without metabolic team input'],
          ['Glucose perioperative', 'GIR 6–8 mg/kg/min IV dextrose; NEVER fast >4h; resume feeds ASAP post-procedure'],
          ['Lactic acidosis monitoring', 'Lactate + pH pre/intra/post-op; NaHCO3 for pH <7.1'],
          ['Drug warnings', 'VPA ABSOLUTE CI; metformin ABSOLUTE CI; linezolid ABSOLUTE CI; chloramphenicol ABSOLUTE CI; KD CONTRAINDICATED'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#b71c1c' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '10–30 mg/kg/day — electron shuttle bypass support'],
          ['Riboflavin B2 (Level C)', '100–400 mg/day — CI and CIII cofactor support'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '5–10 mg/kg/day — ALL Leigh-like presentations; exclude SLC19A3/BTD (curable mimics)'],
          ['Biotin (Level C — MANDATORY empiric)', '5–20 mg/day — biotinidase deficiency is a curable Leigh mimic'],
          ['L-Carnitine (Level C)', '50–100 mg/kg/day — secondary carnitine deficiency in OXPHOS failure'],
          ['LEV (levetiracetam)', '20–60 mg/kg/day — preferred AED; no mito toxicity; renal excretion'],
          ['GIR 6–8 perioperative', 'Glucose infusion rate — prevents catabolism; never fast >4h'],
          ['NIV / BiPAP', 'Respiratory compromise 60–75%; early non-invasive ventilation referral'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab 4: Definitions ────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="Inheritance">
        <p className="small text-muted">{data.inheritance_detail}</p>
      </SectionCard>

      <SectionCard title="Management Summary">
        <p className="small text-muted">{data.management_summary}</p>
      </SectionCard>

      <SectionCard title="Glossary">
        {(data.glossary || []).map(g => (
          <div key={g.term} className="mb-3">
            <div className="fw-semibold small mb-1" style={{ color: COLOR }}>{g.term}</div>
            <p className="small text-muted mb-0">{g.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Notes" borderColor="#2e7d32">
        {(data.clinical_notes || []).map((note, i) => (
          <Alert key={i} variant="success" text={note} />
        ))}
      </SectionCard>

      <SectionCard title="References">
        {(data.references || []).map((ref, i) => (
          <div key={i} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{ref.citation}</span>
            {ref.note && <span className="text-muted"> — {ref.note}</span>}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function COX14Page() {
  const [tab, setTab]             = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cox14/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1) {
      fetch(`${API}/api/cox14/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 2) {
      fetch(`${API}/api/cox14/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 3) {
      fetch(`${API}/api/cox14/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>🧬</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            COX14 — Complex IV Deficiency, Nuclear Type 6 (COXPD6)
          </h4>
          <div className="text-muted small">
            COX14 (C12orf62) · 66aa · 12q24.31 · AR · OMIM *614478 / #614749 ·
            MITRAC Early MT-CO1 Co-translational Assembly Factor ·
            NO HCM ⚑ · NO Hepatopathy ⚑ · NO Tubulopathy ⚑ ·
            Leigh-dominant Encephalopathy · COX &lt;5% · 40-patient cohort seed-617 ·
            VPA / Metformin / Propofol / Linezolid ABSOLUTE CI · KD CONTRAINDICATED
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={t}>
            <button
              className={`nav-link ${tab === i ? 'active' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <DDxTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
