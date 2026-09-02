'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#1565c0';   // deep blue — Complex IV / COX encephalomyopathic
const LIGHT = '#e3f2fd';

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

function Bar({ label, value, color = COLOR }) {
  const numVal = typeof value === 'string' ? parseInt(value) || 0 : value;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}{typeof value === 'number' ? '%' : ''}</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${Math.min(numVal, 100)}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
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
            ['Gene', data.gene],
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

      <SectionCard title="Mechanism — COX6B1 / Subunit VIb1 / Complex IV Late-Stage Assembly">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="Key Clinical Differentiator — NO HCM · NO Hepatopathy · NO Tubulopathy" borderColor="#2e7d32">
        <Alert variant="success" text={data.differentiator_note} />
      </SectionCard>

      <SectionCard title={`KPIs — ${data.cohort}`}>
        <div className="row g-2">
          {(data.kpis || []).map(k => (
            <KPI key={k.label} label={k.label} value={k.value} color={k.color} />
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Absolute Contraindications & Drug Safety" borderColor="#c62828">
        {(data.contraindications || []).map(ci => (
          <div key={ci.drug} className="mb-3">
            <Alert
              variant={ci.severity.startsWith('ABSOLUTE') ? 'danger' : 'warning'}
              text={<><strong>{ci.drug}</strong> — {ci.severity}: {ci.mechanism}</>}
            />
          </div>
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
      <SectionCard title="Feature Prevalence (40-patient COX6B1 cohort, seed-606)">
        {(data.feature_prevalence || []).map(f => (
          <Bar
            key={f.feature}
            label={`${f.feature} — ${f.pct}`}
            value={parseInt(f.pct) || 0}
            color={
              f.feature.toLowerCase().includes('no hcm') || f.feature.toLowerCase().includes('no hepato') || f.feature.toLowerCase().includes('no tubu') ? '#2e7d32' :
              f.feature.toLowerCase().includes('lactic') || f.feature.toLowerCase().includes('resp') ? '#b71c1c' :
              f.feature.toLowerCase().includes('leigh') || f.feature.toLowerCase().includes('seizure') ? '#6a1b9a' :
              COLOR
            }
          />
        ))}
      </SectionCard>

      <SectionCard title="Genotype Distribution">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead className="table-dark">
              <tr><th>Genotype</th><th>n</th><th>%</th></tr>
            </thead>
            <tbody>
              {(data.genotype_dist || []).map(g => (
                <tr key={g.genotype}>
                  <td>{g.genotype}</td>
                  <td>{g.n}</td>
                  <td style={{ color: COLOR }}>{g.pct}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Outcome Summary">
        <div className="row g-3 small">
          {Object.entries(data.outcome_summary || {}).map(([k, v]) => (
            <div key={k} className="col-6 col-md-3">
              <div className="p-2 rounded text-center" style={{ background: LIGHT }}>
                <div className="fw-bold" style={{ color: COLOR }}>{String(v)}</div>
                <div className="text-muted">{k.replace(/_/g, ' ')}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Patient Cohort (first 20 of 40, seed-606)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th><th>Lactate</th>
                <th>COX%</th><th>Genotype</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {(data.patient_rows || []).map(p => (
                <tr key={p.id}>
                  <td className="fw-semibold">{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.onset_mo}</td>
                  <td style={{ color: parseFloat(p.lactate) >= 8 ? '#b71c1c' : parseFloat(p.lactate) >= 5 ? '#e65100' : '#2e7d32' }}>
                    {p.lactate}
                  </td>
                  <td style={{ color: parseFloat(p.cox_pct) < 10 ? '#b71c1c' : parseFloat(p.cox_pct) < 15 ? '#e65100' : '#2e7d32' }}>
                    {p.cox_pct}
                  </td>
                  <td className="text-muted" style={{ maxWidth: 180, wordBreak: 'break-word' }}>{p.genotype}</td>
                  <td style={{ color: p.outcome.startsWith('Died') ? '#b71c1c' : '#2e7d32', maxWidth: 150, wordBreak: 'break-word' }}>
                    {p.outcome}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Treatment Summary">
        <p className="small text-muted">{data.treatment_summary}</p>
      </SectionCard>
    </div>
  );
}

// ── Tab 3: DDx & Treatments ──────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <Spinner />;
  return (
    <div>
      <SectionCard title="COX6B1 vs Other COX Assembly Factor Diseases — Differential Diagnosis" borderColor="#c62828">
        <Alert variant="danger" text="All listed diseases share ISOLATED Complex IV deficiency — WES/WGS is MANDATORY to distinguish them. Bedside clues: HCM → SCO2/COX15; Hepatopathy → SCO1; Tubulopathy+Anaemia → COX10; Combined CI+CIV → LRPPRC; Childhood dysarthria → TACO1. COX6B1: NO HCM, NO hepatopathy, NO tubulopathy — similar to SURF1 but different locus." />
        <div className="table-responsive mt-2">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>Gene</th><th>Locus</th><th>HCM</th><th>Hepatopathy</th>
                <th>Tubulopathy</th><th>COX Defect</th><th>Distinguisher</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_table || []).map(row => (
                <tr key={row.gene} style={{ fontWeight: row.gene === 'COX6B1' ? 'bold' : 'normal', background: row.gene === 'COX6B1' ? LIGHT : undefined }}>
                  <td style={{ color: row.gene === 'COX6B1' ? COLOR : undefined }}>{row.gene}</td>
                  <td>{row.locus}</td>
                  <td style={{ color: row.hcm.startsWith('0%') ? '#2e7d32' : row.hcm.includes('100') ? '#b71c1c' : '#e65100' }}>
                    {row.hcm}
                  </td>
                  <td style={{ color: row.hepatopathy.startsWith('0%') ? '#2e7d32' : row.hepatopathy.includes('100') ? '#b71c1c' : '#e65100' }}>
                    {row.hepatopathy}
                  </td>
                  <td style={{ color: row.tubulopathy.startsWith('0%') ? '#2e7d32' : row.tubulopathy.includes('65') ? '#b71c1c' : '#e65100' }}>
                    {row.tubulopathy}
                  </td>
                  <td>{row.cox_defect}</td>
                  <td className="text-muted" style={{ maxWidth: 220, wordBreak: 'break-word' }}>{row.distinguisher}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Anaesthesia Protocol — COX6B1 / Complex IV Deficiency" borderColor="#c62828">
        {[
          ['Induction (AVOID Propofol — ABSOLUTE CI)', 'Sevoflurane inhalational induction; alternative: ketamine IV in low-dose with careful monitoring'],
          ['Maintenance', 'Sevoflurane (preferred); isoflurane acceptable; STRICTLY AVOID propofol for any indication'],
          ['Sedation (ICU / procedural)', 'Dexmedetomidine preferred; chloral hydrate short-term; NEVER propofol'],
          ['Glucose management perioperative', 'GIR 6–8 mg/kg/min IV dextrose; NEVER fast >4h; resume feeds ASAP postop'],
          ['Preoperative', 'Thiamine + biotin empiric if not on maintenance; IV glucose from midnight; avoid dehydration'],
          ['Drug warnings', 'VPA ABSOLUTE CI; metformin ABSOLUTE CI; linezolid ABSOLUTE CI; chloramphenicol ABSOLUTE CI; KD CONTRAINDICATED'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#c62828' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '10–30 mg/kg/day (children); ubiquinol preferred for bioavailability in COX deficiency'],
          ['Riboflavin B2 (Level C)', '100–300 mg/day; FMN/FAD cofactor for Complex I and Complex II (both normal in COX6B1, but empiric)'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '100–300 mg/day — ALL Leigh-like presentations until SLC19A3/biotinidase excluded (TREATABLE mimic)'],
          ['Biotin (Level C — MANDATORY empiric)', '5–20 mg/day — BTD and BTBGD are CURABLE Leigh mimics; give empirically until enzyme assay returns'],
          ['L-Carnitine (Level C)', '50–100 mg/kg/day — secondary carnitine deficiency occurs with OXPHOS failure'],
          ['LEV (preferred AED)', 'Levetiracetam first-line for seizures; renal excretion; no mito toxicity; IV formulation available'],
          ['NIV / mechanical ventilation', 'Respiratory compromise ~60% — early NIV referral; airway management with mito-safe agents'],
          ['Nutritional support / PEG', 'Growth failure ~75%; gastrostomy when oral feeding insufficient or dysphagia develops'],
          ['Fever management', 'Aggressive cooling + GIR augmentation; fever increases energy demand catastrophically in COX6B1'],
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
export default function COX6B1Page() {
  const [tab, setTab]             = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    fetch(`${API}/api/cox6b1/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2) {
      fetch(`${API}/api/cox6b1/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 3) {
      fetch(`${API}/api/cox6b1/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
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
            COX6B1 — Encephalomyopathic Complex IV Deficiency, Nuclear Type 7 (COXPD7)
          </h4>
          <div className="text-muted small">
            COX6B1-109aa-10.2kDa · 19q13.3 · AR · OMIM *124977 / #607266 ·
            Structural Subunit VIb1 (constitutive) · IMS-exposed · COX homodimer stabiliser ·
            Isolated COX deficiency (CI/CII/CIII NORMAL) ·
            NO HCM (KEY DDx SCO2/COX15) · NO Hepatopathy (KEY DDx SCO1) · NO Tubulopathy (KEY DDx COX10) ·
            p.Trp38Ser Turkish founder ~35% · 40-patient cohort seed-606 ·
            VPA / Metformin / Propofol / Linezolid ABSOLUTE CI · KD CONTRAINDICATED · LEV preferred AED
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
      {tab === 2 && <DDxTab data={defs} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
