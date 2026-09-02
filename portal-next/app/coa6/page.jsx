'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'DDx & Treatments', 'Definitions'];
const COLOR = '#c62828';   // deep crimson — cardiac / HCM predominant
const LIGHT = '#ffebee';

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

      <SectionCard title="Mechanism — COA6 / Twin-CX9C Copper Chaperone / COX2 CuA Metalation">
        <p className="small text-muted mb-0">{data.mechanism}</p>
      </SectionCard>

      <SectionCard title="Key Clinical Differentiator — HCM 90% (CARDINAL) · Liver 35% (KEY DDx vs SCO2) · NO Tubulopathy · NO Anaemia" borderColor="#e65100">
        <Alert variant="warning" text={data.differentiator_note} />
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
      <SectionCard title="Feature Prevalence (40-patient COA6 cohort, seed-611)">
        {(data.feature_prevalence || []).map(f => (
          <Bar
            key={f.feature}
            label={`${f.feature} — ${f.pct}`}
            value={parseInt(f.pct) || 0}
            color={
              f.feature.toLowerCase().includes('hcm') ? COLOR :
              f.feature.toLowerCase().includes('no tubu') || f.feature.toLowerCase().includes('no anaemia') ? '#2e7d32' :
              f.feature.toLowerCase().includes('lactic') || f.feature.toLowerCase().includes('resp') ? '#b71c1c' :
              f.feature.toLowerCase().includes('hepatic') ? '#e65100' :
              f.feature.toLowerCase().includes('leigh') ? '#6a1b9a' :
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

      <SectionCard title="Patient Cohort (first 20 of 40, seed-611)">
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Onset</th><th>Lactate</th>
                <th>COX%</th><th>HCM</th><th>Liver</th><th>Outcome</th>
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
                  <td style={{ color: parseFloat(p.cox_pct) < 8 ? '#b71c1c' : parseFloat(p.cox_pct) < 12 ? '#e65100' : '#2e7d32' }}>
                    {p.cox_pct}
                  </td>
                  <td style={{ color: p.hcm === 'YES' ? COLOR : '#2e7d32', fontWeight: p.hcm === 'YES' ? 'bold' : 'normal' }}>
                    {p.hcm}
                  </td>
                  <td style={{ color: p.liver === 'YES' ? '#e65100' : '#2e7d32' }}>{p.liver}</td>
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
      <SectionCard title="COA6 vs Other COX Assembly Factor Diseases — Differential Diagnosis" borderColor="#c62828">
        <Alert variant="danger" text="All listed diseases share ISOLATED Complex IV deficiency — WES/WGS is MANDATORY. Bedside clues: HCM + liver → COA6; HCM only (no liver) → SCO2/COX15; Hepatic failure 100% (no HCM) → SCO1; Tubulopathy+Anaemia → COX10; Encephalomyopathy (no HCM) → SURF1/COX6B1. COA6: HCM CARDINAL (90%) + liver 35% + NO tubulopathy." />
        <div className="table-responsive mt-2">
          <table className="table table-sm table-hover small">
            <thead className="table-dark">
              <tr>
                <th>Gene</th><th>Locus</th><th>HCM</th><th>Liver</th>
                <th>Tubulopathy</th><th>COX Defect</th><th>Distinguisher</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_table || []).map(row => (
                <tr key={row.gene} style={{ fontWeight: row.gene === 'COA6' ? 'bold' : 'normal', background: row.gene === 'COA6' ? LIGHT : undefined }}>
                  <td style={{ color: row.gene === 'COA6' ? COLOR : undefined }}>{row.gene}</td>
                  <td>{row.locus}</td>
                  <td style={{ color: row.hcm.includes('90%') || row.hcm.includes('100%') ? COLOR : row.hcm.includes('78%') ? '#e65100' : row.hcm.startsWith('0%') || row.hcm.includes('Rare') ? '#2e7d32' : '#555' }}>
                    {row.hcm}
                  </td>
                  <td style={{ color: row.liver.includes('100%') ? '#b71c1c' : row.liver.includes('35%') ? '#e65100' : '#2e7d32' }}>
                    {row.liver}
                  </td>
                  <td style={{ color: row.tubulopathy.startsWith('0%') ? '#2e7d32' : '#b71c1c' }}>
                    {row.tubulopathy}
                  </td>
                  <td>{row.cox_defect}</td>
                  <td className="text-muted" style={{ maxWidth: 240, wordBreak: 'break-word' }}>{row.distinguisher}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Cardiac Management — COA6 HCM (Metabolic / ATP-Failure Cardiomyopathy)" borderColor="#c62828">
        {[
          ['Rate control (PREFERRED)', 'Propranolol or atenolol — reduce myocardial O2 demand in HCM; monitor HR + BP'],
          ['AVOID positive inotropes', 'Digoxin / dobutamine / milrinone HIGH RISK — increase O2 demand in ATP-depleted myocardium → fatal arrhythmia; do NOT use reflexively for "heart failure"'],
          ['AVOID afterload reducers for HCM', 'Nitrates / hydralazine reduce preload in hypertrophied ventricle → may worsen obstruction'],
          ['ECHO monitoring', 'Serial echocardiography every 3–6 months: wall thickness, LVOT gradient, function'],
          ['Paediatric cardiology', 'Mandatory co-management for all COA6 patients with HCM'],
          ['Transplant', 'Cardiac transplantation has been performed in select COA6 cases; multidisciplinary decision'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: COLOR }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Anaesthesia Protocol — COA6 / Complex IV Deficiency + HCM" borderColor="#c62828">
        {[
          ['Induction (AVOID Propofol — ABSOLUTE CI)', 'Sevoflurane inhalational; cardiac-safe induction; pre-op ECHO mandatory'],
          ['Maintenance', 'Sevoflurane preferred; NEVER propofol for any indication'],
          ['Sedation (ICU / procedural)', 'Dexmedetomidine preferred; NEVER propofol'],
          ['Glucose perioperative', 'GIR 6–8 mg/kg/min IV dextrose; NEVER fast >4h; resume feeds ASAP'],
          ['Cardiac monitoring', 'Continuous ECG + echo-guided fluid management; avoid tachycardia'],
          ['Drug warnings', 'VPA ABSOLUTE CI; metformin ABSOLUTE CI; linezolid ABSOLUTE CI; positive inotropes HIGH RISK; KD CONTRAINDICATED'],
        ].map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-semibold" style={{ color: '#c62828' }}>{k}:</span>{' '}
            <span className="text-muted">{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Mitochondrial Cofactor & Supportive Therapy">
        {[
          ['CoQ10 / Ubiquinol (Level C)', '10–30 mg/kg/day; ubiquinol preferred'],
          ['Riboflavin B2 (Level C)', '100–300 mg/day — CI and CII cofactor (both normal in COA6, but empiric)'],
          ['Thiamine B1 (Level C — MANDATORY empiric)', '100–300 mg/day — ALL Leigh-like presentations until SLC19A3/BTD excluded'],
          ['Biotin (Level C — MANDATORY empiric)', '5–20 mg/day — BTD/BTBGD are CURABLE Leigh mimics'],
          ['L-Carnitine (Level C)', '50–100 mg/kg/day — secondary carnitine deficiency in OXPHOS failure'],
          ['UDCA (Level C — hepatic involvement)', '15–20 mg/kg/day if liver involvement confirmed (35% of COA6 patients)'],
          ['Copper histidinate (investigational)', 'Theoretical copper chaperone support; discuss with metabolic specialist; limited evidence'],
          ['NIV / mechanical ventilation', 'Respiratory compromise ~50%; early NIV referral; mito-safe anaesthesia'],
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
export default function COA6Page() {
  const [tab, setTab]             = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    fetch(`${API}/api/coa6/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 1 || tab === 2) {
      fetch(`${API}/api/coa6/breakdown`).then(r => r.json()).then(setBreakdown).catch(() => {});
    }
    if (tab === 3) {
      fetch(`${API}/api/coa6/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
    }
  }, [tab]);

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      <div className="d-flex align-items-center mb-3 gap-3">
        <div style={{
          width: 48, height: 48, borderRadius: '50%',
          background: COLOR, display: 'flex', alignItems: 'center',
          justifyContent: 'center', color: '#fff', fontWeight: 'bold', fontSize: 22, flexShrink: 0,
        }}>🫀</div>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            COA6 — Infantile Cardiomyopathic Complex IV Deficiency, Nuclear Type 14 (COXPD14)
          </h4>
          <div className="text-muted small">
            COA6-73aa-8.3kDa · 1q42.2 · AR · OMIM *614772 / #614924 ·
            Copper Chaperone (twin-CX9C) · COX2 CuA Metalation · Cooperates with SCO1/SCO2 ·
            HCM 90% CARDINAL (KEY DDx COX6B1 0%) · Liver 35% (KEY DDx SCO2 0%) · NO Tubulopathy · NO Anaemia ·
            p.Trp59Cys Australian/UK founder ~30% · 40-patient cohort seed-611 ·
            VPA / Metformin / Propofol / Positive-Inotropes ABSOLUTE CI · KD CONTRAINDICATED
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
