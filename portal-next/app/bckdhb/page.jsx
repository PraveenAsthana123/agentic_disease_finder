'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#6a1b9a';   // deep purple — BCKDHB E1β structural role / complex assembly
const ACCENT2 = '#b71c1c';   // dark red — acute crisis / cerebral oedema / lethal if untreated
const ACCENT3 = '#e65100';   // deep orange — BCAA accumulation / leucine surge HAZARD
const ACCENT4 = '#0277bd';   // steel blue — thiamine B1 / treatment (lower response ~5% vs 10-20% BCKDHA)
const ACCENT5 = '#4a148c';   // deeper purple — alloisoleucine PATHOGNOMONIC / mechanism
const ACCENT6 = '#1b5e20';   // dark green — KEY NEGATIVES (PDH normal / αKGDH normal / DLD normal)
const ACCENT7 = '#37474f';   // dark slate — variant data / gene card
const ACCENT8 = '#004d40';   // deep teal — BCAA-free formula / diet therapy cornerstone

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

function PctBar({ label, pct, color = ACCENT }) {
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function Section({ title, children, color = ACCENT }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function useFetch(url) {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  useEffect(() => {
    fetch(url).then(r => r.json()).then(setData).catch(setErr);
  }, [url]);
  return { data, err };
}

// ── Tab components ───────────────────────────────────────────────────────────

function OverviewTab() {
  const { data, err } = useFetch(`${API}/api/bckdhb/overview`);
  if (err)  return <div className="alert alert-danger">Error loading overview.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const k = data.kpis || {};
  return (
    <div>
      <Alert
        text="🔑 PATHOGNOMONIC MARKER: Alloisoleucine >5 µmol/L in plasma — diagnostic ONLY in MSUD (BCKDHA/BCKDHB/DBT deficiency). Alloisoleucine is absent from all non-MSUD conditions. BCKDHB causes MSUD Type 1B — biochemically IDENTICAL to Type 1A (BCKDHA). ONLY gene panel (BCKDHA + BCKDHB + DBT + DLD) distinguishes between MSUD subtypes."
        variant="info"
      />
      <Alert
        text="⚠️ LEUCINE NEUROTOXICITY — TREATMENT URGENCY: Plasma leucine >400 µmol/L → acute encephalopathy; >1000 µmol/L → cerebral oedema risk; >2000 µmol/L → life-threatening brainstem herniation. Classic neonatal MSUD Type 1B presents day 3–7 with maple syrup odour + progressive encephalopathy. IV glucose (GIR ≥8 mg/kg/min) + insulin + BCAA-free formula + HD if leucine rising → IMMEDIATE ACTION REQUIRED."
        variant="danger"
      />
      <Alert
        text="✅ KEY NEGATIVES (confirm isolated BCKDH block): PDH activity NORMAL · αKGDH complex activity NORMAL · GCS intact · DLD free enzyme NORMAL. If any of these are reduced → consider DLD deficiency (all four complexes blocked). BCKDHB: E1β has NO TPP-binding domain — thiamine responsiveness ~5% (lower than BCKDHA 10–20%)."
        variant="success"
      />
      <Alert
        text={`💊 FIRST-LINE: (1) BCAA-restricted diet + BCAA-free synthetic formula (Level A — lifelong); (2) Thiamine B1 10–20 mg/kg/day trial ≥3 months ALL patients (Level A — ~5% thiamine-responsive in BCKDHB, lower than BCKDHA 10–20% because E1β has NO TPP active site). VPA ABSOLUTE CI (triple mechanism). Fasting EXTREME HAZARD. Liver transplant (Level B) markedly improves leucine tolerance.`}
        variant="warning"
      />

      <Section title="Gene Summary" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'38%'}}>Gene</td><td>{data.gene} — {data.protein}</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{data.locus} — Autosomal Recessive (chromosome 6q14.1)</td></tr>
            <tr><td className="fw-bold">Protein length</td><td>{data.aa_length} aa precursor (~342 aa mature after MTS cleavage); E1β STRUCTURAL subunit of BCKDH E1 decarboxylase (α₂β₂ heterotetramer — NO TPP active site; that is on E1α/BCKDHA)</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td style={{color: ACCENT4, fontWeight:'600'}}>{data.cofactor}</td></tr>
            <tr><td className="fw-bold">Mechanism</td><td>{data.mechanism}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease}</td></tr>
            <tr><td className="fw-bold">Inheritance</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.inheritance}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT2}}>Pathognomonic marker</td><td style={{color: ACCENT2, fontWeight:'600'}}>{data.pathognomonic_marker}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT3}}>Key distinguishing feature</td><td style={{color: ACCENT3}}>{data.key_distinguishing_feature}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT6}}>Structural brain hallmark</td><td style={{color: ACCENT6}}>{data.structural_brain_hallmark}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="BCKDH Complex Components" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Component</th><th>Gene(s)</th><th>Function in BCKDHB deficiency context</th></tr>
            </thead>
            <tbody>
              {(data.bckdh_complex_components || []).map((comp, i) => (
                <tr key={i} style={comp.component.includes('THIS GENE') ? {background:'#f3e5f5'} : {}}>
                  <td className="fw-bold small">{comp.component}</td>
                  <td className="small">{comp.component.includes('E1α') ? 'BCKDHA' : comp.component.includes('E1β') ? 'BCKDHB' : comp.component.includes('E2') ? 'DBT' : comp.component.includes('E3') ? 'DLD' : 'BCKDK/PPM2C'}</td>
                  <td className="small">{comp.function}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="BCAA Catabolism Pathway (BCKDH Block)" color={ACCENT}>
        {data.bcaa_catabolism_pathway && (
          <table className="table table-sm table-bordered small">
            <tbody>
              <tr><td className="fw-bold" style={{width:'35%',color:ACCENT6}}>Step 1 (reversible — transamination)</td><td>{data.bcaa_catabolism_pathway.step_1_reversible}</td></tr>
              <tr style={{background:'#ffebee'}}><td className="fw-bold" style={{color:ACCENT2}}>Step 2 — IRREVERSIBLE — BLOCKED 🚫</td><td style={{color:ACCENT2,fontWeight:'600'}}>{data.bcaa_catabolism_pathway.step_2_irreversible_blocked}</td></tr>
              <tr><td className="fw-bold small">Downstream Leu path</td><td className="small">{data.bcaa_catabolism_pathway.downstream_leu}</td></tr>
              <tr><td className="fw-bold small">Downstream Val path</td><td className="small">{data.bcaa_catabolism_pathway.downstream_val}</td></tr>
              <tr><td className="fw-bold small">Downstream Ile path</td><td className="small">{data.bcaa_catabolism_pathway.downstream_ile}</td></tr>
              <tr style={{background:'#f3e5f5'}}><td className="fw-bold" style={{color:ACCENT5}}>Alloisoleucine formation</td><td style={{color:ACCENT5}}>{data.bcaa_catabolism_pathway.alloisoleucine_formation}</td></tr>
            </tbody>
          </table>
        )}
      </Section>

      <Section title="KPI Summary — 40-Patient BCKDHB Cohort" color={ACCENT}>
        <div className="row g-2">
          <KPI label="Cohort (n)" value={k.cohort_n} color={ACCENT7} />
          <KPI label="Avg Plasma Leu (µmol/L)" value={k.avg_plasma_leucine_umol} color={ACCENT2} />
          <KPI label="Avg Alloisoleucine (µmol/L)" value={k.avg_alloisoleucine_umol} color={ACCENT5} />
          <KPI label="Maple Odour (%)" value={`${k.maple_odour_pct}%`} color={ACCENT3} />
          <KPI label="Cerebral Oedema Acute (%)" value={`${k.cerebral_oedema_acute_pct}%`} color={ACCENT2} />
          <KPI label="MRI Diffusion Restriction (%)" value={`${k.mri_diffusion_restriction_pct}%`} color={ACCENT} />
          <KPI label="Thiamine-Responsive (%)" value={`${k.thiamine_responsive_pct}%`} color={ACCENT4} />
          <KPI label="Liver Transplant (%)" value={`${k.liver_transplant_pct}%`} color={ACCENT7} />
          <KPI label="Carnitine Deficient (%)" value={`${k.carnitine_deficient_pct}%`} color={ACCENT3} />
          <KPI label="DRE (%)" value={`${k.dre_pct}%`} color={ACCENT2} />
          <KPI label="VPA Avoided (%)" value={`${k.vpa_avoided_pct}%`} color={ACCENT6} />
          <KPI label="Male (%)" value={`${k.male_pct}%`} color={ACCENT7} />
        </div>
      </Section>

      <Section title="Phenotype Distribution" color={ACCENT}>
        {(data.phenotype_distribution || []).map((p, i) => (
          <PctBar key={i} label={p.label} pct={p.pct} color={p.color} />
        ))}
      </Section>

      <Section title="High-Risk Drugs / Exposures" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{background: d.risk === 'ABSOLUTE CI' ? '#ffcdd2' : d.risk === 'EXTREME HAZARD' ? '#ffe0b2' : '#fff3e0', border: `1px solid ${d.risk === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}`}}>
            <strong style={{color: d.risk === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}}>{d.drug}</strong>
            <span className="badge ms-2" style={{background: d.risk === 'ABSOLUTE CI' ? ACCENT2 : d.risk === 'EXTREME HAZARD' ? ACCENT3 : '#f9a825', color:'#fff'}}>{d.risk}</span>
            <div className="small text-muted mt-1">{d.mechanism}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

function PatientsTab() {
  const { data, err } = useFetch(`${API}/api/bckdhb/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Diagnostic Biomarkers" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Biomarker</th><th>Normal range</th><th>BCKDHB range</th><th>Clinical significance</th></tr>
            </thead>
            <tbody>
              {(data.biomarkers || []).map((b, i) => (
                <tr key={i} style={b.bckdhb_range && (b.bckdhb_range.includes('NORMAL') || b.bckdhb_range.includes('ABSENT')) ? {background:'#e8f5e9'} : b.bckdhb_range && b.bckdhb_range.includes('PATHOGNOMONIC') ? {background:'#f3e5f5'} : {}}>
                  <td className="fw-bold">{b.name}</td>
                  <td style={{color:'#388e3c'}}>{b.normal}</td>
                  <td style={{color: b.bckdhb_range && (b.bckdhb_range.includes('NORMAL') || b.bckdhb_range.includes('ABSENT')) ? '#388e3c' : b.bckdhb_range && b.bckdhb_range.includes('PATHOGNOMONIC') ? ACCENT5 : ACCENT2, fontWeight:'600'}}>{b.bckdhb_range}</td>
                  <td>{b.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Key Variants" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Variant</th><th>Molecular effect</th><th>Phenotype</th></tr>
            </thead>
            <tbody>
              {(data.key_variants || []).map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{color: ACCENT5}}>{v.variant}</td>
                  <td>{v.effect}</td>
                  <td style={{color: v.phenotype.includes('Classic') ? ACCENT2 : v.phenotype.includes('Thiamine') ? ACCENT4 : ACCENT3}}>{v.phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Patient Sample (first 10 of 40)" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Leu (µmol/L)</th><th>Ile</th><th>Val</th><th>Allo-Ile</th>
                <th>Maple</th><th>Oedema</th><th>MRI-DWI</th>
                <th>Thiamine-R</th><th>Diet-OK</th><th>Transplant</th><th>VPA avoided</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td>{p.sex}</td>
                  <td style={{color: p.phenotype.includes('Classic') ? ACCENT2 : p.phenotype.includes('Thiamine') ? ACCENT4 : ACCENT3, fontSize:11}}>{p.phenotype}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{color: p.plasma_leucine_umol > 1000 ? ACCENT2 : ACCENT3, fontWeight:'600'}}>{p.plasma_leucine_umol}</td>
                  <td>{p.plasma_isoleucine_umol}</td>
                  <td>{p.plasma_valine_umol}</td>
                  <td style={{color: ACCENT5, fontWeight:'600'}}>{p.alloisoleucine_umol}</td>
                  <td>{p.maple_odour ? '✓' : '—'}</td>
                  <td style={{color: p.cerebral_oedema ? ACCENT2 : 'inherit'}}>{p.cerebral_oedema ? '⚠' : '—'}</td>
                  <td>{p.mri_diffusion_restriction ? '✓' : '—'}</td>
                  <td style={{color: p.thiamine_responsive ? ACCENT4 : 'inherit'}}>{p.thiamine_responsive ? '✓' : '—'}</td>
                  <td style={{color: p.diet_well_controlled ? ACCENT6 : ACCENT3}}>{p.diet_well_controlled ? '✓' : '✗'}</td>
                  <td>{p.liver_transplant ? '✓' : '—'}</td>
                  <td style={{color: p.vpa_avoided ? ACCENT6 : ACCENT2, fontWeight:'600'}}>{p.vpa_avoided ? '✓' : '✗'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function SeizuresTab() {
  const { data, err } = useFetch(`${API}/api/bckdhb/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading data.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="⚡ Seizures in BCKDHB deficiency are primarily METABOLIC — driven by acute leucine/KIC neurotoxicity, cerebral oedema, and energy failure (myelinopathy). IDENTICAL seizure pattern to BCKDHA (same BCKDH block). Seizure control requires METABOLIC NORMALISATION first (leucine reduction). AED monotherapy without BCAA control is ineffective. Burst suppression on neonatal EEG is common in classic MSUD acute crisis."
        variant="warning"
      />
      <Section title="Seizure Types" color={ACCENT}>
        {(data.seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.pct} color={ACCENT} />
        ))}
      </Section>
      <Section title="Metabolic Crisis Triggers" color={ACCENT2}>
        <Alert text="ALL triggers cause BCAA surge via muscle protein catabolism → leucine crisis within hours. Metabolic emergency protocol must be pre-arranged for every MSUD patient (sick-day plan + emergency letter + nearest metabolic centre). BCKDHB: IDENTICAL trigger profile to BCKDHA." variant="danger" />
        {(data.trigger_types || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT2} />
        ))}
      </Section>
      <Section title="High-Risk Drugs" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{background: d.risk === 'ABSOLUTE CI' ? '#ffcdd2' : d.risk === 'EXTREME HAZARD' ? '#ffe0b2' : '#fff9c4', border: `1px solid ${d.risk === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}`}}>
            <strong style={{color: d.risk === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3}}>{d.drug}</strong>
            <span className="badge ms-2" style={{background: d.risk === 'ABSOLUTE CI' ? ACCENT2 : ACCENT3, color:'#fff'}}>{d.risk}</span>
            <div className="small text-muted mt-1">{d.mechanism}</div>
          </div>
        ))}
      </Section>
    </div>
  );
}

function TreatmentsTab() {
  const { data, err } = useFetch(`${API}/api/bckdhb/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading data.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="🔑 TREATMENT PRIORITY: (1) BCAA-restricted diet + BCAA-free synthetic formula — lifelong, non-negotiable (Level A). (2) Thiamine B1 high-dose trial ALL patients ≥3 months (Level A) — NOTE: BCKDHB thiamine response ~5% (lower than BCKDHA 10–20%) because E1β has NO TPP-binding domain. (3) Acute crisis: IV glucose GIR 8–12 + insulin + BCAA-free amino acid infusion ± HD if leucine rising (Level A). (4) Carnitine supplementation (Level B). (5) Liver transplant — consider before age 10 (Level B)."
        variant="info"
      />
      <Section title="Treatment Ladder" color={ACCENT4}>
        {(data.treatments || []).map((t, i) => (
          <div key={i} className="mb-3 p-3 rounded" style={{border:`2px solid ${t.color || ACCENT4}`, background:'#f8f9fa'}}>
            <div className="d-flex justify-content-between align-items-center mb-1">
              <strong style={{color: t.color || ACCENT4}}>{t.drug}</strong>
              <div>
                <span className="badge me-2" style={{background: t.level === 'A' ? '#1b5e20' : t.level === 'B' ? '#0277bd' : '#827717', color:'#fff'}}>Level {t.level}</span>
                <span className="badge" style={{background: '#546e7a', color:'#fff'}}>{t.response_pct}% response</span>
              </div>
            </div>
            <div className="small text-muted">{t.note}</div>
            <div className="mt-1">
              <div className="progress" style={{height:8}}>
                <div className="progress-bar" style={{width:`${t.response_pct}%`, backgroundColor: t.color || ACCENT4}} />
              </div>
            </div>
          </div>
        ))}
      </Section>
    </div>
  );
}

function DefinitionsTab() {
  const { data, err } = useFetch(`${API}/api/bckdhb/definitions`);
  if (err)  return <div className="alert alert-danger">Error loading definitions.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const gc = data.gene_card || {};
  return (
    <div>
      <Section title="Gene Card" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {Object.entries(gc).map(([k, v], i) => (
              <tr key={i}>
                <td className="fw-bold" style={{width:'35%', color: ACCENT7}}>{k}</td>
                <td style={{color: k.includes('CI') || k.includes('ABSOLUTE') ? ACCENT2 : k.includes('First-line') ? ACCENT4 : 'inherit', fontWeight: k.includes('CI') || k.includes('First-line') || k.includes('Pathognomonic') ? '600' : 'normal'}}>{Array.isArray(v) ? v.join(' · ') : v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Key Concepts" color={ACCENT5}>
        {(data.key_concepts || []).map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{border:`1px solid ${ACCENT5}`, background:'#f3e5f5'}}>
            <div className="fw-bold small mb-1" style={{color: ACCENT5}}>{c.term}</div>
            <div className="small">{c.definition}</div>
          </div>
        ))}
      </Section>

      <Section title="Diagnostic Thresholds" color={ACCENT}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {Object.entries(data.diagnostic_thresholds || {}).map(([k, v], i) => (
              <tr key={i} style={k.includes('ci') || k.includes('absolute') ? {background:'#ffebee'} : k.includes('pathognomonic') ? {background:'#f3e5f5'} : {}}>
                <td className="fw-bold" style={{width:'40%', color: k.includes('ci') ? ACCENT2 : ACCENT7}}>{k.replace(/_/g,' ')}</td>
                <td style={{color: k.includes('pathognomonic') ? ACCENT5 : k.includes('hd') ? ACCENT2 : 'inherit', fontWeight: k.includes('pathognomonic') || k.includes('hd') ? '600' : 'normal'}}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th style={{width:'30%'}}>Disease</th><th>How to distinguish from BCKDHB deficiency</th></tr>
            </thead>
            <tbody>
              {(data.differential_diagnosis || []).map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{color: ACCENT5}}>{d.disease}</td>
                  <td>{d.distinguishing_features}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function BCKDHBPage() {
  const [tab, setTab] = useState(0);
  const COMPONENTS = [OverviewTab, PatientsTab, SeizuresTab, TreatmentsTab, DefinitionsTab];
  const Active = COMPONENTS[tab];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold" style={{ color: ACCENT }}>
          🧬 BCKDHB Epilepsy
        </h4>
        <div className="text-muted small">
          Maple Syrup Urine Disease Type 1B · Branched-Chain Keto Acid Dehydrogenase E1β Subunit Deficiency ·
          BCKDH complex E1 structural subunit (α₂β₂ heterotetramer with BCKDHA E1α — NO TPP active site) ·
          Irreversible BCAA catabolism step 2 BLOCKED · Alloisoleucine PATHOGNOMONIC ·
          Leucine neurotoxicity (LAT1 / mTOR / myelinopathy) · Thiamine-responsive ~5% (lower than BCKDHA 10–20%) ·
          VPA ABSOLUTE CI (triple mechanism) ·
          Autosomal Recessive · 6q14.1 · OMIM *248611 / #248600 · ~1:185,000 globally
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      <Active />
    </div>
  );
}
