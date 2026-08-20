'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4e342e';   // dark brown — αKGDH / mitochondrial E2 succinyltransferase
const ACCENT2 = '#b71c1c';   // dark red — severe phenotype / lactic acidosis
const ACCENT3 = '#e65100';   // deep orange — 2-oxoglutarate elevated / HAZARD triggers
const ACCENT4 = '#0277bd';   // steel blue — lipoic acid / riboflavin / treatments
const ACCENT5 = '#4a148c';   // deep purple — TCA mechanism / definitions / E2 lipoamide
const ACCENT6 = '#1b5e20';   // dark green — NORMAL key negatives (BCAA / glycine / DLD / PDH / E1)
const ACCENT7 = '#37474f';   // dark slate — L:P intermediate / E1 intact / glutamate
const ACCENT8 = '#1565c0';   // deep blue — lipoamide / Lys114 / Lys150 / lipoic acid direct

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
  const { data, err } = useFetch(`${API}/api/dlst/overview`);
  if (err)  return <div className="alert alert-danger">Error loading overview.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const k = data.kpis || {};
  const tca = data.tca_cycle_position || {};
  return (
    <div>
      <Alert
        text="🔑 KEY BIOMARKER: Urine 2-oxoglutarate (α-KG) MARKEDLY elevated (180–1800 µmol/mmol Cr; normal <30) — PATHOGNOMONIC for αKGDH complex deficiency. TCA step 4 block: α-KG cannot be turned over because DLST E2 (succinyltransferase) is absent. ALWAYS request urine organic acids (GC-MS) in any child with lactic acidosis + Leigh syndrome."
        variant="info"
      />
      <Alert
        text="🧪 PATHOGNOMONIC ASSAY PATTERN: Isolated OGDH E1 activity NORMAL (E1 is intact — can still form succinyl-TPP) + αKGDH COMPLEX activity <10% (E2 missing → complex stalls). This E1-normal/complex-low combination is THE diagnostic fingerprint of DLST (E2) deficiency, distinguishing it from OGDH (E1) deficiency where isolated E1 activity is severely reduced."
        variant="primary"
      />
      <Alert
        text="✅ BCAA NORMAL · GLYCINE NORMAL · DLD/E3 free activity NORMAL · PDH activity NORMAL — all KEY NEGATIVES. DLST deficiency blocks ONLY αKGDH E2. BCKDH uses DBT (separate E2 gene), GCS uses GCSH/AMT, PDH uses DLAT (separate E2). DLD (shared E3) protein is intact."
        variant="success"
      />
      <Alert
        text="💊 LIPOIC ACID (Level B) — MOST DIRECTLY RELEVANT in DLST deficiency. Lipoamide (covalently attached to Lys114-L1 and Lys150-L2 on DLST) is the primary deficient cofactor. Stronger rationale than in OGDH or DLAT deficiency. Riboflavin (B2) Level B for DLD E3. Thiamine Level C (E1 is intact; thiamine targets E1 cofactor TPP, not E2 lipoamide)."
        variant="warning"
      />

      <Section title="Gene Summary" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'38%'}}>Gene</td><td>{data.gene} ({data.protein})</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{data.locus} — Autosomal (chromosome 14q24.3)</td></tr>
            <tr><td className="fw-bold">Protein length</td><td>{data.aa_length} aa (mature protein); E2 succinyltransferase; TWO lipoyl domains: Lys114-L1 and Lys150-L2</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td style={{color: ACCENT8, fontWeight:'600'}}>{data.cofactor}</td></tr>
            <tr><td className="fw-bold">Mechanism</td><td>{data.mechanism}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease}</td></tr>
            <tr><td className="fw-bold">Inheritance</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.inheritance}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT}}>DLST vs OGDH</td><td style={{color: ACCENT7, fontWeight:'600'}}>{data.key_distinguishing_from_ogdh}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT2}}>Critical biochemical key</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.key_distinguishing_feature}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT6}}>Structural brain hallmark</td><td style={{color: ACCENT6}}>{data.structural_brain_hallmark}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="αKGDH Complex Components" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Component</th><th>Gene(s)</th><th>Function in DLST deficiency context</th></tr>
            </thead>
            <tbody>
              {(data.akgdh_complex_components || []).map((row, i) => (
                <tr key={i} style={
                  row.component.includes('DLST') ? {backgroundColor:'#efebe9', fontWeight:'600'} :
                  row.component.includes('DLD') ? {backgroundColor:'#e8f5e9'} :
                  {backgroundColor:'#e3f2fd'}
                }>
                  <td className="fw-bold" style={
                    row.component.includes('DLST') ? {color: ACCENT} :
                    row.component.includes('DLD') ? {color: ACCENT6} :
                    {color: ACCENT8}
                  }>{row.component}</td>
                  <td style={{fontSize:11}}>{
                    row.component.includes('DLST') ? 'DLST ← THIS GENE (DEFICIENT)' :
                    row.component.includes('OGDH') ? 'OGDH (E1 INTACT — isolated E1 activity normal in DLST deficiency)' :
                    'DLD (E3 shared with PDH, BCKDH, GCS — INTACT in DLST deficiency)'
                  }</td>
                  <td>{row.function}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="TCA Cycle Position — Block at Step 4 (E2 Transfer Step)" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>TCA Step</th><th>Reaction</th></tr>
            </thead>
            <tbody>
              <tr style={{backgroundColor:'#e3f2fd'}}>
                <td className="text-muted">Step 3</td>
                <td className="small">{tca.step_3}</td>
              </tr>
              <tr style={{backgroundColor:'#ffebee'}}>
                <td className="fw-bold" style={{color: ACCENT2}}>Step 4 — BLOCKED at E2</td>
                <td className="fw-bold small" style={{color: ACCENT2}}>{tca.step_4_blocked}</td>
              </tr>
              <tr style={{backgroundColor:'#fff8e1'}}>
                <td className="fw-bold" style={{color: ACCENT7}}>E1 intact (E2 absent)</td>
                <td className="small" style={{color: ACCENT7}}>OGDH E1 forms succinyl-TPP from α-KG normally — E1 is not the deficient protein. Succinyl-TPP CANNOT transfer to absent DLST E2 lipoamide arm → complex stalls → α-KG accumulates.</td>
              </tr>
              <tr>
                <td className="text-muted">Step 5</td><td className="small">{tca.step_5}</td>
              </tr>
              <tr>
                <td className="text-muted">Step 6</td><td className="small">{tca.step_6}</td>
              </tr>
              <tr style={{backgroundColor:'#e8f5e9'}}>
                <td className="fw-bold" style={{color: ACCENT6}}>Partial compensation</td>
                <td className="small" style={{color: ACCENT6}}>{tca.consequence}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Cohort KPIs (n=40)" color={ACCENT}>
        <div className="row g-2">
          <KPI label="Cohort N" value={k.cohort_n} color={ACCENT} />
          <KPI label="Male patients %" value={`${k.male_pct}%`} color={ACCENT5} />
          <KPI label="Avg plasma lactate (mmol/L)" value={k.avg_plasma_lactate_mmol} color={ACCENT2} />
          <KPI label="Avg plasma pyruvate (mmol/L)" value={k.avg_plasma_pyruvate_mmol} color={ACCENT3} />
          <KPI label="Avg L:P ratio (intermediate 15–30)" value={k.avg_lp_ratio} color={ACCENT7} />
          <KPI label="L:P elevated >20 %" value={`${k.elevated_lp_ratio_gt20_pct}%`} color={ACCENT3} />
          <KPI label="Avg urine 2-OG (µmol/mmolCr)" value={k.avg_urine_2_oxoglutarate_umol_mmolCr} color={ACCENT2} />
          <KPI label="OGDH E1 isolated NORMAL %" value={`${k.ogdh_e1_isolated_normal_pct}%`} color={ACCENT6} />
          <KPI label="Leigh lesions (MRI) %" value={`${k.leigh_lesions_pct}%`} color={ACCENT2} />
          <KPI label="Basal ganglia lesions %" value={`${k.basal_ganglia_lesions_pct}%`} color={ACCENT2} />
          <KPI label="On lipoic acid %" value={`${k.on_lipoic_acid_pct}%`} color={ACCENT4} />
          <KPI label="Lipoic acid responsive %" value={`${k.lipoic_acid_responsive_pct}%`} color={ACCENT4} />
          <KPI label="On thiamine %" value={`${k.on_thiamine_pct}%`} color={ACCENT8} />
          <KPI label="DRE %" value={`${k.dre_pct}%`} color={ACCENT2} />
        </div>
      </Section>

      <Section title="High-Risk Drugs / Exposures" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <Alert
            key={i}
            text={`${d.risk === 'ABSOLUTE CI' ? '⛔' : d.risk === 'EXTREME HAZARD' ? '🚨' : d.risk === 'HAZARD' ? '🔴' : '⚠️'} ${d.drug} — ${d.risk}: ${d.mechanism}`}
            variant={d.risk === 'ABSOLUTE CI' ? 'danger' : d.risk === 'HAZARD' ? 'danger' : 'warning'}
          />
        ))}
      </Section>

      <Section title="Structural Brain Anomalies" color={ACCENT6}>
        {(data.structural_anomalies || []).map((a, i) => (
          <div key={i} className="mb-2">
            <PctBar label={a.anomaly} pct={a.frequency_pct} color={ACCENT6} />
            <div className="small text-muted ms-1">{a.significance}</div>
          </div>
        ))}
      </Section>

      <Section title="Phenotype Distribution" color={ACCENT}>
        {(data.phenotype_distribution || []).map((p, i) => (
          <PctBar key={i} label={p.label} pct={p.pct} color={p.color || ACCENT} />
        ))}
      </Section>
    </div>
  );
}

function PhenotypeTab() {
  const { data, err } = useFetch(`${API}/api/dlst/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Biomarkers — αKGDH E2 Biochemical Fingerprint" color={ACCENT6}>
        <Alert
          text="URINE ORGANIC ACIDS (GC-MS) MANDATORY — 2-oxoglutarate is the pathognomonic marker. CRITICAL ASSAY: Request both ISOLATED OGDH E1 activity (should be NORMAL in DLST deficiency) AND αKGDH COMPLEX activity (should be <10%). This E1-normal/complex-low pattern is pathognomonic for DLST (E2) deficiency. Also request plasma amino acids, simultaneous plasma lactate+pyruvate (L:P ratio), and fibroblast enzyme studies."
          variant="info"
        />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Biomarker</th><th>Normal</th><th>DLST range</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {(data.biomarkers || []).map((b, i) => (
                <tr key={i} style={
                  b.name.includes('2-oxoglutarate') ? {backgroundColor:'#fff3e0', fontWeight:'600'} :
                  b.name.includes('Isolated OGDH E1') ? {backgroundColor:'#e8f5e9', fontWeight:'600'} :
                  b.name.includes('αKGDH complex') ? {backgroundColor:'#fce4ec', fontWeight:'600'} :
                  b.name.includes('L:P') ? {backgroundColor:'#fff8e1'} :
                  b.name.includes('BCAA') || b.name.includes('Glycine') || b.name.includes('DLD/E3 free') || b.name.includes('PDH enzyme') ? {backgroundColor:'#e8f5e9'} :
                  {}
                }>
                  <td className="fw-bold">{b.name}</td>
                  <td className="text-success small">{b.normal}</td>
                  <td className={
                    b.dlst_range && (b.dlst_range.includes('NORMAL') || b.dlst_range.includes('intact') || b.dlst_range.includes('mildly elevated'))
                      ? (b.dlst_range.includes('PATHOGNOMONIC') || b.dlst_range.includes('MARKEDLY') || b.dlst_range.includes('<10%') ? 'text-danger small fw-bold' : 'text-success small fw-bold')
                      : 'text-danger small'
                  }>{b.dlst_range}</td>
                  <td className="small">{b.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Key Variants" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Variant</th><th>Effect / Structural impact</th><th>Phenotype</th></tr>
            </thead>
            <tbody>
              {(data.key_variants || []).map((v, i) => (
                <tr key={i} style={
                  v.variant.includes('splice') || v.variant.includes('del') ? {backgroundColor:'#fce4ec'} :
                  v.variant.includes('Arg415') ? {backgroundColor:'#fff3e0'} : {}
                }>
                  <td className="fw-bold font-monospace">{v.variant}</td>
                  <td>{v.effect}</td>
                  <td><span className="badge" style={{backgroundColor: ACCENT3, color:'#000'}}>{v.phenotype}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">Pink = null/splice/deletion — no lipoic acid response expected (no DLST protein). Orange = lipoyl-tethering domain variants — may show partial response to lipoic acid supplementation.</div>
      </Section>

      <Section title="Patient Sample (n=10)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Lactate</th><th>Pyruvate</th><th>L:P</th><th>2-OG (µmol/mmolCr)</th>
                <th>E1 normal</th><th>Leigh</th><th>Lipoic acid</th><th>LA resp</th><th>Thiamine</th><th>VPA avoided</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map((p, i) => (
                <tr key={i} style={p.lipoic_acid_responsive ? {backgroundColor:'#e3f2fd'} : {}}>
                  <td>{p.id}</td>
                  <td><span className="badge" style={{backgroundColor: p.sex==='M'?'#1565c0':'#c62828'}}>{p.sex}</span></td>
                  <td style={{fontSize:10}}>{p.phenotype.split('(')[0].trim()}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{color:ACCENT2}}>{p.plasma_lactate_mmol}</td>
                  <td style={{color:ACCENT7}}>{p.plasma_pyruvate_mmol}</td>
                  <td style={{color: p.lp_ratio > 25 ? ACCENT2 : ACCENT7, fontWeight:'600'}}>{p.lp_ratio}</td>
                  <td style={{color:ACCENT3, fontWeight:'600'}}>{p.urine_2_oxoglutarate}</td>
                  <td style={{color: p.ogdh_e1_isolated_normal ? ACCENT6 : ACCENT2, fontWeight:'600'}}>{p.ogdh_e1_isolated_normal ? '✅ NORMAL' : '↓ Low'}</td>
                  <td>{p.leigh_lesions ? '✅' : '—'}</td>
                  <td>{p.on_lipoic_acid ? '✅' : '—'}</td>
                  <td style={{color: p.lipoic_acid_responsive ? ACCENT4 : 'inherit'}}>{p.lipoic_acid_responsive ? '✅' : '—'}</td>
                  <td>{p.on_thiamine ? '✅' : '—'}</td>
                  <td>{p.vpa_avoided ? '✅' : '⚠️'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">Blue rows = lipoic acid responsive. E1 normal = isolated OGDH E1 activity normal (pathognomonic for DLST). L:P &gt;25 shown in red. 2-OG = urine 2-oxoglutarate.</div>
      </Section>
    </div>
  );
}

function SeizuresTab() {
  const { data, err } = useFetch(`${API}/api/dlst/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Seizure Types" color={ACCENT2}>
        {(data.seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.pct} color={ACCENT2} />
        ))}
      </Section>

      <Section title="Metabolic Crisis Triggers" color={ACCENT3}>
        <Alert
          text="⚠️ HIGH-PROTEIN MEAL IS A KEY TRIGGER IN DLST DEFICIENCY (same as OGDH deficiency) — amino acid catabolism generates glutamate → α-KG (via glutamate dehydrogenase), flooding the blocked αKGDH step 4. The blocked E2 cannot transfer succinyl away, so α-KG accumulates acutely. Moderate protein restriction and lipoic acid pre-loading are essential during illness."
          variant="warning"
        />
        {(data.trigger_types || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT3} />
        ))}
      </Section>

      <Section title="Phenotype Classes" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Phenotype</th><th>Frequency %</th><th>Sex bias</th><th>Key features</th></tr>
            </thead>
            <tbody>
              <tr>
                <td className="fw-bold" style={{color:'#b71c1c'}}>Leigh Syndrome (Infantile)</td>
                <td>45%</td>
                <td>None (AR, equal)</td>
                <td>Symmetric BG + brainstem lesions; putamen and caudate preferentially affected; psychomotor regression; 2-oxoglutarate markedly elevated; E1 isolated activity normal; lactic acidosis; most common DLST phenotype</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT3}}>Severe Neonatal</td>
                <td>35%</td>
                <td>None (AR, equal)</td>
                <td>Neonatal lactic acidosis; marked 2-oxoglutaric aciduria; E1 isolated activity normal (key diagnostic); null variants (splice, exonic deletion); early metabolic decompensation; poor prognosis without early lipoic acid + metabolic management</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT4}}>Childhood Episodic Encephalopathy</td>
                <td>15%</td>
                <td>None (AR, equal)</td>
                <td>Illness/protein-load-triggered metabolic crises; relatively normal interictal development; lipoyl-tethering domain variants more common; partial lipoic acid response possible; exercise intolerance</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color:'#558b2f'}}>Mild/Juvenile (Partial Deficiency)</td>
                <td>5%</td>
                <td>None (AR, equal)</td>
                <td>Chronic partial DLST E2 deficiency; intellectual disability; ataxia; 2-oxoglutarate mildly-moderately elevated; sometimes partially lipoic acid responsive; ultra-rare — fewer than 5 confirmed cases</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function TreatmentsTab() {
  const { data, err } = useFetch(`${API}/api/dlst/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="💊 LIPOIC ACID (Level B) — MOST DIRECTLY RELEVANT in DLST deficiency. Lipoamide (the reduced form of lipoic acid) is covalently attached to DLST at Lys114 (L1) and Lys150 (L2). These lipoamide-carrying arms ARE the primary catalytic cofactor of DLST E2. For partial-loss variants with residual DLST protein, supplemental lipoic acid may augment residual lipoamide arm function. Dose: 100–600 mg/day. STRONGER RATIONALE than in OGDH (E1), DLAT (acetyl, not succinyl), or PDH deficiency."
        variant="primary"
      />
      <Alert
        text="✅ RIBOFLAVIN (B2) Level B — FAD cofactor for DLD (shared E3). DLD is intact in DLST deficiency, but riboflavin supports optimal DLD function. Co-administered with lipoic acid. Safe and inexpensive."
        variant="info"
      />
      <Alert
        text="⚠️ THIAMINE (B1) Level C — TPP is the cofactor for OGDH E1 (not DLST E2). In DLST deficiency, E1 (OGDH) is INTACT and can already form succinyl-TPP efficiently. Thiamine does not address the E2 transfer block. Trial is reasonable but response rate is LOWER than in OGDH deficiency (~22% vs 40–55% for OGDH). Do not substitute thiamine for lipoic acid as first-line."
        variant="warning"
      />
      <Alert
        text="⚠️ KETOGENIC DIET — Level C / EXPERIMENTAL in DLST deficiency (same as OGDH). KD provides ketones → acetyl-CoA → TCA entry (step 1), but TCA is blocked at step 4 (αKGDH E2). KD does NOT bypass DLST block as it bypasses PDH block. Not first-line. VPA: CAUTION (not absolute CI). High-protein diet: HAZARD."
        variant="warning"
      />

      <Section title="Treatment Ladder" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>Level</th><th>Mechanism in DLST deficiency</th><th>Response %</th></tr>
            </thead>
            <tbody>
              {(data.treatments || []).map((t, i) => (
                <tr key={i} style={t.drug.includes('Lipoic') ? {backgroundColor:'#e3f2fd'} : {}}>
                  <td className="fw-bold" style={{color: t.color}}>{t.drug}</td>
                  <td><span className="badge" style={{backgroundColor: t.level==='A'?'#1b5e20':t.level==='B'?'#1565c0':'#827717'}}>{t.level}</span></td>
                  <td style={{fontSize:11}}>{t.note}</td>
                  <td>
                    <div className="progress" style={{height:12}}>
                      <div className="progress-bar" style={{width:`${t.response_pct}%`, backgroundColor: t.color}} />
                    </div>
                    <small>{t.response_pct}%</small>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="DLST vs OGDH — Treatment Comparison (Same Complex, Different Subunit)" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>DLST (αKGDH E2)</th><th>OGDH (αKGDH E1)</th><th>Key difference</th></tr>
            </thead>
            <tbody>
              <tr style={{backgroundColor:'#e3f2fd'}}>
                <td>Lipoic acid</td>
                <td className="text-primary fw-bold">Level B — MOST DIRECT (lipoamide on Lys114/Lys150 is the primary DLST cofactor)</td>
                <td className="text-muted">Level C — experimental (lipoic acid targets E2; indirect when E1 is deficient)</td>
                <td>Lipoic acid directly addresses DLST (E2) lipoamide arm. In OGDH deficiency, E2/DLST is intact — lipoic acid at E2 is less relevant when E1 is the deficient protein.</td>
              </tr>
              <tr>
                <td>Thiamine (B1)</td>
                <td className="text-warning fw-bold">Level C — ~22% response (E1 is intact; TPP already adequate)</td>
                <td className="text-success fw-bold">Level A — ~40–55% response (TPP is E1 cofactor; E1 is deficient)</td>
                <td>TPP is the E1/OGDH cofactor. In DLST deficiency, E1 is intact and TPP-sufficient — thiamine benefits less. In OGDH deficiency, E1 is deficient and TPP augmentation directly helps.</td>
              </tr>
              <tr>
                <td>Riboflavin (B2)</td>
                <td className="text-primary fw-bold">Level B (DLD E3 support)</td>
                <td className="text-primary fw-bold">Level B (DLD E3 support)</td>
                <td>DLD (E3, shared) is intact in both DLST and OGDH deficiency; riboflavin FAD supplementation provides comparable support in both conditions.</td>
              </tr>
              <tr>
                <td>Ketogenic diet</td>
                <td className="text-warning fw-bold">Level C — does not bypass αKGDH E2 block</td>
                <td className="text-warning fw-bold">Level C — does not bypass αKGDH E1 block</td>
                <td>KD does NOT bypass αKGDH block at step 4 (E1 or E2). Unlike PDH deficiency where KD is Level A (ketones → acetyl-CoA → TCA). Both DLST and OGDH deficiency: KD experimental only.</td>
              </tr>
              <tr>
                <td>VPA</td>
                <td className="text-warning fw-bold">CAUTION — not absolute CI</td>
                <td className="text-warning fw-bold">CAUTION — not absolute CI</td>
                <td>Same in both DLST and OGDH: mitochondrial hepatotoxicity risk; KD not first-line (so VPA is caution, not absolute CI as in PDH deficiency). Prefer LEV.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="High-Risk Drugs / Exposures" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <Alert
            key={i}
            text={`${d.risk === 'ABSOLUTE CI' ? '⛔' : d.risk === 'HAZARD' ? '🔴' : '⚠️'} ${d.drug} — ${d.risk}: ${d.mechanism}`}
            variant={d.risk === 'ABSOLUTE CI' ? 'danger' : d.risk === 'HAZARD' ? 'danger' : 'warning'}
          />
        ))}
      </Section>
    </div>
  );
}

function DefinitionsTab() {
  const { data, err } = useFetch(`${API}/api/dlst/definitions`);
  if (err)  return <div className="alert alert-danger">Error loading definitions.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const gc = data.gene_card || {};
  const dt = data.diagnostic_thresholds || {};
  return (
    <div>
      <Section title="Gene Card" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {Object.entries(gc).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold" style={{width:'32%'}}>{k.replace(/_/g,' ')}</td>
                <td style={
                  k === 'Primary biomarker' ? {color: ACCENT3, fontWeight:'600'} :
                  k === 'Pathognomonic assay pattern' ? {color: ACCENT2, fontWeight:'600'} :
                  k === 'Key negative biomarkers' ? {color: ACCENT6, fontWeight:'600'} :
                  k === 'Most directly relevant treatment' ? {color: ACCENT8, fontWeight:'600'} :
                  k === 'Key distinguishing assay from OGDH' ? {color: ACCENT5, fontWeight:'600'} :
                  {}
                }>{Array.isArray(v) ? v.join(' · ') : String(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>

      <Section title="Key Concepts" color={ACCENT5}>
        {(data.key_concepts || []).map((c, i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{color: ACCENT5}}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        ))}
      </Section>

      <Section title="Diagnostic Thresholds" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Parameter</th><th>Value / Threshold</th></tr>
            </thead>
            <tbody>
              {Object.entries(dt).map(([k, v], i) => (
                <tr key={i} style={
                  k.includes('oxoglutarate') ? {backgroundColor:'#fff3e0', fontWeight:'600'} :
                  k.includes('isolated_ogdh_e1') ? {backgroundColor:'#e8f5e9', fontWeight:'600'} :
                  k.includes('akgdh_complex') ? {backgroundColor:'#fce4ec', fontWeight:'600'} :
                  k.includes('bcaa') || k.includes('glycine') || k.includes('dld_free') || k.includes('pdh_enzyme') ? {backgroundColor:'#e8f5e9'} :
                  k.includes('lp_ratio') ? {backgroundColor:'#fff8e1'} : {}
                }>
                  <td className="fw-bold" style={{width:'40%', fontSize:11}}>{k.replace(/_/g,' ')}</td>
                  <td className="small">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Condition</th><th>Key distinguishing feature from DLST</th></tr>
            </thead>
            <tbody>
              {(data.differential_diagnosis || []).map((d, i) => (
                <tr key={i} style={
                  d.disease.includes('OGDH') ? {backgroundColor:'#efebe9'} :
                  d.disease.includes('DLD') ? {backgroundColor:'#e8f5e9'} :
                  d.disease.includes('DLAT') || d.disease.includes('PDHA1') ? {backgroundColor:'#e0f2f1'} : {}
                }>
                  <td className="fw-bold" style={{fontSize:11, color:
                    d.disease.includes('OGDH') ? ACCENT :
                    d.disease.includes('DLD') ? ACCENT6 :
                    d.disease.includes('DLAT') || d.disease.includes('PDHA1') ? '#00695c' : 'inherit'
                  }}>{d.disease}</td>
                  <td className="small">{d.distinguishing_features}</td>
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
export default function DlstPage() {
  const [tab, setTab] = useState(0);

  const panels = [
    <OverviewTab key="ov" />,
    <PhenotypeTab key="ph" />,
    <SeizuresTab key="sz" />,
    <TreatmentsTab key="tx" />,
    <DefinitionsTab key="df" />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold" style={{ color: ACCENT }}>
          🧬 DLST Epilepsy — Dihydrolipoamide S-Succinyltransferase Deficiency (αKGDH Complex E2 Subunit)
        </h4>
        <div className="text-muted small">
          DLST · 14q24.3 · Autosomal Recessive · ~453 aa · E2 (succinyltransferase, two lipoyl domains Lys114-L1 / Lys150-L2) ·
          αKGDH complex: OGDH (E1) + DLST (E2 ← THIS) + DLD (E3 shared) ·
          TCA step 4: α-KG → succinyl-CoA (BLOCKED at E2 transfer step) ·
          DLST LOF → α-KG accumulates → 2-oxoglutaric aciduria (urine 2-OG 180–1800 µmol/mmolCr — PATHOGNOMONIC) ·
          L:P mildly elevated (15–30; intermediate same as OGDH) ·
          PATHOGNOMONIC ASSAY: isolated OGDH E1 activity NORMAL + αKGDH complex &lt;10% ·
          BCAA NORMAL · Glycine NORMAL · DLD free NORMAL · PDH NORMAL ·
          Lipoic acid (B) MOST DIRECTLY RELEVANT (lipoamide on Lys114/Lys150) ·
          Thiamine Level C only (E1 is intact; TPP not DLST cofactor) ·
          OMIM *126063 / #203740 · ~10–15 cases worldwide (2026)
        </div>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {panels[tab]}
    </div>
  );
}
