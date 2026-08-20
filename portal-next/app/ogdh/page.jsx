'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#4e342e';   // dark brown — TCA cycle / mitochondrial matrix / αKGDH
const ACCENT2 = '#b71c1c';   // dark red — severe phenotype / lactic acidosis / neonatal death
const ACCENT3 = '#e65100';   // deep orange — 2-oxoglutarate elevated / HAZARD triggers
const ACCENT4 = '#1565c0';   // deep blue — thiamine TPP / riboflavin / treatments
const ACCENT5 = '#4a148c';   // deep purple — TCA mechanism / definitions / E1 decarboxylase
const ACCENT6 = '#1b5e20';   // dark green — NORMAL key negatives (BCAA / glycine / DLD / PDH)
const ACCENT7 = '#37474f';   // dark slate — L:P intermediate / DLD intact / glutamate

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
  const { data, err } = useFetch(`${API}/api/ogdh/overview`);
  if (err)  return <div className="alert alert-danger">Error loading overview.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const k = data.kpis || {};
  const tca = data.tca_cycle_position || {};
  return (
    <div>
      <Alert
        text="🔑 KEY BIOMARKER: Urine 2-oxoglutarate (α-KG) MARKEDLY elevated (200–2200 µmol/mmol Cr; normal <30) — PATHOGNOMONIC for αKGDH complex deficiency. TCA step 4 block: α-KG cannot be decarboxylated by OGDH E1 (TPP-dependent). ALWAYS request urine organic acids (GC-MS) in any child with lactic acidosis + Leigh syndrome."
        variant="info"
      />
      <Alert
        text="⚗️ L:P RATIO INTERMEDIATE (15–30) — NOT the normal 10–20 of PDH block, NOT the dramatically elevated >25 of Complex I. OGDH occupies a middle position: TCA impairment builds NADH, but PDH is intact so pyruvate is consumed normally. Do NOT use L:P alone to rule out OGDH — measure 2-oxoglutarate."
        variant="primary"
      />
      <Alert
        text="✅ BCAA NORMAL · GLYCINE NORMAL · DLD/E3 free activity NORMAL · PDH activity NORMAL — all KEY NEGATIVES. OGDH deficiency blocks ONLY αKGDH E1. BCKDH (BCAA catabolism), GCS (glycine), and PDH (pyruvate) all remain intact because DLD (E3 shared enzyme) and their respective E1 subunits are unaffected."
        variant="success"
      />
      <Alert
        text="💊 THIAMINE (B1) LEVEL A — TPP is the OGDH E1 cofactor (same mechanism as PDHA1 for PDH E1α). Trial mandatory in ALL patients (100–600 mg/day). ~40–55% partial response for TPP-binding domain variants (higher than most PDH subunit deficiencies). Riboflavin (B2) Level B — FAD cofactor for shared DLD E3."
        variant="warning"
      />

      <Section title="Gene Summary" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'38%'}}>Gene</td><td>{data.gene} ({data.protein})</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{data.locus} — Autosomal (chromosome 7p)</td></tr>
            <tr><td className="fw-bold">Protein length</td><td>{data.aa_length} aa (includes mitochondrial targeting sequence); E1 decarboxylase; TPP-dependent</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td>{data.cofactor}</td></tr>
            <tr><td className="fw-bold">Mechanism</td><td>{data.mechanism}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease}</td></tr>
            <tr><td className="fw-bold">Inheritance</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.inheritance}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT}}>OGDH vs PDH complex</td><td style={{color: ACCENT7, fontWeight:'600'}}>{data.key_distinguishing_from_pdh}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT2}}>Critical biochemical key</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.key_distinguishing_feature}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT6}}>Structural brain hallmark</td><td style={{color: ACCENT6}}>{data.structural_brain_hallmark}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="αKGDH Complex Components" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Component</th><th>Gene(s)</th><th>Function</th></tr>
            </thead>
            <tbody>
              {(data.akgdh_complex_components || []).map((row, i) => (
                <tr key={i} style={row.component.includes('OGDH') ? {backgroundColor:'#efebe9'} : row.component.includes('DLD') ? {backgroundColor:'#e8f5e9'} : {}}>
                  <td className="fw-bold" style={row.component.includes('OGDH') ? {color: ACCENT} : row.component.includes('DLD') ? {color: ACCENT6} : {}}>{row.component}</td>
                  <td style={{fontSize:11}}>{
                    row.component.includes('OGDH') ? 'OGDH ← THIS GENE' :
                    row.component.includes('DLST') ? 'DLST' : 'DLD (shared with PDH, BCKDH, GCS)'
                  }</td>
                  <td>{row.function}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="TCA Cycle Position — Block at Step 4" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>TCA Step</th><th>Reaction</th></tr>
            </thead>
            <tbody>
              <tr><td className="text-muted">Step 3</td><td className="small">{tca.step_3}</td></tr>
              <tr style={{backgroundColor:'#ffebee'}}>
                <td className="fw-bold" style={{color: ACCENT2}}>Step 4 — BLOCKED</td>
                <td className="fw-bold small" style={{color: ACCENT2}}>{tca.step_4_blocked}</td>
              </tr>
              <tr><td className="text-muted">Step 5</td><td className="small">{tca.step_5}</td></tr>
              <tr><td className="text-muted">Step 6</td><td className="small">{tca.step_6}</td></tr>
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
          <KPI label="Leigh lesions (MRI) %" value={`${k.leigh_lesions_pct}%`} color={ACCENT2} />
          <KPI label="Basal ganglia lesions %" value={`${k.basal_ganglia_lesions_pct}%`} color={ACCENT2} />
          <KPI label="On thiamine %" value={`${k.on_thiamine_pct}%`} color={ACCENT4} />
          <KPI label="Thiamine responsive %" value={`${k.thiamine_responsive_pct}%`} color={ACCENT4} />
          <KPI label="DRE %" value={`${k.dre_pct}%`} color={ACCENT2} />
        </div>
      </Section>

      <Section title="High-Risk Drugs / Exposures" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <Alert
            key={i}
            text={`${d.risk === 'ABSOLUTE CI' ? '⛔' : d.risk === 'EXTREME HAZARD' ? '🚨' : d.risk === 'HAZARD' ? '🔴' : '⚠️'} ${d.drug} — ${d.risk}: ${d.mechanism}`}
            variant={d.risk === 'ABSOLUTE CI' ? 'danger' : d.risk === 'HAZARD' ? 'danger' : d.risk === 'CAUTION' ? 'warning' : 'warning'}
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
  const { data, err } = useFetch(`${API}/api/ogdh/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Biomarkers — αKGDH Biochemical Fingerprint" color={ACCENT6}>
        <Alert
          text="URINE ORGANIC ACIDS (GC-MS) MANDATORY — 2-oxoglutarate is the pathognomonic marker; it is not detectable on standard amino acid screens or plasma lactate/pyruvate alone. Request urine organic acids, plasma amino acids, plasma lactate+pyruvate (simultaneous L:P ratio), and fibroblast αKGDH enzyme activity for definitive diagnosis."
          variant="info"
        />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Biomarker</th><th>Normal</th><th>OGDH range</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {(data.biomarkers || []).map((b, i) => (
                <tr key={i} style={
                  b.name.includes('2-oxoglutarate') ? {backgroundColor:'#fff3e0', fontWeight:'600'} :
                  b.name.includes('L:P') ? {backgroundColor:'#fff8e1'} :
                  b.name.includes('BCAA') || b.name.includes('Glycine') || b.name.includes('DLD/E3 free') || b.name.includes('PDH enzyme') ? {backgroundColor:'#e8f5e9'} :
                  b.name.includes('αKGDH complex') ? {backgroundColor:'#fce4ec'} : {}
                }>
                  <td className="fw-bold">{b.name}</td>
                  <td className="text-success small">{b.normal}</td>
                  <td className={
                    b.ogdh_range && (b.ogdh_range.includes('NORMAL') || b.ogdh_range.includes('intact') || b.ogdh_range.includes('mildly elevated'))
                      ? (b.ogdh_range.includes('PATHOGNOMONIC') || b.ogdh_range.includes('MARKEDLY') ? 'text-danger small fw-bold' : 'text-success small fw-bold')
                      : 'text-danger small'
                  }>{b.ogdh_range}</td>
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
                <tr key={i} style={v.variant.includes('splice') ? {backgroundColor:'#fce4ec'} : v.variant.includes('Arg263') ? {backgroundColor:'#e8f5e9'} : {}}>
                  <td className="fw-bold font-monospace">{v.variant}</td>
                  <td>{v.effect}</td>
                  <td><span className="badge" style={{backgroundColor: ACCENT3, color:'#000'}}>{v.phenotype}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">Green = thiamine-responsive TPP-binding domain variant. Pink = null/splice — no thiamine response expected.</div>
      </Section>

      <Section title="Patient Sample (n=10)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Lactate</th><th>Pyruvate</th><th>L:P</th><th>2-OG (µmol/mmolCr)</th>
                <th>Leigh</th><th>Thiamine</th><th>B1 resp</th><th>KD</th><th>VPA avoided</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map((p, i) => (
                <tr key={i} style={p.thiamine_responsive ? {backgroundColor:'#e8f5e9'} : {}}>
                  <td>{p.id}</td>
                  <td><span className="badge" style={{backgroundColor: p.sex==='M'?'#1565c0':'#c62828'}}>{p.sex}</span></td>
                  <td style={{fontSize:10}}>{p.phenotype.split('(')[0].trim()}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{color:ACCENT2}}>{p.plasma_lactate_mmol}</td>
                  <td style={{color:ACCENT7}}>{p.plasma_pyruvate_mmol}</td>
                  <td style={{color: p.lp_ratio > 25 ? ACCENT2 : ACCENT7, fontWeight:'600'}}>{p.lp_ratio}</td>
                  <td style={{color:ACCENT3, fontWeight:'600'}}>{p.urine_2_oxoglutarate}</td>
                  <td>{p.leigh_lesions ? '✅' : '—'}</td>
                  <td>{p.on_thiamine ? '✅' : '—'}</td>
                  <td style={{color: p.thiamine_responsive ? ACCENT6 : 'inherit'}}>{p.thiamine_responsive ? '✅' : '—'}</td>
                  <td>{p.on_kd ? '✅' : '—'}</td>
                  <td>{p.vpa_avoided ? '✅' : '⚠️'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">Green rows = thiamine-responsive. L:P &gt;25 shown in red. 2-OG = urine 2-oxoglutarate (µmol/mmol Cr).</div>
      </Section>
    </div>
  );
}

function SeizuresTab() {
  const { data, err } = useFetch(`${API}/api/ogdh/breakdown`);
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
          text="⚠️ HIGH-PROTEIN MEAL IS A KEY TRIGGER IN OGDH DEFICIENCY — amino acid catabolism generates glutamate → α-KG (via glutamate dehydrogenase), flooding the blocked αKGDH step. Unlike PDH deficiency (where carbohydrate/glucose is the primary trigger), OGDH crisis is driven by PROTEIN CATABOLISM. Moderate protein restriction and thiamine pre-loading are essential during illness."
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
                <td>40%</td>
                <td>None (AR, equal)</td>
                <td>Symmetric BG + brainstem lesions; putamen and caudate preferentially affected; psychomotor regression; marked 2-oxoglutarate elevation; lactic acidosis; most common OGDH phenotype</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT3}}>Severe Neonatal</td>
                <td>30%</td>
                <td>None (AR, equal)</td>
                <td>Neonatal lactic acidosis; marked 2-oxoglutaric aciduria; early metabolic decompensation; null variants (splice, frameshift); poor prognosis without early thiamine + metabolic management</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT4}}>Childhood Episodic Encephalopathy</td>
                <td>20%</td>
                <td>None (AR, equal)</td>
                <td>Illness/protein-load-triggered metabolic crises; relatively normal interictal development; TPP-binding domain variants more common; thiamine-responsive subset; exercise intolerance</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color:'#558b2f'}}>Mild/Juvenile (Partial Deficiency)</td>
                <td>10%</td>
                <td>None (AR, equal)</td>
                <td>Chronic partial OGDH E1 deficiency; intellectual disability; ataxia; 2-oxoglutarate mildly-moderately elevated; sometimes thiamine-responsive; long-term survival possible with treatment</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function TreatmentsTab() {
  const { data, err } = useFetch(`${API}/api/ogdh/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="✅ THIAMINE (B1) LEVEL A — FIRST-LINE and MANDATORY TRIAL in ALL OGDH patients. TPP (thiamine pyrophosphate) is the essential cofactor for OGDH E1 (same role as TPP for PDHA1 in PDH deficiency). TPP-binding domain variants show 40–55% partial-to-complete response. Dose 100–600 mg/day; measure plasma α-KG and urine 2-oxoglutarate after 2–4 weeks."
        variant="success"
      />
      <Alert
        text="💊 RIBOFLAVIN (B2) LEVEL B — FAD cofactor for DLD (shared E3). Used in combination with thiamine. Safe and inexpensive. May improve residual αKGDH complex activity if DLD (E3) regeneration is partially rate-limiting."
        variant="info"
      />
      <Alert
        text="⚠️ KETOGENIC DIET — LEVEL C / EXPERIMENTAL IN OGDH (CONTRAST WITH PDH DEFICIENCY WHERE KD IS LEVEL A). KD does NOT bypass the OGDH block: ketones → acetyl-CoA → TCA entry at step 1 (citrate synthase), but TCA then hits the OGDH block at step 4 (α-KG → succinyl-CoA). KD does not provide succinyl-CoA. Use KD only if other options fail."
        variant="warning"
      />
      <Alert
        text="⚠️ VPA — CAUTION (not absolute CI as in PDH deficiency). VPA carries mitochondrial hepatotoxicity risk in any mitochondrial disorder. Unlike PDH deficiency where KD is first-line (so VPA destruction of KD is critical), OGDH treatment does not depend on KD. Monitor LFTs if VPA is prescribed. Prefer LEV."
        variant="warning"
      />

      <Section title="Treatment Ladder" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>Level</th><th>Mechanism in OGDH</th><th>Response %</th></tr>
            </thead>
            <tbody>
              {(data.treatments || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{color: t.color}}>{t.drug}</td>
                  <td><span className="badge" style={{backgroundColor: t.level==='A'?'#1b5e20':t.level==='B'?'#1565c0':'#827717'}}>{t.level}</span></td>
                  <td style={{fontSize:11}}>{
                    t.drug.includes('Thiamine') ? 'TPP cofactor for OGDH E1 (same mechanism as PDHA1 for PDH E1); high-dose TPP restores E1 decarboxylase activity in TPP-binding domain variants by mass action; reduces α-KG accumulation; Level A mandatory trial all patients' :
                    t.drug.includes('Riboflavin') ? 'FAD cofactor for DLD (shared E3); regenerates lipoamide arm of DLST E2; combined with thiamine; safe co-administration; may improve residual αKGDH complex activity' :
                    t.drug.includes('Lipoic') ? 'Lipoamide cofactor for DLST E2; experimental; OGDH-specific rationale (lipoamide on DLST E2 is the succinyl-group carrier); limited clinical evidence; Level C experimental' :
                    t.drug.includes('Succinate') ? 'Anaplerotic bypass: oral succinate → fumarate → malate → OAA (TCA downstream of OGDH block); replenishes TCA intermediates; provides anaplerotic carbon downstream of step 4 block; experimental; some anecdotal benefit mild phenotypes' :
                    t.drug.includes('Leve') ? 'First-line AED; no αKGDH pathway interaction; safe in all metabolic epilepsies including OGDH deficiency' :
                    t.drug.includes('Ketogenic') ? 'Experimental/limited: ketones → acetyl-CoA → TCA entry (citrate synthase), but TCA STILL BLOCKED at step 4. KD does not bypass OGDH block as it bypasses PDH block. Use only if other options fail; may reduce carbohydrate-driven α-KG flux modestly.' :
                    '—'
                  }</td>
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

      <Section title="OGDH vs PDH Complex — Treatment Comparison" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>OGDH (αKGDH E1)</th><th>PDHA1/PDHB/DLAT/PDHX (PDH complex)</th><th>Key difference</th></tr>
            </thead>
            <tbody>
              <tr><td>Thiamine (B1)</td><td className="text-success fw-bold">Level A — ~45% response</td><td className="text-success fw-bold">Level A — PDHA1 ~55%; PDHB/DLAT/PDHX ~30–40%</td><td>Same TPP-E1 mechanism; OGDH E1 uses TPP identical to PDHA1 E1α; response rates comparable</td></tr>
              <tr><td>Riboflavin (B2)</td><td className="text-primary fw-bold">Level B</td><td className="text-muted">Not standard</td><td>DLD (E3) is shared; riboflavin FAD cofactor relevant for DLD in αKGDH; not typically given in PDH deficiency</td></tr>
              <tr><td>Ketogenic diet</td><td className="text-warning fw-bold">Level C — experimental</td><td className="text-success fw-bold">Level A — FIRST LINE</td><td>KD BYPASSES PDH block (ketones → acetyl-CoA → TCA). KD does NOT bypass OGDH block (TCA block at step 4 downstream of acetyl-CoA entry). Critical difference.</td></tr>
              <tr><td>Succinate supplementation</td><td className="text-warning fw-bold">Level C — experimental</td><td className="text-muted">Not applicable</td><td>OGDH-specific: succinate provides TCA anaplerosis downstream of αKGDH block. No role in PDH deficiency (TCA entry is the problem, not internal TCA block).</td></tr>
              <tr><td>VPA</td><td className="text-warning fw-bold">CAUTION — not absolute CI</td><td className="text-danger fw-bold">ABSOLUTE CI</td><td>PDH deficiency: VPA destroys KD (first-line treatment) → absolute CI. OGDH: KD is not first-line; VPA is caution (mitochondrial hepatotoxicity), not absolute CI.</td></tr>
              <tr><td>High-carbohydrate diet</td><td className="text-warning fw-bold">CAUTION (moderate)</td><td className="text-danger fw-bold">EXTREME HAZARD</td><td>PDH block: high CHO floods pyruvate → acute crisis. OGDH block: high protein is the main trigger (α-KG from AA catabolism); CHO less dangerous than in PDH deficiency.</td></tr>
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
  const { data, err } = useFetch(`${API}/api/ogdh/definitions`);
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
                <td style={k === 'Primary biomarker' ? {color: ACCENT3, fontWeight:'600'} : k === 'Key negative biomarkers' ? {color: ACCENT6, fontWeight:'600'} : {}}>{Array.isArray(v) ? v.join(' · ') : String(v)}</td>
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
                  k.includes('bcaa') || k.includes('glycine') || k.includes('dld_free') || k.includes('pdh_enzyme') ? {backgroundColor:'#e8f5e9'} :
                  k.includes('akgdh_complex') ? {backgroundColor:'#fce4ec'} :
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
              <tr><th>Condition</th><th>Key distinguishing feature from OGDH</th></tr>
            </thead>
            <tbody>
              {(data.differential_diagnosis || []).map((d, i) => (
                <tr key={i} style={
                  d.disease.includes('DLD') ? {backgroundColor:'#e8f5e9'} :
                  d.disease.includes('DLST') ? {backgroundColor:'#efebe9'} :
                  d.disease.includes('PDH') ? {backgroundColor:'#e0f2f1'} : {}
                }>
                  <td className="fw-bold" style={{fontSize:11, color: d.disease.includes('DLD') ? ACCENT6 : d.disease.includes('DLST') ? ACCENT : d.disease.includes('PDH') ? '#00695c' : 'inherit'}}>{d.disease}</td>
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
export default function OgdhPage() {
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
          🧬 OGDH Epilepsy — 2-Oxoglutarate Dehydrogenase Deficiency (αKGDH Complex E1 Subunit)
        </h4>
        <div className="text-muted small">
          OGDH · 7p13 · Autosomal Recessive · 1023 aa · E1 (oxoglutarate decarboxylase) · TPP cofactor ·
          αKGDH complex: OGDH (E1) + DLST (E2) + DLD (E3 shared) ·
          TCA step 4: α-KG → succinyl-CoA (BLOCKED) ·
          OGDH LOF → 2-oxoglutaric aciduria (urine 2-OG 200–2200 µmol/mmolCr — PATHOGNOMONIC) ·
          L:P mildly elevated (15–30; intermediate between PDH 10–20 and Complex I &gt;25) ·
          BCAA NORMAL · Glycine NORMAL · DLD free activity NORMAL · PDH activity NORMAL ·
          Thiamine (B1) FIRST-LINE Level A (TPP cofactor for E1; ~45% response) ·
          KD Level C only (does NOT bypass αKGDH block unlike PDH deficiency) ·
          OMIM *604290/#203740 · ~20–40 cases worldwide (2026)
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
