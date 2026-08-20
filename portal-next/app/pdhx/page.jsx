'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00695c';   // dark teal — E3BP structural bridging role / PDHX linker
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / VPA / neonatal death
const ACCENT3 = '#e65100';   // deep orange — L:P normal fingerprint / episodic / hazard
const ACCENT4 = '#1565c0';   // deep blue — KD / thiamine / DCA / treatments
const ACCENT5 = '#4a148c';   // deep purple — definitions / E3BP mechanism
const ACCENT6 = '#1b5e20';   // dark green — biomarkers / lipoyl domain / normal key negatives
const ACCENT7 = '#37474f';   // dark slate — DLD normal key negative / large deletion

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
  const { data, err } = useFetch(`${API}/api/pdhx/overview`);
  if (err)  return <div className="alert alert-danger">Error loading overview.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const k = data.kpis || {};
  return (
    <div>
      <Alert
        text="🔑 KEY BIOCHEMICAL FINGERPRINT: L:P ratio NORMAL (10–20) in PDHX deficiency — BOTH lactate AND pyruvate accumulate equally. Biochemically IDENTICAL to PDHA1, PDHB, and DLAT. Complex I deficiency: L:P >25. Gene panel (PDHA1 + PDHB + DLAT + PDHX) MANDATORY to assign correct gene."
        variant="info"
      />
      <Alert
        text="🔗 PDHX ENCODES E3BP (E3-BINDING PROTEIN / COMPONENT X) — a structural LINKER with NO catalytic activity. E3BP anchors E3 (DLD) to the E2 (DLAT) cubic core. PDHX LOF: E3 dissociates from the PDH complex → lipoamide arms cannot be regenerated → complex stalls. E2 (DLAT) core remains structurally intact (unlike DLAT LOF)."
        variant="primary"
      />
      <Alert
        text="🔑 CRITICAL PDHX-SPECIFIC FINGERPRINT: DLD/E3 FREE enzyme activity is NORMAL or slightly reduced in PDHX deficiency — DLD protein is intact, just unanchored. In DLD deficiency: DLD activity is severely reduced (<10% normal). This biochemical difference is key to distinguishing PDHX from DLD without gene panel."
        variant="success"
      />
      <Alert
        text="⛔ VPA ABSOLUTE CONTRAINDICATED — mitochondrial hepatotoxicity + carnitine depletion (destroys ketogenic diet therapy — the primary treatment for ALL PDH complex deficiencies including PDHX). NEVER use valproate in PDHX deficiency."
        variant="danger"
      />
      <Alert
        text="🧬 LARGE GENOMIC DELETIONS IN ~20% OF PDHX ALLELES — 11p13 structural instability makes copy number variants more frequent in PDHX than in PDHA1/PDHB/DLAT. Standard sequencing alone may miss ~20% of alleles. CNV analysis / MLPA RECOMMENDED alongside sequencing."
        variant="warning"
      />

      <Section title="Gene Summary" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'38%'}}>Gene</td><td>{data.gene} ({data.protein})</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{data.locus} — Autosomal (chromosome 11p)</td></tr>
            <tr><td className="fw-bold">Protein length</td><td>{data.aa_length} aa (incl. mitochondrial targeting sequence); E3BP structural linker; NO catalytic activity; bridges E2 core and E3</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td>{data.cofactor}</td></tr>
            <tr><td className="fw-bold">Mechanism</td><td>{data.mechanism}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease} (PDH Complex Deficiency, E3BP type)</td></tr>
            <tr><td className="fw-bold">Inheritance</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.inheritance}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT}}>PDHX vs PDHA1</td><td style={{color: ACCENT, fontWeight:'600'}}>{data.key_distinguishing_from_pdha1}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT2}}>Critical biochemical key</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.key_distinguishing_feature}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT6}}>Structural brain hallmark</td><td style={{color: ACCENT6}}>{data.structural_brain_hallmark}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="PDH Complex Components & Regulation" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Component</th><th>Gene(s)</th><th>Function</th></tr>
            </thead>
            <tbody>
              {(data.pdh_complex_components || []).map((row, i) => (
                <tr key={i} style={row.component.includes('PDHX') ? {backgroundColor:'#e0f2f1'} : {}}>
                  <td className="fw-bold" style={row.component.includes('PDHX') ? {color: ACCENT} : {}}>{row.component}</td>
                  <td style={{fontSize:11}}>{
                    row.component.includes('E1 (PDHA1') ? 'PDHA1 + PDHB' :
                    row.component.includes('DLAT') ? 'DLAT' :
                    row.component.includes('PDHX') ? 'PDHX ← THIS GENE' :
                    row.component.includes('E3') ? 'DLD' :
                    row.component.includes('PDK') ? 'PDK1/2/3/4' : 'PDP1/2'
                  }</td>
                  <td>{row.function}</td>
                </tr>
              ))}
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
          <KPI label="Avg L:P ratio (target 10–20)" value={k.avg_lp_ratio} color={ACCENT4} />
          <KPI label="Normal L:P ratio (10–20) %" value={`${k.normal_lp_ratio_10_20_pct}%`} color={ACCENT6} />
          <KPI label="Avg plasma alanine (µmol/L)" value={k.avg_plasma_alanine_umol} color={ACCENT5} />
          <KPI label="CC agenesis/dysgenesis %" value={`${k.cc_agenesis_pct}%`} color={ACCENT2} />
          <KPI label="Leigh lesions (MRI) %" value={`${k.leigh_lesions_pct}%`} color={ACCENT2} />
          <KPI label="On Ketogenic Diet %" value={`${k.on_kd_pct}%`} color={ACCENT6} />
          <KPI label="DRE %" value={`${k.dre_pct}%`} color={ACCENT2} />
          <KPI label="VPA avoided %" value={`${k.vpa_avoided_pct}%`} color={ACCENT6} />
          <KPI label="Large genomic deletion %" value={`${k.large_deletion_pct}%`} color={ACCENT7} />
        </div>
      </Section>

      <Section title="High-Risk Drugs" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <Alert
            key={i}
            text={`${d.risk === 'ABSOLUTE CI' ? '⛔' : d.risk === 'EXTREME HAZARD' ? '🚨' : d.risk === 'HIGH RISK' ? '🔴' : '⚠️'} ${d.drug} — ${d.risk}: ${d.mechanism}`}
            variant={d.risk === 'ABSOLUTE CI' ? 'danger' : d.risk === 'EXTREME HAZARD' ? 'danger' : d.risk === 'HIGH RISK' ? 'danger' : 'warning'}
          />
        ))}
      </Section>

      <Section title="Structural Brain Anomalies (Frequency in Cohort)" color={ACCENT6}>
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
  const { data, err } = useFetch(`${API}/api/pdhx/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Biomarkers — PDH Biochemical Fingerprint (identical to PDHA1/PDHB/DLAT)" color={ACCENT6}>
        <Alert
          text="SIMULTANEOUS plasma lactate + plasma pyruvate (L:P ratio) + plasma alanine MANDATORY. L:P ratio 10–20 (NORMAL) is the KEY fingerprint. PDHX, PDHA1, PDHB, DLAT are biochemically IDENTICAL — gene panel is the ONLY discriminator. Normal BCAA + normal 2-HG + NORMAL free DLD/E3 activity distinguish PDHX from DLD (E3 deficiency)."
          variant="info"
        />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Biomarker</th><th>Normal</th><th>PDHX range</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {(data.biomarkers || []).map((b, i) => (
                <tr key={i} style={
                  b.name.includes('L:P') ? {backgroundColor:'#fff8e1', fontWeight:'600'} :
                  b.name.includes('BCAA') || b.name.includes('2-Hydroxy') || b.name.includes('DLD/E3 free') ? {backgroundColor:'#e8f5e9'} :
                  b.name.includes('PDHX/E3BP') ? {backgroundColor:'#fce4ec'} : {}
                }>
                  <td className="fw-bold">{b.name}</td>
                  <td className="text-success small">{b.normal}</td>
                  <td className={
                    b.pdhx_range && (b.pdhx_range.includes('NORMAL') || b.pdhx_range.includes('intact') || b.pdhx_range.includes('slightly reduced'))
                      ? 'text-success small fw-bold'
                      : 'text-danger small'
                  }>{b.pdhx_range}</td>
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
                <tr key={i} style={v.variant.includes('deletion') ? {backgroundColor:'#fff3e0'} : {}}>
                  <td className="fw-bold font-monospace">{v.variant}</td>
                  <td>{v.effect}</td>
                  <td><span className="badge" style={{backgroundColor: ACCENT3, color:'#000'}}>{v.phenotype}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">⚠️ Large genomic deletions (highlighted) account for ~20% of PDHX alleles — CNV/MLPA analysis recommended alongside sequencing.</div>
      </Section>

      <Section title="Patient Sample (n=10)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Lactate</th><th>Pyruvate</th><th>L:P</th><th>Alanine</th>
                <th>CC</th><th>Leigh</th><th>KD</th><th>Thiamine</th><th>VPA avoided</th><th>Del</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map((p, i) => (
                <tr key={i} style={p.large_deletion ? {backgroundColor:'#fff8e1'} : {}}>
                  <td>{p.id}</td>
                  <td><span className="badge" style={{backgroundColor: p.sex==='M'?'#1565c0':'#c62828'}}>{p.sex}</span></td>
                  <td style={{fontSize:10}}>{p.phenotype.split('(')[0].trim()}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{color:ACCENT2}}>{p.plasma_lactate_mmol}</td>
                  <td style={{color:ACCENT3}}>{p.plasma_pyruvate_mmol}</td>
                  <td style={{color: p.lp_ratio<=20 ? ACCENT6 : ACCENT2, fontWeight:'600'}}>{p.lp_ratio}</td>
                  <td>{p.plasma_alanine_umol}</td>
                  <td>{p.cc_agenesis ? '✅' : '—'}</td>
                  <td>{p.leigh_lesions ? '✅' : '—'}</td>
                  <td>{p.on_kd ? '✅' : '—'}</td>
                  <td>{p.on_thiamine ? '✅' : '—'}</td>
                  <td>{p.vpa_avoided ? '✅' : '⛔'}</td>
                  <td style={{color: p.large_deletion ? ACCENT3 : 'inherit'}}>{p.large_deletion ? 'DEL' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">Del = large genomic deletion allele (CNV/MLPA confirmed). Highlighted rows carry deletion allele.</div>
      </Section>
    </div>
  );
}

function SeizuresTab() {
  const { data, err } = useFetch(`${API}/api/pdhx/breakdown`);
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
          text="🌡️ FEBRILE ILLNESS IS THE MOST COMMON TRIGGER IN PDHX deficiency — fever increases metabolic demands and pyruvate flux; PDHX LOF means E3 cannot regenerate lipoamide arms → PDH complex stalls → acute lactic acidosis. Glucose load and glucose-only IV feeds are equally dangerous. KD must be maintained even during illness."
          variant="warning"
        />
        {(data.trigger_types || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT3} />
        ))}
      </Section>

      <Section title="Phenotype Distribution" color={ACCENT}>
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
                <td>Symmetric BG + brainstem lesions (putamen, periaqueductal grey); psychomotor regression; episodic decompensation; most common PDHX phenotype</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT3}}>Severe Neonatal</td>
                <td>25%</td>
                <td>None (AR, equal)</td>
                <td>Neonatal lactic acidosis; CC agenesis/dysgenesis; early death without KD; null variants (large deletions, frameshift); DLD free activity normal distinguishes from DLD deficiency</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT4}}>Childhood Episodic</td>
                <td>25%</td>
                <td>None (AR, equal)</td>
                <td>Illness/exercise-triggered lactic acidosis; relatively normal development between episodes; partial E3BP function (p.Arg445Cys, p.Arg384Cys); more common in PDHX than DLAT</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color:'#558b2f'}}>Mild Subacute/Juvenile</td>
                <td>10%</td>
                <td>None (AR, equal)</td>
                <td>Chronic partial enzyme deficiency; ataxia; intellectual disability; elevated alanine as chronic marker</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function TreatmentsTab() {
  const { data, err } = useFetch(`${API}/api/pdhx/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="✅ KETOGENIC DIET (Level A) is FIRST-LINE therapy for PDHX — same mechanism as PDHA1/PDHB/DLAT: ketones bypass the blocked E3BP-dependent E3 anchoring step. MANDATORY to start KD early in all severe and Leigh phenotypes. Ketones → Acetyl-CoA via thiolase → TCA cycle bypasses the PDH block completely."
        variant="success"
      />
      <Alert
        text="⚠️ THIAMINE (B1) TRIAL MANDATORY but LESS RESPONSIVE in PDHX — PDHX does not directly bind TPP (that is E1α/PDHA1). Thiamine may help only if residual E3BP function allows partial E3 engagement. Expect ~30–35% partial response. High-dose trial (100–600 mg/day) is still Level A — cannot predict response without trial."
        variant="warning"
      />
      <Alert
        text="⛔ VPA ABSOLUTE CI in PDHX — same mitochondrial hepatotoxicity + carnitine depletion risk as PDHA1/PDHB/DLAT. NEVER prescribe VPA in PDH complex deficiency."
        variant="danger"
      />

      <Section title="Treatment Ladder" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>Level</th><th>Mechanism in PDHX</th><th>Response %</th></tr>
            </thead>
            <tbody>
              {(data.treatments || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{color: t.color}}>{t.drug}</td>
                  <td><span className="badge" style={{backgroundColor: t.level==='A'?'#1b5e20':t.level==='B'?'#1565c0':'#827717'}}>{t.level}</span></td>
                  <td style={{fontSize:11}}>{
                    t.drug.includes('Ketogenic') ? 'Ketones → Acetyl-CoA via SCOT + thiolase; bypasses E3BP-blocked PDH step entirely; provides TCA substrate downstream of blocked PDH complex; same as PDHA1/PDHB/DLAT' :
                    t.drug.includes('Thiamine') ? 'TPP cofactor for E1α; stabilises E1 complex; may help if partial E3BP-E3 engagement remains; less responsive in PDHX (~32%) — E3BP anchoring (not TPP) is the primary block' :
                    t.drug.includes('Carnitine') ? 'Secondary depletion support; essential for β-oxidation → fatty acids → ketones for KD; must supplement during KD' :
                    t.drug.includes('Dichlo') ? 'Inhibits PDK1/3 → prevents E1α Ser293 phosphorylation → maximises residual E1 activity; only effective if partial E3BP function remains; monitor for peripheral neuropathy' :
                    t.drug.includes('Leve') ? 'First-line AED; no PDH pathway interaction; safe in all metabolic epilepsies' :
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

      <Section title="PDHX vs Other PDH Complex Genes — Treatment Comparison" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>PDHX</th><th>PDHA1</th><th>PDHB / DLAT</th><th>Key difference</th></tr>
            </thead>
            <tbody>
              <tr><td>Ketogenic Diet</td><td className="text-success fw-bold">Level A — First line</td><td className="text-success fw-bold">Level A — First line</td><td className="text-success fw-bold">Level A — First line</td><td>Identical indication and mechanism across all PDH complex subunit deficiencies</td></tr>
              <tr><td>Thiamine (B1)</td><td className="text-warning fw-bold">Level A — ~32% response</td><td className="text-success fw-bold">Level A — ~55% response</td><td className="text-warning fw-bold">Level A — ~35% response</td><td>PDHX: less responsive (E3BP anchoring is the block, not TPP-binding E1α); still trial all patients</td></tr>
              <tr><td>L-Carnitine</td><td>Level B</td><td>Level B</td><td>Level B</td><td>Same indication (secondary depletion, KD support)</td></tr>
              <tr><td>DCA</td><td>Level B (partial-loss only)</td><td>Level B</td><td>Level B (partial-loss only)</td><td>PDHX: only if some E3BP-mediated E3 anchoring remains; null variants unlikely to respond</td></tr>
              <tr><td>Lipoic acid</td><td className="text-muted fw-bold">Not applicable</td><td className="text-muted fw-bold">Not applicable</td><td className="text-secondary">DLAT: Level C (experimental)</td><td>Lipoic acid is DLAT-specific (Lys173/Lys259); PDHX E3BP Lys173 is not accessible to exogenous supplementation</td></tr>
              <tr><td>VPA</td><td className="text-danger fw-bold">ABSOLUTE CI</td><td className="text-danger fw-bold">ABSOLUTE CI</td><td className="text-danger fw-bold">ABSOLUTE CI</td><td>Identical contraindication — same mitochondrial hepatotoxicity + carnitine depletion</td></tr>
              <tr><td>High-CHO diet</td><td className="text-danger fw-bold">EXTREME HAZARD</td><td className="text-danger fw-bold">EXTREME HAZARD</td><td className="text-danger fw-bold">EXTREME HAZARD</td><td>Identical — floods pyruvate, worsens lactic acidosis</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="High-Risk Drugs" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <Alert
            key={i}
            text={`${d.risk === 'ABSOLUTE CI' ? '⛔' : d.risk === 'EXTREME HAZARD' ? '🚨' : d.risk === 'HIGH RISK' ? '🔴' : '⚠️'} ${d.drug} — ${d.risk}: ${d.mechanism}`}
            variant={d.risk === 'ABSOLUTE CI' ? 'danger' : d.risk === 'EXTREME HAZARD' ? 'danger' : d.risk === 'HIGH RISK' ? 'danger' : 'warning'}
          />
        ))}
      </Section>
    </div>
  );
}

function DefinitionsTab() {
  const { data, err } = useFetch(`${API}/api/pdhx/definitions`);
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
                <td style={k === 'Unique feature' ? {color: ACCENT7, fontWeight:'600'} : {}}>{Array.isArray(v) ? v.join(' · ') : String(v)}</td>
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
                  k.includes('lp_ratio') ? {backgroundColor:'#fff8e1', fontWeight:'600'} :
                  k.includes('bcaa') || k.includes('2hg') || k.includes('dld_free') ? {backgroundColor:'#e8f5e9'} :
                  k.includes('pdhx_e3bp') ? {backgroundColor:'#fce4ec'} :
                  k.includes('cnv') ? {backgroundColor:'#fff3e0'} : {}
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
              <tr><th>Condition</th><th>Key distinguishing feature from PDHX</th></tr>
            </thead>
            <tbody>
              {(data.differential_diagnosis || []).map((d, i) => (
                <tr key={i} style={d.disease.includes('DLD') ? {backgroundColor:'#e8f5e9'} : d.disease.includes('PDHA1') || d.disease.includes('PDHB') || d.disease.includes('DLAT') ? {backgroundColor:'#e0f2f1'} : {}}>
                  <td className="fw-bold" style={{fontSize:11, color: d.disease.includes('DLD') ? ACCENT6 : d.disease.includes('PDHA1') || d.disease.includes('PDHB') || d.disease.includes('DLAT') ? ACCENT : 'inherit'}}>{d.disease}</td>
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
export default function PdhxPage() {
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
          🧬 PDHX Epilepsy — Pyruvate Dehydrogenase Complex Deficiency (E3BP / Component X / E3-Binding Protein)
        </h4>
        <div className="text-muted small">
          PDHX · 11p13 · Autosomal Recessive · 501 aa · E3BP (structural LINKER, NO catalytic activity) ·
          E3BP anchors E3 (DLD) to E2 (DLAT) cubic core via C-terminal docking domain ·
          E3BP lipoyl domain Lys173 — E3&apos;s substrate within PDH complex ·
          PDHX LOF → E3 dissociates → lipoamide arms not regenerated → PDH complex stalls ·
          Leigh syndrome (40%) + CC agenesis (~40%) · L:P ratio NORMAL (key fingerprint) ·
          CRITICAL: DLD free enzyme activity NORMAL (unlike DLD deficiency — key distinguisher) ·
          Large genomic deletions in ~20% of PDHX alleles (CNV/MLPA needed) ·
          Biochemically IDENTICAL to PDHA1/PDHB/DLAT — gene panel mandatory ·
          Ketogenic Diet FIRST-LINE (Level A) · VPA Absolute CI ·
          OMIM *608769/#245349 · ~30–50 cases worldwide (2026)
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
