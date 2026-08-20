'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#d84315';   // deep red-orange — E1β structural / AR PDH / Leigh syndrome
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / VPA / neonatal death
const ACCENT3 = '#f57f17';   // amber — episodic / L:P ratio normal / carbohydrate hazard
const ACCENT4 = '#1565c0';   // deep blue — KD / thiamine / DCA / treatments
const ACCENT5 = '#4a148c';   // deep purple — definitions / PDH complex mechanism
const ACCENT6 = '#006064';   // dark teal — biochemistry / PDH components / regulation
const ACCENT7 = '#1b5e20';   // dark green — biomarkers / alanine / structural anomalies

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
  const { data, err } = useFetch(`${API}/api/pdhb/overview`);
  if (err)  return <div className="alert alert-danger">Error loading overview.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const k = data.kpis || {};
  return (
    <div>
      <Alert
        text="🔑 KEY BIOCHEMICAL FINGERPRINT: L:P ratio NORMAL (10–20) in PDHB deficiency — BOTH lactate AND pyruvate accumulate equally. Biochemically IDENTICAL to PDHA1. Complex I deficiency: L:P >25. Gene panel (PDHA1 + PDHB + DLAT + PDHX) MANDATORY to assign correct gene."
        variant="info"
      />
      <Alert
        text="⚠️ PDHB vs PDHA1: PDHB is AUTOSOMAL RECESSIVE (AR, 3p14.3) — no sex bias; males and females equally affected. PDHA1 is X-linked dominant (Xp22.12) — males severely affected. Biochemistry IDENTICAL: gene panel is the ONLY way to distinguish."
        variant="warning"
      />
      <Alert
        text="⛔ VPA ABSOLUTE CONTRAINDICATED — mitochondrial hepatotoxicity + carnitine depletion (destroys ketogenic diet therapy — the primary treatment for PDHB). NEVER use valproate in any PDH complex subunit deficiency."
        variant="danger"
      />
      <Alert
        text="🥑 KETOGENIC DIET IS FIRST-LINE THERAPY (Level A) — KD provides ketone bodies that enter TCA cycle as Acetyl-CoA, BYPASSING the blocked E1 component of PDH complex. Same mechanism and efficacy as in PDHA1."
        variant="success"
      />

      <Section title="Gene Summary" color={ACCENT6}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'38%'}}>Gene</td><td>{data.gene} ({data.protein})</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{data.locus} — Autosomal (chromosome 3)</td></tr>
            <tr><td className="fw-bold">Protein length</td><td>{data.aa_length} aa (incl. mitochondrial targeting sequence); structural E1β; pairs with E1α (PDHA1) to form E1α₂β₂ heterotetramer</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td>{data.cofactor}</td></tr>
            <tr><td className="fw-bold">Mechanism</td><td>{data.mechanism}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease} (PDH E1β Deficiency)</td></tr>
            <tr><td className="fw-bold">Inheritance</td><td style={{color: ACCENT3, fontWeight:'600'}}>{data.inheritance}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT}}>PDHB vs PDHA1</td><td style={{color: ACCENT, fontWeight:'600'}}>{data.key_distinguishing_from_pdha1}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT2}}>Critical biochemical key</td><td style={{color: ACCENT, fontWeight:'600'}}>{data.key_distinguishing_feature}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT7}}>Structural brain hallmark</td><td style={{color: ACCENT7}}>{data.structural_brain_hallmark}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="PDH Complex Components & Regulation" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Component</th><th>Gene(s)</th><th>Function</th></tr>
            </thead>
            <tbody>
              {(data.pdh_complex_components || []).map((row, i) => (
                <tr key={i} style={row.component.includes('PDHA1 + PDHB') ? {backgroundColor:'#fff3e0'} : {}}>
                  <td className="fw-bold" style={row.component.includes('PDHA1 + PDHB') ? {color: ACCENT} : {}}>{row.component}</td>
                  <td style={{fontSize:11}}>{
                    row.component.includes('E1 (PDHA1') ? 'PDHA1 + PDHB' :
                    row.component.includes('E2') ? 'DLAT' :
                    row.component.includes('E3BP') ? 'PDHX' :
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
          <KPI label="Male patients %" value={`${k.male_pct}%`} color={ACCENT6} />
          <KPI label="Avg plasma lactate (mmol/L)" value={k.avg_plasma_lactate_mmol} color={ACCENT2} />
          <KPI label="Avg plasma pyruvate (mmol/L)" value={k.avg_plasma_pyruvate_mmol} color={ACCENT3} />
          <KPI label="Avg L:P ratio (target 10–20)" value={k.avg_lp_ratio} color={ACCENT4} />
          <KPI label="Normal L:P ratio (10–20) %" value={`${k.normal_lp_ratio_10_20_pct}%`} color={ACCENT7} />
          <KPI label="Avg plasma alanine (µmol/L)" value={k.avg_plasma_alanine_umol} color={ACCENT6} />
          <KPI label="CC agenesis/dysgenesis %" value={`${k.cc_agenesis_pct}%`} color={ACCENT2} />
          <KPI label="Leigh lesions (MRI) %" value={`${k.leigh_lesions_pct}%`} color={ACCENT2} />
          <KPI label="On Ketogenic Diet %" value={`${k.on_kd_pct}%`} color={ACCENT7} />
          <KPI label="DRE %" value={`${k.dre_pct}%`} color={ACCENT2} />
          <KPI label="VPA avoided %" value={`${k.vpa_avoided_pct}%`} color={ACCENT7} />
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

      <Section title="Structural Brain Anomalies (Frequency in Cohort)" color={ACCENT7}>
        {(data.structural_anomalies || []).map((a, i) => (
          <div key={i} className="mb-2">
            <PctBar label={a.anomaly} pct={a.frequency_pct} color={ACCENT7} />
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
  const { data, err } = useFetch(`${API}/api/pdhb/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Biomarkers — PDH Biochemical Fingerprint (identical to PDHA1)" color={ACCENT7}>
        <Alert
          text="SIMULTANEOUS plasma lactate + plasma pyruvate (L:P ratio) + plasma alanine MANDATORY. L:P ratio 10–20 (NORMAL) is the KEY fingerprint. PDHB and PDHA1 are biochemically IDENTICAL — gene panel is the only discriminator. Normal BCAA and 2-HG distinguish PDHB from DLD (E3 deficiency)."
          variant="info"
        />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Biomarker</th><th>Normal</th><th>PDHB range</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {(data.biomarkers || []).map((b, i) => (
                <tr key={i} style={b.name.includes('L:P') ? {backgroundColor:'#fff8e1', fontWeight:'600'} : b.name.includes('BCAA') || b.name.includes('2-hydroxy') ? {backgroundColor:'#e8f5e9'} : {}}>
                  <td className="fw-bold">{b.name}</td>
                  <td className="text-success small">{b.normal}</td>
                  <td className={b.pdhb_range.includes('NORMAL') ? 'text-success small fw-bold' : 'text-danger small'}>{b.pdhb_range}</td>
                  <td className="small">{b.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Key Variants" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Variant</th><th>Effect / Structural impact</th><th>Phenotype</th></tr>
            </thead>
            <tbody>
              {(data.key_variants || []).map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold font-monospace">{v.variant}</td>
                  <td>{v.effect}</td>
                  <td><span className="badge" style={{backgroundColor: ACCENT3, color:'#000'}}>{v.phenotype}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Patient Sample (n=10)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Lactate</th><th>Pyruvate</th><th>L:P</th><th>Alanine</th>
                <th>CC</th><th>Leigh</th><th>KD</th><th>Thiamine</th><th>VPA avoided</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map((p, i) => (
                <tr key={i}>
                  <td>{p.id}</td>
                  <td><span className="badge" style={{backgroundColor: p.sex==='M'?'#1565c0':'#c62828'}}>{p.sex}</span></td>
                  <td style={{fontSize:10}}>{p.phenotype.split('(')[0].trim()}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{color:ACCENT2}}>{p.plasma_lactate_mmol}</td>
                  <td style={{color:ACCENT3}}>{p.plasma_pyruvate_mmol}</td>
                  <td style={{color: p.lp_ratio<=20 ? ACCENT7 : ACCENT2, fontWeight:'600'}}>{p.lp_ratio}</td>
                  <td>{p.plasma_alanine_umol}</td>
                  <td>{p.cc_agenesis ? '✅' : '—'}</td>
                  <td>{p.leigh_lesions ? '✅' : '—'}</td>
                  <td>{p.on_kd ? '✅' : '—'}</td>
                  <td>{p.on_thiamine ? '✅' : '—'}</td>
                  <td>{p.vpa_avoided ? '✅' : '⛔'}</td>
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
  const { data, err } = useFetch(`${API}/api/pdhb/breakdown`);
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
          text="🏃 EXERCISE IS A KEY TRIGGER IN PDHB — exertion raises pyruvate flux; PDHB LOF means E1 cannot handle pyruvate surge → acute lactic acidosis. Exercise-induced lactic acidosis is more prominent in childhood episodic phenotype. Glucose load / febrile illness trigger same as PDHA1."
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
                <td>BG + brainstem lesions (putamen, periaqueductal grey); psychomotor regression; episodic decompensation; most common PDHB phenotype</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT}}>Severe Neonatal</td>
                <td>20%</td>
                <td>None (AR, equal)</td>
                <td>Neonatal lactic acidosis; early death without KD; CC agenesis possible; less common than in PDHA1 (no hemizygous male bias)</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color: ACCENT3}}>Childhood Episodic</td>
                <td>30%</td>
                <td>None (AR, equal)</td>
                <td>Exercise/illness-triggered lactic acidosis; relatively normal development between episodes; partial E1β function (partial-loss variants p.Ile199Thr, p.Arg161His)</td>
              </tr>
              <tr>
                <td className="fw-bold" style={{color:'#558b2f'}}>Mild Subacute/Juvenile</td>
                <td>10%</td>
                <td>None (AR, equal)</td>
                <td>Chronic partial enzyme deficiency; ataxia; intellectual disability; elevated alanine as chronic marker; some normal lifespan</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function TreatmentsTab() {
  const { data, err } = useFetch(`${API}/api/pdhb/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="✅ KETOGENIC DIET (Level A) is FIRST-LINE therapy for PDHB — same mechanism as PDHA1: ketones bypass the blocked E1 component. MANDATORY to start KD early in all severe and Leigh phenotypes. Identical efficacy to PDHA1."
        variant="success"
      />
      <Alert
        text="⚠️ THIAMINE (B1) RESPONSIVENESS LESS COMMON IN PDHB — E1β does not directly bind TPP (that is E1α). However, high-dose thiamine trial (100–600 mg/day) is still recommended as TPP may stabilise residual E1 complex activity. Expect ~20–35% partial response (vs 40–55% in PDHA1)."
        variant="warning"
      />
      <Alert
        text="⛔ VPA ABSOLUTE CI in PDHB — same mitochondrial hepatotoxicity risk as PDHA1 + carnitine depletion impairs β-oxidation (ketone production for KD therapy). NEVER prescribe VPA in PDH complex deficiency."
        variant="danger"
      />

      <Section title="Treatment Ladder" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>Level</th><th>Mechanism in PDHB</th><th>Response %</th></tr>
            </thead>
            <tbody>
              {(data.treatments || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{color: t.color}}>{t.drug}</td>
                  <td><span className="badge" style={{backgroundColor: t.level==='A'?'#1b5e20':t.level==='B'?'#1565c0':'#827717'}}>{t.level}</span></td>
                  <td style={{fontSize:11}}>{
                    t.drug.includes('Ketogenic') ? 'Ketones → Acetyl-CoA via thiolase; bypasses E1-deficient PDH block; provides TCA substrate downstream of E1 step; same as PDHA1' :
                    t.drug.includes('Thiamine') ? 'TPP cofactor for E1α; stabilises E1α which may partially reconstitute E1 complex with residual E1β; less responsive than PDHA1 (~35% vs 55%) — E1β does not bind TPP' :
                    t.drug.includes('Carnitine') ? 'Secondary depletion support; essential for β-oxidation → fatty acids → ketones for KD; must supplement during KD' :
                    t.drug.includes('Dichloro') ? 'Inhibits PDK1/3 → prevents E1α Ser293 phosphorylation → maximises residual E1 complex activity (only effective if some E1β remains); monitor for peripheral neuropathy' :
                    t.drug.includes('Ribo') ? 'Minimal direct PDHB effect; trial only if DLD/E3 component involvement suspected' :
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

      <Section title="PDHB vs PDHA1 — Treatment Comparison" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Treatment</th><th>PDHB</th><th>PDHA1</th><th>Key difference</th></tr>
            </thead>
            <tbody>
              <tr><td>Ketogenic Diet</td><td className="text-success fw-bold">Level A — First line</td><td className="text-success fw-bold">Level A — First line</td><td>Identical indication and mechanism; same efficacy</td></tr>
              <tr><td>Thiamine (B1)</td><td className="text-warning fw-bold">Level A — ~35% response</td><td className="text-success fw-bold">Level A — ~55% response</td><td>PDHB: less responsive (E1β does not bind TPP); still trial all patients</td></tr>
              <tr><td>L-Carnitine</td><td>Level B</td><td>Level B</td><td>Same indication (secondary depletion, KD support)</td></tr>
              <tr><td>DCA</td><td>Level B (partial-loss only)</td><td>Level B</td><td>PDHB: only if some E1β residual activity; null variants unlikely to respond</td></tr>
              <tr><td>VPA</td><td className="text-danger fw-bold">ABSOLUTE CI</td><td className="text-danger fw-bold">ABSOLUTE CI</td><td>Identical contraindication — same mitochondrial hepatotoxicity + carnitine depletion</td></tr>
              <tr><td>High-CHO diet</td><td className="text-danger fw-bold">EXTREME HAZARD</td><td className="text-danger fw-bold">EXTREME HAZARD</td><td>Identical — floods pyruvate, worsens lactic acidosis</td></tr>
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
  const { data, err } = useFetch(`${API}/api/pdhb/definitions`);
  if (err)  return <div className="alert alert-danger">Error loading definitions.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const gc = data.gene_card || {};
  return (
    <div>
      <Section title="Gene Card" color={ACCENT5}>
        <table className="table table-sm table-bordered small">
          <tbody>
            {Object.entries(gc).map(([k, v]) => (
              <tr key={k}>
                <td className="fw-bold" style={{width:'32%'}}>{k.replace(/_/g,' ')}</td>
                <td>{Array.isArray(v) ? v.join(' · ') : String(v)}</td>
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

      <Section title="Diagnostic Thresholds" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Parameter</th><th>Normal / Threshold</th><th>PDHB range</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i} style={t.parameter.includes('L:P') ? {backgroundColor:'#fff8e1', fontWeight:'600'} : t.parameter.includes('BCAA') ? {backgroundColor:'#e8f5e9'} : {}}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td className="text-success small">{t.threshold}</td>
                  <td className={t.pdhb_range.includes('NORMAL') ? 'text-success small fw-bold' : 'text-danger small'}>{t.pdhb_range}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Condition</th><th>Key distinguishing feature from PDHB</th></tr>
            </thead>
            <tbody>
              {(data.differential_diagnosis || []).map((d, i) => (
                <tr key={i} style={d.condition.includes('PDHA1') ? {backgroundColor:'#fff3e0'} : {}}>
                  <td className="fw-bold" style={{fontSize:11, color: d.condition.includes('PDHA1') ? ACCENT : 'inherit'}}>{d.condition}</td>
                  <td className="small">{d.distinguishing}</td>
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
export default function PdhbPage() {
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
          🧬 PDHB Epilepsy — Pyruvate Dehydrogenase Complex Deficiency (E1β Subunit)
        </h4>
        <div className="text-muted small">
          PDHB · 3p14.3 · Autosomal Recessive · 359 aa · PDC-E1β (structural; assembles E1α₂β₂ tetramer) ·
          No direct TPP binding · E1β LOF → E1α degraded → PDH complex inactive → Lactic acidosis ·
          Leigh syndrome (most prominent) + CC agenesis (~40%) · L:P ratio NORMAL (key fingerprint) ·
          Biochemically IDENTICAL to PDHA1 — gene panel mandatory · Ketogenic Diet FIRST-LINE (Level A) ·
          VPA Absolute CI · OMIM *179060/#614111 · ~50–80 cases worldwide (2026)
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
