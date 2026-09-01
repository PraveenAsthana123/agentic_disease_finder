'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Hearing & MRI', 'Treatments', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — SERAC1/MEGDEL (MAM/mito, Leigh-like encephalopathy)
const LIGHT = '#e8eaf6';

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

function Bar({ label, value, max, color = COLOR }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ variant, text }) {
  const bg = variant === 'danger' ? '#ffebee' : variant === 'warning' ? '#fff8e1' : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger' ? '#c62828' : variant === 'warning' ? '#f57f17' : variant === 'success' ? '#2e7d32' : COLOR;
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

function OverviewTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading overview...</div>;
  const kpis = data.kpis || {};
  const highlights = data.clinical_highlights || [];
  const cis = data.contraindications || [];
  const thresholds = data.thresholds || [];
  const ddx = data.ddx_table || [];

  return (
    <div>
      {/* Identity */}
      <SectionCard title="🧬 Disease Identity">
        <div className="row g-2 small">
          <div className="col-md-6"><strong>Disease:</strong> {data.disease}</div>
          <div className="col-md-6"><strong>Gene:</strong> {data.gene?.split(';')[0]}</div>
          <div className="col-md-4"><strong>Chromosome:</strong> {data.chromosome}</div>
          <div className="col-md-4"><strong>OMIM Gene:</strong> {data.omim_gene} &nbsp; <strong>Disease:</strong> {data.omim_disease}</div>
          <div className="col-md-4"><strong>Inheritance:</strong> {data.inheritance?.split(';')[0]}</div>
          <div className="col-md-6"><strong>Prevalence:</strong> {data.prevalence}</div>
          <div className="col-md-6"><strong>First described:</strong> {data.first_described}</div>
          <div className="col-md-6"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-md-6"><strong>Category:</strong> {data.category}</div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Cohort KPIs (n=40, seed-539)">
        <div className="row g-2">
          <KPI label="SNHL" value="100%" color="#b71c1c" />
          <KPI label="Encephalopathy/ID" value="100%" color={COLOR} />
          <KPI label="3-MGA-uria" value="100%" color={COLOR} />
          <KPI label="Leigh-like MRI" value={`${kpis.leigh_like_mri_pct}%`} color="#4a148c" />
          <KPI label="Epilepsy" value={`${kpis.epilepsy_pct}%`} color="#e65100" />
          <KPI label="Neonatal Liver" value={`${kpis.neonatal_liver_pct}%`} color="#f57f17" />
          <KPI label="Dystonia" value={`${kpis.dystonia_pct}%`} color="#1565c0" />
          <KPI label="Optic Atrophy" value="0%" color="#9e9e9e" />
          <KPI label="DCM" value="0%" color="#9e9e9e" />
          <KPI label="GP Iron" value="0%" color="#9e9e9e" />
          <KPI label="Cochlear Implant" value={`${kpis.cochlear_implant_placed_pct}%`} color="#2e7d32" />
          <KPI label="Mean 3-MGA" value={`${kpis.mean_mga_mmol} mmol/Cr`} color={COLOR} />
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="⚡ Clinical Highlights">
        {highlights.map((h, i) => <Alert key={i} variant={
          h.includes('CARDINAL') || h.includes('100%') && h.includes('SNHL') ? 'danger' :
          h.includes('NO optic') || h.includes('NO DCM') || h.includes('NO GP') || h.includes('NO neutropenia') ? 'warning' :
          h.includes('LEV') || h.includes('Cochlear') || h.includes('highly effective') ? 'success' : 'info'
        } text={h} />)}
      </SectionCard>

      {/* Contraindications */}
      <SectionCard title="🚫 Contraindications & Prescribing Cautions">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: 'white' }}>
              <tr><th>Drug</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {cis.map((ci, i) => (
                <tr key={i}>
                  <td className="fw-bold">{ci.drug}</td>
                  <td>{ci.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Thresholds */}
      <SectionCard title="📏 Clinical Thresholds & Action Points">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: 'white' }}>
              <tr><th>Parameter</th><th>Threshold</th><th>Action</th></tr>
            </thead>
            <tbody>
              {thresholds.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td>{t.threshold}</td>
                  <td>{t.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* DDx Table */}
      {ddx.length > 0 && (
        <SectionCard title="🔬 DDx Table: SERAC1 vs DNAJC19 vs OPA3 vs Barth (TAZ) vs MECR">
          <div className="table-responsive">
            <table className="table table-sm table-bordered small">
              <thead style={{ backgroundColor: COLOR, color: 'white' }}>
                <tr>
                  <th>Feature</th>
                  <th>SERAC1 MEGDEL</th>
                  <th>DNAJC19 DCMA</th>
                  <th>OPA3 Costeff</th>
                  <th>Barth (TAZ)</th>
                  <th>MECR MEPAN</th>
                </tr>
              </thead>
              <tbody>
                {ddx.map((row, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{row.feature}</td>
                    <td style={{ background: i % 2 === 0 ? LIGHT : 'white' }}>{row.serac1_megdel}</td>
                    <td>{row.dnajc19_dcma}</td>
                    <td>{row.opa3_costeff}</td>
                    <td>{row.barth_taz}</td>
                    <td>{row.mecr_mepan}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Gene Biology */}
      {data.gene_biology && (
        <SectionCard title="🔬 Protein Biology — SERAC1 (MAM Phospholipid Remodeling)">
          <div className="row g-2 small mb-3">
            <div className="col-md-4"><strong>Length:</strong> {data.gene_biology.protein_length} aa</div>
            <div className="col-md-4"><strong>Complex:</strong> {data.gene_biology.complex}</div>
            <div className="col-md-4"><strong>Partners:</strong> {data.gene_biology.partners}</div>
            <div className="col-12"><strong>Pathway:</strong> {data.gene_biology.pathway}</div>
            <div className="col-12"><strong>LOF consequence:</strong> {data.gene_biology.lof_consequence}</div>
          </div>
          <div className="table-responsive">
            <table className="table table-sm table-bordered small">
              <thead style={{ backgroundColor: LIGHT }}>
                <tr><th>Domain</th><th>Residues</th><th>Function</th></tr>
              </thead>
              <tbody>
                {(data.gene_biology.domains || []).map((d, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{d.domain}</td>
                    <td>{d.residues}</td>
                    <td>{d.function}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}
    </div>
  );
}

function PatientsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const phenos = data.phenotype_groups || [];
  const variants = data.variant_distribution || [];
  const neuro = data.neurological_outcomes || {};
  const liver = data.liver_outcomes || {};

  return (
    <div>
      <SectionCard title="👥 Phenotype Distribution (n=40)">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: 'white' }}>
              <tr><th>Phenotype Group</th><th>n</th><th>%</th></tr>
            </thead>
            <tbody>
              {phenos.map((p, i) => (
                <tr key={i}>
                  <td>{p.group}</td>
                  <td>{p.n}</td>
                  <td>{p.pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🧬 Variant Distribution (n=40 patients, seed-539)">
        {variants.map((v, i) => (
          <div key={i} className="mb-3 p-2 rounded small" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between mb-1">
              <strong>{v.variant}</strong>
              <span className="badge" style={{ backgroundColor: COLOR }}>{v.pct}%</span>
            </div>
            <div className="progress mb-1" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${v.pct}%`, backgroundColor: COLOR }} />
            </div>
            <div className="text-muted">{v.effect}</div>
          </div>
        ))}
        <Alert variant="warning" text="No dominant founder mutation: WES/gene panel required for SERAC1 diagnosis — unlike DNAJC19 (Hutterite c.130-1G>C 88%), MECR (Bedouin p.Tyr200His), OPA3 (Iraqi-Jewish p.Gln105*)" />
      </SectionCard>

      <SectionCard title="🧠 Neurological Outcomes">
        <Bar label="Leigh-like MRI (bilateral putamen)" value={neuro.leigh_like_mri_pct || 0} max={100} color="#4a148c" />
        <Bar label="Epilepsy (any)" value={neuro.epilepsy_pct || 0} max={100} color="#e65100" />
        <Bar label="Drug-resistant epilepsy" value={neuro.drug_resistant_epilepsy_pct || 0} max={100} color="#b71c1c" />
        <Bar label="Infantile spasms" value={neuro.infantile_spasms_pct || 0} max={100} color="#880e4f" />
        <Bar label="Dystonia" value={neuro.dystonia_pct || 0} max={100} color="#1565c0" />
        <Bar label="Spasticity" value={neuro.spasticity_pct || 0} max={100} color="#0d47a1" />
        <Bar label="Independent ambulation" value={neuro.independent_ambulation_pct || 0} max={100} color="#2e7d32" />
        <Bar label="Moderate-severe ID" value={neuro.moderate_severe_id_pct || 0} max={100} color="#e65100" />
        <Alert variant="warning" text={`Non-verbal (no spoken language): ${neuro.nonverbal_pct || 0}% — cochlear implant + AAC is primary communication strategy`} />
      </SectionCard>

      <SectionCard title="🫀 Neonatal Liver Dysfunction Outcomes">
        <Bar label="Liver affected (neonatal)" value={liver.liver_affected_pct || 0} max={100} color="#f57f17" />
        <Bar label="Self-limited (resolves by 6-12 mo)" value={liver.self_limited_pct || 0} max={100} color="#2e7d32" />
        <Bar label="Fulminant hepatic failure" value={liver.fulminant_hepatic_failure_pct || 0} max={100} color="#b71c1c" />
        <div className="small mt-2 text-muted"><strong>Median resolution:</strong> {liver.median_resolution_months || 9} months</div>
        <Alert variant="danger" text="Fulminant hepatic failure (~15% of liver-affected cases): poor prognosis; may require liver transplant in neonatal period. Identify SERAC1 early via NBS + urine 3-MGA + SNHL screening." />
        <Alert variant="warning" text="Neonatal liver disease alters VPA prescribing: NEVER start VPA during liver dysfunction period. Wait until LFTs normalise. UDCA for cholestasis support." />
      </SectionCard>
    </div>
  );
}

function HearingMRITab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const ci = data.cochlear_implant_outcomes || [];
  const biomarkers = data.biomarker_summary || {};
  const mga_pheno = data.mga_by_phenotype || [];

  return (
    <div>
      <SectionCard title="👂 Sensorineural Hearing Loss — 100% CARDINAL" borderColor="#b71c1c">
        <Alert variant="danger" text="SNHL is the SINGLE MOST DISTINCTIVE feature of SERAC1 MEGDEL among ALL 3-MGA-uria diseases. No other disease in the 3-MGA classification has SNHL as a cardinal feature." />
        <Alert variant="info" text="Bilateral, sensorineural; onset within first year (detected on newborn hearing screen in most); profound (>70 dB) in ~60% by 18 months." />
        <Alert variant="info" text="Mechanism: cochlear inner hair cells have extreme energy demands → OXPHOS failure in SERAC1 → hair cell death → permanent SNHL. Unlike DCAF17 (SNHL from cochlear maldevelopment), SERAC1 SNHL is degenerative." />
        <Alert variant="warning" text="DDx of 3-MGA-uria + SNHL: SERAC1 alone. If 3-MGA urine elevated AND bilateral SNHL detected on newborn screen → request SERAC1 gene immediately." />
      </SectionCard>

      <SectionCard title="🔊 Cochlear Implant Outcomes (n=35 implanted)" borderColor="#2e7d32">
        {ci.map((c, i) => (
          <div key={i} className="mb-3 p-2 rounded small" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
            <div className="d-flex justify-content-between mb-1">
              <strong>{c.outcome}</strong>
              <span className="badge bg-success">{c.pct_of_ci}% of CI patients</span>
            </div>
            <div className="progress mb-1" style={{ height: 8 }}>
              <div className="progress-bar bg-success" style={{ width: `${c.pct_of_ci}%` }} />
            </div>
            <div className="text-muted">{c.notes}</div>
          </div>
        ))}
        <Alert variant="success" text="Best outcomes: CI placed <18 months + intensive post-CI auditory-verbal therapy. Even patients with severe ID benefit — hearing restoration supports communication development." />
        <Alert variant="warning" text="Anaesthetic alert for CI surgery: SERAC1 = mitochondrial disease → avoid propofol (PRIS risk); use 5% glucose IV; minimise fasting; inform anaesthesia team of OXPHOS dysfunction." />
      </SectionCard>

      <SectionCard title="🧠 Leigh-like MRI — 87% (Bilateral Putamen)" borderColor="#4a148c">
        <Alert variant="info" text="Pattern: bilateral putamen T2 hyperintensity (most common); caudate, brainstem nuclei also involved. 'Leigh-LIKE' — resembles Leigh syndrome but may be incomplete/asymmetric." />
        <Alert variant="danger" text="KEY NEGATIVE: NO globus pallidus iron on T2*/SWI — distinguishes SERAC1 from MECR/MEPAN (bilateral GP iron is PATHOGNOMONIC in MECR) and all NBIA diseases (GP hypointensity on T2*). SERAC1 Leigh-like = T2 bright, not T2 dark." />
        <Alert variant="info" text="Evolution: Leigh-like changes may appear or worsen during metabolic crises (infection, fever, surgery). Post-crisis MRI may partially recover." />
        <Alert variant="warning" text="CSF lactate: often elevated during Leigh-like episodes (energy failure in basal ganglia). Serum lactate + L:P ratio useful during acute encephalopathy. Avoid prolonged fasting (worsens lactic acidosis)." />
      </SectionCard>

      <SectionCard title="🧪 Biomarker Summary">
        <div className="row g-2 small">
          <div className="col-md-4"><strong>3-MGA range:</strong> {biomarkers.mga_range_mmol_cr} mmol/mol Cr</div>
          <div className="col-md-4"><strong>3-MGA mean:</strong> {biomarkers.mga_mean} mmol/mol Cr</div>
          <div className="col-md-4"><strong>Acylcarnitine:</strong> {biomarkers.acylcarnitine_normal_pct}% normal</div>
          <div className="col-md-4"><strong>C4-DC normal:</strong> {biomarkers.c4dc_normal_pct}%</div>
          <div className="col-md-4"><strong>SNHL bilateral:</strong> {biomarkers.snhl_bilateral_pct}%</div>
          <div className="col-md-4"><strong>SNHL profound:</strong> {biomarkers.snhl_profound_pct}%</div>
          <div className="col-md-4"><strong>Low carnitine:</strong> {biomarkers.c0_carnitine_low_pct}%</div>
          <div className="col-md-4"><strong>Lactate elevation:</strong> {biomarkers.lactate_mild_elevation_pct}%</div>
        </div>
        <Alert variant="info" text="KEY NEGATIVE biomarker: acylcarnitine profile NORMAL (100%) in SERAC1 — distinguishes from Barth (TAZ) where C4-DC (3-methylglutarylcarnitine) is elevated. If C4-DC elevated → send TAZ gene testing." />
      </SectionCard>

      <SectionCard title="🔬 3-MGA Level by Phenotype Subtype">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: 'white' }}>
              <tr><th>Phenotype</th><th>Mean 3-MGA (mmol/mol Cr)</th><th>Range</th><th>n</th></tr>
            </thead>
            <tbody>
              {mga_pheno.map((m, i) => (
                <tr key={i}>
                  <td>{m.phenotype}</td>
                  <td className="fw-bold">{m.mean_mga}</td>
                  <td>{m.range}</td>
                  <td>{m.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="info" text="Higher 3-MGA levels correlate with more severe phenotype — Severe MEGDEL has highest overflow (mean 122 mmol/Cr), Mild MEGDEL lowest (mean 52 mmol/Cr)" />
      </SectionCard>
    </div>
  );
}

function TreatmentsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const tx = data.treatment_distribution || [];

  return (
    <div>
      <SectionCard title="💊 Treatment Distribution (n=40)">
        {tx.map((t, i) => (
          <div key={i} className="mb-3 p-2 rounded small" style={{ background: LIGHT }}>
            <div className="d-flex justify-content-between mb-1">
              <div>
                <strong>{t.treatment}</strong>
                <span className="ms-2 badge bg-secondary">{t.indication}</span>
              </div>
              <span className="badge" style={{ backgroundColor: COLOR }}>{t.n}/{data.cohort} ({t.pct}%)</span>
            </div>
            <div className="progress" style={{ height: 8 }}>
              <div className="progress-bar" style={{ width: `${t.pct}%`, backgroundColor: COLOR }} />
            </div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔊 Cochlear Implant — The Most Impactful Intervention" borderColor="#2e7d32">
        <Alert variant="success" text="Cochlear implant (Level B): highly effective in MEGDEL. Even with severe intellectual disability, hearing restoration fundamentally transforms communication ability and quality of life. Recommend CI evaluation before 12-18 months." />
        <Alert variant="warning" text="Anaesthetic safety protocol: SERAC1 = mitochondrial Complex I/IV deficiency → AVOID propofol (PRIS risk). Use sevoflurane or ketamine. Maintain 5% glucose IV. Minimize NPO time. Alert anaesthesiology." />
        <Alert variant="info" text="Post-CI: intensive auditory-verbal therapy required. AAC (augmentative communication) used in combination with CI in most patients. Speech-language pathology is mandatory post-CI." />
      </SectionCard>

      <SectionCard title="⚠️ VPA Prescribing in SERAC1 — Moderate Caution">
        <Alert variant="warning" text="VPA MODERATE CAUTION in SERAC1: Complex I already deficient → VPA Complex I inhibition is additive. NOT absolute CI (lipoic acid pathway intact — contrast MECR where VPA = absolute CI). However: neonatal liver disease history in 67% → hepatotoxicity risk elevated." />
        <Alert variant="danger" text="NEVER start VPA during neonatal liver dysfunction period. Wait until LFTs return to normal. Once liver stable: VPA may be used if LEV fails, with close LFT + NH3 monitoring every 3 months." />
        <Alert variant="success" text="LEV (Levetiracetam) PREFERRED: renal excretion; no hepatic metabolism; no Complex I interaction; broad-spectrum seizure coverage. Same preferred choice as OPA3, DNAJC19, MECR — the 3-MGA-uria AED preference." />
        <Alert variant="info" text="ACTH + VGB (Level A per UKISS): for infantile spasms. VGB not contraindicated in SERAC1 (unlike CP aceruloplasminemia where VGB is absolute CI for retinal toxicity — SERAC1 has no optic atrophy)." />
      </SectionCard>

      <SectionCard title="🍃 KD (Ketogenic Diet) — Investigational">
        <Alert variant="info" text="KD investigational (Level D): some benefit reported in Leigh-like mito encephalopathies with Complex I deficiency. Rationale: ketone bodies bypass Complex I → provide energy to basal ganglia neurons." />
        <Alert variant="warning" text="SERAC1 KD safety: No absolute CI (ketogenesis intact, unlike HMGCL). However: delay KD until liver function normalises — hepatic involvement in neonatal period contraindications KD transiently. Monitor liver enzymes during KD initiation." />
      </SectionCard>

      <SectionCard title="🧪 Mitochondrial Cofactor Cocktail — Level C">
        <Alert variant="info" text="CoQ10 (100-300 mg/day): Complex I/IV support; commonly prescribed in mito disease despite weak RCT evidence; reasonable empiric use in SERAC1." />
        <Alert variant="info" text="Riboflavin (Vit B2, 100-200 mg/day): flavoprotein support for respiratory chain; especially for Complex I flavin subunit stability." />
        <Alert variant="info" text="L-Carnitine (50-100 mg/kg/day): secondary depletion in OXPHOS disorders; supplement if C0 (free carnitine) below 25 µmol/L." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 Clinical Definitions — SERAC1 MEGDEL Syndrome (10 concepts)">
        {defs.map((d, i) => (
          <div key={i} className="mb-4 p-3 rounded" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <div className="fw-bold mb-1" style={{ color: COLOR }}>{i + 1}. {d.term}</div>
            <div className="small mb-2">{d.definition}</div>
            <div className="small text-muted"><strong>Clinical relevance:</strong> {d.relevance}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

export default function SERAC1Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/serac1/overview`).then(r => r.json()),
      fetch(`${API}/api/serac1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/serac1/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '1.25rem 1.5rem' }}>
        <h4 className="mb-1 fw-bold">🧬 SERAC1 — MEGDEL Syndrome</h4>
        <div className="small opacity-75">
          3-Methylglutaconic aciduria + Deafness + Encephalopathy + Leigh-Like · 3-MGA-uria Type V ·
          MAM Phospholipid Remodeling · 6q22.1 · AR · OMIM Gene 614725 · Disease 614739 · n=40 cohort seed-539
        </div>
        <div className="small opacity-75 mt-1">
          SNHL-100%-CARDINAL-No-Other-3-MGA-Disease-Has-SNHL · Leigh-like-MRI-87%-Bilateral-Putamen ·
          NO-GP-Iron-DDx-MECR · NO-DCM-DDx-DNAJC19-Barth · NO-Optic-Atrophy-DDx-OPA3-MECR ·
          Cochlear-Implant-Level-B · LEV-Preferred · VPA-Moderate-Caution · Neonatal-Liver-67%-Transient
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: 'white', borderBottom: '1px solid #dee2e6' }}>
        <div className="container-fluid px-3">
          <ul className="nav nav-tabs border-0">
            {TABS.map((t, i) => (
              <li key={i} className="nav-item">
                <button
                  className={`nav-link ${tab === i ? 'active' : ''}`}
                  style={tab === i ? { color: COLOR, borderBottomColor: COLOR, fontWeight: 600 } : {}}
                  onClick={() => setTab(i)}
                >{t}</button>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Content */}
      <div className="container-fluid px-3 py-3">
        {loading && <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>}
        {error && <div className="alert alert-danger">Error: {error}</div>}
        {!loading && !error && (
          <>
            {tab === 0 && <OverviewTab data={overview} />}
            {tab === 1 && <PatientsTab data={breakdown} />}
            {tab === 2 && <HearingMRITab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
