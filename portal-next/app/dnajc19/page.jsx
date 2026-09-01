'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Cardiac & Biomarkers', 'Treatments', 'Definitions'];
const COLOR = '#004d40';   // dark teal — DNAJC19/DCMA (TIM23 mitochondrial, DCM-dominant)
const LIGHT = '#e0f2f1';

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
          <div className="col-12"><strong>Founder mutation:</strong> {data.founder_mutation}</div>
          <div className="col-md-6"><strong>Protein:</strong> {data.protein}</div>
          <div className="col-md-6"><strong>Category:</strong> {data.category}</div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <SectionCard title="📊 Cohort KPIs (n=40, seed-537)">
        <div className="row g-2">
          <KPI label="DCM" value="100%" color="#b71c1c" />
          <KPI label="Cerebellar Ataxia" value={`${kpis.cerebellar_ataxia_pct}%`} color={COLOR} />
          <KPI label="3-MGA-uria" value="100%" color={COLOR} />
          <KPI label="Male Genital Anom." value={`${kpis.male_genital_anomalies_pct}%`} color="#4a148c" />
          <KPI label="Mild ID" value={`${kpis.mild_id_pct}%`} color="#e65100" />
          <KPI label="Transplant Rate" value={`${kpis.transplant_pct}%`} color="#b71c1c" />
          <KPI label="Seizures (rare)" value={`${kpis.seizure_pct}%`} color="#1565c0" />
          <KPI label="Optic Atrophy" value="0%" color="#9e9e9e" />
          <KPI label="Mean EF at Dx" value={`${kpis.mean_ef_at_dx}%`} color="#b71c1c" />
          <KPI label="Mean 3-MGA" value={`${kpis.mean_mga_mmol} mmol/Cr`} color={COLOR} />
          <KPI label="DCM Onset" value={`${kpis.mean_dcm_onset_yr}yr`} color="#b71c1c" />
          <KPI label="Ataxia Onset" value={`${kpis.mean_ataxia_onset_yr}yr`} color={COLOR} />
        </div>
      </SectionCard>

      {/* Clinical Highlights */}
      <SectionCard title="⚡ Clinical Highlights">
        {highlights.map((h, i) => <Alert key={i} variant={
          h.includes('CARDINAL') || h.includes('~30%') || h.includes('transplant') ? 'danger' :
          h.includes('NO optic') || h.includes('NO chorea') || h.includes('NON-PROGRESSIVE') ? 'warning' :
          h.includes('LEV') || h.includes('ACE') ? 'success' : 'info'
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
        <SectionCard title="🔬 DDx Table: DNAJC19 vs OPA3 vs MECR vs Barth (TAZ)">
          <div className="table-responsive">
            <table className="table table-sm table-bordered small">
              <thead style={{ backgroundColor: COLOR, color: 'white' }}>
                <tr>
                  <th>Feature</th>
                  <th>DNAJC19 DCMA</th>
                  <th>OPA3 Costeff</th>
                  <th>MECR MEPAN</th>
                  <th>Barth (TAZ)</th>
                </tr>
              </thead>
              <tbody>
                {ddx.map((row, i) => (
                  <tr key={i}>
                    <td className="fw-bold">{row.feature}</td>
                    <td>{row.dnajc19_dcma}</td>
                    <td>{row.opa3_costeff}</td>
                    <td>{row.mecr_mepan}</td>
                    <td>{row.barth_taz}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      {/* Gene Biology */}
      {data.gene_biology && (
        <SectionCard title="🔬 Protein Biology — DNAJC19 (TIM23 Co-chaperone)">
          <div className="row g-2 small mb-3">
            <div className="col-md-4"><strong>Length:</strong> {data.gene_biology.protein_length} aa</div>
            <div className="col-md-4"><strong>Complex:</strong> {data.gene_biology.complex}</div>
            <div className="col-md-4"><strong>Partner:</strong> {data.gene_biology.partner}</div>
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
  const sex = data.sex_specific || {};

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

      <SectionCard title="🧬 Variant Distribution (n=40 patients, seed-537)">
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
      </SectionCard>

      <SectionCard title="🧠 Neurological Outcomes">
        <Bar label="Cerebellar Ataxia" value={neuro.cerebellar_ataxia_pct || 0} max={100} />
        <Bar label="Non-progressive Course" value={neuro.nonprogressive_pct || 0} max={100} color="#2e7d32" />
        <Bar label="Mild ID" value={neuro.mild_id_pct || 0} max={100} color="#e65100" />
        <Bar label="Independent Ambulation" value={neuro.independent_ambulation_pct || 0} max={100} color="#1565c0" />
        <Bar label="Seizures (rare)" value={neuro.seizure_pct || 0} max={100} color="#6a1b9a" />
        <Alert variant="success" text="Seizure control rate: 100% (rare seizures, all controlled with LEV)" />
      </SectionCard>

      <SectionCard title="⚧ Sex-Specific Features (Male Genital Anomalies)">
        <div className="row g-2 small">
          <div className="col-md-3"><strong>Males:</strong> {sex.male_n || 0}</div>
          <div className="col-md-3"><strong>Females:</strong> {sex.female_n || 0}</div>
          <div className="col-md-3"><strong>Cryptorchidism:</strong> {sex.cryptorchidism_n || 0}/{sex.male_n || 0} males ({sex.cryptorchidism_pct_males || 0}%)</div>
          <div className="col-md-3"><strong>Hypospadias:</strong> {sex.hypospadias_n || 0}/{sex.male_n || 0} males ({sex.hypospadias_pct_males || 0}%)</div>
        </div>
        <Alert variant="warning" text="Male genital anomalies (cryptorchidism ~77%, hypospadias ~23%) are UNIQUE to DNAJC19 among all 3-MGA-uria diseases — pathognomonic DDx clue in male infants with DCM + 3-MGA" />
        <Alert variant="info" text="Absent in all females; surgical correction (orchidopexy) is standard for cryptorchidism" />
      </SectionCard>
    </div>
  );
}

function CardiacTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const cardiac = data.cardiac_outcomes || {};
  const ef_age = data.ef_by_age_group || [];
  const biomarkers = data.biomarker_summary || {};
  const mga_pheno = data.mga_by_phenotype || [];

  return (
    <div>
      <SectionCard title="❤️ Cardiac Outcomes (n=40)">
        <Bar label="LV Dilation" value={cardiac.lv_dilation_pct || 0} max={100} color="#b71c1c" />
        <Bar label="LBBB on ECG" value={cardiac.lbbb_ecg_pct || 0} max={100} color="#b71c1c" />
        <Bar label="Conduction Defects" value={cardiac.conduction_defect_pct || 0} max={100} color="#c62828" />
        <Bar label="Heart Transplant" value={cardiac.transplant_rate_pct || 0} max={100} color="#880e4f" />
        <Bar label="Stable on Medical Mgmt" value={cardiac.stable_medical_mgmt_pct || 0} max={100} color="#2e7d32" />
        <Alert variant="danger" text={`Cardiac death (no transplant): ${cardiac.cardiac_death_no_transplant_pct || 0}% — emphasises need for early cardiac monitoring and transplant evaluation when EF<25%`} />
      </SectionCard>

      <SectionCard title="📈 EF at Diagnosis by Age Group">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead style={{ backgroundColor: COLOR, color: 'white' }}>
              <tr><th>Age Group</th><th>Mean EF (%)</th><th>EF Range</th><th>Transplants</th></tr>
            </thead>
            <tbody>
              {ef_age.map((e, i) => (
                <tr key={i}>
                  <td>{e.age_group}</td>
                  <td className="fw-bold">{e.mean_ef}%</td>
                  <td>{e.range}%</td>
                  <td>{e.transplant_n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Alert variant="warning" text="EF < 25% = transplant evaluation threshold; early-onset (<1yr) patients have lower mean EF and higher transplant risk" />
      </SectionCard>

      <SectionCard title="🧪 Biomarker Summary">
        <div className="row g-2 small">
          <div className="col-md-4"><strong>3-MGA range:</strong> {biomarkers.mga_range_mmol_cr} mmol/mol Cr</div>
          <div className="col-md-4"><strong>3-MGA mean:</strong> {biomarkers.mga_mean} mmol/mol Cr</div>
          <div className="col-md-4"><strong>EF range:</strong> {biomarkers.ef_range_pct}%</div>
          <div className="col-md-4"><strong>EF mean at Dx:</strong> {biomarkers.ef_mean}%</div>
          <div className="col-md-4"><strong>Low carnitine:</strong> {biomarkers.c0_carnitine_low_pct}%</div>
          <div className="col-md-4"><strong>Mild lactate elevation:</strong> {biomarkers.lactate_mild_elevation_pct}%</div>
        </div>
        <Alert variant="info" text="KEY NEGATIVE biomarker: acylcarnitine profile NORMAL in DNAJC19 — DDx from Barth (TAZ) which shows elevated C4-DC (3-methylglutarylcarnitine) on acylcarnitine profile" />
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

      <SectionCard title="🫀 Cardiac Pharmacotherapy — Level A">
        <Alert variant="info" text="ACE Inhibitors (captopril/enalapril/lisinopril) — 100% of patients. Reduce afterload, prevent LV remodeling. Start at diagnosis regardless of symptoms. Monitor: BP, renal function, K+." />
        <Alert variant="info" text="Beta-Blockers (carvedilol preferred; bisoprolol/metoprolol succinate alternatives) — 93% of patients. Reverse cardiac remodeling, reduce HF mortality. Start AFTER ACE stabilisation; titrate from low dose." />
        <Alert variant="info" text="Diuretics (furosemide + spironolactone) — 75% of patients. Furosemide for acute decompensation; spironolactone for aldosterone blockade and K+ preservation." />
      </SectionCard>

      <SectionCard title="⚠️ NOT Indicated in DNAJC19-DCMA">
        <Alert variant="danger" text="Tetrabenazine / Deutetrabenazine: NOT indicated — no chorea in DCMA. Chorea is a feature of OPA3/Costeff, NOT DNAJC19. Prescribing VMAT2 inhibitors would worsen cardiac function (reduced BP, sedation)." />
        <Alert variant="danger" text="Baclofen: NOT indicated — no spastic paraplegia in DCMA. Spastic paraplegia is a feature of OPA3/Costeff (50-60%), absent in DNAJC19." />
        <Alert variant="warning" text="VPA: MODERATE CAUTION (NOT absolute CI). Lipoic acid pathway is INTACT in DNAJC19 — VPA absolute CI mechanism (MECR/MEPAN) does not apply. However, OXPHOS dysfunction may impair urea cycle → hyperammonemia risk; monitor NH3 + LFTs. Use LEV first." />
        <Alert variant="warning" text="PHT/CBZ: Avoid if possible due to CYP450 induction affecting cardiac drug levels (warfarin, amiodarone, beta-blockers). Cardiac drug interaction is primary concern, different from OPA3 reasoning." />
      </SectionCard>

      <SectionCard title="🏥 Heart Transplant Decision">
        <Alert variant="danger" text="Transplant threshold: EF < 25% with medical-refractory symptoms → transplant evaluation." />
        <Alert variant="warning" text="Post-transplant: Cardiac function improves, but cerebellar ataxia, mild ID, and male genital anomalies persist — they are systemic mitochondrial manifestations, not cardiac-only." />
        <Alert variant="info" text="Counselling: ~30% require transplant; ~15% cardiac death without transplant. Outcomes generally good post-transplant. Multidisciplinary team: paediatric cardiology + neurology + clinical genetics required." />
      </SectionCard>

      <SectionCard title="💊 LEV Preference Rationale">
        <Alert variant="success" text="LEV (Levetiracetam) PREFERRED for rare seizures in DNAJC19-DCMA: renal excretion (no hepatic metabolism); no mitochondrial interactions; broad-spectrum; no cardiac drug interactions. Same preference as OPA3, MECR, and DCAF17." />
        <Alert variant="info" text="L-Carnitine (50-100 mg/kg/day): supplemented in 70% of patients for secondary carnitine depletion common in OXPHOS disorders. Check C0 (free carnitine) at each visit." />
      </SectionCard>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <div className="text-center py-4 text-muted">Loading...</div>;
  const defs = data.definitions || [];

  return (
    <div>
      <SectionCard title="📖 Clinical Definitions — DNAJC19 DCMA Syndrome (10 concepts)">
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

export default function DNAJC19Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/dnajc19/overview`).then(r => r.json()),
      fetch(`${API}/api/dnajc19/breakdown`).then(r => r.json()),
      fetch(`${API}/api/dnajc19/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#f8f9fa' }}>
      {/* Header */}
      <div style={{ background: COLOR, color: 'white', padding: '1.25rem 1.5rem' }}>
        <h4 className="mb-1 fw-bold">🧬 DNAJC19 — DCMA Syndrome</h4>
        <div className="small opacity-75">
          Dilated Cardiomyopathy with Ataxia · 3-MGA-uria Type III · TIM23 Co-chaperone · 3q26.33 · AR ·
          OMIM Gene 608977 · Disease 610198 · n=40 cohort seed-537
        </div>
        <div className="small opacity-75 mt-1">
          DCM-100% CARDINAL · Cerebellar Ataxia 95% Nonprogressive · Male Genital Anomalies 75% Males ·
          Hutterite Founder c.130-1G>C (88%) · NO Optic Atrophy (DDx OPA3) · NO Chorea (DDx OPA3) ·
          ACE+BB Level A · Heart Transplant 30%
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
            {tab === 2 && <CardiacTab data={breakdown} />}
            {tab === 3 && <TreatmentsTab data={breakdown} />}
            {tab === 4 && <DefinitionsTab data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
