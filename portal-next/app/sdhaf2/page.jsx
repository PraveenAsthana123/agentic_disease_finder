'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Tumours & Features', 'CII Assembly & DDx', 'Definitions'];
const COLOR = '#880e4f';   // deep pink/maroon — PGL2 paraganglioma, distinct from SDHAF1 purple
const LIGHT = '#fce4ec';

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
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{value}%</span>
      </div>
      <div className="progress" style={{ height: 12 }}>
        <div className="progress-bar" style={{ width: `${value}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const mod = data.sdhaf2_module_summary || {};
  const cs  = data.cohort_summary || {};
  const kf  = data.key_clinical_facts || {};

  return (
    <div>
      <SectionCard>
        <strong>🩺 SDHAF2 (SDH5) — PGL2 Hereditary Paraganglioma / SDHA Flavinylation Factor / AD + Maternal Imprinting</strong> —{' '}
        SDHAF2 is a 166-aa mitochondrial matrix protein that covalently attaches FAD to SDHA histidine-99
        (His99), enabling succinate oxidation by Complex II (SDH). Without SDHAF2, SDHA remains apo-protein,
        CII assembly fails, succinate accumulates, PHD enzymes are inhibited, HIF1α stabilises (pseudo-hypoxia),
        and paraganglioma/pheochromocytoma develops.
        <br /><br />
        <span className="badge me-1" style={{ background: COLOR }}>AD — Dominant</span>
        <span className="badge bg-warning text-dark me-1">MATERNAL IMPRINTING</span>
        <span className="badge bg-secondary me-1">11q13.1</span>
        <span className="badge bg-danger me-1">Paraganglioma PGL2</span>
        <br /><br />
        <strong>⚠️ MATERNAL IMPRINTING — CRITICAL:</strong> Only <em>paternal</em> SDHAF2 mutations cause disease.
        Female carriers are unaffected and their children are <em>not</em> at risk.
        Children of <em>male</em> carriers: 50% risk. This is NOT conventional autosomal dominant inheritance.
      </SectionCard>

      <div className="row g-2 mb-4">
        <KPI label="Gene"           value="SDHAF2"           color={COLOR} />
        <KPI label="OMIM Gene"      value="*613019"          color={COLOR} />
        <KPI label="Disease"        value="PGL2 #601650"     color={COLOR} />
        <KPI label="Chromosome"     value="11q13.1"          color={COLOR} />
        <KPI label="Inheritance"    value="AD+Imprinting"    color="#c62828" />
        <KPI label="Cohort (N)"     value={data.n_patients}  color={COLOR} />
        <KPI label="HNPGL"          value="~78%"             color={COLOR} />
        <KPI label="Adrenal PCC"    value="~15%"             color={COLOR} />
        <KPI label="Bilateral PGL"  value="~30%"             color={COLOR} />
        <KPI label="Malignancy"     value="~5%"              color="#388e3c" />
        <KPI label="SDH-GIST"       value="~8%"              color={COLOR} />
        <KPI label="Paternal Only"  value="100%"             color="#c62828" />
      </div>

      <SectionCard title="⚠️ SDHAF2 Maternal Imprinting — Genetic Counselling">
        <table className="table table-sm table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>Carrier Parent</th><th>Mutation Allele Transmitted As</th>
              <th>Imprinting Status in Child</th><th>Disease Risk in Child</th>
            </tr>
          </thead>
          <tbody>
            <tr className="table-success">
              <td>Mother (female carrier)</td>
              <td>MATERNAL allele</td>
              <td>SILENCED (methylated)</td>
              <td className="fw-bold text-success">0% — NOT at risk</td>
            </tr>
            <tr className="table-danger">
              <td>Father (male carrier)</td>
              <td>PATERNAL allele</td>
              <td>ACTIVE (expressed)</td>
              <td className="fw-bold text-danger">50% — AT RISK (PGL2)</td>
            </tr>
          </tbody>
        </table>
        <p className="small text-muted mb-0">
          Analogous to SDHD (PGL1, 11q23.1) — also maternal imprinting. Contrast SDHB/SDHC: NO imprinting.
          WES must confirm paternal origin of SDHAF2 mutation for correct risk assessment.
        </p>
      </SectionCard>

      <SectionCard title="⚙️ SDHAF2 Role in CII Assembly (Step 1 — SDHA Flavinylation)">
        <p className="small">
          <strong>SDHAF2 role:</strong> Step 1 — binds SDHA → positions His99 → autocatalytic FAD covalent attachment →
          flavinylated SDHA can enter Step 2 (SDHAF1 delivers FeS to SDHB) → CII holoenzyme
        </p>
        {mod.sdhaf2_vs_sdhaf1 && (
          <p className="small text-muted mb-0"><strong>vs SDHAF1:</strong> {mod.sdhaf2_vs_sdhaf1}</p>
        )}
      </SectionCard>

      <SectionCard title="📊 Key Clinical Facts">
        <div className="row">
          <div className="col-md-6">
            <Bar label="Head-neck PGL (HNPGL)"    value={78} />
            <Bar label="Carotid body tumour"       value={55} />
            <Bar label="Jugulotympanic PGL"        value={35} />
            <Bar label="Vagal paraganglioma"       value={22} />
          </div>
          <div className="col-md-6">
            <Bar label="Bilateral / multicentric"  value={30} color="#c62828" />
            <Bar label="Adrenal PCC"               value={15} />
            <Bar label="Catecholamine excess"      value={20} />
            <Bar label="SDH-deficient GIST"        value={8}  />
          </div>
        </div>
        <div className="alert alert-warning small mt-2 mb-0">
          <strong>Malignancy risk ~5%</strong> — significantly lower than SDHB (~20–50%).
          Annual surveillance imaging mandatory given bilateral tumour risk (30%).
        </div>
      </SectionCard>

      <SectionCard title="🧬 Cohort Summary (N=40, Seed 703)">
        <div className="row small">
          <div className="col-md-4">
            <strong>Sex:</strong> {cs.male}M / {cs.female}F<br />
            <strong>Mean age at Dx:</strong> {cs.avg_age_at_dx} yr
          </div>
          <div className="col-md-4">
            <strong>Severity:</strong>
            <ul className="mb-0">
              {cs.severity_distribution && Object.entries(cs.severity_distribution).map(([k,v]) => (
                <li key={k}>{k}: {v}</li>
              ))}
            </ul>
          </div>
          <div className="col-md-4">
            <strong>Tumour types:</strong>
            <ul className="mb-0">
              {cs.tumour_type_distribution && Object.entries(cs.tumour_type_distribution).slice(0, 5).map(([k,v]) => (
                <li key={k}>{k}: {v}</li>
              ))}
            </ul>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="💊 Surveillance Protocol">
        <table className="table table-sm table-bordered small">
          <thead className="table-light">
            <tr><th>Target</th><th>Modality</th><th>Frequency</th></tr>
          </thead>
          <tbody>
            <tr><td>Head-neck PGL</td><td>MRI/MRA (skull base → aortic arch)</td><td>Annual</td></tr>
            <tr><td>Adrenal / retroperitoneal</td><td>Contrast CT or MRI</td><td>Annual</td></tr>
            <tr><td>Biochemistry</td><td>Plasma/urine metanephrines, CgA</td><td>Annual</td></tr>
            <tr><td>SDH-deficient GIST</td><td>Upper GI endoscopy / abdominal MRI</td><td>If symptomatic</td></tr>
            <tr><td>Start age</td><td>—</td><td>15 yr (or 5 yr before youngest affected)</td></tr>
            <tr><td>Genetic testing</td><td>Confirm paternal origin of SDHAF2 mutation</td><td>At diagnosis</td></tr>
          </tbody>
        </table>
        <div className="alert alert-info small mb-0">
          <strong>Only children of MALE SDHAF2 carriers</strong> require surveillance.
          Children of female carriers: NOT at risk (maternal imprinting silences the mutation).
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Tumours & Features ─────────────────────────────────────────────────────
function TumoursTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb  = data.variant_breakdown || [];
  const tb  = data.tumour_type_breakdown || [];
  const cb  = data.clinical_feature_breakdown || [];
  const sl  = data.severity_logic || {};
  const tx  = data.treatment_summary || {};

  return (
    <div>
      <SectionCard title="🔬 SDHAF2 Variant Breakdown">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>Protein</th><th>cDNA</th><th>Domain</th>
                <th>Severity</th><th>Penetrance</th><th>N</th><th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {vb.map((v, i) => (
                <tr key={i} className={v.severity === 'severe' ? 'table-danger' : v.severity === 'intermediate' ? 'table-warning' : ''}>
                  <td className="fw-bold">{v.hgvs_p}</td>
                  <td>{v.hgvs_c}</td>
                  <td>{v.domain}</td>
                  <td>{v.severity}</td>
                  <td>{v.penetrance_pct}%</td>
                  <td>{v.n_patients}</td>
                  <td className="text-muted">{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="row mt-2 small">
          {Object.entries(sl).map(([k,v]) => (
            <div key={k} className="col-md-4 mb-2">
              <strong>{k}:</strong> {v}
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🎯 Tumour Type Distribution">
        <div className="row">
          <div className="col-md-6">
            {tb.map((t, i) => (
              <Bar key={i} label={t.tumour_type} value={t.freq_pct} />
            ))}
          </div>
          <div className="col-md-6">
            <strong>Clinical Features:</strong>
            {cb.map((c, i) => (
              <Bar key={i} label={c.feature} value={c.freq_pct} color="#6a1b9a" />
            ))}
          </div>
        </div>
      </SectionCard>

      <SectionCard title="💊 Treatment Summary">
        <table className="table table-sm table-bordered small">
          <thead className="table-light">
            <tr><th>Modality</th><th>Indication / Notes</th></tr>
          </thead>
          <tbody>
            {Object.entries(tx).map(([k,v]) => (
              <tr key={k}>
                <td className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="alert alert-danger small mb-0">
          <strong>⚠️ PCC Pre-operative Protocol:</strong> Alpha-blockade (phenoxybenzamine) BEFORE
          beta-blockade — reversing this order risks fatal hypertensive crisis from unopposed
          alpha-adrenergic stimulation.
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: CII Assembly & DDx ─────────────────────────────────────────────────────
function AssemblyDDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const pathway  = data.cii_assembly_pathway || [];
  const ddx      = data.ddx_table || [];
  const imp      = data.imprinting_analysis || {};

  return (
    <div>
      <SectionCard title="⚙️ CII Assembly Pathway — SDHAF2 at Step 1 (SDHA Flavinylation)">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Step</th><th>Factor</th><th>Role</th><th>Disease</th></tr>
            </thead>
            <tbody>
              {pathway.map((s, i) => (
                <tr key={i} className={s.highlight ? 'table-danger fw-bold' : ''}>
                  <td>{s.step}</td>
                  <td>{s.factor}</td>
                  <td>{s.role}</td>
                  <td>{s.disease}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="⚠️ Maternal Imprinting — Mechanism & Counselling">
        <p className="small"><strong>Mechanism:</strong> {imp.mechanism}</p>
        <p className="small"><strong>Consequence:</strong> {imp.consequence}</p>
        <p className="small"><strong>Penetrance rule:</strong> {imp.penetrance_rule}</p>
        <p className="small"><strong>Clinical counselling:</strong> {imp.clinical_counselling}</p>
        <p className="small"><strong>Genetic testing:</strong> {imp.genetic_testing_implication}</p>
        {imp.analogous_loci && (
          <p className="small text-muted mb-0">
            <strong>Analogous imprinted loci:</strong> {imp.analogous_loci.join(' | ')}
          </p>
        )}
      </SectionCard>

      <SectionCard title="🔑 DDx — Hereditary Paraganglioma Syndromes">
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>Gene / Syndrome</th><th>Locus</th><th>Inheritance</th>
                <th>Disease</th><th>Malignancy</th><th>Imprinting</th><th>Distinguishing Feature</th>
              </tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} className={d.gene.includes('SDHAF2') ? 'table-danger fw-bold' : ''}>
                  <td>{d.gene}</td>
                  <td>{d.locus}</td>
                  <td>{d.inheritance}</td>
                  <td>{d.disease}</td>
                  <td>{d.malignancy}</td>
                  <td>{d.imprinting}</td>
                  <td className="text-muted">{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="alert alert-warning small mt-2 mb-0">
          <strong>SDHB vs SDHAF2:</strong> SDHB has NO maternal imprinting and carries the highest
          malignancy risk (~20–50%). SDHAF2 has maternal imprinting and low malignancy (~5%).
          Panel testing mandatory — phenotype alone cannot distinguish.
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const kd = data.key_distinctions || {};
  const ci = data.drug_contraindications || [];

  const fields = [
    ['gene_definition',        '🧬 Gene Definition'],
    ['disease_definition',     '🩺 Disease Definition'],
    ['inheritance_definition', '🔗 Inheritance + Maternal Imprinting'],
    ['mechanism_definition',   '⚙️ Mechanism: CII Assembly → Pseudo-Hypoxia'],
    ['imprinting_definition',  '⚠️ Genomic Imprinting — Detail'],
    ['surveillance_definition','📋 Surveillance Protocol'],
    ['treatment_definition',   '💊 Treatment'],
  ];

  return (
    <div>
      {fields.map(([key, label]) => data[key] && (
        <SectionCard key={key} title={label}>
          <p className="small mb-0">{data[key]}</p>
        </SectionCard>
      ))}

      <SectionCard title="🔑 Key Distinctions">
        {Object.entries(kd).map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <strong>{k.replace(/_/g, ' → ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Drug Considerations (PCC Context)">
        {ci.map((d, i) => (
          <div key={i} className="mb-3 border-bottom pb-2 small">
            <strong className={d.level.includes('ABSOLUTE') || d.level.includes('CRITICAL') ? 'text-danger' : 'text-warning'}>
              {d.drug}
            </strong>
            <span className="badge ms-2 bg-danger">{d.level}</span>
            <br /><em>Mechanism:</em> {d.mechanism}
            <br /><em>Alternative:</em> {d.alternative}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Root component ─────────────────────────────────────────────────────────────
export default function SDHAF2Page() {
  const [tab, setTab]   = useState(0);
  const [ov,  setOv]   = useState(null);
  const [bd,  setBd]   = useState(null);
  const [def, setDef]  = useState(null);
  const [err, setErr]  = useState('');

  useEffect(() => {
    const load = async (endpoint, setter) => {
      try {
        const r = await fetch(`${API}${endpoint}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        setter(await r.json());
      } catch (e) { setErr(e.message); }
    };
    load('/api/sdhaf2/overview',    setOv);
    load('/api/sdhaf2/breakdown',   setBd);
    load('/api/sdhaf2/definitions', setDef);
  }, []);

  const gene = ov?.gene || 'SDHAF2';
  const disease = 'PGL2 — Paragangliomas 2 / SDHA Flavinylation Factor / AD + Maternal Imprinting';

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <span className="fs-4 me-2">🩺</span>
        <div>
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>{gene}</h4>
          <div className="text-muted small">{disease}</div>
        </div>
        <span className="badge ms-auto" style={{ background: COLOR }}>OMIM *613019 / #601650</span>
        <span className="badge bg-warning text-dark">MATERNAL IMPRINTING</span>
      </div>

      {err && <div className="alert alert-danger">API error: {err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab    data={ov}  />}
      {tab === 1 && <TumoursTab     data={bd}  />}
      {tab === 2 && <AssemblyDDxTab data={bd}  />}
      {tab === 3 && <DefinitionsTab data={def} />}

      <div className="text-muted small mt-4 border-top pt-2">
        <strong>SDHAF2 (SDH5)</strong> — SDHA Flavinylation Factor / PGL2 Paraganglioma /
        AD with Maternal Imprinting / 11q13.1 / OMIM Gene *613019 / Disease #601650 /
        40-patient cohort seed-703 / 3 endpoints /api/sdhaf2/overview|breakdown|definitions
      </div>
    </div>
  );
}
