'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Features', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — CIII, dual-disease
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';   // secondary indigo
const COLOR3 = '#b71c1c';   // red for GRACILE danger

function KPI({ label, value, color = COLOR }) {
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const s = data.cohort_statistics || {};
  const feats = data.cohort_summary_features || [];

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 BCS1L — BCS1 Homolog, Ubiquinol-Cytochrome C Reductase Complex Chaperone
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Complex:</strong> {data.complex}
        </p>
        <p className="mb-1 small">
          <strong>Diseases:</strong> GRACILE Syndrome (OMIM #{data.omim_gracile}) · Björnstad Syndrome (OMIM #{data.omim_bjornstad}) &nbsp;|&nbsp;
          <strong>Inheritance:</strong> {data.inheritance}
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🔴 BCS1L: DUAL-DISEASE GENE — biallelic severe alleles → GRACILE (lethal neonatal);
          milder alleles → Björnstad (SNHL + pili torti, long survival).
          AAA+ ATPase inserts RISP into CIII Qo site — rate-limiting CIII assembly step.
          KD / Metformin / VPA / Linezolid: ABSOLUTE CONTRAINDICATIONS.
        </p>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Patients (n)"       value={s.n_patients} />
        <KPI label="GRACILE"            value={`${s.gracile_pct}%`}          color={COLOR3} />
        <KPI label="Björnstad"          value={`${s.bjornstad_pct}%`}        color={COLOR} />
        <KPI label="Encephalomyopathy"  value={`${s.encephalomyopathy_pct}%`} />
        <KPI label="Leigh-like"         value={`${s.leigh_pct}%`}            color={COLOR3} />
        <KPI label="Lactic acidosis"    value={`${s.lactic_acidosis_pct}%`}  color={COLOR3} />
        <KPI label="Aminoaciduria"      value={`${s.aminoaciduria_pct}%`} />
        <KPI label="Cholestasis"        value={`${s.cholestasis_pct}%`} />
        <KPI label="Iron overload"      value={`${s.iron_overload_pct}%`} />
        <KPI label="SNHL"               value={`${s.snhl_pct}%`}             color={COLOR} />
        <KPI label="Pili torti"         value={`${s.pili_torti_pct}%`}       color={COLOR2} />
        <KPI label="Survived"           value={`${s.survived_pct}%`} />
      </div>
      <div className="row mb-3">
        <KPI label="CIII activity mean" value={`${s.ciii_mean_pct}%`} />
        <KPI label="CIII range"         value={s.ciii_range} />
        <KPI label="Cardiomyopathy"     value={`${s.cardiomyopathy_pct}%`} />
        <KPI label="Seizures"           value={`${s.seizures_pct}%`} />
        <KPI label="IUGR"               value={`${s.iugr_pct}%`} />
        <KPI label="Age mean (yr)"      value={s.age_mean} />
      </div>

      {/* Clinical feature bars */}
      <SectionCard title="Clinical Features (Frequency %)">
        {feats.map(f => (
          <Bar key={f.feature} label={f.feature} value={f.freq_pct}
               color={
                 f.feature === 'GRACILE syndrome' ? COLOR3 :
                 f.feature === 'Lactic acidosis'  ? COLOR3 :
                 f.feature === 'Björnstad syndrome' ? COLOR :
                 f.freq_pct >= 40 ? COLOR : COLOR2
               } />
        ))}
      </SectionCard>

      {/* Key facts */}
      <SectionCard title="Key Clinical Facts" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {(data.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Top variants */}
      <SectionCard title="Top Variant Genotypes in Cohort">
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Genotype (compound het / hom)</th><th>Count</th><th>Freq%</th></tr></thead>
            <tbody>
              {(data.top_variants_cohort || []).map((v, i) => (
                <tr key={i}>
                  <td><code>{v.variant}</code></td>
                  <td>{v.count}</td>
                  <td>{v.freq_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Patient table */}
      <SectionCard title={`Patient Table (n=${s.n_patients}, seed ${data.seed})`}>
        <div className="table-responsive" style={{ maxHeight: 400, overflowY: 'auto' }}>
          <table className="table table-sm table-striped mb-0">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Sex</th><th>Phenotype</th><th>Age Dx</th>
                <th>Genotype</th><th>LA</th><th>Chol</th>
                <th>Amino</th><th>Fe↑</th><th>SNHL</th><th>PT</th>
                <th>CIII%</th><th>Survived</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id} style={p.phenotype === 'GRACILE' ? { background: '#ffebee' } : {}}>
                  <td>{p.id}</td>
                  <td>{p.sex}</td>
                  <td>
                    <span className={`badge ${
                      p.phenotype === 'GRACILE' ? 'bg-danger' :
                      p.phenotype === 'Björnstad' ? 'bg-primary' :
                      p.phenotype === 'CIII-Leigh' ? 'bg-warning text-dark' :
                      'bg-secondary'
                    }`}>{p.phenotype}</span>
                  </td>
                  <td>{p.age_at_diagnosis}</td>
                  <td><code style={{ fontSize: '0.7rem' }}>{p.variant}</code></td>
                  <td>{p.lactic_acidosis ? '✓' : ''}</td>
                  <td>{p.cholestasis ? '✓' : ''}</td>
                  <td>{p.aminoaciduria ? '✓' : ''}</td>
                  <td>{p.iron_overload ? <span style={{ color: COLOR3 }} className="fw-bold">✓</span> : ''}</td>
                  <td>{p.snhl ? '✓' : ''}</td>
                  <td>{p.pili_torti ? '✓' : ''}</td>
                  <td><small>{p.ciii_residual_activity_pct}%</small></td>
                  <td>{p.survived
                    ? <span style={{ color: 'green' }}>✓</span>
                    : <span style={{ color: COLOR3 }} className="fw-bold">✗</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">
          LA = lactic acidosis · Chol = cholestasis · Amino = aminoaciduria · Fe↑ = iron overload · PT = pili torti
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Features ──────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      {/* Structural features */}
      <SectionCard title="BCS1L Structural Features">
        {data.structural_features && Object.entries(data.structural_features).map(([k, v]) => (
          <p key={k} className="mb-1 small"><strong>{k.replace(/_/g, ' ')}:</strong> {String(v)}</p>
        ))}
      </SectionCard>

      {/* Variants table */}
      <SectionCard title="Pathogenic / Likely-Pathogenic Variants in BCS1L">
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead className="table-dark">
              <tr>
                <th>cDNA</th><th>Protein</th><th>Location</th>
                <th>Pathogenicity%</th><th>Severity</th>
                <th>Phenotype</th><th>Population</th><th>Cohort n</th>
              </tr>
            </thead>
            <tbody>
              {(data.variants || []).map((v, i) => (
                <tr key={i}>
                  <td><code>{v.cDNA}</code></td>
                  <td><code>{v.protein}</code></td>
                  <td><small>{v.location}</small></td>
                  <td>
                    <div className="progress" style={{ height: 10, minWidth: 60 }}>
                      <div className="progress-bar"
                           style={{ width: `${v.pathogenicity_pct}%`, backgroundColor: COLOR }} />
                    </div>
                    <small>{v.pathogenicity_pct}%</small>
                  </td>
                  <td>
                    <span className={`badge ${
                      v.severity.includes('GRACILE') ? 'bg-danger' :
                      v.severity.includes('Björnstad') ? 'bg-primary' :
                      v.severity.includes('Leigh') ? 'bg-warning text-dark' :
                      v.severity.includes('Severe') ? 'bg-warning text-dark' :
                      v.severity.includes('Moderate-Severe') ? 'bg-warning text-dark' :
                      v.severity.includes('Moderate') ? 'bg-info text-dark' : 'bg-secondary'
                    }`}>{v.severity}</span>
                  </td>
                  <td><small>{v.phenotype}</small></td>
                  <td><small>{v.population}</small></td>
                  <td>{v.cohort_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Variant consequence cards */}
      {(data.variants || []).map((v, i) => (
        <div key={i} className="card mb-2 shadow-sm">
          <div className="card-body py-2">
            <div className="d-flex align-items-start gap-2">
              <code className="text-nowrap" style={{ color: COLOR }}>{v.protein}</code>
              <div className="small">{v.consequence}</div>
            </div>
            <div className="small text-muted mt-1"><em>{v.reference}</em></div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const t   = data.treatment_summary || {};
  const ddx = data.key_ddx || [];

  return (
    <div>
      {/* CRITICAL pharmacology alert */}
      <div className="alert mb-4" style={{ background: '#ffebee', borderLeft: `5px solid ${COLOR3}` }}>
        <strong style={{ color: COLOR3 }}>🔴 BCS1L / CIII Deficiency — ABSOLUTE CONTRAINDICATIONS</strong>
        <ul className="mb-0 mt-2 small">
          {(data.pharmacology_alerts || []).map((a, i) => (
            <li key={i} className="mb-1">{a}</li>
          ))}
        </ul>
      </div>

      {/* GRACILE vs Björnstad summary */}
      <SectionCard title="BCS1L Dual-Disease Spectrum" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-dark">
              <tr>
                <th>Feature</th>
                <th style={{ background: '#b71c1c', color: '#fff' }}>GRACILE (OMIM #603358)</th>
                <th style={{ background: '#1a237e', color: '#fff' }}>Björnstad (OMIM #262000)</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Alleles", "Severe (e.g. Ser78Gly homozygous, Ser78Phe)", "Milder (e.g. Arg144Gln, Gln302Arg)"],
                ["CIII residual", "<25%", "25–55%"],
                ["Onset", "Neonatal (days 1–30)", "Congenital–childhood"],
                ["Lactic acidosis", "Severe, refractory", "Mild or absent"],
                ["Aminoaciduria", "Present (Fanconi tubulopathy)", "Absent"],
                ["Cholestasis", "Present", "Absent"],
                ["Iron overload", "Hepatic siderosis", "Absent"],
                ["IUGR", "Severe", "Mild or absent"],
                ["SNHL", "Terminal seizures only", "Bilateral, moderate-severe"],
                ["Pili torti", "Absent", "Characteristic (90%)"],
                ["Cognition", "Globally impaired / pre-terminal", "Typically normal"],
                ["Survival", "~90% dead by 5 months", "Adult-compatible"],
                ["Treatment", "Supportive only; comfort care", "Cochlear implants; MRC cocktail"],
              ].map(([feat, gracile, bjorn], i) => (
                <tr key={i}>
                  <td className="fw-bold small">{feat}</td>
                  <td className="small">{gracile}</td>
                  <td className="small">{bjorn}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* DDx table */}
      <SectionCard title="Differential Diagnosis — CIII Assembly Factors & Related Genes">
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead className="table-dark">
              <tr>
                <th>Gene</th><th>Disease</th><th>Locus</th>
                <th>Key DDx Point</th><th>CIII residual</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ background: LIGHT }}>
                <td><strong>BCS1L (THIS)</strong></td>
                <td>GRACILE #603358 / Björnstad #262000</td>
                <td>2q35</td>
                <td>DUAL DISEASE — GRACILE (iron overload + aminoaciduria + cholestasis) vs Björnstad (SNHL + pili torti); BN-PAGE CIII precomplex; Finnish founder p.Ser78Gly</td>
                <td>&lt;5–55%</td>
              </tr>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td><strong>{d.gene}</strong></td>
                  <td><small>{d.disease}</small></td>
                  <td>{d.locus}</td>
                  <td><small>{d.ddx_point}</small></td>
                  <td>{d.residual_ciii}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Treatment */}
      <SectionCard title="Treatment Summary">
        {Object.entries(t).map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <strong className="text-capitalize">{k.replace(/_/g, ' ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <div>
      <SectionCard title="Gene Summary">
        <p className="small mb-1"><strong>Gene:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp; <strong>GRACILE:</strong> #{data.omim_gracile} &nbsp;|&nbsp; <strong>Björnstad:</strong> #{data.omim_bjornstad}</p>
        <p className="small mb-1"><strong>Chromosome:</strong> {data.chromosome} &nbsp;|&nbsp; <strong>Protein:</strong> {data.protein_size}</p>
        <p className="small mb-1"><strong>Domain:</strong> {data.tm_helices}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {data.inheritance}</p>
        <p className="small mb-0"><strong>Complex:</strong> {data.complex}</p>
      </SectionCard>

      <SectionCard title="Clinical Definitions">
        {(data.definitions || []).map((d, i) => (
          <div key={i} className="mb-2 small">
            <strong style={{ color: COLOR }}>{d.term}:</strong> {d.definition}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Standards & Guidelines">
        <ul className="mb-0 small">
          {(data.standards || []).map((s, i) => <li key={i} className="mb-1">{s}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="Key References">
        {(data.references || []).map((r, i) => (
          <div key={i} className="mb-3 small">
            <div><em>{r.citation}</em></div>
            <div className="text-muted mt-1">{r.significance}</div>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function BCS1LPage() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/bcs1l/overview`).then(r => r.json()),
      fetch(`${API}/api/bcs1l/breakdown`).then(r => r.json()),
      fetch(`${API}/api/bcs1l/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Tab bar */}
      <ul className="nav nav-tabs mb-4">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === t ? 'active fw-bold' : ''}`}
              style={tab === t ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(t)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {loading && <div className="text-center py-5"><div className="spinner-border" style={{ color: COLOR }} /></div>}
      {error   && <div className="alert alert-danger">{error}</div>}
      {!loading && !error && (
        <>
          {tab === 'Overview'            && <OverviewTab   data={overview}  />}
          {tab === 'Variants & Features' && <VariantsTab   data={breakdown} />}
          {tab === 'DDx & Treatment'     && <DDxTab        data={breakdown} />}
          {tab === 'Definitions'         && <DefinitionsTab data={defs}     />}
        </>
      )}
    </div>
  );
}
