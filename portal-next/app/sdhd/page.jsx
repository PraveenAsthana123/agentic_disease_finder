'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Features', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#4a148c';   // deep purple — maternal imprinting, highest penetrance PGL1
const LIGHT  = '#f3e5f5';
const COLOR2 = '#6a1b9a';   // secondary purple

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
          🧬 SDHD — Succinate Dehydrogenase Subunit D (Cytochrome b Small Subunit)
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>Structure:</strong> {data.tm_helices}
        </p>
        <p className="mb-1 small">
          <strong>Disease:</strong> Paraganglioma 1 (PGL1, OMIM #{data.omim_disease}) — AD, MATERNALLY IMPRINTED &nbsp;|&nbsp;
          <strong>Penetrance:</strong> {data.penetrance} &nbsp;|&nbsp;
          <strong>Malignancy:</strong> <span className="fw-bold" style={{ color: COLOR2 }}>{data.malignancy}</span>
        </p>
        <p className="mb-0 small fw-semibold" style={{ color: COLOR }}>
          🔴 SDHD: MATERNALLY IMPRINTED — only PATERNAL transmission causes PGL1.
          Female carriers' children NOT at risk. HIGHEST penetrance (~70-80%). HNPGL predominant.
          Bilateral PGL ~38%. CRITICAL DDx: SDHAF2 (11q13.1) — same chr11, same maternal imprinting.
        </p>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Patients (n)" value={s.n_patients} />
        <KPI label="HNPGL (any)" value={`${s.hnpgl_pct}%`} color={COLOR} />
        <KPI label="Carotid body" value={`${s.carotid_body_pct}%`} color={COLOR} />
        <KPI label="Jugulotympanic" value={`${s.jugulotympanic_pct}%`} />
        <KPI label="Vagal PGL" value={`${s.vagal_pct}%`} />
        <KPI label="Adrenal PCC" value={`${s.adrenal_pcc_pct}%`} color={COLOR2} />
        <KPI label="Bilateral" value={`${s.bilateral_pct}%`} color={COLOR} />
        <KPI label="Secretory" value={`${s.secretory_pct}%`} />
        <KPI label="Malignant" value={`${s.malignant_pct}%`} color={COLOR2} />
        <KPI label="GIST (CSS rare)" value={`${s.gist_pct}%`} />
        <KPI label="DOTATATE+" value={`${s.dotatate_positive_pct}%`} />
        <KPI label="Age mean" value={`${s.age_mean}yr`} />
      </div>

      {/* Clinical feature bars */}
      <SectionCard title="Clinical Features (Frequency %)">
        {feats.map(f => (
          <Bar key={f.feature} label={f.feature} value={f.freq_pct}
               color={f.freq_pct >= 40 ? COLOR : COLOR2} />
        ))}
      </SectionCard>

      {/* Key facts */}
      <SectionCard title="Key Clinical Facts" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {(data.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Top variants */}
      <SectionCard title="Top Variants in Cohort">
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead><tr><th>Variant</th><th>Count</th><th>Freq%</th></tr></thead>
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
                <th>ID</th><th>Age</th><th>PGL Type</th><th>Variant</th>
                <th>Bilateral</th><th>Secretory</th><th>Malignant</th>
                <th>GIST</th><th>DOTATATE+</th><th>Treatment</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.age_at_diagnosis_years}</td>
                  <td>{p.pgl_type}</td>
                  <td><code>{p.variant}</code></td>
                  <td>{p.bilateral ? '✓' : ''}</td>
                  <td>{p.secretory ? '✓' : ''}</td>
                  <td>{p.malignant ? <span style={{ color: COLOR2 }} className="fw-bold">✓</span> : ''}</td>
                  <td>{p.gist ? <span style={{ color: COLOR2 }}>✓</span> : ''}</td>
                  <td>{p.dotatate_positive ? '✓' : ''}</td>
                  <td><small>{p.treatment}</small></td>
                </tr>
              ))}
            </tbody>
          </table>
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
      <SectionCard title="SDHD Structural Features">
        {data.structural_features && Object.entries(data.structural_features).map(([k, v]) => (
          <p key={k} className="mb-1 small"><strong>{k.replace(/_/g, ' ')}:</strong> {String(v)}</p>
        ))}
      </SectionCard>

      {/* Variants table */}
      <SectionCard title="Pathogenic / Likely-Pathogenic Variants in SDHD">
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
                      v.severity === 'Severe (null)' || v.severity === 'Severe (catastrophic)' ? 'bg-danger' :
                      v.severity === 'Severe' ? 'bg-warning text-dark' :
                      v.severity === 'Moderate-Severe' ? 'bg-warning text-dark' :
                      v.severity === 'Intermediate' ? 'bg-info text-dark' : 'bg-secondary'
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
  const t = data.treatment_summary || {};
  const imp = data.imprinting_counselling || {};
  const ddx = data.key_ddx || [];

  return (
    <div>
      {/* CRITICAL maternal imprinting alert */}
      <div className="alert mb-4" style={{ background: '#fce4ec', borderLeft: `5px solid ${COLOR}` }}>
        <strong style={{ color: COLOR }}>🔴 SDHD MATERNALLY IMPRINTED — CRITICAL for Genetic Counselling</strong>
        <ul className="mb-0 mt-2 small">
          <li><strong>Father with SDHD mutation:</strong> 50% of all children (male + female) inherit active (paternal) SDHD → at risk for PGL1 → cascade testing recommended</li>
          <li><strong>Mother with SDHD mutation:</strong> children inherit silenced (maternal) SDHD allele → NOT at risk → surveillance of children NOT required</li>
          <li><strong>SDHD vs SDHC:</strong> SDHD female carrier = children NOT at risk; SDHC female carrier = children ARE at 50% risk (SDHC NOT imprinted)</li>
          <li><strong>SDHD vs SDHAF2 (11q13.1):</strong> BOTH chr11, BOTH maternally imprinted — cannot be distinguished by imprinting rule; WES mandatory</li>
        </ul>
      </div>

      {/* Imprinting counselling table */}
      <SectionCard title="Maternal Imprinting — Inheritance Counselling" borderColor={COLOR}>
        {Object.entries(imp).map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <strong className="text-capitalize" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}:</strong> {v}
          </div>
        ))}
      </SectionCard>

      {/* DDx table */}
      <SectionCard title="Differential Diagnosis — SDH Gene Comparison">
        <div className="table-responsive">
          <table className="table table-sm table-striped mb-0">
            <thead className="table-dark">
              <tr>
                <th>Gene</th><th>Locus</th><th>Key DDx Point</th>
                <th>Malignancy</th><th>Penetrance</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ background: LIGHT }}>
                <td><strong>SDHD (THIS)</strong></td>
                <td>11q23.1</td>
                <td>MATERNALLY IMPRINTED; HNPGL predominant; bilateral ~38%; HIGHEST penetrance ~70-80%</td>
                <td style={{ color: COLOR2 }} className="fw-bold">~3-5%</td>
                <td>~70-80% paternal</td>
              </tr>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td><strong>{d.gene}</strong></td>
                  <td>{d.locus}</td>
                  <td><small>{d.ddx_point}</small></td>
                  <td>{d.malignancy}</td>
                  <td>{d.penetrance}</td>
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

      {/* Pharmacology alerts */}
      <SectionCard title="Pharmacology Alerts" borderColor="#b71c1c">
        <ul className="mb-0 small">
          <li className="mb-1"><strong className="text-danger">Alpha-blockade BEFORE beta-blockade:</strong> Phenoxybenzamine must precede any beta-blocker — beta-first causes hypertensive crisis (unopposed alpha vasoconstriction).</li>
          <li className="mb-1"><strong>Surgery:</strong> Primary curative treatment for localised HNPGL; ENT and skull-base expertise mandatory for jugulotympanic PGL.</li>
          <li className="mb-1"><strong>177Lu-DOTATATE PRRT:</strong> SSTR2-positive progressive/metastatic PGL — SDHD ~75% DOTATATE+ → most patients eligible.</li>
          <li className="mb-1"><strong>Sunitinib:</strong> Anti-VEGFR/PDGFR — best systemic evidence for metastatic SDH-deficient PGL (including rare malignant SDHD).</li>
          <li className="mb-1"><strong>Belzutifan (HIF-2α):</strong> Emerging for unresectable/metastatic SDH-deficient PGL — same pseudo-hypoxia axis as VHL.</li>
          <li className="mb-1"><strong>No unique metabolic CI:</strong> SDHD PGL is a tumour suppressor gene — no specific metabolic contraindications (unlike SDHA Leigh, which has ABSOLUTE CI for KD, metformin, VPA).</li>
        </ul>
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
        <p className="small mb-1"><strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp; <strong>Disease:</strong> PGL1 #{data.omim_disease} / CSS #{data.omim_css}</p>
        <p className="small mb-1"><strong>Chromosome:</strong> {data.chromosome} &nbsp;|&nbsp; <strong>Protein:</strong> {data.protein_size}</p>
        <p className="small mb-1"><strong>TM Helices:</strong> {data.tm_helices}</p>
        <p className="small mb-1"><strong>Inheritance:</strong> {data.inheritance}</p>
        <p className="small mb-1"><strong>Penetrance:</strong> {data.penetrance}</p>
        <p className="small mb-1"><strong>Malignancy:</strong> {data.malignancy}</p>
        <p className="small mb-0"><strong>Imprinting:</strong> {data.imprinting}</p>
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
export default function SDHDPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/sdhd/overview`).then(r => r.json()),
      fetch(`${API}/api/sdhd/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sdhd/definitions`).then(r => r.json()),
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
      {error && <div className="alert alert-danger">{error}</div>}
      {!loading && !error && (
        <>
          {tab === 'Overview'            && <OverviewTab  data={overview}   />}
          {tab === 'Variants & Features' && <VariantsTab  data={breakdown}  />}
          {tab === 'DDx & Treatment'     && <DDxTab       data={breakdown}  />}
          {tab === 'Definitions'         && <DefinitionsTab data={defs}     />}
        </>
      )}
    </div>
  );
}
