'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Features', 'DDx & Treatment', 'Definitions'];
const COLOR   = '#b71c1c';   // deep red — high-malignancy gene
const LIGHT   = '#ffebee';
const COLOR2  = '#880e4f';   // secondary — pheochromocytoma/RCC

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

  return (
    <div>
      {/* Gene header */}
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <h5 className="fw-bold mb-1" style={{ color: COLOR }}>
          🧬 SDHB — Succinate Dehydrogenase Subunit B (Iron-Sulfur Subunit)
        </h5>
        <p className="mb-1 small">
          <strong>OMIM Gene:</strong> *{data.omim_gene} &nbsp;|&nbsp;
          <strong>Chr:</strong> {data.chromosome} &nbsp;|&nbsp;
          <strong>Protein:</strong> {data.protein_size} &nbsp;|&nbsp;
          <strong>FeS Clusters:</strong> {data.fes_clusters}
        </p>
        <p className="mb-1 small">
          <strong>Disease:</strong> Paraganglioma 4 (PGL4, OMIM #{data.omim_disease}) — AD, NOT imprinted &nbsp;|&nbsp;
          <strong>Penetrance:</strong> {data.penetrance} &nbsp;|&nbsp;
          <strong>Malignancy:</strong> <span className="text-danger fw-bold">20–50% — HIGHEST of all SDH genes</span>
        </p>
        <p className="mb-0 small text-danger fw-semibold">
          ⚠️ SDHB = THE malignancy gene. Extra-adrenal PGL predominant. RCC in 15%. IHC: SDHB null ONLY (SDHA proficient). NOT maternally imprinted.
        </p>
      </div>

      {/* KPIs */}
      <div className="row mb-3">
        <KPI label="Patients (n)" value={s.n_patients} />
        <KPI label="Malignancy" value={`${s.malignant_pct}%`} color="#c62828" />
        <KPI label="Extra-adrenal PGL" value={`${s.extra_adrenal_pgl_pct}%`} color={COLOR} />
        <KPI label="Head-neck PGL" value={`${s.head_neck_pgl_pct}%`} />
        <KPI label="Adrenal PCC" value={`${s.adrenal_pcc_pct}%`} color={COLOR2} />
        <KPI label="RCC" value={`${s.rcc_pct}%`} color={COLOR2} />
        <KPI label="Bilateral" value={`${s.bilateral_pct}%`} />
        <KPI label="Secretory" value={`${s.secretory_pct}%`} />
        <KPI label="DOTATATE+" value={`${s.dotatate_positive_pct}%`} />
        <KPI label="Variants (P/LP)" value={s.n_unique_variants} />
      </div>

      {/* Clinical feature bars */}
      <SectionCard title="Clinical Features (Frequency %)">
        {(data.cohort_summary || []).map(f => (
          <Bar key={f.feature} label={f.feature} value={f.freq_pct} color={f.freq_pct >= 40 ? COLOR : COLOR2} />
        ))}
      </SectionCard>

      {/* Key facts */}
      <SectionCard title="Key Clinical Facts" borderColor={COLOR2}>
        <ul className="mb-0 small">
          {(data.key_facts || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Variant summary */}
      <SectionCard title="Pathogenic Variants (8 representative P/LP)">
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0 small">
            <thead><tr>
              <th>HGVS_p</th><th>HGVS_c</th><th>Domain</th><th>Severity %</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {(data.variant_summary || []).map(v => (
                <tr key={v.hgvs_p}>
                  <td><code>{v.hgvs_p}</code></td>
                  <td><code>{v.hgvs_c}</code></td>
                  <td>{v.domain}</td>
                  <td><span className="badge" style={{ background: v.severity_pct >= 85 ? '#c62828' : '#e65100' }}>{v.severity_pct}%</span></td>
                  <td>{v.notes}</td>
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
      <SectionCard title="Pathogenic Variant Breakdown">
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-hover mb-0 small">
            <thead className="table-dark"><tr>
              <th>HGVS_p</th><th>HGVS_c</th><th>Domain / FeS</th><th>Severity %</th><th>Mechanism</th><th>Notes</th>
            </tr></thead>
            <tbody>
              {(data.variant_breakdown || []).map(v => (
                <tr key={v.hgvs_p}>
                  <td><code>{v.hgvs_p}</code></td>
                  <td><code>{v.hgvs_c}</code></td>
                  <td>{v.domain}</td>
                  <td><span className="badge bg-danger">{v.severity_pct}%</span></td>
                  <td className="small">{v.mechanism}</td>
                  <td className="small">{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Clinical Features (Frequency %)">
        {(data.clinical_features || []).map(f => (
          <Bar key={f.feature} label={f.feature} value={f.freq_pct} color={f.freq_pct >= 40 ? COLOR : COLOR2} />
        ))}
      </SectionCard>

      <SectionCard title="Patient Cohort (40 patients, seed-707)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0 small">
            <thead><tr>
              <th>ID</th><th>Age</th><th>Sex</th><th>Variant</th><th>Tumor</th><th>Malignant</th><th>Bilateral</th><th>RCC</th><th>Secretory</th><th>IHC</th>
            </tr></thead>
            <tbody>
              {(data.patients || []).map(p => (
                <tr key={p.patient_id} className={p.malignant ? 'table-danger' : ''}>
                  <td>{p.patient_id}</td>
                  <td>{p.age_at_diagnosis_years}</td>
                  <td>{p.sex}</td>
                  <td><code>{p.variant_hgvs_p}</code></td>
                  <td>{p.tumor_location}</td>
                  <td>{p.malignant ? <span className="badge bg-danger">Yes</span> : <span className="badge bg-success">No</span>}</td>
                  <td>{p.bilateral ? 'Yes' : 'No'}</td>
                  <td>{p.rcc ? <span className="badge bg-warning text-dark">Yes</span> : 'No'}</td>
                  <td>{p.secretory ? 'Yes' : 'No'}</td>
                  <td><span className="badge bg-danger">SDHB null</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const tx = data.treatment || {};

  return (
    <div>
      <div className="alert alert-danger mb-4">
        <strong>⚠️ Critical DDx Rule:</strong> SDHB = THE highest malignancy SDH gene (20–50%).
        IHC shows SDHB null + SDHA proficient. NOT maternally imprinted. Extra-adrenal PGL predominant.
        Sunitinib is the best-evidenced systemic therapy for metastatic disease.
      </div>

      <SectionCard title="Differential Diagnosis Table">
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-hover mb-0 small">
            <thead className="table-dark"><tr>
              <th>Gene / Disease</th><th>Locus</th><th>Key DDx vs SDHB</th><th>Malignancy</th><th>Imprinting</th>
            </tr></thead>
            <tbody>
              {(data.ddx_table || []).map(d => (
                <tr key={d.gene}>
                  <td><strong>{d.gene}</strong><br/><small className="text-muted">{d.disease}</small></td>
                  <td><code>{d.locus}</code></td>
                  <td className="small">{d.key_ddx}</td>
                  <td><span className="badge bg-danger">{d.malignancy}</span></td>
                  <td>{d.imprinting}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Pre-operative Sequence (CRITICAL)" borderColor="#c62828">
        <div className="alert alert-warning mb-3">
          <strong>Alpha-blockade BEFORE beta-blockade:</strong> Phenoxybenzamine (start ≥7–14 days pre-op)
          MUST precede any beta-blocker. Reversing order → unopposed alpha vasoconstriction → hypertensive crisis.
        </div>
      </SectionCard>

      <SectionCard title="Recommended Treatments">
        <div className="table-responsive">
          <table className="table table-sm table-bordered table-hover mb-0 small">
            <thead className="table-dark"><tr>
              <th>Drug / Intervention</th><th>Dose</th><th>Level</th><th>Rationale</th>
            </tr></thead>
            <tbody>
              {(tx.recommended || []).map(t => (
                <tr key={t.drug}>
                  <td><strong>{t.drug}</strong></td>
                  <td><code>{t.dose}</code></td>
                  <td><span className="badge bg-primary">{t.level}</span></td>
                  <td className="small">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {tx.surveillance && (
        <SectionCard title="Surveillance Protocol" borderColor={COLOR2}>
          <ul className="mb-0 small">
            {tx.surveillance.map((s, i) => <li key={i} className="mb-1">{s}</li>)}
          </ul>
        </SectionCard>
      )}
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  const g = data.gene || {};
  const d = data.disease || {};
  const ihc = data.ihc_interpretation || {};
  const refs = data.key_references || [];

  return (
    <div>
      <SectionCard title="Gene Definition">
        <dl className="row small mb-0">
          <dt className="col-sm-4">Full name</dt><dd className="col-sm-8">{g.full_name}</dd>
          <dt className="col-sm-4">OMIM Gene</dt><dd className="col-sm-8">*{g.omim_gene}</dd>
          <dt className="col-sm-4">Chromosome</dt><dd className="col-sm-8">{g.chromosome}</dd>
          <dt className="col-sm-4">Size</dt><dd className="col-sm-8">{g.size_aa} aa, ~{g.size_kda} kDa</dd>
          <dt className="col-sm-4">FeS Clusters</dt><dd className="col-sm-8">{g.fes_clusters}</dd>
          <dt className="col-sm-4">Function</dt><dd className="col-sm-8">{g.function}</dd>
          <dt className="col-sm-4">Domains</dt><dd className="col-sm-8">
            <ul className="mb-0 ps-3">{(g.domains || []).map((dom, i) => <li key={i}>{dom}</li>)}</ul>
          </dd>
        </dl>
      </SectionCard>

      <SectionCard title="Disease: Paraganglioma 4 (PGL4)" borderColor="#c62828">
        <dl className="row small mb-0">
          <dt className="col-sm-4">OMIM</dt><dd className="col-sm-8">#{d.omim}</dd>
          <dt className="col-sm-4">Inheritance</dt><dd className="col-sm-8">{d.inheritance}</dd>
          <dt className="col-sm-4">Penetrance</dt><dd className="col-sm-8">{d.penetrance}</dd>
          <dt className="col-sm-4">Tumor sites</dt><dd className="col-sm-8">{d.sites}</dd>
          <dt className="col-sm-4">Malignancy</dt><dd className="col-sm-8 text-danger fw-bold">{d.malignancy}</dd>
          <dt className="col-sm-4">RCC</dt><dd className="col-sm-8">{d.rcc}</dd>
          <dt className="col-sm-4">IHC pattern</dt><dd className="col-sm-8">{d.ihc_pattern}</dd>
          <dt className="col-sm-4">Surveillance</dt><dd className="col-sm-8">{d.surveillance}</dd>
        </dl>
      </SectionCard>

      <SectionCard title="IHC Interpretation" borderColor={COLOR2}>
        <dl className="row small mb-0">
          <dt className="col-sm-4">SDHB loss only</dt><dd className="col-sm-8">{ihc.sdhb_loss_only}</dd>
          <dt className="col-sm-4">SDHA+SDHB loss</dt><dd className="col-sm-8">{ihc.sdha_sdhb_loss}</dd>
          <dt className="col-sm-4">Rationale</dt><dd className="col-sm-8">{ihc.rationale}</dd>
          <dt className="col-sm-4">Clinical use</dt><dd className="col-sm-8">{ihc.clinical_use}</dd>
        </dl>
      </SectionCard>

      <SectionCard title="Imprinting Comparison">
        {Object.entries(data.imprinting_comparison || {}).map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-bold text-uppercase">{k.replace(/_/g, ' ')}: </span>{v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Malignancy Comparison">
        {Object.entries(data.malignancy_comparison || {}).map(([k, v]) => (
          <div key={k} className="mb-2 small">
            <span className="fw-bold text-uppercase">{k.replace(/_/g, ' ')}: </span>{v}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR2}>
        <ol className="mb-0 small">
          {refs.map((r, i) => (
            <li key={i} className="mb-2">
              <span className="fw-semibold">{r.citation}</span>
              {r.relevance && <span className="text-muted d-block ms-2">→ {r.relevance}</span>}
            </li>
          ))}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function SdhbPage() {
  const [activeTab, setActiveTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sdhb/overview`).then(r => r.json()),
      fetch(`${API}/api/sdhb/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sdhb/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefs(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error loading SDHB dashboard: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 gap-3">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 SDHB — Paraganglioma 4 (PGL4)</h4>
          <small className="text-muted">
            Iron-Sulfur Subunit · Complex II · 1p36.13 · OMIM *185470 / #115310 ·
            <span className="text-danger fw-bold ms-1">Highest Malignancy SDH Gene (20–50%)</span>
          </small>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${i === activeTab ? 'active fw-bold' : ''}`}
              style={i === activeTab ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setActiveTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {activeTab === 0 && <OverviewTab data={overview} />}
      {activeTab === 1 && <VariantsTab data={{ ...breakdown, patients: overview?.patients, variant_breakdown: breakdown?.variant_breakdown, clinical_features: breakdown?.clinical_features }} />}
      {activeTab === 2 && <DDxTab data={breakdown} />}
      {activeTab === 3 && <DefsTab data={defs} />}
    </div>
  );
}
