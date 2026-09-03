'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#4527a0';   // deep indigo — tRNA-Ala / L-strand / EIGHTH tRNA
const LIGHT  = '#ede7f6';
const COLOR2 = '#6a1b9a';   // medium purple — CPEO / myopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / L-strand pitfall alert
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#0277bd';   // dark blue — AARS2 DDx / nuclear gene

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
  const s        = data.cohort_statistics || {};
  const feats    = data.cohort_summary_features || [];
  const pheno_dist = data.phenotype_distribution || [];
  const hmap     = data.heteroplasmy_clinical_map || [];
  const mol_feats = data.key_molecular_features || [];
  const alerts   = data.clinical_alerts || [];
  const gf       = data.gene_facts || {};
  const bf       = data.biochemical_fingerprint || {};

  return (
    <>
      {/* Gene header */}
      <SectionCard borderColor={COLOR}>
        <div className="d-flex align-items-center gap-3 mb-2">
          <span style={{ fontSize: 40 }}>🧬</span>
          <div>
            <h4 className="fw-bold mb-0" style={{ color: COLOR }}>{data.title}</h4>
            <div className="text-muted small">{data.subtitle}</div>
            <span className="badge" style={{ backgroundColor: COLOR, fontSize: '0.75rem' }}>{data.omim}</span>
          </div>
        </div>
      </SectionCard>

      {/* L-strand NGS pitfall alert — prominent */}
      <div className="alert border-0 mb-4" style={{ backgroundColor: '#fce4ec', borderLeft: `6px solid ${COLOR3}` }}>
        <strong style={{ color: COLOR3 }}>🔴 L-STRAND NGS PITFALL — MT-TA + THREE CONSECUTIVE L-strand tRNAs — MANDATORY QC SWITCH:</strong>
        <br />
        MT-TA (rCRS 5587–5655) is <strong>L-strand encoded</strong>. Standard NGS pipelines using H-strand (rCRS) calls
        will <strong>MISS or MIS-CALL</strong> MT-TA variants. Reverse-complement QC is mandatory.
        The same pitfall extends to <strong>MT-TN, MT-TC, MT-TY</strong> (rCRS 5657–5891) — all L-strand.
        Labs must apply L-strand QC from rCRS 5580 (after MT-TW) through rCRS 5891 (end of MT-TY).
      </div>

      {/* AARS2 DDx note */}
      <div className="alert border-0 mb-4" style={{ backgroundColor: '#e3f2fd', borderLeft: `6px solid ${COLOR5}` }}>
        <strong style={{ color: COLOR5 }}>🔵 AARS2 NUCLEAR DDx — OVARIAN INSUFFICIENCY distinguishes AARS2 from MT-TA:</strong>
        <br />
        AARS2 biallelic (AR, WES-detectable): females get <strong>ovario-leukodystrophy</strong> (premature ovarian
        insufficiency + white-matter MRI) — NOT adult CPEO. Males get lethal infant cardiomyopathy.
        MT-TA (maternal, heteroplasmic) does NOT cause ovarian insufficiency and does NOT cause leukodystrophy.
      </div>

      {/* KPIs */}
      <SectionCard title="40-Patient Cohort — Key Statistics (seed-815)" borderColor={COLOR2}>
        <div className="row g-2">
          <KPI label="Patients" value={s.n_patients} />
          <KPI label="Avg Heteroplasmy" value={`${s.avg_heteroplasmy_blood_pct}%`} />
          <KPI label="CI Activity" value={`${s.avg_ci_activity_pct_normal}%`} color={COLOR3} />
          <KPI label="CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR3} />
          <KPI label="CII Activity" value={`${s.avg_cii_activity_pct_normal}%`} color={COLOR4} />
          <KPI label="Avg Onset (yr)" value={s.avg_age_onset_yr} color={COLOR2} />
        </div>
        <div className="row g-2 mt-1">
          <KPI label="CPEO" value={`${s.pct_cpeo}%`} />
          <KPI label="Myopathy" value={`${s.pct_myopathy}%`} />
          <KPI label="Exercise Intol" value={`${s.pct_exercise_intolerance}%`} color={COLOR2} />
          <KPI label="Cardiomyopathy" value={`${s.pct_cardiomyopathy}%`} color={COLOR3} />
          <KPI label="Compound TW+TA" value={`${s.pct_compound_tw_loss}%`} color={COLOR3} />
          <KPI label="Leigh-like MRI" value={`${s.pct_leigh_like_mri}%`} color={COLOR5} />
        </div>
      </SectionCard>

      {/* Phenotype features */}
      <SectionCard title="Phenotype Summary — 40-Patient Cohort" borderColor={COLOR}>
        {feats.map((f, i) => (
          <div key={i} className="d-flex justify-content-between border-bottom py-1 small">
            <span>{f.feature}</span>
            <span>
              <strong style={{ color: COLOR }}>{f.value}</strong>
              <span className="text-muted ms-2">— {f.note}</span>
            </span>
          </div>
        ))}
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="Variant Distribution in 40-Patient Cohort" borderColor={COLOR2}>
        {pheno_dist.map((p, i) => (
          <div key={i} className="mb-3">
            <Bar label={`${p.variant} — ${p.phenotype} (${p.position})`} value={p.pct} color={COLOR2} />
          </div>
        ))}
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint — OXPHOS Enzymology" borderColor={COLOR4}>
        <p className="fw-bold mb-2" style={{ color: COLOR4 }}>{bf.summary}</p>
        <div className="row g-3">
          {['complex_i','complex_ii','complex_iv'].map(k => (
            <div key={k} className="col-md-4">
              <div className="border rounded p-2 small" style={{ backgroundColor: k === 'complex_ii' ? '#e8f5e9' : '#fce4ec' }}>
                <div className="fw-bold">{k.replace('complex_','Complex ').toUpperCase()}</div>
                <div>{bf[k]}</div>
              </div>
            </div>
          ))}
        </div>
        <p className="small text-muted mt-3">{bf.mechanism}</p>
      </SectionCard>

      {/* Heteroplasmy map */}
      <SectionCard title="Heteroplasmy → Clinical Threshold Map" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead><tr><th>Heteroplasmy</th><th>Expected Phenotype</th></tr></thead>
            <tbody>
              {hmap.map((h, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR }}>{h.threshold_pct}%</td>
                  <td>{h.expected_phenotype}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Gene facts */}
      <SectionCard title="Gene Facts — MT-TA" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <tbody>
              {Object.entries(gf).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-capitalize" style={{ color: COLOR, whiteSpace: 'nowrap' }}>{k.replace(/_/g,' ')}</td>
                  <td>{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Key molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR5}>
        <ul className="mb-0 small">
          {mol_feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts" borderColor={COLOR3}>
        {alerts.map((a, i) => (
          <div key={i} className="alert alert-danger py-2 mb-2 small">
            <strong>⚠ {a.alert}:</strong> {a.detail}
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Variants & Cohort ────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vb   = data.variant_breakdown || [];
  const mgmt = data.management_by_variant || [];
  const ci   = data.absolute_contraindications || [];
  const si   = data.safe_interventions || [];

  return (
    <>
      <SectionCard title="Variant Breakdown — 40-Patient Cohort (seed-815)" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-light">
              <tr>
                <th>Variant</th><th>N</th><th>% Cohort</th>
                <th>Avg Hetero</th><th>Avg CI%</th><th>Avg CIV%</th>
                <th>CPEO%</th><th>Myopathy%</th><th>Cardio%</th><th>Compound TW+TA%</th>
              </tr>
            </thead>
            <tbody>
              {vb.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR }}>{v.variant}</td>
                  <td>{v.n}</td>
                  <td>{v.pct_of_cohort}%</td>
                  <td>{v.avg_heteroplasmy}%</td>
                  <td className="text-danger">{v.avg_ci_pct_normal}%</td>
                  <td className="text-danger">{v.avg_civ_pct_normal}%</td>
                  <td>{v.pct_cpeo}%</td>
                  <td>{v.pct_myopathy}%</td>
                  <td>{v.pct_cardiomyopathy}%</td>
                  <td>{v.pct_compound_tw_loss}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Management by Variant" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-light">
              <tr><th>Variant</th><th>CPEO Risk</th><th>Cardio Risk</th><th>Key Actions</th></tr>
            </thead>
            <tbody>
              {mgmt.map((m, i) => (
                <tr key={i}>
                  <td className="fw-bold" style={{ color: COLOR }}>{m.variant}</td>
                  <td>{m.cpeo_risk}</td>
                  <td>{m.cardio_risk}</td>
                  <td className="small">{m.key_action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Absolute Contraindications (ALL mt disease)" borderColor={COLOR3}>
        {ci.map((c, i) => (
          <div key={i} className="d-flex border-bottom py-1 small gap-2">
            <span className="fw-bold text-danger" style={{ minWidth: 160 }}>{c.drug}</span>
            <span className="text-muted">{c.reason}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Safe Interventions" borderColor={COLOR4}>
        {si.map((s, i) => (
          <div key={i} className="d-flex border-bottom py-1 small gap-2">
            <span className="fw-bold" style={{ color: COLOR4, minWidth: 180 }}>{s.intervention}</span>
            <span className="text-muted">{s.evidence}</span>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: DDx & Management ─────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ddx = data.ddx_table || [];

  return (
    <>
      <SectionCard title="Differential Diagnosis Table" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-light">
              <tr><th>Entity</th><th>Inheritance</th><th>Phenotype</th><th>Biochemistry</th><th>NGS</th><th>Distinctive</th></tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} className={d.entity.startsWith('MT-TA') ? 'table-info' : ''}>
                  <td className="fw-bold">{d.entity}</td>
                  <td>{d.inheritance}</td>
                  <td>{d.phenotype}</td>
                  <td>{d.biochemistry}</td>
                  <td>{d.ngs}</td>
                  <td className="small">{d.distinctive}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${COLOR}` }}>
        <div className="card-body">
          <h6 className="fw-bold mb-3" style={{ color: COLOR }}>MT-TA L-strand NGS Pitfall — Clinical Decision Tree</h6>
          <div className="p-3 rounded" style={{ backgroundColor: LIGHT, fontSize: '0.85rem' }}>
            <p className="mb-2"><strong>CPEO + CI+CIV deficiency (CII NORMAL) — suspected mt-tRNA disease?</strong></p>
            <ul>
              <li>→ <strong>Was mtDNA panel run with L-strand QC?</strong> If H-strand only: MT-TA variants may be MISSED — retest with L-strand reverse-complement QC for rCRS 5587–5655</li>
              <li>→ <strong>AARS2 workup if female + ovarian insufficiency:</strong> Premature ovarian insufficiency (POI) + white-matter MRI = AARS2 (AR, WES) — NOT MT-TA</li>
              <li>→ <strong>LargeDel check:</strong> Deletions spanning rCRS ~5480–5700 may cause compound MT-TW + MT-TA loss — both tRNAs checked simultaneously</li>
              <li>→ <strong>Stroke-like episodes?</strong>: MT-TL1 (MELAS) — NOT MT-TA</li>
              <li>→ <strong>Myoclonic epilepsy?</strong>: MT-TK (MERRF) — NOT MT-TA</li>
              <li>→ <strong>Isolated HCM?</strong>: MT-TI m.4300AG — NOT MT-TA</li>
              <li>→ <strong>Leigh-like MRI?</strong>: Exclude BTBGD (SLC19A3) — biotin+thiamine trial FIRST</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${COLOR3}` }}>
        <div className="card-body">
          <h6 className="fw-bold mb-3" style={{ color: COLOR3 }}>Perioperative Protocol (ALL MT-TA patients)</h6>
          <div className="row g-3 small">
            {[
              { label: 'GIR 6–8 mg/kg/min', desc: 'MANDATORY — never fast; glucose infusion during any perioperative period' },
              { label: 'Avoid Propofol', desc: 'PRIS risk — use sevoflurane; total IV anaesthesia only if no alternative' },
              { label: 'Avoid Metformin', desc: 'Complex I inhibition → fatal lactic acidosis — stop before admission' },
              { label: 'Lactate monitoring', desc: 'Pre-/intra-/post-op; target <2 mmol/L; escalate if rising' },
              { label: 'Thiamine + Biotin', desc: 'Empiric perioperative — BTBGD exclusion + mt-energy support' },
              { label: 'ICU for >50% heteroplasmy', desc: 'High-heteroplasmy MT-TA: ICU perioperative; anaesthetic + metabolic co-management' },
            ].map((item, i) => (
              <div key={i} className="col-md-6">
                <div className="border rounded p-2" style={{ backgroundColor: '#fff3e0' }}>
                  <div className="fw-bold text-danger">{item.label}</div>
                  <div className="text-muted">{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const sections = [
    { key: 'gene_definitions',       title: 'Gene & Molecular',      color: COLOR  },
    { key: 'biochemical_definitions', title: 'Biochemical',           color: COLOR4 },
    { key: 'clinical_definitions',   title: 'Clinical Syndromes',    color: COLOR2 },
    { key: 'ngs_definitions',        title: 'NGS & Variant Analysis', color: COLOR5 },
    { key: 'drug_definitions',       title: 'Drug & Treatment',       color: COLOR3 },
  ];

  return (
    <>
      {sections.map(({ key, title, color }) => (
        <SectionCard key={key} title={title} borderColor={color}>
          {(data[key] || []).map((d, i) => (
            <div key={i} className="border-bottom py-2 small">
              <span className="fw-bold me-2" style={{ color }}>{d.term}:</span>
              <span className="text-muted">{d.definition}</span>
            </div>
          ))}
        </SectionCard>
      ))}

      <SectionCard title="References" borderColor={COLOR2}>
        {(data.references || []).map((r, i) => (
          <div key={i} className="border-bottom py-1 small">
            <span className="fw-bold me-2" style={{ color: COLOR2 }}>[{r.ref}]</span>
            <span className="text-muted">{r.citation}</span>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function MTTAPage() {
  const [tab, setTab] = useState(0);
  const [overview,  setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,      setDefs]      = useState(null);
  const [error,     setError]     = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mtta/overview`).then(r => r.json()),
      fetch(`${API}/api/mtta/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mtta/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefs(df);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Error: {error}</div>
    </div>
  );

  const tabContent = [
    <OverviewTab  key="ov" data={overview}  />,
    <VariantsTab  key="vr" data={breakdown} />,
    <DDxTab       key="dx" data={breakdown} />,
    <DefinitionsTab key="df" data={defs}  />,
  ];

  return (
    <div className="container-fluid py-3">
      {/* Page header */}
      <div className="d-flex align-items-center gap-2 mb-3">
        <span style={{ fontSize: 28 }}>🧬</span>
        <div>
          <h5 className="mb-0 fw-bold" style={{ color: COLOR }}>
            MT-TA — tRNA-Ala — L-strand NGS Pitfall — EIGHTH tRNA — L-strand rCRS 5587–5655
          </h5>
          <small className="text-muted">
            Combined CI+CIV Deficiency · CPEO · Myopathy · L-strand cluster first (MT-TA/TN/TC/TY) · AARS2 DDx · OMIM *590000 · seed-815
          </small>
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { borderBottomColor: COLOR, color: COLOR, fontWeight: 600 } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tabContent[tab]}
    </div>
  );
}
