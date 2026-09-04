'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1b5e20';   // deep green — CoQ10/ubiquinone biosynthesis
const LIGHT  = '#e8f5e9';
const COLOR2 = '#2e7d32';   // PDSS isoprene tail synthesis
const COLOR3 = '#388e3c';   // ring attachment / COQ2
const COLOR4 = '#43a047';   // ring modification enzymes
const COLOR5 = '#1565c0';   // CoQ kinases (ADCK family)
const COLOR6 = '#b71c1c';   // drug CIs / absolute CI
const COLOR7 = '#e65100';   // SRNS / renal phenotype
const COLOR8 = '#6a1b9a';   // ataxia / ARCA2 phenotype
const COLOR9 = '#880e4f';   // Leigh / neonatal severe

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
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

function Badge({ text, color = COLOR }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const agg   = data.aggregate_clinical || {};
  const drug  = data.drug_contraindications || {};
  const wes   = data.wes_utility || {};
  const pheno = data.hallmark_phenotypes || {};
  const genes = data.gene_list || {};
  const rules = data.key_rules || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="CoQ10-Biosynthesis-Atlas — Complete 10-Gene Primary Coenzyme Q10 (Ubiquinone) Deficiency Reference">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">CoQ10 function: </span>{data.function}</div>
            <div><span className="fw-semibold">Biosynthesis pathway: </span>{data.pathway}</div>
            <div><span className="fw-semibold">Total Genes: </span>{data.n_genes} nuclear ({data.n_decaprenyl_pp} PDSS tail + {data.n_ring_attachment} ring attachment + {data.n_ring_modification} ring modification + {data.n_coq_kinase} COQ kinases)</div>
            <div><span className="fw-semibold">Cohort: </span>{data.cohort_formula}</div>
            <div className="alert alert-danger py-1 px-2 mt-2 small">
              <strong>CRITICAL:</strong> {data.plasma_unreliable}
            </div>
          </div>
          <div className="col-12 col-md-6">
            <div className="fw-semibold mb-1" style={{ color: COLOR2 }}>2 Decaprenyl-PP Synthesis (Tail):</div>
            <div className="mb-2">{(genes.decaprenyl_pp_2 || []).join(' · ')}</div>
            <div className="fw-semibold mb-1" style={{ color: COLOR3 }}>1 Ring Attachment:</div>
            <div className="mb-2">{(genes.ring_attachment_1 || []).join(' · ')}</div>
            <div className="fw-semibold mb-1" style={{ color: COLOR4 }}>5 Ring Modification (COQ Complex):</div>
            <div className="mb-2 small">{(genes.ring_modification_4 || []).join(' · ')}</div>
            <div className="fw-semibold mb-1" style={{ color: COLOR5 }}>2 COQ Kinases (ADCK):</div>
            <div className="small">{(genes.coq_kinase_2 || []).join(' · ')}</div>
          </div>
        </div>
        <div className="row g-2 mt-2">
          <KPI label="PDSS Tail Genes"       value={data.n_decaprenyl_pp}   color={COLOR2} />
          <KPI label="Ring Attachment"        value={data.n_ring_attachment}  color={COLOR3} />
          <KPI label="Ring Modification"      value={data.n_ring_modification} color={COLOR4} />
          <KPI label="COQ Kinases"            value={data.n_coq_kinase}       color={COLOR5} />
          <KPI label="Total Genes"            value={data.n_genes}            color={COLOR}  />
          <KPI label="Total Patients"         value={data.n_patients}         color={COLOR}  />
        </div>
        <div className="alert alert-warning py-1 px-2 mt-2 small">
          <strong>CII ALWAYS NORMAL:</strong> {data.cii_always_normal}
        </div>
        <div className="alert alert-info py-1 px-2 mt-1 small">
          <strong>BTBGD Exclusion:</strong> {data.btbgd_exclusion}
        </div>
      </SectionCard>

      {/* Aggregate clinical */}
      <SectionCard title="📊 Aggregate Clinical Phenotypes — 400 Patients (10 × 40)" borderColor={COLOR4}>
        <div className="row g-2 small">
          <KPI label="Avg CoQ10 Muscle %" value={`${agg.avg_coq10_muscle_pct}%`} color={COLOR6} />
          <KPI label="SRNS"               value={`${agg.srns_pct}%`}             color={COLOR7} />
          <KPI label="SNHL"               value={`${agg.snhl_pct}%`}             color={COLOR}  />
          <KPI label="Ataxia"             value={`${agg.ataxia_pct}%`}           color={COLOR8} />
          <KPI label="Leigh / Enceph"     value={`${agg.leigh_pct}%`}            color={COLOR9} />
          <KPI label="Myopathy"           value={`${agg.myopathy_pct}%`}         color={COLOR4} />
          <KPI label="Lactic Acidosis"    value={`${agg.lactic_acidosis_pct}%`}  color={COLOR6} />
          <KPI label="HCM"                value={`${agg.hcm_pct}%`}             color={COLOR9} />
        </div>
      </SectionCard>

      {/* Key clinical rules */}
      <SectionCard title="🔑 Key Clinical Rules" borderColor={COLOR6}>
        <div className="row g-3 small">
          {Object.entries(rules).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR6}` }}>
                <span className="fw-semibold">{k.replace(/_/g, ' ').toUpperCase()}: </span>{v}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Hallmark phenotypes */}
      <SectionCard title="🧬 Hallmark Phenotypes by Phenotype Cluster" borderColor={COLOR8}>
        <div className="table-responsive small">
          <table className="table table-bordered table-sm">
            <thead><tr style={{ background: COLOR8, color: 'white' }}>
              <th>Phenotype Pattern</th><th>Genes</th>
            </tr></thead>
            <tbody>
              {Object.entries(pheno).map(([k, v]) => (
                <tr key={k}>
                  <td className="fw-semibold text-nowrap">{k.replace(/_/g, ' ')}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Drug contraindications */}
      <SectionCard title="💊 Drug Contraindications — CoQ10 Biosynthesis Deficiency" borderColor={COLOR6}>
        <div className="row g-3 small">
          {Object.values(drug).map((d, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="card h-100" style={{ borderLeft: `4px solid ${d.risk?.startsWith('ABSOLUTE') ? COLOR6 : d.risk?.startsWith('Yes') ? COLOR : '#ff8f00'}` }}>
                <div className="card-body py-2 px-3">
                  <div className="fw-bold mb-1">{d.drug}</div>
                  <Badge text={d.risk} color={d.risk?.startsWith('ABSOLUTE') ? COLOR6 : d.risk?.startsWith('Yes') ? COLOR : '#ff8f00'} />
                  <div className="text-muted mt-1">{d.mechanism}</div>
                  {d.action && <div className="mt-1"><span className="fw-semibold">Action: </span>{d.action}</div>}
                  {d.applies_to && <div className="mt-1 text-muted fst-italic">Applies to: {d.applies_to}</div>}
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* WES utility */}
      <SectionCard title="🧪 WES Utility + Diagnostic Notes" borderColor={COLOR5}>
        <div className="row g-3 small">
          {Object.entries(wes).map(([k, v]) => (
            <div key={k} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ background: '#e8eaf6', borderLeft: `3px solid ${COLOR5}` }}>
                <span className="fw-semibold">{k.replace(/_/g, ' ')}: </span>{v}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  const [filter, setFilter] = useState('all');
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];

  const classColors = {
    decaprenyl_pp:    COLOR2,
    ring_attachment:  COLOR3,
    ring_modification: COLOR4,
    coq_kinase:       COLOR5,
  };

  const filtered = filter === 'all' ? genes : genes.filter(g => g.gene_class === filter);

  return (
    <>
      <SectionCard title="Filter by Pathway Class">
        <div className="d-flex flex-wrap gap-2 mb-3">
          {['all', 'decaprenyl_pp', 'ring_attachment', 'ring_modification', 'coq_kinase'].map(f => (
            <button key={f} className="btn btn-sm"
              style={{ background: filter === f ? (classColors[f] || COLOR) : '#e0e0e0', color: filter === f ? 'white' : '#333', border: 'none' }}
              onClick={() => setFilter(f)}>
              {f === 'all' ? 'All Genes' : f.replace(/_/g, ' ').toUpperCase()}
            </button>
          ))}
        </div>
      </SectionCard>

      <div className="table-responsive small">
        <table className="table table-bordered table-hover table-sm align-middle">
          <thead><tr style={{ background: COLOR, color: 'white' }}>
            <th>Gene</th><th>Class</th><th>Pathway Step</th><th>Chr</th><th>aa</th>
            <th>SRNS%</th><th>SNHL%</th><th>Ataxia%</th><th>Leigh%</th><th>HCM%</th>
            <th>CoQ10 Muscle%</th><th>CoQ10 Response</th>
          </tr></thead>
          <tbody>
            {filtered.map(g => (
              <tr key={g.gene}>
                <td className="fw-bold" style={{ color: classColors[g.gene_class] || COLOR }}>{g.gene}</td>
                <td><Badge text={g.gene_class.replace(/_/g, '-')} color={classColors[g.gene_class] || COLOR} /></td>
                <td className="small text-muted">{g.pathway_step}</td>
                <td>{g.chromosome}</td>
                <td>{g.aa}</td>
                <td>{g.srns_pct}%</td>
                <td>{g.snhl_pct}%</td>
                <td>{g.ataxia_pct}%</td>
                <td>{g.leigh_pct}%</td>
                <td>{g.hcm_pct}%</td>
                <td style={{ color: g.avg_coq10_muscle_pct < 20 ? COLOR6 : COLOR4 }}>
                  <strong>{g.avg_coq10_muscle_pct}%</strong>
                </td>
                <td className="small">{g.coq10_response?.split(' — ')[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const genes = data.genes || [];

  const classColors = {
    decaprenyl_pp:    COLOR2,
    ring_attachment:  COLOR3,
    ring_modification: COLOR4,
    coq_kinase:       COLOR5,
  };

  return (
    <>
      {genes.map(g => (
        <div key={g.gene} className="card mb-4 shadow-sm" style={{ borderLeft: `4px solid ${classColors[g.gene_class] || COLOR}` }}>
          <div className="card-body small">
            <div className="d-flex flex-wrap align-items-start gap-2 mb-2">
              <h6 className="fw-bold mb-0" style={{ color: classColors[g.gene_class] || COLOR }}>{g.gene}</h6>
              <Badge text={g.gene_class.replace(/_/g, '-')} color={classColors[g.gene_class] || COLOR} />
              <Badge text={g.chromosome} color="#546e7a" />
              <Badge text={g.aa} color="#546e7a" />
              <Badge text={`Onset ${g.median_onset_months}m`} color="#795548" />
              <Badge text={`CoQ10 ${g.avg_coq10_muscle_pct}% in muscle`} color={g.avg_coq10_muscle_pct < 15 ? COLOR6 : COLOR4} />
            </div>

            <div className="mb-2">
              <span className="fw-semibold">Pathway Step: </span>{g.pathway_step}
            </div>
            <div className="mb-2">
              <span className="fw-semibold">Phenotype: </span>{g.phenotype_summary}
            </div>

            <div className="row g-2 mb-2">
              {[
                ['SRNS', g.srns_pct, COLOR7],
                ['SNHL', g.snhl_pct, COLOR],
                ['Ataxia', g.ataxia_pct, COLOR8],
                ['Leigh', g.leigh_pct, COLOR9],
                ['Myopathy', g.myopathy_pct, COLOR4],
                ['Exer. Intol.', g.exercise_intol_pct, COLOR5],
                ['Lactic Ac.', g.lactic_ac_pct, COLOR6],
                ['HCM', g.hcm_pct, COLOR9],
              ].map(([label, pct, col]) => (
                <div key={label} className="col-6 col-md-3">
                  <div className="text-center p-1 rounded" style={{ background: '#f5f5f5' }}>
                    <div className="fw-bold small" style={{ color: col }}>{pct}%</div>
                    <div className="text-muted" style={{ fontSize: '0.7rem' }}>{label}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mb-1">
              <span className="fw-semibold" style={{ color: COLOR }}>Hallmark: </span>{g.hallmark}
            </div>
            <div className="mb-1">
              <span className="fw-semibold">Key DDx: </span>{g.key_ddx}
            </div>
            <div className="mb-1">
              <span className="fw-semibold">Founder Variants: </span>{g.founder_variant}
            </div>
            <div className="p-2 rounded mt-2" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>
              <span className="fw-semibold">CoQ10 Response: </span>{g.coq10_response}
            </div>
          </div>
        </div>
      ))}
    </>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const terms  = data.terms  || {};
  const classes = data.gene_classes || {};
  const protos = data.supplementation_protocols || {};

  return (
    <>
      <SectionCard title="Gene Class Definitions" borderColor={COLOR4}>
        <dl className="row small">
          {Object.entries(classes).map(([k, v]) => (
            <>
              <dt key={`k-${k}`} className="col-12 col-md-3 fw-bold" style={{ color: COLOR4 }}>{k.replace(/_/g, '-')}</dt>
              <dd key={`v-${k}`} className="col-12 col-md-9">{v}</dd>
            </>
          ))}
        </dl>
      </SectionCard>

      <SectionCard title="CoQ10 Supplementation Protocols" borderColor={COLOR}>
        <dl className="row small">
          {Object.entries(protos).map(([k, v]) => (
            <>
              <dt key={`k-${k}`} className="col-12 col-md-3 fw-bold" style={{ color: COLOR }}>{k.replace(/_/g, ' ')}</dt>
              <dd key={`v-${k}`} className="col-12 col-md-9">{v}</dd>
            </>
          ))}
        </dl>
      </SectionCard>

      <SectionCard title="Glossary — CoQ10 Biosynthesis Terms" borderColor={COLOR5}>
        <dl className="row small">
          {Object.entries(terms).map(([k, v]) => (
            <>
              <dt key={`k-${k}`} className="col-12 col-md-3 fw-bold" style={{ color: COLOR5 }}>{k.replace(/_/g, ' ')}</dt>
              <dd key={`v-${k}`} className="col-12 col-md-9">{v}</dd>
            </>
          ))}
        </dl>
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function CoQ10BiosynthesisAtlasPage() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bk,   setBk]   = useState(null);
  const [def,  setDef]  = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    fetch(`${API}/api/coq10-biosynthesis-atlas/overview`)
      .then(r => r.json()).then(setOv).catch(() => setErr('overview failed'));
    fetch(`${API}/api/coq10-biosynthesis-atlas/breakdown`)
      .then(r => r.json()).then(setBk).catch(() => setErr('breakdown failed'));
    fetch(`${API}/api/coq10-biosynthesis-atlas/definitions`)
      .then(r => r.json()).then(setDef).catch(() => setErr('definitions failed'));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex flex-wrap align-items-center gap-2 mb-3">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          🌿 CoQ10-Biosynthesis-Atlas
        </h4>
        <span className="badge" style={{ backgroundColor: COLOR }}>10 Genes</span>
        <span className="badge" style={{ backgroundColor: COLOR4 }}>400 Patients</span>
        <span className="badge" style={{ backgroundColor: COLOR6 }}>Statins ABSOLUTE CI</span>
        <span className="badge" style={{ backgroundColor: COLOR5 }}>WES Detects All 10</span>
        <Link href="/" className="btn btn-sm btn-outline-secondary ms-auto">← Portal</Link>
      </div>
      {err && <div className="alert alert-danger py-1">Error: {err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottom: `2px solid ${COLOR}` } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab   data={ov}  />}
      {tab === 1 && <GeneTableTab  data={bk}  />}
      {tab === 2 && <ClinicalAtlasTab data={bk} />}
      {tab === 3 && <DefinitionsTab data={def} />}
    </div>
  );
}
