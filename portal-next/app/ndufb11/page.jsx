'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#006064';   // dark teal-cyan — PP-module ND6/ND1-face / X-linked ESSS
const LIGHT = '#e0f7fa';

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

function Alert({ variant, text }) {
  const bg     = variant === 'danger'  ? '#ffebee' : variant === 'warning' ? '#fff8e1'
               : variant === 'success' ? '#e8f5e9' : LIGHT;
  const border = variant === 'danger'  ? '#c62828' : variant === 'warning' ? '#f57f17'
               : variant === 'success' ? '#2e7d32' : COLOR;
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
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

// ── Tab: Overview ──────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const ff = data.feature_frequencies_pct || {};
  const bf = data.biochemical_fingerprint || {};
  const p  = data.protein || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"              value={data.gene}                        color={COLOR} />
        <KPI label="Also known as"     value="ESSS"                             color={COLOR} />
        <KPI label="OMIM Gene"         value={`*${data.omim_gene}`}             color={COLOR} />
        <KPI label="Chromosome"        value={data.chromosome}                  color={COLOR} />
        <KPI label="Inheritance"       value="X-LINKED"                         color="#c62828" />
        <KPI label="Protein (mature)"  value={`${p.size_kda} kDa`}             color={COLOR} />
      </div>

      <Alert variant="danger" text="⚠️ X-LINKED inheritance — ONLY X-linked nuclear NDUFB gene. Hemizygous males: severe/lethal neonatal. Heterozygous females: variable (X-inactivation-dependent). Pedigree showing X-linked pattern + CI deficiency + Leigh = NDUFB11 Xp11.3 first." />

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-1"><strong>TM helices:</strong> {p.tm_helices} (single IMM-spanning anchor at ND6/ND1 boundary)</p>
        <p className="small mb-1"><strong>Size:</strong> {p.size_aa} aa / {p.size_kda} kDa</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔑 Pathway Note">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      <SectionCard title="🧪 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <p key={k} className="small mb-1">
            <strong>{k.replace(/_/g, ' ')}:</strong>{' '}
            <span style={{ color: v.includes('NORMAL') ? '#2e7d32' : v.includes('ELEVATED') || v.includes('SEVERELY') ? '#c62828' : undefined }}>
              {v}
            </span>
          </p>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (n={data.cohort_n})">
        {Object.entries(ff).map(([k, v]) => <Bar key={k} label={k} value={v} />)}
      </SectionCard>

      <SectionCard title="👥 Cohort Demographics">
        <div className="row">
          <div className="col-md-6">
            <p className="small mb-1"><strong>Total patients:</strong> {data.cohort_n}</p>
            <p className="small mb-1"><strong>Males (hemizygous):</strong> {data.cohort_males} ({Math.round(data.cohort_males/data.cohort_n*100)}%)</p>
            <p className="small mb-1"><strong>Females (heterozygous):</strong> {data.cohort_females} ({Math.round(data.cohort_females/data.cohort_n*100)}%)</p>
          </div>
          <div className="col-md-6">
            <p className="small mb-1"><strong>Mean onset age:</strong> {data.avg_onset_years} years</p>
            <p className="small mb-1"><strong>Mean CI activity:</strong> {data.avg_ci_activity_pct}% of control</p>
            <p className="small mb-1"><strong>Seed:</strong> {data.seed}</p>
          </div>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const bd = data.breakdown || {};
  const sd = bd.sex_distribution || {};
  const patients = bd.patients_sample || [];
  const onset    = bd.onset_age_buckets || {};

  return (
    <>
      <SectionCard title="⚥ Sex Distribution (X-linked)">
        <div className="row mb-3">
          <div className="col-md-6">
            <Bar label={`Males hemizygous (${sd.males_hemizygous})`}   value={sd.male_pct}   color="#1565c0" />
            <Bar label={`Females heterozygous (${sd.females_heterozygous})`} value={sd.female_pct} color="#ad1457" />
          </div>
          <div className="col-md-6">
            <Alert variant="warning" text="Males: severe neonatal CI-Leigh (hemizygous). Females: variable phenotype — may be carrier-only, mild, or asymptomatic depending on X-inactivation pattern." />
          </div>
        </div>
      </SectionCard>

      <SectionCard title="🕒 Age of Onset Distribution">
        <div className="row">
          {[
            ['Neonatal (<6 mo)', onset.neonatal_under_6mo],
            ['Infantile (6 mo–2 yr)', onset.infantile_6mo_to_2yr],
            ['Childhood (2–10 yr)', onset.childhood_2_to_10yr],
            ['Juvenile (>10 yr)', onset.juvenile_over_10yr],
          ].map(([label, count]) => (
            <div key={label} className="col-6 col-md-3 mb-2 text-center">
              <div className="fw-bold fs-4" style={{ color: COLOR }}>{count}</div>
              <div className="small text-muted">{label}</div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🧬 Known Mutations">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>Variant</th><th>cDNA</th><th>Domain</th><th>Effect</th><th>Severity</th>
              </tr>
            </thead>
            <tbody>
              {(bd.known_mutations || []).map((m, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{m.variant}</td>
                  <td className="small">{m.cdna}</td>
                  <td className="small">{m.domain}</td>
                  <td className="small">{m.effect}</td>
                  <td className="small">
                    <span className="badge" style={{
                      backgroundColor: m.severity.includes('Severe') ? '#c62828' :
                                       m.severity.includes('Moderate') ? '#f57f17' : '#2e7d32',
                      color: 'white'
                    }}>{m.severity}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="👥 Patient Sample (first 15)">
        <div className="table-responsive">
          <table className="table table-sm table-striped">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr>
                <th>#</th><th>Sex</th><th>Onset (yr)</th><th>Mutation</th><th>Region</th>
                <th>CI%</th><th>Leigh MRI</th><th>Lactic↑</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {patients.map(p => (
                <tr key={p.id}>
                  <td className="small">{p.id}</td>
                  <td className="small"><span style={{ color: p.sex === 'M' ? '#1565c0' : '#ad1457', fontWeight: 'bold' }}>{p.sex}</span></td>
                  <td className="small">{p.age_onset_years}</td>
                  <td className="small" style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.mutation}</td>
                  <td className="small">{p.region}</td>
                  <td className="small">{p.ci_activity_pct_control}%</td>
                  <td className="small">{p.leigh_mri ? '✓' : '—'}</td>
                  <td className="small">{p.lactic_acidosis ? '✓' : '—'}</td>
                  <td className="small">{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🌍 Regional Distribution">
        {(bd.region_distribution || []).map(r => (
          <Bar key={r.region} label={`${r.region} (${r.count})`} value={r.pct} />
        ))}
      </SectionCard>

      <SectionCard title="📈 Outcome Distribution">
        {(bd.outcome_distribution || []).map(o => (
          <Bar key={o.outcome} label={`${o.outcome} (${o.count})`} value={o.pct}
               color={o.outcome.includes('deceased') ? '#c62828' : o.outcome.includes('progressing') ? '#f57f17' : '#2e7d32'} />
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments & DDx ─────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const tx  = (data.breakdown || {}).treatments || {};
  const ddx = (data.breakdown || {}).ddx_key_negatives || [];

  return (
    <>
      <Alert variant="danger" text="🚨 ABSOLUTE CI: Metformin · Valproate (VPA) · Linezolid · Chloramphenicol — all directly block CI or mtDNA-encoded ND subunit synthesis." />
      <Alert variant="danger" text="🚨 CONTRAINDICATED: Ketogenic diet — forces NADH → β-oxidation; CI cannot re-oxidise NADH with NDUFB11 PP-module scaffold lost." />
      <Alert variant="warning" text="⚠️ GENETIC COUNSELLING MANDATORY: X-linked recessive. Offer prenatal diagnosis; cascade testing of maternal relatives." />

      <SectionCard title="🚫 Absolute Contraindications">
        {(tx.absolute_contraindicated || []).map(d => (
          <Alert key={d.drug} variant="danger" text={`${d.drug}: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="❌ Contraindicated">
        {(tx.contraindicated || []).map(d => (
          <Alert key={d.drug} variant="warning" text={`${d.drug}: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Avoid / High Caution">
        {(tx.avoid_caution || []).map(d => (
          <Alert key={d.drug} variant="warning" text={`${d.drug}: ${d.reason}`} />
        ))}
      </SectionCard>

      <SectionCard title="💊 Level C Cofactors (CI Supportive)">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr><th>Agent</th><th>Dose</th><th>Rationale</th></tr>
            </thead>
            <tbody>
              {(tx.level_c_cofactors || []).map(c => (
                <tr key={c.agent}>
                  <td className="small fw-bold">{c.agent}</td>
                  <td className="small">{c.dose}</td>
                  <td className="small">{c.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="⚡ Acute Management">
        <p className="small mb-1"><strong>Preferred AED:</strong> {tx.preferred_aed}</p>
        <p className="small mb-1"><strong>Glucose protocol:</strong> {tx.glucose_protocol}</p>
        <p className="small mb-0"><strong>Anaesthesia:</strong> {tx.anaesthesia}</p>
      </SectionCard>

      <SectionCard title="🧬 Genetic Counselling (X-linked)">
        <Alert variant="warning" text={tx.genetic_counselling} />
      </SectionCard>

      <SectionCard title="🔍 DDx Key Negatives">
        <div className="table-responsive">
          <table className="table table-sm table-bordered">
            <thead style={{ backgroundColor: LIGHT }}>
              <tr><th>Negative Finding</th><th>Conditions Excluded</th></tr>
            </thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td className="small fw-bold">{d.finding}</td>
                  <td className="small">{d.ddx_excluded}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const defs = data.definitions || {};
  const refs  = defs.references || [];
  const others = Object.entries(defs).filter(([k]) => k !== 'references');

  return (
    <>
      {others.map(([key, val]) => (
        <SectionCard key={key} title={key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}>
          <p className="small mb-0">{val}</p>
        </SectionCard>
      ))}
      <SectionCard title="📚 References">
        <ul className="small mb-0">
          {refs.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function NDUFB11Page() {
  const [tab, setTab]   = useState(0);
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufb11/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufb11/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufb11/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(String(e)));
  }, []);

  const tabData = {
    0: overview,
    1: breakdown ? { breakdown } : null,
    2: breakdown ? { breakdown } : null,
    3: definitions ? { definitions } : null,
  };

  return (
    <div className="container-fluid py-4" style={{ maxWidth: 1100 }}>
      {/* Header */}
      <div className="mb-4 p-3 rounded" style={{ background: COLOR, color: 'white' }}>
        <h4 className="mb-1 fw-bold">🧬 NDUFB11 (ESSS) — Leigh Syndrome · X-Linked CI Deficiency</h4>
        <p className="mb-0 small opacity-75">
          PP-Module ND6/ND1-Face · Xp11.3 · OMIM *300403 · X-Linked Recessive ·
          Only X-linked nuclear NDUFB subunit · CI 5–20%
        </p>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

      {/* Clinical disclaimer */}
      <div className="alert alert-warning mb-4 small">
        <strong>⚠️ Clinical Disclaimer:</strong> This dashboard presents synthetic cohort data
        (seed-651, n=40) derived from published literature distributions for educational and
        AI governance research purposes. Not validated for direct clinical decision-making.
        Consult metabolic neurology and clinical genetics for individual patient care.
        <strong> X-LINKED inheritance: genetic counselling is mandatory.</strong>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-4">
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

      {/* Tab content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={tabData[1]} />}
      {tab === 2 && <TreatmentsTab data={tabData[2]} />}
      {tab === 3 && <DefinitionsTab data={tabData[3]} />}
    </div>
  );
}
