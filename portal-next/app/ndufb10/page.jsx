'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — PD-module ND4L-face / PDSW peripheral scaffold
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
        <KPI label="Gene"              value={data.gene}                   color={COLOR} />
        <KPI label="Also known as"     value="PDSW"                        color={COLOR} />
        <KPI label="OMIM Gene"         value={`*${data.omim_gene}`}        color={COLOR} />
        <KPI label="Chromosome"        value={data.chromosome}             color={COLOR} />
        <KPI label="Inheritance"       value={data.inheritance}            color={COLOR} />
        <KPI label="Protein (mature)"  value={`${p.size_kda} kDa`}        color={COLOR} />
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-1"><strong>TM helices:</strong> {p.tm_helices} (peripheral — no canonical IMM-spanning TM helix; ND4L-face beta-hairpin anchor)</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="⚡ Key Pathway Note — PD-Module ND4L-Face Beta-Hairpin Scaffold (NDUFB10 / PDSW)">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between small mb-1">
            <span className="fw-semibold">{k.replace(/_/g, ' ').toUpperCase()}</span>
            <span className={v.includes('NORMAL') ? 'text-success' : v.includes('ELEVATED') || v.includes('SEVERELY') ? 'text-danger fw-bold' : 'text-dark'}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title={`📊 Feature Frequencies (${data.cohort_n}-patient cohort, seed-${data.seed})`}>
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k} value={v} />
        ))}
      </SectionCard>

      <SectionCard title="📋 Cohort Summary">
        <div className="row g-2">
          <div className="col-md-4">
            <div className="border rounded p-2 text-center small">
              <div className="fw-bold fs-5" style={{ color: COLOR }}>{data.cohort_n}</div>
              <div className="text-muted">Patients (synthetic cohort)</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="border rounded p-2 text-center small">
              <div className="fw-bold fs-5" style={{ color: COLOR }}>{data.avg_onset_years}y</div>
              <div className="text-muted">Mean onset age</div>
            </div>
          </div>
          <div className="col-md-4">
            <div className="border rounded p-2 text-center small">
              <div className="fw-bold fs-5" style={{ color: COLOR }}>{data.avg_ci_activity_pct}%</div>
              <div className="text-muted">Mean CI activity (% control)</div>
            </div>
          </div>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ───────────────────────────────────────────────────
function PatientsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const onset = data.onset_age_buckets || {};
  const muts  = data.mutation_distribution || [];
  const regs  = data.region_distribution  || [];
  const outs  = data.outcome_distribution || [];
  const pts   = data.patients_sample      || [];

  return (
    <>
      <SectionCard title="🧬 Known Pathogenic Variants">
        {(data.known_mutations || []).map((m, i) => (
          <div key={i} className="border rounded p-2 mb-2 small">
            <span className="fw-bold text-danger">{m.variant}</span>
            <span className="ms-2 text-muted">({m.cdna})</span>
            <span className="ms-2 badge" style={{ background: COLOR }}>{m.severity}</span>
            <div className="mt-1"><strong>Domain:</strong> {m.domain}</div>
            <div><strong>Effect:</strong> {m.effect}</div>
          </div>
        ))}
      </SectionCard>

      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <SectionCard title="📅 Onset Age Distribution">
            {Object.entries(onset).map(([k, v]) => (
              <div key={k} className="d-flex justify-content-between small mb-1">
                <span>{k.replace(/_/g, ' ')}</span>
                <span className="fw-bold" style={{ color: COLOR }}>{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🌍 Cohort Region">
            {regs.slice(0, 6).map((r, i) => (
              <Bar key={i} label={r.region} value={r.pct} />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="📊 Mutation Distribution">
        {muts.slice(0, 6).map((m, i) => (
          <Bar key={i} label={m.mutation.length > 60 ? m.mutation.slice(0, 60) + '…' : m.mutation} value={m.pct} />
        ))}
      </SectionCard>

      <SectionCard title="📈 Outcome Distribution">
        <div className="row g-2">
          {outs.map((o, i) => (
            <div key={i} className="col-md-4">
              <div className="border rounded p-2 text-center small">
                <div className="fw-bold fs-5" style={{ color: COLOR }}>{o.count}</div>
                <div className="text-muted">{o.outcome}</div>
                <div className="text-muted small">{o.pct}%</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="🗂 Patient Sample (first 15)">
        <div style={{ overflowX: 'auto' }}>
          <table className="table table-sm table-bordered small">
            <thead>
              <tr style={{ background: COLOR, color: '#fff' }}>
                <th>ID</th><th>Sex</th><th>Onset (y)</th><th>Mutation</th>
                <th>Region</th><th>CI%</th><th>Leigh MRI</th><th>Lactate↑</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {pts.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_years}</td>
                  <td style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.mutation}</td>
                  <td>{p.region}</td>
                  <td>{p.ci_activity_pct_control}%</td>
                  <td>{p.leigh_mri ? '✅' : '❌'}</td>
                  <td>{p.lactic_acidosis ? '✅' : '❌'}</td>
                  <td>{p.outcome}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Treatments & DDx ─────────────────────────────────────────────────────
function TreatmentsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const tx  = data.treatments || {};
  const ddx = data.ddx_key_negatives || [];

  return (
    <>
      <SectionCard title="🚫 Absolute Contraindications" borderColor="#c62828">
        {(tx.absolute_contraindicated || []).map((d, i) => (
          <Alert key={i} variant="danger" text={<><strong>{d.drug}:</strong> {d.reason}</>} />
        ))}
      </SectionCard>

      <SectionCard title="⛔ Contraindicated" borderColor="#b71c1c">
        {(tx.contraindicated || []).map((d, i) => (
          <Alert key={i} variant="danger" text={<><strong>{d.drug}:</strong> {d.reason}</>} />
        ))}
      </SectionCard>

      <SectionCard title="⚠️ Avoid / High Caution" borderColor="#f57f17">
        {(tx.avoid_caution || []).map((d, i) => (
          <Alert key={i} variant="warning" text={<><strong>{d.drug}:</strong> {d.reason}</>} />
        ))}
      </SectionCard>

      <SectionCard title="💊 Level C Cofactors (standard CI-Leigh supportive)">
        {(tx.level_c_cofactors || []).map((c, i) => (
          <div key={i} className="border rounded p-2 mb-2 small">
            <span className="fw-bold" style={{ color: COLOR }}>{c.agent}</span>
            <span className="ms-2 text-muted">{c.dose}</span>
            <div className="mt-1 text-muted">{c.note}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="✅ Supportive Protocols">
        <Alert variant="success" text={<><strong>Preferred AED:</strong> {tx.preferred_aed}</>} />
        <Alert variant="success" text={<><strong>Glucose protocol:</strong> {tx.glucose_protocol}</>} />
        <Alert variant="success" text={<><strong>Anaesthesia:</strong> {tx.anaesthesia}</>} />
      </SectionCard>

      <SectionCard title="🔍 Key Negatives — Differential Diagnosis">
        {ddx.map((d, i) => (
          <div key={i} className="border rounded p-2 mb-2 small">
            <span className="badge me-2" style={{ background: '#2e7d32' }}>ABSENT</span>
            <strong>{d.finding}</strong>
            <div className="text-muted mt-1">DDx excluded: {d.ddx_excluded}</div>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <>
      <SectionCard title="📚 NDUFB10 / PDSW Definition">
        <p className="small mb-0">{data.ndufb10_pdsw}</p>
      </SectionCard>

      <SectionCard title="🧩 PD-Module">
        <p className="small mb-0">{data.pd_module}</p>
      </SectionCard>

      <SectionCard title="🔬 ND4L Face">
        <p className="small mb-0">{data.nd4l_face}</p>
      </SectionCard>

      <SectionCard title="⚗️ Isolated CI Deficiency">
        <p className="small mb-0">{data.isolated_ci_deficiency}</p>
      </SectionCard>

      <SectionCard title="🔵 BN-PAGE Absent CI Pattern">
        <p className="small mb-0">{data.bn_page_absent_ci}</p>
      </SectionCard>

      <SectionCard title="⚠️ PDSW vs NDUFB6 — Critical DDx Distinction">
        <Alert variant="warning" text={data.pdsw_vs_ndufb6} />
      </SectionCard>

      <SectionCard title="📄 References">
        <ul className="small mb-0">
          {(data.references || []).map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function NDUFB10Page() {
  const [tab, setTab] = useState(0);
  const [overview,  setOverview]  = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs,      setDefs]      = useState(null);
  const [error,     setError]     = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [ov, bd, df] = await Promise.all([
          fetch(`${API}/api/ndufb10/overview`).then(r => r.json()),
          fetch(`${API}/api/ndufb10/breakdown`).then(r => r.json()),
          fetch(`${API}/api/ndufb10/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bd);
        setDefs(df);
      } catch (e) {
        setError(e.message);
      }
    };
    load();
  }, []);

  if (error) return (
    <div className="container py-4">
      <div className="alert alert-danger">Failed to load NDUFB10 data: {error}</div>
    </div>
  );

  return (
    <div className="container-fluid py-3">
      <div className="mb-3" style={{ borderBottom: `3px solid ${COLOR}`, paddingBottom: 8 }}>
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
          🧬 NDUFB10 — Leigh Syndrome / Isolated Complex I Deficiency
        </h4>
        <p className="text-muted small mb-0">
          PDSW · PD-Module ND4L-Face Beta-Hairpin Peripheral Scaffold · 16p13.3 ·
          OMIM *603843 / #256000 · AR biallelic · 40-patient cohort seed-649
        </p>
      </div>

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <TreatmentsTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={defs} />}
    </div>
  );
}
