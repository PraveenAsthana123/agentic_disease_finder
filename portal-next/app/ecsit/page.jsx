'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'MCIA Assembly', 'Definitions'];
const COLOR = '#1b5e20';   // deep green — ECSIT / innate immunity / MCIA tetramer
const LIGHT = '#e8f5e9';

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
  const co = data.cohort || {};
  const mc = data.mcia_complex_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"              value={data.gene}                  color={COLOR} />
        <KPI label="Riboflavin Resp."  value="0% (None)"                  color="#b71c1c" />
        <KPI label="OMIM Gene"         value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"        value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"       value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"           value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: '#ffebee', borderLeft: '4px solid #c62828' }}>
        <strong>🔴 NO Riboflavin Response — Critical DDx vs ACAD9</strong> — ECSIT has NO FAD-binding domain.
        High-dose riboflavin does NOT rescue ECSIT deficiency. If MCIA-type CI deficiency shows riboflavin
        response: ACAD9 is the diagnosis. No response → consider ECSIT / NDUFAF1 / TMEM126B (after excluding ACAD9).
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🟩 ECSIT — TMEM126B-Recruiting Scaffold &amp; Dual Innate Immunity Role</strong> — ECSIT bridges
        the ACAD9-NDUFAF1 binary core to TMEM126B, completing the MCIA tetramer. ECSIT is the ONLY MCIA complex
        member originally identified in innate immunity (TLR adaptor, Kopp 1999) before its CI assembly role
        was discovered (Vogel 2007).
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔄 Key Pathway Note — MCIA Complex / TMEM126B Recruiter / No Riboflavin Response">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      {mc.tetramer && (
        <SectionCard title="⚙️ MCIA Tetramer Assembly Order">
          <div className="alert mb-2" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Tetramer:</strong> {mc.tetramer}
          </div>
          {(mc.assembly_order || []).map((step, i) => (
            <div key={i} className="d-flex align-items-start mb-2">
              <span className="badge me-2 mt-1" style={{ background: COLOR, minWidth: 28, fontSize: '0.7rem' }}>{i + 1}</span>
              <span className="small">{step}</span>
            </div>
          ))}
          <p className="small text-muted mt-2 mb-0">{mc.ecsit_unique_position}</p>
        </SectionCard>
      )}

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span className={k === 'complex_I' ? 'text-danger fw-bold' : k === 'riboflavin_response' ? 'text-danger' : 'text-success'}>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-679)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v}
            color={k === 'riboflavin_responder' ? '#b71c1c' : k === 'peripheral_neuropathy' || k === 'hepatopathy' ? '#4caf50' : COLOR} />
        ))}
      </SectionCard>

      <SectionCard title="⚖️ Key DDx" borderColor="#1565c0">
        {(data.key_ddx || []).map((d, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: '#e3f2fd', borderLeft: '4px solid #1565c0' }}>
            <div className="fw-bold small mb-1" style={{ color: '#1565c0' }}>{d.feature}</div>
            <p className="small mb-1">{d.significance}</p>
            <div className="text-muted small">Target gene: <strong>{d.target_gene}</strong></div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚨 Clinical Alerts" borderColor="#b71c1c">
        {(data.clinical_alerts || []).map((a, i) => (
          <div key={i} className="mb-1 p-2 rounded small"
            style={{ background: a.startsWith('🔴') ? '#ffebee' : '#fff8e1', borderLeft: `4px solid ${a.startsWith('🔴') ? '#c62828' : '#f57f17'}` }}>
            {a}
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [search, setSearch] = useState('');
  if (!data?.patients) return <p className="text-muted">Loading…</p>;

  const filtered = data.patients.filter(p =>
    p.mutation.toLowerCase().includes(search.toLowerCase()) ||
    p.region.toLowerCase().includes(search.toLowerCase()) ||
    p.outcome.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <div className="row g-3 mb-4">
        {(data.outcome_distribution || []).map(o => (
          <div key={o.outcome} className="col-6 col-md-3">
            <div className="card shadow-sm text-center">
              <div className="card-body py-2">
                <div className="fw-bold fs-5" style={{ color: o.outcome.includes('deceased') ? '#c62828' : COLOR }}>{o.count}</div>
                <div className="text-muted small">{o.outcome}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="📊 Onset Distribution">
            {(data.onset_distribution || []).map(o => (
              <Bar key={o.bin} label={o.bin} value={Math.round(o.count / data.patients.length * 100)} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="🌍 Region Distribution">
            {(data.region_distribution || []).map(r => (
              <div key={r.region} className="d-flex justify-content-between border-bottom py-1 small">
                <span>{r.region}</span><span className="fw-bold">{r.count}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🧬 Variant Distribution">
        {(data.variant_distribution || []).map(v => (
          <div key={v.mutation_class} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted font-monospace">{v.mutation_class}</span>
            <span className="fw-bold">{v.count}</span>
          </div>
        ))}
      </SectionCard>

      <div className="row mb-3">
        <div className="col-md-6">
          <input className="form-control form-control-sm" placeholder="Filter by mutation / region / outcome…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>

      <div className="table-responsive">
        <table className="table table-sm table-hover small">
          <thead style={{ background: COLOR, color: '#fff' }}>
            <tr>
              <th>#</th><th>Onset (m)</th><th>Sex</th><th>Mutation</th><th>CI %</th>
              <th>Leigh MRI</th><th>Lactic Ac.</th><th>HCM</th><th>Seizures</th><th>Outcome</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(p => (
              <tr key={p.id}>
                <td>{p.id}</td>
                <td>{p.age_onset_months}</td>
                <td>{p.sex}</td>
                <td className="font-monospace" style={{ fontSize: '0.7rem', maxWidth: 220 }}>{p.mutation}</td>
                <td><span className="badge bg-danger">{p.ci_activity_pct}%</span></td>
                <td>{p.leigh_mri ? '✅' : '—'}</td>
                <td>{p.lactic_acidosis ? '✅' : '—'}</td>
                <td>{p.hcm ? '⚠️' : '—'}</td>
                <td>{p.seizures ? '⚠️' : '—'}</td>
                <td><span className="badge" style={{ background: p.outcome.includes('deceased') ? '#c62828' : COLOR, fontSize: '0.65rem' }}>{p.outcome}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SectionCard title="🚫 Contraindicated Drugs" borderColor="#c62828">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Drug</th><th>Mechanism</th><th>Class</th></tr></thead>
            <tbody>
              {(data.contraindicated_drugs || []).map(d => (
                <tr key={d.drug}>
                  <td className="fw-bold">{d.drug}</td>
                  <td>{d.mechanism}</td>
                  <td><span className="badge" style={{ background: d.class === 'ABSOLUTE CI' ? '#c62828' : d.class === 'CONTRAINDICATED' ? '#e53935' : d.class === 'AVOID' ? '#f57f17' : '#ff8f00' }}>{d.class}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="💊 Treatment Protocols" borderColor="#2e7d32">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Agent</th><th>Dose</th><th>Evidence</th><th>Rationale</th></tr></thead>
            <tbody>
              {(data.treatment_protocols || []).map(t => (
                <tr key={t.agent}>
                  <td className="fw-bold">{t.agent}</td>
                  <td className="font-monospace small">{t.dose}</td>
                  <td><span className="badge" style={{ background: t.evidence.includes('Level B') ? '#1565c0' : t.evidence.includes('Level C') ? COLOR : '#6a1b9a', fontSize: '0.65rem' }}>{t.evidence}</span></td>
                  <td className="text-muted small">{t.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: MCIA Assembly ────────────────────────────────────────────────────────
function MCIATab({ data }) {
  if (!data?.mcia_assembly_steps) return <p className="text-muted">Loading…</p>;

  const stepColors = ['#1b5e20', '#2e7d32', '#c62828', '#c62828'];

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>MCIA Tetramer:</strong> ACAD9 – NDUFAF1 – ECSIT – TMEM126B<br />
        <span className="small text-muted">
          ECSIT joins as the 3rd member, after ACAD9-NDUFAF1 binary forms. ECSIT then recruits TMEM126B to complete the tetramer.
          Loss of ECSIT: ACAD9-NDUFAF1 binary still intact, but TMEM126B cannot join — MCIA tetramer incomplete.
        </span>
      </div>

      {data.mcia_assembly_steps.map((step) => (
        <div key={step.step} className="card mb-3 shadow-sm" style={{ borderLeft: `6px solid ${stepColors[step.step - 1]}` }}>
          <div className="card-body py-3">
            <div className="d-flex align-items-start">
              <span className="badge me-3 mt-1" style={{ background: stepColors[step.step - 1], minWidth: 32, fontSize: '0.85rem' }}>
                {step.step}
              </span>
              <div className="flex-grow-1">
                <div className="fw-bold mb-1">{step.event}</div>
                <div className="mb-1">
                  <span className="badge" style={{
                    background: step.status_in_ecsit_deficiency.startsWith('INTACT') ? '#2e7d32' : '#c62828',
                    fontSize: '0.7rem'
                  }}>
                    {step.status_in_ecsit_deficiency}
                  </span>
                </div>
                <p className="small text-muted mb-0">{step.note}</p>
              </div>
            </div>
          </div>
        </div>
      ))}

      <SectionCard title="🔗 MCIA Member Comparison" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Member</th><th>MCIA Step</th><th>Chromosome</th><th>Riboflavin Response</th><th>Unique Feature</th>
              </tr>
            </thead>
            <tbody>
              {[
                { member: 'ACAD9', step: '1st — scaffold', chr: '3q21.3',     ribo: 'Level B (50-60%)', feat: 'FAD-binding domain; exercise-intolerance dominant (p.Arg518His)' },
                { member: 'NDUFAF1', step: '2nd — CIA30 binary', chr: '15q11.2-q13', ribo: 'None (0%)', feat: 'First CI assembly factor ever discovered (Kuffner 1998)' },
                { member: 'ECSIT',  step: '3rd — TMEM126B recruiter', chr: '19p13.3', ribo: 'None (0%)',   feat: 'ONLY MCIA member with dual innate immunity role (TLR adaptor → CI assembly)' },
                { member: 'TMEM126B', step: '4th — integral IMM', chr: '11q14.1', ribo: 'None (0%)', feat: '2 TM helices (only integral membrane MCIA member); recruited by ECSIT' },
              ].map(r => (
                <tr key={r.member} style={{ background: r.member === 'ECSIT' ? LIGHT : undefined }}>
                  <td className="fw-bold">{r.member} {r.member === 'ECSIT' && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>YOU ARE HERE</span>}</td>
                  <td>{r.step}</td>
                  <td className="font-monospace">{r.chr}</td>
                  <td><span className="badge" style={{ background: r.ribo.includes('None') ? '#c62828' : '#1565c0', fontSize: '0.65rem' }}>{r.ribo}</span></td>
                  <td className="text-muted" style={{ fontSize: '0.75rem' }}>{r.feat}</td>
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
  if (!data?.concepts) return <p className="text-muted">Loading…</p>;

  return (
    <>
      <SectionCard title="📖 Concepts & Definitions">
        {data.concepts.map(c => (
          <div key={c.term} className="mb-3 border-bottom pb-2">
            <div className="fw-bold small" style={{ color: COLOR }}>{c.term}</div>
            <p className="small text-muted mb-0">{c.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📏 Clinical Thresholds">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Parameter</th><th>Threshold</th><th>Significance</th></tr></thead>
            <tbody>
              {(data.thresholds || []).map(t => (
                <tr key={t.parameter}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td className="font-monospace text-danger">{t.threshold}</td>
                  <td className="text-muted">{t.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="📋 Standards & Guidelines">
        {(data.standards || []).map(s => (
          <div key={s.code} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="font-monospace fw-bold" style={{ color: COLOR }}>{s.code}</span>
            <span className="text-muted">{s.title}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 References">
        {(data.references || []).map(r => (
          <div key={r.id} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <p className="small fw-bold mb-1">{r.citation}</p>
            <p className="small text-muted mb-0"><em>{r.relevance}</em></p>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function ECSITPage() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ecsit/overview`).then(r => r.json()),
      fetch(`${API}/api/ecsit/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ecsit/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 ECSIT — Complex I Deficiency</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>MCIA Complex</span>
        <span className="badge ms-1 bg-danger">No Riboflavin Response</span>
        <span className="badge ms-1" style={{ background: '#1565c0' }}>TMEM126B Recruiter</span>
        <span className="badge ms-1" style={{ background: '#6a1b9a' }}>Dual Innate Immunity</span>
        <span className="badge ms-1 bg-secondary">19p13.3</span>
        <span className="badge ms-1 bg-secondary">OMIM *608196</span>
        <span className="badge ms-1 bg-secondary">AR</span>
      </div>

      {err && <div className="alert alert-danger small">{err}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <PatientsTab data={breakdown} />}
      {tab === 2 && <MCIATab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
