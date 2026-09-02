'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'ND1-Module & IMM Assembly', 'Definitions'];
const COLOR = '#004d40';   // dark teal — integral IMM / TM-helix anchor / TIMMDC1
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
  const ff  = data.feature_frequencies_pct || {};
  const bf  = data.biochemical_fingerprint || {};
  const p   = data.protein || {};
  const mod = data.timmdc1_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="HCM Rate"         value=">80% (HIGH)"                color="#b71c1c" />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: '#ffebee', borderLeft: '4px solid #c62828' }}>
        <strong>🔴 HCM &gt;80% — Highest in Class-3 ND1-Module Group — Critical DDx vs NDUFAF3/4/5</strong> —
        TIMMDC1 deficiency causes hypertrophic cardiomyopathy in &gt;80% of patients. This is the highest
        HCM rate of any Class-3 ND1-module gene (NDUFAF3/4: 15–25%; NDUFAF5: &lt;20%). High HCM + Class-3
        BN-PAGE + isolated CI + 3q25.1 = TIMMDC1 fingerprint. Echocardiography at diagnosis and 6-monthly.
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 ONLY INTEGRAL IMM MEMBER OF CLASS 3 — 2 TM Helices</strong> —
        NDUFAF3, NDUFAF4, and NDUFAF5 are all soluble matrix proteins (zero TM helices).
        TIMMDC1 is the SOLE Class-3 factor anchored to the inner mitochondrial membrane via 2 TM helices.
        TM-helix mutations have a compound effect: disrupt both IMM anchoring AND matrix-loop
        ND1-sub-assembly contact. This IMM-anchored scaffold role explains the high HCM rate in cardiac tissue.
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🔵 TIMMDC1 (3q25.1) — Same Chromosome 3 as ACAD9 (3q21.3) — CRITICAL DDx</strong> —
        TIMMDC1 and ACAD9 are both on chromosome 3. ACAD9 is MCIA-class (ND2/ND5 module, riboflavin-responsive
        50-60%). TIMMDC1 is Class-3 ND1-module (0% riboflavin response, no FAD domain). Riboflavin
        response is the KEY clinical distinguisher on chromosome 3. WES essential: 3q25.1 vs 3q21.3.
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔄 Key Pathway Note — Integral IMM / HCM / No Riboflavin / Same Chr3 as ACAD9">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      {mod.gene && (
        <SectionCard title="⚙️ TIMMDC1 Module Summary — Integral IMM Scaffold">
          <div className="alert mb-2" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Gene:</strong> {mod.gene}
          </div>
          <div className="row g-3">
            <div className="col-md-6">
              <div className="small text-muted mb-1">Module class</div>
              <div className="fw-bold small">{mod.module_class}</div>
            </div>
            <div className="col-md-6">
              <div className="small text-muted mb-1">Assembly position</div>
              <div className="fw-bold small">{mod.assembly_position}</div>
            </div>
          </div>
          <div className="mt-2">
            <div className="small text-muted mb-1">Integral IMM — unique in Class 3</div>
            <p className="small mb-1">{mod.integral_imm_unique}</p>
            <div className="small text-muted mb-1">HCM mechanism (&gt;80%)</div>
            <p className="small mb-1">{mod.hcm_mechanism}</p>
            <div className="small text-muted mb-1">Effect of TIMMDC1 loss</div>
            <p className="small mb-0">{mod.timmdc1_loss_effect}</p>
          </div>
        </SectionCard>
      )}

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span
              className={
                k === 'Complex_I' ? 'text-danger fw-bold' :
                k === 'Riboflavin_response' ? 'text-danger' :
                k === 'HCM_rate' ? 'text-danger fw-bold' :
                k.startsWith('Complex_') ? 'text-success' : 'fw-bold'
              }
              style={k === 'Integral_IMM_status' ? { color: COLOR } : undefined}
            >{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-689)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v}
            color={
              k === 'HCM' ? '#c62828' :  // red for HCM — key distinguishing high feature
              k === 'Riboflavin_responder' || k === 'Peripheral_neuropathy' ||
              k === 'Leukodystrophy' || k === 'Hepatopathy' || k === 'Olfactory_bulb_lesions'
                ? '#4caf50'   // green for hard-0 protective features
                : COLOR
            } />
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
            style={{
              background: a.startsWith('🔴') ? '#ffebee' : a.startsWith('🟠') ? '#fff3e0' : a.startsWith('🟡') ? '#fff8e1' : a.startsWith('🟢') ? '#e8f5e9' : '#e3f2fd',
              borderLeft: `4px solid ${a.startsWith('🔴') ? '#c62828' : a.startsWith('🟠') ? '#e65100' : a.startsWith('🟡') ? '#f57f17' : a.startsWith('🟢') ? '#2e7d32' : '#1565c0'}`
            }}>
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
                <td>{p.hcm ? <span className="badge bg-danger" style={{ fontSize: '0.6rem' }}>HCM</span> : '—'}</td>
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
                  <td><span className="badge" style={{
                    background: d.class === 'ABSOLUTE CI' ? '#c62828' :
                                d.class === 'CONTRAINDICATED' ? '#e53935' :
                                d.class === 'AVOID' || d.class === 'AVOID in HCM' ? '#f57f17' : '#ff8f00'
                  }}>{d.class}</span></td>
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
                  <td><span className="badge" style={{
                    background: t.evidence.includes('Level B') ? '#1565c0' :
                                t.evidence.includes('Level C') ? COLOR :
                                '#2e7d32',
                    fontSize: '0.65rem'
                  }}>{t.evidence}</span></td>
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

// ── Tab: ND1-Module & IMM Assembly ────────────────────────────────────────────
function ND1ModuleTab({ data }) {
  if (!data?.nd1_module_steps) return <p className="text-muted">Loading…</p>;

  const stepColors = [COLOR, '#00695c', '#b71c1c', '#c62828'];

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>TIMMDC1 — Only Integral-IMM Class-3 ND1-Module Factor (2 TM Helices)</strong><br />
        <span className="small text-muted">
          TIMMDC1 is anchored to the inner mitochondrial membrane via 2 TM helices, positioning its
          matrix-exposed loop to contact the ND1-containing sub-assembly. NDUFAF3, NDUFAF4, and NDUFAF5
          are all soluble matrix proteins. This IMM-anchored scaffold explains the dominant HCM (&gt;80%)
          seen in cardiomyocytes that are densely packed with mitochondria requiring IMM-surface CI assembly.
          WES mandatory to confirm 3q25.1 locus. Same chromosome 3 as ACAD9 (3q21.3) — different arm.
        </span>
      </div>

      {data.nd1_module_steps.map((step) => (
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
                    background: step.status_in_timmdc1_deficiency.startsWith('INTACT') ? '#2e7d32' :
                                step.status_in_timmdc1_deficiency.startsWith('DISRUPTED') ? '#c62828' :
                                step.status_in_timmdc1_deficiency.startsWith('ABSENT') ? '#e65100' : '#b71c1c',
                    fontSize: '0.7rem'
                  }}>
                    {step.status_in_timmdc1_deficiency}
                  </span>
                </div>
                <p className="small text-muted mb-0">{step.note}</p>
              </div>
            </div>
          </div>
        </div>
      ))}

      <SectionCard title="🔗 CI Assembly Intermediate Classes — BN-PAGE" borderColor="#1565c0">
        <p className="small text-muted mb-3">
          BN-PAGE of CI assembly intermediates identifies three distinct accumulating sub-assembly classes.
          TIMMDC1 belongs to Class 3 (ND1-module) — same class as NDUFAF3, NDUFAF4, and NDUFAF5.
          TIMMDC1 is the only integral-IMM member of this class.
        </p>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Class</th><th>Members</th><th>Module</th><th>BN-PAGE Band</th><th>Riboflavin</th>
              </tr>
            </thead>
            <tbody>
              {(data.assembly_module_comparison || []).map((c, i) => (
                <tr key={i} style={{ background: c.class.includes('Class 3') ? LIGHT : undefined }}>
                  <td className="fw-bold" style={{ color: c.class.includes('Class 3') ? COLOR : undefined }}>
                    {c.class}
                    {c.class.includes('Class 3') && (
                      <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>TIMMDC1</span>
                    )}
                  </td>
                  <td className="font-monospace small">{c.members}</td>
                  <td>{c.module}</td>
                  <td className="text-muted small">{c.bnpage}</td>
                  <td>
                    <span className="badge" style={{
                      background: c.riboflavin.includes('50-60%') ? '#1565c0' : '#c62828',
                      fontSize: '0.65rem'
                    }}>
                      {c.riboflavin}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ Class 3 ND1-Module Comparison — NDUFAF3 / NDUFAF4 / NDUFAF5 / TIMMDC1" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#fff8e1', borderLeft: '4px solid #f57f17' }}>
          <strong>All four genes produce Class-3 BN-PAGE ND1-module intermediates.</strong> WES chromosomal
          locus is mandatory to distinguish them. Pre-WES clinical clue: <strong>HCM &gt;80% → TIMMDC1;
          HCM 15–25% → NDUFAF3/4; HCM &lt;20% → NDUFAF5.</strong> TM-helix status distinguishes
          protein topology (integral IMM vs soluble matrix).
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr>
              <th>Gene</th><th>Chromosome</th><th>TM Helices</th><th>Heterodimer</th><th>HCM</th><th>Integral IMM</th>
            </tr></thead>
            <tbody>
              {(data.nd1_class3_comparison || []).map(r => (
                <tr key={r.gene} style={{ background: r.gene === 'TIMMDC1' ? LIGHT : undefined }}>
                  <td className="fw-bold" style={{ color: r.gene === 'TIMMDC1' ? COLOR : undefined }}>{r.gene}</td>
                  <td className="font-monospace">{r.chromosome}</td>
                  <td className={r.tm_helices.includes('2') ? 'text-danger fw-bold' : 'text-muted'}>{r.tm_helices}</td>
                  <td>{r.heterodimer}</td>
                  <td className={r.hcm.includes('>80%') ? 'text-danger fw-bold' : ''}>{r.hcm}</td>
                  <td>{r.integral_imm === 'Yes — ONLY integral-IMM Class-3 member'
                    ? <span className="badge bg-danger" style={{ fontSize: '0.65rem' }}>YES — Only one</span>
                    : <span className="text-muted">{r.integral_imm}</span>}
                  </td>
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
export default function TIMMDC1Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/timmdc1/overview`).then(r => r.json()),
      fetch(`${API}/api/timmdc1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/timmdc1/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 TIMMDC1 — Complex I Deficiency (C3orf58)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>ND1-Module Assembly</span>
        <span className="badge ms-1 bg-danger">HCM &gt;80%</span>
        <span className="badge ms-1" style={{ background: '#e65100' }}>Integral IMM — 2 TM Helices</span>
        <span className="badge ms-1 bg-danger">No Riboflavin Response</span>
        <span className="badge ms-1 bg-secondary">3q25.1</span>
        <span className="badge ms-1 bg-secondary">OMIM *615530</span>
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
      {tab === 2 && <ND1ModuleTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
