'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'ND1-Module Assembly', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — NDUFAF3 / early ND1-module / distinct from MCIA teal
const LIGHT = '#ede7f6';

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
  const mod = data.ndufaf3_ndufaf4_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="Riboflavin Resp." value="0% (None)"                  color="#b71c1c" />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: '#ffebee', borderLeft: '4px solid #c62828' }}>
        <strong>🔴 NO Riboflavin Response — Critical DDx vs ACAD9</strong> — NDUFAF3 has NO FAD-binding domain.
        High-dose riboflavin does NOT rescue NDUFAF3 deficiency. If MCIA-class CI deficiency shows riboflavin
        response: ACAD9 is the diagnosis. No response → consider NDUFAF3 / NDUFAF4 / other assembly factors.
      </div>

      <div className="alert mb-4" style={{ background: '#fff8e1', borderLeft: '4px solid #f57f17' }}>
        <strong>🟡 SAME CHROMOSOME ARM DDx: NDUFAF3 (2q33.1) vs NDUFS1 (2q33.3)</strong> — Both at 2q33.
        NDUFS1 causes peripheral neuropathy (50%). NDUFAF3 causes NONE. No peripheral neuropathy
        strongly favours NDUFAF3 over NDUFS1 on the same chromosomal arm.
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🟣 NDUFAF3 — Earliest ND1-Module CI Assembly Factor / NDUFAF4 Obligate Heterodimer</strong> —
        NDUFAF3 (C3orf60, 2q33.1) forms an obligate heterodimer with NDUFAF4 (C6orf66, 6q16.3). This
        heterodimer is the EARLIEST committed CI assembly complex, acting on the ND1-containing P-module
        sub-assembly — upstream of and completely separate from the MCIA tetramer (ND2/ND5 module).
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔄 Key Pathway Note — ND1-Module / NDUFAF3-NDUFAF4 Heterodimer / No Riboflavin">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      {mod.heterodimer && (
        <SectionCard title="⚙️ NDUFAF3-NDUFAF4 Module Summary">
          <div className="alert mb-2" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Heterodimer:</strong> {mod.heterodimer}
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
            <div className="small text-muted mb-1">Distinct from MCIA tetramer</div>
            <p className="small mb-1">{mod.distinct_from_mcia}</p>
            <div className="small text-muted mb-1">Effect of NDUFAF3 loss</div>
            <p className="small mb-0">{mod.ndufaf3_loss_effect}</p>
          </div>
        </SectionCard>
      )}

      <SectionCard title="🔬 Biochemical Fingerprint">
        {Object.entries(bf).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span className={
              k === 'Complex_I' ? 'text-danger fw-bold' :
              k === 'Riboflavin_response' ? 'text-danger' :
              k.startsWith('Complex_') ? 'text-success' : ''
            }>{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-683)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v}
            color={
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
                  <td><span className="badge" style={{
                    background: d.class === 'ABSOLUTE CI' ? '#c62828' :
                                d.class === 'CONTRAINDICATED' ? '#e53935' :
                                d.class === 'AVOID' ? '#f57f17' : '#ff8f00'
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

// ── Tab: ND1-Module Assembly ──────────────────────────────────────────────────
function ND1ModuleTab({ data }) {
  if (!data?.nd1_module_steps) return <p className="text-muted">Loading…</p>;

  const stepColors = ['#4a148c', '#6a1b9a', '#b71c1c', '#c62828'];

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>NDUFAF3-NDUFAF4 Obligate Heterodimer:</strong> NDUFAF3 (2q33.1) + NDUFAF4 (6q16.3)<br />
        <span className="small text-muted">
          The earliest committed CI assembly complex. Acts on the ND1-containing P-module sub-assembly —
          upstream of and completely separate from the MCIA tetramer (ACAD9–NDUFAF1–ECSIT–TMEM126B) which
          handles ND2/ND5. Loss of NDUFAF3 destabilizes NDUFAF4, stalling the ND1-module assembly.
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
                    background: step.status_in_ndufaf3_deficiency.startsWith('INTACT') ? '#2e7d32' :
                                step.status_in_ndufaf3_deficiency.startsWith('DISRUPTED') ? '#c62828' :
                                step.status_in_ndufaf3_deficiency.startsWith('STALLED') ? '#e65100' : '#b71c1c',
                    fontSize: '0.7rem'
                  }}>
                    {step.status_in_ndufaf3_deficiency}
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
          NDUFAF3 belongs exclusively to Class 3 (ND1-module).
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
                      <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>NDUFAF3</span>
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

      <SectionCard title="🏗️ NDUFAF3 vs NDUFAF4 — Obligate Heterodimer Partners" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>Near-identical CI deficiency phenotype.</strong> NDUFAF3 (2q33.1) and NDUFAF4 (6q16.3)
          are obligate heterodimer partners. Loss of either gene destabilizes the other protein.
          Only WES with chromosomal locus resolution distinguishes them.
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead><tr><th>Feature</th><th>NDUFAF3</th><th>NDUFAF4</th></tr></thead>
            <tbody>
              {[
                { feature: 'Chromosome',         n3: '2q33.1',         n4: '6q16.3' },
                { feature: 'OMIM gene',          n3: '*612911',        n4: '*611776' },
                { feature: 'CI deficiency',      n3: '5-20%',          n4: '5-20%' },
                { feature: 'Phenotype',          n3: 'Leigh / CI',     n4: 'Leigh / CI' },
                { feature: 'Riboflavin response',n3: 'None (0%)',      n4: 'None (0%)' },
                { feature: 'Partner protein effect', n3: 'NDUFAF4 secondarily reduced', n4: 'NDUFAF3 secondarily reduced' },
                { feature: 'BN-PAGE class',      n3: 'Class 3 (ND1)', n4: 'Class 3 (ND1)' },
                { feature: 'Distinguishing DDx', n3: 'WES: 2q33.1',   n4: 'WES: 6q16.3' },
              ].map(r => (
                <tr key={r.feature}>
                  <td className="text-muted">{r.feature}</td>
                  <td className="fw-bold" style={{ color: COLOR }}>{r.n3}</td>
                  <td>{r.n4}</td>
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
export default function NDUFAF3Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufaf3/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufaf3/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufaf3/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 NDUFAF3 — Complex I Deficiency (MC1DN19)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>ND1-Module Assembly</span>
        <span className="badge ms-1 bg-danger">No Riboflavin Response</span>
        <span className="badge ms-1" style={{ background: '#1565c0' }}>NDUFAF4 Obligate Heterodimer</span>
        <span className="badge ms-1" style={{ background: '#e65100' }}>Earliest CI Assembly Complex</span>
        <span className="badge ms-1 bg-secondary">2q33.1</span>
        <span className="badge ms-1 bg-secondary">OMIM *612911</span>
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
