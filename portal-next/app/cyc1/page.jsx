'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Biochemistry', 'DDx & Treatment', 'Definitions'];
const COLOR  = '#1b5e20';   // deep green — CYC1 / heme c core structural subunit
const LIGHT  = '#e8f5e9';
const COLOR2 = '#2e7d32';
const COLOR3 = '#b71c1c';   // crimson — severity / contraindications
const COLOR4 = '#4527a0';   // deep purple — key DDx

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
          🧬 {data.gene} — {data.disease}
        </h5>
        <div className="row g-2 small">
          <div className="col-md-4"><strong>Gene:</strong> {data.gene} ({data.alias}) · OMIM *{data.omim_gene}</div>
          <div className="col-md-4"><strong>Disease OMIM:</strong> #{data.omim_disease} · {data.disease}</div>
          <div className="col-md-4"><strong>Chromosome:</strong> {data.chromosome} · {data.inheritance}</div>
          <div className="col-md-6"><strong>Protein:</strong> {data.protein_size}</div>
          <div className="col-md-6"><strong>Complex:</strong> {data.complex}</div>
          <div className="col-12 mt-1"><strong>Function:</strong> {data.function}</div>
        </div>
      </div>

      {/* KPI row */}
      <div className="row mb-4">
        <KPI label="Cohort (n)" value={data.cohort_n} />
        <KPI label="Neonatal onset" value={`${s.neonatal_onset_pct}%`} color={COLOR3} />
        <KPI label="Hypotonia" value={`${s.hypotonia_pct}%`} />
        <KPI label="Avg CIII activity" value={`${s.avg_ciii_activity_pct}%`} color={COLOR2} />
        <KPI label="Avg Lactate (mM)" value={s.avg_lactic_acid_mmolL} color={COLOR2} />
        <KPI label="Deceased" value={`${s.deceased_pct}%`} color={COLOR3} />
      </div>

      {/* Clinical features */}
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Clinical Features (% of cohort)">
            {feats.map((f, i) => (
              <Bar key={i} label={f.feature} value={f.pct}
                color={f.pct === 0 ? '#9e9e9e' : (f.pct < 20 ? '#9e9e9e' : (f.pct < 50 ? COLOR2 : COLOR))} />
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Top Variants (allele frequency)">
            {(data.top_variant_counts || []).map((v, i) => (
              <div key={i} className="d-flex justify-content-between small py-1 border-bottom">
                <code>{v.variant}</code>
                <span className="badge" style={{ background: COLOR }}>{v.count} alleles</span>
              </div>
            ))}
          </SectionCard>
          <SectionCard title="🔑 CYC1 — Critical DDx vs LYRM7/BCS1L" borderColor={COLOR4}>
            <ul className="small mb-0 ps-3">
              <li><strong>Immunoblot UQCRC1:</strong> <span style={{ color: COLOR3 }}>ABSENT in CYC1</span> · <span style={{ color: COLOR }}>PRESERVED in LYRM7/BCS1L</span> — KEY distinguisher</li>
              <li><strong>BN-PAGE CIII:</strong> Absent (CYC1) · Reduced (LYRM7) · Precomplex (BCS1L)</li>
              <li><strong>Hepatic involvement:</strong> ~78% in CYC1 · uncommon in LYRM7/BCS1L</li>
              <li><strong>Cataracts:</strong> ~35% in CYC1 · absent in LYRM7/BCS1L — distinctive</li>
              <li><strong>Mechanism:</strong> CYC1 = structural subunit (holocomplex loss) vs LYRM7/BCS1L = chaperone/assembly factor (core preserved)</li>
            </ul>
          </SectionCard>
        </div>
      </div>

      {/* Clinical alerts */}
      <SectionCard title="⚠️ Clinical Alerts — Contraindications & Safety" borderColor={COLOR3}>
        <div className="row">
          {(data.key_clinical_alerts || []).map((a, i) => (
            <div key={i} className="col-md-6 mb-2">
              <div className="p-2 rounded small" style={{
                background: a.startsWith('🚫') ? '#fff3e0' : a.startsWith('⚠️') ? '#fff8e1' : '#e8f5e9',
                borderLeft: `3px solid ${a.startsWith('🚫') ? COLOR3 : a.startsWith('⚠️') ? '#f9a825' : COLOR}`
              }}>
                {a}
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Sample patients */}
      <SectionCard title="Sample Patient Records (first 10)">
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0">
            <thead><tr>
              <th>ID</th><th>Sex</th><th>Onset (mo)</th><th>Dx (mo)</th><th>Origin</th>
              <th>Variant 1</th><th>Variant 2</th><th>CIII%</th><th>Lactate</th><th>Outcome</th>
            </tr></thead>
            <tbody>
              {(data.patients || []).map((p, i) => (
                <tr key={i}>
                  <td><code>{p.id}</code></td>
                  <td>{p.sex}</td>
                  <td>{p.age_onset_months}</td>
                  <td>{p.age_dx_months}</td>
                  <td>{p.origin}</td>
                  <td><code>{p.variant_allele1}</code></td>
                  <td><code>{p.variant_allele2}</code></td>
                  <td><span style={{ color: p.ciii_activity_pct < 8 ? COLOR3 : COLOR }}>{p.ciii_activity_pct}%</span></td>
                  <td><span style={{ color: p.lactic_acid_mmolL > 10 ? COLOR3 : COLOR }}>{p.lactic_acid_mmolL}</span></td>
                  <td><span className="badge" style={{ background: p.outcome.includes('Deceased') ? COLOR3 : COLOR, fontSize: '0.7em' }}>{p.outcome.substring(0,25)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Biochemistry ──────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const bio = data.biochemistry_distribution || {};
  const imm = data.immunoblot_pattern || {};
  const bn  = data.bn_page_pattern || {};
  const gc  = data.genetic_counselling || {};

  return (
    <div>
      <SectionCard title="Pathogenic Variants in CYC1">
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead className="table-success">
              <tr>
                <th>Protein</th><th>cDNA</th><th>Domain</th><th>Type</th>
                <th>Severity</th><th>Penetrance</th><th>Mechanism</th>
              </tr>
            </thead>
            <tbody>
              {(data.all_variants || []).map((v, i) => (
                <tr key={i}>
                  <td><code>{v.protein}</code></td>
                  <td><code>{v.cdna}</code></td>
                  <td>{v.domain}</td>
                  <td>{v.type}</td>
                  <td><span className="badge" style={{ background: v.severity.includes('Severe') ? COLOR3 : (v.severity.includes('Intermediate') ? '#f9a825' : COLOR) }}>{v.severity}</span></td>
                  <td>{v.penetrance_pct}%</td>
                  <td className="small text-muted">{v.mechanism}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Biochemistry Distribution">
            <p className="small mb-2">Avg CIII activity: <strong>{bio.avg_ciii_activity_pct}%</strong> · Avg lactate: <strong>{bio.avg_lactic_acid_mmolL} mM</strong></p>
            <Bar label="CIII activity <5%" value={bio.ciii_below_15_pct || 0} color={COLOR3} />
            <Bar label="CIII activity 5-10%" value={bio.ciii_15to25_pct || 0} color="#e53935" />
            <Bar label="CIII activity >10%" value={bio.ciii_above_25_pct || 0} color={COLOR} />
            <hr />
            <Bar label="Lactate >15 mM" value={bio.lactic_above_10_pct || 0} color={COLOR3} />
            <Bar label="Lactate 8-15 mM" value={bio.lactic_6_to10_pct || 0} color="#e53935" />
            <Bar label="Lactate <8 mM" value={bio.lactic_below_6_pct || 0} color={COLOR} />
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="BN-PAGE Pattern">
            <p className="small"><strong>Finding:</strong> {bn.finding}</p>
            <p className="small"><strong>Interpretation:</strong> {bn.interpretation}</p>
            <p className="small mb-0"><strong>DDx value:</strong> {bn.ddx_value}</p>
          </SectionCard>
          <SectionCard title="Immunoblot Pattern">
            {Object.entries(imm).map(([k, v], i) => (
              <div key={i} className="d-flex justify-content-between small py-1 border-bottom">
                <strong>{k.replace(/_/g, ' ')}</strong>
                <span style={{ color: v.includes('ABSENT') ? COLOR3 : COLOR }} className="fw-bold">{v.substring(0, 65)}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Outcome Distribution">
            {(data.outcome_distribution || []).map((o, i) => (
              <div key={i} className="d-flex justify-content-between small py-1 border-bottom">
                <span>{o.outcome}</span>
                <span className="badge" style={{ background: o.outcome.includes('Deceased') ? COLOR3 : COLOR }}>{o.count}</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Genetic Counselling">
            {Object.entries(gc).map(([k, v], i) => (
              <div key={i} className="mb-2 small">
                <strong>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}:</strong>
                <span className="text-muted ms-1">{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>
    </div>
  );
}

// ── Tab: DDx & Treatment ──────────────────────────────────────────────────────
function DDxTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const abs = data.absolute_contraindications || [];
  const rel = data.relative_contraindications || [];
  const tx  = data.recommended_treatments || [];
  const ddx = data.key_ddx || [];

  return (
    <div>
      <SectionCard title="Differential Diagnosis" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered mb-0">
            <thead><tr><th>Condition</th><th>Distinguishing from CYC1</th></tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i}>
                  <td><strong style={{ color: COLOR4 }}>{d.condition}</strong></td>
                  <td className="small">{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <div className="row">
        <div className="col-md-6">
          <SectionCard title="🚫 Absolute Contraindications" borderColor={COLOR3}>
            {abs.map((a, i) => (
              <div key={i} className="p-2 mb-2 rounded small" style={{ background: '#fff3e0', borderLeft: `3px solid ${COLOR3}` }}>{a}</div>
            ))}
          </SectionCard>
          {rel.length > 0 && (
            <SectionCard title="⚠️ Relative Contraindications / Cautions" borderColor="#f9a825">
              {rel.map((r, i) => (
                <div key={i} className="p-2 mb-2 rounded small" style={{ background: '#fff8e1', borderLeft: `3px solid #f9a825` }}>{r}</div>
              ))}
            </SectionCard>
          )}
        </div>
        <div className="col-md-6">
          <SectionCard title="✅ Recommended Treatments" borderColor={COLOR}>
            {tx.map((t, i) => (
              <div key={i} className="p-2 mb-2 rounded small" style={{ background: LIGHT, borderLeft: `3px solid ${COLOR}` }}>{t}</div>
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="CIII Assembly Cascade — CYC1 Position" borderColor={COLOR2}>
        <div className="row text-center small">
          {[
            { step: '1', label: 'UQCC1-UQCC2 scaffold', detail: 'Bind nascent MT-CYB · CIII★ (20q11.22 / 6p21.2)', color: '#1565c0' },
            { step: '2', label: 'UQCC3 cooperates', detail: '1-TM-Helix IMM anchor (11q12.3)', color: '#2e7d32' },
            { step: '3', label: 'CYC1 integrates', detail: 'Core structural subunit · heme c · 8q24.13', color: COLOR },
            { step: '4', label: 'TTC19 stabilises', detail: 'TPR scaffold intermediate (17p12)', color: '#4a148c' },
            { step: '5', label: 'LYRM7 chaperones RISP', detail: 'LYR motif · UQCRFS1 (5q33.1)', color: '#f57f17' },
            { step: '6', label: 'BCS1L inserts RISP', detail: 'Rate-limiting Qo-site insertion (2q35)', color: COLOR3 },
          ].map((s, i) => (
            <div key={i} className="col-md-2 mb-2">
              <div className="rounded p-2" style={{ background: s.color, color: '#fff' }}>
                <div className="fw-bold">Step {s.step}</div>
                <div>{s.label}</div>
                <div style={{ fontSize: '0.7em', opacity: 0.85 }}>{s.detail}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="alert mt-3 mb-0 small" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR3}` }}>
          <strong>CYC1 block effect:</strong> CYC1 absent → ENTIRE CIII holocomplex destabilised and degraded →
          ALL core subunits (UQCRC1, UQCRC2, RISP, MT-CYB) absent on immunoblot →
          BN-PAGE: CIII2 and supercomplexes completely absent.
          Distinguish from LYRM7/BCS1L: UQCRC1 (Core1) preserved in LYRM7/BCS1L, ABSENT in CYC1.
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const p = data.protein || {};

  return (
    <div>
      <div className="row">
        <div className="col-md-6">
          <SectionCard title="Gene & Disease Reference">
            <table className="table table-sm mb-0">
              <tbody>
                <tr><td><strong>Gene</strong></td><td>{data.gene} ({data.alias})</td></tr>
                <tr><td><strong>Full name</strong></td><td>{data.full_name}</td></tr>
                <tr><td><strong>OMIM Gene</strong></td><td>*{data.omim_gene}</td></tr>
                <tr><td><strong>OMIM Disease</strong></td><td>#{data.omim_disease} — {data.disease_name}</td></tr>
                <tr><td><strong>Chromosome</strong></td><td>{data.chromosome}</td></tr>
                <tr><td><strong>Inheritance</strong></td><td>{data.inheritance}</td></tr>
                <tr><td><strong>CIII role</strong></td><td>{data.ciii_assembly_step}</td></tr>
              </tbody>
            </table>
          </SectionCard>
          <SectionCard title="Protein Details">
            <table className="table table-sm mb-0">
              <tbody>
                <tr><td><strong>Size</strong></td><td>{p.size_aa} aa, {p.kDa} kDa</td></tr>
                <tr><td><strong>TM helices</strong></td><td>{p.tm_helices} (C-terminal IMM anchor)</td></tr>
                <tr><td><strong>Localization</strong></td><td>{p.localization}</td></tr>
                <tr><td><strong>Partners</strong></td><td>{p.partner}</td></tr>
                <tr><td><strong>Function</strong></td><td>{p.function}</td></tr>
              </tbody>
            </table>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="Key Biochemical Features">
            <ul className="small ps-3 mb-0">
              {(data.key_biochemical_features || []).map((f, i) => <li key={i} className="mb-1">{f}</li>)}
            </ul>
          </SectionCard>
          <SectionCard title="BN-PAGE Pattern">
            <p className="small mb-0">{data.bn_page}</p>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="Key References">
        <ul className="small ps-3 mb-0">
          {(data.key_references || []).map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ul>
      </SectionCard>

      <SectionCard title="Glossary">
        {(data.terms || []).map((t, i) => (
          <div key={i} className="mb-2 pb-2 border-bottom small">
            <strong style={{ color: COLOR }}>{t.term}</strong>
            <span className="text-muted ms-2">— {t.definition}</span>
          </div>
        ))}
      </SectionCard>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function CYC1Page() {
  const [tab, setTab]     = useState(0);
  const [overview, setOv] = useState(null);
  const [bkdown, setBk]   = useState(null);
  const [defs, setDefs]   = useState(null);
  const [err, setErr]     = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/cyc1/overview`).then(r => r.json()),
      fetch(`${API}/api/cyc1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/cyc1/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOv(ov); setBk(bk); setDefs(df); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div className="container-fluid py-4">
      <div className="mb-3" style={{ borderBottom: `3px solid ${COLOR}` }}>
        <h4 className="fw-bold" style={{ color: COLOR }}>
          🧬 CYC1 — Complex III Deficiency, Nuclear Type 3 (CIII-D3)
        </h4>
        <p className="text-muted mb-2 small">
          OMIM Gene *123980 · Disease #615158 · AR biallelic · 8q24.13 · 40-patient cohort (seed 725)
        </p>
        <ul className="nav nav-tabs border-0">
          {TABS.map((t, i) => (
            <li key={i} className="nav-item">
              <button
                className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
                style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
                onClick={() => setTab(i)}
              >{t}</button>
            </li>
          ))}
        </ul>
      </div>

      {err && <div className="alert alert-danger small">API error: {err}</div>}

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={bkdown} />}
      {tab === 2 && <DDxTab data={defs} />}
      {tab === 3 && <DefsTab data={defs} />}
    </div>
  );
}
