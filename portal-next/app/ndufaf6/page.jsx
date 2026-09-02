'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Dioxygenase Activity', 'Definitions'];
const COLOR = '#1a237e';   // deep indigo — dioxygenase/hydroxylase/late-stage CI maturation
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
  const mod = data.ndufaf6_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="HCM Rate"         value="&lt;5-10% (Very Low)"       color="#b71c1c" />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: '#e8eaf6', borderLeft: `4px solid ${COLOR}` }}>
        <strong>🔵 NDUFAF6 (C8orf38) — ONLY CI Assembly Factor with 2OG-Fe(II) Oxygenase/Hydroxylase Activity</strong> —
        NDUFAF6 uniquely catalyzes post-translational hydroxylation of the NDUFS7/PSST subunit (Q-module) using
        2-oxoglutarate and Fe(II) as cofactors. This covalent modification is required for Q-module maturation
        and late-stage CI assembly. All other CI assembly factors are chaperones, scaffolds, or structural factors —
        none catalyze a PTM of a CI subunit. 8q22.1, 369aa, ~42kDa, 2OG-dioxygenase superfamily (jelly-roll beta-barrel fold).
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9 (50-60% Riboflavin-Responsive)</strong> —
        NDUFAF6 is a 2OG-dioxygenase with NO FAD domain. Riboflavin supplementation CANNOT rescue the
        NDUFS7/PSST hydroxylation defect. Any riboflavin response in a CI patient points to ACAD9 (MCIA/ND2-ND5
        module), not NDUFAF6. Riboflavin trial + WES (8q22.1 vs 3q21.3) is the mandatory discriminator.
        NDUFAF6 (8q22.1) vs NDUFAF5 (20p12.1) — both no-FAD, no riboflavin; WES is mandatory.
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🔵 Very Low HCM (&lt;5-10%) — Critical DDx Marker</strong> —
        NDUFAF6 shows very low HCM, contrasting sharply with TIMMDC1 (&gt;80%), NDUFV2 (~80%),
        ACAD9 (55-65%), SCO2 (65%). High HCM (&gt;60%) in a CI patient points strongly away from NDUFAF6.
        Late-stage Q-module assembly defect; BN-PAGE shows Q-module maturation intermediates stalled.
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔄 Key Pathway Note — 2OG-Dioxygenase / Q-Module / No Riboflavin / Late-Stage CI Assembly">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      {mod.gene && (
        <SectionCard title="⚙️ NDUFAF6 Module Summary — 2OG-Fe(II) Dioxygenase / Q-Module Late-Stage CI Assembly">
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
            <div className="small text-muted mb-1">ONLY CI assembly factor with 2OG-Fe(II) oxygenase/hydroxylase activity</div>
            <p className="small mb-1">{mod.unique_2og_dioxygenase}</p>
            <div className="small text-muted mb-1">NDUFAF6 vs FOXRED1 — 2OG-dioxygenase vs FAD-oxidoreductase</div>
            <p className="small mb-1">{mod.ndufaf6_vs_foxred1}</p>
            <div className="small text-muted mb-1">NDUFAF6 vs NUBPL — hydroxylase vs [4Fe-4S] delivery</div>
            <p className="small mb-1">{mod.ndufaf6_vs_nubpl}</p>
            <div className="small text-muted mb-1">NDUFAF6 vs ACAD9 — no riboflavin vs riboflavin-responsive</div>
            <p className="small mb-0">{mod.ndufaf6_vs_acad9}</p>
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
                k === 'HCM_rate' ? 'text-warning fw-bold' :
                k === '2OG_dioxygenase_unique' ? 'text-primary fw-bold' :
                k === 'Late_stage_assembly' ? 'text-info fw-bold' :
                k.startsWith('Complex_') ? 'text-success' : 'fw-bold'
              }
            >{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-695)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v}
            color={
              k === 'HCM' ? '#b71c1c' :           // dark red — very low HCM
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
  if (!data) return <p className="text-muted">Loading…</p>;

  const pts = data.patients || [];
  const filtered = pts.filter(p =>
    (p.allele1 || '').toLowerCase().includes(search.toLowerCase()) ||
    (p.allele2 || '').toLowerCase().includes(search.toLowerCase()) ||
    (p.outcome || '').toLowerCase().includes(search.toLowerCase())
  );

  const od = data.outcome_distribution || {};

  return (
    <>
      <div className="row g-3 mb-4">
        {Object.entries(od).map(([outcome, count]) => (
          <div key={outcome} className="col-6 col-md-3">
            <div className="card shadow-sm text-center">
              <div className="card-body py-2">
                <div className="fw-bold fs-5" style={{ color: outcome.includes('deceased') ? '#c62828' : COLOR }}>{count}</div>
                <div className="text-muted small">{outcome.replace(/_/g, ' ')}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="📊 CI Activity Distribution">
            {Object.entries(data.ci_activity_stats?.bands || {}).map(([band, count]) => (
              <Bar key={band} label={band} value={Math.round(count / (data.cohort_n || 40) * 100)} />
            ))}
            <div className="small text-muted mt-2">
              Mean: {data.ci_activity_stats?.mean}% | Min: {data.ci_activity_stats?.min}% | Max: {data.ci_activity_stats?.max}%
            </div>
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="⏱️ Onset Distribution">
            {Object.entries(data.onset_stats?.bands || {}).map(([band, count]) => (
              <Bar key={band} label={band} value={Math.round(count / (data.cohort_n || 40) * 100)} />
            ))}
            <div className="small text-muted mt-2">
              Mean onset: {data.onset_stats?.mean_months} months
            </div>
          </SectionCard>
        </div>
      </div>

      <SectionCard title="🧬 Variant Frequency">
        {Object.entries(data.variant_frequency || {}).map(([v, count]) => (
          <div key={v} className="d-flex justify-content-between border-bottom py-1 small">
            <span className={`font-monospace ${v.includes('Arg85Trp') ? 'fw-bold text-danger' : v.includes('Gly59Arg') ? 'fw-bold text-danger' : 'text-muted'}`}>{v}</span>
            <span className="fw-bold">{count}{v.includes('Arg85Trp') ? ' ← active site' : ''}{v.includes('Leu213Pro') ? ' ← helix-breaking' : ''}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔑 2OG-Dioxygenase Key Features">
        {Object.entries(data.key_2og_dioxygenase_features || {}).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span className={
              v === true ? 'text-success fw-bold' :
              v === false ? 'text-danger' :
              typeof v === 'number' ? (v < 15 ? 'text-info fw-bold' : 'fw-bold') :
              'fw-bold'
            }>{String(v)}{typeof v === 'number' && k.includes('HCM') ? '%' : ''}</span>
          </div>
        ))}
      </SectionCard>

      <div className="row mb-3">
        <div className="col-md-6">
          <input className="form-control form-control-sm" placeholder="Filter by allele / outcome…"
            value={search} onChange={e => setSearch(e.target.value)} />
        </div>
      </div>

      {filtered.length > 0 && (
        <div className="table-responsive">
          <table className="table table-sm table-hover small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>#</th><th>Onset (m)</th><th>Sex</th><th>Allele 1</th><th>Allele 2</th>
                <th>CI %</th><th>Leigh MRI</th><th>Lactic Ac.</th><th>HCM</th><th>Seizures</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.onset_age_months}</td>
                  <td>{p.sex}</td>
                  <td className={`font-monospace ${p.allele1 === 'p.Arg85Trp' ? 'fw-bold text-danger' : ''}`} style={{ fontSize: '0.7rem' }}>{p.allele1}</td>
                  <td className={`font-monospace ${p.allele2 === 'p.Arg85Trp' ? 'fw-bold text-danger' : ''}`} style={{ fontSize: '0.7rem' }}>{p.allele2}</td>
                  <td><span className="badge bg-danger">{p.ci_activity_pct}%</span></td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.lactic_acidosis ? '✅' : '—'}</td>
                  <td>{p.hcm ? <span className="badge" style={{ background: '#b71c1c', fontSize: '0.6rem' }}>HCM</span> : '—'}</td>
                  <td>{p.seizures ? '⚠️' : '—'}</td>
                  <td><span className="badge" style={{ background: p.outcome?.includes('deceased') ? '#c62828' : COLOR, fontSize: '0.65rem' }}>{p.outcome}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <SectionCard title="💊 Treatment Summary" borderColor="#2e7d32">
        {Object.entries(data.treatment_summary || {}).map(([k, v]) => (
          <div key={k} className="mb-2 border-bottom pb-1">
            <div className="small fw-bold" style={{ color: k === 'absolute_ci' || k === 'do_not_use' ? '#c62828' : k === 'avoid' ? '#e65100' : k === 'diagnostic_priority' ? '#1565c0' : '#2e7d32' }}>
              {k.replace(/_/g, ' ').toUpperCase()}
            </div>
            <div className="small text-muted">{Array.isArray(v) ? v.join(' · ') : v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🧪 Variant Table" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>cDNA</th><th>Protein</th><th>Domain</th><th>Severity</th><th>CI Range</th><th>Notes</th></tr>
            </thead>
            <tbody>
              {(data.variant_table || []).map(v => (
                <tr key={v.hgvs_p} style={{ background: v.hgvs_p === 'p.Arg85Trp' ? '#ffebee' : undefined }}>
                  <td className="font-monospace small">{v.hgvs_c}</td>
                  <td className={`font-monospace small fw-bold ${v.hgvs_p === 'p.Arg85Trp' ? 'text-danger' : ''}`}>{v.hgvs_p}</td>
                  <td className="text-muted small">{v.domain}</td>
                  <td><span className="badge" style={{ background: v.severity === 'severe' ? '#c62828' : v.severity === 'moderate' ? '#f57f17' : '#1565c0', fontSize: '0.65rem' }}>{v.severity}</span></td>
                  <td className="text-muted small">{v.ci_range}</td>
                  <td className="text-muted small">{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Dioxygenase Activity ─────────────────────────────────────────────────
function DioxygenaseTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const mod = data.ndufaf6_module_summary || {};

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>NDUFAF6 (C8orf38) — ONLY CI Assembly Factor with 2OG-Fe(II) Oxygenase/Hydroxylase Activity</strong><br />
        <span className="small text-muted">
          NDUFAF6 is unique among all known CI assembly factors: it is the sole factor that catalyzes a
          post-translational modification (hydroxylation) of a CI subunit. Using 2-oxoglutarate (2OG) as
          co-substrate and Fe(II) as cofactor (jelly-roll beta-barrel fold), NDUFAF6 hydroxylates
          NDUFS7/PSST (Q-module subunit) at a leucyl or asparaginyl residue. This covalent PTM is
          required for Q-module maturation and late-stage CI assembly. Loss of NDUFAF6 stalls CI assembly
          at the Q-module maturation step. BN-PAGE: late-stage Q-module intermediates accumulate —
          distinct from N-module (FOXRED1/NUBPL), MCIA/ND2-ND5 (ACAD9/NDUFAF1/ECSIT/TMEM126B),
          and ND1-module/Class3 (NDUFAF3/4/5/TIMMDC1) stalling patterns.
        </span>
      </div>

      {mod.gene && (
        <SectionCard title="⚙️ NDUFAF6 2OG-Dioxygenase Mechanism Detail" borderColor={COLOR}>
          <div className="alert mb-3" style={{ background: '#e8eaf6', borderLeft: `4px solid ${COLOR}` }}>
            <strong>Gene: {mod.gene}</strong>
          </div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="small text-muted mb-1">Module class</div>
              <div className="fw-bold small">{mod.module_class}</div>
            </div>
            <div className="col-md-6">
              <div className="small text-muted mb-1">Assembly position</div>
              <div className="fw-bold small">{mod.assembly_position}</div>
            </div>
          </div>
          <div className="small text-muted mb-1">2OG-Fe(II) oxygenase activity — unique CI assembly factor</div>
          <p className="small mb-2">{mod.unique_2og_dioxygenase}</p>
          <div className="small text-muted mb-1">NDUFAF6 vs FOXRED1</div>
          <p className="small mb-2">{mod.ndufaf6_vs_foxred1}</p>
          <div className="small text-muted mb-1">NDUFAF6 vs NUBPL</div>
          <p className="small mb-2">{mod.ndufaf6_vs_nubpl}</p>
          <div className="small text-muted mb-1">NDUFAF6 vs ACAD9</div>
          <p className="small mb-0">{mod.ndufaf6_vs_acad9}</p>
        </SectionCard>
      )}

      <SectionCard title="🔬 DDx Matrix — NDUFAF6 vs Key CI Genes" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Comparator</th><th>NDUFAF6</th><th>Comparator</th><th>Key Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map((row, i) => (
                <tr key={i} style={{ background: row.comparator.includes('ACAD9') ? '#fff8e1' : row.comparator.includes('NDUFAF5') ? '#f3e5f5' : undefined }}>
                  <td className="fw-bold small">{row.comparator}</td>
                  <td className="text-muted small">{row.ndufaf6}</td>
                  <td className="text-muted small">{row.comparator_val}</td>
                  <td className="fw-bold small" style={{ color: '#1565c0' }}>{row.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ CI Assembly — NDUFAF6 2OG-Dioxygenase vs All Other CI Assembly Factors" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#fff8e1', borderLeft: '4px solid #f57f17' }}>
          <strong>NDUFAF6 is the ONLY CI assembly factor with 2OG-Fe(II) oxygenase/hydroxylase activity.</strong>{' '}
          All other factors are chaperones, scaffolds, or Fe-S delivery factors. NDUFAF6 catalyzes a covalent
          PTM of NDUFS7/PSST — a post-translational modification unique in CI assembly biology.
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>Factor/Gene</th><th>Module</th><th>Mechanism</th><th>Riboflavin Response</th><th>HCM</th><th>Chromosome</th>
              </tr>
            </thead>
            <tbody>
              {[
                { factor: 'NDUFAF6', module: 'Q-module / late-stage', mech: '2OG-Fe(II) dioxygenase hydroxylase — NDUFS7/PSST PTM', ribo: '0%', hcm: '<5-10%', chr: '8q22.1', highlight: true },
                { factor: 'FOXRED1', module: 'N-module (chaperone)', mech: 'FAD-oxidoreductase protein chaperone (N-module protein folding)', ribo: '0%', hcm: '~10%', chr: '11q24.2' },
                { factor: 'NUBPL', module: 'N-module ([4Fe-4S] delivery)', mech: 'P-loop GTPase Fe-S carrier (IND1) — [4Fe-4S] delivery to NDUFS1/NDUFV1', ribo: '0%', hcm: '~25%', chr: '14q12' },
                { factor: 'ACAD9', module: 'MCIA/ND2-ND5 (Class 1)', mech: 'FAD-binding MCIA scaffold — riboflavin-responsive', ribo: '50-60% (Level B)', hcm: '55-65%', chr: '3q21.3' },
                { factor: 'NDUFAF1', module: 'MCIA/ND2-ND5 (Class 1)', mech: 'CIA30 obligate ACAD9 binary partner', ribo: '0%', hcm: '20-30%', chr: '15q11.2-q13' },
                { factor: 'ECSIT', module: 'MCIA/ND2-ND5 (Class 1)', mech: 'TMEM126B-recruiting scaffold; dual innate immunity/CI role', ribo: '0%', hcm: '<20%', chr: '19p13.3' },
                { factor: 'TMEM126B', module: 'MCIA/ND2-ND5 (Class 1)', mech: 'Integral IMM (2-TM) terminal MCIA member; ECSIT-recruited', ribo: '0%', hcm: '<20%', chr: '11q14.1' },
                { factor: 'TIMMDC1', module: 'ND1-module (Class 3)', mech: 'Integral IMM ND1-module scaffold (2-TM helices)', ribo: '0%', hcm: '>80%', chr: '3q25.1' },
                { factor: 'NDUFAF3', module: 'ND1-module (Class 3)', mech: 'NDUFAF4-obligate heterodimer; earliest ND1 CI complex', ribo: '0%', hcm: '15-25%', chr: '2q33.1' },
                { factor: 'NDUFAF5', module: 'ND1-module (Class 3)', mech: 'Independent ND1-module actor; NDUFAF3/4 protein normal', ribo: '0%', hcm: '<20%', chr: '20p12.1' },
              ].map(r => (
                <tr key={r.factor} style={{ background: r.highlight ? LIGHT : undefined }}>
                  <td className="fw-bold" style={{ color: r.highlight ? COLOR : undefined }}>
                    {r.factor}
                    {r.highlight && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>THIS</span>}
                  </td>
                  <td className="small text-muted">{r.module}</td>
                  <td className="small text-muted">{r.mech}</td>
                  <td>
                    <span className="badge" style={{
                      background: r.ribo.includes('50-60%') ? '#1565c0' : '#c62828',
                      fontSize: '0.65rem'
                    }}>{r.ribo}</span>
                  </td>
                  <td className={r.hcm.includes('>80%') ? 'text-danger fw-bold' : r.hcm.includes('55-65%') ? 'fw-bold' : r.hcm.includes('<5-10%') ? 'text-info fw-bold' : ''}>{r.hcm}</td>
                  <td className="font-monospace small">{r.chr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔍 2OG-Fe(II)-Dioxygenase Reaction — NDUFAF6 Catalytic Mechanism" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>NDUFAF6 catalyzes:</strong> NDUFS7/PSST + 2-oxoglutarate + O₂ → NDUFS7/PSST-OH + succinate + CO₂
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>Component</th><th>Role in NDUFAF6 Reaction</th><th>Effect if Absent</th></tr>
            </thead>
            <tbody>
              {[
                { comp: '2-Oxoglutarate (2OG)', role: 'Co-substrate; oxidatively decarboxylated to succinate during reaction; provides energy for oxygen activation', eff: 'No substrate for oxygenation; Fe(IV)=O intermediate cannot form; NDUFS7 hydroxylation fails' },
                { comp: 'Fe(II) cofactor', role: 'Coordinated in HXD...H facial triad of jelly-roll beta-barrel; required for O₂ activation to Fe(IV)=O intermediate', eff: 'p.Arg85Trp disrupts Fe(II) coordination → no Fe(IV)=O → no NDUFS7 hydroxylation' },
                { comp: 'O₂ (molecular oxygen)', role: 'Oxidant; activated to Fe(IV)=O (ferryl) intermediate; inserts one O atom into NDUFS7 substrate', eff: 'No reaction possible; anaerobic conditions block NDUFAF6 function' },
                { comp: 'NDUFS7/PSST (substrate)', role: 'Q-module CI subunit; Leu or Asn residue is hydroxylated by NDUFAF6; required for Q-module integration', eff: 'Without NDUFAF6 action: NDUFS7 cannot integrate into Q-module; late-stage CI assembly stalls' },
                { comp: 'Jelly-roll beta-barrel fold', role: 'Core structural scaffold of 2OG-dioxygenase superfamily; coordinates Fe(II) and 2OG; positions substrate', eff: 'p.Gly59Arg (core glycine) or p.Leu213Pro (helix-breaking) disrupt fold → complete loss of activity' },
                { comp: 'C-terminal domain', role: 'Substrate (NDUFS7) recognition and binding interface', eff: 'p.Trp320Ter truncates C-terminal substrate-binding domain → NDUFS7 cannot be recognized' },
              ].map(r => (
                <tr key={r.comp}>
                  <td className="fw-bold small" style={{ color: COLOR }}>{r.comp}</td>
                  <td className="text-muted small">{r.role}</td>
                  <td className="small text-danger">{r.eff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🧬 Riboflavin Test — The Key NDUFAF6 vs ACAD9 Discriminator" borderColor="#2e7d32">
        <div className="row g-3">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#2e7d32' }}>ACAD9 — FAD-Binding MCIA Scaffold (Riboflavin-Responsive)</div>
            <ul className="list-unstyled small">
              <li>• FAD domain present — flavin adenine dinucleotide binding</li>
              <li>• Riboflavin supplementation increases FAD availability</li>
              <li>• Increased FAD → more functional ACAD9 → improved MCIA formation</li>
              <li>• <strong className="text-success">50-60% riboflavin response rate (Level B evidence)</strong></li>
              <li>• HCM 55-65%; MCIA/ND2-ND5 module; 3q21.3</li>
              <li>• Riboflavin trial POSITIVE → strongly favors ACAD9</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#c62828' }}>NDUFAF6 — 2OG-Fe(II) Dioxygenase (No Riboflavin Response)</div>
            <ul className="list-unstyled small">
              <li>• NO FAD domain — no flavin binding whatsoever</li>
              <li>• Uses 2OG + Fe(II), not FAD, as cofactors</li>
              <li>• Riboflavin supplementation has no biochemical pathway to NDUFAF6</li>
              <li>• <strong className="text-danger">0% riboflavin response (no FAD domain to saturate)</strong></li>
              <li>• HCM &lt;5-10%; Q-module late stage; 8q22.1</li>
              <li>• Riboflavin trial NEGATIVE → pursue WES 8q22.1 locus</li>
            </ul>
          </div>
        </div>
        <div className="alert mt-3 mb-0" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
          <strong>Clinical Action:</strong> Infant with isolated CI deficiency + Leigh syndrome → start riboflavin trial immediately.
          Response (within 4-8 weeks) → ACAD9 leading diagnosis; continue riboflavin.
          No response → pursue WES including NDUFAF6 (8q22.1), NDUFAF5 (20p12.1), FOXRED1 (11q24.2), NUBPL (14q12), TIMMDC1 (3q25.1).
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <>
      <SectionCard title="📖 Gene & Disease Definition">
        <p className="small mb-2">{data.gene_definition}</p>
        <p className="small mb-0">{data.disease_definition}</p>
      </SectionCard>

      <SectionCard title="🔬 Module Definitions">
        {(data.module_definitions || []).map(c => (
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
              {(data.clinical_thresholds || []).map(t => (
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

      <SectionCard title="🧬 Inheritance">
        <p className="small mb-0">{data.inheritance_definition}</p>
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
export default function NDUFAF6Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufaf6/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufaf6/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufaf6/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 NDUFAF6 (C8orf38) — Complex I Deficiency MC1DN26 (2OG-Dioxygenase / Late-Stage CI Assembly)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>2OG-Fe(II)-Dioxygenase</span>
        <span className="badge ms-1" style={{ background: '#c62828' }}>Only CI Hydroxylase</span>
        <span className="badge ms-1" style={{ background: '#b71c1c' }}>HCM &lt;5-10% (Very Low)</span>
        <span className="badge ms-1 bg-secondary">8q22.1</span>
        <span className="badge ms-1 bg-secondary">OMIM *612392</span>
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
      {tab === 2 && <DioxygenaseTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
