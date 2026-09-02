'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'N-Module Fe-S Assembly', 'Definitions'];
const COLOR = '#004d40';   // deep teal — [4Fe-4S] iron-sulfur / N-module / NUBPL
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
  const mod = data.nubpl_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="HCM Rate"         value="~25% (Inter.)"              color="#e65100" />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
        <strong>🟢 EUROPEAN FOUNDER ALLELE p.Gly56Arg (c.166G>A) — ~70% of NUBPL Patients — Walker A P-loop Disruption</strong> —
        p.Gly56Arg disrupts the Walker A P-loop (GxGxxG) GTPase motif of NUBPL, impairing [4Fe-4S] cluster transfer to CI N-module.
        Found in ~70% of published NUBPL patients, typically compound heterozygous with c.815-27T>C (deep intronic branch-point, intron 9)
        missed by standard exome sequencing. European CI-deficiency patient + heterozygous p.Gly56Arg = suspect NUBPL + request deep intronic sequencing.
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9 (50-60% Riboflavin-Responsive)</strong> —
        NUBPL has no FAD domain. Riboflavin supplementation CANNOT rescue the [4Fe-4S] cluster delivery defect in NUBPL deficiency.
        Any riboflavin response in a CI patient points to ACAD9 (MCIA/ND2-ND5 module), not NUBPL.
        Riboflavin trial + WES (14q12 vs 3q21.3) is the mandatory discriminator.
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🔵 NUBPL (14q12) — Only CI-Specific [4Fe-4S] Cluster Delivery Factor — N-Module</strong> —
        NUBPL/IND1 delivers [4Fe-4S] clusters to CI N-module subunits NDUFS1 (N1b/N3/N4/N5) and NDUFV1 (N3).
        Same N-module as FOXRED1 (FAD-oxidoreductase chaperone) but different step. HCM ~25% — intermediate between FOXRED1 (~10%) and TIMMDC1 (&gt;80%).
        Deep intronic c.815-27T>C second allele is invisible to exome sequencing.
      </div>

      <SectionCard title="🧬 Gene & Protein">
        <p className="small mb-1"><strong>Full name:</strong> {data.gene_full_name}</p>
        <p className="small mb-1"><strong>Also known as:</strong> {data.also_known_as}</p>
        <p className="small mb-1"><strong>Fold / domain:</strong> {p.fold}</p>
        <p className="small mb-1"><strong>Module:</strong> {p.module}</p>
        <p className="small mb-0"><strong>Function:</strong> {p.function}</p>
      </SectionCard>

      <SectionCard title="🔄 Key Pathway Note — [4Fe-4S] Delivery / N-Module / No Riboflavin / p.Gly56Arg Founder">
        <p className="small mb-0">{data.key_pathway_note}</p>
      </SectionCard>

      {mod.gene && (
        <SectionCard title="⚙️ NUBPL Module Summary — N-Module [4Fe-4S] Cluster Delivery">
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
            <div className="small text-muted mb-1">[4Fe-4S] delivery — only CI-specific factor</div>
            <p className="small mb-1">{mod.fes_delivery_unique}</p>
            <div className="small text-muted mb-1">European founder allele p.Gly56Arg</div>
            <p className="small mb-1">{mod.european_founder_role}</p>
            <div className="small text-muted mb-1">NUBPL vs FOXRED1 — same N-module, different step</div>
            <p className="small mb-0">{mod.nubpl_vs_foxred1_same_module}</p>
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
                k === 'European_founder' ? 'text-success fw-bold' :
                k.startsWith('Complex_') ? 'text-success' : 'fw-bold'
              }
            >{v}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📊 Feature Frequencies (40-patient cohort, seed-693)">
        {Object.entries(ff).map(([k, v]) => (
          <Bar key={k} label={k.replace(/_/g, ' ')} value={v}
            color={
              k === 'HCM' ? '#e65100' :           // orange — intermediate HCM
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
            <span className={`font-monospace ${v.includes('Gly56Arg') ? 'fw-bold text-success' : 'text-muted'}`}>{v}</span>
            <span className="fw-bold">{count}{v.includes('Gly56Arg') ? ' ← European founder' : ''}</span>
          </div>
        ))}
        <div className="small text-muted mt-1">
          p.Gly56Arg (European founder): {data.gly56arg_carrier_pct}% of cohort carry this allele
        </div>
      </SectionCard>

      <SectionCard title="🔑 N-Module [4Fe-4S] Key Features">
        {Object.entries(data.key_n_module_fes_features || {}).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span className={
              v === true ? 'text-success fw-bold' :
              v === false ? 'text-danger' :
              typeof v === 'number' ? (v < 35 ? 'text-warning fw-bold' : 'fw-bold') :
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
                  <td className={`font-monospace ${p.allele1 === 'p.Gly56Arg' ? 'fw-bold text-success' : ''}`} style={{ fontSize: '0.7rem' }}>{p.allele1}</td>
                  <td className={`font-monospace ${p.allele2 === 'p.Gly56Arg' ? 'fw-bold text-success' : ''}`} style={{ fontSize: '0.7rem' }}>{p.allele2}</td>
                  <td><span className="badge bg-danger">{p.ci_activity_pct}%</span></td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.lactic_acidosis ? '✅' : '—'}</td>
                  <td>{p.hcm ? <span className="badge" style={{ background: '#e65100', fontSize: '0.6rem' }}>HCM</span> : '—'}</td>
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
                <tr key={v.hgvs_p} style={{ background: v.hgvs_p === 'p.Gly56Arg' ? '#e8f5e9' : undefined }}>
                  <td className="font-monospace small">{v.hgvs_c}</td>
                  <td className={`font-monospace small fw-bold ${v.hgvs_p === 'p.Gly56Arg' ? 'text-success' : ''}`}>{v.hgvs_p}</td>
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

// ── Tab: N-Module Fe-S Assembly ─────────────────────────────────────────────────
function FeSTabb({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>NUBPL — N-Module CI-Specific [4Fe-4S] Cluster Delivery Factor (P-loop GTPase/IND1)</strong><br />
        <span className="small text-muted">
          NUBPL/IND1 is the ONLY CI-specific [4Fe-4S] cluster delivery factor. It delivers [4Fe-4S] clusters
          to N-module subunits NDUFS1 (N1b/N3/N4/N5) and NDUFV1 (N3). Walker A P-loop GTPase domain
          (GxGxxG motif) drives cluster transfer. p.Gly56Arg European founder disrupts Walker A — the most
          common NUBPL allele (~70% of patients). Deep intronic c.815-27T>C (branch point, intron 9) is
          the frequent second allele — invisible to standard exome. Both NUBPL and FOXRED1 act on the N-module
          but at different steps (Fe-S delivery vs FAD-oxidoreductase protein chaperone). Neither has riboflavin response.
        </span>
      </div>

      <SectionCard title="🔬 DDx Matrix — NUBPL vs Key CI Genes" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Comparator</th><th>NUBPL</th><th>Comparator</th><th>Key Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map((row, i) => (
                <tr key={i} style={{ background: row.comparator.includes('FOXRED1') ? '#e0f2f1' : row.comparator.includes('ACAD9') ? '#fff8e1' : undefined }}>
                  <td className="fw-bold small">{row.comparator}</td>
                  <td className="text-muted small">{row.nubpl}</td>
                  <td className="text-muted small">{row.comparator_val}</td>
                  <td className="fw-bold small" style={{ color: '#1565c0' }}>{row.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ CI Assembly — N-Module Factor Comparison (NUBPL vs FOXRED1 vs Structural)" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#fff8e1', borderLeft: '4px solid #f57f17' }}>
          <strong>NUBPL and FOXRED1 both act on the N-module — but at different biochemical steps.</strong>{' '}
          NUBPL delivers [4Fe-4S] clusters (electron transfer cofactors). FOXRED1 is a FAD-oxidoreductase chaperone for protein folding.
          Both = 0% riboflavin response. NUBPL HCM ~25%; FOXRED1 HCM ~10%. WES mandatory to distinguish.
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
                { factor: 'NUBPL', module: 'N-module ([4Fe-4S] delivery)', mech: 'P-loop GTPase Fe-S carrier (IND1)', ribo: '0%', hcm: '~25%', chr: '14q12', highlight: true },
                { factor: 'FOXRED1', module: 'N-module (chaperone)', mech: 'FAD-oxidoreductase protein chaperone', ribo: '0%', hcm: '~10%', chr: '11q24.2' },
                { factor: 'NDUFAF2', module: 'N-Q module area', mech: 'NDUFA12-paralog assembly-swap', ribo: '0%', hcm: '<15%', chr: '5q12.1' },
                { factor: 'ACAD9', module: 'MCIA/ND2-ND5 (Class 1)', mech: 'FAD-binding MCIA scaffold', ribo: '50-60% (Level B)', hcm: '55-65%', chr: '3q21.3' },
                { factor: 'NDUFAF1', module: 'MCIA/ND2-ND5 (Class 1)', mech: 'CIA30 obligate ACAD9 partner', ribo: '0%', hcm: '<20%', chr: '15q11.2-q13' },
                { factor: 'TIMMDC1', module: 'ND1-module (Class 3)', mech: 'Integral IMM ND1-module scaffold', ribo: '0%', hcm: '>80%', chr: '3q25.1' },
                { factor: 'NDUFV1', module: 'N-module (structural)', mech: 'FMN-binding structural subunit (N3 cluster recipient)', ribo: '0%', hcm: '<15%', chr: '11q13.2' },
                { factor: 'NDUFS1', module: 'N-module (structural)', mech: '75kDa structural subunit (N1b/N3/N4/N5 recipient)', ribo: '0%', hcm: '<15%', chr: '2q33.3' },
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
                  <td className={r.hcm.includes('>80%') ? 'text-danger fw-bold' : r.hcm.includes('55-65%') ? 'fw-bold' : r.hcm.includes('~25%') ? 'text-warning fw-bold' : ''}>{r.hcm}</td>
                  <td className="font-monospace small">{r.chr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🔍 [4Fe-4S] Cluster Delivery — NUBPL N-Module Targets" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>NUBPL delivers [4Fe-4S] clusters to these N-module positions:</strong>
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>Cluster</th><th>Recipient Subunit</th><th>Position / Role</th><th>NUBPL-Dependent</th></tr>
            </thead>
            <tbody>
              {[
                { cluster: 'N1b', subunit: 'NDUFS1 (75kDa)', role: 'N-module matrix arm; first electron relay from NADH', dep: true },
                { cluster: 'N3', subunit: 'NDUFV1 (51kDa)', role: 'NADH-oxidizing subunit; FMN + N3 Fe-S pair', dep: true },
                { cluster: 'N4', subunit: 'NDUFS1 (75kDa)', role: 'Matrix arm electron relay', dep: true },
                { cluster: 'N5', subunit: 'NDUFS1 (75kDa)', role: 'Matrix arm Fe-S; electron transfer chain', dep: true },
                { cluster: 'N6a', subunit: 'NDUFS8', role: 'Q-module/N-module boundary; terminal N-cluster before N2', dep: false },
                { cluster: 'N6b', subunit: 'NDUFS8', role: 'Q-module boundary Fe-S', dep: false },
                { cluster: 'N2', subunit: 'NDUFS7 (PSST)', role: 'Q-module; penultimate Fe-S; electron donor to CoQ', dep: false },
                { cluster: 'N1a', subunit: 'NDUFV2 (24kDa)', role: '[2Fe-2S] cluster; NUBPL-independent (ISC direct)', dep: false },
              ].map(r => (
                <tr key={r.cluster}>
                  <td className="font-monospace fw-bold" style={{ color: r.dep ? COLOR : '#9e9e9e' }}>{r.cluster}</td>
                  <td className="fw-bold small">{r.subunit}</td>
                  <td className="text-muted small">{r.role}</td>
                  <td>{r.dep
                    ? <span className="badge" style={{ background: COLOR, fontSize: '0.65rem' }}>NUBPL-dependent</span>
                    : <span className="badge bg-secondary" style={{ fontSize: '0.65rem' }}>Not NUBPL</span>
                  }</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🧬 European Founder — p.Gly56Arg + c.815-27T>C Deep Intronic" borderColor="#2e7d32">
        <div className="row g-3">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#2e7d32' }}>p.Gly56Arg (c.166G>A) — Walker A Founder</div>
            <ul className="list-unstyled small">
              <li>• ~70% of NUBPL patients carry this allele</li>
              <li>• European enrichment (Scottish/British)</li>
              <li>• Disrupts Walker A P-loop GxGxxG motif</li>
              <li>• Hypomorphic — partial GTPase residual activity</li>
              <li>• <strong>Detectable by standard exome sequencing</strong></li>
              <li>• Intermediate severity; typically compound het</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#c62828' }}>c.815-27T>C — Deep Intronic Branch Point (Intron 9)</div>
            <ul className="list-unstyled small">
              <li>• 27 bp upstream of exon 10 acceptor site</li>
              <li>• Disrupts branch-point adenosine → aberrant splicing</li>
              <li>• Partial exon 9/10 skipping; partial normal mRNA</li>
              <li>• <strong className="text-danger">Missed by standard exome sequencing</strong></li>
              <li>• Requires RNA-seq, RT-PCR, or targeted deep intronic PCR</li>
              <li>• Hypomorphic second allele — explains survival to infantile onset</li>
            </ul>
          </div>
        </div>
        <div className="alert mt-3 mb-0" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
          <strong>Clinical Action:</strong> European infant with isolated CI deficiency + heterozygous p.Gly56Arg →
          NUBPL is the leading diagnosis. Request NUBPL deep intronic sequencing (c.815-27T>C) or NUBPL mRNA/cDNA analysis.
          Do not stop at heterozygous exome finding — the second allele is deep intronic.
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
export default function NUBPLPage() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/nubpl/overview`).then(r => r.json()),
      fetch(`${API}/api/nubpl/breakdown`).then(r => r.json()),
      fetch(`${API}/api/nubpl/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 NUBPL (IND1) — Complex I Deficiency (N-Module [4Fe-4S] Delivery)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>N-Module [4Fe-4S] Assembly</span>
        <span className="badge ms-1" style={{ background: '#2e7d32' }}>p.Gly56Arg European Founder ~70%</span>
        <span className="badge ms-1" style={{ background: '#e65100' }}>HCM ~25% (Intermediate)</span>
        <span className="badge ms-1 bg-secondary">14q12</span>
        <span className="badge ms-1 bg-secondary">OMIM *613621</span>
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
      {tab === 2 && <FeSTabb data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
