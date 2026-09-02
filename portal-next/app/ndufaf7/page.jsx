'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Methyltransferase Activity', 'Definitions'];
const COLOR = '#004d40';   // deep teal — SAM-methyltransferase / methyl-donor biochemistry
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
  const mod = data.ndufaf7_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="HCM Rate"         value="&lt;10% (Very Low)"         color="#b71c1c" />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_kda} kDa`}       color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🟢 NDUFAF7 (C2orf56/MIDI1IP1) — ONLY CI Assembly Factor with SAM-Dependent Methyltransferase Activity</strong> —
        NDUFAF7 uniquely catalyzes S-adenosylmethionine (SAM)-dependent arginine methylation of NDUFS2 (R-85, mature)
        during CI assembly. This post-translational modification is required for NDUFB9 to bind the Q-module/
        membrane-arm interface, enabling late-stage CI assembly progression.
        Alongside NDUFAF6 (2OG-dioxygenase), NDUFAF7 is one of only TWO enzymatically active CI assembly factors.
        2q11.2, 371aa, ~41kDa, class I SAM-dependent methyltransferase (Rossmann-like beta/alpha fold).
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9 (50-60% Riboflavin-Responsive)</strong> —
        NDUFAF7 is a SAM-methyltransferase with NO FAD domain. Riboflavin supplementation CANNOT rescue the
        NDUFS2 R-85 arginine methylation defect. Any riboflavin response in a CI patient points to ACAD9 (MCIA/ND2-ND5),
        not NDUFAF7. Riboflavin trial + WES (3q21.3 vs 2q11.2) is the mandatory discriminator.
        <strong> IMPORTANT: NDUFAF7 (2q11.2) and NDUFS1 (2q33.3) are on the SAME CHROMOSOME 2q — WES locus
        discrimination is mandatory. Clinical DDx: NDUFS1 peripheral neuropathy ~50%; NDUFAF7 ~0%.</strong>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="🔬 Biochemical Fingerprint">
            {Object.entries(bf).map(([k, v]) => (
              <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
                <span className="text-muted">{k.replace(/_/g, ' ')}</span>
                <span className={
                  k === 'Complex_I' ? 'text-danger fw-bold' :
                  k === 'Riboflavin_response' ? 'text-danger fw-bold' :
                  k === 'SAM_methyltransferase_unique' ? 'text-success fw-bold' :
                  k === 'HCM_rate' ? 'text-info fw-bold' :
                  'text-success fw-bold'
                } style={{ maxWidth: '55%', textAlign: 'right', fontSize: '0.7rem' }}>{v}</span>
              </div>
            ))}
          </SectionCard>
        </div>
        <div className="col-md-6">
          <SectionCard title="📊 Feature Frequencies (40-patient cohort)">
            {Object.entries(ff).slice(0, 10).map(([feat, pct]) => (
              <Bar key={feat} label={feat}
                value={pct}
                color={
                  feat.includes('HCM') ? '#b71c1c' :
                  feat.includes('Lactic') ? '#c62828' :
                  feat.includes('Leigh') ? '#b71c1c' :
                  feat.includes('Basal') ? '#e65100' :
                  COLOR
                }
              />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="⚗️ NDUFAF7 SAM-Methyltransferase Mechanism Summary">
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>SAM reaction:</strong> NDUFS2 + SAM → NDUFS2(R-85-methyl) + SAH → NDUFB9 recruitment → Q-module/membrane-arm assembly proceeds
        </div>
        <p className="small mb-1">{mod.unique_sam_methyltransferase}</p>
        {mod.ndufs2_r85_methylation_mechanism && (
          <p className="small text-muted mb-0">{mod.ndufs2_r85_methylation_mechanism}</p>
        )}
      </SectionCard>

      <SectionCard title="⚠️ Key DDx Summary" borderColor="#c62828">
        {(data.key_ddx || []).slice(0, 4).map(d => (
          <div key={d.feature} className="mb-3 border-bottom pb-2">
            <div className="small fw-bold" style={{ color: '#c62828' }}>{d.feature}</div>
            <p className="small text-muted mb-0">{d.significance}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚨 Clinical Alerts">
        {(data.clinical_alerts || []).map((a, i) => (
          <div key={i} className="py-1 border-bottom small">{a}</div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Tab: Patients & Features ──────────────────────────────────────────────────
function PatientsTab({ data }) {
  const [search, setSearch] = useState('');
  if (!data) return <p className="text-muted">Loading…</p>;

  const patients = (data.patients || []).map((p, i) => ({
    id: p.patient_id || `P${String(i + 1).padStart(3, '0')}`,
    onset_age_months: p.onset_age_months,
    sex: p.sex,
    allele1: p.allele1,
    allele2: p.allele2,
    ci_activity_pct: p.ci_activity_pct,
    hcm: p.hcm,
    leigh_mri: p.leigh_mri,
    lactic_acidosis: p.lactic_acidosis,
    basal_ganglia: p.basal_ganglia,
    seizures: p.features?.['Seizures (multiple types)'],
    outcome: p.outcome,
  }));

  const filtered = patients.filter(p => {
    if (!search) return true;
    const s = search.toLowerCase();
    return p.allele1?.toLowerCase().includes(s) || p.allele2?.toLowerCase().includes(s) ||
           p.outcome?.toLowerCase().includes(s) || p.sex?.toLowerCase() === s;
  });

  return (
    <>
      <div className="row g-3 mb-4">
        <div className="col-md-3">
          <SectionCard title="📊 Cohort">
            <KPI label="Total Patients"  value={data.cohort_n}  color={COLOR} />
            <KPI label="Consanguineous" value={`${data.consanguineous_pct}%`} color="#e65100" />
          </SectionCard>
        </div>
        <div className="col-md-3">
          <SectionCard title="🧬 CI Activity">
            <KPI label="Mean CI %"  value={`${data.ci_activity_stats?.mean}%`} color="#c62828" />
            <KPI label="Min / Max" value={`${data.ci_activity_stats?.min}–${data.ci_activity_stats?.max}%`} color="#c62828" />
          </SectionCard>
        </div>
        <div className="col-md-3">
          <SectionCard title="⏱️ Sex">
            <KPI label="Male"   value={data.sex_distribution?.M} color={COLOR} />
            <KPI label="Female" value={data.sex_distribution?.F} color={COLOR} />
          </SectionCard>
        </div>
        <div className="col-md-3">
          <SectionCard title="📈 Outcomes">
            {Object.entries(data.outcome_distribution || {}).map(([o, c]) => (
              <div key={o} className="d-flex justify-content-between small border-bottom py-1">
                <span className="text-muted">{o}</span>
                <span className="fw-bold">{c}</span>
              </div>
            ))}
          </SectionCard>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-6">
          <SectionCard title="📊 CI Activity Bands">
            {Object.entries(data.ci_activity_stats?.bands || {}).map(([band, count]) => (
              <Bar key={band} label={band} value={Math.round(count / (data.cohort_n || 40) * 100)} color="#c62828" />
            ))}
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
            <span className={`font-monospace ${v.includes('Arg321Pro') ? 'fw-bold text-danger' : 'text-muted'}`}>{v}</span>
            <span className="fw-bold">{count}{v.includes('Arg321Pro') ? ' ← Zurita Rendón 2014 founder' : ''}{v.includes('Leu152Pro') ? ' ← helix-breaking' : ''}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔑 SAM-Methyltransferase Key Features">
        {Object.entries(data.key_sam_methyltransferase_features || {}).map(([k, v]) => (
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
                <th>CI %</th><th>Leigh MRI</th><th>Lactic Ac.</th><th>Basal Gang.</th><th>HCM</th><th>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(p => (
                <tr key={p.id}>
                  <td>{p.id}</td>
                  <td>{p.onset_age_months}</td>
                  <td>{p.sex}</td>
                  <td className={`font-monospace ${p.allele1 === 'p.Arg321Pro' ? 'fw-bold text-danger' : ''}`} style={{ fontSize: '0.7rem' }}>{p.allele1}</td>
                  <td className={`font-monospace ${p.allele2 === 'p.Arg321Pro' ? 'fw-bold text-danger' : ''}`} style={{ fontSize: '0.7rem' }}>{p.allele2}</td>
                  <td><span className="badge bg-danger">{p.ci_activity_pct}%</span></td>
                  <td>{p.leigh_mri ? '✅' : '—'}</td>
                  <td>{p.lactic_acidosis ? '✅' : '—'}</td>
                  <td>{p.basal_ganglia ? '⚠️' : '—'}</td>
                  <td>{p.hcm ? <span className="badge" style={{ background: '#b71c1c', fontSize: '0.6rem' }}>HCM</span> : '—'}</td>
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
                <tr key={v.hgvs_p} style={{ background: v.hgvs_p === 'p.Arg321Pro' ? '#e8f5e9' : undefined }}>
                  <td className="font-monospace small">{v.hgvs_c}</td>
                  <td className={`font-monospace small fw-bold ${v.hgvs_p === 'p.Arg321Pro' ? 'text-danger' : ''}`}>{v.hgvs_p}</td>
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

// ── Tab: Methyltransferase Activity ────────────────────────────────────────────
function MethyltransferaseTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const mod = data.ndufaf7_module_summary || {};

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>NDUFAF7 (C2orf56/MIDI1IP1) — ONLY CI Assembly Factor with SAM-Dependent Methyltransferase Activity</strong><br />
        <span className="small text-muted">
          NDUFAF7 is unique among all known CI assembly factors: it is the sole factor that catalyzes
          S-adenosylmethionine (SAM)-dependent arginine methylation of a CI subunit. Using SAM as methyl
          donor (class I Rossmann-like methyltransferase fold), NDUFAF7 methylates arginine-85 of the mature
          NDUFS2 subunit (Q-module). This methylation is required for NDUFB9 to bind the Q-module/membrane-arm
          interface, enabling late-stage CI assembly to proceed. Loss stalls assembly at this checkpoint.
          BN-PAGE: Q-module/membrane-arm interface intermediates accumulate with unmethylated NDUFS2 R-85.
          Alongside NDUFAF6 (2OG-dioxygenase), NDUFAF7 is one of only TWO enzymatically active CI assembly factors.
        </span>
      </div>

      {mod.gene && (
        <SectionCard title="⚙️ NDUFAF7 SAM-Methyltransferase Mechanism Detail" borderColor={COLOR}>
          <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Gene: {mod.gene} | Module class: {mod.module_class}</strong>
          </div>
          <div className="small text-muted mb-1">SAM-dependent methyltransferase reaction mechanism</div>
          <p className="small mb-2">{mod.sam_methyltransferase_mechanism}</p>
          <div className="small text-muted mb-1">NDUFB9 recruitment and membrane-arm assembly</div>
          <p className="small mb-2">{mod.ndufb9_recruitment}</p>
          <div className="small text-muted mb-1">NDUFAF7 vs NDUFAF6 — Two enzymatic CI assembly factors</div>
          <p className="small mb-0">{mod.ndufaf7_vs_ndufaf6}</p>
        </SectionCard>
      )}

      <SectionCard title="🔬 DDx Matrix — NDUFAF7 vs Key CI Genes" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Comparator</th><th>NDUFAF7</th><th>Comparator</th><th>Key Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map((row, i) => (
                <tr key={i} style={{
                  background: row.comparator.includes('ACAD9') ? '#fff8e1' :
                              row.comparator.includes('NDUFS1') ? '#fce4ec' :
                              row.comparator.includes('NDUFAF6') ? '#e8f5e9' :
                              undefined
                }}>
                  <td className="fw-bold small">{row.comparator}</td>
                  <td className="text-muted small">{row.ndufaf7}</td>
                  <td className="text-muted small">{row.comparator_val}</td>
                  <td className="fw-bold small" style={{ color: '#1565c0' }}>{row.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ CI Assembly — NDUFAF7 SAM-Methyltransferase vs All CI Assembly Factors" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
          <strong>Only TWO CI assembly factors have enzymatic (catalytic) activity:</strong>{' '}
          NDUFAF7 (SAM-methyltransferase) and NDUFAF6 (2OG-dioxygenase).
          All others are chaperones, scaffolds, or Fe-S delivery factors — no catalytic PTM activity.
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>Factor/Gene</th><th>Module</th><th>Enzymatic Activity?</th><th>Riboflavin Response</th><th>HCM</th><th>Chromosome</th>
              </tr>
            </thead>
            <tbody>
              {[
                { factor: 'NDUFAF7', module: 'Q-module/membrane-arm interface', enzymatic: 'SAM-methyltransferase (NDUFS2 R-85 methylation)', ribo: '0%', hcm: '<10%', chr: '2q11.2', highlight: true },
                { factor: 'NDUFAF6', module: 'Q-module (late-stage)', enzymatic: '2OG-Fe(II) dioxygenase/hydroxylase (NDUFS7/PSST PTM)', ribo: '0%', hcm: '<5-10%', chr: '8q22.1', enzymatic_yes: true },
                { factor: 'FOXRED1', module: 'N-module', enzymatic: 'NO — FAD-oxidoreductase chaperone (non-covalent)', ribo: '0%', hcm: '~10%', chr: '11q24.2' },
                { factor: 'NUBPL', module: 'N-module', enzymatic: 'GTPase (IND1) — [4Fe-4S] cluster transfer (cofactor insertion, not substrate PTM)', ribo: '0%', hcm: '~25%', chr: '14q12' },
                { factor: 'ACAD9', module: 'MCIA/ND2-ND5 (Class 1)', enzymatic: 'FAD-binding dehydrogenase (ancestral) — assembly scaffold role, riboflavin-responsive', ribo: '50-60%', hcm: '55-65%', chr: '3q21.3' },
                { factor: 'NDUFAF1', module: 'MCIA/ND2-ND5 (Class 1)', enzymatic: 'NO — structural CIA30 scaffold', ribo: '0%', hcm: '20-30%', chr: '15q13.3' },
                { factor: 'ECSIT', module: 'MCIA/ND2-ND5 (Class 1)', enzymatic: 'NO — TMEM126B-recruiting scaffold', ribo: '0%', hcm: '<20%', chr: '19p13.3' },
                { factor: 'TMEM126B', module: 'MCIA/ND2-ND5 (Class 1)', enzymatic: 'NO — integral IMM (2-TM) structural scaffold', ribo: '0%', hcm: '<20%', chr: '11q14.1' },
                { factor: 'TIMMDC1', module: 'ND1-module (Class 3)', enzymatic: 'NO — integral IMM (2-TM) ND1 scaffold', ribo: '0%', hcm: '>80%', chr: '3q25.1' },
                { factor: 'NDUFAF3', module: 'ND1-module (Class 3)', enzymatic: 'NO — NDUFAF4-obligate heterodimer scaffold', ribo: '0%', hcm: '15-25%', chr: '2q33.1' },
                { factor: 'NDUFAF5', module: 'ND1-module (Class 3)', enzymatic: 'NO — independent ND1 scaffold', ribo: '0%', hcm: '<20%', chr: '20p12.1' },
                { factor: 'NDUFAF2', module: 'N-Q module interface', enzymatic: 'NO — assembly factor swap chaperone (NDUFA12L paralog)', ribo: '0%', hcm: '<10%', chr: '5q12.1' },
              ].map(r => (
                <tr key={r.factor} style={{ background: r.highlight ? LIGHT : r.enzymatic_yes ? '#e8f5e9' : undefined }}>
                  <td className="fw-bold" style={{ color: r.highlight ? COLOR : r.enzymatic_yes ? '#1b5e20' : undefined }}>
                    {r.factor}
                    {r.highlight && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>THIS</span>}
                    {r.enzymatic_yes && <span className="badge ms-1" style={{ background: '#1b5e20', fontSize: '0.6rem' }}>ENZYMATIC</span>}
                  </td>
                  <td className="small text-muted">{r.module}</td>
                  <td className="small" style={{ color: r.highlight || r.enzymatic_yes ? '#1b5e20' : '#757575' }}>{r.enzymatic}</td>
                  <td>
                    <span className="badge" style={{
                      background: r.ribo.includes('50-60%') ? '#1565c0' : '#c62828',
                      fontSize: '0.65rem'
                    }}>{r.ribo}</span>
                  </td>
                  <td className={r.hcm.includes('>80%') ? 'text-danger fw-bold' : r.hcm.includes('55-65%') ? 'fw-bold' : r.hcm.includes('<10%') || r.hcm.includes('<5') ? 'text-info fw-bold' : ''}>{r.hcm}</td>
                  <td className="font-monospace small">{r.chr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="⚗️ SAM-Dependent Methyltransferase Reaction — NDUFAF7 Catalytic Mechanism" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>NDUFAF7 catalyzes:</strong> NDUFS2 (R-85) + SAM → NDUFS2 (R-85-methyl) + SAH → NDUFB9 can bind → Q/membrane-arm assembly proceeds
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr><th>Component</th><th>Role in NDUFAF7 Reaction</th><th>Effect if Absent/Mutated</th></tr>
            </thead>
            <tbody>
              {[
                { comp: 'SAM (S-adenosylmethionine)', role: 'Methyl donor; SAM → SAH (S-adenosylhomocysteine) after methyl transfer; coordinates in GxGxxG-like loop', eff: 'No SAM → no methylation; NDUFS2 R-85 remains unmethylated; NDUFB9 cannot bind; assembly stalls' },
                { comp: 'NDUFS2 Arg-85 (substrate)', role: 'Target arginine of mature NDUFS2 (Q-module subunit); side chain guanidinium accepts methyl group from SAM', eff: 'NDUFS2 R85K/R85C mutation → NDUFB9 binding fails even if NDUFAF7 is functional; same phenotype as NDUFAF7 LOF' },
                { comp: 'Rossmann-like beta/alpha fold', role: 'Core class I methyltransferase scaffold; coordinates SAM in βαβαβ motif; positions NDUFS2 substrate', eff: 'p.Gly214Arg (glycine-rich SAM loop) or p.Leu152Pro (helix-breaking) disrupt fold → complete loss' },
                { comp: 'C-terminal SAM-binding helix cluster', role: 'Helix cluster that contributes to SAM-binding C-terminal domain; p.Arg321Pro (Zurita Rendón 2014) disrupts this region', eff: 'p.Arg321Pro = helix-breaking C-terminal SAM helix disruption → SAM cannot bind → first reported NDUFAF7 mutation' },
                { comp: 'NDUFS2 substrate-recognition interface (Arg-179)', role: 'NDUFAF7 helix C contributes to NDUFS2 substrate positioning; Arg-179 helps orient NDUFS2 R-85 in active site', eff: 'p.Arg179Cys disrupts electrostatic NDUFS2 docking → substrate methylation fails even if SAM binds' },
                { comp: 'NDUFB9 (membrane arm accessory subunit)', role: 'Downstream effector; NDUFB9 binds Q-module only after NDUFS2 R-85 methylation; enables membrane-arm assembly progression', eff: 'Without NDUFS2 R-85 methylation: NDUFB9 cannot bind; Q-module/membrane-arm assembly stalls; holoenzyme CI cannot form' },
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

      <SectionCard title="🧬 Riboflavin Test — NDUFAF7 vs ACAD9 Discriminator" borderColor="#2e7d32">
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
            <div className="fw-bold small mb-2" style={{ color: '#c62828' }}>NDUFAF7 — SAM-Methyltransferase (No Riboflavin Response)</div>
            <ul className="list-unstyled small">
              <li>• NO FAD domain — no flavin binding whatsoever</li>
              <li>• Uses SAM as cofactor, not FAD/riboflavin</li>
              <li>• Riboflavin supplementation has no pathway to NDUFAF7 SAM-methylation</li>
              <li>• <strong className="text-danger">0% riboflavin response (no FAD domain; SAM-dependent)</strong></li>
              <li>• HCM &lt;10%; Q-module/membrane-arm interface; 2q11.2</li>
              <li>• Riboflavin trial NEGATIVE → pursue WES 2q11.2 locus</li>
            </ul>
          </div>
        </div>
        <div className="alert mt-3 mb-0" style={{ background: '#fce4ec', borderLeft: '4px solid #c62828' }}>
          <strong>⚠️ Chr2q Same-Chromosome Warning:</strong> NDUFAF7 (2q11.2) and NDUFS1 (2q33.3) are both on
          chromosome 2q. NDUFS1 causes peripheral neuropathy ~50%; NDUFAF7 ~0%.
          If CI patient has peripheral neuropathy → favor NDUFS1 (2q33.3). No peripheral neuropathy → consider
          both; WES locus is the discriminator (2q11.2 vs 2q33.3).
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
      <SectionCard title="📖 Gene Definition">
        <p className="small mb-0">{data.gene_definition}</p>
      </SectionCard>

      <SectionCard title="🏥 Disease Definition">
        <p className="small mb-0">{data.disease_definition}</p>
      </SectionCard>

      <SectionCard title="🧬 Inheritance">
        <p className="small mb-0">{data.inheritance_definition}</p>
      </SectionCard>

      <SectionCard title="🔬 Module & Mechanism Definitions">
        {(data.module_definitions || []).map(c => (
          <div key={c.term} className="mb-3 border-bottom pb-2">
            <div className="fw-bold small" style={{ color: COLOR }}>{c.term}</div>
            <p className="small text-muted mb-0">{c.definition}</p>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🚫 Key Contraindication Definitions" borderColor="#c62828">
        {(data.contraindication_definitions || []).map(c => (
          <div key={c.drug} className="mb-3 border-bottom pb-2">
            <div className="d-flex gap-2 align-items-center mb-1">
              <span className="badge bg-danger">{c.level}</span>
              <span className="fw-bold small">{c.drug}</span>
            </div>
            <p className="small text-muted mb-1">{c.mechanism}</p>
            {c.alternative && <p className="small mb-0" style={{ color: '#2e7d32' }}>Alternative: {c.alternative}</p>}
          </div>
        ))}
      </SectionCard>

      <SectionCard title="📚 References" borderColor="#1565c0">
        {(data.reference_list || []).map((r, i) => (
          <div key={i} className="mb-3 p-2 rounded" style={{ background: LIGHT }}>
            <p className="small fw-bold mb-1">{r.citation}</p>
            <p className="small text-muted mb-0"><em>{r.significance}</em></p>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main Page ───────────────��──────────────────────────────────────────────────
export default function NDUFAF7Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufaf7/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufaf7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufaf7/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 NDUFAF7 (C2orf56) — Complex I Deficiency MC1DN30 (SAM-Methyltransferase / Q-Module / NDUFS2-R85)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>SAM-Methyltransferase</span>
        <span className="badge ms-1" style={{ background: '#1b5e20' }}>Only CI Methyltransferase</span>
        <span className="badge ms-1" style={{ background: '#b71c1c' }}>HCM &lt;10% (Very Low)</span>
        <span className="badge ms-1 bg-secondary">2q11.2</span>
        <span className="badge ms-1 bg-secondary">OMIM *615898</span>
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
      {tab === 2 && <MethyltransferaseTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
