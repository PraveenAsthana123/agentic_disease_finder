'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Assembly Scaffold', 'Definitions'];
const COLOR = '#1a237e';   // dark indigo — intermediate CI assembly scaffold / structural chaperone
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
  const mod = data.ndufaf8_module_summary || {};

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
        <strong>🟢 NDUFAF8 (C17orf89) — CI Assembly Chaperone/Scaffold (Intermediate Stage, Structural, Non-Enzymatic)</strong> —
        NDUFAF8 is a soluble mitochondrial matrix CI assembly factor that stabilises CI assembly intermediates
        during intermediate stages of holoenzyme maturation. Unlike NDUFAF6 (2OG-dioxygenase) and NDUFAF7
        (SAM-methyltransferase) — the only two enzymatically active CI assembly factors — NDUFAF8 has NO
        confirmed enzymatic activity. It acts as a purely structural scaffold, analogous to NDUFAF3/4/5.
        17p13.2 locus is unique among all NDUFAF family members. ~230aa, ~27kDa, soluble matrix.
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 NO RIBOFLAVIN RESPONSE — Critical DDx vs ACAD9 (50-60% Riboflavin-Responsive)</strong> —
        NDUFAF8 is a structural scaffold with NO FAD domain. Riboflavin supplementation CANNOT rescue the
        CI assembly chaperone defect. Any riboflavin response in a CI patient points to ACAD9 (MCIA/ND2-ND5),
        not NDUFAF8. Riboflavin trial + WES (3q21.3 vs 17p13.2) is the mandatory discriminator.
        <strong> NDUFAF8 (17p13.2) unique chromosomal locus — distinct from all other NDUFAF family members.</strong>
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
                  k === 'Enzymatic_activity' ? 'text-warning fw-bold' :
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

      <SectionCard title="⚙️ NDUFAF8 CI Assembly Scaffold Summary">
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>NDUFAF8 role:</strong> Structural chaperone/scaffold → stabilises CI assembly intermediates → enables holoenzyme CI maturation
        </div>
        <p className="small mb-1">{mod.unique_structural_scaffold}</p>
        {mod.assembly_chaperone_mechanism && (
          <p className="small text-muted mb-0">{mod.assembly_chaperone_mechanism}</p>
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
            <span className="font-monospace text-muted">{v}</span>
            <span className="fw-bold">{count}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="🔑 Structural Scaffold Key Features">
        {Object.entries(data.key_structural_scaffold_features || {}).map(([k, v]) => (
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
                  <td className="font-monospace" style={{ fontSize: '0.7rem' }}>{p.allele1}</td>
                  <td className="font-monospace" style={{ fontSize: '0.7rem' }}>{p.allele2}</td>
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
                <tr key={v.hgvs_p}>
                  <td className="font-monospace small">{v.hgvs_c}</td>
                  <td className="font-monospace small fw-bold">{v.hgvs_p}</td>
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

// ── Tab: Assembly Scaffold ────────────────────────────────────────────────────
function AssemblyScaffoldTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const mod = data.ndufaf8_module_summary || {};

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>NDUFAF8 (C17orf89) — Structural CI Assembly Chaperone (Non-Enzymatic Scaffold)</strong><br />
        <span className="small text-muted">
          NDUFAF8 is a purely structural CI assembly chaperone/scaffold with NO confirmed enzymatic activity.
          This places it in the same structural-scaffold category as NDUFAF3/4/5, but at a distinct
          intermediate CI assembly stage. Critically different from NDUFAF6 (2OG-dioxygenase) and NDUFAF7
          (SAM-methyltransferase) — the ONLY two enzymatically active CI assembly factors.
          BN-PAGE shows intermediate CI assembly stalling at the NDUFAF8-dependent stage.
          Chromosome 17p13.2 — unique locus among all NDUFAF family members.
        </span>
      </div>

      {mod.gene && (
        <SectionCard title="⚙️ NDUFAF8 CI Assembly Chaperone Mechanism" borderColor={COLOR}>
          <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Gene: {mod.gene} | Module class: {mod.module_class}</strong>
          </div>
          <div className="small text-muted mb-1">CI assembly chaperone/scaffold mechanism</div>
          <p className="small mb-2">{mod.assembly_chaperone_mechanism}</p>
          <div className="small text-muted mb-1">NDUFAF8 vs enzymatic CI assembly factors (NDUFAF6/7)</div>
          <p className="small mb-0">{mod.ndufaf8_vs_enzymatic_ci_factors}</p>
        </SectionCard>
      )}

      <SectionCard title="🔬 DDx Matrix — NDUFAF8 vs Key CI Genes" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Comparator</th><th>NDUFAF8</th><th>Comparator</th><th>Key Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map((row, i) => (
                <tr key={i} style={{
                  background: row.comparator.includes('ACAD9') ? '#fff8e1' :
                              row.comparator.includes('NDUFAF7') ? '#e8f5e9' :
                              row.comparator.includes('TIMMDC1') ? '#fce4ec' :
                              undefined
                }}>
                  <td className="fw-bold small">{row.comparator}</td>
                  <td className="text-muted small">{row.ndufaf8}</td>
                  <td className="text-muted small">{row.comparator_val}</td>
                  <td className="fw-bold small" style={{ color: '#1565c0' }}>{row.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ CI Assembly — Enzymatic vs Structural Factors" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#e8f5e9', borderLeft: '4px solid #2e7d32' }}>
          <strong>Only TWO CI assembly factors have enzymatic (catalytic) activity:</strong>{' '}
          NDUFAF6 (2OG-dioxygenase) and NDUFAF7 (SAM-methyltransferase).
          NDUFAF8 — like NDUFAF3/4/5, FOXRED1, NUBPL, ACAD9-scaffold, TIMMDC1 — acts as a structural factor.
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>Factor/Gene</th><th>Module</th><th>Enzymatic?</th><th>Riboflavin</th><th>HCM</th><th>Chromosome</th>
              </tr>
            </thead>
            <tbody>
              {[
                { factor: 'NDUFAF8', module: 'Intermediate CI assembly', enzymatic: 'NO — structural chaperone/scaffold', ribo: '0%', hcm: '<10%', chr: '17p13.2', highlight: true },
                { factor: 'NDUFAF7', module: 'Q-module/membrane-arm interface', enzymatic: 'SAM-methyltransferase (NDUFS2 R-85 methylation)', ribo: '0%', hcm: '<10%', chr: '2q11.2', enzymatic_yes: true },
                { factor: 'NDUFAF6', module: 'Q-module (late-stage)', enzymatic: '2OG-Fe(II) dioxygenase/hydroxylase (NDUFS7/PSST PTM)', ribo: '0%', hcm: '<10%', chr: '8q22.1', enzymatic_yes: true },
                { factor: 'FOXRED1', module: 'N-module', enzymatic: 'NO — FAD-oxidoreductase chaperone (non-covalent)', ribo: '0%', hcm: '~10%', chr: '11q24.2' },
                { factor: 'NUBPL', module: 'N-module', enzymatic: 'GTPase — [4Fe-4S] cluster transfer', ribo: '0%', hcm: '~25%', chr: '14q12' },
                { factor: 'ACAD9', module: 'MCIA/ND2-ND5 (Class 1)', enzymatic: 'FAD-binding (ancestral) — scaffold; riboflavin-responsive', ribo: '50-60%', hcm: '55-65%', chr: '3q21.3' },
                { factor: 'NDUFAF1', module: 'MCIA/ND2-ND5 (Class 1)', enzymatic: 'NO — CIA30 structural scaffold', ribo: '0%', hcm: '20-30%', chr: '15q13.3' },
                { factor: 'TIMMDC1', module: 'ND1-module (Class 3)', enzymatic: 'NO — integral IMM (2-TM) ND1 scaffold', ribo: '0%', hcm: '>80%', chr: '3q25.1' },
                { factor: 'NDUFAF3', module: 'ND1-module (Class 3)', enzymatic: 'NO — NDUFAF4-obligate heterodimer scaffold', ribo: '0%', hcm: '15-25%', chr: '2q33.1' },
                { factor: 'NDUFAF5', module: 'ND1-module (Class 3)', enzymatic: 'NO — independent ND1 scaffold', ribo: '0%', hcm: '<20%', chr: '20p12.1' },
              ].map(r => (
                <tr key={r.factor} style={{ background: r.highlight ? LIGHT : r.enzymatic_yes ? '#e8f5e9' : undefined }}>
                  <td className="fw-bold" style={{ color: r.highlight ? COLOR : r.enzymatic_yes ? '#1b5e20' : undefined }}>
                    {r.factor}
                    {r.highlight && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>THIS</span>}
                    {r.enzymatic_yes && <span className="badge ms-1" style={{ background: '#1b5e20', fontSize: '0.6rem' }}>ENZYMATIC</span>}
                  </td>
                  <td className="small text-muted">{r.module}</td>
                  <td className="small" style={{ color: r.highlight ? '#c62828' : r.enzymatic_yes ? '#1b5e20' : '#757575' }}>{r.enzymatic}</td>
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

      <SectionCard title="🧬 Riboflavin Test — NDUFAF8 vs ACAD9 Discriminator" borderColor="#2e7d32">
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
            <div className="fw-bold small mb-2" style={{ color: '#c62828' }}>NDUFAF8 — Structural Scaffold (No Riboflavin Response)</div>
            <ul className="list-unstyled small">
              <li>• NO FAD domain — no flavin binding whatsoever</li>
              <li>• Structural chaperone/scaffold — no enzymatic cofactor</li>
              <li>• Riboflavin supplementation has no pathway to NDUFAF8 assembly scaffold</li>
              <li>• <strong className="text-danger">0% riboflavin response (no FAD domain; structural mechanism)</strong></li>
              <li>• HCM &lt;10%; intermediate CI assembly; 17p13.2</li>
              <li>• Riboflavin trial NEGATIVE → pursue WES 17p13.2 locus</li>
            </ul>
          </div>
        </div>
        <div className="alert mt-3 mb-0" style={{ background: '#e8eaf6', borderLeft: `4px solid ${COLOR}` }}>
          <strong>🔵 NDUFAF8 17p13.2 Unique Locus:</strong> NDUFAF8 (17p13.2) is on a different chromosome from
          all other NDUFAF family members: NDUFAF1 (15q11.2), NDUFAF2 (5q12.1), NDUFAF3 (2q33.1),
          NDUFAF4 (6q16.3), NDUFAF5 (20p12.1), NDUFAF6 (8q22.1), NDUFAF7 (2q11.2).
          WES locus discrimination is straightforward — 17p13.2 is a unique address.
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

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function NDUFAF8Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ndufaf8/overview`).then(r => r.json()),
      fetch(`${API}/api/ndufaf8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ndufaf8/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 NDUFAF8 (C17orf89) — Complex I Deficiency (CI Assembly Scaffold / Intermediate-Stage)</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>CI Assembly Scaffold</span>
        <span className="badge ms-1" style={{ background: '#c62828' }}>Non-Enzymatic</span>
        <span className="badge ms-1" style={{ background: '#b71c1c' }}>HCM &lt;10% (Very Low)</span>
        <span className="badge ms-1 bg-secondary">17p13.2</span>
        <span className="badge ms-1 bg-secondary">OMIM *616051</span>
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
      {tab === 2 && <AssemblyScaffoldTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
