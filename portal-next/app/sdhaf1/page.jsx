'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'CII Assembly', 'Definitions'];
const COLOR = '#4a148c';   // deep purple — CII assembly factor, distinct from CI blue/indigo
const LIGHT = '#f3e5f5';

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
  const mod = data.sdhaf1_module_summary || {};

  return (
    <>
      <div className="row g-3 mb-4">
        <KPI label="Gene"             value={data.gene}                  color={COLOR} />
        <KPI label="Alias"            value="LYRM8"                      color={COLOR} />
        <KPI label="OMIM Gene"        value={`*${data.omim_gene}`}       color={COLOR} />
        <KPI label="Chromosome"       value={data.chromosome}            color={COLOR} />
        <KPI label="Inheritance"      value={data.inheritance}           color={COLOR} />
        <KPI label="Protein"          value={`${p.size_aa}aa / ${p.size_kda}kDa`} color={COLOR} />
      </div>

      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>🟣 SDHAF1 (LYRM8) — CII Assembly Factor / LYR-Motif SDHB FeS Cluster Delivery / Only CII-Specific FeS Insertion Factor</strong> —
        SDHAF1 is a 111-aa LYR-motif (LYRM) protein that delivers [2Fe-2S] and [4Fe-4S] iron-sulfur clusters
        to SDHB (the FeS subunit of Complex II/SDH) via the HSC20/HSPA9 co-chaperone system.
        Without SDHAF1, SDHB is apo-protein, CII cannot assemble, and succinate accumulates.
        Isolated CII deficiency (10–30%); infantile leukoencephalopathy; brain MRS succinate elevated ~85% (pathognomonic).
        No FAD domain. No TM helices. No paraganglioma. No HCM. 19q13.12.
      </div>

      <div className="alert mb-4" style={{ background: '#fce4ec', borderLeft: '4px solid #b71c1c' }}>
        <strong>🔴 KETOGENIC DIET — ABSOLUTELY CONTRAINDICATED in CII Deficiency</strong> —
        Beta-oxidation generates FADH2 which enters ETC exclusively via Complex II.
        SDHAF1 patients have deficient CII (10–30%) — FADH2 cannot be oxidized → metabolic crisis.
        Also: succinate supplementation NOT recommended (CII itself is deficient).
        Riboflavin NOT indicated (SDHAF1 has no FAD domain — unlike SDHA).
      </div>

      <div className="alert mb-4" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
        <strong>🟠 BRAIN MRS SUCCINATE ELEVATED (~85%) — PATHOGNOMONIC for CII Deficiency</strong> —
        Elevated succinate peak on 1H-MRS at 2.4 ppm is pathognomonic for Complex II deficiency.
        Distinguishes SDHAF1 from Canavan (NAA peak), fumaric aciduria (fumarate), and other leukodystrophies.
        <strong> NO paraganglioma — critical DDx vs SDHB/SDHC/SDHD (dominant) and SDHAF2 (PGL2).</strong>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-md-6">
          <SectionCard title="🔬 Biochemical Fingerprint">
            {Object.entries(bf).map(([k, v]) => (
              <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
                <span className="text-muted">{k.replace(/_/g, ' ')}</span>
                <span className={
                  k === 'Complex_II_SDH' ? 'text-danger fw-bold' :
                  k === 'Riboflavin_response' ? 'text-warning fw-bold' :
                  k === 'KD_risk' ? 'text-danger fw-bold' :
                  k === 'Paraganglioma' ? 'text-success fw-bold' :
                  k === 'Succinate_MRS' ? 'text-info fw-bold' :
                  k === 'HCM_rate' ? 'text-success fw-bold' :
                  k.startsWith('Complex_I') || k.startsWith('Complex_III') || k.startsWith('Complex_IV') || k.startsWith('Complex_V') ? 'text-success fw-bold' :
                  'fw-bold'
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
                  feat.includes('MRS') ? '#4a148c' :
                  feat.includes('Leuko') ? '#b71c1c' :
                  feat.includes('Lactic') ? '#c62828' :
                  feat.includes('Spastic') ? '#e65100' :
                  feat.includes('HCM') ? '#1b5e20' :
                  feat.includes('Paraganglioma') ? '#1b5e20' :
                  COLOR
                }
              />
            ))}
          </SectionCard>
        </div>
      </div>

      <SectionCard title="⚙️ SDHAF1 LYR-Motif CII Assembly Summary">
        <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>SDHAF1 role:</strong> LYR-motif → recruits HSC20/HSPA9 → delivers [2Fe-2S] + [4Fe-4S] to SDHB → enables SDHA-SDHB catalytic core → CII holoenzyme
        </div>
        <p className="small mb-1">{mod.lyr_motif_mechanism}</p>
        {mod.sdhaf1_vs_sdhaf2_sdhaf3 && (
          <p className="small text-muted mb-0">{mod.sdhaf1_vs_sdhaf2_sdhaf3}</p>
        )}
      </SectionCard>

      <SectionCard title="⚠️ Key DDx Summary" borderColor="#c62828">
        {(data.ddx_table || []).slice(0, 4).map(d => (
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
    onset_age_months:   p.onset_age_months,
    sex:                p.sex,
    allele1:            p.allele1,
    allele2:            p.allele2,
    cii_activity_pct:   p.cii_activity_pct,
    hcm:                p.hcm,
    leukodystrophy:     p.leukodystrophy,
    brain_mrs_succinate: p.brain_mrs_succinate,
    lactic_acidosis:    p.lactic_acidosis,
    spastic_paraplegia: p.spastic_paraplegia,
    outcome:            p.outcome,
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
          <SectionCard title="🧬 CII Activity">
            <KPI label="Mean CII %"  value={`${data.cii_activity_stats?.mean}%`} color="#c62828" />
            <KPI label="Min / Max" value={`${data.cii_activity_stats?.min}–${data.cii_activity_stats?.max}%`} color="#c62828" />
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
          <SectionCard title="📊 CII Activity Bands (SDH %)">
            {Object.entries(data.cii_activity_stats?.bands || {}).map(([band, count]) => (
              <Bar key={band} label={band} value={Math.round(count / (data.cohort_n || 40) * 100)} color="#c62828" />
            ))}
            <div className="small text-muted mt-2">
              CII/SDH activity — isolated deficiency (CI/CIII/CIV normal)
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

      <div className="row mb-4">
        <div className="col-md-4">
          <div className="card shadow-sm" style={{ borderTop: `3px solid ${COLOR}` }}>
            <div className="card-body text-center">
              <div className="fw-bold" style={{ color: COLOR, fontSize: '2rem' }}>{data.brain_mrs_succinate_pct}%</div>
              <div className="small text-muted">Brain MRS Succinate Elevated</div>
              <div className="small" style={{ color: '#4a148c' }}>~85% — Pathognomonic</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm" style={{ borderTop: '3px solid #b71c1c' }}>
            <div className="card-body text-center">
              <div className="fw-bold" style={{ color: '#b71c1c', fontSize: '2rem' }}>{data.leukoencephalopathy_pct}%</div>
              <div className="small text-muted">Leukoencephalopathy</div>
              <div className="small text-muted">White matter disease (hallmark)</div>
            </div>
          </div>
        </div>
        <div className="col-md-4">
          <div className="card shadow-sm" style={{ borderTop: '3px solid #1b5e20' }}>
            <div className="card-body text-center">
              <div className="fw-bold" style={{ color: '#1b5e20', fontSize: '2rem' }}>{data.hcm_pct}%</div>
              <div className="small text-muted">HCM Rate (Very Low)</div>
              <div className="small text-muted">No HCM in SDHAF1</div>
            </div>
          </div>
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

      <SectionCard title="🔑 Key LYR-Motif / SDHAF1 Features">
        {Object.entries(data.key_lyr_motif_features || {}).map(([k, v]) => (
          <div key={k} className="d-flex justify-content-between border-bottom py-1 small">
            <span className="text-muted">{k.replace(/_/g, ' ')}</span>
            <span className={
              v === true ? 'text-success fw-bold' :
              v === false ? 'text-danger' :
              v === 0 ? 'text-success fw-bold' :
              typeof v === 'number' ? (v < 5 ? 'text-success fw-bold' : v > 80 ? 'text-danger fw-bold' : 'fw-bold') :
              'fw-bold'
            }>{String(v)}{typeof v === 'number' && k.includes('pct') ? '%' : ''}</span>
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
                <th>CII %</th><th>Leuko</th><th>MRS Succ.</th><th>Lactic Ac.</th><th>Spastic</th><th>HCM</th><th>Outcome</th>
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
                  <td><span className="badge bg-danger">{p.cii_activity_pct}%</span></td>
                  <td>{p.leukodystrophy ? '✅' : '—'}</td>
                  <td>{p.brain_mrs_succinate ? <span className="badge" style={{ background: COLOR, fontSize: '0.6rem' }}>↑Succ</span> : '—'}</td>
                  <td>{p.lactic_acidosis ? '⚠️' : '—'}</td>
                  <td>{p.spastic_paraplegia ? '⚠️' : '—'}</td>
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
            <div className="small fw-bold" style={{ color: k === 'absolute_ci' ? '#c62828' : k === 'not_recommended' ? '#b71c1c' : k === 'avoid' ? '#e65100' : k === 'diagnostic_priority' ? '#1565c0' : '#2e7d32' }}>
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
              <tr><th>cDNA</th><th>Protein</th><th>Domain</th><th>Severity</th><th>CII Range</th><th>Notes</th></tr>
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

// ── Tab: CII Assembly ──────────────────────────────────────────────────────────
function CIIAssemblyTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const mod = data.sdhaf1_module_summary || {};

  return (
    <>
      <div className="alert mb-4" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
        <strong>SDHAF1 (LYRM8) — LYR-Motif CII Assembly Factor / Only CII-Specific FeS Insertion Factor</strong><br />
        <span className="small text-muted">
          SDHAF1 is the sole LYR-motif protein dedicated to delivering [2Fe-2S] and [4Fe-4S] iron-sulfur
          clusters to SDHB (Complex II iron-sulfur subunit) via the HSC20/HSPA9 co-chaperone system.
          Without SDHAF1, SDHB remains apo-protein, the SDHA-SDHB catalytic core cannot form,
          and CII/SDH activity is severely deficient (10–30%). Succinate accumulates, causing
          infantile leukoencephalopathy with pathognomonic brain MRS succinate elevation.
          19q13.12 — distinct from all SDH structural subunits and other SDHAF assembly factors.
        </span>
      </div>

      {mod.cii_assembly_pathway && (
        <SectionCard title="⚙️ CII Assembly Pathway — SDHAF1 Role at Step 2 (SDHB FeS Insertion)" borderColor={COLOR}>
          <div className="alert mb-3" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
            <strong>Gene: {mod.gene} | Module: {mod.module_class}</strong>
          </div>
          <p className="small mb-2">{mod.cii_assembly_pathway}</p>

          <div className="table-responsive mt-3">
            <table className="table table-sm small">
              <thead style={{ background: COLOR, color: '#fff' }}>
                <tr><th>Step</th><th>Factor</th><th>Role</th><th>Disease if mutated</th></tr>
              </thead>
              <tbody>
                {[
                  { step: '1', factor: 'SDHAF2 (PGL2, 11q13.1)', role: 'FAD insertion into SDHA; SDHA stabilisation', disease: 'Hereditary Paraganglioma type 2 (dominant) — NO leukoencephalopathy', highlight: false },
                  { step: '2', factor: 'SDHAF1 (LYRM8, 19q13.12) ← THIS GENE', role: '[2Fe-2S] + [4Fe-4S] FeS cluster delivery to SDHB via LYR–HSC20/HSPA9', disease: 'CII Deficiency + Infantile Leukoencephalopathy (AR, biallelic)', highlight: true },
                  { step: '3', factor: 'SDHAF3 (1q21.2)', role: 'Protects FeS-loaded SDHB from oxidative damage', disease: 'CII Deficiency (AR) — similar to SDHAF1 phenotype', highlight: false },
                  { step: '4', factor: 'SDHA + SDHB assembly', role: 'Catalytic core formation (requires FeS-SDHB from step 2)', disease: 'N/A — requires SDHAF1 at step 2', highlight: false },
                  { step: '5', factor: 'SDHC + SDHD', role: 'Membrane anchor integration (inner mitochondrial membrane)', disease: 'SDHC/SDHD: Dominant paraganglioma (NO CII deficiency leukoencephalopathy)', highlight: false },
                ].map(r => (
                  <tr key={r.step} style={{ background: r.highlight ? LIGHT : undefined }}>
                    <td className="fw-bold" style={{ color: r.highlight ? COLOR : undefined }}>{r.step}</td>
                    <td className="fw-bold small" style={{ color: r.highlight ? COLOR : undefined }}>
                      {r.factor}
                      {r.highlight && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>THIS</span>}
                    </td>
                    <td className="text-muted small">{r.role}</td>
                    <td className="small" style={{ color: r.highlight ? '#c62828' : '#757575' }}>{r.disease}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      )}

      <SectionCard title="🔬 DDx Matrix — SDHAF1 vs Key CII/Succinate-Pathway Genes" borderColor="#1565c0">
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: '#1565c0', color: '#fff' }}>
              <tr>
                <th>Comparator</th><th>SDHAF1</th><th>Comparator</th><th>Key Test</th>
              </tr>
            </thead>
            <tbody>
              {(data.ddx_matrix || []).map((row, i) => (
                <tr key={i} style={{
                  background: row.comparator.includes('SDHB') ? '#fff8e1' :
                              row.comparator.includes('SDHA') ? '#fce4ec' :
                              row.comparator.includes('SDHAF2') ? '#e8f5e9' :
                              undefined
                }}>
                  <td className="fw-bold small">{row.comparator}</td>
                  <td className="text-muted small">{row.sdhaf1}</td>
                  <td className="text-muted small">{row.comparator_val}</td>
                  <td className="fw-bold small" style={{ color: '#1565c0' }}>{row.key_test}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🏗️ SDHAF Factors — CII Assembly Factor Comparison" borderColor={COLOR}>
        <div className="alert mb-3" style={{ background: '#fff3e0', borderLeft: '4px solid #e65100' }}>
          <strong>Critical clinical distinction:</strong> SDHAF2 (FAD insertion) → dominant paraganglioma.
          SDHAF1 (FeS insertion) → infantile leukoencephalopathy (recessive). Completely different diseases.
        </div>
        <div className="table-responsive">
          <table className="table table-sm small">
            <thead style={{ background: COLOR, color: '#fff' }}>
              <tr>
                <th>Factor</th><th>Function</th><th>Target</th><th>Inheritance</th><th>Disease</th><th>Chromosome</th>
              </tr>
            </thead>
            <tbody>
              {[
                { factor: 'SDHAF1 (LYRM8)', fn: 'FeS cluster delivery to SDHB via LYR–HSC20', target: 'SDHB [2Fe-2S] + [4Fe-4S]', inh: 'AR (biallelic)', disease: 'Infantile CII deficiency leukoencephalopathy', chr: '19q13.12', highlight: true },
                { factor: 'SDHAF2 (PGL2)', fn: 'FAD insertion into SDHA', target: 'SDHA FAD binding site', inh: 'AD (dominant)', disease: 'Hereditary paraganglioma type 2 (PGL2)', chr: '11q13.1', para: true },
                { factor: 'SDHAF3', fn: 'Protects FeS-SDHB from oxidative damage', target: 'SDHB (post-FeS insertion)', inh: 'AR (biallelic)', disease: 'CII deficiency (similar to SDHAF1)', chr: '1q21.2' },
                { factor: 'SDHAF4', fn: 'SDHA stabilisation after FAD insertion', target: 'SDHA (post-FAD insertion)', inh: 'AR?', disease: 'CII deficiency', chr: 'Unknown' },
              ].map(r => (
                <tr key={r.factor} style={{ background: r.highlight ? LIGHT : r.para ? '#fff8e1' : undefined }}>
                  <td className="fw-bold" style={{ color: r.highlight ? COLOR : r.para ? '#e65100' : undefined }}>
                    {r.factor}
                    {r.highlight && <span className="badge ms-1" style={{ background: COLOR, fontSize: '0.6rem' }}>THIS</span>}
                    {r.para && <span className="badge ms-1 bg-warning text-dark" style={{ fontSize: '0.6rem' }}>PARAGANGLIOMA</span>}
                  </td>
                  <td className="small text-muted">{r.fn}</td>
                  <td className="small text-muted">{r.target}</td>
                  <td className="small" style={{ color: r.inh.includes('AD') ? '#b71c1c' : '#1b5e20' }}>{r.inh}</td>
                  <td className="small" style={{ color: r.highlight ? '#c62828' : r.para ? '#e65100' : '#757575' }}>{r.disease}</td>
                  <td className="font-monospace small">{r.chr}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="🧬 LYR-Motif Family — SDHAF1 vs LYRM7 (CIII)" borderColor={COLOR}>
        <div className="row g-3">
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: COLOR }}>SDHAF1 (LYRM8) — CII Assembly</div>
            <ul className="list-unstyled small">
              <li>• LYR tripeptide recruits HSC20 for SDHB FeS delivery</li>
              <li>• Delivers [2Fe-2S] + [4Fe-4S] to SDHB</li>
              <li>• Target: SDHB (CII iron-sulfur subunit)</li>
              <li>• Deficiency: <strong className="text-danger">Isolated CII deficiency — SDH 10-30%</strong></li>
              <li>• Disease: infantile leukoencephalopathy; brain MRS succinate ↑</li>
              <li>• Chromosome: 19q13.12 | OMIM *612848 / #252011</li>
            </ul>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#1b5e20' }}>LYRM7 — CIII Assembly (5q33.1)</div>
            <ul className="list-unstyled small">
              <li>• LYR tripeptide recruits HSC20 for UQCRFS1 (RISP) FeS delivery</li>
              <li>• Delivers [2Fe-2S] to UQCRFS1 (Rieske iron-sulfur protein of CIII)</li>
              <li>• Target: UQCRFS1 / RISP (CIII Rieske subunit)</li>
              <li>• Deficiency: <strong className="text-info">Isolated CIII deficiency — CII normal</strong></li>
              <li>• Disease: CIII deficiency encephalopathy; different metabolomics</li>
              <li>• Chromosome: 5q33.1 | Different gene, different complex</li>
            </ul>
          </div>
        </div>
        <div className="alert mt-3 mb-0" style={{ background: LIGHT, borderLeft: `4px solid ${COLOR}` }}>
          <strong>Key discriminator:</strong> Enzyme assay — isolated CII deficiency (SDHAF1) vs isolated CIII deficiency (LYRM7).
          Biochemical fingerprint is definitive. WES locus (19q13.12 vs 5q33.1) confirms.
          Brain MRS: SDHAF1 shows succinate peak; LYRM7 does not.
        </div>
      </SectionCard>

      <SectionCard title="🔴 KD Contraindication — Metabolic Mechanism in CII Deficiency" borderColor="#c62828">
        <div className="row g-3">
          <div className="col-md-6">
            <div className="fw-bold small mb-2 text-danger">Why KD is Dangerous in SDHAF1 (CII deficiency)</div>
            <ol className="small">
              <li>KD shifts energy metabolism to fat oxidation (beta-oxidation)</li>
              <li>Beta-oxidation generates FADH2 at each cycle</li>
              <li>FADH2 donates electrons to ubiquinone <strong>EXCLUSIVELY via Complex II (CII)</strong></li>
              <li>SDHAF1 patients have CII deficiency (10–30%) — FADH2 cannot be oxidized</li>
              <li>FADH2 accumulates → ETC backup → lactic acidosis → metabolic crisis</li>
              <li>KD-driven fat oxidation makes the defect much worse, not better</li>
            </ol>
          </div>
          <div className="col-md-6">
            <div className="fw-bold small mb-2" style={{ color: '#2e7d32' }}>Safe Dietary Approach for SDHAF1</div>
            <ul className="list-unstyled small">
              <li>✅ High-complex-carbohydrate diet (glucose → pyruvate → acetyl-CoA; NADH via CI)</li>
              <li>✅ Avoid prolonged fasting (mobilizes fat → FADH2 surge)</li>
              <li>✅ IV dextrose GIR 6-8 mg/kg/min during illness, surgery, or fasting</li>
              <li>✅ Frequent meals; no overnight fast longer than 4-6 hours in infants</li>
              <li>❌ Avoid ketogenic diet</li>
              <li>❌ Avoid medium-chain triglyceride (MCT) oil supplements</li>
            </ul>
          </div>
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
              <span className="badge" style={{ background: c.level.includes('ABSOLUTELY') ? '#7b0000' : c.level.includes('ABSOLUTE') ? '#c62828' : c.level.includes('NOT') ? '#e65100' : '#f57f17' }}>{c.level}</span>
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
export default function SDHAF1Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [definitions, setDef] = useState(null);
  const [err, setErr]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/sdhaf1/overview`).then(r => r.json()),
      fetch(`${API}/api/sdhaf1/breakdown`).then(r => r.json()),
      fetch(`${API}/api/sdhaf1/definitions`).then(r => r.json()),
    ]).then(([ov, bk, def]) => { setOv(ov); setBk(bk); setDef(def); })
      .catch(e => setErr(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-3 flex-wrap gap-2">
        <h4 className="fw-bold mb-0" style={{ color: COLOR }}>🧬 SDHAF1 (LYRM8) — Complex II Deficiency / Infantile Leukoencephalopathy / LYR-Motif SDHB FeS Delivery</h4>
        <span className="badge ms-2" style={{ background: COLOR }}>CII Assembly Factor</span>
        <span className="badge ms-1" style={{ background: '#7b0000' }}>LYR-Motif LYRM8</span>
        <span className="badge ms-1" style={{ background: '#b71c1c' }}>Only CII-Specific FeS Factor</span>
        <span className="badge ms-1 bg-secondary">19q13.12</span>
        <span className="badge ms-1 bg-secondary">OMIM *612848</span>
        <span className="badge ms-1 bg-secondary">AR</span>
        <span className="badge ms-1" style={{ background: '#1b5e20' }}>No Paraganglioma</span>
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
      {tab === 2 && <CIIAssemblyTab data={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
