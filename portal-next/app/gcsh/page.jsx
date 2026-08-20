'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#5d4037';   // dark brown — H-protein / lipoamide / lipoic acid / central carrier
const ACCENT2 = '#b71c1c';   // dark red — HIGH RISK / VPA CI / classic neonatal
const ACCENT3 = '#e65100';   // deep orange — CAUTION / attenuated / triggers
const ACCENT4 = '#1565c0';   // deep blue — treatments / benzoate / LEV
const ACCENT5 = '#4a148c';   // deep purple — GCS mechanism / NMDAr / DXM
const ACCENT6 = '#283593';   // dark indigo — dual block / GLDC-AMT stall / glycine biology
const ACCENT7 = '#1b5e20';   // dark green — H-protein carrier / lipoyl / Lys59

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

function PctBar({ label, pct, color = ACCENT }) {
  const numPct = typeof pct === 'string' ? parseInt(pct) : pct;
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span><span className="text-muted">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${numPct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function Alert({ text, variant = 'warning' }) {
  return (
    <div className={`alert alert-${variant} py-2 mb-2`} style={{ fontSize: 13 }}>
      {text}
    </div>
  );
}

function SectionCard({ title, children, borderColor = ACCENT }) {
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body">
        <h6 className="card-title fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span className="badge me-1" style={{ backgroundColor: color, color: '#fff', fontSize: 11 }}>{text}</span>
  );
}

function CICard({ drug, level, reason, alternative }) {
  const color = level?.includes('HIGH RISK') || level?.includes('ABSOLUTE')
    ? ACCENT2
    : level?.includes('CAUTION')
    ? ACCENT3 : '#546e7a';
  return (
    <div className="card mb-2 shadow-sm" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold small">{drug}</span>
          <Badge text={level?.split('—')[0]?.trim().split(' ').slice(0, 4).join(' ')} color={color} />
        </div>
        <p className="small text-danger mb-1">{reason}</p>
        {alternative && <p className="small text-muted mb-0"><strong>Alternative:</strong> {alternative}</p>}
      </div>
    </div>
  );
}

function TreatmentCard({ drug, level, dose, moa, efficacy, monitoring, gcsh_note }) {
  const isHighRisk = level?.includes('HIGH RISK');
  const isCaution = level?.includes('CAUTION');
  const borderColor = isHighRisk ? ACCENT2 : isCaution ? ACCENT3 : ACCENT4;
  return (
    <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${borderColor}` }}>
      <div className="card-body py-2 px-3">
        <div className="d-flex justify-content-between align-items-start mb-1">
          <span className="fw-bold">{drug}</span>
          <Badge text={level?.split('—')[0]?.trim().replace('Level ', 'Lvl ')} color={borderColor} />
        </div>
        <p className="small mb-1"><strong>Dose:</strong> {dose}</p>
        <p className="small mb-1"><strong>MoA:</strong> {moa}</p>
        <p className="small mb-1"><strong>Efficacy:</strong> {efficacy}</p>
        <p className="small mb-1"><strong>Monitoring:</strong> {monitoring}</p>
        {gcsh_note && (
          <p className="small mb-0 mt-2 p-2 rounded" style={{ backgroundColor: '#efebe9', color: '#3e2723' }}>
            <strong>GCSH Note:</strong> {gcsh_note}
          </p>
        )}
      </div>
    </div>
  );
}

export default function GCSHPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/gcsh/overview`).then(r => r.json()),
      fetch(`${API}/api/gcsh/breakdown`).then(r => r.json()),
      fetch(`${API}/api/gcsh/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefinitions(df);
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-2 text-muted">Loading GCSH dashboard…</p></div>;
  if (error) return <div className="container py-5"><div className="alert alert-danger">Error: {error}</div></div>;
  if (!overview) return null;

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1400 }}>
      {/* Header */}
      <div className="card mb-3 shadow-sm" style={{ borderLeft: `6px solid ${ACCENT}`, background: 'linear-gradient(135deg, #efebe9 0%, #fbe9e7 100%)' }}>
        <div className="card-body py-3">
          <div className="d-flex align-items-start gap-3">
            <div style={{ fontSize: 40 }}>🧬</div>
            <div>
              <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>GCSH Epilepsy — Non-Ketotic Hyperglycinemia (NKH)</h4>
              <p className="mb-1 small"><strong>Gene:</strong> {overview.gene}</p>
              <p className="mb-1 small"><strong>Inheritance:</strong> {overview.inheritance}</p>
              <div className="d-flex gap-2 flex-wrap mt-2">
                <Badge text={`OMIM *${overview.omim_gene}`} color={ACCENT} />
                <Badge text={`Disease #${overview.omim_disease}`} color={ACCENT6} />
                <Badge text={overview.locus} color={ACCENT7} />
                <Badge text="H-protein (GCS Carrier Step 2)" color={ACCENT5} />
                <Badge text="~1% of NKH" color={ACCENT4} />
                <Badge text="AR Biallelic LOF" color="#546e7a" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* H-Protein Dual Block Alert */}
      <div className="alert mb-3 py-2" style={{ backgroundColor: '#fbe9e7', border: `1px solid ${ACCENT}`, fontSize: 13 }}>
        <strong>⚗️ GCSH H-Protein Dual GCS Block:</strong> H-protein (GCSH) is the CENTRAL CARRIER of GCS — lipoamide swinging arm (Lys59) shuttles aminomethyl group from P-protein (GLDC) to T-protein (AMT). GCSH LOF → <strong>DUAL BLOCK</strong>: (1) P-protein cannot discharge aminomethyl (no H-protein) → Step 1 stalls; (2) T-protein has no loaded H-protein substrate → Step 3 stalls simultaneously. Most complete GCS block of all NKH types. Biochemically IDENTICAL to GLDC-NKH and AMT-NKH — gene panel mandatory. H-protein has NO catalytic activity — diagnosis by sequencing only. ~1% of NKH; NO founder allele; ~20–50 cases worldwide 2026.
      </div>

      {/* KPI strip */}
      <div className="row mb-3">
        <KPI label="Cohort Size" value={overview.cohort_size} color={ACCENT} />
        <KPI label="Epilepsy" value={`${overview.epilepsy_pct}%`} color={ACCENT2} />
        <KPI label="DRE" value={`${overview.dre_pct}%`} color={ACCENT2} />
        <KPI label="Infantile Spasms" value={`${overview.is_pct}%`} color={ACCENT3} />
        <KPI label="Burst-Suppression" value={`${overview.burst_suppression_pct}%`} color={ACCENT5} />
        <KPI label="Classic Neonatal" value={`${overview.classic_pct}%`} color={ACCENT2} />
        <KPI label="Attenuated" value={`${overview.attenuated_pct}%`} color={ACCENT3} />
        <KPI label="On Benzoate" value={`${overview.benzoate_pct}%`} color={ACCENT4} />
        <KPI label="On DXM" value={`${overview.dxm_pct}%`} color={ACCENT5} />
        <KPI label="Folate Low" value={`${overview.folate_low_pct}%`} color={ACCENT7} />
        <KPI label="Avg Plasma Gly" value={`${overview.avg_plasma_glycine} µM`} color={ACCENT6} />
        <KPI label="Avg CSF Ratio" value={overview.avg_csf_plasma_ratio} color={ACCENT} />
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`} onClick={() => setTab(t)}
              style={tab === t ? { color: ACCENT, borderBottomColor: ACCENT } : {}}>{t}</button>
          </li>
        ))}
      </ul>

      {/* TAB: Overview */}
      {tab === 'Overview' && overview && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Phenotype Classes" borderColor={ACCENT}>
              {overview.phenotype_classes?.map((pc, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between mb-1">
                    <strong className="small">{pc.name}</strong>
                    <Badge text={`${pc.pct}%`} color={ACCENT} />
                  </div>
                  <p className="small text-muted mb-1">{pc.description}</p>
                  <div className="small"><strong>EEG:</strong> {pc.eeg}</div>
                  <div className="small"><strong>Seizures:</strong> {pc.seizure_types}</div>
                  <div className="small"><strong>Outcome:</strong> {pc.outcome}</div>
                  <div className="small"><strong>CSF ratio:</strong> {pc.csf_ratio}</div>
                  {i < overview.phenotype_classes.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>

            <SectionCard title="Etiologies — 40-Patient GCSH-NKH Cohort" borderColor={ACCENT7}>
              {overview.etiologies?.map((e, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between">
                    <span className="small fw-bold">{e.etiology}</span>
                    <Badge text={`${e.pct}% (n=${e.n})`} color={ACCENT7} />
                  </div>
                  <div className="small text-muted">CSF ratio: {e.csf_plasma_ratio} · {e.outcome}</div>
                  {i < overview.etiologies.length - 1 && <hr className="my-1" />}
                </div>
              ))}
            </SectionCard>
          </div>

          <div className="col-md-6">
            <SectionCard title="Key GCSH-NKH Concepts" borderColor={ACCENT5}>
              {overview.key_concepts?.map((c, i) => (
                <div key={i} className="small mb-1">
                  <span style={{ color: ACCENT5 }}>▶</span> {c}
                </div>
              ))}
            </SectionCard>

            <div className="card mb-3 shadow-sm" style={{ borderLeft: `4px solid ${ACCENT2}` }}>
              <div className="card-body py-2 px-3">
                <h6 className="fw-bold mb-2" style={{ color: ACCENT2 }}>Unique GCSH Features</h6>
                <div className="small mb-1"><strong>Worldwide cases 2026:</strong> {overview.worldwide_cases_2026}</div>
                <div className="small mb-1"><strong>Founder allele:</strong> {overview.founder_allele}</div>
                <div className="small mb-1"><strong>Unique mechanism:</strong> {overview.unique_mechanism}</div>
                <div className="small mb-1"><strong>Pathognomonic sign:</strong> {overview.pathognomonic_sign}</div>
                <div className="small mb-1"><strong>Diagnostic biomarker:</strong> {overview.diagnostic_biomarker}</div>
              </div>
            </div>

            <SectionCard title="Evidence Standards" borderColor="#546e7a">
              {overview.standards?.map((s, i) => (
                <div key={i} className="small mb-1">📚 {s}</div>
              ))}
            </SectionCard>

            <SectionCard title="GCS Pathway Summary (GCSH = Central Carrier Step 2)" borderColor={ACCENT6}>
              <div className="small">
                <p><strong>Step 1 (P-protein / GLDC):</strong> Glycine + H-protein(ox) → CO₂ + aminomethyl-H-protein</p>
                <p className="p-2 rounded" style={{ backgroundColor: '#efebe9', color: '#3e2723' }}>
                  <strong>Step 2 (H-protein / GCSH) ← DEFECTIVE CARRIER IN GCSH-NKH:</strong> H-protein (lipoamide Lys59 swinging arm) shuttles aminomethyl group from P-protein to T-protein. GCSH LOF → DUAL BLOCK: P-protein stalls (no H-protein to receive) AND T-protein stalls (no loaded H-protein substrate).
                </p>
                <p><strong>Step 3 (T-protein / AMT):</strong> Aminomethyl-H-protein + THF → 5,10-methyleneTHF + NH₄⁺ + H-protein(ox). [Cannot function in GCSH-NKH — no loaded H-protein]</p>
                <p><strong>Step 4 (L-protein / DLD):</strong> H-protein(red) + NAD⁺ → H-protein(ox) + NADH [cannot act on absent H-protein]</p>
                <p><strong>Net reaction:</strong> Glycine + THF + NAD⁺ → 5,10-methyleneTHF + CO₂ + NH₄⁺ + NADH</p>
                <p className="text-danger mb-0"><strong>GCSH LOF consequence:</strong> DUAL BLOCK at both Step 1 (GLDC stalls upstream) AND Step 3 (AMT stalls downstream) → GCS completely inoperable → glycine accumulates (identical to GLDC-NKH and AMT-NKH) + 5,10-methyleneTHF NOT produced</p>
              </div>
            </SectionCard>
          </div>
        </div>
      )}

      {/* TAB: Patients & Phenotype */}
      {tab === 'Patients & Phenotype' && breakdown && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Phenotype Distribution" borderColor={ACCENT}>
              {breakdown.phenotype_class_distribution?.map((pc, i) => (
                <PctBar key={i} label={`${pc.class} (n=${pc.n})`} pct={pc.pct} color={pc.colour} />
              ))}
            </SectionCard>

            <SectionCard title="CSF:Plasma Glycine Ratio Histogram" borderColor={ACCENT5}>
              <p className="small text-muted mb-2">Normal &lt;0.02 · Diagnostic ≥0.08 · Attenuated 0.08–0.18 · Classic 0.20–0.55</p>
              {breakdown.csf_plasma_ratio_histogram?.map((r, i) => (
                <PctBar key={i} label={r.range} pct={r.pct} color={ACCENT5} />
              ))}
            </SectionCard>

            <SectionCard title="Plasma Glycine Histogram" borderColor={ACCENT6}>
              <p className="small text-muted mb-2">Target on benzoate: &lt;500 µmol/L (ideally &lt;300 µmol/L)</p>
              {breakdown.plasma_glycine_histogram?.map((r, i) => (
                <PctBar key={i} label={r.range} pct={r.pct} color={ACCENT6} />
              ))}
            </SectionCard>
          </div>

          <div className="col-md-6">
            <SectionCard title="Treatment Counts (40 patients)" borderColor={ACCENT4}>
              {breakdown.treatment_counts && Object.entries(breakdown.treatment_counts).map(([drug, n], i) => (
                <PctBar key={i} label={`${drug} (n=${n})`} pct={Math.round(100 * n / 40)} color={ACCENT4} />
              ))}
            </SectionCard>

            <SectionCard title="Patient Profiles (Top 10 by CSF ratio)" borderColor={ACCENT}>
              <div style={{ overflowX: 'auto' }}>
                <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
                  <thead>
                    <tr>
                      <th>ID</th><th>Sex</th><th>Class</th><th>CSF Ratio</th><th>Plasma Gly</th><th>DRE</th><th>Folate↓</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.per_patient_profiles?.slice(0, 10).map((p, i) => (
                      <tr key={i} style={{ color: p.drug_resistant ? ACCENT2 : 'inherit' }}>
                        <td>{p.patient_id}</td>
                        <td>{p.sex}</td>
                        <td>{p.phenotype_class.split(' ')[0]}</td>
                        <td>{p.csf_plasma_ratio.toFixed(3)}</td>
                        <td>{p.plasma_glycine_umol}</td>
                        <td>{p.drug_resistant ? '⚠️' : '✓'}</td>
                        <td>{p.folate_low ? '⚠️' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>
          </div>
        </div>
      )}

      {/* TAB: Seizures & Triggers */}
      {tab === 'Seizures & Triggers' && breakdown && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Seizure Types" borderColor={ACCENT2}>
              {breakdown.seizure_type_distribution?.map((s, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between">
                    <strong className="small">{s.type}</strong>
                    <Badge text={`${s.pct}%`} color={ACCENT2} />
                  </div>
                  <div className="small"><strong>EEG:</strong> {s.eeg}</div>
                  <div className="small"><strong>Semiology:</strong> {s.semiology}</div>
                  <div className="small text-info"><strong>Tips:</strong> {s.tips}</div>
                  {i < breakdown.seizure_type_distribution.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>
          </div>
          <div className="col-md-6">
            <SectionCard title="Seizure Triggers" borderColor={ACCENT3}>
              {breakdown.trigger_distribution?.map((t, i) => (
                <div key={i} className="mb-3">
                  <div className="d-flex justify-content-between">
                    <strong className="small">{t.trigger}</strong>
                    <Badge text={`${t.pct}%`} color={ACCENT3} />
                  </div>
                  <PctBar label="" pct={t.pct} color={ACCENT3} />
                  <p className="small text-muted mb-0">{t.mechanism}</p>
                  {i < breakdown.trigger_distribution.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>

            <SectionCard title="Monitoring Parameters" borderColor={ACCENT7}>
              {breakdown.monitoring?.map((m, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between">
                    <strong className="small">{m.parameter}</strong>
                    <span className="badge bg-secondary small">{m.frequency}</span>
                  </div>
                  <div className="small text-muted">{m.target}</div>
                  {i < breakdown.monitoring.length - 1 && <hr className="my-1" />}
                </div>
              ))}
            </SectionCard>
          </div>
        </div>
      )}

      {/* TAB: Treatments */}
      {tab === 'Treatments' && breakdown && (
        <div className="row">
          <div className="col-md-6">
            <Alert text="⚠️ VPA HIGH RISK in ALL NKH (GLDC/AMT/GCSH): In GCSH-NKH, H-protein has no catalytic activity so VPA cannot inhibit it directly — but raises glycine via secondary mechanisms with equivalent clinical risk. Avoid. Use LEV + CLB + DXM + benzoate." variant="danger" />
            <Alert text="⚠️ VGB HIGH RISK for IS in GCSH-NKH: May raise glycine via GABA-glycine co-transporter + retinal toxicity. ACTH preferred for IS." variant="warning" />
            {breakdown.contraindications?.map((ci, i) => (
              <CICard key={i} {...ci} />
            ))}
          </div>
          <div className="col-md-6">
            {definitions?.treatments?.map((tx, i) => (
              <TreatmentCard key={i} {...tx} />
            ))}
          </div>
        </div>
      )}

      {/* TAB: Definitions */}
      {tab === 'Definitions' && definitions && (
        <div className="row">
          <div className="col-md-6">
            <SectionCard title="Gene Card — GCSH (H-protein, 16q23.2)" borderColor={ACCENT}>
              {definitions.gene_card && Object.entries(definitions.gene_card).map(([k, v], i) => (
                <div key={i} className="mb-1">
                  <strong className="small" style={{ color: ACCENT }}>{k.replace(/_/g, ' ').toUpperCase()}:</strong>{' '}
                  <span className="small">{v}</span>
                </div>
              ))}
            </SectionCard>

            <SectionCard title="GCS Pathway — GCSH = H-Protein (Central Carrier, Step 2)" borderColor={ACCENT6}>
              <p className="small fw-bold mb-2">{definitions.pathway?.net_reaction}</p>
              <p className="small text-danger mb-2">{definitions.pathway?.gcsh_lof_consequence}</p>
              {definitions.pathway?.steps?.map((s, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ backgroundColor: i === 1 ? '#efebe9' : '#f5f5f5' }}>
                  <div className="small fw-bold" style={{ color: i === 1 ? ACCENT : '#333' }}>
                    Step {s.step}: {s.enzyme} ({s.gene}) {s.cofactor && `— Cofactor: ${s.cofactor}`}
                  </div>
                  <div className="small font-monospace">{s.reaction}</div>
                  <div className="small text-muted">{s.clinical}</div>
                </div>
              ))}
            </SectionCard>

            <SectionCard title="Biomarkers" borderColor={ACCENT5}>
              {definitions.biomarkers?.map((bm, i) => (
                <div key={i} className="mb-3">
                  <strong className="small">{bm.marker}</strong>
                  <div className="small"><strong>Method:</strong> {bm.method}</div>
                  <div className="small"><strong>Normal:</strong> {bm.reference_range} · <strong>GCSH-NKH:</strong> {bm.nkh_range}</div>
                  <div className="small text-muted">{bm.notes}</div>
                  {i < definitions.biomarkers.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>
          </div>

          <div className="col-md-6">
            <SectionCard title="Key GCSH-NKH Concepts (11)" borderColor={ACCENT5}>
              {definitions.key_concepts?.map((c, i) => (
                <div key={i} className="mb-3">
                  <strong className="small" style={{ color: ACCENT5 }}>{c.term}</strong>
                  <p className="small text-muted mb-0">{c.definition}</p>
                  {i < definitions.key_concepts.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>

            <SectionCard title="Diagnostic Thresholds" borderColor={ACCENT7}>
              <div style={{ overflowX: 'auto' }}>
                <table className="table table-sm" style={{ fontSize: 12 }}>
                  <thead><tr><th>Parameter</th><th>Value</th><th>Clinical</th></tr></thead>
                  <tbody>
                    {definitions.thresholds?.map((t, i) => (
                      <tr key={i}><td>{t.parameter}</td><td className="fw-bold" style={{ color: ACCENT }}>{t.value}</td><td className="text-muted">{t.clinical}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </SectionCard>

            <SectionCard title="Differential Diagnosis" borderColor={ACCENT3}>
              {definitions.differential_diagnosis?.map((d, i) => (
                <div key={i} className="mb-2">
                  <strong className="small">{d.condition}</strong>
                  <p className="small text-muted mb-0">{d.distinction}</p>
                  {i < definitions.differential_diagnosis.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>

            <SectionCard title="Lifecycle" borderColor={ACCENT4}>
              {breakdown?.lifecycle?.map((l, i) => (
                <div key={i} className="mb-2">
                  <div className="d-flex justify-content-between">
                    <strong className="small">{l.stage}</strong>
                    <Badge text={l.age} color={ACCENT4} />
                  </div>
                  <p className="small text-muted mb-0">{l.description}</p>
                  {i < breakdown.lifecycle.length - 1 && <hr className="my-2" />}
                </div>
              ))}
            </SectionCard>

            <SectionCard title="References" borderColor="#546e7a">
              {definitions.references?.map((r, i) => (
                <div key={i} className="small mb-1">📚 {r}</div>
              ))}
            </SectionCard>
          </div>
        </div>
      )}
    </div>
  );
}
