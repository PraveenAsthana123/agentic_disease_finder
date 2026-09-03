'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Variants & Cohort', 'DDx & Management', 'Definitions'];
const COLOR  = '#1a237e';   // deep indigo — tRNA-Phe / CPEO-dominant / FIRST tRNA
const LIGHT  = '#e8eaf6';
const COLOR2 = '#283593';   // medium indigo — CPEO / myopathy
const COLOR3 = '#b71c1c';   // dark red — absolute CIs / severe
const COLOR4 = '#1b5e20';   // dark green — biochemical fingerprint / OXPHOS
const COLOR5 = '#e65100';   // deep orange — FARS2 neonatal DDx / cardiomyopathy alerts

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
  const pheno_dist = data.phenotype_distribution || [];
  const mol_feats = data.key_molecular_features || [];
  const alerts = data.clinical_alerts || [];
  const hmap = data.heteroplasmy_clinical_map || [];

  return (
    <div>
      {/* Gene header */}
      <div className="p-3 mb-4 rounded" style={{ background: LIGHT, borderLeft: `5px solid ${COLOR}` }}>
        <div className="d-flex flex-wrap align-items-start gap-3">
          <div>
            <h4 className="fw-bold mb-1" style={{ color: COLOR }}>MT-TF — tRNA-Phe</h4>
            <div className="text-muted small">
              <span className="badge me-1" style={{ background: COLOR }}>OMIM *590070</span>
              <span className="badge me-1" style={{ background: COLOR2 }}>Combined CI+CIV Deficiency</span>
              <span className="badge me-1 bg-dark">Phe codons: UUU/UUC · GAA anticodon</span>
              <span className="badge me-1" style={{ background: COLOR4 }}>H-STRAND · No NGS Pitfall</span>
              <span className="badge me-1 bg-secondary">FIRST tRNA in mitochondrial genome</span>
            </div>
            <p className="mt-2 mb-0 small">
              MT-TF encodes mitochondrial tRNA-Phe (GAA anticodon, 71 nt) — <strong>H-strand</strong> rCRS 577–647 —
              the <strong>FIRST tRNA gene</strong> of the human mitochondrial genome, immediately after the D-loop (rCRS 576).
              Mutations cause <strong>combined CI + CIV deficiency</strong> (mt-translation fingerprint: CII NORMAL).
              CPEO + myopathy are the dominant phenotypes (~78–90%). Nuclear DDx:
              <strong style={{ color: COLOR5 }}> FARS2 (mt-Phe-tRNA Synthetase) — causes NEONATAL epileptic encephalopathy</strong>
              (not adult CPEO) — phenotypically opposite but biochemically identical fingerprint.
            </p>
          </div>
        </div>
      </div>

      {/* FIRST tRNA + FARS2 banner */}
      <div className="alert mb-4" style={{ background: '#e8eaf6', border: `2px solid ${COLOR}`, borderRadius: 8 }}>
        <div className="fw-bold" style={{ color: COLOR }}>🧬 FIRST tRNA Gene — D-loop Adjacent — FARS2 Nuclear DDx Critical</div>
        <div className="small mt-1">
          MT-TF (577–647) is the <strong>first gene after the D-loop</strong> — large deletions extending from MT-TF into the D-loop
          can disrupt the heavy-strand replication origin (OH), reducing mtDNA copy number.
          <span style={{ color: COLOR5 }}> FARS2 nuclear DDx: biallelic AR FARS2 causes <strong>neonatal epileptic encephalopathy</strong> (PEBEI),
          not adult CPEO — the same CI+CIV biochemical fingerprint but radically different age of onset and phenotype.
          Always confirm maternal inheritance pattern before attributing to MT-TF vs FARS2.</span>
        </div>
      </div>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Patients" value={s.n_patients} />
        <KPI label="Mean Heteroplasmy (blood)" value={`${s.avg_heteroplasmy_blood_pct}%`} color={COLOR2} />
        <KPI label="Mean CI Activity" value={`${s.avg_ci_activity_pct_normal}%`} color={COLOR3} />
        <KPI label="Mean CIV Activity" value={`${s.avg_civ_activity_pct_normal}%`} color={COLOR3} />
        <KPI label="CII (NORMAL)" value={`${s.avg_cii_activity_pct_normal}%`} color={COLOR4} />
        <KPI label="CPEO" value={`${s.pct_cpeo}%`} color={COLOR2} />
        <KPI label="Myopathy" value={`${s.pct_myopathy}%`} color={COLOR} />
        <KPI label="Cardiomyopathy" value={`${s.pct_cardiomyopathy}%`} color={COLOR5} />
        <KPI label="SNHL" value={`${s.pct_snhl}%`} color={COLOR2} />
        <KPI label="Diabetes (low)" value={`${s.pct_diabetes_mellitus}%`} color={COLOR4} />
        <KPI label="RRF" value={`${s.pct_ragged_red_fibres}%`} color={COLOR} />
        <KPI label="Mean Onset (yr)" value={s.avg_age_onset_yr} />
      </div>

      {/* FIRST tRNA + D-loop adjacency feature */}
      <SectionCard title="FIRST tRNA Gene — D-loop Adjacency — H-strand Encoding" borderColor={COLOR}>
        <div className="row">
          <div className="col-md-6">
            <div className="p-3 rounded mb-2" style={{ background: LIGHT, border: `2px solid ${COLOR}` }}>
              <div className="fw-bold small" style={{ color: COLOR }}>FIRST tRNA — H-strand Encoded</div>
              <div className="small mt-1">MT-TF (577–647) is the first gene after the D-loop on the H-strand</div>
              <div className="small">H-strand encoding: no L-strand NGS coverage pitfall (contrast MT-TE, MT-TP, MT-ND6)</div>
              <div className="small text-success fw-bold mt-1">Still requires dedicated mtDNA panel — WES misses RNA genes</div>
            </div>
          </div>
          <div className="col-md-6">
            <div className="p-3 rounded mb-2" style={{ background: '#fff3e0', border: `2px solid ${COLOR5}` }}>
              <div className="fw-bold small" style={{ color: COLOR5 }}>D-loop Adjacency — mtDNA Copy Number Risk</div>
              <div className="small mt-1">D-loop (16024–576) contains OH (heavy-strand replication origin)</div>
              <div className="small">Large deletions spanning MT-TF into D-loop → OH deletion → reduced mtDNA copy number</div>
              <div className="small text-danger fw-bold mt-1">mtDNA quantification (ND1/B2M ratio) mandatory in KSS MT-TF deletion</div>
            </div>
          </div>
        </div>
        <p className="small text-muted mb-0 mt-2">
          MT-TF flanks the D-loop on its 3&apos; side (D-loop ends at 576; MT-TF starts at 577).
          Nuclear DDx FARS2 (mt-Phe-tRNA Synthetase) causes NEONATAL epileptic encephalopathy — AR biallelic, WES-detectable.
          Maternal inheritance history and age of onset are the critical first-line distinguishers.
        </p>
      </SectionCard>

      {/* Heteroplasmy map */}
      <SectionCard title="Heteroplasmy–Phenotype Map (blood; muscle underestimates by 10–15%)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Blood heteroplasmy / Variant</th><th>Clinical phenotype</th><th>Management</th>
            </tr></thead>
            <tbody>
              {hmap.map((r, i) => (
                <tr key={i}>
                  <td className="fw-bold">{r.range}</td>
                  <td>{r.phenotype}</td>
                  <td className="text-muted">{r.management}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Clinical alerts */}
      <SectionCard title="Clinical Alerts — Contraindications" borderColor={COLOR3}>
        <div className="row">
          {alerts.map((a, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#fce4ec', border: '1px solid #c62828' }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{a.alert}</div>
                <div className="small text-muted">{a.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Phenotype distribution */}
      <SectionCard title="Phenotype Distribution (40-patient cohort, seed-803)" borderColor={COLOR}>
        {pheno_dist.map((p, i) => (
          <Bar key={i} label={`${p.phenotype} (n=${p.count})`} value={p.pct} color={COLOR} />
        ))}
      </SectionCard>

      {/* Molecular features */}
      <SectionCard title="Key Molecular Features" borderColor={COLOR4}>
        <ul className="small mb-0">
          {mol_feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>

      {/* Cohort summary */}
      <SectionCard title="Cohort Summary" borderColor={COLOR2}>
        <ul className="small mb-0">
          {feats.map((f, i) => <li key={i} className="mb-1">{f}</li>)}
        </ul>
      </SectionCard>
    </div>
  );
}

// ── Tab: Variants & Cohort ─────────────────────────────────────────────────────
function VariantsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const vs = data.variant_summaries || [];
  const pts = data.per_patient || [];
  const triggers = data.trigger_rates || [];
  const txs = data.treatment_info || [];
  const bf = data.biochemical_fingerprint || {};

  return (
    <div>
      {/* Variant summaries */}
      <SectionCard title="Variant Summaries by Mutation" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Variant</th><th>Region</th><th>N</th>
              <th>Het%</th><th>CI%</th><th>CIV%</th>
              <th>CPEO%</th><th>Myopathy%</th><th>Cardio%</th><th>SNHL%</th><th>DM%</th>
            </tr></thead>
            <tbody>
              {vs.map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold">{v.variant}</td>
                  <td>{v.region}</td>
                  <td>{v.n}</td>
                  <td>{v.avg_heteroplasmy_blood_pct}</td>
                  <td style={{ color: COLOR3 }}>{v.avg_ci_activity_pct}</td>
                  <td style={{ color: COLOR3 }}>{v.avg_civ_activity_pct}</td>
                  <td>{v.pct_cpeo}%</td>
                  <td>{v.pct_myopathy}%</td>
                  <td style={{ color: COLOR5 }}>{v.pct_cardiomyopathy}%</td>
                  <td>{v.pct_snhl}%</td>
                  <td>{v.pct_diabetes_mellitus}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          {vs.map((v, i) => (
            <div key={i} className="mb-2 p-2 rounded" style={{ background: LIGHT }}>
              <span className="fw-bold small me-2">{v.variant}</span>
              <span className="text-muted small">{v.note}</span>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Biochemical fingerprint */}
      <SectionCard title="Biochemical Fingerprint (BN-PAGE / Spectrophotometry)" borderColor={COLOR4}>
        <div className="row">
          {[
            { label: 'CI Activity', val: bf.CI_pct_normal, color: COLOR3 },
            { label: 'CIV Activity', val: bf.CIV_pct_normal, color: COLOR3 },
            { label: 'CII Activity (NORMAL)', val: bf.CII_pct_normal, color: COLOR4 },
          ].map((x, i) => (
            <div className="col-md-4 mb-2" key={i}>
              <div className="p-2 rounded text-center" style={{ background: LIGHT }}>
                <div className="fw-bold" style={{ color: x.color }}>{x.val}%</div>
                <div className="small text-muted">{x.label}</div>
              </div>
            </div>
          ))}
        </div>
        <div className="small mt-2">
          <div><strong>Pattern:</strong> {bf.pattern}</div>
          <div><strong>BN-PAGE:</strong> {bf.BN_PAGE}</div>
          <div><strong>Histochem:</strong> {bf.muscle_histochemistry}</div>
          {bf.H_strand_note && (
            <div className="mt-2 p-2 rounded" style={{ background: '#e8f5e9', border: `1px solid ${COLOR4}` }}>
              <strong style={{ color: COLOR4 }}>ℹ H-Strand Note (No NGS Pitfall):</strong> {bf.H_strand_note}
            </div>
          )}
        </div>
      </SectionCard>

      {/* Per-patient table (first 20) */}
      <SectionCard title="Per-Patient Cohort Data (40 patients, seed-803 — first 20 shown)" borderColor={COLOR2}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>ID</th><th>Variant</th><th>Sex</th><th>Onset</th>
              <th>Het%</th><th>CI%</th><th>CIV%</th><th>CII%</th>
              <th>Lactate</th><th>CPEO</th><th>Myo</th><th>Cardio</th><th>SNHL</th><th>DM</th><th>RRF</th><th>Leigh</th>
            </tr></thead>
            <tbody>
              {pts.slice(0, 20).map((p, i) => (
                <tr key={i} style={{ background: p.cardiomyopathy ? '#fff3e0' : 'inherit' }}>
                  <td className="fw-bold">{p.id}</td>
                  <td>{p.variant}</td>
                  <td>{p.sex}</td>
                  <td>{`${p.age_onset_yr}yr`}</td>
                  <td>{p.heteroplasmy_blood_pct}%</td>
                  <td style={{ color: COLOR3 }}>{p.ci_pct}</td>
                  <td style={{ color: COLOR3 }}>{p.civ_pct}</td>
                  <td style={{ color: COLOR4 }}>{p.cii_pct}</td>
                  <td>{p.lactate_mmol_L}</td>
                  <td>{p.cpeo ? '✓' : '—'}</td>
                  <td>{p.myopathy ? '✓' : '—'}</td>
                  <td style={{ color: COLOR5 }}>{p.cardiomyopathy ? '♥' : '—'}</td>
                  <td>{p.snhl ? '✓' : '—'}</td>
                  <td>{p.diabetes_mellitus ? '✓' : '—'}</td>
                  <td>{p.ragged_red_fibres ? '✓' : '—'}</td>
                  <td>{p.leigh_like ? '✓' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small text-muted mt-1">♥ = cardiomyopathy present (annual echo + Holter required)</div>
      </SectionCard>

      {/* Trigger rates */}
      <SectionCard title="Crisis Trigger Rates" borderColor={COLOR3}>
        {triggers.map((t, i) => (
          <Bar key={i} label={t.trigger} value={t.pct} color={COLOR3} />
        ))}
      </SectionCard>

      {/* Treatments */}
      <SectionCard title="Treatment Summary" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Agent</th><th>Evidence</th><th>Note</th>
            </tr></thead>
            <tbody>
              {txs.map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.agent}</td>
                  <td><span className="badge" style={{ background: COLOR }}>{t.evidence}</span></td>
                  <td className="text-muted">{t.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: DDx & Management ──────────────────────────────────────────────────────
function DDxTab({ bdata }) {
  if (!bdata) return <p className="text-muted">Loading…</p>;
  const ddx = bdata.ddx_comparison || [];
  const ci_info = bdata.contraindication_info || [];

  return (
    <div>
      {/* DDx table */}
      <SectionCard title="Differential Diagnosis — MT-TF vs Key Mimics" borderColor={COLOR}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Gene / Disease</th><th>Primary Disease</th><th>OXPHOS</th><th>Key Distinguisher from MT-TF</th>
            </tr></thead>
            <tbody>
              {ddx.map((d, i) => (
                <tr key={i} style={{ background: d.gene.includes('FARS2') ? '#fff8e1' : 'inherit' }}>
                  <td className="fw-bold">{d.gene}</td>
                  <td>{d.disease}</td>
                  <td>{d.oxphos}</td>
                  <td className="text-muted">{d.distinguisher}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small fw-bold mt-2 mb-0" style={{ color: COLOR5 }}>
          ⚠ FARS2 (mt-Phe-tRNA Synthetase) is the MOST IMPORTANT nuclear DDx — same CI+CIV biochemical fingerprint as MT-TF.
          CRITICAL DISTINCTION: FARS2 causes NEONATAL/INFANTILE epileptic encephalopathy (NOT adult CPEO).
          If a neonate/infant presents with CI+CIV deficiency + seizures + Leigh-like MRI, consider FARS2 (WES) BEFORE assuming MT-TF.
          Maternal inheritance history and age of onset are the decisive distinguishers.
        </p>
      </SectionCard>

      {/* Contraindication details */}
      <SectionCard title="Contraindications — Detailed Rationale" borderColor={COLOR3}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead style={{ background: LIGHT }}><tr>
              <th>Agent</th><th>Category</th><th>Rationale</th>
            </tr></thead>
            <tbody>
              {ci_info.map((c, i) => (
                <tr key={i}>
                  <td className="fw-bold">{c.agent}</td>
                  <td><span className="badge" style={{ background: c.category.includes('ABSOLUTE') ? COLOR3 : '#e65100' }}>{c.category}</span></td>
                  <td className="text-muted">{c.rationale}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* KSS (large deletion) management */}
      <SectionCard title="Large Deletion / KSS Protocol (MT-TF + D-loop spanning)" borderColor={COLOR2}>
        <div className="row">
          {[
            { step: 'Pacemaker — AV block', detail: 'Holter + EP study if PR >200ms or Mobitz I; pacemaker if Mobitz II / complete block; sudden death risk without pacemaker in KSS AV block; deletions spanning MT-TF into D-loop → additional mtDNA copy number reduction' },
            { step: 'mtDNA copy number — D-loop deletion', detail: 'MT-TF-spanning deletions extending into D-loop can delete the OH replication origin → reduced mtDNA copy number; quantify with ND1/B2M ratio on muscle biopsy; copy number <20% normal = severe depletion risk' },
            { step: 'Ophthalmology — Retinal pigmentation', detail: 'Pigmentary retinopathy ~68% KSS; annual fundoscopy; Amsler grid home monitoring; no curative treatment; low vision aids; worsens over decades' },
            { step: 'Endocrinology — DM + endocrinopathies', detail: 'DM (insulin-dependent) in ~12% KSS MT-TF deletion; hypothyroidism; hypoparathyroidism (tetany risk); annual screen: glucose, TFT, calcium, PTH' },
            { step: 'Avoid Fasting + GIR protocol', detail: 'Continuous NG/IV feeds during illness; GIR 6–8 mg/kg/min; KSS large deletion: multi-OXPHOS → energy crisis with ANY fasting; NEVER fast perioperatively' },
            { step: 'Genetic — Mostly sporadic', detail: 'Large deletions usually sporadic; maternal recurrence risk low but not zero; mtDNA deletion analysis (Southern/long-range PCR/NanoSeq) on muscle biopsy required; blood may underrepresent deletion burden' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: LIGHT, border: `1px solid ${COLOR2}` }}>
                <div className="fw-bold small" style={{ color: COLOR2 }}>{item.step}</div>
                <div className="small text-muted">{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Emergency protocol */}
      <SectionCard title="Acute Crisis Protocol" borderColor={COLOR3}>
        <div className="row">
          {[
            { step: '1. GIR 6–8 mg/kg/min', detail: 'IV dextrose — NEVER fast; maintain normoglycaemia; prevents catabolism and lactate surge' },
            { step: '2. IV Thiamine 10-20 mg/kg', detail: 'Empiric — PDH/KGDH cofactor; give BEFORE glucose if Wernicke risk; continue oral maintenance' },
            { step: '3. Avoid Propofol', detail: 'Use sevoflurane — PRIS risk amplified in CI+CIV deficiency; propofol ABSOLUTE CI' },
            { step: '4. Cardiac monitoring', detail: 'Continuous ECG in large deletion KSS; AV block can emerge acutely; temporary pacing readiness; not routine in CPEO-only phenotype' },
            { step: '5. Bicarbonate if pH <7.2', detail: 'Target pH >7.2; do not fully correct lactic acidosis; treat underlying trigger; isotonic bicarbonate preferred' },
            { step: '6. Use LEV for seizures', detail: 'VPA ABSOLUTE CI; LEV first-line; if already on VPA → taper rapidly, transition to LEV + bicarb cover for LA risk during transition' },
          ].map((item, i) => (
            <div className="col-md-6 mb-2" key={i}>
              <div className="p-2 rounded" style={{ background: '#fce4ec', border: `1px solid ${COLOR3}` }}>
                <div className="fw-bold small" style={{ color: COLOR3 }}>{item.step}</div>
                <div className="small text-muted">{item.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}

// ── Tab: Definitions ───────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const gb = data.gene_biology || {};
  const ct = data.clinical_terms || {};
  const ph = data.pharmacology || {};
  const refs = data.key_references || [];

  return (
    <div>
      <SectionCard title="Gene Biology" borderColor={COLOR4}>
        {Object.entries(gb).map(([k, v], i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR4 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Clinical Terms" borderColor={COLOR2}>
        {Object.entries(ct).map(([k, v], i) => (
          <div key={i} className="mb-3">
            <div className="fw-bold small" style={{ color: COLOR2 }}>{k.replace(/_/g, ' ')}</div>
            <div className="small text-muted">{v}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Pharmacology" borderColor={COLOR3}>
        {ph.preferred_aed && (
          <div className="mb-2">
            <span className="fw-bold small">Preferred AED: </span>
            <span className="small">{ph.preferred_aed}</span>
          </div>
        )}
        {ph.cardiac_management && (
          <div className="mb-2">
            <span className="fw-bold small" style={{ color: COLOR5 }}>Cardiac Management: </span>
            <span className="small">{ph.cardiac_management}</span>
          </div>
        )}
        {ph.emergency_protocol && (
          <div className="mb-2">
            <span className="fw-bold small">Emergency Protocol: </span>
            <span className="small">{ph.emergency_protocol}</span>
          </div>
        )}
        {ph.anaesthetic_guidance && (
          <div className="mb-2">
            <span className="fw-bold small">Anaesthetic Guidance: </span>
            <span className="small">{ph.anaesthetic_guidance}</span>
          </div>
        )}
        {ph.absolute_ci && (
          <div className="mt-3">
            <div className="fw-bold small mb-2" style={{ color: COLOR3 }}>Absolute Contraindications:</div>
            {Object.entries(ph.absolute_ci).map(([drug, reason], i) => (
              <div key={i} className="mb-2 p-2 rounded" style={{ background: '#fce4ec' }}>
                <span className="fw-bold small me-2" style={{ color: COLOR3 }}>{drug}:</span>
                <span className="small text-muted">{reason}</span>
              </div>
            ))}
          </div>
        )}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR}>
        <ol className="small mb-0">
          {refs.map((r, i) => <li key={i} className="mb-1">{r}</li>)}
        </ol>
      </SectionCard>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────────
export default function MTTFPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mttf/overview`).then(r => r.json()),
      fetch(`${API}/api/mttf/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mttf/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => {
      setOverview(o);
      setBreakdown(b);
      setDefinitions(d);
    }).catch(e => setError(String(e)));
  }, []);

  if (error) return <div className="alert alert-danger m-4">Error: {error}</div>;

  return (
    <div className="container-fluid py-4">
      <h2 className="fw-bold mb-1" style={{ color: COLOR }}>
        🧬 MT-TF — tRNA-Phe Dashboard
      </h2>
      <p className="text-muted mb-3 small">
        Combined CI+CIV Deficiency · CPEO / Myopathy / Exercise Intolerance ·
        m.611T&gt;C most common · FIRST tRNA in mt-genome · D-loop Adjacent ·
        H-strand rCRS 577–647 · FARS2 nuclear DDx (neonatal encephalopathy) · OMIM *590070
      </p>

      <ul className="nav nav-tabs mb-4">
        {TABS.map((t, i) => (
          <li className="nav-item" key={i}>
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <VariantsTab data={breakdown} />}
      {tab === 2 && <DDxTab bdata={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
