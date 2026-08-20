'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Phenotype', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#00695c';   // dark teal — L-protein / FAD / four-complex hub
const ACCENT2 = '#b71c1c';   // dark red — ABSOLUTE CI / VPA / neonatal
const ACCENT3 = '#e65100';   // deep orange — EXTREME HAZARD / fasting / catabolism
const ACCENT4 = '#1565c0';   // deep blue — treatments / IV glucose / LEV
const ACCENT5 = '#4a148c';   // deep purple — definitions / DLD mechanism / FAD cycle
const ACCENT6 = '#283593';   // dark indigo — four-complex block / combined disorder
const ACCENT7 = '#1b5e20';   // dark green — biomarkers / thiamine / riboflavin

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

function Section({ title, children, color = ACCENT }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function useFetch(url) {
  const [data, setData] = useState(null);
  const [err, setErr]   = useState(null);
  useEffect(() => {
    fetch(url).then(r => r.json()).then(setData).catch(setErr);
  }, [url]);
  return { data, err };
}

// ── Tab components ───────────────────────────────────────────────────────────

function OverviewTab() {
  const { data, err } = useFetch(`${API}/api/dld/overview`);
  if (err)  return <div className="alert alert-danger">Error loading overview.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const k = data.kpis || {};
  return (
    <div>
      <Alert
        text="⚠️ DLD deficiency is NOT pure NKH. CSF:plasma glycine ratio is typically <0.08 (below the ≥0.08 NKH diagnostic threshold). Look for COMBINED elevation of lactate, BCAA, and glycine — always order plasma amino acids + urine organic acids + lactate/pyruvate simultaneously."
        variant="warning"
      />
      <Alert
        text="🧬 DLD (L-protein / E3 subunit) is shared by FOUR mitochondrial complexes: GCS (glycine cleavage) · PDH (pyruvate DH) · αKGDH (alpha-ketoglutarate DH) · BCKDH (branched-chain keto acid DH). DLD LOF → simultaneous block of all four."
        variant="info"
      />
      <Alert
        text="⛔ VPA ABSOLUTE CONTRAINDICATED — triple CI: (1) mitochondrial hepatotoxicity (POLG1-equivalent risk), (2) carnitine depletion (worsens secondary deficit), (3) directly inhibits BCKDH. NEVER use valproate in DLD deficiency."
        variant="danger"
      />

      <Section title="Gene Summary" color={ACCENT6}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'38%'}}>Gene</td><td>{data.gene} ({data.protein})</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{data.locus}</td></tr>
            <tr><td className="fw-bold">Protein length</td><td>{data.aa_length} aa; FAD-dependent homodimer; mitochondrial matrix</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td>{data.cofactor}</td></tr>
            <tr><td className="fw-bold">Mechanism</td><td>{data.mechanism}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{data.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{data.omim_disease} (DLD deficiency / MSUD Type III)</td></tr>
            <tr><td className="fw-bold">Inheritance</td><td>{data.inheritance}</td></tr>
            <tr><td className="fw-bold">Prevalence</td><td>{data.prevalence}</td></tr>
            <tr><td className="fw-bold">Founder allele</td><td>{data.founder_allele}</td></tr>
            <tr><td className="fw-bold" style={{color: ACCENT2}}>DLD vs pure NKH (critical)</td><td style={{color: ACCENT2, fontWeight:'600'}}>{data.dld_vs_nkh_key_distinction}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="Four-Complex Simultaneous Block" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Complex</th><th>DLD role</th><th>LOF result</th><th>Diagnostic marker</th></tr>
            </thead>
            <tbody>
              {(data.four_complex_block || []).map((row, i) => (
                <tr key={i}>
                  <td className="fw-bold">{row.complex}</td>
                  <td>E3 / L-protein</td>
                  <td style={{color: ACCENT2}}>{row.result}</td>
                  <td>{row.diagnostic}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Cohort KPIs (n=40)" color={ACCENT}>
        <div className="row g-2">
          <KPI label="Cohort N" value={k.cohort_n} color={ACCENT} />
          <KPI label="Avg plasma lactate (mmol/L)" value={k.avg_plasma_lactate_mmol} color={ACCENT2} />
          <KPI label="Avg plasma glycine (µmol/L)" value={k.avg_plasma_glycine_umol} color={ACCENT3} />
          <KPI label="Avg plasma Leu (µmol/L)" value={k.avg_plasma_leu_umol} color={ACCENT6} />
          <KPI label="Avg CSF:plasma glycine ratio" value={k.avg_csf_plasma_glycine_ratio} color={ACCENT4} />
          <KPI label="CSF ratio <0.08 (below NKH threshold) %" value={`${k.csf_ratio_below_nkh_threshold_008_pct}%`} color={ACCENT7} />
          <KPI label="DRE %" value={`${k.dre_pct}%`} color={ACCENT2} />
          <KPI label="VPA avoided %" value={`${k.vpa_avoided_pct}%`} color={ACCENT7} />
        </div>
      </Section>

      <Section title="High-Risk Drugs" color={ACCENT2}>
        {(data.high_risk_drugs || []).map((d, i) => (
          <Alert
            key={i}
            text={`${d.risk === 'ABSOLUTE CI' ? '⛔' : d.risk === 'EXTREME HAZARD' ? '🚨' : '⚠️'} ${d.drug} — ${d.risk}: ${d.mechanism}`}
            variant={d.risk === 'ABSOLUTE CI' ? 'danger' : d.risk === 'EXTREME HAZARD' ? 'danger' : 'warning'}
          />
        ))}
      </Section>

      <Section title="Phenotype Distribution" color={ACCENT}>
        {(data.phenotype_distribution || []).map((p, i) => (
          <PctBar key={i} label={p.label} pct={p.pct} color={p.color || ACCENT} />
        ))}
      </Section>
    </div>
  );
}

function PhenotypeTab() {
  const { data, err } = useFetch(`${API}/api/dld/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Biomarkers — Combined Pattern (Diagnostic)" color={ACCENT7}>
        <Alert
          text="Simultaneous plasma amino acids + urine organic acids + plasma lactate/pyruvate MANDATORY. No single marker is sufficient. The COMBINATION of lactate↑ + BCAA↑ + glycine↑ (mild) + 2-HG↑ is diagnostic of DLD deficiency."
          variant="info"
        />
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Biomarker</th><th>Normal</th><th>DLD range</th><th>Significance</th></tr>
            </thead>
            <tbody>
              {(data.biomarkers || []).map((b, i) => (
                <tr key={i}>
                  <td className="fw-bold">{b.name}</td>
                  <td>{b.normal}</td>
                  <td style={{color: ACCENT3}}>{b.dld_range}</td>
                  <td>{b.significance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Phenotype Distribution" color={ACCENT}>
        {(data.phenotype_distribution || []).map((p, i) => (
          <PctBar key={i} label={p.label} pct={p.pct} color={p.color || ACCENT} />
        ))}
      </Section>

      <Section title="Key Variants" color={ACCENT5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Variant</th><th>Effect / Mechanism</th><th>Phenotype</th></tr>
            </thead>
            <tbody>
              {(data.key_variants || []).map((v, i) => (
                <tr key={i}>
                  <td className="fw-bold font-monospace">{v.variant}</td>
                  <td>{v.effect}</td>
                  <td><span className="badge" style={{background: ACCENT}}>{v.phenotype}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Patient Sample (first 10)" color={ACCENT}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>ID</th><th>Phenotype</th><th>Onset (mo)</th>
                <th>Lactate (mmol/L)</th><th>Leu (µmol/L)</th>
                <th>Glycine plasma</th><th>CSF:plasma ratio</th>
                <th>VPA avoided</th><th>DRE</th>
              </tr>
            </thead>
            <tbody>
              {(data.patients_sample || []).map((p, i) => (
                <tr key={i}>
                  <td className="font-monospace">{p.id}</td>
                  <td>{p.phenotype}</td>
                  <td>{p.onset_age_months}</td>
                  <td style={{color: p.plasma_lactate_mmol > 5 ? ACCENT2 : 'inherit'}}>{p.plasma_lactate_mmol}</td>
                  <td style={{color: p.plasma_leu_umol > 300 ? ACCENT3 : 'inherit'}}>{p.plasma_leu_umol}</td>
                  <td>{p.plasma_glycine_umol}</td>
                  <td style={{color: p.csf_plasma_ratio >= 0.08 ? ACCENT2 : ACCENT7}}>
                    {p.csf_plasma_ratio} {p.csf_plasma_ratio >= 0.08 ? '⚠ ≥0.08' : '✓ <0.08'}
                  </td>
                  <td>{p.vpa_avoided ? <span className="text-success fw-bold">Yes</span> : <span className="text-danger fw-bold">No</span>}</td>
                  <td>{p.dre ? <span className="text-danger">DRE</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function SeizuresTab() {
  const { data, err } = useFetch(`${API}/api/dld/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Section title="Seizure Types" color={ACCENT2}>
        {(data.seizure_types || []).map((s, i) => (
          <PctBar key={i} label={s.type} pct={s.pct} color={ACCENT2} />
        ))}
      </Section>

      <Section title="Metabolic Crisis Triggers" color={ACCENT3}>
        <Alert
          text="🚨 Metabolic crisis is the dominant acute risk in DLD deficiency. Each trigger surges ALL FOUR substrate streams simultaneously. Crisis = medical emergency requiring IV dextrose (GIR ≥6–8 mg/kg/min) + bicarb + avoid fasting."
          variant="danger"
        />
        {(data.trigger_types || []).map((t, i) => (
          <PctBar key={i} label={t.trigger} pct={t.pct} color={ACCENT3} />
        ))}
      </Section>
    </div>
  );
}

function TreatmentsTab() {
  const { data, err } = useFetch(`${API}/api/dld/breakdown`);
  if (err)  return <div className="alert alert-danger">Error loading breakdown.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  return (
    <div>
      <Alert
        text="⛔ VPA ABSOLUTE CI — triple mechanism: hepatotoxicity (mitochondrial disease risk) + carnitine depletion + BCKDH inhibition. Equivalent to POLG1 context. Never use in DLD deficiency."
        variant="danger"
      />
      <Alert
        text="🚨 Fasting / catabolism EXTREME HAZARD — IV dextrose (GIR ≥6–8 mg/kg/min) MANDATORY in any metabolic crisis. Nil-by-mouth is contraindicated."
        variant="danger"
      />
      <Alert
        text="💊 Always trial thiamine (B1, 100–300 mg/day) AND riboflavin (B2, 50–200 mg/day) — ~40–55% have partial biochemical response. Do NOT skip this trial before classifying as non-responsive."
        variant="info"
      />

      <Section title="Treatment Ladder" color={ACCENT4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Intervention</th><th>Evidence Level</th><th>Response %</th></tr>
            </thead>
            <tbody>
              {(data.treatments || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.drug}</td>
                  <td><span className="badge" style={{background: t.level === 'A' ? ACCENT7 : t.level === 'B' ? ACCENT4 : ACCENT5}}>Level {t.level}</span></td>
                  <td>
                    <div className="d-flex align-items-center gap-2">
                      <div className="progress flex-grow-1" style={{ height: 8 }}>
                        <div className="progress-bar" style={{ width: `${t.response_pct}%`, backgroundColor: t.color || ACCENT4 }} />
                      </div>
                      <span className="text-muted small">{t.response_pct}%</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function DefinitionsTab() {
  const { data, err } = useFetch(`${API}/api/dld/definitions`);
  if (err)  return <div className="alert alert-danger">Error loading definitions.</div>;
  if (!data) return <div className="text-muted">Loading...</div>;

  const gc = data.gene_card || {};
  return (
    <div>
      <Section title="Gene Card" color={ACCENT6}>
        <table className="table table-sm table-bordered small">
          <tbody>
            <tr><td className="fw-bold" style={{width:'36%'}}>Gene / Protein</td><td>{gc.gene} — {gc.full_name}</td></tr>
            <tr><td className="fw-bold">Alias</td><td>{gc.alias}</td></tr>
            <tr><td className="fw-bold">Locus</td><td>{gc.locus}</td></tr>
            <tr><td className="fw-bold">Length</td><td>{gc.aa_length} aa; {gc.structure}</td></tr>
            <tr><td className="fw-bold">Cofactor</td><td>{gc.cofactor}</td></tr>
            <tr><td className="fw-bold">Complexes shared</td><td>{(gc.complexes_shared || []).join(' · ')}</td></tr>
            <tr><td className="fw-bold">Reaction</td><td><code>{gc.reaction}</code></td></tr>
            <tr><td className="fw-bold">Inheritance</td><td>{gc.inheritance}</td></tr>
            <tr><td className="fw-bold">OMIM Gene</td><td>{gc.omim_gene}</td></tr>
            <tr><td className="fw-bold">OMIM Disease</td><td>{gc.omim_disease}</td></tr>
            <tr><td className="fw-bold">Other name</td><td>{gc.other_name}</td></tr>
          </tbody>
        </table>
      </Section>

      <Section title="Key Concepts" color={ACCENT5}>
        {(data.key_concepts || []).map((c, i) => (
          <div key={i} className="mb-2 p-2 rounded" style={{background:'#f8f9fa', border:`1px solid ${ACCENT5}22`}}>
            <div className="fw-bold small" style={{color: ACCENT5}}>{c.term}</div>
            <div className="small text-muted">{c.definition}</div>
          </div>
        ))}
      </Section>

      <Section title="Diagnostic Thresholds" color={ACCENT7}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Parameter</th><th>Normal / threshold</th><th>DLD range / target</th></tr>
            </thead>
            <tbody>
              {(data.thresholds || []).map((t, i) => (
                <tr key={i}>
                  <td className="fw-bold">{t.parameter}</td>
                  <td>{t.threshold}</td>
                  <td style={{color: ACCENT3}}>{t.dld_range}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr><th>Condition</th><th>Key distinguishing feature</th></tr>
            </thead>
            <tbody>
              {(data.differential_diagnosis || []).map((d, i) => (
                <tr key={i}>
                  <td className="fw-bold">{d.condition}</td>
                  <td>{d.distinguishing}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

export default function DLDPage() {
  const [tab, setTab] = useState(0);

  const TAB_COMPONENTS = [
    <OverviewTab key="ov" />,
    <PhenotypeTab key="ph" />,
    <SeizuresTab key="sz" />,
    <TreatmentsTab key="tx" />,
    <DefinitionsTab key="df" />,
  ];

  return (
    <div className="container-fluid py-3">
      <div className="mb-3">
        <h4 className="fw-bold mb-1" style={{ color: ACCENT }}>
          &#x1f9ec; DLD Epilepsy — Dihydrolipoamide Dehydrogenase Deficiency (L-protein / E3 subunit)
        </h4>
        <p className="text-muted small mb-1">
          DLD (7q31.1, 509 aa, FAD homodimer) · OMIM *238331/#246900 · AR biallelic LOF · ~200 cases worldwide
        </p>
        <p className="text-muted small mb-0">
          <strong>Four-complex block:</strong> GCS (glycine↑) · PDH (lactate↑↑) · αKGDH (2-HG↑) · BCKDH (BCAA↑) simultaneously impaired.
          NOT pure NKH — CSF:plasma glycine ratio typically &lt;0.08. Founder allele p.Gly194Cys (Ashkenazi Jewish).
        </p>
      </div>

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {TAB_COMPONENTS[tab]}
    </div>
  );
}
