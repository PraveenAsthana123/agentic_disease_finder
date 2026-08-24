'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Triggers', 'Treatments', 'Definitions'];

const ACCENT  = '#1a237e';   // deep indigo — MAT1A / SAM deficiency
const ACCENT2 = '#b71c1c';   // deep red — extreme methionine / severe
const ACCENT3 = '#4a148c';   // deep purple — SAM very low / SAM crisis
const ACCENT4 = '#1565c0';   // blue — Level A treatment
const ACCENT5 = '#37474f';   // slate — KEY NEGATIVES / normal values
const ACCENT6 = '#1b5e20';   // dark green — benign / normal biomarkers
const ACCENT7 = '#e65100';   // deep orange — liver disease / white matter
const ACCENT8 = '#006064';   // teal — methionine restriction / SAMe

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

export default function MAT1APage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [br, setBr]       = useState(null);
  const [df, setDf]       = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mat1a/overview`).then(r => r.json()),
      fetch(`${API}/api/mat1a/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mat1a/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBr(b); setDf(d); })
      .catch(e => setError(e.message));
  }, []);

  if (error) return <div className="alert alert-danger m-4">{error}</div>;
  if (!ov)   return <div className="text-center mt-5"><div className="spinner-border" /></div>;

  const kpi = ov.kpis || {};
  const kpiPcts = br?.kpi_pcts || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${ACCENT} 0%, ${ACCENT3} 100%)` }}>
        <div className="d-flex justify-content-between align-items-start flex-wrap gap-2">
          <div>
            <h4 className="mb-1 fw-bold">🧬 MAT1A Epilepsy Dashboard</h4>
            <div style={{ fontSize: 13, opacity: 0.9 }}>
              Methionine Adenosyltransferase I/III Deficiency — Hypermethioninemia with SAM Deficiency
            </div>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              MAT1A-395aa · Mg²⁺/K⁺-Hepatic · 10q22.3 · AR (severe) / AD (benign p.Arg264His) ·
              Methionine+ATP → SAM (BLOCKED) · OMIM *610550 / #250850
            </div>
          </div>
          <div className="text-end">
            <span className="badge bg-light text-dark me-1">n={ov.cohort_n}</span>
            <span className="badge bg-danger me-1">Met {kpi.avg_methionine_umol_l} µmol/L avg</span>
            <span className="badge bg-warning text-dark me-1">SAM↓↓ {kpi.avg_sam_umol_l} µmol/L avg</span>
          </div>
        </div>
      </div>

      {/* Critical alerts */}
      <Alert variant="danger" text="🚨 BETAINE (TMG) ABSOLUTELY CONTRAINDICATED — BHMT: Hcy→methionine. Already extreme hypermethioninemia will WORSEN catastrophically. Never prescribe betaine for 'elevated Hcy' if methionine is very high + tHcy normal." />
      <Alert variant="success" text="✅ SAM (SAMe) SUPPLEMENTS ARE THE TREATMENT — SAMe corrects the product deficiency in MAT1A. OPPOSITE of AHCY where SAM is contraindicated. Confirm SAM/SAH levels before prescribing." />
      <Alert variant="warning" text="⚠️ BENIGN FORM COMMON — AD p.Arg264His heterozygotes (~50% of MAT1A NBS positives) are asymptomatic. Do NOT restrict diet without genetic confirmation — unnecessary restriction harms growth." />
      <Alert variant="info"    text="ℹ️ BREATH ODOR PATHOGNOMONIC — Dimethylsulfide / garlic-cabbage smell from methionine catabolism. Bedside clue unique to MAT1A deficiency among HHcy disorders." />

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row 1 — biomarkers */}
          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Cohort Biomarker KPIs (n={ov.cohort_n})</h6>
          <div className="row g-2 mb-3">
            <KPI label="Avg Methionine (µmol/L)" value={kpi.avg_methionine_umol_l} color={ACCENT2} />
            <KPI label="Avg SAM (µmol/L)" value={kpi.avg_sam_umol_l} color={ACCENT3} />
            <KPI label="Avg SAH (µmol/L)" value={kpi.avg_sah_umol_l} color={ACCENT6} />
            <KPI label="Avg tHcy (µmol/L)" value={kpi.avg_homocysteine_umol_l} color={ACCENT6} />
            <KPI label="Avg AST (U/L)" value={kpi.avg_ast_u_l} color={ACCENT7} />
            <KPI label="NBS Detected" value={`${kpi.pct_nbs_detected}%`} color={ACCENT4} />
          </div>
          {/* KPI row 2 — clinical */}
          <div className="row g-2 mb-4">
            <KPI label="White Matter Disease" value={`${kpi.pct_white_matter}%`} color={ACCENT7} />
            <KPI label="Liver Disease" value={`${kpi.pct_liver_disease}%`} color={ACCENT7} />
            <KPI label="IDD" value={`${kpi.pct_idd}%`} color={ACCENT3} />
            <KPI label="Seizures" value={`${kpi.pct_seizures}%`} color={ACCENT} />
            <KPI label="Breath Odor" value={`${kpi.pct_breath_odor}%`} color={ACCENT2} />
            <KPI label="Myopathy" value="0% (ABSENT)" color={ACCENT6} />
          </div>

          {/* Phenotype distribution */}
          <div className="row mb-4">
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT }}>Phenotypic Class Distribution</h6>
              {Object.entries(ov.phenotype_distribution || {}).map(([ph, cnt]) => (
                <PctBar
                  key={ph}
                  label={ph}
                  pct={Math.round(cnt / ov.cohort_n * 100)}
                  color={ph.includes('Severe') ? ACCENT2 : ph.includes('Classic') ? ACCENT : ACCENT6}
                />
              ))}
              <div className="alert alert-success mt-2" style={{ fontSize: 12 }}>
                <strong>~50% of MAT1A NBS positives are BENIGN</strong> (AD p.Arg264His heterozygotes).
                No treatment needed. Unnecessary methionine restriction causes growth failure.
              </div>
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>KEY METABOLIC FINGERPRINT — MAT1A vs AHCY vs CBS</h6>
              <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
                <thead className="table-dark">
                  <tr><th>Biomarker</th><th>MAT1A</th><th>AHCY</th><th>CBS</th></tr>
                </thead>
                <tbody>
                  <tr><td>Methionine</td><td className="text-danger fw-bold">↑↑↑ 200–2000</td><td className="text-danger">↑↑ 200–600</td><td className="text-warning">↑ 60–500</td></tr>
                  <tr><td>SAM</td><td className="text-danger fw-bold">↓↓ VERY LOW</td><td className="text-danger fw-bold">↑↑ ELEVATED</td><td>Normal</td></tr>
                  <tr><td>SAH</td><td className="text-success fw-bold">NORMAL ✓</td><td className="text-danger fw-bold">↑↑↑ PATHOGN.</td><td className="text-success">NORMAL ✓</td></tr>
                  <tr><td>tHcy</td><td className="text-success fw-bold">NORMAL ✓</td><td className="text-warning">40–150 (mod)</td><td className="text-danger fw-bold">100–500 HIGHEST</td></tr>
                  <tr><td>MMA</td><td className="text-success fw-bold">NORMAL ✓</td><td className="text-success fw-bold">NORMAL ✓</td><td className="text-success fw-bold">NORMAL ✓</td></tr>
                  <tr><td>Myopathy</td><td className="text-success fw-bold">ABSENT ✓</td><td className="text-danger fw-bold">85-90% HALLMARK</td><td className="text-success">ABSENT ✓</td></tr>
                  <tr><td>Ectopia lentis</td><td className="text-success fw-bold">ABSENT ✓</td><td className="text-success fw-bold">ABSENT ✓</td><td className="text-danger fw-bold">90% PATHOGN.</td></tr>
                  <tr><td>Breath odor</td><td className="text-danger fw-bold">PRESENT (DMS)</td><td className="text-success">ABSENT</td><td className="text-success">ABSENT</td></tr>
                  <tr><td>White matter</td><td className="text-warning">40-50%</td><td className="text-warning">Present</td><td className="text-success">10-15%</td></tr>
                  <tr><td>SAM treatment</td><td className="text-success fw-bold">Level A ✅</td><td className="text-danger fw-bold">ABSOLUTE CI 🚫</td><td>Level B</td></tr>
                  <tr><td>Betaine tx</td><td className="text-danger fw-bold">ABSOLUTE CI 🚫</td><td className="text-warning">CAUTION</td><td className="text-success fw-bold">Level A ✅</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Pathway summary */}
          <div className="card mb-3">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: 'white' }}>
              Methionine Cycle — MAT1A Is the SAM Synthesis Step (UPSTREAM of all methyltransferases)
            </div>
            <div className="card-body">
              <pre style={{ fontSize: 12, background: '#f8f9fa', padding: 12, borderRadius: 6, whiteSpace: 'pre-wrap' }}>
{`  L-Methionine + ATP
        │
   MAT1A (blocked in MAT1A deficiency)
        ↓
   SAM (S-Adenosylmethionine) ← CANNOT BE MADE
   ↑ Methionine accumulates (200–2000+ µmol/L)
   ↓ SAM is VERY LOW
        │
   [If SAM were made:]
   Methyltransferases (DNMT, COMT, HNMT, GAMT, PEMT, MBP…)
   SAM donates CH₃ → biological methylation reactions
        │
   SAH (S-Adenosylhomocysteine) ← LESS PRODUCED (less SAM → less methylation)
        │
   AHCY → Adenosine + Homocysteine
   tHcy LOW/NORMAL (less SAH hydrolysis → less Hcy produced)

MAT ISOFORMS:
  MAT I (homotetramer, alpha1×4) — liver, Km ~1 mM  ← LOST in AR MAT1A
  MAT III (homodimer, alpha1×2) — liver, Km ~8 mM   ← LOST in AR MAT1A
  MAT II (heterotetramer, MAT2A/MAT2B) — brain/ubiquitous ← INTACT (brain protected)`}
              </pre>
            </div>
          </div>

          {/* Gene / disease summary */}
          <div className="row g-3">
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT3, color: 'white' }}>Disease Summary</div>
                <div className="card-body small">
                  <p><strong>Gene:</strong> {ov.gene} (Methionine Adenosyltransferase 1A)</p>
                  <p><strong>Disease:</strong> {ov.disease_name}</p>
                  <p><strong>Protein:</strong> {ov.protein_size}</p>
                  <p><strong>Chromosome:</strong> {ov.chromosome}</p>
                  <p><strong>Inheritance:</strong> {ov.inheritance}</p>
                  <p><strong>OMIM Gene:</strong> {ov.omim_gene} · <strong>Disease:</strong> {ov.omim_disease}</p>
                  <p><strong>Prevalence:</strong> {ov.prevalence}</p>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card h-100">
                <div className="card-header fw-bold" style={{ background: ACCENT4, color: 'white' }}>NBS Detection</div>
                <div className="card-body small">
                  <p><strong>Primary:</strong> {ov.nbs_primary}</p>
                  <p><strong>Confirmatory:</strong> {ov.nbs_secondary}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 1: Patients & Biomarkers ── */}
      {tab === 1 && br && (
        <div>
          <div className="row mb-4">
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT }}>Clinical Feature Prevalence</h6>
              <PctBar label="IDD (intellectual disability)" pct={kpiPcts.idd} color={ACCENT3} />
              <PctBar label="Liver disease (hepatomegaly/transaminasemia)" pct={kpiPcts.liver_disease} color={ACCENT7} />
              <PctBar label="White matter disease (demyelination)" pct={kpiPcts.white_matter_disease} color={ACCENT7} />
              <PctBar label="Breath odor (dimethylsulfide)" pct={kpiPcts.breath_odor} color={ACCENT2} />
              <PctBar label="NBS detected (methionine ↑↑)" pct={kpiPcts.nbs_detected} color={ACCENT4} />
              <PctBar label="Seizures" pct={kpiPcts.seizures} color={ACCENT} />
              <PctBar label="Myopathy (ABSENT — not a feature)" pct={0} color={ACCENT6} />
              <div className="alert alert-success mt-2" style={{ fontSize: 12 }}>
                <strong>Myopathy = 0%</strong> — distinguishes MAT1A from AHCY (85-90% myopathy).
                CK is NORMAL. No cardiomyopathy.
              </div>
            </div>
            <div className="col-md-6">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>Biomarker Reference Ranges</h6>
              {br.biomarker_ranges && Object.entries(br.biomarker_ranges).map(([key, val]) => (
                <div key={key} className="mb-2 p-2 rounded" style={{ background: '#f8f9fa', fontSize: 12 }}>
                  <span className="fw-bold text-capitalize">{key.replace(/_/g, ' ')}: </span>
                  {typeof val === 'string' ? val :
                    Object.entries(val).map(([k, v]) => `${k}: ${v}`).join(' | ')}
                </div>
              ))}
            </div>
          </div>

          <h6 className="fw-bold mb-2" style={{ color: ACCENT }}>Patient Sample (first 12)</h6>
          <div className="table-responsive">
            <table className="table table-sm table-striped" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr>
                  <th>ID</th><th>Phenotype</th><th>Variant</th><th>Onset (mo)</th>
                  <th>Met µmol/L</th><th>SAM µmol/L</th><th>SAH µmol/L</th><th>tHcy</th>
                  <th>AST</th><th>WM</th><th>Liver</th><th>Sz</th><th>Odor</th><th>NBS</th>
                </tr>
              </thead>
              <tbody>
                {(br.patient_sample || []).map(p => (
                  <tr key={p.id}>
                    <td>{p.id}</td>
                    <td style={{ maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                        title={p.phenotype}>{p.phenotype.split(' (')[0]}</td>
                    <td><code>{p.variant}</code></td>
                    <td>{p.age_onset_months}</td>
                    <td><span className={p.methionine_umol_l > 500 ? 'text-danger fw-bold' : 'text-warning'}>{p.methionine_umol_l}</span></td>
                    <td><span className={p.sam_umol_l < 40 ? 'text-danger fw-bold' : 'text-warning'}>{p.sam_umol_l}</span></td>
                    <td><span className="text-success">{p.sah_umol_l}</span></td>
                    <td><span className="text-success">{p.homocysteine_umol_l}</span></td>
                    <td><span className={p.ast_u_l > 100 ? 'text-warning' : ''}>{p.ast_u_l}</span></td>
                    <td>{p.white_matter_disease ? '✓' : '—'}</td>
                    <td>{p.liver_disease ? '✓' : '—'}</td>
                    <td>{p.seizures ? '✓' : '—'}</td>
                    <td>{p.breath_odor ? '🌿' : '—'}</td>
                    <td>{p.nbs_detected ? '✓' : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h6 className="fw-bold mt-3 mb-2" style={{ color: ACCENT3 }}>Genetic Variants</h6>
          <div className="table-responsive">
            <table className="table table-sm table-bordered" style={{ fontSize: 12 }}>
              <thead className="table-dark">
                <tr><th>Variant</th><th>Domain</th><th>Prevalence</th><th>Severity</th></tr>
              </thead>
              <tbody>
                {(br.variant_breakdown || []).map((v, i) => (
                  <tr key={i} style={{ background: v.variant === 'p.Arg264His' ? '#e8f5e9' : undefined }}>
                    <td><code>{v.variant}</code>{v.variant === 'p.Arg264His' && <span className="badge bg-success ms-1">AD Benign</span>}</td>
                    <td>{v.domain}</td>
                    <td>{v.prevalence}</td>
                    <td>{v.severity}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Tab 2: Seizures & Triggers ── */}
      {tab === 2 && br && (
        <div>
          <div className="row mb-4">
            <div className="col-md-5">
              <h6 className="fw-bold" style={{ color: ACCENT }}>Seizure Type Distribution</h6>
              {(br.seizure_types || []).map((s, i) => (
                <PctBar key={i} label={s.type} pct={s.pct}
                  color={i === 0 ? ACCENT2 : i === 1 ? ACCENT3 : i === 2 ? ACCENT7 : ACCENT} />
              ))}
              <div className="alert alert-secondary mt-2" style={{ fontSize: 12 }}>
                Seizures in {kpiPcts.seizures ?? kpi.pct_seizures}% of cohort — LESS PROMINENT than white matter disease / liver.
                Mechanism: cerebral demyelination (SAM-deficient myelin) + hypermethioninemia neurotoxicity.
                Brain MAT II (MAT2A) is intact → partial neurological protection.
              </div>
            </div>
            <div className="col-md-7">
              <h6 className="fw-bold" style={{ color: ACCENT2 }}>Metabolic Triggers / Risk Factors</h6>
              {(br.metabolic_triggers || []).map((t, i) => (
                <div key={i} className="mb-2 p-2 rounded border" style={{ fontSize: 12 }}>
                  <div className="d-flex justify-content-between mb-1">
                    <span className="fw-bold">{t.trigger}</span>
                    <span className="badge" style={{ backgroundColor: t.trigger.includes('CONTRAINDICATED') ? '#b71c1c' : ACCENT2 }}>{t.pct}%</span>
                  </div>
                  <div className="text-muted">{t.mechanism}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Tab 3: Treatments ── */}
      {tab === 3 && br && (
        <div>
          <Alert variant="danger" text="🚨 BETAINE (TMG) ABSOLUTELY CONTRAINDICATED — BHMT raises methionine further in already extreme hypermethioninemia." />
          <Alert variant="success" text="✅ SAM (SAMe) IS THE TREATMENT — corrects product deficiency. Opposite of AHCY where SAM is the toxin." />
          <Alert variant="info" text="ℹ️ Benign AD p.Arg264His heterozygotes need NO treatment — confirm genetics before starting any intervention." />
          <Alert variant="success" text="✅ First-line AED: Levetiracetam — safe with liver disease; no impact on methionine pathway." />

          <h6 className="fw-bold mb-3" style={{ color: ACCENT4 }}>Treatment Protocols</h6>
          {(br.treatments || []).map((t, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header d-flex justify-content-between align-items-center py-2">
                <span className="fw-bold" style={{ fontSize: 14 }}>{t.treatment}</span>
                <span className="badge" style={{
                  backgroundColor: t.level.includes('Level A') ? '#1b5e20' : t.level.includes('Level B') ? '#1565c0' : '#37474f'
                }}>{t.level}</span>
              </div>
              <div className="card-body py-2 small text-muted">{t.mechanism}</div>
            </div>
          ))}

          <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT2 }}>Drug Risks & Contraindications</h6>
          {(br.drug_risks || []).map((d, i) => (
            <div key={i} className="card mb-3 border-danger">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{ background: d.risk.includes('ABSOLUTE') ? '#ffebee' : '#fff8e1' }}>
                <span className="fw-bold" style={{ fontSize: 14 }}>{d.agent}</span>
                <span className="badge" style={{
                  backgroundColor: d.risk.includes('ABSOLUTE') ? '#b71c1c' : d.risk.includes('HIGH') ? '#e65100' : d.risk.includes('MODERATE') ? '#f57f17' : '#78909c'
                }}>{d.risk.split(' —')[0]}</span>
              </div>
              <div className="card-body py-2 small text-muted">{d.mechanism}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Tab 4: Definitions ── */}
      {tab === 4 && df && (
        <div>
          {/* Gene card */}
          {df.gene_card && (
            <div className="card mb-4">
              <div className="card-header fw-bold" style={{ background: ACCENT, color: 'white' }}>Gene Card — MAT1A</div>
              <div className="card-body">
                <div className="row g-2">
                  {Object.entries(df.gene_card).map(([k, v]) => (
                    <div key={k} className="col-12 col-md-6">
                      <div className="p-2 rounded" style={{ background: '#f8f9fa', fontSize: 12 }}>
                        <span className="fw-bold text-capitalize">{k.replace(/_/g, ' ')}: </span>
                        <span className="text-muted">{v}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Key concepts */}
          <h6 className="fw-bold mb-3" style={{ color: ACCENT3 }}>Key Concepts</h6>
          {(df.key_concepts || []).map((c, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header fw-bold py-2" style={{ background: '#e8eaf6', fontSize: 13 }}>
                {c.concept}
              </div>
              <div className="card-body py-2 small text-muted">{c.explanation}</div>
            </div>
          ))}

          {/* Differential */}
          <h6 className="fw-bold mb-3 mt-4" style={{ color: ACCENT2 }}>Differential Diagnosis</h6>
          {(df.differential_diagnosis || []).map((d, i) => (
            <div key={i} className="card mb-3">
              <div className="card-header fw-bold py-2" style={{ background: '#fce4ec', fontSize: 13 }}>
                {d.disease}
              </div>
              <div className="card-body py-2 small text-muted">{d.distinguishing}</div>
            </div>
          ))}
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top text-center small text-muted">
        <Link href="/ahcy" className="me-3">← AHCY (Adenosylhomocysteinase Deficiency)</Link>
        <Link href="/" className="me-3">Home</Link>
        <span>MAT1A · OMIM *610550 / #250850 · 10q22.3 · AR/AD · Mg²⁺/K⁺-dependent</span>
      </div>
    </div>
  );
}
