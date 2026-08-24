'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// OAT color scheme — ornithine accumulation / gyrate atrophy / retinal degeneration
const ACCENT  = '#bf360c';   // deep burnt orange — ornithine VERY HIGH / retinotoxic accumulation
const ACCENT2 = '#1a237e';   // deep navy — gyrate atrophy / chorioretinal degeneration
const ACCENT3 = '#1b5e20';   // deep forest green — B6-responsive / partial restoration
const ACCENT4 = '#4a148c';   // deep purple — pathway position / P5C intermediate
const ACCENT5 = '#f57f17';   // amber — arginine restriction / dietary management
const ACCENT6 = '#006064';   // teal — key negatives / differentials
const ACCENT7 = '#880e4f';   // dark pink — absolute CI (high-arginine diet)
const ACCENT8 = '#37474f';   // blue-grey — secondary features

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
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between small mb-1">
        <span>{label}</span>
        <span className="fw-bold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: 10 }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function InfoBox({ title, children, color = ACCENT }) {
  return (
    <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${color}` }}>
      <div className="card-body py-2">
        <div className="fw-bold small mb-1" style={{ color }}>{title}</div>
        <div className="small text-muted">{children}</div>
      </div>
    </div>
  );
}

export default function OATPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/oat/overview`).then(r => r.json()),
      fetch(`${API}/api/oat/breakdown`).then(r => r.json()),
      fetch(`${API}/api/oat/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="container py-5 text-center"><div className="spinner-border" style={{ color: ACCENT }} /><p className="mt-3 text-muted">Loading OAT dashboard…</p></div>;
  if (err)     return <div className="container py-5"><div className="alert alert-danger">Error: {err}</div></div>;

  const kpi = ov?.kpi || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: `linear-gradient(135deg, ${ACCENT}18, ${ACCENT2}18)`, borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex align-items-center gap-2 mb-1">
          <span style={{ fontSize: 24 }}>🧬</span>
          <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>OAT Epilepsy Dashboard</h4>
          <span className="badge" style={{ background: ACCENT2, fontSize: 11 }}>Gyrate Atrophy</span>
          <span className="badge" style={{ background: ACCENT3, fontSize: 11 }}>B6-Responsive Subset</span>
        </div>
        <div className="small text-muted">{ov?.subtitle}</div>
        <div className="d-flex gap-3 mt-2 flex-wrap">
          {[
            ['Gene', ov?.gene],
            ['Chr', ov?.chromosome],
            ['Protein', ov?.protein_size?.split(';')[0]],
            ['OMIM Disease', ov?.omim_disease],
            ['Inheritance', 'Autosomal Recessive'],
            ['Cohort', `n=${ov?.cohort_n}`],
          ].map(([k, v]) => (
            <span key={k} className="badge bg-light text-dark border small">{k}: {v}</span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active fw-bold' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div className="row g-2 mb-3">
            <KPI label="Avg Plasma Ornithine (µmol/L)" value={kpi.avg_plasma_ornithine_umol_l} color={ACCENT} />
            <KPI label="Avg Proline (µmol/L)" value={kpi.avg_proline_umol_l} color={ACCENT4} />
            <KPI label="Avg PLP (nmol/L)" value={kpi.avg_plp_nmol_l} color={ACCENT6} />
            <KPI label="Seizures %" value={`${kpi.pct_seizures}%`} color={ACCENT2} />
            <KPI label="DRE %" value={`${kpi.pct_dre}%`} color={ACCENT8} />
            <KPI label="Cataracts %" value={`${kpi.pct_cataracts}%`} color={ACCENT7} />
            <KPI label="High Myopia %" value={`${kpi.pct_high_myopia}%`} color={ACCENT5} />
            <KPI label="Proximal Myopathy %" value={`${kpi.pct_proximal_myopathy}%`} color={ACCENT8} />
            <KPI label="IDD %" value={`${kpi.pct_idd}%`} color={ACCENT2} />
            <KPI label="Focal Seizures %" value={`${kpi.pct_focal_seizures}%`} color={ACCENT} />
          </div>

          {/* Phenotype distribution */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Phenotype Distribution (n={ov?.cohort_n})</div>
                <div className="card-body">
                  {Object.entries(ov?.phenotype_distribution || {}).map(([ph, v]) => (
                    <PctBar
                      key={ph} label={`${ph} (n=${v.n})`} pct={v.pct}
                      color={ph === 'Classic-Severe' ? ACCENT : ph === 'B6-Responsive' ? ACCENT3 : ACCENT8}
                    />
                  ))}
                  <div className="small text-muted mt-2">
                    Classic-Severe: biallelic null; no B6 response; ornithine &gt;800 µmol/L; retinal loss by 45–55y<br/>
                    B6-Responsive: partial PLP augmentation; ornithine falls 40–80%; slower progression<br/>
                    Mild-Attenuated: &gt;10% residual OAT; hyperornithinemia 200–400 µmol/L; late-onset
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>Key Features</div>
                <div className="card-body">
                  <PctBar label="Gyrate atrophy (pathognomonic)" pct={100} color={ACCENT} />
                  <PctBar label="Night blindness / nyctalopia" pct={95} color={ACCENT} />
                  <PctBar label="Progressive visual field loss" pct={90} color={ACCENT2} />
                  <PctBar label="High myopia (>6 D)" pct={kpi.pct_high_myopia} color={ACCENT2} />
                  <PctBar label="Posterior subcapsular cataracts" pct={kpi.pct_cataracts} color={ACCENT4} />
                  <PctBar label="Proximal myopathy" pct={kpi.pct_proximal_myopathy} color={ACCENT8} />
                  <PctBar label="Epilepsy (any)" pct={kpi.pct_seizures} color={ACCENT5} />
                  <PctBar label="Intellectual disability" pct={kpi.pct_idd} color={ACCENT6} />
                </div>
              </div>
            </div>
          </div>

          {/* Mechanism + Key positives/negatives */}
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <InfoBox title="Mechanism (OAT LOF → ornithine catabolism blocked)" color={ACCENT}>
                {ov?.mechanism}
              </InfoBox>
              <InfoBox title="KEY POSITIVE Biomarkers" color={ACCENT}>
                {ov?.key_positive_features}
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="KEY NEGATIVE Biomarkers (critical differentials)" color={ACCENT6}>
                {ov?.key_negative_features}
              </InfoBox>
              <InfoBox title="B6-Responsiveness (mandatory trial at diagnosis)" color={ACCENT3}>
                {ov?.b6_response_note}
              </InfoBox>
            </div>
          </div>

          {/* Pathway position + NBS */}
          <div className="row g-3">
            <div className="col-md-6">
              <InfoBox title="Pathway Position (Ornithine/Proline Cycle)" color={ACCENT4}>
                {ov?.pathway_position}
              </InfoBox>
            </div>
            <div className="col-md-6">
              <InfoBox title="Newborn Screening" color={ACCENT6}>
                <strong>Primary:</strong> {ov?.nbs_primary}<br/>
                <strong>Secondary:</strong> {ov?.nbs_secondary}
              </InfoBox>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && bd && (
        <div>
          {/* Biomarkers table */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT }}>Diagnostic Biomarker Profile</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Biomarker</th><th>Value / Range</th><th>Direction</th><th>Clinical Significance</th></tr>
                  </thead>
                  <tbody>
                    {bd.biomarkers?.map((b, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{b.name}</td>
                        <td className="small font-monospace">{b.value}</td>
                        <td>
                          <span className="badge" style={{
                            background:
                              b.direction === 'VERY HIGH' ? ACCENT :
                              b.direction === 'HIGH' ? '#e65100' :
                              b.direction === 'LOW-NORMAL' ? ACCENT4 :
                              b.direction === 'NORMAL-LOW' ? ACCENT8 :
                              ACCENT6,
                            fontSize: 10
                          }}>{b.direction}</span>
                        </td>
                        <td className="small text-muted">{b.significance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>OAT Variants (8 key pathogenic variants)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Variant</th><th>Domain</th><th>Freq %</th><th>Severity</th><th>Notes</th></tr>
                  </thead>
                  <tbody>
                    {bd.variants?.map((v, i) => (
                      <tr key={i}>
                        <td className="fw-bold small font-monospace">{v.variant}</td>
                        <td className="small">{v.domain}</td>
                        <td className="small text-center fw-bold" style={{ color: ACCENT }}>{v.freq_pct}%</td>
                        <td>
                          <span className="badge" style={{
                            background:
                              v.severity === 'Severe' ? ACCENT :
                              v.severity === 'Moderate-Severe' ? '#e65100' :
                              v.severity === 'Moderate' ? ACCENT5 :
                              ACCENT3,
                            fontSize: 10
                          }}>{v.severity}</span>
                        </td>
                        <td className="small text-muted">{v.note}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Patient cohort */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>
              Patient Cohort (n={bd.n}) — Deterministic Seed {187}
            </div>
            <div className="card-body p-0">
              <div className="table-responsive" style={{ maxHeight: 420 }}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>ID</th><th>Phenotype</th><th>Ornithine (µmol/L)</th><th>Proline (µmol/L)</th><th>PLP (nmol/L)</th><th>Dx Age (y)</th><th>Gyrate</th><th>Cataracts</th><th>Epilepsy</th><th>DRE</th><th>Myopathy</th><th>B6 Resp</th></tr>
                  </thead>
                  <tbody>
                    {bd.patients?.map((p, i) => (
                      <tr key={i}>
                        <td className="small font-monospace">{p.id}</td>
                        <td>
                          <span className="badge" style={{
                            background:
                              p.phenotype === 'Classic-Severe' ? ACCENT :
                              p.phenotype === 'B6-Responsive' ? ACCENT3 : ACCENT8,
                            fontSize: 9
                          }}>{p.phenotype}</span>
                        </td>
                        <td className="small fw-bold" style={{ color: ACCENT }}>{p.plasma_ornithine_umol_l}</td>
                        <td className="small" style={{ color: ACCENT4 }}>{p.proline_umol_l}</td>
                        <td className="small">{p.plp_nmol_l}</td>
                        <td className="small">{p.age_diagnosis_years}</td>
                        <td className="small text-center">✅</td>
                        <td className="small text-center">{p.cataracts ? '✅' : '—'}</td>
                        <td className="small text-center">{p.epilepsy ? '⚡' : '—'}</td>
                        <td className="small text-center">{p.dre ? '🔴' : '—'}</td>
                        <td className="small text-center">{p.myopathy ? '💪' : '—'}</td>
                        <td className="small text-center">{p.b6_response ? '✅' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ── */}
      {tab === 2 && bd && (
        <div>
          <div className="row g-3 mb-3">
            {/* Seizure types */}
            <div className="col-md-5">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>Seizure Types</div>
                <div className="card-body">
                  {bd.seizure_types?.map((s, i) => (
                    <div key={i} className="mb-3">
                      <PctBar label={s.type} pct={s.pct} color={i === 0 ? ACCENT : i < 3 ? ACCENT2 : ACCENT8} />
                      <div className="small text-muted">{s.note}</div>
                    </div>
                  ))}
                  <InfoBox title="Seizure rate in OAT (~30–40%)" color={ACCENT5}>
                    Lower than most metabolic epilepsies. Focal and absence are modal types. Ornithine has
                    NMDA agonist-like effects at very high concentrations → cortical hyperexcitability.
                    DRE rate (~15–20%) is lower than most metabolic epilepsies; seizures may improve with
                    ornithine-lowering treatment.
                  </InfoBox>
                </div>
              </div>
            </div>

            {/* Clinical features */}
            <div className="col-md-7">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Clinical Features</div>
                <div className="card-body">
                  {bd.clinical_features?.map((f, i) => (
                    <div key={i} className="mb-2">
                      <PctBar label={f.feature} pct={f.pct}
                        color={f.pct === 100 ? ACCENT : f.pct > 80 ? ACCENT2 : f.pct > 50 ? ACCENT5 : ACCENT8} />
                      <div className="small text-muted">{f.note}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Treatments */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT3 }}>Treatments (Evidence Levels)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Treatment</th><th>Level</th><th>Mechanism</th></tr>
                  </thead>
                  <tbody>
                    {bd.treatments?.map((t, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{t.tx}</td>
                        <td>
                          <span className="badge" style={{
                            background:
                              t.level === 'A' ? '#388e3c' :
                              t.level === 'A (when indicated)' ? '#388e3c' :
                              t.level === 'MANDATORY' ? ACCENT :
                              t.level === 'B' ? ACCENT5 :
                              t.level === 'B / MODERATE RISK' ? '#e65100' :
                              ACCENT8,
                            fontSize: 10
                          }}>{t.level}</span>
                        </td>
                        <td className="small text-muted">{t.mechanism}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Drug risks */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT7 }}>Drug Risks & Contraindications</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Drug/Substance</th><th>Risk Level</th><th>Reason</th></tr>
                  </thead>
                  <tbody>
                    {bd.drug_risks?.map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{d.drug}</td>
                        <td>
                          <span className="badge" style={{
                            background:
                              d.risk === 'ABSOLUTE CI' ? '#c62828' :
                              d.risk === 'ABSOLUTE CI (B6-responsive) / HIGH RISK (others)' ? '#c62828' :
                              d.risk === 'HIGH RISK' ? '#e65100' :
                              '#616161',
                            fontSize: 10
                          }}>{d.risk}</span>
                        </td>
                        <td className="small text-muted">{d.reason}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Differentials */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>Differential Diagnoses</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th>Disease</th><th>Shared Features</th><th>Distinguishing Feature</th></tr>
                  </thead>
                  <tbody>
                    {bd.differentials?.map((d, i) => (
                      <tr key={i}>
                        <td className="fw-bold small">{d.disease}</td>
                        <td className="small text-muted">{d.shared}</td>
                        <td className="small" style={{ color: ACCENT6 }}>{d.distinguishing}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && def && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT }}>Gene & Disease</div>
                <div className="card-body small">
                  {[
                    ['Gene', def.gene],
                    ['Full name', def.full_name],
                    ['Disease', def.disease_name],
                    ['OMIM', `Gene ${def.omim_gene} · Disease ${def.omim_disease}`],
                    ['Chromosome', def.chromosome],
                    ['Inheritance', def.inheritance],
                    ['Protein', def.protein],
                  ].map(([k, v]) => (
                    <div key={k} className="mb-1"><span className="fw-bold" style={{ color: ACCENT }}>{k}:</span> {v}</div>
                  ))}
                  <div className="mt-2 text-muted">{def.enzyme_function}</div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-bold small" style={{ color: ACCENT2 }}>Pathway</div>
                <div className="card-body small text-muted">
                  <div className="fw-bold text-dark mb-2">{def.pathway}</div>
                  {def.pathway_summary}
                </div>
              </div>
            </div>
          </div>

          {/* Key pathway comparisons */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold small" style={{ color: ACCENT4 }}>Key Pathway Comparisons (Proline/Ornithine Group)</div>
            <div className="card-body">
              {def.key_pathway_comparisons?.map((c, i) => (
                <InfoBox key={i} title={c.pair} color={i % 2 === 0 ? ACCENT : ACCENT2}>
                  {c.description}
                </InfoBox>
              ))}
            </div>
          </div>

          {/* Glossary */}
          <div className="card shadow-sm">
            <div className="card-header fw-bold small" style={{ color: ACCENT6 }}>Glossary ({def.key_terms?.length} terms)</div>
            <div className="card-body p-0">
              <div className="table-responsive">
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-light">
                    <tr><th style={{ width: '25%' }}>Term</th><th>Definition</th></tr>
                  </thead>
                  <tbody>
                    {def.key_terms?.map((t, i) => (
                      <tr key={i}>
                        <td className="fw-bold small" style={{ color: ACCENT4 }}>{t.term}</td>
                        <td className="small text-muted">{t.definition}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-4 pt-3 border-top text-muted small d-flex justify-content-between">
        <span>🧬 OAT Deficiency (Gyrate Atrophy of Choroid and Retina) · OMIM #258870 · 10q26.13 · AR · ~300–500 cases worldwide 2026</span>
        <Link href="/" className="text-decoration-none" style={{ color: ACCENT }}>← Back to Portal</Link>
      </div>
    </div>
  );
}
