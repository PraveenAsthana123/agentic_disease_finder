'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const TABS = ['Overview', 'Patients & Biomarkers', 'Seizures & Treatments', 'Definitions'];

// LCHAD colour scheme — teal-indigo (retinal / neural / maternal)
const ACCENT  = '#004d40';   // deep teal — LCHAD / MTP primary
const ACCENT2 = '#1a237e';   // deep indigo — C16-OH primary NBS marker
const ACCENT3 = '#b71c1c';   // deep red — ABSOLUTE CI / severe
const ACCENT4 = '#1b5e20';   // deep green — KEY NEGATIVES (C8/C14:1 normal)
const ACCENT5 = '#6a1b9a';   // deep purple — retinopathy / neuropathy (UNIQUE)
const ACCENT6 = '#0d47a1';   // deep blue — MCT therapeutic
const ACCENT7 = '#37474f';   // dark slate — epidemiology
const ACCENT8 = '#4e342e';   // dark brown — maternal AFLP (UNIQUE)

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

export default function LCHADPage() {
  const [tab, setTab]     = useState(0);
  const [ov, setOv]       = useState(null);
  const [bd, setBd]       = useState(null);
  const [def, setDef]     = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]     = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/lchad/overview`).then(r => r.json()),
      fetch(`${API}/api/lchad/breakdown`).then(r => r.json()),
      fetch(`${API}/api/lchad/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDef(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading LCHAD / MTP Dashboard…</div>;
  if (err)     return <div className="p-4 text-center text-danger">Error: {err}</div>;

  const phDist     = ov?.phenotype_distribution || {};
  const clinSum    = ov?.clinical_summary || {};
  const varDist    = ov?.variant_distribution || {};

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3">
        <h4 className="fw-bold mb-0" style={{ color: ACCENT }}>
          &#x1f9ec; LCHAD / MTP Deficiency Dashboard
        </h4>
        <div className="text-muted small">
          Long-Chain 3-Hydroxyacyl-CoA Dehydrogenase / Mitochondrial Trifunctional Protein Deficiency &mdash; HADHA / HADHB &middot; 2p23.3 &middot; AR &middot; OMIM #609016 / #609015
        </div>
        <div className="text-muted small">
          C16-OH PRIMARY NBS MARKER &middot; Retinopathy + Neuropathy UNIQUE &middot; Maternal AFLP/HELLP UNIQUE &middot; MCT oil therapeutic &middot; KD absolute CI
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button
              className={`nav-link${tab === i ? ' active fw-bold' : ''}`}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── TAB 0: OVERVIEW ── */}
      {tab === 0 && (
        <div>
          <Alert
            text="⚠ FASTING: ABSOLUTE CONTRAINDICATION — triggers long-chain FAO demand on blocked LCHAD/MTP → hypoketotic hypoglycaemia + lactic acidosis + rhabdomyolysis. IV Glucose 10% immediately during any crisis."
            variant="danger"
          />
          <Alert
            text="⚠ KETOGENIC DIET: ABSOLUTE CI — long-chain fat floods the blocked MTP complex → massive 3-OH-acylcarnitine accumulation → retinal toxicity + rhabdomyolysis + cardiomyopathy crisis. VPA: HIGH RISK."
            variant="warning"
          />
          <Alert
            text="🧠 UNIQUE FEATURES: Retinal pigmentary degeneration + peripheral neuropathy — the ONLY FAO disorder with these features. DHA supplementation (Level B) for retinal/neural protection. Annual ERG + EMG/NCS monitoring."
            variant="info"
          />
          <Alert
            text="🤰 MATERNAL ALERT: Unexplained AFLP (Acute Fatty Liver of Pregnancy) or HELLP in 3rd trimester → investigate newborn for LCHAD deficiency. UNIQUE maternal-fetal metabolite interaction."
            variant="warning"
          />

          {/* KPI strip */}
          <div className="row mb-3">
            <KPI label="Total Patients"     value={ov?.n_patients}                  color={ACCENT}  />
            <KPI label="Retinopathy"        value={clinSum.retinopathy_n}            color={ACCENT5} />
            <KPI label="Neuropathy"         value={clinSum.neuropathy_n}             color={ACCENT5} />
            <KPI label="Maternal AFLP"      value={clinSum.maternal_aflp_n}          color={ACCENT8} />
            <KPI label="Cardiomyopathy"     value={clinSum.cardiomyopathy_n}         color={ACCENT3} />
            <KPI label="Rhabdomyolysis"     value={clinSum.rhabdo_n}                 color={ACCENT3} />
          </div>
          <div className="row mb-3">
            <KPI label="Seizures"           value={clinSum.seizures_n}              color={ACCENT}  />
            <KPI label="Avg C16-OH (µmol/L)" value={ov?.mean_c16_oh}               color={ACCENT2} />
            <KPI label="Avg CK (U/L)"      value={ov?.mean_ck}                     color={ACCENT3} />
            <KPI label="Avg Glucose (mmol)" value={ov?.mean_glucose}               color={ACCENT7} />
            <KPI label="Avg Lactate (mmol)" value={ov?.mean_lactate}               color={ACCENT7} />
          </div>

          <div className="row">
            {/* Phenotype distribution */}
            <div className="col-md-5 mb-3">
              <div className="card shadow-sm h-100">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Phenotype Distribution</div>
                  {Object.entries(phDist).map(([ph, n]) => (
                    <PctBar
                      key={ph}
                      label={ph}
                      pct={Math.round(n / ov.n_patients * 100)}
                      color={ph.includes('Severe') ? ACCENT3 : ph.includes('Adult') ? ACCENT6 : ACCENT}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Unique features */}
            <div className="col-md-7 mb-3">
              <div className="row h-100">
                <div className="col-12 mb-3">
                  <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                    <div className="card-body py-2">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT5 }}>
                        &#x1f441;&#xfe0f; Retinopathy ({clinSum.retinopathy_pct}%) + &#x1f9e0; Neuropathy ({clinSum.neuropathy_pct}%)
                      </div>
                      <div className="small text-muted">
                        <strong>UNIQUE among ALL FAO disorders.</strong> Retinal pigmentary degeneration: 3-OH-acylcarnitines toxic to RPE + DHA synthesis impaired.
                        Annual ERG + fundoscopy. Peripheral neuropathy: axonal, length-dependent, progressive.
                        DHA supplementation (100–200 mg/day) slows progression. No other FAO disorder causes retinopathy or neuropathy.
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-12 mb-3">
                  <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT8}` }}>
                    <div className="card-body py-2">
                      <div className="fw-bold small mb-1" style={{ color: ACCENT8 }}>
                        &#x1f930; Maternal AFLP/HELLP ({clinSum.maternal_aflp_pct}% maternal history)
                      </div>
                      <div className="small text-muted">
                        <strong>UNIQUE LCHAD maternal-fetal interaction.</strong> LCHAD-deficient fetus cannot oxidize 3-OH-fatty acids →
                        cross placenta → accumulate in maternal liver → AFLP in 3rd trimester. Also HELLP (hemolysis + elevated LFTs + low platelets).
                        Mother presenting with unexplained AFLP → test newborn for LCHAD. Exclusive to LCHAD/TFP.
                      </div>
                    </div>
                  </div>
                </div>
                <div className="col-12">
                  <InfoBox title="NBS Marker: C16-OH (3-Hydroxypalmitoylcarnitine)" color={ACCENT2}>
                    Primary NBS marker: C16-OH &gt;0.08 µmol/L on tandem MS/MS.
                    C16-OH/C16 ratio &gt;0.07 highly specific. Also C14-OH + C18:1-OH elevated.
                    KEY NEGATIVES: C8 NORMAL (vs MCAD) · C14:1 NORMAL (vs VLCAD) · No HG/SG/PPG (vs MCAD).
                    Founder: p.Glu474Gln (c.1528G&gt;A) HADHA — 50–90% Northern European.
                  </InfoBox>
                </div>
              </div>
            </div>
          </div>

          {/* Unique features vs other FAO */}
          <div className="row mt-2">
            {(ov?.unique_features || []).map((f, i) => (
              <div key={i} className="col-md-6 mb-2">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                  <div className="card-body py-2">
                    <div className="small" style={{ color: ACCENT5 }}>&#x2605; {f}</div>
                  </div>
                </div>
              </div>
            ))}
            {(ov?.key_negatives || []).map((f, i) => (
              <div key={i} className="col-md-4 mb-2">
                <div className="card shadow-sm" style={{ borderLeft: `4px solid ${ACCENT4}` }}>
                  <div className="card-body py-2">
                    <div className="small" style={{ color: ACCENT4 }}>&#x2714; {f}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB 1: PATIENTS & BIOMARKERS ── */}
      {tab === 1 && (
        <div>
          <Alert
            text="C16-OH primary NBS | C16-OH/C16 ratio specific | C8 NORMAL (vs MCAD) | C14:1 NORMAL (vs VLCAD) | 3-OH-acylcarnitines diagnostic"
            variant="info"
          />

          {/* Biomarker reference */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Biomarker Reference Panel</div>
              <div className="row">
                {Object.entries(bd?.biomarkers || {}).map(([key, bm]) => (
                  <div key={key} className="col-md-6 col-lg-4 mb-2">
                    <div
                      className="card shadow-sm h-100"
                      style={{ borderLeft: `4px solid ${bm.color === 'danger' ? ACCENT3 : bm.color === 'success' ? ACCENT4 : ACCENT7}` }}
                    >
                      <div className="card-body py-2 px-2">
                        <div className="fw-bold" style={{ fontSize: 12, color: bm.color === 'danger' ? ACCENT3 : bm.color === 'success' ? ACCENT4 : ACCENT }}>
                          {bm.label}
                        </div>
                        <div style={{ fontSize: 11, color: '#666' }}>Normal: {bm.normal}</div>
                        <div className="fw-bold" style={{ fontSize: 11, color: bm.color === 'danger' ? ACCENT3 : bm.color === 'success' ? ACCENT4 : ACCENT7 }}>
                          {bm.direction} {bm.status}
                        </div>
                        <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>{bm.rationale?.slice(0, 120)}…</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Patient table */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Patient Cohort (n=25 shown / 40 total · seed=273)</div>
              <div className="table-responsive">
                <table className="table table-sm table-hover" style={{ fontSize: 11 }}>
                  <thead>
                    <tr>
                      <th>ID</th><th>Phenotype</th><th>Onset(mo)</th>
                      <th>C16-OH</th><th>C16-OH/C16</th><th>C14-OH</th><th>C18:1-OH</th>
                      <th>C8</th><th>C14:1</th><th>C0</th>
                      <th>Glu</th><th>CK</th><th>Lact</th>
                      <th>Retinop</th><th>Neuropath</th><th>Cardio</th><th>m.AFLP</th><th>Sz</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd?.patients || []).map(p => (
                      <tr key={p.id}>
                        <td className="fw-bold">{p.id}</td>
                        <td>
                          <span className="badge" style={{
                            backgroundColor: p.phenotype.includes('Severe') ? ACCENT3 : p.phenotype.includes('Adult') ? ACCENT6 : ACCENT,
                            fontSize: 9
                          }}>
                            {p.phenotype.split('(')[0].trim().slice(0, 10)}
                          </span>
                        </td>
                        <td>{p.onset_mo}</td>
                        <td style={{ color: ACCENT2, fontWeight: 'bold' }}>{p.c16_oh}</td>
                        <td style={{ color: p.c16_oh_c16 > 0.07 ? ACCENT3 : ACCENT4 }}>{p.c16_oh_c16}</td>
                        <td>{p.c14_oh}</td>
                        <td>{p.c18_1_oh}</td>
                        <td style={{ color: ACCENT4 }}>{p.c8}</td>
                        <td style={{ color: ACCENT4 }}>{p.c14_1}</td>
                        <td style={{ color: p.c0 < 15 ? ACCENT3 : ACCENT7 }}>{p.c0}</td>
                        <td style={{ color: p.glucose < 3.0 ? ACCENT3 : 'inherit' }}>{p.glucose}</td>
                        <td style={{ color: p.ck > 5000 ? ACCENT3 : ACCENT7 }}>{p.ck.toLocaleString()}</td>
                        <td style={{ color: p.lactate > 4 ? ACCENT3 : ACCENT7 }}>{p.lactate}</td>
                        <td>{p.retinopathy ? <span style={{ color: ACCENT5 }}>&#x1f441;</span> : <span style={{ color: ACCENT4 }}>&#x2713;</span>}</td>
                        <td>{p.neuropathy ? <span style={{ color: ACCENT5 }}>&#x26a0;</span> : <span style={{ color: ACCENT4 }}>&#x2713;</span>}</td>
                        <td>{p.cardiomyopathy ? <span style={{ color: ACCENT3 }}>&#x2764;</span> : '—'}</td>
                        <td>{p.maternal_aflp ? <span style={{ color: ACCENT8 }}>&#x1f930;</span> : '—'}</td>
                        <td>{p.seizures ? <span style={{ color: ACCENT3 }}>&#x26a1;</span> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Phenotype profiles */}
          <div className="row">
            {Object.entries(bd?.phenotype_profiles || {}).map(([ph, prof]) => (
              <div key={ph} className="col-md-4 mb-3">
                <div className="card shadow-sm h-100" style={{
                  borderLeft: `4px solid ${ph.includes('Severe') ? ACCENT3 : ph.includes('Adult') ? ACCENT6 : ACCENT}`
                }}>
                  <div className="card-body py-2">
                    <div className="fw-bold small mb-1" style={{
                      color: ph.includes('Severe') ? ACCENT3 : ph.includes('Adult') ? ACCENT6 : ACCENT
                    }}>{ph}</div>
                    <div style={{ fontSize: 11 }}>
                      <div><strong>Prevalence:</strong> {prof.prevalence}</div>
                      <div><strong>Genotype:</strong> {prof.genotype}</div>
                      <div><strong>Onset:</strong> {prof.onset}</div>
                      <div className="text-muted">{prof.features}</div>
                      <div className="mt-1"><strong>Key:</strong> {prof.key_point}</div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB 2: SEIZURES & TREATMENTS ── */}
      {tab === 2 && (
        <div>
          <Alert
            text="FASTING = ABSOLUTE CI · KD = ABSOLUTE CI · VPA = HIGH RISK · MCT Oil = Level A (therapeutic) · DHA = Level B (retinal/neural UNIQUE) · Avoid Long-Chain Fat = Level A"
            variant="danger"
          />

          <div className="row mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT3 }}>&#x26a0; Contraindications</div>
                  <div className="small">
                    <div className="p-2 mb-2 rounded" style={{ background: '#ffebee' }}>
                      <strong style={{ color: ACCENT3 }}>FASTING — ABSOLUTE CI (Level A)</strong>
                      <div className="text-muted" style={{ fontSize: 11 }}>Forces long-chain FAO → LCHAD blocked → 3-OH-acylcarnitine surge → crisis. Never fast &gt;4h (infants &lt;6mo), &gt;6h (6-12mo), &gt;8h (children), &gt;10h (adults). IV glucose during surgery/illness.</div>
                    </div>
                    <div className="p-2 mb-2 rounded" style={{ background: '#ffebee' }}>
                      <strong style={{ color: ACCENT3 }}>KETOGENIC DIET — ABSOLUTE CI</strong>
                      <div className="text-muted" style={{ fontSize: 11 }}>Floods long-chain fat (C12-C18) on blocked MTP → retinal toxicity + rhabdomyolysis + cardiomyopathy. ABSOLUTE CI in LCHAD same as VLCAD.</div>
                    </div>
                    <div className="p-2 rounded" style={{ background: '#fff3e0' }}>
                      <strong style={{ color: '#e65100' }}>VPA — HIGH RISK (avoid)</strong>
                      <div className="text-muted" style={{ fontSize: 11 }}>Inhibits FAO globally; depletes carnitine; worsens LCHAD crisis. Use: levetiracetam, lamotrigine, oxcarbazepine.</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="col-md-6">
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT6 }}>&#x2705; Treatments</div>
                  <div className="small">
                    {Object.entries(bd?.treatments || {}).map(([key, tx]) => (
                      <div key={key} className="p-2 mb-2 rounded" style={{
                        background: key.includes('ci') ? '#ffebee' : key.includes('avoid') ? '#fff3e0' : '#e8f5e9',
                        borderLeft: `3px solid ${key.includes('ci') ? ACCENT3 : key.includes('avoid') ? '#e65100' : ACCENT6}`
                      }}>
                        <strong style={{ color: key.includes('ci') ? ACCENT3 : key.includes('avoid') ? '#e65100' : ACCENT6 }}>
                          {tx.label}
                        </strong>
                        <div className="text-muted" style={{ fontSize: 10 }}>{tx.level}</div>
                        <div className="text-muted" style={{ fontSize: 11 }}>{tx.rationale?.slice(0, 130)}…</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Seizure data */}
          <div className="card shadow-sm mb-3">
            <div className="card-body">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Seizure & Complication Summary</div>
              <div className="row">
                <div className="col-md-6">
                  <PctBar label={`Retinopathy (${clinSum.retinopathy_pct}%)`} pct={clinSum.retinopathy_pct || 0} color={ACCENT5} />
                  <PctBar label={`Peripheral Neuropathy (${clinSum.neuropathy_pct}%)`} pct={clinSum.neuropathy_pct || 0} color={ACCENT5} />
                  <PctBar label={`Cardiomyopathy (${clinSum.cardiomyopathy_pct}%)`} pct={clinSum.cardiomyopathy_pct || 0} color={ACCENT3} />
                </div>
                <div className="col-md-6">
                  <PctBar label={`Maternal AFLP/HELLP history (${clinSum.maternal_aflp_pct}%)`} pct={clinSum.maternal_aflp_pct || 0} color={ACCENT8} />
                  <PctBar label={`Rhabdomyolysis (CK≥5000) (${clinSum.rhabdo_pct}%)`} pct={clinSum.rhabdo_pct || 0} color={ACCENT3} />
                  <PctBar label={`Seizures (${clinSum.seizures_pct}%)`} pct={clinSum.seizures_pct || 0} color={ACCENT} />
                </div>
              </div>
            </div>
          </div>

          {/* Variants */}
          <div className="card shadow-sm">
            <div className="card-body">
              <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Pathogenic Variants</div>
              <div className="row">
                {(bd?.variants || []).map((v, i) => (
                  <div key={i} className="col-md-6 mb-2">
                    <div className="card shadow-sm h-100" style={{ borderLeft: `4px solid ${i === 0 ? ACCENT3 : ACCENT}` }}>
                      <div className="card-body py-2">
                        <div className="fw-bold" style={{ fontSize: 12, color: i === 0 ? ACCENT3 : ACCENT }}>
                          {v.variant} {i === 0 && <span className="badge bg-danger" style={{ fontSize: 9 }}>FOUNDER</span>}
                        </div>
                        <div style={{ fontSize: 11, color: '#666' }}>Freq: {v.frequency} · Gene: {v.gene}</div>
                        <div style={{ fontSize: 11, color: '#888' }}>{v.effect}</div>
                        <div style={{ fontSize: 11, color: ACCENT7 }}>{v.exam_pearl}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB 3: DEFINITIONS ── */}
      {tab === 3 && (
        <div>
          {def && (
            <>
              <div className="row mb-3">
                <div className="col-md-6">
                  <div className="card shadow-sm h-100">
                    <div className="card-body">
                      <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Disease Identity</div>
                      <table className="table table-sm" style={{ fontSize: 12 }}>
                        <tbody>
                          <tr><th>Disease</th><td>{def.disease}</td></tr>
                          <tr><th>Gene (α)</th><td>{def.gene_primary}</td></tr>
                          <tr><th>Gene (β)</th><td>{def.gene_secondary}</td></tr>
                          <tr><th>Locus</th><td>{def.locus}</td></tr>
                          <tr><th>OMIM HADHA</th><td>{def.omim_gene_hadha}</td></tr>
                          <tr><th>OMIM HADHB</th><td>{def.omim_gene_hadhb}</td></tr>
                          <tr><th>OMIM LCHAD</th><td>{def.omim_disease_lchad}</td></tr>
                          <tr><th>OMIM TFP</th><td>{def.omim_disease_tfp}</td></tr>
                          <tr><th>Inheritance</th><td>{def.inheritance}</td></tr>
                          <tr><th>Prevalence</th><td>{def.prevalence}</td></tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="card shadow-sm h-100">
                    <div className="card-body">
                      <div className="fw-bold small mb-2" style={{ color: ACCENT }}>NBS Markers</div>
                      <table className="table table-sm" style={{ fontSize: 12 }}>
                        <tbody>
                          <tr><th>Primary</th><td style={{ color: ACCENT2 }}>{def.nbs_markers?.primary}</td></tr>
                          <tr><th>Secondary</th><td>{def.nbs_markers?.secondary?.join(', ')}</td></tr>
                          <tr><th>Best Ratio</th><td>{def.nbs_markers?.best_ratio}</td></tr>
                        </tbody>
                      </table>
                      <div className="fw-bold small mb-1 mt-2" style={{ color: ACCENT4 }}>Key Negatives</div>
                      {(def.nbs_markers?.key_negatives || []).map((k, i) => (
                        <div key={i} className="small p-1 mb-1 rounded" style={{ background: '#e8f5e9', color: ACCENT4, fontSize: 11 }}>
                          ✓ {k}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Unique clinical features */}
              <div className="card shadow-sm mb-3" style={{ borderLeft: `4px solid ${ACCENT5}` }}>
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT5 }}>Unique Clinical Features (No Other FAO Disorder)</div>
                  {(def.unique_clinical_triad || []).map((f, i) => (
                    <div key={i} className="small p-2 mb-1 rounded" style={{ background: '#f3e5f5', color: ACCENT5 }}>
                      ⭐ {f}
                    </div>
                  ))}
                </div>
              </div>

              {/* Enzyme function */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Enzyme Function &amp; Protein Structure</div>
                  <div className="small text-muted mb-2">{def.enzyme_function}</div>
                  <div className="fw-bold small mb-1" style={{ color: ACCENT }}>Protein Structure</div>
                  <div className="small text-muted">{def.protein_structure}</div>
                </div>
              </div>

              {/* Pathomechanism */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT3 }}>Pathomechanism</div>
                  <div className="small text-muted">{def.pathomechanism}</div>
                </div>
              </div>

              {/* Key differentials */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT }}>Key Differentials</div>
                  <div className="row">
                    {Object.entries(def.key_differentials || {}).map(([vs, text]) => (
                      <div key={vs} className="col-md-4 mb-2">
                        <div className="card shadow-sm h-100" style={{ borderLeft: `3px solid ${ACCENT7}` }}>
                          <div className="card-body py-2">
                            <div className="fw-bold small" style={{ color: ACCENT7 }}>{vs}</div>
                            <div className="small text-muted" style={{ fontSize: 11 }}>{text}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Exam pearls */}
              <div className="card shadow-sm mb-3">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT2 }}>Key Exam Pearls</div>
                  <div className="row">
                    {(def.key_exam_pearls || []).map((pearl, i) => (
                      <div key={i} className="col-md-6 mb-1">
                        <div className="small p-2 rounded" style={{ background: '#e3f2fd', color: ACCENT2, fontSize: 11 }}>
                          &#x1f4a1; {pearl}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Related disorders */}
              <div className="card shadow-sm">
                <div className="card-body">
                  <div className="fw-bold small mb-2" style={{ color: ACCENT7 }}>Related FAO Disorders</div>
                  <div className="row">
                    {(def.related_disorders || []).map((d, i) => (
                      <div key={i} className="col-12 mb-1">
                        <div className="small p-1 rounded" style={{ background: '#f5f5f5', fontSize: 11 }}>
                          {d}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
