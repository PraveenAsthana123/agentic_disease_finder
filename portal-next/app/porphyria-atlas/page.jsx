'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

// Porphyria Atlas color palette
const COLOR  = '#4a0072';  // deep purple — heme pathway / porphyrin rings
const LIGHT  = '#f3e5f5';  // purple tint
const COLOR2 = '#b71c1c';  // danger / acute crisis
const COLOR3 = '#e65100';  // orange — cutaneous/phototoxicity
const COLOR4 = '#1b5e20';  // green — safe / curative treatments
const COLOR5 = '#0d47a1';  // blue — hepatic
const COLOR6 = '#880e4f';  // pink/maroon — erythropoietic

const GENE_COLORS = {
  HMBS:  '#b71c1c',  // AIP — most common acute (danger red)
  ALAD:  '#4a148c',  // ADP — rarest, recessive (deep purple)
  CPOX:  '#e65100',  // HCP — dual phenotype (orange)
  PPOX:  '#1565c0',  // VP — SA founder (blue)
  FECH:  '#f57f17',  // EPP — most common erythropoietic (amber)
  UROD:  '#2e7d32',  // PCT — most common overall, curative (green)
  UROS:  '#880e4f',  // CEP — rarest/most severe (maroon)
  ALAS2: '#4527a0',  // XLP — X-linked GOF (deep purple-blue)
};

const GROUP_COLORS = {
  'Acute Hepatic': '#b71c1c',
  'Acute Hepatic + Cutaneous': '#e65100',
  'Cutaneous / Erythropoietic': '#2e7d32',
};

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function BarRow({ label, pct, color = COLOR, note }) {
  return (
    <div className="mb-2">
      <div className="d-flex justify-content-between mb-1">
        <span className="small fw-semibold">{label}</span>
        <span className="small text-muted">{typeof pct === 'number' ? `${pct}%` : pct}{note ? ` — ${note}` : ''}</span>
      </div>
      {typeof pct === 'number' && (
        <div className="progress" style={{ height: 8 }}>
          <div className="progress-bar" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
        </div>
      )}
    </div>
  );
}

function AlertBox({ type = 'danger', title, children }) {
  const icons = { danger: '🚨', warning: '⚠️', info: 'ℹ️', success: '✅' };
  return (
    <div className={`alert alert-${type} py-2 px-3 mb-3`}>
      <strong>{icons[type]} {title}</strong>
      <div className="small mt-1">{children}</div>
    </div>
  );
}

export default function PorphyriaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/porphyria-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/porphyria-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/porphyria-atlas/definitions`).then(r => r.json()),
    ])
      .then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center text-muted">Loading Porphyria Atlas…</div>;
  if (err) return <div className="p-4 text-danger">Error: {err}</div>;

  return (
    <div className="container-fluid py-3 px-3">
      {/* Header */}
      <div className="rounded-3 p-3 mb-3 text-white" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #7b1fa2 100%)` }}>
        <h4 className="mb-1 fw-bold">🟣 Porphyria Atlas — Complete 8-Gene Porphyria Disorders Atlas</h4>
        <div className="small opacity-90">
          HMBS/AIP · ALAD/ADP · CPOX/HCP · PPOX/VP · FECH/EPP · UROD/PCT · UROS/CEP · ALAS2/XLP
        </div>
        <div className="small opacity-75 mt-1">
          {ov?.total_patients} patients · 8 genes · seeds {ov?.seeds_used} · Heme biosynthesis pathway disorders
        </div>
      </div>

      {/* Critical drug safety alert */}
      <AlertBox type="danger" title="CRITICAL — Porphyrinogenic AED Contraindications">
        <strong>PHT · PB · CBZ · OXC · Primidone</strong> are ABSOLUTELY CONTRAINDICATED in acute porphyria (AIP, HCP, VP, ADP).
        These CYP450 inducers upregulate ALAS1 → excess ALA/PBG → worsens neurovisceral crisis → can be FATAL.
        <br /><strong>SAFE AEDs for porphyric seizures: Levetiracetam (LEV) — first-line. Gabapentin. Clonazepam.</strong>
        <br />Always verify at: <span className="text-decoration-underline">drugs-porphyria.org</span>
      </AlertBox>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li className="nav-item" key={t}>
            <button className={`nav-link ${tab === t ? 'active fw-bold' : ''}`} style={tab === t ? { color: COLOR } : {}} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW TAB ── */}
      {tab === 'Overview' && ov && (
        <div>
          <div className="row g-2 mb-3">
            <KPI label="Total patients" value={ov.total_patients} color={COLOR} />
            <KPI label="Genes" value={ov.genes?.length} color={COLOR} />
            <KPI label="Avg onset (y)" value={ov.avg_onset_y} color={COLOR3} />
            <KPI label="Avg dx delay (y)" value={ov.avg_dx_delay_y} color={COLOR2} />
            <KPI label="Hepatic disease" value={`${ov.hepatic_disease_pct}%`} color={COLOR5} />
            <KPI label="Unsafe AED events" value={`${ov.unsafe_aed_pct}%`} color={COLOR2} />
          </div>

          {/* Gene pills */}
          <div className="mb-3">
            <div className="fw-semibold mb-2" style={{ color: COLOR }}>8-Gene Atlas Coverage</div>
            <div className="d-flex flex-wrap gap-2">
              {ov.genes?.map((g, i) => (
                <span key={g} className="badge rounded-pill px-3 py-2" style={{ backgroundColor: GENE_COLORS[g] || COLOR, fontSize: '0.82rem' }}>
                  {g} <span className="opacity-75">/ {ov.subtypes?.[i]?.split('—')[0]?.trim()}</span>
                </span>
              ))}
            </div>
          </div>

          <div className="row g-3 mb-3">
            {/* Severity */}
            <div className="col-md-5">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold small" style={{ color: COLOR }}>Severity Distribution (n=320)</div>
                <div className="card-body">
                  {ov.severity_distribution && Object.entries(ov.severity_distribution).map(([sev, n]) => (
                    <BarRow key={sev} label={sev} pct={Math.round(100 * n / ov.total_patients)} color={sev === 'Severe' ? COLOR2 : sev === 'Moderate' ? COLOR3 : COLOR4} />
                  ))}
                </div>
              </div>
            </div>

            {/* Acute vs Cutaneous */}
            <div className="col-md-7">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold small" style={{ color: COLOR }}>Porphyria Type Groups</div>
                <div className="card-body">
                  <BarRow label="Acute Hepatic Porphyria (AHP)" pct={Math.round(100 * ov.acute_n / ov.total_patients)} color={COLOR2} note="HMBS, ALAD, CPOX, PPOX" />
                  <BarRow label="Cutaneous / Erythropoietic" pct={Math.round(100 * ov.cutaneous_erythropoietic_n / ov.total_patients)} color={COLOR3} note="FECH, UROD, UROS, ALAS2" />
                  <hr className="my-2" />
                  <BarRow label="Hepatic disease complications" pct={ov.hepatic_disease_pct} color={COLOR5} />
                  <BarRow label="Unsafe AED given historically" pct={ov.unsafe_aed_pct} color={COLOR2} />
                </div>
              </div>
            </div>
          </div>

          {/* Key facts */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold small" style={{ color: COLOR }}>Key Clinical Facts</div>
            <div className="card-body">
              <ul className="list-unstyled mb-0">
                {ov.key_facts?.map((f, i) => (
                  <li key={i} className="mb-2 small">
                    <span className="me-2">🔑</span>{f}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Critical distinctions */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold small" style={{ color: COLOR2 }}>Critical Differential Diagnoses</div>
            <div className="card-body">
              <div className="row g-2">
                {ov.critical_distinctions && Object.entries(ov.critical_distinctions).map(([pair, dist]) => (
                  <div key={pair} className="col-md-6">
                    <div className="border rounded p-2 h-100" style={{ borderColor: COLOR + '40' }}>
                      <div className="fw-bold small mb-1" style={{ color: COLOR2 }}>{pair}</div>
                      <div className="text-muted" style={{ fontSize: '0.78rem' }}>{dist}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── GENE TABLE TAB ── */}
      {tab === 'Gene Table' && bd && (
        <div>
          <AlertBox type="warning" title="Diagnostic Biomarker Priority">
            Urine PBG (acute attack, AIP/HCP/VP) · Plasma porphyrin fluorescence peak 626-628nm (VP, between attacks) ·
            Erythrocyte free PP IX (EPP) · Erythrocyte ZnPP + free PP IX (XLP) · Urine uroporphyrin I (PCT, CEP) · Erythrodontia UV fluorescence (CEP PATHOGNOMONIC)
          </AlertBox>
          <div className="table-responsive">
            <table className="table table-bordered table-hover table-sm align-middle small">
              <thead style={{ backgroundColor: LIGHT }}>
                <tr>
                  <th>Gene</th>
                  <th>Subtype</th>
                  <th>Locus</th>
                  <th>Inh.</th>
                  <th>Pts</th>
                  <th>Avg onset (y)</th>
                  <th>Avg delay (y)</th>
                  <th>Hepatic (%)</th>
                  <th>Key biomarker</th>
                  <th>Drug CI</th>
                </tr>
              </thead>
              <tbody>
                {bd.breakdown?.map(g => (
                  <tr key={g.gene}>
                    <td>
                      <span className="badge rounded-pill" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR }}>
                        {g.gene}
                      </span>
                    </td>
                    <td style={{ maxWidth: 180 }}>{g.subtype?.split('—')[0]?.trim()}</td>
                    <td className="font-monospace">{g.locus}</td>
                    <td>{g.inheritance?.split('.')[0]}</td>
                    <td className="text-center">{g.n_patients}</td>
                    <td className="text-center">{g.avg_onset_y}</td>
                    <td className="text-center">{g.avg_dx_delay_y}</td>
                    <td className="text-center">
                      <span style={{ color: g.hepatic_disease_pct > 15 ? COLOR2 : COLOR4, fontWeight: g.hepatic_disease_pct > 15 ? 700 : 400 }}>
                        {g.hepatic_disease_pct}%
                      </span>
                    </td>
                    <td style={{ maxWidth: 200, fontSize: '0.73rem' }}>{g.key_biomarker?.split('.')[0]}</td>
                    <td style={{ maxWidth: 160, fontSize: '0.73rem', color: COLOR2 }}>{g.ci_drugs?.split('(')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-gene cards */}
          <div className="row g-3 mt-2">
            {bd.breakdown?.map(g => (
              <div key={g.gene} className="col-md-6 col-lg-4">
                <div className="card shadow-sm h-100" style={{ borderTop: `3px solid ${GENE_COLORS[g.gene] || COLOR}` }}>
                  <div className="card-header py-1 px-2 d-flex align-items-center gap-2">
                    <span className="badge" style={{ backgroundColor: GENE_COLORS[g.gene] || COLOR }}>{g.gene}</span>
                    <span className="small fw-semibold text-truncate">{g.subtype?.split('—')[0]?.trim()}</span>
                  </div>
                  <div className="card-body py-2 px-2">
                    <div className="small mb-1"><strong>Hallmark:</strong> <span className="text-muted" style={{ fontSize: '0.75rem' }}>{g.hallmark?.slice(0, 120)}…</span></div>
                    <div className="small mb-1"><strong>Emergency:</strong> <span className="text-muted" style={{ fontSize: '0.75rem', color: COLOR2 }}>{g.emergency?.slice(0, 100)}…</span></div>
                    <div className="small mb-1"><strong>Top Tx:</strong> {g.top_treatments?.map(t => t.treatment.split('(')[0].trim()).slice(0, 2).join(' / ')}</div>
                    {g.top_triggers && g.top_triggers.length > 0 && (
                      <div className="small"><strong>Triggers:</strong> {g.top_triggers?.slice(0, 3).map(t => t.trigger).join(' · ')}</div>
                    )}
                    <div className="mt-2 d-flex gap-1">
                      {Object.entries(g.severity_distribution).map(([sev, n]) => (
                        <span key={sev} className="badge" style={{ backgroundColor: sev === 'Severe' ? COLOR2 : sev === 'Moderate' ? COLOR3 : COLOR4, fontSize: '0.68rem' }}>
                          {sev}: {n}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS TAB ── */}
      {tab === 'Clinical Atlas' && bd && (
        <div>
          <div className="row g-3 mb-3">
            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold small" style={{ color: COLOR2 }}>Acute Hepatic Porphyrias — Treatment Protocol</div>
                <div className="card-body small">
                  <div className="mb-3">
                    <div className="fw-bold mb-1">🔴 Mild Attack</div>
                    <ol className="mb-0 ps-3">
                      <li>Stop all triggering drugs immediately</li>
                      <li>IV high-glucose saline (300g glucose/day)</li>
                      <li>Opioid analgesia for pain</li>
                      <li>LEV IV if seizures develop</li>
                      <li>Monitor electrolytes (Na⁺ for SIADH)</li>
                    </ol>
                  </div>
                  <div className="mb-3">
                    <div className="fw-bold mb-1">🚨 Moderate-Severe Attack</div>
                    <ol className="mb-0 ps-3">
                      <li>IV hematin (Panhematin/Normosang) 3-4 mg/kg/day × 4 days — give EARLY</li>
                      <li>IV glucose + saline</li>
                      <li>LEV IV for seizures (NEVER PHT/PB/CBZ/OXC)</li>
                      <li>ICU if ascending motor neuropathy (respiratory failure risk)</li>
                      <li>Correct Na⁺ slowly if SIADH (max 8-10 mmol/L/24h)</li>
                      <li>Propofol IV for status epilepticus (safe)</li>
                    </ol>
                  </div>
                  <div>
                    <div className="fw-bold mb-1">📅 Prophylaxis (recurrent attacks, ≥2/year)</div>
                    <ul className="mb-0 ps-3">
                      <li>Givosiran 2.5 mg/kg SC monthly (FDA 2019)</li>
                      <li>Monitor LFTs monthly + homocysteine</li>
                      <li>Avoid all triggering drugs (check porphyria database)</li>
                      <li>Liver transplant for refractory severe AIP unresponsive to givosiran</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="col-md-6">
              <div className="card shadow-sm h-100">
                <div className="card-header fw-semibold small" style={{ color: COLOR3 }}>Cutaneous / Erythropoietic Porphyrias — Treatment</div>
                <div className="card-body small">
                  <div className="mb-2">
                    <span className="badge mb-1" style={{ backgroundColor: GENE_COLORS['FECH'] }}>EPP (FECH)</span>
                    <ul className="mb-0 ps-3">
                      <li>Afamelanotide implant (Scenesse) — FDA 2019</li>
                      <li>Physical sunscreens (ZnO, TiO₂) + sun avoidance</li>
                      <li>Cholestyramine (reduce hepatic PP IX)</li>
                      <li>Liver transplant if hepatic failure; HSCT (curative)</li>
                    </ul>
                  </div>
                  <div className="mb-2">
                    <span className="badge mb-1" style={{ backgroundColor: GENE_COLORS['UROD'] }}>PCT (UROD)</span>
                    <ul className="mb-0 ps-3">
                      <li>Phlebotomy (450mL q2wks → ferritin 15-20 μg/L) — CURATIVE</li>
                      <li>Chloroquine 125mg twice weekly (if phlebotomy CI)</li>
                      <li>HCV DAA therapy if HCV+</li>
                      <li>Alcohol cessation + stop oestrogens</li>
                    </ul>
                  </div>
                  <div className="mb-2">
                    <span className="badge mb-1" style={{ backgroundColor: GENE_COLORS['UROS'] }}>CEP (UROS)</span>
                    <ul className="mb-0 ps-3">
                      <li>HSCT — potentially curative (initiate early before mutilation)</li>
                      <li>Chronic red cell transfusion (suppress erythropoiesis)</li>
                      <li>Splenectomy (severe haemolysis)</li>
                      <li>Gene therapy (lentiviral, clinical trial)</li>
                    </ul>
                  </div>
                  <div>
                    <span className="badge mb-1" style={{ backgroundColor: GENE_COLORS['ALAS2'] }}>XLP (ALAS2)</span>
                    <ul className="mb-0 ps-3">
                      <li>Afamelanotide (off-label) + sun avoidance</li>
                      <li>Aggressive hepatic monitoring (ALT + ZnPP quarterly)</li>
                      <li>Liver transplant + HSCT for advanced hepatic disease</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Diagnostic algorithm */}
          <div className="card shadow-sm mb-3">
            <div className="card-header fw-semibold small" style={{ color: COLOR }}>Diagnostic Algorithm — Which Porphyria?</div>
            <div className="card-body">
              <div className="row g-2 small">
                <div className="col-md-6">
                  <div className="p-2 rounded mb-2" style={{ background: '#fce4ec' }}>
                    <div className="fw-bold mb-1">Step 1: Acute attack present?</div>
                    <div>YES → Urine PBG (quantitative, spot): elevated → acute porphyria</div>
                    <div className="ms-3">↳ Plasma porphyrins fluorescence: 626-628nm → <strong>VP (PPOX)</strong></div>
                    <div className="ms-3">↳ Fecal coproporphyrins elevated → <strong>HCP (CPOX)</strong></div>
                    <div className="ms-3">↳ PBG elevated, fecal normal → <strong>AIP (HMBS)</strong></div>
                    <div className="ms-3">↳ ALA elevated but PBG NORMAL → <strong>ADP (ALAD)</strong> (exclude lead poisoning)</div>
                  </div>
                </div>
                <div className="col-md-6">
                  <div className="p-2 rounded mb-2" style={{ background: '#fff3e0' }}>
                    <div className="fw-bold mb-1">Step 2: No acute attacks — skin disease?</div>
                    <div>Blistering + adults + iron/HCV → <strong>PCT (UROD)</strong> (phlebotomy)</div>
                    <div>Blistering + neonatal + erythrodontia → <strong>CEP (UROS)</strong> (HSCT)</div>
                    <div>Blistering + dual (VP outside attack) → plasma porphyrins 626nm</div>
                    <div>Non-blistering burning pain (immediate, child) → erythrocyte PP IX</div>
                    <div className="ms-3">↳ Free PP IX high, ZnPP normal → <strong>EPP (FECH)</strong></div>
                    <div className="ms-3">↳ Both free PP IX + ZnPP high → <strong>XLP (ALAS2)</strong></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Monitoring schedule */}
          <div className="card shadow-sm">
            <div className="card-header fw-semibold small" style={{ color: COLOR5 }}>Long-term Monitoring by Gene</div>
            <div className="card-body">
              <div className="table-responsive">
                <table className="table table-sm table-bordered small mb-0">
                  <thead style={{ backgroundColor: LIGHT }}>
                    <tr><th>Gene</th><th>Frequency</th><th>Tests</th><th>Key Complication</th></tr>
                  </thead>
                  <tbody>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.HMBS }}>HMBS</span></td><td>Annual</td><td>Urine PBG, eGFR, LFTs, liver US (from age 50)</td><td>CKD (ALA nephrotoxicity), HCC (40× risk)</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.ALAD }}>ALAD</span></td><td>6-monthly</td><td>Urine ALA, eGFR, blood lead, LFTs</td><td>CKD, progressive neuropathy</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.CPOX }}>CPOX</span></td><td>Annual</td><td>Urine PBG, fecal porphyrins, eGFR</td><td>Cutaneous disease flares</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.PPOX }}>PPOX</span></td><td>Annual (plasma porphyrins lifelong)</td><td>Plasma porphyrins, urine PBG, liver US</td><td>Dual neuro + skin; plasma porphyrins never normalise</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.FECH }}>FECH</span></td><td>6-monthly</td><td>Erythrocyte PP IX, LFTs, bilirubin, ALP</td><td>Protoporphyric liver disease (5-10%)</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.UROD }}>UROD</span></td><td>6-monthly (until remission)</td><td>Urine uroporphyrins, ferritin, Hb, LFTs</td><td>Relapse if triggers return; HCV-related liver disease</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.UROS }}>UROS</span></td><td>Monthly (HSCT pre-) then 3-monthly</td><td>Hb, erythrocyte porphyrins, LFTs, ophthalmology</td><td>Haemolytic anaemia, photomutilation, corneal ulceration</td></tr>
                    <tr><td><span className="badge" style={{ backgroundColor: GENE_COLORS.ALAS2 }}>ALAS2</span></td><td>Quarterly</td><td>Erythrocyte ZnPP + free PP IX, ALT, bilirubin, liver US</td><td>Hepatic EPP (20-30%); earlier and more severe than EPP</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS TAB ── */}
      {tab === 'Definitions' && defs && (
        <div>
          <div className="row g-3">
            {defs.definitions?.map((d, i) => (
              <div key={i} className="col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header py-1 px-2 fw-semibold small" style={{ color: COLOR, backgroundColor: LIGHT }}>
                    {d.term}
                  </div>
                  <div className="card-body py-2 px-2 text-muted" style={{ fontSize: '0.8rem' }}>
                    {d.definition}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
