'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';  // deep indigo — urea cycle / nitrogen metabolism
const LIGHT  = '#e8eaf6';
const COLOR2 = '#b71c1c';  // danger — VPA CI / ARG1 arginine CI
const COLOR3 = '#1b5e20';  // NCG curative — NAGS
const COLOR4 = '#e65100';  // VPA risk
const COLOR5 = '#4a148c';  // liver transplant
const COLOR6 = '#006064';  // NBS
const COLOR7 = '#37474f';  // orotate marker

const CLASS_COLORS = {
  nag_synthase:               '#1b5e20',  // green — NCG curative
  carbamoyl_phosphate_synthetase: '#1a237e',  // indigo — CPS1
  ornithine_transcarbamylase: '#880e4f',  // X-linked red-purple
  argininosuccinate_synthase: '#0d47a1',  // dark blue — ASS1
  argininosuccinate_lyase:    '#bf360c',  // burnt orange — ASL
  arginase_1:                 '#4e342e',  // brown — ARG1 (arginine CI)
};

const CLASS_LABELS = {
  nag_synthase:               'NAG Synthase — NCG-curative allosteric master switch (NAGS)',
  carbamoyl_phosphate_synthetase: 'Carbamoyl Phosphate Synthetase — most severe neonatal UCD (CPS1)',
  ornithine_transcarbamylase: 'Ornithine Transcarbamylase — MOST COMMON, ONLY X-linked UCD (OTC)',
  argininosuccinate_synthase: 'Argininosuccinate Synthase — Citrullinemia type 1, very high citrulline (ASS1)',
  argininosuccinate_lyase:    'Argininosuccinate Lyase — ASA disease, trichorrhexis nodosa, neuro despite control (ASL)',
  arginase_1:                 'Arginase 1 — Hyperargininemia, spastic diplegia, arginine CI (ARG1)',
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

function BarRow({ label, pct, color = COLOR }) {
  return (
    <div className="mb-1">
      <div className="d-flex justify-content-between mb-0" style={{ fontSize: '0.78rem' }}>
        <span>{label}</span><span className="fw-semibold">{pct}%</span>
      </div>
      <div className="progress" style={{ height: '7px' }}>
        <div className="progress-bar" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

export default function UCDAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [loading, setLoading]   = useState(true);
  const [err, setErr]           = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/ucd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/ucd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/ucd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov); setBreakdown(bd); setDefs(df); setLoading(false);
    }).catch(e => { setErr(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: COLOR }} /><p className="mt-2">Loading UCD-Atlas…</p></div>;
  if (err)     return <div className="p-4 alert alert-danger">Error: {err}</div>;

  const ac  = overview?.aggregate_clinical || {};
  const ds  = overview?.drug_safety || {};
  const uc  = overview?.urea_cycle || {};
  const op  = overview?.orotate_profile || {};
  const genes = breakdown?.genes || [];

  return (
    <div className="container-fluid py-3">
      {/* Header */}
      <div className="mb-3 p-3 rounded text-white" style={{ background: `linear-gradient(135deg, ${COLOR} 0%, #283593 100%)` }}>
        <h4 className="mb-1 fw-bold">🧬 UCD-Atlas — Complete 6-Gene Urea Cycle Disorders Atlas</h4>
        <div style={{ fontSize: '0.82rem', opacity: 0.92 }}>
          {overview?.atlas_scope} &nbsp;·&nbsp; {overview?.n_genes} genes &nbsp;·&nbsp; {overview?.n_patients} patients &nbsp;·&nbsp; seeds {overview?.seeds}
        </div>
        <div className="mt-1" style={{ fontSize: '0.78rem', opacity: 0.85 }}>
          {uc.pathway}
        </div>
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t} className="nav-item">
            <button className={`nav-link${tab === t ? ' active fw-semibold' : ''}`} onClick={() => setTab(t)}>{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview ── */}
      {tab === 'Overview' && (
        <div>
          {/* KPIs */}
          <div className="row g-2 mb-3">
            <KPI label="Genes" value={overview?.n_genes} color={COLOR} />
            <KPI label="Patients" value={overview?.n_patients} color={COLOR} />
            <KPI label="Neonatal Severe" value={`${ac.neonatal_severe_pct}%`} color={COLOR2} />
            <KPI label="NH3 > 500 µmol/L" value={`${ac.nh3_over_500_pct}%`} color={COLOR2} />
            <KPI label="Liver Tx" value={`${ac.liver_tx_pct}%`} color={COLOR5} />
            <KPI label="NCG Responsive" value={`${ac.ncg_responsive_pct}%`} color={COLOR3} />
          </div>

          <div className="row g-3">
            {/* Urea Cycle Overview */}
            <div className="col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Urea Cycle Pathway</div>
                <div className="card-body" style={{ fontSize: '0.82rem' }}>
                  <div className="p-2 rounded mb-2" style={{ background: '#e8eaf6', fontFamily: 'monospace', fontSize: '0.78rem' }}>
                    NH₃ + HCO₃⁻ → Carbamoyl-P <span className="text-muted">(CPS1, NAG-activated by NAGS)</span><br/>
                    &nbsp;&nbsp;→ Citrulline <span className="text-muted">(OTC)</span> [mitochondrial steps]<br/>
                    &nbsp;&nbsp;→ Argininosuccinate <span className="text-muted">(ASS1)</span> [cytosolic]<br/>
                    &nbsp;&nbsp;→ Arginine + Fumarate <span className="text-muted">(ASL)</span><br/>
                    &nbsp;&nbsp;→ Urea + Ornithine <span className="text-muted">(ARG1)</span>
                  </div>
                  {[
                    ['NAGS role', uc.nags_role],
                    ['X-linked gene', uc.x_linked_gene],
                    ['NCG-curative', uc.ncg_curative_gene],
                    ['Arginine CI', uc.arginine_ci_gene],
                    ['Most common', uc.most_common],
                  ].map(([k,v]) => (
                    <div key={k} className="d-flex gap-2 mb-1" style={{ fontSize: '0.78rem' }}>
                      <span className="fw-semibold" style={{ minWidth: 120, color: COLOR }}>{k}:</span>
                      <span>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Clinical aggregate */}
            <div className="col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>Aggregate Clinical (n=240)</div>
                <div className="card-body">
                  <BarRow label="Neonatal/severe form" pct={ac.neonatal_severe_pct} color={COLOR2} />
                  <BarRow label="Encephalopathy" pct={ac.encephalopathy_pct} color={COLOR2} />
                  <BarRow label="Cerebral oedema" pct={ac.cerebral_oedema_pct} color={COLOR2} />
                  <BarRow label="Protein aversion (self-selection)" pct={ac.protein_aversion_pct} color={COLOR4} />
                  <BarRow label="Liver transplant" pct={ac.liver_tx_pct} color={COLOR5} />
                  <BarRow label="NH3 > 500 µmol/L" pct={ac.nh3_over_500_pct} color={COLOR2} />
                  <BarRow label="NCG responsive (NAGS cohort)" pct={ac.ncg_responsive_pct} color={COLOR3} />
                  <div className="mt-2" style={{ fontSize: '0.75rem', color: '#888' }}>
                    ASL-specific: spastic diplegia {ac.spastic_diplegia_arg1_pct}% (ARG1) · trichorrhexis nodosa {ac.trichorrhexis_asl_pct}% (ASL) · hypertension {ac.hypertension_asl_pct}% (ASL)
                  </div>
                </div>
              </div>
            </div>

            {/* Orotate discriminator */}
            <div className="col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR7 }}>Orotate — KEY UCD Discriminator</div>
                <div className="card-body" style={{ fontSize: '0.8rem' }}>
                  <div className="alert alert-warning py-1 px-2 mb-2" style={{ fontSize: '0.78rem' }}>
                    <strong>Rule:</strong> {op.key_rule}
                  </div>
                  {Object.entries(op).filter(([k]) => k !== 'key_rule').map(([gene, note]) => (
                    <div key={gene} className="d-flex gap-2 mb-1">
                      <span className="badge" style={{ background: CLASS_COLORS[Object.keys(CLASS_COLORS)[['NAGS','CPS1','OTC','ASS1','ASL','ARG1'].indexOf(gene)]] || COLOR, minWidth: 48 }}>{gene}</span>
                      <span style={{ fontSize: '0.77rem' }}>{note}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Drug safety */}
            <div className="col-md-6">
              <div className="card h-100 shadow-sm">
                <div className="card-header fw-semibold" style={{ background: '#ffebee', color: COLOR2 }}>Drug Safety — All 6 Genes</div>
                <div className="card-body" style={{ fontSize: '0.8rem' }}>
                  <div className="mb-2 p-2 rounded" style={{ background: '#ffebee', border: `1px solid ${COLOR2}` }}>
                    <strong style={{ color: COLOR2 }}>VPA HIGH RISK ALL 6:</strong> {ds.vpa_mechanism}
                  </div>
                  <div className="mb-2 p-2 rounded" style={{ background: '#e8f5e9', border: '1px solid #1b5e20' }}>
                    <strong style={{ color: COLOR3 }}>NCG (Carbaglu) CURATIVE:</strong> ONLY for NAGS deficiency — 100% normalise NH3. Not indicated for other 5 genes.
                  </div>
                  <div className="mb-2 p-2 rounded" style={{ background: '#fbe9e7', border: `1px solid ${COLOR4}` }}>
                    <strong style={{ color: COLOR4 }}>ARG1 — Arginine ABSOLUTE CI:</strong> arginine IS the toxic metabolite; never supplement in ARG1. Opposite rule for all other 5 UCDs.
                  </div>
                  {[
                    ['Preferred AED', ds.preferred_aed],
                    ['Ammonia scavengers', ds.ammonia_scavengers],
                    ['GIR (crisis)', ds.gir],
                    ['Zero protein (crisis)', ds.zero_protein_crisis],
                    ['Haemodialysis', ds.haemodialysis],
                  ].map(([k, v]) => (
                    <div key={k} className="d-flex gap-2 mb-1" style={{ fontSize: '0.77rem' }}>
                      <span className="fw-semibold" style={{ minWidth: 140, color: COLOR }}>{k}:</span>
                      <span>{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 'Gene Table' && (
        <div>
          <div className="row g-3">
            {genes.map(g => (
              <div key={g.gene} className="col-12 col-lg-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header text-white fw-semibold d-flex justify-content-between align-items-center"
                    style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>
                    <span>{g.gene} — {g.aa} · {g.kDa}</span>
                    <span className="badge bg-light" style={{ color: CLASS_COLORS[g.gene_class] || COLOR, fontSize: '0.7rem' }}>{g.locus}</span>
                  </div>
                  <div className="card-body" style={{ fontSize: '0.8rem' }}>
                    <div className="fw-semibold mb-1">{CLASS_LABELS[g.gene_class]}</div>
                    <div className="text-muted mb-2" style={{ fontSize: '0.75rem' }}>{g.inheritance}</div>
                    <div className="row g-1 mb-2">
                      {[
                        ['Neonatal/severe', `${g.pct_neonatal_severe}%`],
                        ['Liver Tx', `${g.pct_liver_tx}%`],
                        ['NCG resp.', `${g.pct_ncg_response}%`],
                        ['Peak NH3 (med.)', `${g.median_peak_nh3} µM`],
                      ].map(([label, val]) => (
                        <div key={label} className="col-6">
                          <div className="bg-light rounded p-1 text-center">
                            <div className="fw-bold" style={{ color: CLASS_COLORS[g.gene_class] || COLOR, fontSize: '0.85rem' }}>{val}</div>
                            <div className="text-muted" style={{ fontSize: '0.68rem' }}>{label}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mb-1"><span className="fw-semibold">NBS profile: </span>{g.nbs_profile}</div>
                    <div className="mb-1"><span className="fw-semibold">Orotate: </span>{g.orotate}</div>
                    <div className="mb-1"><span className="fw-semibold">Arginine rule: </span>
                      <span style={{ color: g.gene === 'ARG1' ? COLOR2 : COLOR3 }}>{g.arginine_rule}</span>
                    </div>
                    <div className="mb-1"><span className="fw-semibold">NCG: </span>{g.ncg_response}</div>
                    <div className="mb-1"><span className="fw-semibold">Liver Tx: </span>{g.liver_transplant}</div>
                    <div className="mb-1"><span className="fw-semibold text-danger">Drug CI: </span>{g.critical_ci}</div>
                    <div className="mb-1"><span className="fw-semibold">Key DDx: </span><span className="text-muted" style={{ fontSize: '0.75rem' }}>{g.key_ddx}</span></div>
                    <div><span className="fw-semibold">Founder: </span><span className="text-muted" style={{ fontSize: '0.75rem' }}>{g.founder}</span></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Clinical Atlas ── */}
      {tab === 'Clinical Atlas' && (
        <div>
          <div className="row g-3">
            {genes.map(g => (
              <div key={g.gene} className="col-12">
                <div className="card shadow-sm">
                  <div className="card-header text-white fw-semibold"
                    style={{ background: CLASS_COLORS[g.gene_class] || COLOR }}>
                    {g.gene} — {CLASS_LABELS[g.gene_class]}
                  </div>
                  <div className="card-body">
                    <div className="row g-2">
                      <div className="col-md-6">
                        <div className="fw-semibold mb-1" style={{ color: COLOR }}>Hallmarks</div>
                        <div style={{ fontSize: '0.78rem', whiteSpace: 'pre-line' }}>{g.hallmark}</div>
                      </div>
                      <div className="col-md-6">
                        <div className="fw-semibold mb-1" style={{ color: COLOR }}>Disease Mechanism</div>
                        <div style={{ fontSize: '0.78rem' }}>{g.disease}</div>
                        {g.key_variants && (
                          <>
                            <div className="fw-semibold mt-2 mb-1" style={{ color: COLOR }}>Key Pathogenic Variants</div>
                            <ul className="mb-0" style={{ fontSize: '0.76rem', paddingLeft: '1.2rem' }}>
                              {g.key_variants.map((v, i) => <li key={i}>{v}</li>)}
                            </ul>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Cross-gene rules panel */}
            <div className="col-12">
              <div className="card shadow-sm">
                <div className="card-header fw-semibold" style={{ background: '#fff3e0', color: '#e65100' }}>
                  Cross-Gene Pharmacological Rules — All 6 UCD Genes
                </div>
                <div className="card-body">
                  <div className="row g-2" style={{ fontSize: '0.8rem' }}>
                    {[
                      { title: 'VPA — HIGH RISK ALL 6', color: COLOR2,
                        body: 'VPA directly inhibits NAGS → NAG deficiency → CPS1 hypoactive → hyperammonemia. This mechanism is pharmacological (not disease-specific) and operates in ANY patient on VPA. In UCD patients: potentially fatal hyperammonemic crisis. In OTC female carriers: may precipitate first symptomatic crisis. RULE: Never use VPA in any known or suspected UCD. Always check amino acids + urine orotate before starting VPA in patients with encephalopathy or DD.' },
                      { title: 'NCG (Carbaglu) — ONLY for NAGS', color: COLOR3,
                        body: 'NCG is a structural NAG analogue that bypasses NAGS and directly activates CPS1. CURATIVE for NAGS deficiency (100% NH3 normalisation). NOT indicated for OTC, ASS1, ASL, ARG1 (blocks are downstream of CPS1). May partially help CPS1 (30-50% partial CPS1 activation). Trial of NCG is mandatory in any patient with low citrulline + low orotate to distinguish NAGS (curative) from CPS1 (partial/no response).' },
                      { title: 'Arginine — Supplemented in 5, CONTRAINDICATED in 1', color: COLOR4,
                        body: 'ARGININE SUPPLEMENTATION (200-500 mg/kg/day): INDICATED in NAGS/CPS1/OTC/ASS1/ASL — arginine is an essential amino acid in all these UCDs (cannot be synthesised beyond the block); supplementation drives nitrogen excretion and corrects arginine deficiency. EXCEPTION — ARG1: arginine IS the toxic metabolite → supplementation is ABSOLUTELY CONTRAINDICATED. Use essential AA formula without arginine. THIS IS THE MOST IMPORTANT UCD DRUG RULE.' },
                      { title: 'Ammonia Scavengers — Emergency Protocol', color: COLOR5,
                        body: 'Sodium benzoate 250 mg/kg + sodium phenylacetate 250 mg/kg IV (Ammonul): conjugate glycine (benzoate) and glutamine (phenylacetate) → hippuric acid + phenylacetylglutamine (renal excretion). GIR 8-12 mg/kg/min: anti-catabolic, prevents endogenous protein catabolism. Zero protein 24-48h. Arginine 200-500 mg/kg/day IV (EXCEPT ARG1). CVVHDF/haemodiafiltration for NH3 >500 µmol/L (peritoneal dialysis inadequate). Repeat NH3 every 2-4h.' },
                      { title: 'Liver Transplant — Rule of Benefit', color: '#1a237e',
                        body: 'CURATIVE for metabolic function: CPS1, OTC (males), ASS1 — liver transplant essentially eliminates hyperammonemia risk; patient may liberalise protein intake. PARTIAL benefit in ASL — neurological disease (ArgSuc/NO pathway) persists post-LT even with normalised NH3. PARTIAL benefit in ARG1 — progressive neurological damage may stabilise but not reverse. NOT NEEDED in NAGS — NCG is curative without LT. LT decision: weigh surgical risk against morbidity of dietary restriction + crisis risk.' },
                    ].map(({ title, color, body }) => (
                      <div key={title} className="col-12 col-md-6">
                        <div className="p-2 rounded h-100" style={{ background: '#f8f9fa', borderLeft: `4px solid ${color}` }}>
                          <div className="fw-semibold mb-1" style={{ color }}>{title}</div>
                          <div style={{ fontSize: '0.77rem' }}>{body}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && (
        <div>
          <div className="row g-2">
            {(defs?.definitions || []).map((d, i) => (
              <div key={i} className="col-12 col-md-6">
                <div className="card shadow-sm h-100">
                  <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR, fontSize: '0.85rem' }}>
                    {d.term}
                  </div>
                  <div className="card-body" style={{ fontSize: '0.79rem' }}>{d.definition}</div>
                </div>
              </div>
            ))}
          </div>
          {/* UCD-wide summary card */}
          <div className="mt-3 card shadow-sm">
            <div className="card-header fw-semibold" style={{ background: LIGHT, color: COLOR }}>UCD-Atlas Quick Reference</div>
            <div className="card-body" style={{ fontSize: '0.8rem' }}>
              <div className="row g-2">
                <div className="col-md-6">
                  <div className="fw-semibold mb-1">Orotate Discriminator</div>
                  {Object.entries(defs?.urea_cycle || {}).slice(0,3).map(([k,v]) => (
                    <div key={k} className="mb-1"><span className="fw-semibold">{k}: </span>{Array.isArray(v) ? v.join(' · ') : v}</div>
                  ))}
                </div>
                <div className="col-md-6">
                  <div className="fw-semibold mb-1">Emergency Protocol (ALL 6)</div>
                  <ul className="mb-0" style={{ paddingLeft: '1.2rem', fontSize: '0.77rem' }}>
                    <li>Stop protein 24-48h (zero protein in crisis)</li>
                    <li>GIR 8-12 mg/kg/min (anti-catabolic)</li>
                    <li>Sodium benzoate + phenylacetate + arginine IV (NOT ARG1)</li>
                    <li>CVVHDF if NH3 &gt; 500 µmol/L</li>
                    <li>NCG trial (NAGS) or VPA cessation (VPA-induced)</li>
                    <li>BTBGD mandatory exclusion (Carbaglu + biotin trial)</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
