'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-thoracic-aortic-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'FBN1':    '#1565c0',  // deep blue     — Marfan syndrome, most common HTAD
  'TGFBR1':  '#0277bd',  // medium blue   — LDS1 bifid uvula, aggressive
  'TGFBR2':  '#01579b',  // navy blue     — LDS2 most aggressive, mean 26y
  'SMAD3':   '#4a148c',  // deep purple   — LDS3/AOS, early osteoarthritis
  'ACTA2':   '#b71c1c',  // deep red      — MSMDS, stroke, fixed pupils
  'MYH11':   '#e65100',  // burnt orange  — TAAD + PDA
  'COL3A1':  '#880e4f',  // dark magenta  — vEDS, surgery contraindicated
  'SLC2A10': '#2e7d32',  // dark green    — ATS, AR, 100% tortuosity
};

const GENE_INFO = {
  'FBN1':    { full: 'FBN1 / Fibrillin-1 / 2871aa', locus: '15q21.1', size: '2871 aa / 350 kDa', inh: 'AD', disease: 'Marfan syndrome — aortic root aneurysm + ectopia lentis (superotemporal) + tall stature TRIAD; prophylactic surgery 50 mm (45 mm rapid growth); beta-blocker + losartan lifelong; most common HTAD gene (~65%)' },
  'TGFBR1':  { full: 'TGFBR1 / TGF-β Receptor I / 503aa', locus: '9q22.33', size: '503 aa / 56 kDa', inh: 'AD', disease: 'LDS1 — BIFID UVULA + hypertelorism + craniosynostosis PATHOGNOMONIC; dissection at SMALLER diameter than MFS; surgery 45 mm (42 mm risk factors); full-body MRI/MRA annually' },
  'TGFBR2':  { full: 'TGFBR2 / TGF-β Receptor II / 567aa', locus: '3p24.1', size: '567 aa / 70 kDa', inh: 'AD', disease: 'LDS2 — MOST AGGRESSIVE; dissection mean 26 years; surgery 42-45 mm; full-body MRI/MRA every 6 MONTHS; p.Arg460Cys kinase domain most common; NEVER apply MFS 50 mm threshold' },
  'SMAD3':   { full: 'SMAD3 / SMAD Family Member 3 / 425aa', locus: '15q22.33', size: '425 aa / 48 kDa', inh: 'AD', disease: 'LDS3/AOS — EARLY OSTEOARTHRITIS < 30 years PATHOGNOMONIC (joint replacement before 40); aortic + branch vessel aneurysms; mild bifid uvula (35%); downstream TGF-β signalling' },
  'ACTA2':   { full: 'ACTA2 / Smooth Muscle Actin Alpha 2 / 375aa', locus: '10q23.31', size: '375 aa / 42 kDa', inh: 'AD', disease: 'MSMDS (p.Arg179His): TAAD + STROKE + FIXED DILATED PUPILS TRIAD PATHOGNOMONIC; Moyamoya; premature CAD < 40y; other ACTA2 = familial TAAD only' },
  'MYH11':   { full: 'MYH11 / SM Myosin Heavy Chain / 1972aa', locus: '16p13.11', size: '1972 aa / 227 kDa', inh: 'AD', disease: 'Familial TAAD + PDA PATHOGNOMONIC pairing; PDA often corrected in childhood — ASK about PDA history in relatives; rare (~2% HTAD); smooth muscle myosin' },
  'COL3A1':  { full: 'COL3A1 / Collagen Type III Alpha 1 / 1466aa', locus: '2q32.2', size: '1466 aa / 139 kDa', inh: 'AD', disease: 'vascular EDS — SPONTANEOUS RUPTURE without aneurysm; SURGERY CONTRAINDICATED (>50% mortality); CELIPROLOL 400mg ONLY evidence-based therapy (BBEST RCT); bowel/uterine rupture pathognomonic' },
  'SLC2A10': { full: 'SLC2A10 / GLUT10 / 541aa', locus: '20q13.12', size: '541 aa / 57 kDa', inh: 'AR', disease: 'Arterial Tortuosity Syndrome — AR (BOTH SEXES); 100% ARTERIAL TORTUOSITY universal diagnostic; neonatal onset; pulmonary stenosis 50%; DHA transporter → mitochondrial ascorbate → collagen' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 130, flex: '1 1 130px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: warn ? '#f87171' : '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredThoracicAorticAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (tab === 'Overview') setOverview(d);
        else if (tab === 'Gene Table' || tab === 'Clinical Atlas') setBreakdown(d);
        else setDefinitions(d);
      })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false));
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#1565c0';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '0 0 40px' }}>
      {/* Header */}
      <div style={{ background: card, borderBottom: '1px solid #334155', padding: '18px 28px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>🧬 Hereditary Disease Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          Hereditary Thoracic Aortic Disease Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>
          Complete 8-Gene Hereditary Thoracic Aortic Disease &amp; Connective Tissue Aortopathy Reference ·{' '}
          {['FBN1','TGFBR1','TGFBR2','SMAD3','ACTA2','MYH11','COL3A1','SLC2A10'].map(g => (
            <GeneChip key={g} gene={g} />
          ))}
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#475569' }}>
          320 patients · 8 × 40 · seeds 2566-2573 · FBN1/TGFBR1/TGFBR2/SMAD3 (TGF-β pathway) · ACTA2/MYH11 (smooth muscle proteins) · COL3A1 (vEDS surgery CI) · SLC2A10 (ATS AR)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, padding: '12px 28px', borderBottom: '1px solid #334155', background: '#0f172a' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? accent : '#1e293b',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: 6, padding: '7px 16px', cursor: 'pointer', fontSize: 13, fontWeight: 600,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 28px' }}>
        {loading && <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading…</div>}
        {err && <div style={{ color: '#f87171', padding: 20 }}>Error: {err}</div>}

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && overview && !loading && (
          <div>
            <div style={{ marginBottom: 18 }}>
              <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>{overview.title}</div>
              <div style={{ color: '#94a3b8', fontSize: 13 }}>{overview.subtitle}</div>
            </div>

            {/* Aggregate metrics */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub={`${overview.n_genes} genes · seeds ${overview.seeds}`} />
              <MetricCard label="Aortic Events" value={`${overview.aggregate_metrics.aortic_event_pct}%`} warn sub="dissection/surgery lifetime" />
              <MetricCard label="Surgery Rate" value={`${overview.aggregate_metrics.surgery_pct}%`} sub="aortic repair/replacement" />
              <MetricCard label="On Beta-Blocker" value={`${overview.aggregate_metrics.on_beta_blocker_pct}%`} sub="rate-control therapy" />
              <MetricCard label="On Losartan/ARB" value={`${overview.aggregate_metrics.on_losartan_pct}%`} sub="TGF-β pathway suppression" />
              <MetricCard label="Arterial Tortuosity" value={`${overview.aggregate_metrics.arterial_tortuosity_pct}%`} sub="including SLC2A10 100%" />
              <MetricCard label="Vascular Rupture" value={`${overview.aggregate_metrics.vascular_rupture_pct}%`} warn sub="spontaneous/surgical" />
            </div>

            {/* Disease classes */}
            <div style={{ background: card, borderRadius: 8, padding: 20, marginBottom: 20 }}>
              <div style={{ fontWeight: 700, marginBottom: 12 }}>Disease Classes by Gene</div>
              {(overview.disease_classes || []).map((cls, i) => {
                const gene = cls.split(' —')[0].trim();
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, marginBottom: 8 }}>
                    <GeneChip gene={gene} />
                    <span style={{ color: '#cbd5e1', fontSize: 13, lineHeight: 1.5 }}>{cls.split(' —').slice(1).join(' —').trim()}</span>
                  </div>
                );
              })}
            </div>

            {/* Gene summary table */}
            <div style={{ background: card, borderRadius: 8, padding: 20, marginBottom: 20, overflowX: 'auto' }}>
              <div style={{ fontWeight: 700, marginBottom: 12 }}>Gene Summary — 320 Patients</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Gene</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Locus</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Inh</th>
                    <th style={{ textAlign: 'left', padding: '6px 8px' }}>Disease</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Onset (y)</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Events%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Surgery%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Tortuosity%</th>
                    <th style={{ textAlign: 'right', padding: '6px 8px' }}>Rupture%</th>
                  </tr>
                </thead>
                <tbody>
                  {(overview.gene_summary || []).map((g, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #1e293b', cursor: 'pointer' }}
                        onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}>
                      <td style={{ padding: '7px 8px' }}><GeneChip gene={g.gene} /></td>
                      <td style={{ padding: '7px 8px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '7px 8px', color: '#94a3b8' }}>{g.inheritance}</td>
                      <td style={{ padding: '7px 8px', color: '#cbd5e1', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g.disease_name}</td>
                      <td style={{ padding: '7px 8px', textAlign: 'right', color: '#94a3b8' }}>{g.onset_age_years_median}</td>
                      <td style={{ padding: '7px 8px', textAlign: 'right', color: g.aortic_event_pct > 50 ? '#f87171' : '#86efac' }}>{g.aortic_event_pct}%</td>
                      <td style={{ padding: '7px 8px', textAlign: 'right', color: '#94a3b8' }}>{g.surgery_required_pct}%</td>
                      <td style={{ padding: '7px 8px', textAlign: 'right', color: g.arterial_tortuosity_pct > 80 ? '#fbbf24' : '#94a3b8' }}>{g.arterial_tortuosity_pct}%</td>
                      <td style={{ padding: '7px 8px', textAlign: 'right', color: g.vascular_rupture_risk_pct > 60 ? '#f87171' : '#94a3b8' }}>{g.vascular_rupture_risk_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Clinical pearls */}
            <div style={{ background: card, borderRadius: 8, padding: 20 }}>
              <div style={{ fontWeight: 700, marginBottom: 12, color: '#fbbf24' }}>⚠ Critical Clinical Pearls</div>
              {(overview.clinical_pearls || []).map((p, i) => (
                <div key={i} style={{ marginBottom: 10, color: '#cbd5e1', fontSize: 13, lineHeight: 1.6, paddingLeft: 12, borderLeft: '3px solid #334155' }}>
                  {p}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === 'Gene Table' && breakdown && !loading && (
          <div>
            <div style={{ marginBottom: 16, color: '#94a3b8', fontSize: 13 }}>
              Per-gene breakdown — 40 patients each (seeds 2566-2573). Click a row to expand.
            </div>
            {(breakdown.gene_breakdowns || []).map((g, i) => (
              <div key={i} style={{ background: card, borderRadius: 8, marginBottom: 12, overflow: 'hidden' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 18px', cursor: 'pointer',
                              borderBottom: selGene === g.gene ? '1px solid #334155' : 'none' }}
                     onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}>
                  <GeneChip gene={g.gene} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 700, fontSize: 14 }}>{GENE_INFO[g.gene]?.full || g.gene}</div>
                    <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>
                      {g.locus} · {g.protein_size} · {g.inheritance.split(';')[0]}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#94a3b8' }}>
                    <span>Events: <b style={{ color: g.aortic_event_pct > 50 ? '#f87171' : '#86efac' }}>{g.aortic_event_pct}%</b></span>
                    <span>Surgery: <b>{g.surgery_pct}%</b></span>
                    <span>Rupture: <b style={{ color: g.vascular_rupture_pct > 60 ? '#f87171' : '#94a3b8' }}>{g.vascular_rupture_pct}%</b></span>
                    <span style={{ color: '#475569' }}>{selGene === g.gene ? '▲' : '▼'}</span>
                  </div>
                </div>
                {selGene === g.gene && (
                  <div style={{ padding: '16px 18px', fontSize: 13 }}>
                    <div style={{ marginBottom: 10 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Disease: </span>
                      <span style={{ color: '#cbd5e1' }}>{g.disease_category}</span>
                    </div>
                    <div style={{ marginBottom: 10 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Pathognomonic: </span>
                      <span style={{ color: '#fbbf24' }}>{g.pathognomonic}</span>
                    </div>
                    <div style={{ marginBottom: 10 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Treatment: </span>
                      <span style={{ color: '#86efac' }}>{g.treatment}</span>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>Key Features: </span>
                      <ul style={{ margin: '6px 0 0 16px', color: '#cbd5e1' }}>
                        {(g.key_features || []).map((f, j) => <li key={j} style={{ marginBottom: 3 }}>{f}</li>)}
                      </ul>
                    </div>
                    <div>
                      <span style={{ color: '#64748b', fontWeight: 700 }}>DDx: </span>
                      <ul style={{ margin: '6px 0 0 16px', color: '#94a3b8' }}>
                        {(g.key_ddx || []).map((d, j) => <li key={j} style={{ marginBottom: 3 }}>{d}</li>)}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {tab === 'Clinical Atlas' && breakdown && !loading && (
          <div>
            <div style={{ marginBottom: 16, fontWeight: 700, color: '#f1f5f9' }}>Clinical Atlas — 8 Gene Profiles</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 16 }}>
              {(breakdown.gene_breakdowns || []).map((g, i) => (
                <div key={i} style={{ background: card, borderRadius: 10, padding: 18, borderTop: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <GeneChip gene={g.gene} />
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 13 }}>{GENE_INFO[g.gene]?.full || g.gene}</div>
                      <div style={{ color: '#64748b', fontSize: 11 }}>{g.locus} · {g.protein_size}</div>
                    </div>
                  </div>
                  <div style={{ marginBottom: 8, fontSize: 12, color: '#94a3b8' }}>
                    <b style={{ color: '#64748b' }}>Inheritance:</b> {g.inheritance.split(';')[0]}
                  </div>
                  <div style={{ marginBottom: 8, fontSize: 12, lineHeight: 1.5, color: '#cbd5e1' }}>
                    {GENE_INFO[g.gene]?.disease}
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                    <div style={{ background: '#0f172a', borderRadius: 6, padding: '5px 10px', fontSize: 11 }}>
                      <span style={{ color: '#64748b' }}>Events: </span>
                      <span style={{ color: g.aortic_event_pct > 50 ? '#f87171' : '#86efac', fontWeight: 700 }}>{g.aortic_event_pct}%</span>
                    </div>
                    <div style={{ background: '#0f172a', borderRadius: 6, padding: '5px 10px', fontSize: 11 }}>
                      <span style={{ color: '#64748b' }}>Surgery: </span>
                      <span style={{ fontWeight: 700 }}>{g.surgery_pct}%</span>
                    </div>
                    <div style={{ background: '#0f172a', borderRadius: 6, padding: '5px 10px', fontSize: 11 }}>
                      <span style={{ color: '#64748b' }}>Tortuosity: </span>
                      <span style={{ color: g.arterial_tortuosity_pct > 80 ? '#fbbf24' : '#94a3b8', fontWeight: 700 }}>{g.arterial_tortuosity_pct}%</span>
                    </div>
                    <div style={{ background: '#0f172a', borderRadius: 6, padding: '5px 10px', fontSize: 11 }}>
                      <span style={{ color: '#64748b' }}>Rupture: </span>
                      <span style={{ color: g.vascular_rupture_pct > 60 ? '#f87171' : '#94a3b8', fontWeight: 700 }}>{g.vascular_rupture_pct}%</span>
                    </div>
                  </div>
                  <div style={{ marginTop: 10, background: '#0f172a', borderRadius: 6, padding: '8px 10px', fontSize: 11, color: '#fbbf24', lineHeight: 1.5 }}>
                    ⚡ {g.pathognomonic?.split('. ')[0]}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === 'Definitions' && definitions && !loading && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 14 }}>Gene Reference</div>
              {Object.entries(definitions.gene_entries || {}).map(([gene, info]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: 18, marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                    <GeneChip gene={gene} />
                    <div style={{ fontWeight: 700, fontSize: 14 }}>{info.full_name}</div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 8, marginBottom: 12, fontSize: 12 }}>
                    <div><span style={{ color: '#64748b' }}>Locus: </span><span style={{ color: '#cbd5e1' }}>{info.locus}</span></div>
                    <div><span style={{ color: '#64748b' }}>Size: </span><span style={{ color: '#cbd5e1' }}>{info.protein_size}</span></div>
                    <div><span style={{ color: '#64748b' }}>Inheritance: </span><span style={{ color: '#cbd5e1' }}>{info.inheritance}</span></div>
                    <div><span style={{ color: '#64748b' }}>Aortic Events: </span><span style={{ color: info.aortic_event_pct > 50 ? '#f87171' : '#86efac', fontWeight: 700 }}>{info.aortic_event_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Rupture Risk: </span><span style={{ color: info.vascular_rupture_risk_pct > 60 ? '#f87171' : '#94a3b8', fontWeight: 700 }}>{info.vascular_rupture_risk_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>NBS: </span><span style={{ color: '#94a3b8' }}>{info.nbs_indicated ? 'Yes' : 'Not standard'}</span></div>
                  </div>
                  <div style={{ marginBottom: 8, fontSize: 12, color: '#cbd5e1' }}>
                    <b style={{ color: '#64748b' }}>Pathognomonic: </b>{info.pathognomonic}
                  </div>
                  <div style={{ marginBottom: 8, fontSize: 12, color: '#86efac' }}>
                    <b style={{ color: '#64748b' }}>Treatment: </b>{info.treatment}
                  </div>
                </div>
              ))}
            </div>

            <div>
              <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 14 }}>HTAD Glossary</div>
              {Object.entries(definitions.htad_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ background: card, borderRadius: 8, padding: 18, marginBottom: 10 }}>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#fbbf24', marginBottom: 8 }}>{term}</div>
                  <div style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.7 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
