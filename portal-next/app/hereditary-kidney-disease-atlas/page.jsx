'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-kidney-disease-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'PKD1':   '#1565c0',  // deep blue — ADPKD1 most common; tolvaptan
  'PKD2':   '#0277bd',  // medium blue — ADPKD2 milder; 20yr later ESRD
  'PKHD1':  '#4a148c',  // deep purple — ARPKD; CHF; Potter; AR
  'COL4A5': '#b71c1c',  // deep red — X-linked Alport; lenticonus PATHOGNOMONIC
  'COL4A3': '#e65100',  // burnt orange — AR Alport / TBMN dual phenotype
  'UMOD':   '#1b5e20',  // dark green — ADTKD-UMOD; young gout + CKD
  'HNF1B':  '#f57f17',  // amber — RCAD; MODY5; pancreatic atrophy; mullerian
  'NPHS2':  '#880e4f',  // dark magenta — podocin SRNS2; steroid-resistant; FSGS
};

const GENE_INFO = {
  'PKD1':   { full: 'PKD1 / Polycystin-1 / 4303aa', locus: '16p13.3', size: '4303 aa / 462 kDa', inh: 'AD (two-hit)', disease: 'ADPKD1 ~85%; ESRD ~54yr; tolvaptan FDA 2018; low BP <110/75 ACEi/ARB; ICA 10%; liver cysts 80%; Mayo 1C-1E = rapidly progressive' },
  'PKD2':   { full: 'PKD2 / Polycystin-2 (TRPP2) / 968aa', locus: '4q22.1', size: '968 aa / 110 kDa', inh: 'AD (two-hit)', disease: 'ADPKD2 ~15%; ESRD ~74yr (20yr milder); genotyping essential for prognosis; many die of unrelated causes; tolvaptan if Mayo 1C-1E' },
  'PKHD1':  { full: 'PKHD1 / Fibrocystin-Polyductin / 4074aa', locus: '6p12.2', size: '4074 aa / 447 kDa', inh: 'AR biallelic', disease: 'ARPKD; massive echogenic kidneys + CHF + portal HTN; null/null → Potter; liver synthesis NORMAL despite portal HTN; combined liver-kidney transplant' },
  'COL4A5': { full: 'COL4A5 / Collagen α5(IV) / 1454aa', locus: 'Xq22.3', size: '1454 aa / 161 kDa', inh: 'XLD/XLR', disease: 'X-linked Alport (~80%); anterior lenticonus PATHOGNOMONIC; SNHL 90% males; GBM EM basket-weave; ACEi EARLY delays ESRD ~10yr; ESRD males ~25yr' },
  'COL4A3': { full: 'COL4A3 / Collagen α3(IV) / 1670aa', locus: '2q36.3', size: '1670 aa / 186 kDa', inh: 'AR/AD dual', disease: 'AR biallelic → Alport ESRD ~25yr; AD het → TBMN (1% population; most common microscopic haematuria; GBM thin <150nm NO lamellation; 20% progress to CKD)' },
  'UMOD':   { full: 'UMOD / Uromodulin-Tamm-Horsfall / 640aa', locus: '16p12.3', size: '640 aa / 85 kDa', inh: 'AD', disease: 'ADTKD-UMOD; young gout <35yr + CKD + low FEUA <6% = triad; medullary cysts on MRI (not USS); allopurinol; AVOID NSAIDs; transplant CURATIVE (no recurrence)' },
  'HNF1B':  { full: 'HNF1B / HNF-1β Transcription Factor / 557aa', locus: '17q12', size: '557 aa / 68 kDa', inh: 'AD (50% de novo)', disease: 'RCAD: renal cysts + MODY5 + pancreatic atrophy + mullerian anomalies; 17q12 deletion (chromosomal microarray mandatory); MODY5 insulin-requiring; sulphonylureas LESS effective' },
  'NPHS2':  { full: 'NPHS2 / Podocin / 383aa', locus: '1q25.2', size: '383 aa / 42 kDa', inh: 'AR biallelic', disease: 'SRNS2 steroid-RESISTANT; FSGS on biopsy; p.R138Q European founder; genetics BEFORE steroids; ACEi mandatory; anticoagulate if albumin <20g/L; transplant CURATIVE (no recurrence)' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryKidneyDiseaseAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true); setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#1565c0';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ background: card, borderRadius: 12, padding: '20px 24px', marginBottom: 20, borderLeft: `4px solid ${accent}` }}>
          <h1 style={{ margin: 0, fontSize: 20, color: '#7dd3fc' }}>&#x1fac0; Hereditary Kidney Disease Atlas</h1>
          <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: 13 }}>
            Complete 8-Gene Reference — PKD1 · PKD2 · PKHD1 · COL4A5 · COL4A3 · UMOD · HNF1B · NPHS2 &nbsp;|&nbsp; 320 patients · Seeds 2598–2605
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 10 }}>
            {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
          </div>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)}
              style={{ background: tab === t ? accent : card, color: tab === t ? '#fff' : '#94a3b8', border: 'none', borderRadius: 6, padding: '8px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400 }}>
              {t}
            </button>
          ))}
        </div>

        {loading && <div style={{ color: '#94a3b8', padding: 20 }}>Loading…</div>}
        {err && <div style={{ color: '#ef4444', padding: 20 }}>Error: {err}</div>}

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
              <MetricCard label="Genes" value={overview.n_genes} sub="Kidney Disease" />
              <MetricCard label="Seeds" value={overview.seeds} />
              <MetricCard label="Overall Avg GFR" value={overview.aggregate_metrics?.overall_avg_gfr ?? '—'} sub="ml/min/1.73m²" />
              <MetricCard label="Hypertension" value={`${overview.aggregate_metrics?.overall_hypertension_pct ?? '—'}%`} warn={overview.aggregate_metrics?.overall_hypertension_pct > 70} />
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <h3 style={{ color: '#7dd3fc', margin: '0 0 12px' }}>Disease Classes</h3>
              {overview.disease_classes?.map((dc, i) => {
                const gene = dc.split(' — ')[0].trim();
                return (
                  <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #334155', fontSize: 13 }}>
                    <GeneChip gene={gene} />
                    <span style={{ marginLeft: 8, color: '#cbd5e1' }}>{dc.split(' — ').slice(1).join(' — ')}</span>
                  </div>
                );
              })}
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <h3 style={{ color: '#7dd3fc', margin: '0 0 12px' }}>Gene Summary Table</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#1e3a5f' }}>
                      {['Gene', 'Locus', 'Avg GFR', 'Avg ESRD Age', 'Hypertension %', 'Haematuria %'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#7dd3fc', borderBottom: '1px solid #334155' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_summaries?.map((gs, i) => (
                      <tr key={gs.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#1a2540' }}>
                        <td style={{ padding: '6px 10px' }}><GeneChip gene={gs.gene} /></td>
                        <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{gs.locus}</td>
                        <td style={{ padding: '6px 10px', color: gs.avg_gfr < 30 ? '#f87171' : '#38bdf8' }}>{gs.avg_gfr}</td>
                        <td style={{ padding: '6px 10px', color: gs.avg_esrd_age < 30 ? '#f87171' : '#e2e8f0' }}>{gs.avg_esrd_age}yr</td>
                        <td style={{ padding: '6px 10px' }}>{gs.hypertension_pct}%</td>
                        <td style={{ padding: '6px 10px' }}>{gs.hematuria_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#7dd3fc', margin: '0 0 12px' }}>Clinical Pearls</h3>
              {overview.clinical_pearls?.map((pearl, i) => (
                <div key={i} style={{ padding: '8px 0', borderBottom: '1px solid #334155', fontSize: 13, color: '#cbd5e1', lineHeight: 1.5 }}>
                  <span style={{ color: '#fbbf24', marginRight: 8 }}>▸</span>{pearl}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1e3a5f' }}>
                  {['Gene', 'Locus', 'Inheritance', 'Avg GFR', 'ESRD Age', 'HTN %', 'Haematuria %', 'Key Metric'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#7dd3fc', borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.gene_breakdowns?.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#1a2540', cursor: 'pointer' }}
                    onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}>
                    <td style={{ padding: '8px 10px', fontWeight: 700 }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{g.inheritance?.split(';')[0]?.trim()?.slice(0, 30)}</td>
                    <td style={{ padding: '8px 10px', color: g.avg_gfr < 30 ? '#f87171' : '#38bdf8' }}>{g.avg_gfr}</td>
                    <td style={{ padding: '8px 10px', color: g.avg_esrd_age < 30 ? '#f87171' : '#e2e8f0' }}>{g.avg_esrd_age}yr</td>
                    <td style={{ padding: '8px 10px' }}>{g.hypertension_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{g.hematuria_pct}%</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#86efac' }}>
                      {g.gene === 'PKD1' ? `TKV ${g.avg_tkv_ml}ml · ICA ${g.ica_pct}%` :
                       g.gene === 'PKD2' ? `TKV ${g.avg_tkv_ml}ml · ICA ${g.ica_pct}%` :
                       g.gene === 'PKHD1' ? `Portal HTN ${g.portal_htn_pct}%` :
                       g.gene === 'COL4A5' ? `SNHL ${g.snhl_pct}% · Lenticonus ${g.anterior_lenticonus_pct}%` :
                       g.gene === 'COL4A3' ? `AR Alport ${g.ar_alport_pct}% · SNHL ${g.snhl_pct}%` :
                       g.gene === 'UMOD' ? `Gout ${g.gout_pct}% · FEUA ${g.avg_feua_pct}%` :
                       g.gene === 'HNF1B' ? `DM ${g.diabetes_pct}% · Pancreas ${g.pancreas_atrophy_pct}%` :
                       g.gene === 'NPHS2' ? `FSGS ${g.fsgs_biopsy_pct}% · Proteinuria ${g.avg_proteinuria_g}g` :
                       '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {selGene && breakdown.gene_breakdowns?.filter(g => g.gene === selGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginTop: 16 }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || '#38bdf8', margin: '0 0 10px' }}>
                  {GENE_INFO[g.gene]?.full || g.gene}
                </h3>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
                  <strong style={{ color: '#e2e8f0' }}>Locus:</strong> {g.locus} &nbsp;|&nbsp;
                  <strong style={{ color: '#e2e8f0' }}>Size:</strong> {g.protein_size} &nbsp;|&nbsp;
                  <strong style={{ color: '#e2e8f0' }}>Inheritance:</strong> {g.inheritance?.split(';')[0]}
                </div>
                <div style={{ fontSize: 12, color: '#fbbf24', marginBottom: 8 }}>
                  <strong>Pathognomonic:</strong> {g.pathognomonic?.split(';')[0]}
                </div>
                <div style={{ fontSize: 12, color: '#86efac', marginBottom: 8 }}>
                  <strong>Treatment:</strong> {g.treatment?.split(';')[0]}
                </div>
                <div style={{ marginTop: 10 }}>
                  <strong style={{ fontSize: 12, color: '#e2e8f0' }}>Key Features:</strong>
                  <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                    {g.key_features?.map((f, i) => (
                      <li key={i} style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 4 }}>{f}</li>
                    ))}
                  </ul>
                </div>
                <div style={{ marginTop: 10 }}>
                  <strong style={{ fontSize: 12, color: '#e2e8f0' }}>Key DDx:</strong>
                  <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                    {g.key_ddx?.map((d, i) => (
                      <li key={i} style={{ fontSize: 11, color: '#f87171', marginBottom: 4 }}>{d}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── CLINICAL ATLAS ── */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
            {breakdown.gene_breakdowns?.map(g => (
              <div key={g.gene} style={{ background: card, borderRadius: 8, padding: 16, borderTop: `3px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <GeneChip gene={g.gene} />
                  <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.inheritance?.split(';')[0]?.trim()?.slice(0, 25)}</span>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>{g.disease_category?.split(';')[0]}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: g.avg_gfr < 30 ? '#f87171' : '#38bdf8' }}>{g.avg_gfr}</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Avg GFR</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: g.avg_esrd_age < 30 ? '#f87171' : '#e2e8f0' }}>{g.avg_esrd_age}yr</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Avg ESRD Age</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#fbbf24' }}>{g.hypertension_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Hypertension</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#86efac' }}>{g.hematuria_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Haematuria</div>
                  </div>
                </div>
                <div style={{ fontSize: 11, color: '#fbbf24', background: '#1e1a00', borderRadius: 4, padding: '4px 8px', marginBottom: 6 }}>
                  ⚠ {g.pathognomonic?.split(';')[0]?.trim()?.slice(0, 130)}
                </div>
                <div style={{ fontSize: 11, color: '#86efac' }}>
                  Rx: {g.treatment?.split(';')[0]?.trim()?.slice(0, 110)}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12, marginBottom: 20 }}>
              {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: 14, borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#38bdf8', marginBottom: 4 }}>{gene}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</div>
                  <div style={{ fontSize: 11, color: '#94a3b8' }}>{entry.disease_name?.split(';')[0]}</div>
                  {gene === 'COL4A5' && (
                    <div style={{ fontSize: 11, color: '#ef4444', fontWeight: 700, marginTop: 4 }}>
                      ⚠ Anterior lenticonus PATHOGNOMONIC — slit-lamp MANDATORY
                    </div>
                  )}
                  {gene === 'NPHS2' && (
                    <div style={{ fontSize: 11, color: '#ef4444', fontWeight: 700, marginTop: 4 }}>
                      ⚠ STEROIDS DO NOT WORK — genetic testing BEFORE immunosuppression
                    </div>
                  )}
                  {gene === 'PKHD1' && (
                    <div style={{ fontSize: 11, color: '#a78bfa', fontWeight: 700, marginTop: 4 }}>
                      ⚠ Null/Null → Potter sequence; liver SYNTHESIS normal despite portal HTN
                    </div>
                  )}
                  {gene === 'UMOD' && (
                    <div style={{ fontSize: 11, color: '#86efac', fontWeight: 700, marginTop: 4 }}>
                      Young gout + FEUA &lt;6% + family CKD = UMOD triad; transplant CURATIVE
                    </div>
                  )}
                  {gene === 'HNF1B' && (
                    <div style={{ fontSize: 11, color: '#fbbf24', fontWeight: 700, marginTop: 4 }}>
                      17q12 deletion → chromosomal microarray (not sequencing alone)
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#7dd3fc', margin: '0 0 12px' }}>Kidney Disease Glossary</h3>
              {Object.entries(definitions.kidney_disease_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 16, borderBottom: '1px solid #334155', paddingBottom: 12 }}>
                  <div style={{ fontWeight: 700, color: '#7dd3fc', marginBottom: 6, fontSize: 13 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
