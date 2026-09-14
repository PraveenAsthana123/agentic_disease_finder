'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cardiomyopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MYH7':   '#b71c1c',  // deep red — HCM #1, dominant negative, Arg403Gln malignant
  'MYBPC3': '#1565c0',  // deep blue — HCM #1 (tied), haploinsufficiency, most common
  'TNNT2':  '#e65100',  // burnt orange — malignant SCD without hypertrophy
  'TNNI3':  '#4a148c',  // deep purple — AR RCM infantile, most severe
  'TPM1':   '#1b5e20',  // dark green — same gene HCM or DCM, calcium sensitisation
  'ACTC1':  '#006064',  // dark teal — apical HCM, giant negative T-waves
  'MYL2':   '#f57f17',  // amber — mid-ventricular obstruction, AR neonatal lethal
  'MYL3':   '#4e342e',  // dark brown — mid-cavity, Asp94Ala Middle East founder
};

const GENE_INFO = {
  'MYH7':   { full: 'MYH7 / Beta-Myosin Heavy Chain / 1935aa', locus: '14q11.2', size: '1935 aa / 223 kDa', inh: 'AD dominant negative', disease: 'HCM #1 (~35%); dominant negative poison polypeptide; Arg403Gln MALIGNANT; Mavacamten FDA 2022; full penetrance; earlier onset than MYBPC3; septal hypertrophy + SAM + LVOTO' },
  'MYBPC3': { full: 'MYBPC3 / Myosin Binding Protein C3 / 1274aa', locus: '11p11.2', size: '1274 aa / 150 kDa', inh: 'AD haploinsufficiency', disease: 'HCM #1 (~35%); haploinsufficiency; INCOMPLETE PENETRANCE 40-50% by age 50; South Asian c.3736+1G>A founder 1-in-500; Mavacamten; rescreen every 3-5 years' },
  'TNNT2':  { full: 'TNNT2 / Cardiac Troponin T2 / 298aa', locus: '1q32.1', size: '298 aa / 36 kDa', inh: 'AD thin filament', disease: 'HCM MALIGNANT (~5%); HIGH SCD despite MINIMAL hypertrophy; CMR LGE mandatory; ICD threshold LOWER than standard; family SCD = immediate high-risk' },
  'TNNI3':  { full: 'TNNI3 / Cardiac Troponin I3 / 210aa', locus: '19q13.42', size: '210 aa / 24 kDa', inh: 'AD/AR', disease: 'HCM AD (~5%); AR biallelic → RCM INFANTILE most severe; DDx RCM vs constrictive pericarditis — mandatory RHC; biatrial enlargement; transplant often needed' },
  'TPM1':   { full: 'TPM1 / Tropomyosin Alpha-1 Chain / 284aa', locus: '15q22.2', size: '284 aa / 33 kDa', inh: 'AD dual phenotype', disease: 'HCM (~2%) OR DCM — SAME GENE, variant class determines phenotype; calcium sensitisation mechanism; LVNC associated; VUS needs functional assay' },
  'ACTC1':  { full: 'ACTC1 / Alpha Cardiac Actin / 375aa', locus: '15q14', size: '375 aa / 42 kDa', inh: 'AD thin filament', disease: 'HCM APICAL variant PROMINENT (~1%); GIANT NEGATIVE T-WAVES V4-V6 ≥10mm PATHOGNOMONIC; LVOTO RARE; spade-shaped LV cavity; apical aneurysm thrombus risk' },
  'MYL2':   { full: 'MYL2 / Myosin Regulatory Light Chain 2 / 166aa', locus: '12q24.11', size: '166 aa / 19 kDa', inh: 'AD/AR thick filament', disease: 'HCM (~2%); MID-VENTRICULAR OBSTRUCTION more common than MYH7/MYBPC3; AR biallelic → NEONATAL LETHAL HCM + skeletal myopathy; extended myectomy needed' },
  'MYL3':   { full: 'MYL3 / Myosin Essential Light Chain 3 / 195aa', locus: '3p21.31', size: '195 aa / 22 kDa', inh: 'AD/AR thick filament', disease: 'HCM (~1%); mid-cavity obstruction; Asp94Ala FOUNDER in Middle Eastern/consanguineous populations; AR biallelic → early-onset severe HCM; consanguinity clue' },
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
  const bg = '#1e293b';
  const border = warn ? '#ef4444' : '#334155';
  return (
    <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryCardiomyopathyAtlasPage() {
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
    setLoading(true);
    setErr(null);
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
  const accent = '#b71c1c';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: card, borderRadius: 12, padding: '20px 24px', marginBottom: 20, borderLeft: `4px solid ${accent}` }}>
          <h1 style={{ margin: 0, fontSize: 20, color: '#f87171' }}>&#x2764;&#xfe0f; Hereditary Cardiomyopathy Atlas</h1>
          <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: 13 }}>
            Complete 8-Gene Reference — MYH7 · MYBPC3 · TNNT2 · TNNI3 · TPM1 · ACTC1 · MYL2 · MYL3 &nbsp;|&nbsp; 320 patients · Seeds 2590–2597
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
              <MetricCard label="Genes" value={overview.n_genes} sub="Cardiomyopathy" />
              <MetricCard label="Seeds" value={overview.seeds} />
              <MetricCard label="Avg Septal (mm)" value={overview.aggregate_metrics?.avg_septal_thickness_mm ?? '—'} sub="steady state" />
              <MetricCard label="Avg EF %" value={`${overview.aggregate_metrics?.avg_ef_percent ?? '—'}%`} />
              <MetricCard label="ICD Implanted" value={`${overview.aggregate_metrics?.icd_pct ?? '—'}%`} warn={overview.aggregate_metrics?.icd_pct > 25} />
              <MetricCard label="AF Prevalence" value={`${overview.aggregate_metrics?.af_pct ?? '—'}%`} />
              <MetricCard label="Family SCD Hx" value={`${overview.aggregate_metrics?.family_scd_pct ?? '—'}%`} warn />
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <h3 style={{ color: '#f87171', margin: '0 0 12px' }}>Disease Classes</h3>
              {overview.disease_classes?.map((dc, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #334155', fontSize: 13 }}>
                  <GeneChip gene={dc.split(' — ')[0].trim()} />
                  <span style={{ marginLeft: 8, color: '#cbd5e1' }}>{dc.split(' — ').slice(1).join(' — ')}</span>
                </div>
              ))}
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#f87171', margin: '0 0 12px' }}>Clinical Pearls</h3>
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
                  {['Gene', 'Locus', 'Inheritance', 'Avg Septal mm', 'Avg EF %', 'SCD Risk 5yr', 'ICD %', 'AF %', 'Mavacamten %'].map(h => (
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
                    <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11 }}>{g.inheritance?.split(';')[0]}</td>
                    <td style={{ padding: '8px 10px', color: '#38bdf8' }}>{g.avg_septal_thickness_mm}</td>
                    <td style={{ padding: '8px 10px', color: '#34d399' }}>{g.avg_ef_percent}%</td>
                    <td style={{ padding: '8px 10px', color: g.avg_scd_risk_5yr > 6 ? '#f87171' : '#e2e8f0' }}>{g.avg_scd_risk_5yr}%</td>
                    <td style={{ padding: '8px 10px', color: g.icd_pct > 30 ? '#f87171' : '#e2e8f0' }}>{g.icd_pct}%</td>
                    <td style={{ padding: '8px 10px' }}>{g.af_pct}%</td>
                    <td style={{ padding: '8px 10px', color: '#86efac' }}>{g.mavacamten_eligible_pct}%</td>
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
                  <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.inheritance?.split(';')[0]?.trim()}</span>
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>{g.disease_category?.split(';')[0]}</div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#38bdf8' }}>{g.avg_septal_thickness_mm}</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Avg Septal mm</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#34d399' }}>{g.avg_ef_percent}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Avg EF %</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: g.icd_pct > 30 ? '#f87171' : '#e2e8f0' }}>{g.icd_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>ICD Implanted</div>
                  </div>
                  <div style={{ background: '#0f172a', borderRadius: 4, padding: 6, textAlign: 'center' }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: '#fbbf24' }}>{g.mavacamten_eligible_pct}%</div>
                    <div style={{ fontSize: 10, color: '#64748b' }}>Mavacamten Elig.</div>
                  </div>
                </div>
                <div style={{ fontSize: 11, color: '#fbbf24', background: '#1e1a00', borderRadius: 4, padding: '4px 8px', marginBottom: 6 }}>
                  ⚠ {g.pathognomonic?.split(';')[0]?.trim()?.slice(0, 120)}
                </div>
                <div style={{ fontSize: 11, color: '#86efac' }}>
                  Rx: {g.treatment?.split(';')[0]?.trim()?.slice(0, 100)}
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
                  <div style={{ fontSize: 11, color: '#fbbf24', marginTop: 6 }}>
                    HCM: {entry.hcm_pct}% · LVOTO: {entry.lvoto_pct}% · SCD 5yr: {entry.scd_risk_pct}% · ICD: {entry.icd_pct}%
                  </div>
                  {gene === 'TNNT2' && (
                    <div style={{ fontSize: 11, color: '#ef4444', fontWeight: 700, marginTop: 4 }}>
                      ⚠ MALIGNANT — SCD without hypertrophy; ICD LOWER threshold
                    </div>
                  )}
                  {gene === 'TNNI3' && (
                    <div style={{ fontSize: 11, color: '#ef4444', fontWeight: 700, marginTop: 4 }}>
                      ⚠ AR biallelic → RCM INFANTILE most severe
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <h3 style={{ color: '#f87171', margin: '0 0 12px' }}>Cardiomyopathy Glossary</h3>
              {Object.entries(definitions.cardiomyopathy_glossary || {}).map(([term, def]) => (
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
