'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-surfactant-dysfunction-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'SFTPB':  '#b71c1c',  // deep red — fatal neonatal
  'SFTPC':  '#e65100',  // burnt orange — AD ILD
  'ABCA3':  '#1565c0',  // deep blue — most common
  'NKX2-1': '#4a148c',  // deep purple — triad
  'MARS1':  '#1b5e20',  // dark green — liver+lung
  'SLC34A2':'#795548',  // brown — microlithiasis
  'CSF2RA': '#0277bd',  // medium blue — PAP type 4
  'CSF2RB': '#006064',  // teal — PAP type 5
};

const GENE_INFO = {
  'SFTPB':  { full: 'SFTPB / Surfactant Protein B / 279aa', locus: '2p11.2', size: '279 aa / 8 kDa (mature)', inh: 'AR', disease: 'Fatal neonatal RDS — absent lamellar bodies on EM (pathognomonic); c.121ins2 >70% N. American alleles; lung transplant ONLY curative' },
  'SFTPC':  { full: 'SFTPC / Surfactant Protein C / 197aa', locus: '8p21.3', size: '197 aa / 4 kDa (mature)', inh: 'AD', disease: 'ILD childhood-adult — de novo 40%; p.Ile73Thr BRICHOS misfolding; HCQ first-line children; nintedanib/pirfenidone adults' },
  'ABCA3':  { full: 'ABCA3 / ATP-Binding Cassette A3 / 1704aa', locus: '16p13.3', size: '1704 aa / 191 kDa', inh: 'AR', disease: 'Most common genetic chILD — small DENSE electron-opaque LBs on EM (pathognomonic); biallelic null = neonatal fatal; genotype predicts severity' },
  'NKX2-1': { full: 'NKX2-1 / TTF-1 / 371aa', locus: '14q13.3', size: '371 aa / 42 kDa', inh: 'AD', disease: 'Brain-Thyroid-Lung triad — chorea + hypothyroidism + ILD; NKX2-1 IHC absent on lung biopsy; de novo 60%' },
  'MARS1':  { full: 'MARS1 / Methionyl-tRNA Synthetase 1 / 900aa', locus: '12q13.3', size: '900 aa / 101 kDa', inh: 'AR', disease: 'Interstitial Lung AND Liver Disease (ILLD) — UNIQUE combined phenotype; p.Arg792His most common; liver Tx may stabilise lung' },
  'SLC34A2':{ full: 'SLC34A2 / NaPi-IIb / 689aa', locus: '4p15.31', size: '689 aa / 74 kDa', inh: 'AR', disease: 'Pulmonary Alveolar Microlithiasis (PAM) — snowstorm CXR pathognomonic; serum Ca/PO4 NORMAL; no medical treatment; lung Tx end-stage' },
  'CSF2RA': { full: 'CSF2RA / GM-CSFR-Alpha / 400aa', locus: 'Xp22.33 (PAR1)', size: '400 aa / 45 kDa', inh: 'PAR/XL', disease: 'Hereditary PAP type 4 — anti-GM-CSF Ab ABSENT; WLL effective; inhaled GM-CSF variably effective; HSCT curative severe' },
  'CSF2RB': { full: 'CSF2RB / GM-CSFR-Beta / 897aa', locus: '22q12.3', size: '897 aa / 97 kDa', inh: 'AR', disease: 'Hereditary PAP type 5 — inhaled GM-CSF INEFFECTIVE (beta chain missing); HSCT CURATIVE; low eosinophils (IL-5 also blocked)' },
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

export default function HereditorySurfactantDysfunctionAtlasPage() {
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
  const accent = '#1565c0';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: card, borderRadius: 12, padding: '20px 24px', marginBottom: 20, borderLeft: `4px solid ${accent}` }}>
          <h1 style={{ margin: 0, fontSize: 20, color: '#38bdf8' }}>&#x1f9ec; Hereditary Surfactant Dysfunction / chILD Atlas</h1>
          <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: 13 }}>
            Complete 8-Gene Reference — SFTPB · SFTPC · ABCA3 · NKX2-1 · MARS1 · SLC34A2 · CSF2RA · CSF2RB &nbsp;|&nbsp; 320 patients · Seeds 2574–2581
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

        {loading && <div style={{ color: '#94a3b8', padding: 40, textAlign: 'center' }}>Loading…</div>}
        {err && <div style={{ color: '#ef4444', padding: 20 }}>Error: {err}</div>}

        {/* Overview Tab */}
        {!loading && !err && tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
              <MetricCard label="Genes" value={overview.n_genes} sub="surfactant/chILD" />
              <MetricCard label="Patients" value={overview.total_patients} sub="8×40 seeds" />
              <MetricCard label="Resp. Failure" value={`${overview.aggregate_metrics?.respiratory_failure_pct}%`} warn sub="neonatal/childhood" />
              <MetricCard label="Lung Tx" value={`${overview.aggregate_metrics?.lung_tx_pct}%`} sub="transplanted" />
              <MetricCard label="ILD" value={`${overview.aggregate_metrics?.ild_pct}%`} sub="interstitial lung disease" />
              <MetricCard label="PAP" value={`${overview.aggregate_metrics?.pap_pct}%`} sub="alveolar proteinosis" />
              <MetricCard label="Avg FEV1%" value={`${overview.aggregate_metrics?.avg_fev1_pct}%`} sub="spirometry" />
              <MetricCard label="Avg DLCO%" value={`${overview.aggregate_metrics?.avg_dlco_pct}%`} sub="diffusing capacity" />
            </div>

            <h3 style={{ color: '#38bdf8', marginBottom: 12 }}>Disease Classes</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 24 }}>
              {(overview.disease_classes || []).map((dc, i) => {
                const gene = dc.split(' — ')[0];
                return (
                  <div key={i} style={{ background: card, borderRadius: 6, padding: '8px 14px', borderLeft: `3px solid ${GENE_COLORS[gene] || accent}`, fontSize: 13 }}>
                    {dc}
                  </div>
                );
              })}
            </div>

            <h3 style={{ color: '#38bdf8', marginBottom: 12 }}>Gene Summary</h3>
            <div style={{ overflowX: 'auto', marginBottom: 24 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Size','Inh','Disease','Onset','Resp Fail%','Lung Tx%','ILD%','PAP%'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', color: '#94a3b8', textAlign: 'left', borderBottom: '1px solid #334155' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.gene_summary || []).map((g, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #1e293b', background: i % 2 === 0 ? '#0f172a' : '#111827' }}
                      onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}
                      style={{ cursor: 'pointer', borderBottom: '1px solid #1e293b', background: selGene === g.gene ? '#1e3a5f' : i % 2 === 0 ? '#0f172a' : '#111827' }}>
                      <td style={{ padding: '6px 10px' }}><GeneChip gene={g.gene} /></td>
                      <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '6px 10px', color: '#64748b' }}>{g.protein_size}</td>
                      <td style={{ padding: '6px 10px', color: '#f59e0b' }}>{g.inheritance}</td>
                      <td style={{ padding: '6px 10px', maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{g.disease_name}</td>
                      <td style={{ padding: '6px 10px', color: '#94a3b8', textAlign: 'right' }}>{g.onset_age_years_median}y</td>
                      <td style={{ padding: '6px 10px', color: g.respiratory_failure_pct > 70 ? '#ef4444' : '#f59e0b', textAlign: 'right' }}>{g.respiratory_failure_pct}%</td>
                      <td style={{ padding: '6px 10px', color: '#38bdf8', textAlign: 'right' }}>{g.lung_tx_pct}%</td>
                      <td style={{ padding: '6px 10px', color: '#a78bfa', textAlign: 'right' }}>{g.ild_pct}%</td>
                      <td style={{ padding: '6px 10px', color: '#34d399', textAlign: 'right' }}>{g.pap_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <h3 style={{ color: '#38bdf8', marginBottom: 12 }}>Clinical Pearls</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(overview.clinical_pearls || []).map((p, i) => (
                <div key={i} style={{ background: card, borderRadius: 6, padding: '10px 16px', fontSize: 13, borderLeft: '3px solid #f59e0b' }}>
                  {p}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Gene Table Tab */}
        {!loading && !err && tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
              {Object.keys(GENE_COLORS).map(g => (
                <button key={g} onClick={() => setSelGene(selGene === g ? null : g)}
                  style={{ background: selGene === g ? GENE_COLORS[g] : card, color: selGene === g ? '#fff' : '#94a3b8', border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 20, padding: '4px 14px', cursor: 'pointer', fontSize: 12, fontWeight: selGene === g ? 700 : 400 }}>
                  {g}
                </button>
              ))}
            </div>
            {(breakdown.gene_breakdowns || [])
              .filter(g => !selGene || g.gene === selGene)
              .map((g, i) => (
                <div key={i} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || accent}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                    <GeneChip gene={g.gene} />
                    <span style={{ color: '#94a3b8', fontSize: 13 }}>{g.locus} · {g.protein_size}</span>
                    <span style={{ color: '#f59e0b', fontSize: 12 }}>{g.inheritance.split(';')[0]}</span>
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
                    <MetricCard label="Resp. Failure" value={`${g.respiratory_failure_pct}%`} warn={g.respiratory_failure_pct > 70} />
                    <MetricCard label="Lung Tx" value={`${g.lung_tx_pct}%`} />
                    <MetricCard label="PAP" value={`${g.pap_pct}%`} />
                    <MetricCard label="Microlithiasis" value={`${g.microlithiasis_pct}%`} />
                    <MetricCard label="Avg FEV1%" value={`${g.avg_fev1_pct}%`} />
                    <MetricCard label="Avg DLCO%" value={`${g.avg_dlco_pct}%`} />
                    <MetricCard label="Onset" value={`${Math.round(g.avg_onset_age_months)}mo`} />
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <strong style={{ color: '#f59e0b' }}>Pathognomonic: </strong>
                    <span style={{ fontSize: 13 }}>{g.pathognomonic}</span>
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <strong style={{ color: '#38bdf8' }}>Key Features:</strong>
                    <ul style={{ margin: '6px 0 0 20px', padding: 0 }}>
                      {(g.key_features || []).map((f, j) => <li key={j} style={{ fontSize: 13, marginBottom: 3 }}>{f}</li>)}
                    </ul>
                  </div>
                  <div>
                    <strong style={{ color: '#a78bfa' }}>DDx: </strong>
                    {(g.key_ddx || []).map((d, j) => (
                      <div key={j} style={{ fontSize: 12, color: '#94a3b8', marginLeft: 16, marginTop: 2 }}>• {d}</div>
                    ))}
                  </div>
                </div>
              ))}
          </div>
        )}

        {/* Clinical Atlas Tab */}
        {!loading && !err && tab === 'Clinical Atlas' && breakdown && (
          <div>
            {(breakdown.gene_breakdowns || []).map((g, i) => (
              <div key={i} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 16, borderTop: `3px solid ${GENE_COLORS[g.gene] || accent}` }}>
                <h3 style={{ color: GENE_COLORS[g.gene] || '#38bdf8', margin: '0 0 8px 0', fontSize: 16 }}>
                  {g.gene} — {g.disease_category?.split(';')[0] || ''}
                </h3>
                <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 10 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</div>
                <p style={{ fontSize: 13, lineHeight: 1.6, marginBottom: 10 }}>{g.disease_pathway}</p>
                <div style={{ background: '#0f172a', borderRadius: 6, padding: 12, marginBottom: 10 }}>
                  <strong style={{ color: '#f59e0b' }}>Treatment: </strong>
                  <span style={{ fontSize: 13 }}>{g.treatment}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Definitions Tab */}
        {!loading && !err && tab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
              {Object.keys(GENE_COLORS).map(g => (
                <button key={g} onClick={() => setSelGene(selGene === g ? null : g)}
                  style={{ background: selGene === g ? GENE_COLORS[g] : card, color: selGene === g ? '#fff' : '#94a3b8', border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 20, padding: '4px 14px', cursor: 'pointer', fontSize: 12 }}>
                  {g}
                </button>
              ))}
            </div>

            {Object.entries(definitions.gene_entries || {})
              .filter(([g]) => !selGene || g === selGene)
              .map(([g, info]) => (
                <div key={g} style={{ background: card, borderRadius: 10, padding: 20, marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[g] || accent}` }}>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 10 }}>
                    <GeneChip gene={g} />
                    <span style={{ color: '#94a3b8', fontSize: 13 }}>{info.locus} · {info.protein_size}</span>
                    <span style={{ background: info.nbs_indicated ? '#166534' : '#374151', color: info.nbs_indicated ? '#4ade80' : '#9ca3af', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>
                      NBS: {info.nbs_indicated ? 'YES' : 'no'}
                    </span>
                  </div>
                  <div style={{ fontSize: 13, marginBottom: 8 }}><strong style={{ color: '#f59e0b' }}>Inheritance: </strong>{info.inheritance}</div>
                  <div style={{ fontSize: 13, marginBottom: 8 }}><strong style={{ color: '#38bdf8' }}>Pathognomonic: </strong>{info.pathognomonic}</div>
                  <div style={{ fontSize: 13, marginBottom: 8 }}><strong style={{ color: '#a78bfa' }}>Treatment: </strong>{info.treatment}</div>
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    <MetricCard label="Resp. Failure" value={`${info.respiratory_failure_pct}%`} warn={info.respiratory_failure_pct > 70} />
                    <MetricCard label="Lung Tx" value={`${info.lung_tx_pct}%`} />
                    <MetricCard label="PAP" value={`${info.pap_pct}%`} />
                    <MetricCard label="Microlithiasis" value={`${info.microlithiasis_pct}%`} />
                  </div>
                </div>
              ))}

            <h3 style={{ color: '#38bdf8', marginTop: 24, marginBottom: 12 }}>chILD Glossary</h3>
            {Object.entries(definitions.child_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: card, borderRadius: 8, padding: '12px 16px', marginBottom: 10, borderLeft: '3px solid #334155' }}>
                <div style={{ fontWeight: 700, color: '#f59e0b', marginBottom: 4, fontSize: 13 }}>{term}</div>
                <div style={{ fontSize: 13, color: '#cbd5e1', lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
