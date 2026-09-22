'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-colorectal-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'POLE':   '#1a237e',  // deep navy     -- PPAP-1; TMB>100 MSS PATHOGNOMONIC; exceptional pembrolizumab
  'POLD1':  '#4a148c',  // deep purple   -- PPAP-2; sebaceous PATHOGNOMONIC; brain tumours; CRC 80%
  'EPCAM':  '#006064',  // deep teal     -- Lynch-by-MSH2-silencing; 3'-deletion only; MLPA mandatory
  'NTHL1':  '#b71c1c',  // deep red      -- NAP; SBS30 CpG>TpG PATHOGNOMONIC; CMMRD NOT expected
  'RNF43':  '#e65100',  // deep orange   -- SFPN; sessile serrated lesions PATHOGNOMONIC; BRAF-V600E
  'BMPR1A': '#2e7d32',  // dark green    -- JPS type 1; juvenile polyps PATHOGNOMONIC; gastric 21%; HHT absent
  'MSH3':   '#880e4f',  // deep pink     -- PPAP-like biallelic; EMAST PATHOGNOMONIC; CRC near 100%
  'GREM1':  '#37474f',  // dark blue-grey -- HMPS; mixed polyps PATHOGNOMONIC; Ashkenazi founder; MLPA/CNV mandatory
};

const GENE_INFO = {
  'POLE':   { full: 'PPAP-1 / Ultra-Hypermutated-TMB-GT100-MSS-NOT-MSI-H-PATHOGNOMONIC / SBS10a-SBS10b-COSMIC / Exceptional-Pembrolizumab-Complete-Remissions / L424V-P286R-Hotspots / CRC-60-80% / Endometrial-30-40%',     locus: '12q24.33', size: '2286 aa / 261 kDa', inh: 'AD GOF' },
  'POLD1':  { full: 'PPAP-2 / Sebaceous-Gland-Tumours-PATHOGNOMONIC-POLD1-vs-POLE / Brain-Tumours-Elevated / CRC-80%-HIGHEST-PPAP-2 / L474P-Hotspot / Ceralasertib-Pembrolizumab',                                           locus: '19q13.33', size: '1107 aa / 125 kDa', inh: 'AD GOF' },
  'EPCAM':  { full: 'Lynch-by-MSH2-Silencing / 3prime-Deletion-ONLY-Mechanism / MLPA-MANDATORY-Sanger-MISSES / MSH2-IHC-Loss-with-MSH2-Coding-Normal / Small-Bowel-PATHOGNOMONIC-Highest-Lynch / CRC-50-80%',               locus: '2p21',     size: '314 aa / 35 kDa',   inh: 'AD LOF 3\' del' },
  'NTHL1':  { full: 'NAP-NTHL1-Associated-Polyposis / SBS30-CpG-TpG-PATHOGNOMONIC / CRC-Near-100%-Biallelic / Q90X-Founder / CMMRD-NOT-Expected-Unlike-MMR / Pembrolizumab-MSI-H / Base-Excision-Repair-NTHL1-BER',         locus: '16p13.3',  size: '312 aa / 35 kDa',   inh: 'AR biallelic' },
  'RNF43':  { full: 'SFPN-Serrated-Familial-Polyposis / Sessile-Serrated-Lesions-PATHOGNOMONIC / RSPO-Fusion-Sporadic-NOT-Germline / BRAF-V600E-Serrated-Pathway / LGK-974-Porcupine-Inhibitor / MSI-H-Subset / G659Vfs41', locus: '17q22',    size: '783 aa / 88 kDa',   inh: 'AD LOF' },
  'BMPR1A': { full: 'JPS-Type-1 / Juvenile-Polyps-Mucus-Filled-Cystic-Hamartomatous-PATHOGNOMONIC / HHT-ABSENT-BMPR1A-SMAD4-Only / Gastric-Cancer-21%-HIGHEST-JPS / CRC-39% / PTEN-10q22-Contiguous-Deletion / Rapamycin',   locus: '10q22.3',  size: '532 aa / 60 kDa',   inh: 'AD LOF' },
  'MSH3':   { full: 'PPAP-Like-Biallelic / EMAST-Tetranucleotide-PATHOGNOMONIC / MSH3-IHC-Lost-MSH2-Retained / CRC-Near-100%-Biallelic / CMMRD-Risk-LOWER-Than-MMR / Monoallelic-NOT-Lynch-Carrier / Pembrolizumab-MSI-H',   locus: '5q11.2',   size: '1128 aa / 128 kDa', inh: 'AR biallelic' },
  'GREM1':  { full: 'HMPS-Hereditary-Mixed-Polyposis-Syndrome / Mixed-Polyps-Adenoma-Hyperplastic-Serrated-Juvenile-PATHOGNOMONIC / Exclusively-Ashkenazi-Jewish-Founder / MLPA-CNV-Mandatory-Sanger-Misses-Duplication / CRC-50-80% / NO-BRAF-V600E', locus: '15q13.3', size: '184 aa / 21 kDa',  inh: 'AD GOF 3\' dup' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#1565c0', color: '#fff', borderRadius: 4,
      padding: '2px 8px', fontSize: 11, fontWeight: 700, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneBar({ gene, pct, color }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color }}>{gene}</span>
        <span style={{ color: '#333' }}>{pct}%</span>
      </div>
      <div style={{ background: '#e0e0e0', borderRadius: 4, height: 14 }}>
        <div style={{ background: color, width: `${pct}%`, height: '100%', borderRadius: 4, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  );
}

export default function HereditaryColorectalCancerAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#1a237e' }}>Loading Hereditary Colorectal Cancer Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f9f9f9', minHeight: '100vh', padding: 0 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#0d47a1 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, textTransform: 'uppercase' }}>Hereditary Cancer Predisposition Atlas</div>
        <h1 style={{ margin: '6px 0 4px', fontSize: 26, fontWeight: 800 }}>Hereditary Colorectal Cancer Predisposition Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.85 }}>
          Complete 8-Gene Reference — POLE · POLD1 · EPCAM · NTHL1 · RNF43 · BMPR1A · MSH3 · GREM1
        </div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>
          320-patient aggregate cohort (8 × 40 · seeds 3230–3237) · PPAP-1/2 · Lynch-by-MSH2-Silencing · NAP · SFPN · JPS · EMAST · HMPS
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '2px solid #1a237e', display: 'flex', gap: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '10px 22px', border: 'none', background: tab === t ? '#1a237e' : 'transparent',
            color: tab === t ? '#fff' : '#1a237e', fontWeight: tab === t ? 700 : 400,
            cursor: 'pointer', fontSize: 14, borderBottom: tab === t ? '2px solid #1a237e' : 'none',
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
              {[
                { label: 'Total Patients', value: overview.total_patients, color: '#1a237e' },
                { label: 'Genes Covered', value: overview.genes_n, color: '#4a148c' },
                { label: 'Highest Risk', value: `${overview.highest_risk_gene} ${overview.highest_risk_pct}%`, color: '#b71c1c' },
                { label: 'Severe Overall', value: `${overview.severe_total_pct}%`, color: '#880e4f' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: '#fff', borderRadius: 8, padding: '16px 20px', boxShadow: '0 1px 4px rgba(0,0,0,.08)', borderLeft: `4px solid ${color}` }}>
                  <div style={{ fontSize: 11, color: '#666', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color, marginTop: 4 }}>{value}</div>
                </div>
              ))}
            </div>

            {/* Severe % bars */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)', marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px', color: '#1a237e', fontSize: 15 }}>Severe Clinical Features by Gene (%)</h3>
              {overview.gene_summary?.map(g => (
                <GeneBar key={g.gene} gene={g.gene} pct={g.severe_pct} color={GENE_COLORS[g.gene] || '#1a237e'} />
              ))}
            </div>

            {/* Gene pathognomonic badges */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)' }}>
              <h3 style={{ margin: '0 0 12px', color: '#1a237e', fontSize: 15 }}>Key Pathognomonic / Actionable Features</h3>
              {overview.gene_summary?.map(g => (
                <div key={g.gene} style={{ marginBottom: 10, borderBottom: '1px solid #f0f0f0', paddingBottom: 10 }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#1a237e', marginRight: 8 }}>{g.gene}</span>
                  <span style={{ fontSize: 12, color: '#444' }}>{g.pathognomonic}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fff', borderRadius: 8, boxShadow: '0 1px 4px rgba(0,0,0,.08)', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  {['Gene', 'Locus', 'Size', 'Inheritance', 'Clinical Identity'].map(h => (
                    <th key={h} style={{ padding: '10px 14px', textAlign: 'left', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(GENE_INFO).map(([gene, info], i) => (
                  <tr key={gene} style={{ background: i % 2 ? '#f8f8f8' : '#fff', borderBottom: '1px solid #e8e8e8' }}>
                    <td style={{ padding: '10px 14px', fontWeight: 800, color: GENE_COLORS[gene] }}>{gene}</td>
                    <td style={{ padding: '10px 14px', fontFamily: 'monospace', color: '#333' }}>{info.locus}</td>
                    <td style={{ padding: '10px 14px', color: '#555' }}>{info.size}</td>
                    <td style={{ padding: '10px 14px' }}>
                      <Badge text={info.inh} color={GENE_COLORS[gene]} />
                    </td>
                    <td style={{ padding: '10px 14px', color: '#444', fontSize: 12 }}>{info.full}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ marginBottom: 16, fontSize: 13, color: '#555' }}>
              320-patient aggregate cohort (8 × 40, seeds 3230–3237). Severe = pathognomonic feature / ultra-hypermutated / polyposis + CRC / biallelic LOF.
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              {breakdown.breakdown?.map(g => (
                <div key={g.gene} style={{
                  background: '#fff', borderRadius: 8, padding: 18, boxShadow: '0 1px 4px rgba(0,0,0,.08)',
                  borderTop: `4px solid ${GENE_COLORS[g.gene] || '#1a237e'}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                    <span style={{ fontSize: 11, color: '#888' }}>n={g.n}</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10, fontSize: 13 }}>
                    <div><span style={{ color: '#666' }}>Mean age onset: </span><strong>{g.mean_age_onset} yr</strong></div>
                    <div><span style={{ color: '#666' }}>Severe: </span><strong style={{ color: '#b71c1c' }}>{g.severe_n}/40 ({g.severe_pct}%)</strong></div>
                  </div>
                  <div style={{ fontSize: 12, color: '#444' }}>
                    {g.key_features?.slice(0, 5).map(f => (
                      <div key={f} style={{ marginBottom: 3 }}>• {f}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {definitions.definitions?.map((d, i) => (
              <div key={i} style={{
                background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)',
                marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[d.term?.split('/')[0]?.trim()] || '#1a237e'}`
              }}>
                <div style={{ fontWeight: 700, color: '#1a237e', fontSize: 13, marginBottom: 8 }}>{d.term}</div>
                <pre style={{ fontFamily: 'system-ui,sans-serif', fontSize: 12, color: '#333', whiteSpace: 'pre-wrap', margin: 0, lineHeight: 1.6 }}>
                  {d.definition}
                </pre>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
