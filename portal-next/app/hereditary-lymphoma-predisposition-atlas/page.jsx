'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lymphoma-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':       '#1a237e',  // deep navy     -- CLL 4-7x; biallelic AT radiation ABSOLUTE CI; ceralasertib
  'BRCA2':     '#4a148c',  // deep purple   -- HBOC; NHL 2-3x; HRD cisplatin/olaparib; FA-D1 most severe
  'CARD11':    '#006064',  // deep teal     -- BENTA; constitutive NF-kB; DLBCL 20-30%; ibrutinib
  'PIK3CD':    '#b71c1c',  // deep red      -- APDS1; EBV+ B-NHL PATHOGNOMONIC; leniolisib FDA2023
  'KMT2D':     '#e65100',  // deep orange   -- Kabuki; KMT2D somatic 83% FL; R2 rituximab+lenalidomide
  'TP53':      '#2e7d32',  // dark green    -- LFS; AVOID radiation ABSOLUTELY; WBMRI Toronto
  'TNFRSF13B': '#880e4f',  // deep pink     -- TACI CVID; MALT dominant; rituximab caution; A181E founder
  'LRBA':      '#37474f',  // dark blue-grey -- CVID-like; EBV+ B-NHL PATHOGNOMONIC; abatacept; HSCT curative
};

const GENE_INFO = {
  'ATM':       { full: 'CLL-4-7x-Monoallelic / MCL-3-5x / DLBCL-2-4x / AT-Biallelic-Radiation-ABSOLUTE-CI / AFP-Elevated-95pct-Biallelic-PATHOGNOMONIC / Ceralasertib-ATRi-Olaparib-Clinical-Trials',   locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD/AR LOF' },
  'BRCA2':     { full: 'NHL-2-3x / Hodgkin-3-5x-Emerging / HRD-Cisplatin-Olaparib / FA-D1-Biallelic-Medulloblastoma-PATHOGNOMONIC-lt5yr / Male-Breast-8-9pct-HIGHEST / 6174delT-Ashkenazi-Founder',       locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
  'CARD11':    { full: 'BENTA / B-Cell-Lymphocytosis-Polyclonal-PATHOGNOMONIC / T-Cell-Anergy / DLBCL-20-30pct / MALT / Constitutive-NF-kB-CBM / Ibrutinib-BTKi-Partial / MALT1-Protease-Inhibitor',       locus: '7p22.2',  size: '1154 aa / 130 kDa', inh: 'AD GOF' },
  'PIK3CD':    { full: 'APDS1 / EBV-Plus-B-NHL-PATHOGNOMONIC / Monthly-EBV-PCR-MANDATORY / Leniolisib-FDA2023-PI3Kdelta-Specific / NOT-Idelalisib-Pan-PI3K / Bronchiectasis-Near-Universal / E1021K-60pct', locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD GOF' },
  'KMT2D':     { full: 'Kabuki-Type-1 / KMT2D-Somatic-83pct-FL-MOST-COMMON-DRIVER / Persistent-Fingertip-Pads-PATHOGNOMONIC / R2-Rituximab-Lenalidomide / Tazemetostat-EZH2i / H3K4-Enhancer-Loss',       locus: '12q13.12', size: '5537 aa / 593 kDa', inh: 'AD LOF' },
  'TP53':      { full: 'LFS / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto-Annual / Sarcoma-50-60pct-Dominant / NHL-5-10pct / R337H-Brazilian-Founder-0-3pct / R-CHOP-NOT-XRT / Double-Hit-DLBCL-Poor-Prx',   locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'TNFRSF13B': { full: 'TACI-Deficiency / CVID-10pct-Monogenic / MALT-Dominant-CVID-Lymphoma / B-NHL-3-5x / A181E-C104R-European-Founder / RITUXIMAB-CAUTION / IVIG-Trough-8g-L / GLILD-PATHOGNOMONIC',     locus: '17p11.2', size: '293 aa / 32 kDa',   inh: 'AD/AR LOF' },
  'LRBA':      { full: 'CVID-Like / EBV-Plus-B-NHL-PATHOGNOMONIC / IBD-Crohn-50-60pct / Abatacept-CTLA4-Ig-Mechanism-Targeted / CTLA4-Surface-Deficiency-Diagnostic-Flow / HSCT-Curative / Monthly-EBV-PCR', locus: '4q31.3',  size: '2863 aa / 320 kDa', inh: 'AR LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#1565c0', color: '#fff', borderRadius: 4,
      padding: '2px 8px', fontSize: 11, fontWeight: 700, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneBar({ gene, pct, color, label }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color }}>{gene}{label ? ` — ${label}` : ''}</span>
        <span style={{ color: '#333' }}>{pct}%</span>
      </div>
      <div style={{ background: '#e0e0e0', borderRadius: 4, height: 14 }}>
        <div style={{ background: color, width: `${Math.min(pct, 100)}%`, height: '100%', borderRadius: 4, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  );
}

export default function HereditaryLymphomaAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#1a237e' }}>Loading Hereditary Lymphoma Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f9f9f9', minHeight: '100vh', padding: 0 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#283593 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, textTransform: 'uppercase' }}>Hereditary Cancer Predisposition Atlas</div>
        <h1 style={{ margin: '6px 0 4px', fontSize: 26, fontWeight: 800 }}>Hereditary Lymphoma Predisposition Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.85 }}>
          Complete 8-Gene Reference — ATM · BRCA2 · CARD11 · PIK3CD · KMT2D · TP53 · TNFRSF13B · LRBA
        </div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>
          320-patient aggregate cohort (8 × 40 · seeds 3238–3245) · BENTA · APDS1 · CVID · Kabuki · LFS · FA-D1 · TACI
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '2px solid #1a237e', display: 'flex', gap: 0, flexWrap: 'wrap' }}>
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
                { label: 'Genes Covered', value: overview.genes?.length || 8, color: '#4a148c' },
                { label: 'CR Rate', value: `${overview.cr_pct}%`, color: '#2e7d32' },
                { label: 'EBV+ Lymphoma', value: `${overview.ebv_positive_total} (${overview.ebv_positive_pct}%)`, color: '#b71c1c' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: '#fff', borderRadius: 8, padding: '16px 20px', boxShadow: '0 1px 4px rgba(0,0,0,.08)', borderLeft: `4px solid ${color}` }}>
                  <div style={{ fontSize: 11, color: '#666', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color, marginTop: 4 }}>{value}</div>
                </div>
              ))}
            </div>

            {/* CR % bars per gene */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)', marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px', color: '#1a237e', fontSize: 15 }}>Complete Remission Rate by Gene (%)</h3>
              {overview.gene_summaries?.map(g => (
                <GeneBar key={g.gene} gene={g.gene} pct={g.cr_pct} color={GENE_COLORS[g.gene] || '#1a237e'} />
              ))}
            </div>

            {/* Clinical pearls */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)', marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px', color: '#1a237e', fontSize: 15 }}>Key Clinical Pearls</h3>
              {overview.clinical_pearls?.map((p, i) => (
                <div key={i} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px solid #f0f0f0', fontSize: 13, color: '#333' }}>
                  <span style={{ color: '#1a237e', fontWeight: 700, marginRight: 6 }}>{i + 1}.</span>{p}
                </div>
              ))}
            </div>

            {/* Key management rules */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)' }}>
              <h3 style={{ margin: '0 0 12px', color: '#b71c1c', fontSize: 15 }}>Key Management Rules</h3>
              {overview.key_management_rules?.map((r, i) => (
                <div key={i} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: '1px solid #f0f0f0', fontSize: 13 }}>
                  <Badge text={r.split(':')[0]} color={['#1a237e','#4a148c','#006064','#b71c1c','#e65100','#2e7d32'][i % 6]} />
                  <span style={{ color: '#444' }}>{r.split(':').slice(1).join(':')}</span>
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
              320-patient aggregate cohort (8 × 40, seeds 3238–3245). CR = complete remission. EBV+ = EBV-positive lymphoma subtype.
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16 }}>
              {breakdown.breakdown?.map(g => (
                <div key={g.gene} style={{
                  background: '#fff', borderRadius: 8, padding: 18, boxShadow: '0 1px 4px rgba(0,0,0,.08)',
                  borderTop: `4px solid ${GENE_COLORS[g.gene] || '#1a237e'}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[g.gene] }}>{g.gene}</span>
                    <span style={{ fontSize: 11, color: '#888' }}>n={g.n_patients} · {g.locus}</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10, fontSize: 12 }}>
                    <div><span style={{ color: '#666' }}>Mean age dx: </span><strong>{g.mean_age_dx} yr</strong></div>
                    <div><span style={{ color: '#666' }}>CR: </span><strong style={{ color: '#2e7d32' }}>{g.cr_pct}%</strong></div>
                    <div><span style={{ color: '#666' }}>Relapse: </span><strong style={{ color: '#b71c1c' }}>{g.relapse_pct}%</strong></div>
                    <div><span style={{ color: '#666' }}>EBV+: </span><strong>{g.ebv_positive_n}</strong></div>
                  </div>
                  <div style={{ fontSize: 11, color: '#444', marginBottom: 8 }}>
                    <strong>Top subtypes: </strong>
                    {g.top_subtypes && Object.entries(g.top_subtypes).slice(0, 3).map(([s, n]) => (
                      <span key={s} style={{ marginRight: 8 }}>{s} ({n})</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: '#333' }}>
                    {g.key_distinctions?.slice(0, 4).map(f => (
                      <div key={f} style={{ marginBottom: 2 }}>
                        <Badge text={f.split('-').slice(0, 3).join('-')} color={GENE_COLORS[g.gene] || '#1a237e'} />
                      </div>
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
            {definitions.definitions && Object.entries(definitions.definitions).map(([gene, d]) => (
              <div key={gene} style={{
                background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)',
                marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || '#1a237e'}`
              }}>
                <div style={{ fontWeight: 800, color: GENE_COLORS[gene] || '#1a237e', fontSize: 15, marginBottom: 4 }}>
                  {gene} <span style={{ fontWeight: 400, fontSize: 12, color: '#666' }}>— {d.locus}</span>
                </div>
                <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}>{d.pathognomonic}</div>
                <div style={{ fontSize: 11, color: '#555', marginBottom: 8 }}>
                  <strong>Cancer risk: </strong>{d.cancer_risk}
                </div>
                <div style={{ fontSize: 11, color: '#555', marginBottom: 8 }}>
                  <strong>Surveillance: </strong>{d.surveillance_key}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {d.key_distinctions?.map(k => (
                    <Badge key={k} text={k} color={GENE_COLORS[gene] || '#1a237e'} />
                  ))}
                </div>
              </div>
            ))}
            {definitions.key_rules && (
              <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,.08)', marginBottom: 16 }}>
                <h3 style={{ margin: '0 0 12px', color: '#b71c1c', fontSize: 15 }}>Critical Management Rules</h3>
                {Object.entries(definitions.key_rules).map(([key, val]) => (
                  <div key={key} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #f0f0f0' }}>
                    <div style={{ fontWeight: 700, color: '#1a237e', fontSize: 12, marginBottom: 4 }}>{key.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 12, color: '#444' }}>{val}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
