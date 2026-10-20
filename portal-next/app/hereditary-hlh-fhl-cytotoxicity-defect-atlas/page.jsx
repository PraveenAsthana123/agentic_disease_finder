'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hlh-fhl-cytotoxicity-defect-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'PRF1':   '#b71c1c',  // deep red      -- FHL2; perforin absent; most common FHL
  'UNC13D': '#c62828',  // red           -- FHL3; Munc13-4; NK degranulation absent
  'STX11':  '#e65100',  // burnt orange  -- FHL4; syntaxin-11; Turkish/Kurdish founder
  'STXBP2': '#f57f17',  // amber         -- FHL5; Munc18-2; neutropenia+bleeding unique
  'RAB27A': '#1565c0',  // deep blue     -- Griscelli Type 2; silver hair PATHOGNOMONIC
  'LYST':   '#4a148c',  // deep purple   -- Chediak-Higashi; giant granules PATHOGNOMONIC
  'AP3B1':  '#1b5e20',  // deep green    -- HPS2; AP-3 complex; NK defect + albinism
  'DOCK8':  '#880e4f',  // dark crimson  -- DOCK8 deficiency; IgE+viral CID; NK partial
};

const GENE_INFO = {
  'PRF1':   { full: 'PRF1-555aa / AR / 10q22.1 / FHL2 / Perforin / Pore-Forming / Most-Common-FHL-40-50pct / Perforin-ABSENT-PATHOGNOMONIC / Granzyme-Delivery / HSCT-Curative / Emapalumab',            locus: '10q22.1',  size: '555 aa / 67 kDa', inh: 'Autosomal recessive' },
  'UNC13D': { full: 'UNC13D-1090aa / AR / 17q25.1 / FHL3 / Munc13-4 / SNARE-Priming / NK-Degranulation-ABSENT / Perforin-NORMAL / 25-30pct-FHL / CD107a-Assay / HSCT',                                  locus: '17q25.1',  size: '1090 aa / 124 kDa', inh: 'Autosomal recessive' },
  'STX11':  { full: 'STX11-287aa / AR / 6q24.2 / FHL4 / Syntaxin-11 / t-SNARE / Perforin-NORMAL / Turkish-Kurdish-p.L58P-Founder / Degranulation-ABSENT / HSCT',                                        locus: '6q24.2',   size: '287 aa / 33 kDa', inh: 'Autosomal recessive' },
  'STXBP2': { full: 'STXBP2-593aa / AR / 19p13.2 / FHL5 / Munc18-2 / STX11-Chaperone / HLH+Neutropenia+Bleeding-TRIAD / STX11-Secondary-Loss / Perforin-NORMAL / HSCT',                                 locus: '19p13.2',  size: '593 aa / 66 kDa', inh: 'Autosomal recessive' },
  'RAB27A': { full: 'RAB27A-221aa / AR / 15q21.3 / Griscelli-Syndrome-Type-2 / Rab-GTPase / Granule-Docking / Silver-Gray-Hair-PATHOGNOMONIC / HLH / NK-Degranulation-ABSENT / HSCT',                   locus: '15q21.3',  size: '221 aa / 25 kDa', inh: 'Autosomal recessive' },
  'LYST':   { full: 'LYST-3801aa / AR / 1q42.3 / Chediak-Higashi / BEACH-Domain / Organelle-Fission / Giant-Granules-PATHOGNOMONIC / Partial-Albinism / Accelerated-HLH-Phase / HSCT-Pre-Accelerated',  locus: '1q42.3',   size: '3801 aa / 429 kDa', inh: 'Autosomal recessive' },
  'AP3B1':  { full: 'AP3B1-1094aa / AR / 5q14.1 / HPS2 / AP-3-Beta-3A / LRO-Cargo-Sorting / Partial-OCA / Dense-Granule-Absence / NK-Cytotoxicity-Defect / Neutropenia / EBV-HLH-Risk',                locus: '5q14.1',   size: '1094 aa / 122 kDa', inh: 'Autosomal recessive' },
  'DOCK8':  { full: 'DOCK8-2099aa / AR / 9p24.3 / DOCK8-Deficiency / CDC42-GEF / Actin-IS / IgE-GT1000-PATHOGNOMONIC / Molluscum-Warts-HSV / NK-Partial / aCGH-Mandatory-Large-Deletions / HSCT',       locus: '9p24.3',   size: '2099 aa / 239 kDa', inh: 'Autosomal recessive' },
};

const PHENOTYPE_ROWS = [
  { gene: 'PRF1',   group: 'FHL2-Perforin',       perforin: 'ABSENT',  degranu: 'ABSENT',  age: '3 mo',  hair: '−',        gran: '−',       ige: '−',       inh: 'AR',    unique: 'PERFORIN ABSENT ON FLOW CYTOMETRY = FHL2 PATHOGNOMONIC; most common FHL (40-50%); fatal in weeks without HSCT; HLH-2004 + emapalumab if refractory; p.A91V homozygous = late-onset hypomorphic' },
  { gene: 'UNC13D', group: 'FHL3-Munc13-4',       perforin: 'NORMAL',  degranu: 'ABSENT',  age: '6 mo',  hair: '−',        gran: '−',       ige: '−',       inh: 'AR',    unique: 'PERFORIN NORMAL + DEGRANULATION ABSENT = FHL3; Munc13-4 primes SNARE complex; granules dock at membrane but FAIL TO FUSE (electron microscopy); second most common FHL (25-30%)' },
  { gene: 'STX11',  group: 'FHL4-Syntaxin-11',    perforin: 'NORMAL',  degranu: 'ABSENT',  age: '13 mo', hair: '−',        gran: '−',       ige: '−',       inh: 'AR',    unique: 'TURKISH/KURDISH p.L58P FOUNDER (80% of FHL4 in Turkish families); t-SNARE protein; lymphocyte-specific expression; check STX11 first in consanguineous Turkish/Middle Eastern HLH families' },
  { gene: 'STXBP2', group: 'FHL5-Munc18-2',       perforin: 'NORMAL',  degranu: 'ABSENT',  age: '12 mo', hair: '−',        gran: '−',       ige: '−',       inh: 'AR',    unique: 'HLH + NEUTROPENIA + PLATELET FUNCTION DEFECT = FHL5 PATHOGNOMONIC TRIPLE; STX11 ALSO LOW (secondary degradation without STXBP2); Crohn-like IBD in some; STXBP2 if HLH patient has neutropenia or bleeding' },
  { gene: 'RAB27A', group: 'GS2-Rab27A',          perforin: 'NORMAL',  degranu: 'ABSENT',  age: '18 mo', hair: 'SILVER✓✓',gran: '−',       ige: '−',       inh: 'AR',    unique: 'SILVER-GRAY METALLIC HAIR IN CHILD WITH HLH = GS2 PATHOGNOMONIC; RAB27A granule-docking GTPase; NO giant granules on smear (unlike CHS); hair remains silver post-HSCT; HSCT curative for HLH' },
  { gene: 'LYST',   group: 'CHS-Chediak-Higashi', perforin: 'NORMAL',  degranu: 'ABSENT',  age: '24 mo', hair: 'PALE✓',    gran: 'GIANT✓✓✓',ige: '−',       inh: 'AR',    unique: 'GIANT GRANULES IN BLOOD SMEAR (peroxidase stain, neutrophils) = CHEDIAK-HIGASHI PATHOGNOMONIC; silver-pale hair; accelerated HLH phase in 85-90%; HSCT pre-accelerated phase; LATE NEURO DETERIORATION post-HSCT (neurons not corrected)' },
  { gene: 'AP3B1',  group: 'HPS2-AP3',            perforin: 'NORMAL',  degranu: 'PARTIAL', age: '36 mo', hair: 'PALE±',    gran: '−',       ige: '−',       inh: 'AR',    unique: 'ALBINISM + PLATELET STORAGE POOL DEFECT (absent dense granules on EM) + NK DEFECT = HPS2; NO giant granules (unlike CHS); neutropenia in childhood; EBV-triggered HLH risk; platelet EM diagnostic' },
  { gene: 'DOCK8',  group: 'DOCK8-CID',           perforin: 'NORMAL',  degranu: 'PARTIAL', age: '48 mo', hair: '−',        gran: '−',       ige: 'HIGH✓✓✓', inh: 'AR',    unique: 'IgE >1000 + CID + MOLLUSCUM/WARTS/HSV = DOCK8 DEFICIENCY SIGNATURE; NK partial defect; EBV-HLH risk; LARGE DELETIONS 9p24.3 — aCGH MANDATORY (50% of variants); HSCT before lymphoma; IgE normalizes post-HSCT' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, cohort }) {
  const typeLabel = ['PRF1','UNC13D','STX11','STXBP2'].includes(gene) ? `FHL${gene==='PRF1'?'2':gene==='UNC13D'?'3':gene==='STX11'?'4':'5'} — Classical HLH`
                  : gene === 'RAB27A' ? 'Griscelli Syndrome Type 2'
                  : gene === 'LYST'   ? 'Chediak-Higashi Syndrome'
                  : gene === 'AP3B1'  ? 'Hermansky-Pudlak Type 2'
                  : 'DOCK8 Deficiency';

  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ fontWeight: 700, fontSize: 18, color }}>{gene}</span>
          <span style={{ marginLeft: 8, fontSize: 12, background: color, color: '#fff', padding: '2px 8px', borderRadius: 4 }}>{typeLabel}</span>
        </div>
        <div style={{ fontSize: 12, color: '#888', textAlign: 'right' }}>
          <div>{info.size} · {info.locus}</div>
          <div>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#bbb', marginTop: 6 }}>{info.full}</div>
      {cohort && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`HLH ${cohort.hlh_pct?.toFixed(0)}%`}                     color={color} />
          <Badge text={`NK Degranu ${cohort.nk_degranulation_pct?.toFixed(0)}%`} color='#880e4f' />
          <Badge text={`Ferritin ${cohort.ferritin_median_k?.toFixed(1)}k`}       color='#b71c1c' />
          <Badge text={`HSCT ${cohort.hsct_pct?.toFixed(0)}%`}                   color='#1565c0' />
          <Badge text={`5yr Surv ${cohort.survival_5yr_pct?.toFixed(0)}%`}        color='#1b5e20' />
          <Badge text={`Age dx ${cohort.age_at_diagnosis_mo} mo`}                 color='#555' />
          <Badge text={`n=${cohort.n} seed=${cohort.seed}`}                        color='#444' />
        </div>
      )}
    </div>
  );
}

export default function HLHFHLCytotoxicityDefectAtlas() {
  const [tab,      setTab]      = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown,setBreakdown]= useState(null);
  const [defs,     setDefs]     = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const cohorts = overview?.cohorts || {};

  return (
    <div style={{ background: '#111', minHeight: '100vh', color: '#eee', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: '#1a1a2e', border: '2px solid #b71c1c', borderRadius: 8, padding: 20, marginBottom: 20 }}>
          <h1 style={{ margin: 0, fontSize: 20, color: '#ff6b6b' }}>
            🧫 Hereditary-HLH-FHL-Cytotoxicity-Defect-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', color: '#ccc', fontSize: 13 }}>
            Complete 8-Gene Reference — Familial HLH Types 2-5, Griscelli Syndrome Type 2,
            Chediak-Higashi Syndrome, HPS Type 2, DOCK8 Deficiency · 320 Patients · Seeds 4038-4045
          </p>
          <div style={{ display: 'flex', gap: 8, marginTop: 12, flexWrap: 'wrap' }}>
            {['PRF1','UNC13D','STX11','STXBP2','RAB27A','LYST','AP3B1','DOCK8'].map(g => (
              <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '2px 10px',
                borderRadius: 4, fontSize: 12, fontWeight: 700 }}>{g}</span>
            ))}
          </div>
        </div>

        {/* Key Facts Bar */}
        <div style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 6, padding: 14, marginBottom: 20, fontSize: 12 }}>
          <div style={{ color: '#ff9800', fontWeight: 700, marginBottom: 8 }}>🔑 CRITICAL PATHWAY FACTS</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 8, color: '#ccc' }}>
            <div>🩸 <b style={{color:'#ff6b6b'}}>PERFORIN FLOW:</b> ABSENT only in FHL2/PRF1 · PRESENT in all others</div>
            <div>🧪 <b style={{color:'#ff9800'}}>NK DEGRANULATION (CD107a):</b> ABSENT in FHL2-5 + GS2 + CHS · PARTIAL in HPS2 + DOCK8</div>
            <div>🔬 <b style={{color:'#4a148c'}}>GIANT GRANULES (blood smear):</b> PATHOGNOMONIC for Chediak-Higashi (LYST) only</div>
            <div>💈 <b style={{color:'#1565c0'}}>SILVER HAIR:</b> PATHOGNOMONIC for Griscelli Type 2 (RAB27A) · also pale in CHS</div>
            <div>💊 <b style={{color:'#b71c1c'}}>EMAPALUMAB (anti-IFN-γ):</b> FDA 2018 · refractory primary HLH · bridge to HSCT</div>
            <div>🏥 <b style={{color:'#1b5e20'}}>HSCT = ONLY CURE:</b> All 8 genes · RIC preferred · before accelerated phase</div>
          </div>
        </div>

        {/* Tab Nav */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '8px 18px', borderRadius: 4, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t ? '#b71c1c' : '#222', color: tab === t ? '#fff' : '#aaa',
              fontFamily: 'monospace', fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>

        {loading && <div style={{ color: '#aaa', textAlign: 'center', padding: 40 }}>Loading atlas data…</div>}
        {error   && <div style={{ color: '#f44', textAlign: 'center', padding: 20 }}>Error: {error}</div>}

        {/* ── Overview ── */}
        {!loading && tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 16 }}>
              {Object.entries(cohorts).map(([gene, cohort]) => (
                <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} cohort={cohort} />
              ))}
            </div>
            {overview.key_facts && (
              <div style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 8, padding: 16, marginTop: 20 }}>
                <div style={{ color: '#ff9800', fontWeight: 700, marginBottom: 10 }}>Atlas Key Facts</div>
                {Object.entries(overview.key_facts).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 6, fontSize: 12, color: '#ccc' }}>
                    <span style={{ color: '#ff6b6b', fontWeight: 700 }}>{k.replace(/_/g,' ').toUpperCase()}: </span>{v}
                  </div>
                ))}
              </div>
            )}
            {overview.pathway_overview && (
              <div style={{ background: '#1a1a2e', border: '1px solid #1565c0', borderRadius: 6, padding: 14, marginTop: 16, fontSize: 12, color: '#aad4f5' }}>
                <b style={{color:'#64b5f6'}}>Cytotoxic Granule Pathway: </b>{overview.pathway_overview}
              </div>
            )}
          </div>
        )}

        {/* ── Gene Table ── */}
        {!loading && tab === 'Gene Table' && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a1a2e', color: '#aaa' }}>
                  {['Gene','Group','Perforin','NK Degranu','Age dx','Hair','Granules','IgE','Inh','Key Distinguisher'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#161616' : '#1a1a1a' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa', whiteSpace: 'nowrap' }}>{row.group}</td>
                    <td style={{ padding: '8px 10px', color: row.perforin === 'ABSENT' ? '#f44336' : '#4caf50', fontWeight: 700, whiteSpace: 'nowrap' }}>{row.perforin}</td>
                    <td style={{ padding: '8px 10px', color: row.degranu.includes('ABSENT') ? '#f44336' : row.degranu.includes('PARTIAL') ? '#ff9800' : '#4caf50', fontWeight: 700 }}>{row.degranu}</td>
                    <td style={{ padding: '8px 10px', color: '#ccc', whiteSpace: 'nowrap' }}>{row.age}</td>
                    <td style={{ padding: '8px 10px', color: row.hair !== '−' ? '#64b5f6' : '#555', fontWeight: row.hair !== '−' ? 700 : 400 }}>{row.hair}</td>
                    <td style={{ padding: '8px 10px', color: row.gran.includes('GIANT') ? '#ce93d8' : '#555', fontWeight: row.gran !== '−' ? 700 : 400 }}>{row.gran}</td>
                    <td style={{ padding: '8px 10px', color: row.ige.includes('HIGH') ? '#ffb74d' : '#555', fontWeight: row.ige !== '−' ? 700 : 400 }}>{row.ige}</td>
                    <td style={{ padding: '8px 10px', color: '#aaa' }}>{row.inh}</td>
                    <td style={{ padding: '8px 10px', color: '#ccc', maxWidth: 340 }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* ── Pathway Map ── */}
        {!loading && tab === 'Pathway Map' && breakdown && (
          <div>
            {breakdown.diagnostic_groups && Object.entries(breakdown.diagnostic_groups).map(([key, grp]) => (
              <div key={key} style={{ background: '#1a1a1a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 16 }}>
                <div style={{ fontWeight: 700, color: '#ff9800', marginBottom: 6, fontSize: 14 }}>
                  {key.replace(/_/g,' ').toUpperCase()}
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                  {grp.genes?.map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 12 }}>{g}</span>
                  ))}
                </div>
                <div style={{ fontSize: 12, color: '#ccc', marginBottom: 6 }}><b style={{color:'#aaa'}}>Mechanism: </b>{grp.mechanism}</div>
                <div style={{ fontSize: 12, color: '#ccc', marginBottom: 6 }}><b style={{color:'#aaa'}}>Shared: </b>{grp.shared_feature}</div>
                <div style={{ fontSize: 12, color: '#ffd54f' }}><b>DDx: </b>{grp.key_ddx}</div>
              </div>
            ))}
            {breakdown.pathognomonic_features && (
              <div style={{ background: '#1a1a2e', border: '1px solid #b71c1c', borderRadius: 8, padding: 16, marginTop: 8 }}>
                <div style={{ color: '#ff6b6b', fontWeight: 700, marginBottom: 10 }}>🔴 PATHOGNOMONIC FINDINGS</div>
                {Object.entries(breakdown.pathognomonic_features).map(([key, val]) => (
                  <div key={key} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #2a2a2a' }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: '#ff9800', marginBottom: 4 }}>{key.toUpperCase().replace(/_/g,' ')}</div>
                    <div style={{ fontSize: 12, color: '#ccc' }}>{val}</div>
                  </div>
                ))}
              </div>
            )}
            {breakdown.cascade_testing_order && (
              <div style={{ background: '#111', border: '1px solid #333', borderRadius: 8, padding: 16, marginTop: 16 }}>
                <div style={{ color: '#64b5f6', fontWeight: 700, marginBottom: 10 }}>🔬 CASCADE TESTING ORDER</div>
                {breakdown.cascade_testing_order.map((step, i) => (
                  <div key={i} style={{ marginBottom: 8, fontSize: 12, color: '#ccc', paddingLeft: 12, borderLeft: '2px solid #333' }}>
                    {step}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── Definitions ── */}
        {!loading && tab === 'Definitions' && defs && (
          <div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: 6, padding: 14, marginBottom: 12 }}>
                <div style={{ fontWeight: 700, color: '#ff9800', marginBottom: 6, fontSize: 13 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#ccc', lineHeight: 1.6 }}>{d.definition}</div>
              </div>
            ))}
            {defs.key_distinguishers && (
              <div style={{ background: '#1a1a2e', border: '1px solid #1565c0', borderRadius: 8, padding: 16, marginTop: 8 }}>
                <div style={{ color: '#64b5f6', fontWeight: 700, marginBottom: 10 }}>KEY DISTINGUISHERS</div>
                {Object.entries(defs.key_distinguishers).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8, fontSize: 12 }}>
                    <span style={{ color: '#ff9800', fontWeight: 700 }}>{k.replace(/_/g,' ')}: </span>
                    <span style={{ color: '#ccc' }}>{v}</span>
                  </div>
                ))}
              </div>
            )}
            {defs.treatment_matrix && (
              <div style={{ background: '#1a1a1a', border: '1px solid #2e7d32', borderRadius: 8, padding: 16, marginTop: 12 }}>
                <div style={{ color: '#66bb6a', fontWeight: 700, marginBottom: 10 }}>TREATMENT MATRIX</div>
                {Object.entries(defs.treatment_matrix).map(([k, v]) => (
                  <div key={k} style={{ marginBottom: 8, fontSize: 12 }}>
                    <span style={{ color: '#a5d6a7', fontWeight: 700 }}>{k.replace(/_/g,' ').toUpperCase()}: </span>
                    <span style={{ color: '#ccc' }}>{v}</span>
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
