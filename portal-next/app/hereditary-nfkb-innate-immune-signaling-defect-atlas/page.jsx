'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-nfkb-innate-immune-signaling-defect-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Groups', 'Definitions'];

const GENE_COLORS = {
  'IRAK4':  '#b71c1c',  // deep red      — AR; master TIR kinase; pyogenic bacterial
  'MYD88':  '#1565c0',  // deep blue     — AR; MyD88 adaptor; phenocopy IRAK4
  'TIRAP':  '#2e7d32',  // dark green    — AR; TLR2/4 bridging; malaria S180L balance
  'IKBKG':  '#6a1b9a',  // deep purple   — X-linked; NEMO; EDA-ID; IP in females
  'IKBKB':  '#e65100',  // deep orange   — AR; IKKβ; SCID-like; NF-kB absent all paths
  'IRAK1':  '#00695c',  // dark teal     — X-linked; downstream IRAK4; inflammatory
  'RELA':   '#4a148c',  // indigo        — AD LOF; p65; paradox autoinflammation IBD
  'NFKBIA': '#f57f17',  // amber         — AD GOF; IkBa; EDA-ID identical to NEMO
};

const GENE_INFO = {
  'IRAK4':  { full: 'IRAK4 / IL-1R-Associated Kinase 4 / 378aa', locus: '12q12',   size: '378 aa / 44 kDa (master TIR kinase; death domain + kinase domain; phosphorylates IRAK1; AR LOF; ALL MyD88-dependent TLRs affected; TLR3 intact; age-resistance unique)',           inh: 'AR LOF' },
  'MYD88':  { full: 'MYD88 / MyD88 Adaptor / 296aa',             locus: '3p22.2',  size: '296 aa / 33 kDa (TIR + intermediate + death domains; recruits IRAK4 to TLR/IL-1R; AR LOF; phenotypically identical to IRAK4; L265P = WM/DLBCL somatic GOF — different variant)',     inh: 'AR LOF' },
  'TIRAP':  { full: 'TIRAP / MAL / TLR2-TLR4 Adaptor / 221aa',   locus: '11q24.2', size: '221 aa / 25 kDa (bridges TLR2/4 to MyD88; PIP2 plasma membrane anchor; S180L heterozygous = malaria/TB protective; biallelic = TLR2/4-specific pyogenic susceptibility)',           inh: 'AR LOF' },
  'IKBKG':  { full: 'IKBKG / NEMO / IKKγ / 419aa',               locus: 'Xq28',    size: '419 aa / 48 kDa (IKK regulatory subunit; coiled-coil + ZF; COMPLETE LOF = Incontinentia Pigmenti lethal in males; HYPOMORPHIC = EDA-ID; BCG absolutely CI; EDA triad)',              inh: 'X-linked Hypomorphic' },
  'IKBKB':  { full: 'IKBKB / IKKβ / Catalytic Kinase / 756aa',   locus: '8p11.21', size: '756 aa / 87 kDa (phosphorylates IκBα Ser32/36; AR LOF; NF-κB absent ALL canonical paths; B cells present but nonfunctional; SCID-like; BCG CI; HSCT indicated)',                    inh: 'AR LOF' },
  'IRAK1':  { full: 'IRAK1 / IL-1R-Associated Kinase 1 / 712aa', locus: 'Xq28',    size: '712 aa / 83 kDa (downstream of IRAK4; IRAK2 partial redundancy; X-linked; milder than IRAK4; inflammatory/IBD features in some; somatic GOF in MDS — distinct from germline)',      inh: 'X-linked LOF' },
  'RELA':   { full: 'RELA / NF-κB p65 / 551aa',                  locus: '11q13.1', size: '551 aa / 65 kDa (p65/RelA transactivation subunit; AD haploinsufficiency; early-onset IBD + aphthous ulcers; IEC apoptosis PATHOGNOMONIC; A20 feedback lost; anti-TNF first)',       inh: 'AD LOF' },
  'NFKBIA': { full: 'NFKBIA / IκBα / IκB Inhibitor-α / 317aa',   locus: '14q13.2', size: '317 aa / 36 kDa (7 ankyrin repeats; PEST domain; AD GOF variants resist phosphorylation/degradation; NF-κB trapped; EDA-ID identical to NEMO; IκBα timecourse WB diagnostic)',      inh: 'AD GOF' },
};

const PATHOGNOMONIC = {
  'IRAK4':  'TLR3/TRIF-IFN-β INTACT (viral immunity spared); ALL MyD88-TLRs absent; pyogenic S.pneumo/S.aureus; AGE-RESISTANCE after 8-12yr — only immunodeficiency improving with age; HSCT NOT routine',
  'MYD88':  'Phenotypic COPY of IRAK4 deficiency; ALL MyD88-TLR responses absent; TLR3 normal; IRAK4 vs MYD88 indistinguishable clinically — molecular sequencing required; L265P = WM GOF not this',
  'TIRAP':  'TLR2/TLR4-SPECIFIC defect (other TLRs intact); S180L heterozygous = MALARIA/TB PROTECTIVE (14-20% Africans) — balanced polymorphism; biallelic = pyogenic gram-positive + gram-negative',
  'IKBKG':  'EDA TRIAD: anhidrosis (HEAT STROKE in summer) + hypotrichosis + oligodontia/conical teeth; X-linked; BCG ABSOLUTELY CI; mycobacterial susceptibility; hyper-IgM; COMPLETE LOF = incontinentia pigmenti',
  'IKBKB':  'SCID-LIKE with NF-κB absent ALL pathways; B CELLS PRESENT BUT NONFUNCTIONAL (NF-κB for activation not development); hypogammaglobulinemia + normal B cell count = NF-κB clue; BCG CI; HSCT',
  'IRAK1':  'X-linked (Xq28); IRAK2 partial redundancy → milder than IRAK4; inflammatory + immunodeficiency overlap; somatic IRAK1 GOF in MDS (different from germline LOF); no father-to-son transmission',
  'RELA':   'PARADOXICAL AUTOINFLAMMATION: p65 LOF → reduced A20 → prolonged NF-κB → IBD + aphthous ulcers; IEC apoptosis on biopsy PATHOGNOMONIC; anti-TNF (infliximab) MOST EFFECTIVE; AD inheritance',
  'NFKBIA': 'IκBα GOF: NF-κB TRAPPED — IκBα fails to degrade after TNF stimulation (timecourse WB); EDA-ID IDENTICAL to NEMO but AD (males + females); BCG CI; IκBα degradation assay diagnostic',
};

export default function HereditaryNFkBInnateAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDef, setExpandedDef] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#1a1a2e', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-NF-kB-Innate-Immune-Signaling-Defect-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#1a1a2e', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#1565c0 40%,#6a1b9a 75%,#f57f17 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary NF-κB Pathway · TIR Innate Immune Signaling · Pyogenic Bacterial Susceptibility · EDA-Immunodeficiency</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-NF-kB-Innate-Immune-Signaling-Defect-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · IRAK4 · MYD88 · TIRAP · IKBKG · IKBKB · IRAK1 · RELA · NFKBIA ·
          320 Patients (8×40) · Seeds 4094–4101
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'IRAK4/MYD88', val: 'Age-resistance UNIQUE' },
            { label: 'EDA-ID Genes', val: 'IKBKG + NFKBIA' },
            { label: 'BCG CI Genes', val: 'IKBKG·IKBKB·NFKBIA' },
          ].map(({ label, val }) => (
            <div key={label} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 14px' }}>
              <div style={{ fontSize: 11, opacity: 0.8 }}>{label}</div>
              <div style={{ fontWeight: 700, fontSize: 15 }}>{val ?? '…'}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#161b22', borderBottom: '1px solid #30363d', display: 'flex', gap: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '10px 20px', background: activeTab === t ? '#1f2d3d' : 'transparent',
              color: activeTab === t ? '#58a6ff' : '#8b949e', border: 'none', borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
              cursor: 'pointer', fontSize: 14 }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>

        {/* OVERVIEW TAB */}
        {activeTab === 'Overview' && overview && (
          <div>
            {/* Key facts */}
            <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 16 }}>⚡ Key Clinical Pearls — NF-κB/TIR Pathway</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 12 }}>
                {Object.entries(overview.key_facts || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 12, color: '#f0883e', fontWeight: 600, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 0.5 }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.5 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Gene cards */}
            <h3 style={{ color: '#58a6ff', fontSize: 16, marginBottom: 16 }}>Gene-Specific Cohort Summary (8 Genes × 40 Patients)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(310px,1fr))', gap: 16 }}>
              {(overview.genes || []).map(gene => {
                const cs = overview.cohort_summary?.[gene];
                const info = GENE_INFO[gene];
                const color = GENE_COLORS[gene] || '#444';
                return (
                  <div key={gene} style={{ background: '#161b22', border: `1px solid ${color}44`, borderRadius: 10, overflow: 'hidden' }}>
                    <div style={{ background: color, padding: '10px 14px' }}>
                      <div style={{ fontWeight: 700, fontSize: 15, color: '#fff' }}>{gene}</div>
                      <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.8)', marginTop: 2 }}>{info?.inh} · {info?.locus}</div>
                    </div>
                    <div style={{ padding: '12px 14px' }}>
                      <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>{info?.full}</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginBottom: 10 }}>
                        {[
                          ['Invasive Bacterial', `${cs?.invasive_bact_pct}%`],
                          ['Mycobacterial', `${cs?.mycobacterial_pct}%`],
                          ['EDA Features', `${cs?.eda_pct}%`],
                          ['Autoinflammatory', `${cs?.autoinflammatory_pct}%`],
                          ['Age Resistance', cs?.age_resistance ? 'YES' : 'No'],
                          ['BCG CI', cs?.bcg_ci ? '⛔ YES' : 'Safe'],
                        ].map(([lbl, val]) => (
                          <div key={lbl} style={{ background: '#0d1117', borderRadius: 6, padding: '5px 8px' }}>
                            <div style={{ fontSize: 10, color: '#8b949e' }}>{lbl}</div>
                            <div style={{ fontSize: 13, fontWeight: 600, color: val?.includes('⛔') ? '#f85149' : val === 'YES' ? '#3fb950' : '#e6edf3' }}>{val}</div>
                          </div>
                        ))}
                      </div>
                      <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#e6edf3', lineHeight: 1.5, borderLeft: `3px solid ${color}` }}>
                        {PATHOGNOMONIC[gene]}
                      </div>
                      <div style={{ marginTop: 8, fontSize: 11, color: '#8b949e' }}>Mean onset: <b>{cs?.mean_age_onset} yr</b> · HSCT: <b>{cs?.hsct_indicated ? 'Indicated' : 'Usually not'}</b></div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {activeTab === 'Gene Table' && overview && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#161b22' }}>
                  {['Gene', 'Size', 'Locus', 'Inheritance', 'Syndrome', 'Invs Bact', 'Myco', 'EDA', 'Age Resist', 'BCG CI', 'HSCT', 'Mean Onset'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#8b949e', fontWeight: 600, fontSize: 12, borderBottom: '1px solid #30363d', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.genes || []).map((gene, i) => {
                  const cs = overview.cohort_summary?.[gene];
                  const info = GENE_INFO[gene];
                  const color = GENE_COLORS[gene] || '#444';
                  return (
                    <tr key={gene} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '9px 12px', fontWeight: 700, color: color }}>{gene}</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{PROFILES_INFO[gene]?.size_aa ?? '—'} aa</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{cs?.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3', fontSize: 12 }}>{cs?.inheritance}</td>
                      <td style={{ padding: '9px 12px', color: '#cdd9e5', fontSize: 12, maxWidth: 200 }}>{cs?.syndrome}</td>
                      <td style={{ padding: '9px 12px', color: '#3fb950', fontWeight: 600 }}>{cs?.invasive_bact_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.mycobacterial_pct > 20 ? '#f85149' : '#8b949e' }}>{cs?.mycobacterial_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.eda_pct > 50 ? '#f0883e' : '#8b949e' }}>{cs?.eda_pct}%</td>
                      <td style={{ padding: '9px 12px', color: cs?.age_resistance ? '#3fb950' : '#8b949e', fontWeight: cs?.age_resistance ? 700 : 400 }}>{cs?.age_resistance ? 'YES' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.bcg_ci ? '#f85149' : '#8b949e', fontWeight: cs?.bcg_ci ? 700 : 400 }}>{cs?.bcg_ci ? '⛔ CI' : 'Safe'}</td>
                      <td style={{ padding: '9px 12px', color: cs?.hsct_indicated ? '#f0883e' : '#8b949e', fontWeight: cs?.hsct_indicated ? 700 : 400 }}>{cs?.hsct_indicated ? 'Yes' : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3' }}>{cs?.mean_age_onset} yr</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div style={{ marginTop: 20 }}>
              <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Gene Detail — Protein & Locus Reference</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 12 }}>
                {Object.entries(GENE_INFO).map(([gene, info]) => (
                  <div key={gene} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: '12px 14px' }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 4 }}>{gene}</div>
                    <div style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5 }}>{info.size}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* PATHWAY GROUPS TAB */}
        {activeTab === 'Pathway Groups' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(380px,1fr))', gap: 20, marginBottom: 28 }}>
              {Object.entries(breakdown.pathway_groups || {}).map(([groupKey, group]) => (
                <div key={groupKey} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 18 }}>
                  <div style={{ fontWeight: 700, color: '#58a6ff', fontSize: 15, marginBottom: 8, textTransform: 'capitalize' }}>{groupKey.replace(/_/g, ' ')}</div>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                    {(group.genes || []).map(g => (
                      <span key={g} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', borderRadius: 5, padding: '2px 8px', fontSize: 12, fontWeight: 700 }}>{g}</span>
                    ))}
                  </div>
                  <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.6, marginBottom: 10 }}>{group.description}</div>
                  <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#f0883e' }}>
                    <b>Key Test:</b> {group.key_test}
                  </div>
                </div>
              ))}
            </div>
            <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Gene Summary Stats</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 12 }}>
              {Object.entries(breakdown.gene_summary || {}).map(([gene, gs]) => (
                <div key={gene} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 14, marginBottom: 6 }}>{gene}</div>
                  <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 4 }}>n={gs.n} · Seed {gs.seed}</div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Invasive bact: <b>{gs.invasive_bact_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>EDA: <b>{gs.eda_pct}%</b></div>
                  <div style={{ fontSize: 12, color: '#cdd9e5' }}>Mean onset: <b>{gs.mean_age_onset} yr</b></div>
                  <div style={{ fontSize: 12, color: gs.bcg_ci ? '#f85149' : '#8b949e', marginTop: 4 }}>BCG: <b>{gs.bcg_ci ? '⛔ CI' : 'Safe'}</b></div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            <div style={{ display: 'grid', gap: 12 }}>
              {(definitions.definitions || []).map((def, i) => (
                <div key={i} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, overflow: 'hidden' }}>
                  <button onClick={() => setExpandedDef(expandedDef === i ? null : i)}
                    style={{ width: '100%', textAlign: 'left', padding: '14px 18px', background: 'transparent', border: 'none', cursor: 'pointer',
                      color: '#58a6ff', fontSize: 14, fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{def.term}</span>
                    <span style={{ color: '#8b949e', fontSize: 18, marginLeft: 12 }}>{expandedDef === i ? '▲' : '▼'}</span>
                  </button>
                  {expandedDef === i && (
                    <div style={{ padding: '0 18px 16px', fontSize: 13, color: '#cdd9e5', lineHeight: 1.75, borderTop: '1px solid #21262d' }}>
                      <div style={{ marginTop: 12 }}>{def.definition}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
            <div style={{ marginTop: 28 }}>
              <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Key Distinguishers</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(330px,1fr))', gap: 10 }}>
                {Object.entries(definitions.key_distinguishers || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#161b22', border: '1px solid #21262d', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 11, color: '#f0883e', fontWeight: 700, marginBottom: 4, textTransform: 'uppercase' }}>{k.replace(/_/g, ' ')}</div>
                    <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.5 }}>{v}</div>
                  </div>
                ))}
              </div>
              <h3 style={{ color: '#58a6ff', fontSize: 15, margin: '24px 0 12px' }}>Treatment Matrix</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 10 }}>
                {Object.entries(definitions.treatment_matrix || {}).map(([k, v]) => (
                  <div key={k} style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: 8, padding: '10px 14px' }}>
                    <div style={{ fontSize: 12, color: '#3fb950', fontWeight: 700, marginBottom: 4 }}>{k.replace(/_/g, ' ').toUpperCase()}</div>
                    <div style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5 }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const PROFILES_INFO = {
  'IRAK4':  { size_aa: 378 },
  'MYD88':  { size_aa: 296 },
  'TIRAP':  { size_aa: 221 },
  'IKBKG':  { size_aa: 419 },
  'IKBKB':  { size_aa: 756 },
  'IRAK1':  { size_aa: 712 },
  'RELA':   { size_aa: 551 },
  'NFKBIA': { size_aa: 317 },
};
