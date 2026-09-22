'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-acc-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TP53':    '#c62828',  // brick red       — LFS; pediatric ACC 50-70% HIGHEST; AVOID RADIATION ABSOLUTELY; WBMRI Toronto
  'NF1':     '#1b5e20',  // deep green      — NF1; adrenocortical 3-5%; MPNST dominant risk; café-au-lait PATHOGNOMONIC
  'MEN1':    '#0d47a1',  // deep blue       — MEN1/MEN1; adrenocortical adenoma 30-75%; parathyroid-pNET-pituitary triad
  'ARMC5':   '#4a148c',  // deep purple     — ARMC5; PBMAH bilateral macronodular; subclinical Cushing; bilateral adrenalectomy
  'PRKAR1A': '#880e4f',  // deep magenta    — Carney Complex; PPNAD paradoxical Liddle; cardiac myxoma LIFE-THREATENING
  'CDKN1C':  '#e65100',  // deep orange     — Beckwith-Wiedemann; pediatric ACC 2nd after Wilms; omphalocele-macroglossia PATHOGNOMONIC
  'APC':     '#006064',  // dark teal       — FAP/Gardner; adrenocortical adenoma; CHRPE PATHOGNOMONIC; prophylactic colectomy
  'DICER1':  '#33691e',  // deep olive      — DICER1 syndrome; PPB type I infancy PATHOGNOMONIC; SLCT ovary; ERMS cervix; MNG
};

const GENE_INFO = {
  'TP53':    { full: 'LFS / Pediatric-ACC-50-70%-LIFETIME-HIGHEST / R337H-Brazilian-Founder / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto-Annually',         locus: '17p13.1', size: '393 aa / 43 kDa',    inh: 'AD LOF' },
  'NF1':     { full: 'NF1 / Adrenocortical-3-5% / MPNST-Dominant-Risk / Cafe-au-Lait-6plus-PATHOGNOMONIC / Lisch-Nodules-PATHOGNOMONIC',                    locus: '17q11.2', size: '2839 aa / 327 kDa',  inh: 'AD LOF' },
  'MEN1':    { full: 'MEN1 / Adrenocortical-Adenoma-30-75%-Non-Functional / Parathyroid-95%-FIRST / pNET-40-70% / Pituitary-30-40%',                         locus: '11q13.1', size: '610 aa / 68 kDa',    inh: 'AD LOF' },
  'ARMC5':   { full: 'PBMAH-Bilateral-Macronodular / Subclinical-Cushing-Bilateral / Bilateral-Adrenalectomy-Definitive / ACTH-Independent-5-10%',           locus: '16p11.2', size: '1032 aa / 116 kDa',  inh: 'AD LOF' },
  'PRKAR1A': { full: 'Carney-Complex / PPNAD-Paradoxical-Liddle-Dexamethasone-DIAGNOSTIC / Cardiac-Myxoma-LIFE-THREATENING / Spotty-Pigmentation-PATHOGNOMONIC', locus: '17q24.2', size: '381 aa / 43 kDa',    inh: 'AD LOF' },
  'CDKN1C':  { full: 'Beckwith-Wiedemann / Pediatric-ACC-2nd-Most-Common-After-Wilms / Omphalocele-Macroglossia-Macrosomia-PATHOGNOMONIC / 11p15.4-Imprinting', locus: '11p15.4', size: '316 aa / 35 kDa',    inh: 'Mat LOF' },
  'APC':     { full: 'FAP-Gardner / Adrenocortical-Adenoma-FAP / CHRPE-PATHOGNOMONIC / Prophylactic-Colectomy-20-25yr / Desmoid-Post-Colectomy',              locus: '5q22.2',  size: '2843 aa / 310 kDa',  inh: 'AD LOF' },
  'DICER1':  { full: 'DICER1-Syndrome / PPB-Type-I-Infancy-PATHOGNOMONIC / SLCT-Ovary / ERMS-Cervix / MNG-75%-Females / Chest-CT-3-Monthly-First-3yr',       locus: '14q32.13', size: '1922 aa / 218 kDa', inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color,
      border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px',
      fontSize: 11, fontWeight: 600, marginRight: 4,
    }}>{text}</span>
  );
}

export default function HereditaryAccAtlas() {
  const [tab, setTab]                   = useState('Overview');
  const [overview, setOverview]         = useState(null);
  const [breakdown, setBreakdown]       = useState(null);
  const [definitions, setDefinitions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const containerStyle = {
    fontFamily: 'monospace', background: '#0a0a0a', color: '#e0e0e0',
    minHeight: '100vh', padding: '24px',
  };
  const cardStyle = {
    background: '#141414', border: '1px solid #2a2a2a',
    borderRadius: 8, padding: 20, marginBottom: 16,
  };
  const tabStyle = (active) => ({
    padding: '8px 18px', cursor: 'pointer', borderRadius: 4,
    background: active ? '#c62828' : '#1a1a1a',
    color: active ? '#ffcdd2' : '#888',
    border: `1px solid ${active ? '#c62828' : '#333'}`,
    fontFamily: 'monospace', fontSize: 13, marginRight: 6,
  });

  if (loading) return <div style={containerStyle}><p style={{ color: '#888' }}>Loading Hereditary Adrenocortical Carcinoma Predisposition Atlas…</p></div>;
  if (error)   return <div style={containerStyle}><p style={{ color: '#ef5350' }}>Error: {error}</p></div>;

  return (
    <div style={containerStyle}>
      {/* Header */}
      <div style={{ ...cardStyle, borderLeft: '4px solid #c62828' }}>
        <h1 style={{ color: '#ef9a9a', fontSize: 18, margin: '0 0 6px' }}>
          🧬 Hereditary Adrenocortical Carcinoma (ACC) Predisposition Atlas
        </h1>
        <div style={{ color: '#888', fontSize: 12 }}>
          Complete 8-Gene Reference · TP53-NF1-MEN1-ARMC5-PRKAR1A-CDKN1C-APC-DICER1
          {overview && <span> · Seeds {overview.seed_range} · {overview.total_patients} patients (8×40)</span>}
        </div>
        <div style={{ marginTop: 8, fontSize: 11, color: '#aaa' }}>
          <Badge text="TP53 Pediatric ACC 50-70% HIGHEST" color="#c62828" />
          <Badge text="PRKAR1A Cardiac Myxoma LIFE-THREATENING" color="#880e4f" />
          <Badge text="ARMC5 PBMAH Bilateral" color="#4a148c" />
          <Badge text="CDKN1C BWS Pediatric" color="#e65100" />
          <Badge text="DICER1 PPB Infancy PATHOGNOMONIC" color="#33691e" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} style={tabStyle(tab === t)} onClick={() => setTab(t)}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 15, margin: '0 0 12px' }}>
              Atlas Summary — Seeds {overview.seed_range}
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
              {[
                { label: 'Genes', value: overview.gene_count },
                { label: 'Total Patients', value: overview.total_patients },
                { label: 'Per Gene', value: overview.cohort_per_gene },
                { label: 'Highest ACC Risk', value: 'TP53 50-70%' },
              ].map(s => (
                <div key={s.label} style={{ background: '#1a1a1a', borderRadius: 6, padding: 12, textAlign: 'center' }}>
                  <div style={{ color: '#ef9a9a', fontSize: 20, fontWeight: 700 }}>{s.value}</div>
                  <div style={{ color: '#888', fontSize: 11 }}>{s.label}</div>
                </div>
              ))}
            </div>
            {overview.clinical_pearls && (
              <div>
                <div style={{ color: '#ef9a9a', fontSize: 12, fontWeight: 600, marginBottom: 6 }}>CLINICAL PEARLS:</div>
                {overview.clinical_pearls.map((p, i) => (
                  <div key={i} style={{ color: '#ccc', fontSize: 11, marginBottom: 5, paddingLeft: 10, borderLeft: '2px solid #333' }}>
                    {p}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Cancer Risk Summary */}
          {overview.cancer_risks && (
            <div style={cardStyle}>
              <h3 style={{ color: '#ef9a9a', fontSize: 14, margin: '0 0 10px' }}>ACC / Adrenal Risk by Gene</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr>
                      {['Gene', 'ACC / Adrenal Risk', 'Other Key Risks', 'Critical Warning'].map(h => (
                        <th key={h} style={{ color: '#888', padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.genes && overview.genes.map(g => {
                      const cr = overview.cancer_risks?.[g] || {};
                      const accRisk = cr.acc_pct ? `${cr.acc_pct}% lifetime ACC`
                        : cr.acc_rr ? `${cr.acc_rr}x RR ACC`
                        : cr.pbmah_pct ? `PBMAH ${cr.pbmah_pct}%`
                        : cr.ppnad_pct ? `PPNAD ${cr.ppnad_pct}%`
                        : cr.adrenal_adenoma_pct ? `Adenoma ${cr.adrenal_adenoma_pct}%`
                        : '—';
                      const warning = cr.avoid_radiation ? 'AVOID RADIATION ABSOLUTELY'
                        : cr.cardiac_myxoma_life_threatening ? 'Cardiac Myxoma LIFE-THREATENING'
                        : cr.bilateral_adrenalectomy ? 'Bilateral Adrenalectomy DEFINITIVE'
                        : cr.ppb_pathognomonic ? 'PPB PATHOGNOMONIC infancy'
                        : cr.chrpe_pathognomonic ? 'CHRPE PATHOGNOMONIC'
                        : cr.mpnst_dominant ? 'MPNST DOMINANT risk'
                        : cr.pediatric_dominant ? 'PEDIATRIC DOMINANT'
                        : '—';
                      return (
                        <tr key={g} style={{ borderBottom: '1px solid #1e1e1e' }}>
                          <td style={{ padding: '6px 8px', color: GENE_COLORS[g] || '#aaa', fontWeight: 700 }}>{g}</td>
                          <td style={{ padding: '6px 8px', color: '#90caf9' }}>{accRisk}</td>
                          <td style={{ padding: '6px 8px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[g]?.full?.split('/')[1]?.trim() || ''}</td>
                          <td style={{ padding: '6px 8px', color: '#ef9a9a', fontSize: 10 }}>{warning}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pathognomonic Features */}
          {overview.pathognomonic_features && (
            <div style={cardStyle}>
              <h3 style={{ color: '#ef9a9a', fontSize: 14, margin: '0 0 10px' }}>Pathognomonic Features</h3>
              {Object.entries(overview.pathognomonic_features).map(([gene, feat]) => (
                <div key={gene} style={{ marginBottom: 8, padding: '8px 12px', background: '#1a1a1a', borderRadius: 6, borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <span style={{ color: GENE_COLORS[gene] || '#aaa', fontWeight: 700, fontSize: 12 }}>{gene}: </span>
                  <span style={{ color: '#ccc', fontSize: 11 }}>{feat}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 15, margin: '0 0 12px' }}>Per-Gene Cohort Summary</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr>
                    {['Gene', 'Locus', 'N', 'Mean Age', 'ACC N', 'ACC %', 'Seed', 'Key Distinctions'].map(h => (
                      <th key={h} style={{ color: '#888', padding: '6px 8px', textAlign: 'left', borderBottom: '1px solid #333' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.breakdown && breakdown.breakdown.map(row => (
                    <tr key={row.gene} style={{ borderBottom: '1px solid #1e1e1e' }}>
                      <td style={{ padding: '6px 8px', color: GENE_COLORS[row.gene] || '#aaa', fontWeight: 700 }}>{row.gene}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa' }}>{row.locus}</td>
                      <td style={{ padding: '6px 8px', color: '#ccc' }}>{row.n}</td>
                      <td style={{ padding: '6px 8px', color: '#90caf9' }}>{row.mean_age_onset}</td>
                      <td style={{ padding: '6px 8px', color: '#ef9a9a' }}>{row.acc_n}</td>
                      <td style={{ padding: '6px 8px', color: '#ef9a9a' }}>{row.acc_pct}%</td>
                      <td style={{ padding: '6px 8px', color: '#888' }}>{row.seed}</td>
                      <td style={{ padding: '6px 8px', color: '#aaa', fontSize: 10 }}>{(row.key_distinctions || []).slice(0, 2).join(' / ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-gene patient samples */}
          {breakdown.breakdown && breakdown.breakdown.map(row => (
            <div key={row.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[row.gene] || '#555'}` }}
              onClick={() => setExpandedGene(expandedGene === row.gene ? null : row.gene)}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', cursor: 'pointer' }}>
                <div>
                  <span style={{ color: GENE_COLORS[row.gene] || '#aaa', fontWeight: 700, fontSize: 14 }}>{row.gene}</span>
                  <span style={{ color: '#888', fontSize: 11, marginLeft: 12 }}>{row.locus} · n={row.n} · mean age {row.mean_age_onset}yr · ACC {row.acc_pct}%</span>
                </div>
                <span style={{ color: '#666', fontSize: 12 }}>{expandedGene === row.gene ? '▲' : '▼'}</span>
              </div>
              {expandedGene === row.gene && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#ef9a9a', fontSize: 11 }}>KEY DISTINCTIONS:</span>
                    <div style={{ color: '#ccc', fontSize: 11 }}>{(row.key_distinctions || []).join(' · ')}</div>
                  </div>
                  <div style={{ overflowX: 'auto', marginTop: 10 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
                      <thead>
                        <tr>
                          {['Age', 'Cancer', 'Variant', 'Treatment', 'FH', 'Stage'].map(h => (
                            <th key={h} style={{ color: '#888', padding: '4px 6px', textAlign: 'left', borderBottom: '1px solid #222' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(row.patients || []).slice(0, 10).map((p, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid #1a1a1a' }}>
                            <td style={{ padding: '3px 6px', color: '#90caf9' }}>{p.age_at_diagnosis}</td>
                            <td style={{ padding: '3px 6px', color: '#ef9a9a' }}>{p.cancer_type}</td>
                            <td style={{ padding: '3px 6px', color: '#aaa' }}>{p.variant}</td>
                            <td style={{ padding: '3px 6px', color: '#a5d6a7' }}>{p.treatment}</td>
                            <td style={{ padding: '3px 6px', color: '#888' }}>{p.family_history}</td>
                            <td style={{ padding: '3px 6px', color: '#ffcc80' }}>{p.staging}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {(row.patients || []).length > 10 && (
                      <div style={{ color: '#666', fontSize: 10, marginTop: 4 }}>Showing 10 of {row.patients.length} patients</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && overview && (
        <div>
          {overview.genes && overview.genes.map(gene => (
            <div key={gene} style={{
              ...cardStyle,
              borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}`,
            }}>
              <h3 style={{ color: GENE_COLORS[gene] || '#aaa', margin: '0 0 8px', fontSize: 15 }}>
                {gene} — {GENE_INFO[gene]?.inh || ''}
              </h3>
              <div style={{ color: '#aaa', fontSize: 12, marginBottom: 6 }}>
                <span style={{ color: '#777' }}>Locus: </span>{GENE_INFO[gene]?.locus || ''}
                <span style={{ color: '#777', marginLeft: 12 }}>Size: </span>{GENE_INFO[gene]?.size || ''}
              </div>
              <div style={{ color: '#ccc', fontSize: 12, lineHeight: 1.6 }}>{GENE_INFO[gene]?.full || ''}</div>
              <div style={{ color: '#bbb', fontSize: 11, marginTop: 6 }}>{overview.inheritance_modes?.[gene] || ''}</div>
              {overview.key_distinctions?.[gene] && (
                <div style={{ color: '#90caf9', fontSize: 10, marginTop: 4 }}>{overview.key_distinctions[gene]}</div>
              )}
              {overview.surveillance_start_ages?.[gene] && (
                <div style={{ marginTop: 6, color: '#a5d6a7', fontSize: 10 }}>
                  Surveillance: {JSON.stringify(overview.surveillance_start_ages[gene])}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={cardStyle}>
            <h2 style={{ color: '#ef9a9a', fontSize: 16, margin: '0 0 12px' }}>
              Clinical Definitions — Seeds {definitions.seed_range}
            </h2>
            {definitions.definitions && definitions.definitions.map((d, i) => (
              <div key={i} style={{
                background: '#1a1a1a', borderRadius: 6, padding: 12, marginBottom: 10,
                borderLeft: `3px solid ${
                  i === 0 ? '#c62828' : i === 1 ? '#1b5e20' : i === 2 ? '#0d47a1' :
                  i === 3 ? '#4a148c' : i === 4 ? '#880e4f' : i === 5 ? '#e65100' :
                  i === 6 ? '#006064' : i === 7 ? '#33691e' : '#555'
                }`,
                cursor: 'pointer',
              }}
                onClick={() => setExpandedDef(expandedDef === i ? null : i)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 13 }}>{d.term}</span>
                  <span style={{ color: '#666', fontSize: 11 }}>{expandedDef === i ? '▲' : '▼'}</span>
                </div>
                {expandedDef === i && (
                  <div style={{ color: '#ccc', fontSize: 11, marginTop: 8, lineHeight: 1.7 }}>{d.definition}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
