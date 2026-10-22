'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-msud-bcaa-disorder-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'BCKDHA': '#e65100',  // deep orange      -- MSUD Ia; BCKDH E1alpha; alloisoleucine PATHOGNOMONIC
  'BCKDHB': '#bf360c',  // deep red-orange  -- MSUD Ib; BCKDH E1beta; similar phenotype to Ia
  'DBT':    '#f57f17',  // amber            -- MSUD II; BCKDH E2 dihydrolipoyl transacylase
  'DLD':    '#827717',  // dark yellow-green -- MSUD III / DLD deficiency; E3; combined OA defect
  'BCKDK':  '#1a237e',  // dark indigo      -- BCKDK deficiency; BCCAs VERY LOW; NBS misses; autism/epilepsy
  'PPM1K':  '#4a148c',  // deep purple      -- PPM1K; BCKD phosphatase activator; mild MSUD phenotype
  'HIBCH':  '#1b5e20',  // dark green       -- HIBCH; β-hydroxyisobutyryl-CoA hydrolase; valine catabolism
  'MCCC1':  '#01579b',  // dark teal        -- MCCC1; 3-methylcrotonyl-CoA carboxylase; leucine catabolism
};

const GENE_INFO = {
  'BCKDHA': { full: 'BCKDHA-445aa / AR / E1alpha / BCKDH-complex / 19q13.2 / MSUD-Ia / alloisoleucine-PATHOGNOMONIC / maple-syrup-urine-smell / NBS-elevated-xLeu / thiamine-responsive-subset / BCAA-restrict-diet / liver-transplant-curative', locus: '19q13.2', size: '445 aa / 50 kDa', inh: 'Autosomal recessive (AR; most common MSUD gene; Y393N Mennonite founder allele)' },
  'BCKDHB': { full: 'BCKDHB-392aa / AR / E1beta / BCKDH-complex / 6q14.1 / MSUD-Ib / alloisoleucine-PATHOGNOMONIC / indistinguishable-from-Ia-clinically / NBS-elevated-xLeu / liver-transplant-curative', locus: '6q14.1', size: '392 aa / 43 kDa', inh: 'Autosomal recessive (AR; MSUD Ib; clinically indistinguishable from MSUD Ia; liver transplant curative)' },
  'DBT':    { full: 'DBT-421aa / AR / E2 / BCKDH-complex-dihydrolipoyl-transacylase / 1p21.2 / MSUD-II / alloisoleucine-PATHOGNOMONIC / intermediate-classic-forms / NBS-elevated-xLeu', locus: '1p21.2', size: '421 aa / 52 kDa', inh: 'Autosomal recessive (AR; MSUD II; E2 transacylase; intermediate forms common; alloisoleucine positive)' },
  'DLD':    { full: 'DLD-509aa / AR / E3 / dihydrolipoamide-dehydrogenase / 7q31.1 / MSUD-III-DLD-deficiency / COMBINED-organic-acidemia-PATHOGNOMONIC / PDH+alpha-KG+BCKDH-all-deficient / lactic-acidosis-dominant / Ashkenazi-p.G229C-founder', locus: '7q31.1', size: '509 aa / 56 kDa', inh: 'Autosomal recessive (AR; MSUD III / DLD deficiency; Ashkenazi p.G229C founder; COMBINED complex defect)' },
  'BCKDK':  { full: 'BCKDK-412aa / AR-LOF / BCKD-kinase / BCKDH-inhibitor / 16p11.2 / BCKDK-deficiency / BCCAs-VERY-LOW-PATHOGNOMONIC-REVERSAL / NBS-MISSES-normal-xLeu / autism-epilepsy-BCAA-responsive / BCAA-supplementation-treatment', locus: '16p11.2', size: '412 aa / 46 kDa', inh: 'Autosomal recessive (AR LOF; BCKDK normally inhibits BCKDH; LOF = overactive BCKDH = LOW BCAAs; opposite of MSUD)' },
  'PPM1K':  { full: 'PPM1K-359aa / AR / PP2Cm-phosphatase / BCKDH-activator / 4q22.1 / PPM1K-deficiency / BCKDH-not-activated / mild-MSUD-like / NBS-elevated-xLeu-mild / BCAA-restrict', locus: '4q22.1', size: '359 aa / 42 kDa', inh: 'Autosomal recessive (AR; PP2Cm phosphatase activates BCKDH by dephosphorylation; LOF = mild MSUD phenotype)' },
  'HIBCH':  { full: 'HIBCH-338aa / AR / beta-hydroxyisobutyryl-CoA-hydrolase / valine-catabolism / 2q32.2 / HIBCH-deficiency / valine-restriction / C4-OH-acylcarnitine / progressive-neurological / brain-MRI-abnormal', locus: '2q32.2', size: '338 aa / 38 kDa', inh: 'Autosomal recessive (AR; valine catabolism distal to BCKDH; C4-OH elevated; progressive neurological deterioration)' },
  'MCCC1':  { full: 'MCCC1-725aa / AR / 3-MCC-alpha-subunit / leucine-catabolism / 3q27.1 / 3-MCC-deficiency / 3-methylcrotonyl-CoA-carboxylase / C5-OH-NBS-most-common-false-positive / mostly-benign / maternal-3-MCC', locus: '3q27.1', size: '725 aa / 80 kDa', inh: 'Autosomal recessive (AR; leucine catabolism; C5-OH most common NBS false-positive; majority benign asymptomatic)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'BCKDHA', group: 'MSUD Type Ia (E1α deficiency)',           bcaa_dir: '↑↑↑',  nbs: '✓',  thiamine: '±',  rx_first: 'BCAA restriction + liver Tx', unique: 'MSUD Ia most common; ALLOISOLEUCINE in plasma PATHOGNOMONIC for ALL BCKDH defects; maple syrup urine odour; Y393N Mennonite founder; neonatal encephalopathy if untreated; liver transplant corrects metabolically' },
  { gene: 'BCKDHB', group: 'MSUD Type Ib (E1β deficiency)',           bcaa_dir: '↑↑↑',  nbs: '✓',  thiamine: '±',  rx_first: 'BCAA restriction + liver Tx', unique: 'MSUD Ib clinically INDISTINGUISHABLE from Ia; alloisoleucine positive; elevated leucine/isoleucine/valine on NBS; severe neonatal form most common; liver transplant curative' },
  { gene: 'DBT',    group: 'MSUD Type II (E2 deficiency)',             bcaa_dir: '↑↑',   nbs: '✓',  thiamine: '±',  rx_first: 'BCAA restriction',           unique: 'MSUD II E2 transacylase; intermediate and classic forms; alloisoleucine PATHOGNOMONIC; variable severity; some thiamine-responsive subset; NBS xLeu elevated' },
  { gene: 'DLD',    group: 'MSUD III / DLD Deficiency',               bcaa_dir: '↑',    nbs: '✓',  thiamine: '✗',  rx_first: 'Protein restrict + riboflavin', unique: 'DLD E3 shared by PDH+α-KGDH+BCKDH+GCS; COMBINED deficiency PATHOGNOMONIC; lactic acidosis dominant; Ashkenazi p.G229C founder; riboflavin trial; liver Tx NOT curative (systemic E3)' },
  { gene: 'BCKDK',  group: 'BCKDK Deficiency (BCAA-responsive autism)', bcaa_dir: '↓↓↓', nbs: '✗',  thiamine: '✗',  rx_first: 'BCAA supplementation',       unique: 'BCAA VERY LOW — OPPOSITE direction of MSUD; NBS COMPLETELY MISSES; presents autism/intellectual disability/epilepsy; plasma BCAAs diagnostic; BCAA protein supplementation dramatically improves neurology' },
  { gene: 'PPM1K',  group: 'PPM1K Deficiency (Mild MSUD-like)',        bcaa_dir: '↑',    nbs: '±',  thiamine: '✗',  rx_first: 'BCAA restriction (mild)',     unique: 'PP2Cm phosphatase activates BCKDH; LOF = mild MSUD phenotype; variable presentation; mildly elevated xLeu on NBS; BCAA restriction; rare' },
  { gene: 'HIBCH',  group: 'HIBCH Deficiency (Valine catabolism)',     bcaa_dir: 'V↑',   nbs: '±',  thiamine: '✗',  rx_first: 'Valine restriction',         unique: 'Valine catabolism defect DISTAL to BCKDH; C4-OH acylcarnitine elevated; progressive neurological deterioration; brain MRI abnormalities; valine restriction; alloisoleucine NEGATIVE' },
  { gene: 'MCCC1',  group: '3-MCC Deficiency (Leucine catabolism)',    bcaa_dir: 'L↑',   nbs: '✓',  thiamine: '✗',  rx_first: 'Usually no treatment needed', unique: 'C5-OH MOST COMMON NBS FALSE-POSITIVE; MOST patients completely asymptomatic; leucine catabolism; maternal 3-MCC can screen positive in newborn; biotin NOT responsive; majority benign' },
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
  const groupLabel = {
    BCKDHA: 'MSUD-Ia-E1alpha-Alloisoleucine-PATHOGNOMONIC-Maple-Syrup-Urine-Mennonite-LiverTx',
    BCKDHB: 'MSUD-Ib-E1beta-Alloisoleucine-PATHOGNOMONIC-Indistinguishable-Ia-LiverTx',
    DBT:    'MSUD-II-E2-Transacylase-Alloisoleucine-PATHOGNOMONIC-Intermediate-Classic',
    DLD:    'MSUD-III-DLD-CombinedComplexDefect-PATHOGNOMONIC-LacticAcidosis-Ashkenazi',
    BCKDK:  'BCKDK-BCCAs-VERY-LOW-PATHOGNOMONIC-NBS-MISSES-Autism-Epilepsy-BCAA-Supplement',
    PPM1K:  'PPM1K-MildMSUD-PP2Cm-Phosphatase-BCKDH-Activator',
    HIBCH:  'HIBCH-Valine-Catabolism-C4-OH-Progressive-Neuro-Alloisoleucine-Negative',
    MCCC1:  'MCCC1-3MCC-Leucine-Catabolism-C5OH-NBS-FALSE-POSITIVE-Mostly-Benign',
  }[gene] || gene;

  return (
    <div style={{
      background: '#1a1a2e', border: `2px solid ${color}`, borderRadius: 8,
      padding: 16, marginBottom: 16
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <span style={{
          background: color, color: '#fff', fontWeight: 700, fontSize: 18,
          padding: '4px 12px', borderRadius: 6
        }}>{gene}</span>
        <div>
          <div style={{ color: '#ccc', fontSize: 12 }}>{info.locus} · {info.size}</div>
          <div style={{ color: '#aaa', fontSize: 11 }}>{info.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#e8c270', marginBottom: 6 }}>
        <strong>Disease:</strong> {cohort?.disease || groupLabel}
      </div>
      <div style={{ fontSize: 12, color: '#f0a0a0', marginBottom: 4 }}>
        <strong>BCAA Direction:</strong> {cohort?.bcaa_direction || '—'}
      </div>
      {cohort && (
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#aaa', marginBottom: 6 }}>
          <span>n={cohort.n}</span>
          <span>M:{cohort.male_n} F:{cohort.female_n}</span>
          <span>Onset median: {cohort.median_onset_months}mo</span>
          <span>NBS detected: {(cohort.nbs_detected_pct * 100).toFixed(0)}%</span>
          <span>Thiamine resp: {(cohort.thiamine_responsive_pct * 100).toFixed(0)}%</span>
        </div>
      )}
      <div style={{ fontSize: 11, color: '#9ecfff', marginTop: 6, lineHeight: 1.5 }}>
        <strong>Pathognomonic:</strong> {cohort?.pathognomonic || info.full}
      </div>
      <div style={{ fontSize: 11, color: '#a8e6cf', marginTop: 6, lineHeight: 1.5 }}>
        <strong>Key Rx:</strong> {cohort?.key_drug}
      </div>
    </div>
  );
}

export default function Page() {
  const [tab, setTab]         = useState('Overview');
  const [overview, setOv]     = useState(null);
  const [breakdown, setBk]    = useState(null);
  const [defs, setDefs]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = (tab === 'Definitions') ? 'definitions'
             : (tab === 'Pathway Map' || tab === 'Treatment') ? 'breakdown'
             : 'overview';
    const setter = (tab === 'Definitions') ? setDefs
                 : (tab === 'Pathway Map' || tab === 'Treatment') ? setBk
                 : setOv;
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => { setter(d); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, [tab]);

  const tabStyle = (t) => ({
    padding: '8px 16px', cursor: 'pointer', borderRadius: '6px 6px 0 0',
    background: tab === t ? '#e65100' : '#1a1a2e',
    color: '#fff', border: 'none', marginRight: 4, fontWeight: tab === t ? 700 : 400,
  });

  return (
    <div style={{ background: '#0d0d1a', minHeight: '100vh', color: '#f0f0f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: 'linear-gradient(135deg,#e65100,#bf360c)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f9ec; Hereditary-MSUD-BCAA-Disorder-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: BCKDHA-BCKDHB-DBT-DLD-BCKDK-PPM1K-HIBCH-MCCC1
            · 320 Patients (8×40) · Seeds 4182-4189
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.85 }}>
            MSUD-Ia (BCKDHA-alloisoleucine-liver-Tx) · MSUD-Ib (BCKDHB-indistinguishable-Ia)
            · MSUD-II (DBT-E2-intermediate) · DLD-Deficiency (DLD-combined-complexes-lactic-acidosis)
            · BCKDK-deficiency (BCCAs-VERY-LOW-NBS-MISSES-autism-epilepsy-BCAA-supplement)
            · PPM1K (mild-MSUD) · HIBCH (valine-catabolism-C4-OH) · 3-MCC (MCCC1-C5-OH-NBS-false-positive)
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', marginBottom: 0, flexWrap: 'wrap' }}>
          {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
        </div>
        <div style={{ background: '#12122a', borderRadius: '0 8px 8px 8px', padding: 24, minHeight: 400 }}>
          {loading && <div style={{ color: '#aaa' }}>Loading…</div>}
          {error && <div style={{ color: '#f44336' }}>Error: {error}</div>}

          {/* OVERVIEW TAB */}
          {tab === 'Overview' && overview && (
            <div>
              <h2 style={{ color: '#ffb74d', marginTop: 0 }}>Atlas Overview</h2>
              <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Total Patients</div>
                  <div style={{ color: '#fff', fontSize: 28, fontWeight: 700 }}>{overview.total_patients || 320}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Genes</div>
                  <div style={{ color: '#ffb300', fontSize: 28, fontWeight: 700 }}>8</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 200 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Seeds</div>
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds || '4182-4189'}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 200 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Curative intervention</div>
                  <div style={{ color: '#ce93d8', fontSize: 14, fontWeight: 700 }}>Liver Tx · BCAA supplement</div>
                </div>
              </div>

              {/* BCAA direction spectrum */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#ffb74d', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  BCAA Direction — The Critical Diagnostic Divide
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#2d1000', borderRadius: 6, fontSize: 11, color: '#ff7043' }}>
                    <strong>↑↑↑ Elevated:</strong> BCKDHA/Ib/DBT/DLD — MSUD/classic alloisoleucine positive
                  </div>
                  <div style={{ padding: '6px 10px', background: '#001a2d', borderRadius: 6, fontSize: 11, color: '#64b5f6' }}>
                    <strong>↓↓↓ VERY LOW:</strong> BCKDK — opposite of MSUD; NBS MISSES; autism/epilepsy; BCAA supplement
                  </div>
                  <div style={{ padding: '6px 10px', background: '#1a2d00', borderRadius: 6, fontSize: 11, color: '#aed581' }}>
                    <strong>Downstream / Partial:</strong> PPM1K (mild ↑) · HIBCH (valine only) · MCCC1 (leucine only, mostly benign)
                  </div>
                </div>
              </div>

              {/* Alloisoleucine highlight */}
              <div style={{ background: '#2d1000', border: '1px solid #e65100', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#ff8a65', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  Alloisoleucine — Pathognomonic for ALL BCKDH Complex Defects (MSUD Ia/Ib/II/III)
                </div>
                <div style={{ fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
                  Alloisoleucine is ABSENT in normal plasma and ONLY appears when the BCKDH complex is deficient.
                  It is the stereoisomer of L-isoleucine formed by transamination of (2S,3R)-2-keto-3-methylvalerate.
                  Positive alloisoleucine = confirms MSUD/DLD regardless of exact gene.
                  BCKDK deficiency: alloisoleucine NEGATIVE (BCAAs low, not elevated).
                </div>
              </div>

              {/* NBS pitfall */}
              <div style={{ background: '#001a2d', border: '1px solid #0288d1', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#29b6f6', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  NBS Critical Pitfall: BCKDK Deficiency (xLeu Normal → Completely Missed)
                </div>
                <div style={{ fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
                  BCKDK LOF = overactive BCKDH = BCAAs consumed too fast = LOW plasma BCAAs.
                  NBS screens for elevated leucine/isoleucine — BCKDK has NORMAL or LOW levels → completely missed.
                  Presents as autism/intellectual disability/refractory epilepsy. Plasma amino acids + urine organic acids needed.
                  BCAA protein supplementation (branched-chain amino acids) corrects CNS substrate deficiency.
                </div>
              </div>

              {/* Gene cards */}
              <h3 style={{ color: '#ffb74d', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.gene_cohorts || []).map((c) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#ffb74d', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Disease','aa','Locus','kDa','Inheritance','BCAA Dir','NBS','Thiamine','Key Rx','Pathognomonic Feature'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#ffb74d', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PHENOTYPE_ROWS.map((row, i) => (
                      <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0d0d1a' : '#12122a' }}>
                        <td style={{ padding: '6px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10 }}>{row.group}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[0]}</td>
                        <td style={{ padding: '6px', color: '#9ecfff' }}>{GENE_INFO[row.gene]?.locus}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[1]}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[row.gene]?.inh?.split('(')[0]}</td>
                        <td style={{ padding: '6px', color: row.bcaa_dir.startsWith('↓') ? '#64b5f6' : row.bcaa_dir.startsWith('↑↑↑') ? '#ff5252' : '#ffb300', fontWeight: 700 }}>{row.bcaa_dir}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.nbs === '✓' ? '#69f0ae' : row.nbs === '±' ? '#ffd740' : '#f44336' }}>{row.nbs}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.thiamine === '±' ? '#ffd740' : '#888' }}>{row.thiamine}</td>
                        <td style={{ padding: '6px', color: '#a8e6cf', fontSize: 10 }}>{row.rx_first}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10, maxWidth: 280 }}>{row.unique}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* PATHWAY MAP TAB */}
          {tab === 'Pathway Map' && breakdown && (
            <div>
              <h2 style={{ color: '#ffb74d', marginTop: 0 }}>BCAA Catabolism Pathway Groups</h2>

              {/* Pathway groups */}
              {Object.entries(breakdown.pathway_groups || {}).map(([grp, data]) => (
                <div key={grp} style={{ background: '#12122a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <span style={{ color: '#ffb74d', fontWeight: 700, fontSize: 13 }}>
                      {grp.replace(/_/g, ' ').toUpperCase()}
                    </span>
                    {(data.genes || []).map(g => (
                      <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />
                    ))}
                  </div>
                  {data.mechanism && (
                    <div style={{ fontSize: 11, color: '#e8c270', marginBottom: 6 }}>
                      <strong>Mechanism:</strong> {data.mechanism}
                    </div>
                  )}
                  {data.shared_feature && (
                    <div style={{ fontSize: 11, color: '#9ecfff', marginBottom: 6 }}>
                      <strong>Shared:</strong> {data.shared_feature}
                    </div>
                  )}
                  {data.key_ddx && (
                    <div style={{ fontSize: 11, color: '#a8e6cf', lineHeight: 1.5 }}>
                      <strong>DDx:</strong> {data.key_ddx}
                    </div>
                  )}
                </div>
              ))}

              {/* Cascade testing */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>BCAA Diagnostic Cascade Algorithm</h3>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {(breakdown.bcaa_diagnostic_cascade || []).map((step, i) => (
                  <li key={i} style={{ fontSize: 11, color: '#ccc', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>

              {/* NBS pitfall table */}
              {breakdown.nbs_pitfall_table && (
                <div style={{ marginTop: 20 }}>
                  <h3 style={{ color: '#29b6f6', fontSize: 13, marginBottom: 10 }}>NBS Detection Table by Gene</h3>
                  {Object.entries(breakdown.nbs_pitfall_table).map(([gene, note]) => (
                    <div key={gene} style={{ background: '#001a2d', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
                        <Badge text={gene} color={GENE_COLORS[gene] || '#555'} />
                        <span style={{ color: '#ccc', fontSize: 11 }}>{note}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TREATMENT TAB */}
          {tab === 'Treatment' && breakdown && (
            <div>
              <h2 style={{ color: '#ffb74d', marginTop: 0 }}>Treatment Pathways</h2>
              {(breakdown.treatment_pathways || []).map((tp, i) => (
                <div key={i} style={{ background: '#0d1a0d', border: '1px solid #2e7d32', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 6, alignItems: 'center' }}>
                    <Badge text={tp.gene} color={GENE_COLORS[tp.gene] || '#555'} />
                    <span style={{ color: '#a8e6cf', fontWeight: 700, fontSize: 12 }}>{tp.disease}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#9ecfff', marginBottom: 4 }}>
                    <strong>First-line:</strong> {tp.first_line}
                  </div>
                  {tp.precision_therapy && (
                    <div style={{ fontSize: 11, color: '#ffcc80', marginBottom: 4 }}>
                      <strong>Precision:</strong> {tp.precision_therapy}
                    </div>
                  )}
                  {tp.avoid && (
                    <div style={{ fontSize: 11, color: '#ff5252', marginBottom: 4 }}>
                      <strong>Avoid/CI:</strong> {tp.avoid}
                    </div>
                  )}
                  {tp.monitoring && (
                    <div style={{ fontSize: 11, color: '#aaa' }}>
                      <strong>Monitor:</strong> {tp.monitoring}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* DEFINITIONS TAB */}
          {tab === 'Definitions' && defs && (
            <div>
              <h2 style={{ color: '#ffb74d', marginTop: 0 }}>Key Definitions</h2>

              {/* Key distinguishers */}
              {defs.key_distinguishers && (
                <div style={{ marginBottom: 20 }}>
                  <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Key Distinguishers</h3>
                  {Object.entries(defs.key_distinguishers).map(([k, v]) => (
                    <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ color: '#9ecfff', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                        {k.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Treatment matrix */}
              {defs.treatment_matrix && (
                <div style={{ marginBottom: 20 }}>
                  <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Treatment Matrix</h3>
                  {Object.entries(defs.treatment_matrix).map(([k, v]) => (
                    <div key={k} style={{ background: '#0d1a0d', border: '1px solid #2e7d32', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ color: '#a8e6cf', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                        {k.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}

              {/* Full definitions */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Full Definitions</h3>
              {(defs.definitions || []).map((d, i) => (
                <div key={i} style={{ background: '#12122a', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ color: '#ffb74d', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.6 }}>{d.definition}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
