'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-autoinflammatory-periodic-fever-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'MEFV':     '#b71c1c',  // deep red        -- FMF; short attacks; colchicine curative; AA amyloid
  'TNFRSF1A': '#880e4f',  // dark magenta     -- TRAPS; prolonged attacks >7d; skin patches; etanercept
  'MVK':      '#e65100',  // deep orange      -- HIDS; 3-7d attacks; IgD elevated; Dutch founder
  'NLRP3':    '#1a237e',  // dark indigo      -- CAPS (FCAS/MWS/NOMID); non-pruritic urticaria; canakinumab
  'PSTPIP1':  '#4a148c',  // deep purple      -- PAPA; sterile arthritis + PG + acne; IL-1+TNF
  'NOD2':     '#1b5e20',  // dark green       -- Blau; granulomatous uveitis+arthritis+dermatitis; anti-TNF
  'IL1RN':    '#01579b',  // dark teal        -- DIRA; neonatal periostitis+osteomyelitis; anakinra curative
  'LPIN2':    '#5d4037',  // dark brown       -- Majeed; osteomyelitis+CDA PATHOGNOMONIC; anakinra
};

const GENE_INFO = {
  'MEFV':     { full: 'MEFV-781aa / AR / Pyrin / inflammasome-scaffold / 16p13.3 / FMF-most-common-autoinflammatory / attacks-12-72h / serositis / colchicine-curative / AA-amyloid-preventable / p.M694V-most-severe / ethnic-Turkish-Armenian-Arab-Sephardic', locus: '16p13.3', size: '781 aa / 86 kDa', inh: 'Autosomal recessive (AR, classical); AD low-penetrance variants (E148Q)' },
  'TNFRSF1A': { full: 'TNFRSF1A-455aa / AD / TNFR1 / TNF-receptor-shedding / 12p13.31 / TRAPS / attacks->7-days-PATHOGNOMONIC / migratory-myalgia-skin-patches / etanercept-NOT-infliximab / canakinumab / AA-amyloid-cysteine-mutations', locus: '12p13.31', size: '455 aa / 51 kDa', inh: 'Autosomal dominant (AD; structural cysteine mutations most severe; low-penetrance R92Q/P46L)' },
  'MVK':      { full: 'MVK-396aa / AR / mevalonate-kinase / isoprenoid-pathway / 12q24.11 / HIDS-MKD / attacks-3-7d / cervical-lymphadenopathy / IgD->100-PATHOGNOMONIC / Dutch-p.V377I-founder / vaccine-triggered / canakinumab-approved', locus: '12q24.11', size: '396 aa / 43 kDa', inh: 'Autosomal recessive (AR; p.V377I Dutch/French founder most common ~80% HIDS alleles)' },
  'NLRP3':    { full: 'NLRP3-1036aa / AD-GOF / Cryopyrin / inflammasome-NLR / 1q44 / CAPS-FCAS-MWS-NOMID / non-pruritic-urticaria-PATHOGNOMONIC / cold-triggered-FCAS / deafness-MWS / neonatal-NOMID / canakinumab-approved-ALL-CAPS', locus: '1q44', size: '1036 aa / 118 kDa', inh: 'Autosomal dominant (AD; gain-of-function; de novo or inherited; ~1/360,000 CAPS)' },
  'PSTPIP1':  { full: 'PSTPIP1-416aa / AD / PSTPIP1-coiled-coil / pyrin-binding / 15q24.3 / PAPA / sterile-destructive-arthritis-PATHOGNOMONIC / pyoderma-gangrenosum / acne-conglobata / IL-1+TNF-inhibition / A230T-E250K', locus: '15q24.3', size: '416 aa / 48 kDa', inh: 'Autosomal dominant (AD; PSTPIP1 coiled-coil domain mutations; ultra-rare)' },
  'NOD2':     { full: 'NOD2-1040aa / AD-GOF / NOD2 / NF-kB-granulomatous / 16q12.1 / Blau / granulomatous-uveitis-PATHOGNOMONIC / camptodactyly / lichenoid-dermatitis / NO-lung-involvement / anti-TNF / distinguish-from-sarcoidosis', locus: '16q12.1', size: '1040 aa / 115 kDa', inh: 'Autosomal dominant (AD; GOF; NOD domain mutation; distinct from IBD-associated LOF in LRR domain)' },
  'IL1RN':    { full: 'IL1RN-177aa / AR / IL-1Ra / IL-1-endogenous-inhibitor / 2q14.1 / DIRA / neonatal-sterile-osteomyelitis / periostitis-PATHOGNOMONIC / pustulosis / anakinra-CURATIVE-48-72h / lifelong / p.E77X-Puerto-Rican / p.C80S-Dutch', locus: '2q14.1', size: '177 aa / 25 kDa', inh: 'Autosomal recessive (AR; IL1RN exon 3-5 deletion; founder mutations; <100 patients worldwide)' },
  'LPIN2':    { full: 'LPIN2-986aa / AR / Lipin-2 / phosphatidate-phosphatase / 18p11.31 / Majeed / multifocal-osteomyelitis / CDA-PATHOGNOMONIC-UNIQUE / neutrophilic-dermatosis / anakinra-canakinumab / Middle-Eastern-origin', locus: '18p11.31', size: '986 aa / 111 kDa', inh: 'Autosomal recessive (AR; 18p11.31; ultra-rare; originally described in Jordanian family)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'MEFV',     group: 'Familial Mediterranean Fever (FMF)',         attack_dur: '<72 h',         amyloid: '✓',  il1_resp: '✓✓✓', rx_first: 'Colchicine',         unique: 'SHORT ATTACKS <72h PATHOGNOMONIC; serositis (peritonitis/pleuritis); erysipelas-like erythema lower legs 30-50%; AA amyloid preventable with colchicine; p.M694V homozygous most severe; MOST COMMON autoinflammatory disease' },
  { gene: 'TNFRSF1A', group: 'TRAPS',                                       attack_dur: '>7 days',       amyloid: '✓✓', il1_resp: '✓✓',  rx_first: 'Etanercept/Canakinumab', unique: 'PROLONGED ATTACKS >7 DAYS PATHOGNOMONIC; migratory myalgia + monocyte erythematous skin patches; periorbital oedema; NO infliximab (worsens); cysteine mutations highest amyloid risk ~25-35%' },
  { gene: 'MVK',      group: 'HIDS / Mevalonate Kinase Deficiency (MKD)',   attack_dur: '3-7 days',      amyloid: '✗',  il1_resp: '✓✓',  rx_first: 'Canakinumab',         unique: 'IgD >100 IU/mL PATHOGNOMONIC (>80% HIDS); cervical lymphadenopathy; vaccine-triggered; Dutch p.V377I founder; 3-7 day attacks (between FMF and TRAPS); urine mevalonic acid elevated during crises' },
  { gene: 'NLRP3',    group: 'CAPS (FCAS / MWS / NOMID)',                   attack_dur: 'Continuous',    amyloid: '✓',  il1_resp: '✓✓✓', rx_first: 'Canakinumab',         unique: 'NON-PRURITIC neutrophilic urticaria PATHOGNOMONIC; MWS sensorineural hearing loss; NOMID: neonatal continuous + aseptic meningitis; cold-triggered (FCAS); canakinumab approved all CAPS' },
  { gene: 'PSTPIP1',  group: 'PAPA Syndrome',                               attack_dur: 'Episodic',      amyloid: '✗',  il1_resp: '✓✓',  rx_first: 'Anakinra + Anti-TNF',  unique: 'STERILE destructive arthritis (50-80k WBC/µL sterile synovial fluid) PATHOGNOMONIC + pyoderma gangrenosum (violaceous undermined border) + acne conglobata TRIAD; avoid PG surgery (pathergy)' },
  { gene: 'NOD2',     group: 'Blau Syndrome / Early-Onset Sarcoidosis',     attack_dur: 'Continuous',    amyloid: '✗',  il1_resp: '✓',   rx_first: 'Adalimumab + MTX',     unique: 'Granulomatous UVEITIS (panuveitis blindness risk) + CAMPTODACTYLY boggy arthritis + lichenoid dermatitis TRIAD; NO LUNG INVOLVEMENT (vs sarcoidosis); AD family history; adalimumab for sight-threatening uveitis' },
  { gene: 'IL1RN',    group: 'DIRA (Deficiency of IL-1 Receptor Antagonist)', attack_dur: 'Neonatal continuous', amyloid: '✗', il1_resp: '✓✓✓', rx_first: 'Anakinra (curative)', unique: 'NEONATAL periostitis (radiological periosteal new bone) + sterile multifocal osteomyelitis + pustulosis PATHOGNOMONIC; anakinra CURATIVE within 48-72h; lifelong required; Puerto Rican/Dutch founders' },
  { gene: 'LPIN2',    group: 'Majeed Syndrome',                              attack_dur: 'Recurrent crises', amyloid: '✗', il1_resp: '✓✓', rx_first: 'Anakinra/Canakinumab', unique: 'Multifocal osteomyelitis + CONGENITAL DYSERYTHROPOIETIC ANAEMIA (CDA) PATHOGNOMONIC — CDA is UNIQUE to Majeed in all autoinflammatory diseases; Middle Eastern; anakinra/canakinumab + haematology support' },
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
    MEFV:     'FMF-Serositis-ShortAttacks-lt72h-PATHOGNOMONIC-Colchicine-Curative-AAamyloid',
    TNFRSF1A: 'TRAPS-ProlongedAttacks-gt7days-PATHOGNOMONIC-SkinPatches-Etanercept-Canakinumab',
    MVK:      'HIDS-MKD-IgD-gt100-PATHOGNOMONIC-CervicalLN-Vaccine-Triggered-DutchFounder',
    NLRP3:    'CAPS-NonPruriticUrticaria-PATHOGNOMONIC-MWS-Deafness-NOMID-Neonatal-Canakinumab',
    PSTPIP1:  'PAPA-SterileArthritis-PyodermaGangrenosum-Acne-Triad-PATHOGNOMONIC-IL1-TNF',
    NOD2:     'Blau-GranulomatousUveitis-Camptodactyly-LichenoidDermatitis-Triad-NoLungs',
    IL1RN:    'DIRA-NeonatalPeriostitis-Osteomyelitis-Pustulosis-PATHOGNOMONIC-Anakinra-Curative',
    LPIN2:    'Majeed-MultifocalOsteomyelitis-CDA-PATHOGNOMONIC-Unique-Anakinra-Canakinumab',
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
        <strong>Attack Duration:</strong> {cohort?.attack_duration || '—'}
      </div>
      {cohort && (
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#aaa', marginBottom: 6 }}>
          <span>n={cohort.n}</span>
          <span>M:{cohort.male_n} F:{cohort.female_n}</span>
          <span>Onset median: {cohort.median_onset_months}mo</span>
          <span>IL-1 resp: {(cohort.il1_response_pct * 100).toFixed(0)}%</span>
          <span>Amyloid risk: {(cohort.amyloid_pct * 100).toFixed(0)}%</span>
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
    background: tab === t ? '#b71c1c' : '#1a1a2e',
    color: '#fff', border: 'none', marginRight: 4, fontWeight: tab === t ? 700 : 400,
  });

  return (
    <div style={{ background: '#0d0d1a', minHeight: '100vh', color: '#f0f0f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: 'linear-gradient(135deg,#b71c1c,#880e4f)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f525; Hereditary-Autoinflammatory-Periodic-Fever-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: MEFV-TNFRSF1A-MVK-NLRP3-PSTPIP1-NOD2-IL1RN-LPIN2
            · 320 Patients (8×40) · Seeds 4174-4181
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.85 }}>
            FMF (MEFV-colchicine) · TRAPS (TNFRSF1A-etanercept/canakinumab-NOT-infliximab)
            · HIDS/MKD (MVK-IgD-canakinumab) · CAPS (NLRP3-non-pruritic-urticaria-canakinumab)
            · PAPA (PSTPIP1-PG-IL1+TNF) · Blau (NOD2-granulomatous-uveitis-anti-TNF)
            · DIRA (IL1RN-neonatal-periostitis-anakinra-CURATIVE) · Majeed (LPIN2-CDA-PATHOGNOMONIC)
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
              <h2 style={{ color: '#ef9a9a', marginTop: 0 }}>Atlas Overview</h2>
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
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds || '4174-4181'}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 200 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>IL-1 Approved (3 diseases)</div>
                  <div style={{ color: '#ce93d8', fontSize: 14, fontWeight: 700 }}>TRAPS · HIDS · CAPS</div>
                </div>
              </div>

              {/* Attack duration spectrum */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  Attack Duration Diagnostic Spectrum
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#2d0000', borderRadius: 6, fontSize: 11, color: '#ff7043' }}>
                    <strong>&lt;72 h:</strong> MEFV/FMF — colchicine curative
                  </div>
                  <div style={{ padding: '6px 10px', background: '#1a1a00', borderRadius: 6, fontSize: 11, color: '#ffd740' }}>
                    <strong>3-7 d:</strong> MVK/HIDS — IgD&gt;100; canakinumab
                  </div>
                  <div style={{ padding: '6px 10px', background: '#1a003d', borderRadius: 6, fontSize: 11, color: '#ea80fc' }}>
                    <strong>&gt;7 d:</strong> TNFRSF1A/TRAPS — etanercept/canakinumab; NO infliximab
                  </div>
                  <div style={{ padding: '6px 10px', background: '#002d1a', borderRadius: 6, fontSize: 11, color: '#69f0ae' }}>
                    <strong>Continuous:</strong> NLRP3/CAPS · NOD2/Blau · IL1RN/DIRA · LPIN2/Majeed
                  </div>
                </div>
              </div>

              {/* IL-1 therapeutic panel */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#ffcc80', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  IL-1 Inhibition — Regulatory Approvals &amp; Hierarchy
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#002d1a', borderRadius: 6, fontSize: 11, color: '#69f0ae' }}>
                    <strong>Canakinumab APPROVED:</strong> TRAPS · HIDS/MKD · ALL CAPS (FCAS/MWS/NOMID)
                  </div>
                  <div style={{ padding: '6px 10px', background: '#002d00', borderRadius: 6, fontSize: 11, color: '#a5d6a7' }}>
                    <strong>Anakinra CURATIVE:</strong> IL1RN/DIRA (lifelong); also PSTPIP1/PAPA + Majeed/LPIN2
                  </div>
                  <div style={{ padding: '6px 10px', background: '#2d0000', borderRadius: 6, fontSize: 11, color: '#ff5252' }}>
                    <strong>Colchicine FIRST-LINE:</strong> MEFV/FMF only; prevents AA amyloid; 90% frequency reduction
                  </div>
                  <div style={{ padding: '6px 10px', background: '#1a1a00', borderRadius: 6, fontSize: 11, color: '#ffd740' }}>
                    <strong>Anti-TNF (NOT IL-1):</strong> NOD2/Blau — adalimumab for granulomatous uveitis
                  </div>
                </div>
              </div>

              {/* Pathognomonic highlights */}
              {overview.pathognomonic_highlights && (
                <div style={{ marginBottom: 16 }}>
                  <h3 style={{ color: '#ef9a9a', fontSize: 14 }}>Pathognomonic Highlights</h3>
                  <ul style={{ paddingLeft: 20, margin: 0 }}>
                    {overview.pathognomonic_highlights.map((f, i) => (
                      <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 6, lineHeight: 1.5 }}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Gene cards */}
              <h3 style={{ color: '#ef9a9a', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.gene_cohorts || []).map((c) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#ef9a9a', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Disease','aa','Locus','kDa','Inheritance','Attack Duration','AA Amyloid','IL-1 Resp','Key Rx','Pathognomonic Feature'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#ef9a9a', borderBottom: '1px solid #333' }}>{h}</th>
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
                        <td style={{ padding: '6px', color: row.attack_dur === '>7 days' ? '#ea80fc' : row.attack_dur === '<72 h' ? '#ff7043' : row.attack_dur === '3-7 days' ? '#ffd740' : '#69f0ae', fontWeight: 600 }}>{row.attack_dur}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.amyloid === '✓✓' ? '#f55' : row.amyloid === '✓' ? '#ff8' : '#888' }}>{row.amyloid}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.il1_resp === '✓✓✓' ? '#69f0ae' : row.il1_resp === '✓✓' ? '#a5d6a7' : '#ccc' }}>{row.il1_resp}</td>
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
              <h2 style={{ color: '#ef9a9a', marginTop: 0 }}>Autoinflammatory Pathway Groups</h2>

              {/* HLH/Amyloid risk */}
              {breakdown.hlh_risk_table && (
                <div style={{ background: '#2d1a00', border: '1px solid #ff8f00', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                  <div style={{ color: '#ffb300', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>AA Amyloid &amp; Systemic Complication Risk by Gene</div>
                  {Object.entries(breakdown.hlh_risk_table).map(([gene, risk]) => (
                    <div key={gene} style={{ display: 'flex', gap: 10, marginBottom: 4, fontSize: 11 }}>
                      <Badge text={gene} color={GENE_COLORS[gene] || '#555'} />
                      <span style={{ color: '#ccc' }}>{risk}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Pathway groups */}
              {Object.entries(breakdown.pathway_groups || {}).map(([grp, data]) => (
                <div key={grp} style={{ background: '#12122a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <span style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 13 }}>
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
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Attack Duration Cascade Testing Algorithm</h3>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {(breakdown.attack_duration_cascade || []).map((step, i) => (
                  <li key={i} style={{ fontSize: 11, color: '#ccc', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>

              {/* IL-1 treatment spectrum */}
              {breakdown.il1_treatment_spectrum && (
                <div style={{ marginTop: 20 }}>
                  <h3 style={{ color: '#69f0ae', fontSize: 13, marginBottom: 10 }}>IL-1 Treatment Spectrum</h3>
                  {Object.entries(breakdown.il1_treatment_spectrum).map(([k, v]) => (
                    <div key={k} style={{ background: '#002d1a', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                      <div style={{ color: '#69f0ae', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                        {k.replace(/_/g, ' ').toUpperCase()}
                      </div>
                      <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* TREATMENT TAB */}
          {tab === 'Treatment' && breakdown && (
            <div>
              <h2 style={{ color: '#ef9a9a', marginTop: 0 }}>Treatment Pathways</h2>
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
              <h2 style={{ color: '#ef9a9a', marginTop: 0 }}>Key Definitions</h2>

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
                  <div style={{ color: '#ef9a9a', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
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
