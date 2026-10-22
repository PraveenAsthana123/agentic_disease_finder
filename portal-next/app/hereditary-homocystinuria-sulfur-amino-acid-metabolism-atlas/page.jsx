'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-homocystinuria-sulfur-amino-acid-metabolism-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'CBS':   '#4a148c',  // deep purple       -- Classical Homocystinuria; ectopia lentis PATHOGNOMONIC; B6-trial MANDATORY
  'MTHFR': '#1a237e',  // dark indigo        -- Severe AR deficiency; remethylation; methionine LOW; NOT C677T polymorphism
  'MTR':   '#006064',  // dark cyan          -- cblG; methionine synthase; megaloblastic anemia; OHCbl
  'MTRR':  '#004d40',  // dark teal          -- cblE; reactivates MTR cobalamin; biochemically = cblG
  'MAT1A': '#e65100',  // deep orange        -- Isolated hypermethioninemia; mostly benign; NBS elevated met
  'GNMT':  '#bf360c',  // deep red-orange    -- Hypermethioninemia + AFP elevated PATHOGNOMONIC; liver disease
  'AHCY':  '#827717',  // dark yellow-green  -- SAH hydrolase; myopathy DISTINCTIVE; SAH accumulates
  'CTH':   '#1b5e20',  // dark green         -- Cystathioninuria; MOSTLY BENIGN; B6 responsive
};

const GENE_INFO = {
  'CBS':   { full: 'CBS-551aa / AR / Cystathionine-beta-synthase / Transsulfuration / 21q22.3 / Classical-Homocystinuria / Ectopia-lentis-DOWNWARD-PATHOGNOMONIC / Marfanoid / VTE-young-age / B6-trial-MANDATORY-50pct-responsive / Betaine-3-6g-BID / Low-methionine-diet', locus: '21q22.3', size: '551 aa / 63 kDa', inh: 'Autosomal recessive (AR; PLP-dependent; B6-trial mandatory; 50% B6-responsive; betaine bypass)' },
  'MTHFR': { full: 'MTHFR-656aa / AR-SEVERE / Methylenetetrahydrofolate-reductase / Remethylation / 1p36.22 / Severe-MTHFR-deficiency / Methionine-LOW-DDx-CBS / Homocysteine-HIGH / NO-MMA / C677T-A1298C-POLYMORPHISMS-NOT-DISEASE / 5-methylTHF-betaine-OHCbl', locus: '1p36.22', size: '656 aa / 74 kDa', inh: 'Autosomal recessive (AR — SEVERE disease only; NOT C677T/A1298C polymorphisms; methionine LOW distinguishes from CBS)' },
  'MTR':   { full: 'MTR-1265aa / AR / Methionine-synthase / Remethylation / 1q43 / cblG / Megaloblastic-anemia / Homocysteine-HIGH / Methionine-LOW / NO-MMA-DDx-cblC-cblD / OHCbl-betaine-methylfolate / N2O-ABSOLUTELY-CI', locus: '1q43', size: '1265 aa / 140 kDa', inh: 'Autosomal recessive (AR; cblG; cobalamin-dependent; megaloblastic anemia + homocystinuria WITHOUT MMA)' },
  'MTRR':  { full: 'MTRR-698aa / AR / Methionine-synthase-reductase / Remethylation / 5p15.31 / cblE / Reactivates-MTR-cobalamin / Biochemically-IDENTICAL-cblG / p.I22M-POLYMORPHISM-NOT-pathogenic / OHCbl-betaine / N2O-ABSOLUTELY-CI', locus: '5p15.31', size: '698 aa / 78 kDa', inh: 'Autosomal recessive (AR; cblE; reactivates oxidised MTR methylcobalamin; indistinguishable from cblG/MTR)' },
  'MAT1A': { full: 'MAT1A-395aa / AR-AD / Methionine-adenosyltransferase-I-III / SAM-synthesis / 10q22.3 / Isolated-hypermethioninemia / Methionine-HIGH-Homocysteine-NORMAL-DDx-CBS / Mostly-BENIGN / Low-met-diet-if-symptomatic / White-matter-MRI', locus: '10q22.3', size: '395 aa / 43 kDa', inh: 'Autosomal recessive (AR; liver-specific MAT; methionine HIGH + homocysteine NORMAL; mostly benign)' },
  'GNMT':  { full: 'GNMT-294aa / AR / Glycine-N-methyltransferase / SAM-catabolism / 6p12.1 / Hypermethioninemia / AFP-CHRONICALLY-ELEVATED-PATHOGNOMONIC / Liver-disease / SAM-HIGH / Homocysteine-NORMAL / Very-rare-lt30-cases', locus: '6p12.1', size: '294 aa / 33 kDa', inh: 'Autosomal recessive (AR; SAM regulator; AFP chronically elevated PATHOGNOMONIC; very rare <30 cases worldwide)' },
  'AHCY':  { full: 'AHCY-432aa / AR / SAH-hydrolase / SAH-SAM-axis / 20q11.22 / SAH-accumulates / Global-methylation-failure / Myopathy-CK-ELEVATED-DISTINCTIVE / Developmental-delay / Hepatopathy / Creatine-supp-addresses-GAMT-inhibition', locus: '20q11.22', size: '432 aa / 48 kDa', inh: 'Autosomal recessive (AR; SAH hydrolase; SAH accumulates → global methylation failure; myopathy DISTINCTIVE)' },
  'CTH':   { full: 'CTH-405aa / AR / Cystathionine-gamma-lyase / Transsulfuration / 1p31.3 / Cystathioninuria / Cystathionine-HIGH / Homocysteine-NORMAL / MOSTLY-BENIGN / B6-responsive / Secondary-causes-DDx-B6-deficiency-liver-neuroblastoma', locus: '1p31.3', size: '405 aa / 44 kDa', inh: 'Autosomal recessive (AR; PLP-dependent; cystathioninuria mostly BENIGN; B6 normalises metabolites)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CBS',   group: 'Classical Homocystinuria (CBS Deficiency)',       met_dir: '↑↑↑',  hcy_dir: '↑↑↑',  mma: '—',  nbs: '✓',  b6: '50%', rx_first: 'Betaine + low-met diet + B6 trial', unique: 'Ectopia lentis DOWNWARD (not Marfan UPWARD); VTE at young age; marfanoid + osteoporosis; B6 trial MANDATORY; betaine works regardless; met HIGH distinguishes from remethylation defects' },
  { gene: 'MTHFR', group: 'Severe MTHFR Deficiency (NOT C677T polymorphism)', met_dir: '↓',    hcy_dir: '↑↑',   mma: '—',  nbs: '✗',  b6: '✗', rx_first: 'Methylfolate + betaine + OHCbl', unique: 'C677T/A1298C = COMMON POLYMORPHISMS NOT DISEASE; severe AR disease rare; methionine LOW = critical DDx from CBS; no megaloblastic anemia (different from MTR/MTRR); cerebral calcifications; psychosis' },
  { gene: 'MTR',   group: 'cblG (Methionine Synthase Deficiency)',            met_dir: '↓',    hcy_dir: '↑↑',   mma: '—',  nbs: '✗',  b6: '✗', rx_first: 'Hydroxocobalamin IM + betaine', unique: 'Megaloblastic anemia PROMINENT (methylfolate trap); hcy HIGH + met LOW + MMA ABSENT (DDx cblC/cblD which have MMA); N2O ABSOLUTELY CI; OHCbl preferred over CNCbl; cblG designation' },
  { gene: 'MTRR',  group: 'cblE (Methionine Synthase Reductase Deficiency)',  met_dir: '↓',    hcy_dir: '↑↑',   mma: '—',  nbs: '✗',  b6: '✗', rx_first: 'Hydroxocobalamin IM + betaine', unique: 'BIOCHEMICALLY IDENTICAL to cblG/MTR; differentiate by complementation or gene sequencing; p.I22M POLYMORPHISM NOT pathogenic; treatment identical to cblG; cblE designation; N2O ABSOLUTELY CI' },
  { gene: 'MAT1A', group: 'Isolated Hypermethioninemia (MAT I/III Deficiency)', met_dir: '↑↑↑', hcy_dir: '→',   mma: '—',  nbs: '✓',  b6: '—', rx_first: 'Low-met diet if symptomatic; observe if not', unique: 'Methionine very HIGH + homocysteine NORMAL = DDx from CBS; MOSTLY BENIGN (most asymptomatic); NBS detects elevated methionine; white matter changes on MRI in minority; liver-specific MAT isoform' },
  { gene: 'GNMT',  group: 'GNMT Deficiency (Hypermethioninemia + Hepatopathy)', met_dir: '↑↑↑', hcy_dir: '→',  mma: '—',  nbs: '✓',  b6: '—', rx_first: 'Low-met diet; liver monitoring', unique: 'AFP CHRONICALLY ELEVATED PATHOGNOMONIC; <30 cases worldwide; SAM very high; liver disease + elevated transaminases; AFP not from HCC — from childhood; met HIGH + hcy NORMAL + AFP elevated = GNMT' },
  { gene: 'AHCY',  group: 'AHCY Deficiency (SAH Hydrolase Deficiency)',       met_dir: '↑↑',  hcy_dir: '↑',    mma: '—',  nbs: '✓',  b6: '—', rx_first: 'Low-met diet + creatine supplementation', unique: 'MYOPATHY (CK elevated) DISTINCTIVE — no other hypermethioninemia causes myopathy; SAH accumulates → global methylation failure; GAMT inhibition → creatine deficiency → myopathy; SAH:SAM ratio >1 diagnostic' },
  { gene: 'CTH',   group: 'Cystathioninuria (CTH Deficiency, Mostly Benign)', met_dir: '→',   hcy_dir: '→',    mma: '—',  nbs: '±',  b6: '60%', rx_first: 'No treatment needed (reassure)', unique: 'MOSTLY BENIGN — cystathionine HIGH but homocysteine NORMAL; B6 normalises metabolites biochemically; secondary causes (B6 deficiency, liver disease, neuroblastoma) must be excluded; reassure families; no dietary restriction needed' },
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
    CBS:   'Classical-Homocystinuria-EctopiaLentis-PATHOGNOMONIC-Marfanoid-VTE-B6-50pct-Betaine',
    MTHFR: 'Severe-MTHFR-Remethylation-MetLOW-HcyHIGH-C677T-NOT-Disease',
    MTR:   'cblG-MetSynthase-MegaloblasticAnemia-HcyHIGH-MMA-ABSENT-OHCbl',
    MTRR:  'cblE-MetSynthaseReductase-IDENTICAL-cblG-p.I22M-NOT-pathogenic-OHCbl',
    MAT1A: 'IsolatedHypermethioninemia-MetHIGH-HcyNORMAL-MostlyBenign',
    GNMT:  'GNMT-MetHIGH-AFP-ELEVATED-PATHOGNOMONIC-LiverDisease-VeryRare',
    AHCY:  'SAH-Hydrolase-MetHIGH-Myopathy-DISTINCTIVE-SAH-GlobalMethylationFailure',
    CTH:   'Cystathioninuria-MOSTLY-BENIGN-B6-Responsive-HcyNORMAL',
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
        <strong>Methionine:</strong> {cohort?.methionine_direction || '—'}
        &nbsp;&nbsp;<strong>Homocysteine:</strong> {cohort?.homocysteine_direction || '—'}
        &nbsp;&nbsp;<strong>MMA:</strong> {cohort?.mma_status || '—'}
      </div>
      {cohort && (
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#aaa', marginBottom: 6 }}>
          <span>n={cohort.n}</span>
          <span>M:{cohort.male_n} F:{cohort.female_n}</span>
          <span>Onset median: {cohort.median_onset_months}mo</span>
          <span>NBS: {(cohort.nbs_detected_pct * 100).toFixed(0)}%</span>
          <span>B6 resp: {cohort.b6_responsive_pct > 0 ? (cohort.b6_responsive_pct * 100).toFixed(0) + '%' : 'N/A'}</span>
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
    background: tab === t ? '#4a148c' : '#1a1a2e',
    color: '#fff', border: 'none', marginRight: 4, fontWeight: tab === t ? 700 : 400,
  });

  return (
    <div style={{ background: '#0d0d1a', minHeight: '100vh', color: '#f0f0f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ background: 'linear-gradient(135deg,#4a148c,#1a237e)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f9ec; Hereditary-Homocystinuria-Sulfur-Amino-Acid-Metabolism-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: CBS-MTHFR-MTR-MTRR-MAT1A-GNMT-AHCY-CTH
            · 320 Patients (8×40) · Seeds 4190-4197
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.85 }}>
            CBS-551aa-21q22.3-AR-EctopiaLentis-PATHOGNOMONIC-Marfanoid-VTE-B6-50pct-Betaine ·
            MTHFR-656aa-1p36.22-AR-SevereDeficiency-MetLOW-NOT-C677T-Polymorphism ·
            MTR-1265aa-1q43-AR-cblG-MegaloblasticAnemia-HcyHIGH-MMA-ABSENT-OHCbl ·
            MTRR-698aa-5p15.31-AR-cblE-IDENTICAL-cblG-p.I22M-NOT-pathogenic ·
            MAT1A-395aa-10q22.3-AR-IsolatedHypermethioninemia-MostlyBenign ·
            GNMT-294aa-6p12.1-AR-AFP-ELEVATED-PATHOGNOMONIC-LiverDisease ·
            AHCY-432aa-20q11.22-AR-Myopathy-DISTINCTIVE-SAH-GlobalMethylation ·
            CTH-405aa-1p31.3-AR-Cystathioninuria-MOSTLY-BENIGN-B6-Responsive
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
              <h2 style={{ color: '#ce93d8', marginTop: 0 }}>Atlas Overview</h2>
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
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds || '4190-4197'}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 200 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Pathway Covered</div>
                  <div style={{ color: '#ce93d8', fontSize: 12, fontWeight: 700 }}>Transsulfuration · Remethylation · SAM catabolism</div>
                </div>
              </div>

              {/* Methionine direction spectrum — THE KEY DDx */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#ce93d8', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  Methionine Direction — PRIMARY Diagnostic DDx Split
                </div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#2d1000', borderRadius: 6, fontSize: 11, color: '#ff7043' }}>
                    <strong>↑↑↑ Met HIGH + Hcy HIGH:</strong> CBS (ectopia lentis + VTE + marfanoid) · AHCY (myopathy)
                  </div>
                  <div style={{ padding: '6px 10px', background: '#2d1000', borderRadius: 6, fontSize: 11, color: '#ffb74d' }}>
                    <strong>↑↑↑ Met HIGH + Hcy NORMAL:</strong> MAT1A (mostly benign) · GNMT (AFP elevated PATHOGNOMONIC)
                  </div>
                  <div style={{ padding: '6px 10px', background: '#001a2d', borderRadius: 6, fontSize: 11, color: '#64b5f6' }}>
                    <strong>↓ Met LOW + Hcy HIGH:</strong> MTHFR/MTR(cblG)/MTRR(cblE) — remethylation defects; MMA ABSENT
                  </div>
                  <div style={{ padding: '6px 10px', background: '#1a2d00', borderRadius: 6, fontSize: 11, color: '#aed581' }}>
                    <strong>→ Met NORMAL + Hcy NORMAL:</strong> CTH — cystathionine HIGH but BENIGN; no hcy elevation
                  </div>
                </div>
              </div>

              {/* Critical N2O warning */}
              <div style={{ background: '#2d0000', border: '1px solid #f44336', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#ff5252', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  ⚠ N2O (Nitrous Oxide) — ABSOLUTELY CONTRAINDICATED in All Homocystinurias
                </div>
                <div style={{ fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
                  N2O irreversibly oxidises methylcobalamin within MTR (methionine synthase) active site →
                  permanent MTR inactivation → acute homocysteine surge → CNS decompensation.
                  Affects MTR/MTRR (cblG/cblE) most severely; also exacerbates CBS.
                  Single anaesthetic exposure = life-threatening crisis. Alert all surgical/anaesthetic teams.
                </div>
              </div>

              {/* MTHFR C677T polymorphism alert */}
              <div style={{ background: '#001a2d', border: '1px solid #0288d1', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#29b6f6', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  Critical: MTHFR C677T / A1298C are POLYMORPHISMS — NOT Clinical Homocystinuria Disease
                </div>
                <div style={{ fontSize: 11, color: '#ccc', lineHeight: 1.6 }}>
                  C677T homozygosity (~10% European) mildly elevates tHcy (12-18 µmol/L) but does NOT cause
                  clinical homocystinuria syndrome. Severe AR MTHFR deficiency (both alleles strongly damaging)
                  is rare and distinct. Do NOT diagnose &quot;MTHFR disorder&quot; based on C677T/A1298C alone.
                  Similarly, MTRR p.I22M is a common polymorphism (~20% European) — NOT pathogenic alone.
                </div>
              </div>

              {/* Gene cards */}
              <h3 style={{ color: '#ce93d8', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.gene_cohorts || []).map((c) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#ce93d8', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Disease','aa','Locus','kDa','Inh','Met','Hcy','MMA','NBS','B6 Resp','Key Rx','Pathognomonic'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#ce93d8', borderBottom: '1px solid #333' }}>{h}</th>
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
                        <td style={{ padding: '6px', color: row.met_dir.startsWith('↑') ? '#ff7043' : row.met_dir.startsWith('↓') ? '#64b5f6' : '#888', fontWeight: 700 }}>{row.met_dir}</td>
                        <td style={{ padding: '6px', color: row.hcy_dir.startsWith('↑') ? '#ff5252' : '#888', fontWeight: 700 }}>{row.hcy_dir}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: '#888' }}>{row.mma}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.nbs === '✓' ? '#69f0ae' : row.nbs === '±' ? '#ffd740' : '#f44336' }}>{row.nbs}</td>
                        <td style={{ padding: '6px', color: row.b6 === '50%' || row.b6 === '60%' ? '#ffd740' : '#888' }}>{row.b6}</td>
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
              <h2 style={{ color: '#ce93d8', marginTop: 0 }}>Sulfur Amino Acid Pathway Groups</h2>

              {Object.entries(breakdown.pathway_groups || {}).map(([grp, data]) => (
                <div key={grp} style={{ background: '#12122a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                    <span style={{ color: '#ce93d8', fontWeight: 700, fontSize: 13 }}>
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

              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Homocysteine / Methionine Diagnostic Cascade</h3>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {(breakdown.hcy_methionine_diagnostic_cascade || []).map((step, i) => (
                  <li key={i} style={{ fontSize: 11, color: '#ccc', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>

              {breakdown.mma_absent_rule && (
                <div style={{ marginTop: 16, background: '#001a2d', borderRadius: 8, padding: 12 }}>
                  <div style={{ color: '#29b6f6', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>MMA Absent Rule</div>
                  <div style={{ color: '#ccc', fontSize: 11 }}>{breakdown.mma_absent_rule}</div>
                </div>
              )}
            </div>
          )}

          {/* TREATMENT TAB */}
          {tab === 'Treatment' && breakdown && (
            <div>
              <h2 style={{ color: '#ce93d8', marginTop: 0 }}>Treatment Pathways</h2>
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
              {breakdown.n2o_warning && (
                <div style={{ background: '#2d0000', border: '1px solid #f44336', borderRadius: 8, padding: 12, marginTop: 16 }}>
                  <div style={{ color: '#ff5252', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>⚠ N2O Warning</div>
                  <div style={{ color: '#ccc', fontSize: 11 }}>{breakdown.n2o_warning}</div>
                </div>
              )}
            </div>
          )}

          {/* DEFINITIONS TAB */}
          {tab === 'Definitions' && defs && (
            <div>
              <h2 style={{ color: '#ce93d8', marginTop: 0 }}>Key Definitions</h2>

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

              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Full Definitions</h3>
              {(defs.definitions || []).map((d, i) => (
                <div key={i} style={{ background: '#12122a', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ color: '#ce93d8', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
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
