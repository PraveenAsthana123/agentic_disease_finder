'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cgd-neutrophil-oxidative-burst-defect-atlas';
const TABS = ['Overview', 'Gene Table', 'Diagnostic Map', 'Definitions'];

const GENE_COLORS = {
  'CYBB':   '#b71c1c',  // deep red         -- X-CGD; gp91phox absent; DHR absent PATHOGNOMONIC; IFN-gamma; HSCT
  'NCF1':   '#880e4f',  // dark crimson      -- p47-CGD AR; ΔGT deletion most common AR-CGD; cytochrome b558 present
  'CYBA':   '#c62828',  // medium red        -- p22-CGD AR; gp91phox also absent (mutual destabilization); AR female affected
  'NCF2':   '#ad1457',  // dark pink-red     -- p67-CGD AR; activation domain absent; cytochrome b558 present; severe
  'NCF4':   '#6a1b9a',  // deep purple       -- p40-CGD AR; rarest; IBD-dominant; DHR may be NORMAL; phagosomal assay
  'CYBC1':  '#283593',  // dark indigo       -- EROS chaperone; ultra-rare 2019; gp91+p22 absent; standard panel negative
  'RAC2':   '#e65100',  // deep orange       -- NIS; GTPase; DHR intermediate; combined PMN defects; HSCT often needed
  'MPO':    '#1b5e20',  // dark green        -- MPO deficiency; DHR NORMAL; mildest; Candida+DM only; auto-analyzer flag
};

const GENE_INFO = {
  'CYBB':   { full: 'CYBB-570aa / XLR / gp91phox / cytochrome-b-558-heavy-chain / Xq21.1 / X-CGD-Most-Common-65pct / DHR-ABSENT-PATHOGNOMONIC / Liver-Abscess-CGD-Hallmark / Aspergillus-Most-Lethal / Carrier-Female-BIMODAL-DHR / IFN-gamma-Prophylaxis / HSCT-Curative / Gene-Therapy-Trials', locus: 'Xq21.1', size: '570 aa / 91 kDa', inh: 'X-linked recessive (XLR; hemizygous males; carrier females bimodal DHR)' },
  'NCF1':   { full: 'NCF1-390aa / AR / p47phox / cytosol-organizer / 7q11.23 / p47-CGD-25pct / ΔGT-deletion-codon-75_76-MOST-COMMON-AR-CGD-mutation / cytochrome-b558-PRESENT / milder-than-X-CGD / equal-sex / TMP-SMX-itraconazole', locus: '7q11.23', size: '390 aa / 47 kDa', inh: 'Autosomal recessive (AR; 7q11.23; both sexes; ΔGT founder deletion)' },
  'CYBA':   { full: 'CYBA-195aa / AR / p22phox / cytochrome-b-558-light-chain / 16q24.2 / p22-CGD-5pct / gp91phox-also-absent-mutual-destabilization / carrier-female-NORMAL-DHR-DDx-XCGD / CYBA-sequencing / same-severity-as-XCGD', locus: '16q24.2', size: '195 aa / 22 kDa', inh: 'Autosomal recessive (AR; 16q24.2; both sexes; ~5% of CGD)' },
  'NCF2':   { full: 'NCF2-526aa / AR / p67phox / activation-domain-FAD-transfer / 1q25.3 / p67-CGD-5pct / cytochrome-b558-PRESENT / p67phox-Western-blot-absent / severe-X-CGD-like / RAC2-GTP-binding-TPR-motif-absent', locus: '1q25.3', size: '526 aa / 67 kDa', inh: 'Autosomal recessive (AR; 1q25.3; both sexes; ~5% of CGD)' },
  'NCF4':   { full: 'NCF4-339aa / AR / p40phox / PX-domain-PI3P-phagosomal-targeting / 22q13.1 / p40-CGD-1-2pct-RAREST / IBD-COLITIS-DOMINANT / DHR-MAY-BE-NORMAL-diagnostic-pitfall / phagosomal-oxidase-assay-latex-bead-required / HSCT-rarely-needed', locus: '22q13.1', size: '339 aa / 40 kDa', inh: 'Autosomal recessive (AR; 22q13.1; both sexes; rarest canonical CGD; 1-2%)' },
  'CYBC1':  { full: 'CYBC1-291aa / AR / EROS-chaperone / ER-gp91phox-p22phox-maturation / 1p22.1 / Ultra-rare-2019-discovery / gp91phox+p22phox-ABSENT-despite-CYBB+CYBA-genes-normal / standard-CGD-panel-NEGATIVE / whole-exome-required / X-CGD-like-severity', locus: '1p22.1', size: '291 aa / 33 kDa', inh: 'Autosomal recessive (AR; 1p22.1; both sexes; ultra-rare <10 families; discovered 2019)' },
  'RAC2':   { full: 'RAC2-192aa / AD-GOF-p.D57N-dominant-negative OR AR-LOF / Rac2-GTPase / p67phox-TPR-binding / 22q13.1 / NIS-Neutrophil-Immunodeficiency-Syndrome / DHR-INTERMEDIATE-not-absent / combined-PMN-defects-oxidative+chemotaxis+phagocytosis / HSCT-often-required / de-novo-no-family-history', locus: '22q13.1', size: '192 aa / 21 kDa', inh: 'AD-GOF (de novo p.D57N dominant negative) or AR-LOF (biallelic; 22q13.1)' },
  'MPO':    { full: 'MPO-745aa / AR / Myeloperoxidase / H2O2-to-HOCl-conversion / 17q22 / Most-COMMON-phagocyte-enzyme-defect-1in4000 / DHR-NORMAL-NOX2-intact / Usually-ASYMPTOMATIC / Candida-risk-only-with-DIABETES / p.R569W-most-common / Sysmex-auto-analyzer-flag / NO-HSCT-NO-IFN-gamma', locus: '17q22', size: '745 aa / 84 kDa', inh: 'Autosomal recessive (AR; 17q22; both sexes; 1/4000 prevalence; usually benign)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CYBB',  group: 'X-CGD (gp91phox absent)',     dhr: '✗✗✗', cytb558: '✗✗✗', ibd: '✓✓',  sex: 'M>>F', hsct_urg: '✓✓✓', unique: 'DHR ABSENT; CARRIER FEMALE BIMODAL DHR PATHOGNOMONIC (X-linked); LIVER ABSCESS hallmark; ASPERGILLUS most lethal; IFN-γ 70% infection reduction; HSCT curative; gene therapy trials; CYBB sequencing.' },
  { gene: 'NCF1',  group: 'p47-CGD (cytosol organizer)', dhr: '✗✗✗', cytb558: '✓',   ibd: '✓',   sex: 'M=F', hsct_urg: '✓',   unique: 'DHR ABSENT; cytochrome b-558 PRESENT (membrane intact); ΔGT DELETION MOST COMMON AR-CGD mutation (70-80% of NCF1 alleles); milder than X-CGD; equal sex; TMP-SMX + itraconazole; HSCT threshold higher (milder phenotype).' },
  { gene: 'CYBA',  group: 'p22-CGD (light chain absent)', dhr: '✗✗✗', cytb558: '✗✗✗', ibd: '✓',   sex: 'M=F', hsct_urg: '✓✓',  unique: 'DHR ABSENT; cytochrome b-558 ABSENT (gp91phox also absent — mutual destabilization); AR — FEMALE AFFECTED (DDx from X-CGD); carrier females NORMAL DHR (not bimodal); same severity as X-CGD; CYBA sequencing.' },
  { gene: 'NCF2',  group: 'p67-CGD (activation domain)', dhr: '✗✗✗', cytb558: '✓',   ibd: '✓',   sex: 'M=F', hsct_urg: '✓✓',  unique: 'DHR ABSENT; cytochrome b-558 PRESENT (activation impossible despite intact membrane); severe X-CGD-like phenotype; p67phox Western blot absent; Rac2-GTP cannot bind absent NCF2 TPR repeat; NCF2 sequencing.' },
  { gene: 'NCF4',  group: 'p40-CGD (phagosomal target)', dhr: '±',    cytb558: '✓',   ibd: '✓✓✓', sex: 'M=F', hsct_urg: '✗',   unique: 'DHR MAY BE NORMAL (MISSED ON STANDARD CGD SCREENING); IBD DOMINANT phenotype (Crohn\'s-like colitis often primary presentation); phagosomal oxidase assay (latex bead) absent; HSCT rarely needed; rarest canonical CGD gene.' },
  { gene: 'CYBC1', group: 'EROS chaperone (2019)',       dhr: '✗✗✗', cytb558: '✗✗✗', ibd: '✓',   sex: 'M=F', hsct_urg: '✓✓✓', unique: 'DHR ABSENT; cytochrome b-558 ABSENT (gp91phox+p22phox degraded in ER without EROS chaperone); ALL STANDARD CGD GENES NORMAL → WHOLE EXOME REQUIRED; X-CGD-like severity; 2019 discovery; AR.' },
  { gene: 'RAC2',  group: 'NIS (GTPase dysfunction)',    dhr: '✓±',   cytb558: '✓',   ibd: '±',   sex: 'M=F', hsct_urg: '✓✓',  unique: 'DHR INTERMEDIATE (partial burst — not absent; may be missed on CGD screening); COMBINED PMN DEFECTS (oxidative + chemotaxis + phagocytosis all impaired); de novo p.D57N dominant negative; HSCT often required; RAC2 sequencing.' },
  { gene: 'MPO',   group: 'MPO def (HOCl absent)',       dhr: '✓✓✓', cytb558: '✓',   ibd: '✗',   sex: 'M=F', hsct_urg: '✗',   unique: 'DHR NORMAL (NOX2 intact — H2O2 produced; DISTINGUISHES from CGD); MPO stain ABSENT; USUALLY ASYMPTOMATIC; Candida risk only with DIABETES MELLITUS; auto-analyzer (Sysmex) MPO flag = incidental diagnosis; NO HSCT, NO IFN-γ.' },
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
  const isCYBB  = gene === 'CYBB';
  const isNCF1  = gene === 'NCF1';
  const isCYBA  = gene === 'CYBA';
  const isNCF2  = gene === 'NCF2';
  const isNCF4  = gene === 'NCF4';
  const isCYBC1 = gene === 'CYBC1';
  const isRAC2  = gene === 'RAC2';
  const isMPO   = gene === 'MPO';
  const groupLabel = isCYBB  ? 'X-CGD (XLR-gp91phox-Absent-DHR-ABSENT-Carrier-Female-BIMODAL-DHR-PATHOGNOMONIC-IFN-gamma-HSCT)'
    : isNCF1  ? 'p47-CGD (AR-Cytosol-Organizer-ΔGT-Deletion-Most-Common-AR-Mutation-Cytochrome-b558-PRESENT)'
    : isCYBA  ? 'p22-CGD (AR-Light-Chain-Absent-gp91phox-Also-Absent-Female-Affected-DDx-XCGD)'
    : isNCF2  ? 'p67-CGD (AR-Activation-Domain-Absent-Cytochrome-b558-PRESENT-Severe-X-CGD-Like)'
    : isNCF4  ? 'p40-CGD (AR-Phagosomal-Targeting-Failure-IBD-DOMINANT-DHR-May-Be-NORMAL-Pitfall)'
    : isCYBC1 ? 'EROS-CGD (AR-Chaperone-2019-gp91phox+p22phox-Absent-Standard-Panel-NEGATIVE-Exome-Required)'
    : isRAC2  ? 'NIS (AD-GOF-p.D57N-Dominant-Negative-DHR-Intermediate-Combined-PMN-Defects-HSCT-Often)'
    : 'MPO-Deficiency (AR-DHR-NORMAL-Mildest-Candida-Only-With-DM-Auto-Analyzer-Flag-No-HSCT)';

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
        <strong>Group:</strong> {groupLabel}
      </div>
      {cohort && (
        <div style={{ display: 'flex', gap: 16, fontSize: 12, color: '#aaa', marginBottom: 6 }}>
          <span>n={cohort.n}</span>
          <span>M:{cohort.male_n} F:{cohort.female_n}</span>
          <span>Onset median: {cohort.median_onset_months}mo</span>
          <span>HSCT: {(cohort.hsct_pct*100).toFixed(0)}%</span>
          <span>Infect: {(cohort.recurrent_infect_pct*100).toFixed(0)}%</span>
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
    const ep = tab === 'Definitions' ? 'definitions'
             : tab === 'Diagnostic Map' ? 'breakdown'
             : 'overview';
    const setter = tab === 'Definitions' ? setDefs
                 : tab === 'Diagnostic Map' ? setBk
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
        <div style={{ background: 'linear-gradient(135deg,#b71c1c,#6a1b9a)', borderRadius: 10, padding: 24, marginBottom: 24 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
            &#x1f9ea; Hereditary-CGD-Neutrophil-Oxidative-Burst-Defect-Atlas
          </h1>
          <p style={{ margin: '8px 0 0', fontSize: 13, opacity: 0.9 }}>
            Complete 8-Gene Reference: CYBB-NCF1-CYBA-NCF2-NCF4-CYBC1-RAC2-MPO
            · 320 Patients (8×40) · Seeds 4014-4021
          </p>
          <p style={{ margin: '4px 0 0', fontSize: 11, opacity: 0.8 }}>
            X-CGD (CYBB-gp91phox-DHR-ABSENT-Carrier-BIMODAL-IFN-γ-HSCT) · p47-CGD (NCF1-ΔGT-Deletion-Most-Common-AR)
            · p22-CGD (CYBA-gp91phox-Also-Absent) · p67-CGD (NCF2-Severe) · p40-CGD (NCF4-IBD-Dominant-DHR-Normal-Pitfall)
            · EROS (CYBC1-2019-Chaperone) · NIS (RAC2-DHR-Intermediate) · MPO-Deficiency (DHR-Normal-Mild-DM-Candida)
          </p>
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', marginBottom: 0 }}>
          {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
        </div>
        <div style={{ background: '#12122a', borderRadius: '0 8px 8px 8px', padding: 24, minHeight: 400 }}>
          {loading && <div style={{ color: '#aaa' }}>Loading…</div>}
          {error && <div style={{ color: '#f44336' }}>Error: {error}</div>}

          {/* OVERVIEW TAB */}
          {tab === 'Overview' && overview && (
            <div>
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Atlas Overview — {overview.atlas}</h2>
              <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Total Patients</div>
                  <div style={{ color: '#fff', fontSize: 28, fontWeight: 700 }}>{overview.total_n}</div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>HSCT Rate</div>
                  <div style={{ color: '#ff5252', fontSize: 28, fontWeight: 700 }}>
                    {(overview.aggregate_stats?.hsct_pct * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Recurrent Infections</div>
                  <div style={{ color: '#ffb300', fontSize: 28, fontWeight: 700 }}>
                    {(overview.aggregate_stats?.recurrent_infections_pct * 100).toFixed(1)}%
                  </div>
                </div>
                <div style={{ background: '#1a1a2e', padding: 12, borderRadius: 8, minWidth: 160 }}>
                  <div style={{ color: '#aaa', fontSize: 11 }}>Seeds</div>
                  <div style={{ color: '#80cbc4', fontSize: 20, fontWeight: 700 }}>{overview.seeds}</div>
                </div>
              </div>

              {/* Group badges */}
              <div style={{ marginBottom: 16 }}>
                <div style={{ marginBottom: 8 }}>
                  {['CYBB'].map(g => <Badge key={g} text={`X-CGD: ${g}`} color={GENE_COLORS[g]} />)}
                  {['CYBA','CYBC1'].map(g => <Badge key={g} text={`AR-Membrane: ${g}`} color={GENE_COLORS[g]} />)}
                  {['NCF1','NCF2','NCF4'].map(g => <Badge key={g} text={`AR-Cytosol: ${g}`} color={GENE_COLORS[g]} />)}
                  {['RAC2'].map(g => <Badge key={g} text={`NIS-GTPase: ${g}`} color={GENE_COLORS[g]} />)}
                  {['MPO'].map(g => <Badge key={g} text={`MPO-Def: ${g}`} color={GENE_COLORS[g]} />)}
                </div>
              </div>

              {/* DHR pattern groups */}
              <div style={{ background: '#1a1a2e', borderRadius: 8, padding: 12, marginBottom: 16 }}>
                <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>
                  DHR Test Pattern Guide (CRITICAL Diagnostic Distinction)
                </div>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                  <div style={{ padding: '6px 10px', background: '#2d0000', borderRadius: 6, fontSize: 11, color: '#ff5252' }}>
                    <strong>DHR ABSENT:</strong> {(overview.dhr_absent_group || []).join(', ')} — classic CGD
                  </div>
                  <div style={{ padding: '6px 10px', background: '#2d1a00', borderRadius: 6, fontSize: 11, color: '#ffb300' }}>
                    <strong>DHR PARTIAL:</strong> {(overview.dhr_partial_group || []).join(', ')} — NIS (RAC2)
                  </div>
                  <div style={{ padding: '6px 10px', background: '#002d00', borderRadius: 6, fontSize: 11, color: '#4caf50' }}>
                    <strong>DHR NORMAL:</strong> {(overview.dhr_normal_group || []).join(', ')} — NCF4 (phagosomal) / MPO def
                  </div>
                </div>
              </div>

              {/* Key facts */}
              <div style={{ marginBottom: 20 }}>
                <h3 style={{ color: '#b71c1c', fontSize: 14 }}>Key Clinical Facts</h3>
                <ul style={{ paddingLeft: 20, margin: 0 }}>
                  {(overview.key_facts || []).map((f, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 6, lineHeight: 1.5 }}>{f}</li>
                  ))}
                </ul>
              </div>

              {/* Gene cards */}
              <h3 style={{ color: '#b71c1c', fontSize: 14, marginBottom: 12 }}>Gene Cohort Summaries (40 patients each)</h3>
              {(overview.cohort_summaries || []).map((c) => (
                <GeneCard key={c.gene} gene={c.gene} color={GENE_COLORS[c.gene] || '#555'}
                          info={GENE_INFO[c.gene] || {}} cohort={c} />
              ))}
            </div>
          )}

          {/* GENE TABLE TAB */}
          {tab === 'Gene Table' && overview && (
            <div>
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Gene Comparison Table</h2>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ background: '#1a1a2e' }}>
                      {['Gene','Group','aa','Locus','kDa','Inheritance','DHR Test','Cyt.b558','IBD/Colitis','Sex Bias','HSCT Urg','Unique Finding'].map(h => (
                        <th key={h} style={{ padding: '8px 6px', textAlign: 'left', color: '#b71c1c', borderBottom: '1px solid #333' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PHENOTYPE_ROWS.map((row, i) => (
                      <tr key={row.gene} style={{ background: i % 2 === 0 ? '#0d0d1a' : '#12122a' }}>
                        <td style={{ padding: '6px', color: GENE_COLORS[row.gene], fontWeight: 700 }}>{row.gene}</td>
                        <td style={{ padding: '6px', color: '#ccc' }}>{row.group}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[0]}</td>
                        <td style={{ padding: '6px', color: '#9ecfff' }}>{GENE_INFO[row.gene]?.locus}</td>
                        <td style={{ padding: '6px', color: '#aaa' }}>{GENE_INFO[row.gene]?.size?.split('/')[1]}</td>
                        <td style={{ padding: '6px', color: '#ccc', fontSize: 10 }}>{GENE_INFO[row.gene]?.inh?.split('(')[0]}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.dhr === '✗✗✗' ? '#f44' : row.dhr === '✓±' ? '#fa8' : row.dhr === '±' ? '#ffb' : '#4f4' }}>{row.dhr}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.cytb558 === '✗✗✗' ? '#f44' : '#4f4' }}>{row.cytb558}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.ibd.includes('✓✓✓') ? '#ff7' : row.ibd.includes('✓✓') ? '#ffa' : row.ibd.includes('✓') ? '#ffb' : '#888' }}>{row.ibd}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: '#ccc' }}>{row.sex}</td>
                        <td style={{ padding: '6px', textAlign: 'center', color: row.hsct_urg.includes('✓✓✓') ? '#f55' : row.hsct_urg.includes('✓✓') ? '#fa8' : row.hsct_urg.includes('✓') ? '#ffd' : '#888' }}>{row.hsct_urg}</td>
                        <td style={{ padding: '6px', color: '#a8e6cf', fontSize: 10, maxWidth: 300 }}>{row.unique}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* DIAGNOSTIC MAP TAB */}
          {tab === 'Diagnostic Map' && breakdown && (
            <div>
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Diagnostic Map & Cascade Testing</h2>

              {/* Pathognomonic features */}
              <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Pathognomonic Features & Diagnostic Pitfalls</h3>
              {Object.entries(breakdown.pathognomonic_features || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 12, marginBottom: 8 }}>
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>
                    {k.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.6 }}>{v}</div>
                </div>
              ))}

              {/* Diagnostic groups */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Diagnostic Groups</h3>
              {Object.entries(breakdown.diagnostic_groups || {}).map(([grp, data]) => (
                <div key={grp} style={{ background: '#12122a', border: '1px solid #333', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <div style={{ display: 'flex', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
                    <span style={{ color: '#b71c1c', fontWeight: 700, fontSize: 13 }}>
                      {grp.replace(/_/g, ' ').toUpperCase()}
                    </span>
                    {(data.genes || []).map(g => (
                      <Badge key={g} text={g} color={GENE_COLORS[g] || '#555'} />
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: '#9ecfff', marginBottom: 6 }}>
                    <strong>Mechanism:</strong> {data.mechanism}
                  </div>
                  <div style={{ fontSize: 11, color: '#a8e6cf', marginBottom: 6 }}>
                    <strong>Shared features:</strong> {data.shared_feature}
                  </div>
                  <div style={{ fontSize: 11, color: '#ffcc80', lineHeight: 1.5 }}>
                    <strong>DDx:</strong> {data.key_ddx}
                  </div>
                </div>
              ))}

              {/* Cascade testing */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Cascade Testing Order</h3>
              <ol style={{ paddingLeft: 20, margin: 0 }}>
                {(breakdown.cascade_testing_order || []).map((step, i) => (
                  <li key={i} style={{ fontSize: 11, color: '#ccc', marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
                ))}
              </ol>
            </div>
          )}

          {/* DEFINITIONS TAB */}
          {tab === 'Definitions' && defs && (
            <div>
              <h2 style={{ color: '#b71c1c', marginTop: 0 }}>Key Definitions & Clinical Protocols</h2>

              {/* Key distinguishers */}
              <h3 style={{ color: '#ffb300', fontSize: 13, marginBottom: 10 }}>Key Distinguishers</h3>
              {Object.entries(defs.key_distinguishers || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#1a1a2e', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                  <div style={{ color: '#9ecfff', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                    {k.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                </div>
              ))}

              {/* Treatment matrix */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Treatment Matrix</h3>
              {Object.entries(defs.treatment_matrix || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#0d1a0d', border: '1px solid #2e7d32', borderRadius: 6, padding: 10, marginBottom: 8 }}>
                  <div style={{ color: '#a8e6cf', fontWeight: 700, fontSize: 11, marginBottom: 4 }}>
                    {k.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div style={{ color: '#ccc', fontSize: 11, lineHeight: 1.5 }}>{v}</div>
                </div>
              ))}

              {/* Full definitions */}
              <h3 style={{ color: '#ffb300', fontSize: 13, margin: '20px 0 10px' }}>Full Definitions</h3>
              {(defs.definitions || []).map((d, i) => (
                <div key={i} style={{ background: '#12122a', borderRadius: 8, padding: 14, marginBottom: 10 }}>
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{d.term}</div>
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
