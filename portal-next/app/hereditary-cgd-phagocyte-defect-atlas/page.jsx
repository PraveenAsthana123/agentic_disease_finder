'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cgd-phagocyte-defect-atlas';
const TABS = ['Overview', 'Gene Table', 'Phagocyte Function Map', 'Definitions'];

const GENE_COLORS = {
  'CYBB':  '#b71c1c',  // deep red         -- XL-CGD gp91phox; 65% all CGD; IFN-gamma FDA1990
  'CYBA':  '#bf360c',  // burnt orange      -- p22phox AR-CGD; obligate heterodimer; Middle Eastern founder
  'NCF1':  '#1565c0',  // deep blue         -- p47phox most common AR-CGD; pseudogene MLPA mandatory
  'NCF2':  '#0277bd',  // medium blue       -- p67phox activation domain; no pseudogene
  'NCF4':  '#1b5e20',  // dark green        -- p40phox IBD dominant; phagosomal; partial DHR
  'EROS':  '#4a148c',  // deep purple       -- CYBC1 ER chaperone; ultra-rare 2019 discovery
  'RAC2':  '#006064',  // teal              -- RhoGTPase AD; leukocytosis pathognomonic
  'MPO':   '#4e342e',  // brown/olive       -- myeloperoxidase; most common phagocyte disorder; clinically silent
};

const GENE_INFO = {
  'CYBB': { full: 'gp91phox-NOX2-570aa / XL-CGD-most-common-65pct-all-CGD / DHR-flat-gold-standard / IFN-gamma-FDA1990-FIRST-cytokine-immunotherapy / Aspergillus-35-40pct', locus: 'Xp21.1', size: '570 aa / 65.3 kDa', inh: 'XL' },
  'CYBA': { full: 'p22phox-195aa / AR-CGD-Type2-both-sexes / obligate-heterodimer-gp91-absent-CYBB-intact / Middle-Eastern-540+1G>T-founder / CYBA-flow-when-gp91-absent-CYBB-negative', locus: '16q24.3', size: '195 aa / 21 kDa', inh: 'AR' },
  'NCF1': { full: 'p47phox-390aa / most-common-AR-CGD-25pct / NCF1B-NCF1C-pseudogene-GT-deletion-75-80pct / MLPA-mandatory-NOT-Sanger / IBD-40pct', locus: '7q11.23', size: '390 aa / 43.6 kDa', inh: 'AR' },
  'NCF2': { full: 'p67phox-526aa / AR-CGD-Type4-5pct / activation-domain-electron-transfer / no-pseudogene-standard-sequencing-reliable / IBD-25-30pct', locus: '1q25.3', size: '526 aa / 59.6 kDa', inh: 'AR' },
  'NCF4': { full: 'p40phox-339aa / AR-CGD-Type5-rarest / IBD-DOMINANT-75pct-presenting / partial-DHR-PATHOGNOMONIC / phagosomal-specific-not-plasma-membrane', locus: '22q13.1', size: '339 aa / 40 kDa', inh: 'AR' },
  'EROS': { full: 'CYBC1-EROS-291aa / ER-chaperone-gp91phox / ultra-rare-lt25-cases / discovered-2019 / gp91-absent-CYBB-intact-diagnostic-pitfall', locus: '1p22.1', size: '291 aa / 34 kDa', inh: 'AR' },
  'RAC2': { full: 'RhoGTPase-RAC2-192aa / AD-dominant-negative-D57N / leukocytosis-WBC-20-100-PATHOGNOMONIC / combined-NADPH-migration-degranulation-defect / broader-than-CGD', locus: '22q13.1', size: '192 aa / 21.5 kDa', inh: 'AD' },
  'MPO':  { full: 'Myeloperoxidase-745aa / most-common-phagocyte-disorder-1in2000-4000 / clinically-silent-99pct / DHR-NORMAL-NBT-NORMAL-NOT-CGD / diabetes-candida-PATHOGNOMONIC', locus: '17q22', size: '745 aa / 84.3 kDa', inh: 'AR' },
};

const PHENOTYPE_ROWS = [
  { gene: 'CYBB',  nbt: '−',      dhr: 'FLAT',    aspergillus: '35-40%', ibd: '30-40%', leukocytosis: '−',      frequency: '65% all CGD',    treatment: 'TMP-SMX+IFN-γ+HSCT/gene therapy' },
  { gene: 'CYBA',  nbt: '−',      dhr: 'FLAT',    aspergillus: '~30%',   ibd: '~30%',   leukocytosis: '−',      frequency: '~5% AR-CGD',     treatment: 'TMP-SMX+antifungal+HSCT' },
  { gene: 'NCF1',  nbt: '−',      dhr: 'FLAT',    aspergillus: '~20%',   ibd: '~40%',   leukocytosis: '−',      frequency: '25% AR-CGD',     treatment: 'TMP-SMX+IFN-γ+HSCT (MLPA first)' },
  { gene: 'NCF2',  nbt: '−',      dhr: 'FLAT',    aspergillus: '~22%',   ibd: '25-30%', leukocytosis: '−',      frequency: '~5% AR-CGD',     treatment: 'TMP-SMX+antifungal+HSCT' },
  { gene: 'NCF4',  nbt: 'partial',dhr: 'PARTIAL', aspergillus: '~10%',   ibd: '~75%',   leukocytosis: '−',      frequency: '<1% all CGD',    treatment: 'IBD management+HSCT if severe' },
  { gene: 'EROS',  nbt: '−',      dhr: 'FLAT',    aspergillus: '~32%',   ibd: '~25%',   leukocytosis: '−',      frequency: '<25 cases ever', treatment: 'TMP-SMX+antifungal+HSCT' },
  { gene: 'RAC2',  nbt: 'partial',dhr: 'partial', aspergillus: '~25%',   ibd: '~15%',   leukocytosis: '✓✓✓',   frequency: 'rare AD',        treatment: 'HSCT early (DN form)' },
  { gene: 'MPO',   nbt: '✓NL',   dhr: '✓NL',    aspergillus: '~3%',    ibd: '~5%',    leukocytosis: '−',      frequency: '1:2000-4000',    treatment: 'Glycaemic control; azole if DM' },
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
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: color, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{cohort?.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{info?.full?.replace(/-/g, ' ')}</div>
      {cohort?.pathognomonic && (
        <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>PATHOGNOMONIC:</strong> {cohort.pathognomonic}
        </div>
      )}
      {cohort?.first_line_treatment && (
        <div style={{ background: '#e8f5e9', border: '1px solid #4caf50', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>Treatment:</strong> {cohort.first_line_treatment}
        </div>
      )}
      {cohort?.critical_rules?.length > 0 && (
        <div style={{ marginTop: 8 }}>
          {cohort.critical_rules.map(r => (
            <Badge key={r} text={r.replace(/-/g, ' ')} color={color} />
          ))}
        </div>
      )}
      {cohort?.n && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          <Badge text={`n=${cohort.n}`} color={color} />
          <Badge text={`onset ${cohort.mean_onset_age}yr`} color="#555" />
          <Badge text={`dx-lag ${cohort.mean_dx_lag_yrs}yr`} color="#888" />
          {cohort.nbt_abnormal_n > 0 && <Badge text={`NBT-abn ${cohort.nbt_abnormal_n}/40`} color="#b71c1c" />}
          {cohort.dhr_abnormal_n > 0 && <Badge text={`DHR-abn ${cohort.dhr_abnormal_n}/40`} color="#bf360c" />}
          {cohort.aspergillus_n > 0 && <Badge text={`Asp ${cohort.aspergillus_n}/40`} color="#6a1b9a" />}
          {cohort.staph_aureus_n > 0 && <Badge text={`Staph ${cohort.staph_aureus_n}/40`} color="#c62828" />}
          {cohort.ibd_n > 0 && <Badge text={`IBD ${cohort.ibd_n}/40`} color="#1b5e20" />}
          {cohort.leukocytosis_n > 0 && <Badge text={`Leuko ${cohort.leukocytosis_n}/40`} color="#006064" />}
          {cohort.ifn_gamma_benefit_n > 0 && <Badge text={`IFN-γ ${cohort.ifn_gamma_benefit_n}/40`} color="#1565c0" />}
          {cohort.hsct_curative_n > 0 && <Badge text={`HSCT-cure ${cohort.hsct_curative_n}/40`} color="#0277bd" />}
          {cohort.gene_therapy_eligible_n > 0 && <Badge text={`gene-Rx ${cohort.gene_therapy_eligible_n}/40`} color="#4a148c" />}
        </div>
      )}
    </div>
  );
}

export default function HereditoryCGDPhagocyteDefectAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#888' }}>Loading Hereditary CGD Phagocyte Defect Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  const cohortByGene = {};
  (breakdown?.genes || []).forEach(g => { cohortByGene[g.gene] = g; });

  return (
    <div style={{ padding: 24, fontFamily: 'system-ui, sans-serif', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #b71c1c 0%, #1565c0 50%, #1b5e20 100%)', borderRadius: 10, padding: '20px 28px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>🧬 Hereditary CGD &amp; Phagocyte Defect Atlas</h1>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.9 }}>
          Complete 8-Gene Reference · CYBB · CYBA · NCF1 · NCF2 · NCF4 · EROS · RAC2 · MPO · {overview?.total_patients} patients · seeds {overview?.seeds}
        </div>
        <div style={{ marginTop: 4, fontSize: 12, opacity: 0.8 }}>
          NADPH Oxidase · NBT/DHR Testing · Aspergillus/Staph Risk · IFN-γ FDA1990 · Pseudogene MLPA · Leukocytosis · Diabetes-Candida
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(i)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
              background: tab === i ? '#b71c1c' : '#f5f5f5',
              color: tab === i ? '#fff' : '#333',
            }}
          >{t}</button>
        ))}
      </div>

      {/* Tab 0: Overview */}
      {tab === 0 && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#b71c1c' },
              { label: 'Genes Covered', value: overview.genes?.length, color: '#4a148c' },
              { label: 'Cohort per Gene', value: overview.cohort_size, color: '#1565c0' },
              { label: 'Seed Range', value: overview.seeds, color: '#1b5e20' },
            ].map(k => (
              <div key={k.label} style={{ background: '#fff', border: `2px solid ${k.color}`, borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: k.color }}>{k.value}</div>
                <div style={{ fontSize: 13, color: '#666', marginTop: 4 }}>{k.label}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Key Clinical Rules</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 12, marginBottom: 24 }}>
            {Object.entries(overview.key_facts || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#fff8e1', border: '1px solid #ffa000', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#e65100', textTransform: 'uppercase', marginBottom: 4 }}>
                  {k.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Diagnostic Approach by Test Pattern</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 12, marginBottom: 24 }}>
            {Object.entries(overview.diagnostic_approach || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#e8f5e9', border: '1px solid #43a047', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1b5e20', textTransform: 'uppercase', marginBottom: 4 }}>
                  {k.replace(/_/g, ' ')}
                </div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h2 style={{ color: '#b71c1c', marginBottom: 12 }}>Gene Cohorts Overview</h2>
          {overview.cohorts?.map(c => (
            <GeneCard
              key={c.gene}
              gene={c.gene}
              color={GENE_COLORS[c.gene] || '#555'}
              info={GENE_INFO[c.gene]}
              cohort={c}
            />
          ))}
        </div>
      )}

      {/* Tab 1: Gene Table */}
      {tab === 1 && breakdown && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Gene Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  {['#', 'Gene', 'Protein / Size', 'Locus', 'Inh.', 'Syndrome', 'n', 'Onset (yr)', 'Dx-Lag', 'NBT-abn', 'DHR-abn', 'Asp', 'IBD', 'Leuko', 'IFN-γ', 'HSCT-cure', 'Gene-Rx'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes?.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px', color: '#888' }}>{g.rank}</td>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[g.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{g.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', fontSize: 11, color: '#444' }}>{g.aa}aa / {g.mw_kda}kDa</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.inheritance}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{g.syndrome?.split('/')[0]}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#555' }}>{g.n}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_onset_age}</td>
                    <td style={{ padding: '7px 10px' }}>{g.mean_dx_lag_yrs}yr</td>
                    <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 700 }}>{g.nbt_abnormal_n}</td>
                    <td style={{ padding: '7px 10px', color: '#bf360c', fontWeight: 700 }}>{g.dhr_abnormal_n}</td>
                    <td style={{ padding: '7px 10px', color: '#6a1b9a' }}>{g.aspergillus_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: g.ibd_n > 10 ? 700 : 400 }}>{g.ibd_n}</td>
                    <td style={{ padding: '7px 10px', color: '#006064', fontWeight: g.leukocytosis_n > 0 ? 700 : 400 }}>{g.leukocytosis_n}</td>
                    <td style={{ padding: '7px 10px', color: '#1565c0' }}>{g.ifn_gamma_benefit_n}</td>
                    <td style={{ padding: '7px 10px', color: '#0277bd' }}>{g.hsct_curative_n}</td>
                    <td style={{ padding: '7px 10px', color: '#4a148c' }}>{g.gene_therapy_eligible_n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ marginTop: 20 }}>
            <h3 style={{ color: '#b71c1c', marginBottom: 12 }}>Critical Rules per Gene</h3>
            {breakdown.genes?.map(g => (
              <div key={g.gene} style={{ marginBottom: 14, padding: 12, background: '#fafafa', borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#555', marginBottom: 6 }}>{g.gene} — {g.syndrome?.split('/')[0]}</div>
                <div style={{ marginBottom: 6, fontSize: 12, color: '#444' }}>
                  <strong>PATHOGNOMONIC:</strong> {g.pathognomonic}
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {g.critical_rules?.map(r => (
                    <Badge key={r} text={r.replace(/-/g, ' ')} color={GENE_COLORS[g.gene] || '#555'} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 2: Phagocyte Function Map */}
      {tab === 2 && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Phagocyte Function Classification</h2>

          <div style={{ background: '#fff3e0', border: '2px solid #ff6f00', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ color: '#e65100', margin: '0 0 10px' }}>NBT / DHR / MPO Stain — Three-Test Diagnostic Matrix</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
              {[
                { pattern: 'NBT − / DHR FLAT / gp91 absent', dx: 'CYBB (XL, male) or CYBA (AR, both sexes); if CYBB neg → CYBA; if both neg → EROS', color: '#b71c1c' },
                { pattern: 'NBT − / DHR FLAT / gp91 present', dx: 'Cytosolic CGD: NCF1 (MLPA mandatory), NCF2 (standard seq), NCF4 (partial DHR clue)', color: '#1565c0' },
                { pattern: 'NBT partial / DHR PARTIAL', dx: 'NCF4 (IBD dominant) or RAC2 DN; partial phagosomal oxidase retained', color: '#1b5e20' },
                { pattern: 'NBT + / DHR NORMAL / MPO absent', dx: 'MPO deficiency — NOT CGD; NADPH oxidase intact; diabetes → Candida risk', color: '#4e342e' },
              ].map(p => (
                <div key={p.pattern} style={{ background: '#fff', border: `2px solid ${p.color}`, borderRadius: 6, padding: 12 }}>
                  <div style={{ fontWeight: 700, color: p.color, marginBottom: 4, fontFamily: 'monospace', fontSize: 12 }}>{p.pattern}</div>
                  <div style={{ fontSize: 13 }}>{p.dx}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'NBT', 'DHR', 'Aspergillus', 'IBD', 'Leukocytosis', 'Frequency', 'Treatment'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px' }}>
                      <span style={{ background: GENE_COLORS[row.gene] || '#555', color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{row.gene}</span>
                    </td>
                    <td style={{ padding: '7px 10px', color: row.nbt === '−' ? '#b71c1c' : (row.nbt === '✓NL' ? '#1b5e20' : '#e65100'), fontWeight: 700, fontSize: 15 }}>{row.nbt}</td>
                    <td style={{ padding: '7px 10px', color: row.dhr === 'FLAT' ? '#b71c1c' : (row.dhr === '✓NL' ? '#1b5e20' : '#e65100'), fontWeight: 700, fontSize: row.dhr === 'FLAT' || row.dhr === '✓NL' ? 14 : 13 }}>{row.dhr}</td>
                    <td style={{ padding: '7px 10px', color: '#6a1b9a', fontSize: 12 }}>{row.aspergillus}</td>
                    <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: row.ibd.includes('75') ? 700 : 400, fontSize: 12 }}>{row.ibd}</td>
                    <td style={{ padding: '7px 10px', color: row.leukocytosis !== '−' ? '#006064' : '#ccc', fontWeight: row.leukocytosis !== '−' ? 800 : 400 }}>{row.leukocytosis}</td>
                    <td style={{ padding: '7px 10px', fontSize: 12 }}>{row.frequency}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{row.treatment}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ color: '#c62828', margin: '0 0 10px' }}>⚠️ Critical Safety &amp; Diagnostic Rules</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 14, lineHeight: 1.8 }}>
              <li><strong>NCF1:</strong> NEVER diagnose by Sanger/NGS alone — 75-80% of alleles are pseudogene GT deletions invisible to routine sequencing. Order MLPA or long-range PCR.</li>
              <li><strong>CYBA:</strong> gp91phox absent + CYBB negative does NOT mean CYBB failure — the p22phox heterodimer partner is absent. Test CYBA (p22phox flow + sequencing).</li>
              <li><strong>EROS:</strong> gp91phox absent + CYBB negative + CYBA negative → EROS/CYBC1 panel (ER chaperone, discovered 2019, &lt;25 cases).</li>
              <li><strong>NCF4:</strong> Partial DHR (not flat) + dominant IBD = NCF4, not other CGD. Phagosomal-specific defect — plasma membrane oxidase preserved.</li>
              <li><strong>RAC2 DN:</strong> LEUKOCYTOSIS (WBC 20-100×10^9/L) PATHOGNOMONIC — impaired diapedesis. Broader than CGD: also migration and degranulation defect.</li>
              <li><strong>MPO:</strong> DHR NORMAL + NBT NORMAL = NOT CGD. MPO flow stain confirms MPO deficiency. If diabetic: HIGH RISK severe Candida — tight glycaemic control mandatory.</li>
              <li><strong>IFN-γ (Actimmune):</strong> FDA approved 1990 — FIRST cytokine approved as immunotherapy. TIW subcutaneous 50 mcg/m2. Reduces infections ~70% in XL-CGD (CYBB).</li>
            </ul>
          </div>
        </div>
      )}

      {/* Tab 3: Definitions */}
      {tab === 3 && defs && (
        <div>
          <h2 style={{ color: '#b71c1c', marginBottom: 16 }}>Clinical Definitions</h2>
          {Object.entries(defs.definitions || {}).map(([title, text]) => (
            <div key={title} style={{ marginBottom: 20, padding: 16, background: '#fafafa', borderRadius: 8, borderLeft: '4px solid #b71c1c' }}>
              <h3 style={{ color: '#b71c1c', margin: '0 0 10px', fontSize: 15 }}>{title}</h3>
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.7, color: '#333' }}>{text}</p>
            </div>
          ))}

          <div style={{ marginTop: 24, padding: 16, background: '#e3f2fd', borderRadius: 8 }}>
            <h3 style={{ color: '#1565c0', margin: '0 0 10px' }}>Standards &amp; References</h3>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
              {defs.standards?.map(s => <li key={s} style={{ marginBottom: 6 }}>{s}</li>)}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
