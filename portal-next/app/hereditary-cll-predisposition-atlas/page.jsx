'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cll-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':    '#1a237e',  // deep indigo      — monoallelic CLL 2-3x; del11q 20-25% worst-prognosis; RADIOSENSITIVITY-ABSOLUTE; FCR inferior
  'TP53':   '#880e4f',  // deep crimson      — LFS; del17p 8-10% WORST-PROGNOSIS; FCR-CI; venetoclax backbone; WB-MRI Toronto
  'POT1':   '#004d40',  // deep teal         — MOST-ESTABLISHED familial CLL ~8%; short telomeres PATHOGNOMONIC; ibrutinib/venetoclax
  'NBN':    '#bf360c',  // deep burnt orange — NBS; 657del5 Slavic 1in150; CLL/NHL 60x HIGHEST; cranial RT CI
  'BRCA2':  '#006064',  // dark cyan         — HBOC; del13q somatic 55% MOST-COMMON; del13q-SOLE-GOOD-PROGNOSIS; cisplatin HRD
  'CHEK2':  '#4a148c',  // deep purple       — 1100delC 1in200; I157T Eastern European; CLL 2-3x; NOT HRD; NO PARPi
  'CDKN2A': '#33691e',  // olive green       — FAMM; del9p21 somatic 20%; Richter 10-15%; CDK4/6i INACTIVE CLL
  'RB1':    '#e65100',  // deep amber        — del13q14 55% MOST-COMMON; SOLE=GOOD-PROGNOSIS; CDK4/6i INACTIVE; venetoclax active
};

const GENE_INFO = {
  'ATM':    { full: 'monoallelic-CLL-2-3x / del11q-somatic-20-25pct-worst-prognosis / RADIOSENSITIVITY-ABSOLUTE / FCR-inferior-ibrutinib-venetoclax / ceralasertib-ATRi-trials', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF heterozygous' },
  'TP53':   { full: 'LFS / del17p-8-10pct-WORST-PROGNOSIS / FCR-CI-ABSOLUTELY / venetoclax-backbone-MANDATORY / WB-MRI-Toronto-annually / eprenetapopt-APR246-trials', locus: '17p13.1', size: '393 aa / 43 kDa', inh: 'AD LOF heterozygous' },
  'POT1':   { full: 'MOST-ESTABLISHED-familial-CLL-8pct / short-telomeres-flow-FISH-PATHOGNOMONIC / OB-fold-ssDNA-binding / FCR-inferior-genomic-instability / ibrutinib-venetoclax / BTKi+BCL2i-combination', locus: '7q31.33', size: '634 aa / 71 kDa', inh: 'AD LOF heterozygous' },
  'NBN':    { full: 'NBS / 657del5-Slavic-founder-1in150 / CLL-NHL-60x-HIGHEST / cranial-RT-ABSOLUTELY-CI / IVIG-IgG<3 / fludarabine-RIC-not-TBI / MRN-complex-DSB-repair', locus: '8q21.3', size: '754 aa / 95 kDa', inh: 'AD/AR LOF; AR biallelic = NBS' },
  'BRCA2':  { full: 'HBOC / del13q-somatic-55pct-MOST-COMMON-CLL / del13q-SOLE-GOOD-PROGNOSIS / HRD-60pct / cisplatin-preferred-R/R / FA-D1-AVOID-ALKYLATING-ABSOLUTELY / olaparib-emerging', locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF; AR biallelic = FA-D1' },
  'CHEK2':  { full: 'NOT-HRD-NO-PARPi / 1100delC-1in200-Western-European / I157T-Eastern-European / CLL-2-3x / ATM+CHEK2-additive-8-10x / standard-CLL-Rx-per-FISH', locus: '22q12.1', size: '543 aa / 60 kDa', inh: 'AD LOF heterozygous' },
  'CDKN2A': { full: 'FAMM / del9p21-somatic-20pct-progression / Richter-transform-10-15pct / CDK4-6i-INACTIVE-CLL / dermatology-18yr / pancreatic-EUS-40yr / p16-ARF-dual-locus', locus: '9p21.3', size: '156 aa / 17 kDa', inh: 'AD LOF heterozygous' },
  'RB1':    { full: 'del13q14-somatic-55pct-MOST-COMMON / SOLE=GOOD-PROGNOSIS-CLL / CDK4-6i-INACTIVE-RB1-deleted / venetoclax-ACTIVE-BCL2-upregulated / bilateral-Rb-PATHOGNOMONIC / secondary-CLL-post-RT', locus: '13q14.2', size: '928 aa / 110 kDa', inh: 'AD LOF heterozygous' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, breakdown }) {
  const bg = color;
  const cohort = breakdown?.gene_data?.[gene] || {};
  const geneAtlas = breakdown?.atlas_genes?.find(g => g.gene === gene) || {};
  return (
    <div style={{ border: `2px solid ${bg}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: bg, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`CLL risk: ${geneAtlas.cll_risk?.split(';')[0] || ''}`} color={bg} />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
        ✅ <b>Key rule:</b> {geneAtlas.key_rule}
      </div>
      <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
        🚫 <b>Avoid:</b> {geneAtlas.key_avoid}
      </div>
      {cohort.n && (
        <div style={{ display: 'flex', gap: 12, fontSize: 12, flexWrap: 'wrap', marginTop: 8 }}>
          <span>n={cohort.n}</span>
          <span>Mean age: {cohort.mean_age}yr</span>
          <span>Binet A: {cohort.binet_a} | B: {cohort.binet_b} | C: {cohort.binet_c}</span>
          <span>IGHV unmut: {cohort.ighv_unmutated}</span>
          <span>BTKi: {cohort.tx_btki} | Ven: {cohort.tx_venetoclax}</span>
          <span>CR: {Math.round(cohort.cr_rate * 100)}%</span>
          <span>OS-2yr: {Math.round(cohort.os_2yr * 100)}%</span>
          <span>Richter: {cohort.richter_count}</span>
        </div>
      )}
    </div>
  );
}

export default function HereditoryCLLPredispositionAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchData = async (endpoint, setter) => {
    setLoading(true); setError(null);
    try {
      const r = await fetch(`${API}/api/${SLUG}/${endpoint}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setter(await r.json());
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  useEffect(() => {
    if (tab === 'Overview' && !overview) fetchData('overview', setOverview);
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) fetchData('breakdown', setBreakdown);
    if (tab === 'Definitions' && !definitions) fetchData('definitions', setDefinitions);
  }, [tab]);

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '16px', fontFamily: 'monospace' }}>
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#004d40 100%)', color: '#fff', borderRadius: 8, padding: '20px 24px', marginBottom: 20 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🧬 Hereditary CLL Predisposition Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.9 }}>
          Complete 8-Gene Reference · ATM · TP53 · POT1 · NBN · BRCA2 · CHEK2 · CDKN2A · RB1 · 320-Patient Aggregate · Seeds 3646–3653
        </div>
        <div style={{ fontSize: 11, marginTop: 4, opacity: 0.75 }}>
          ATM-RADIOSENSITIVITY-ABSOLUTE · TP53-del17p-WORST-PROGNOSIS · POT1-FAMILIAL-CLL-8PCT · NBN-60x-HIGHEST · BRCA2-del13q-SOLE-GOOD-PROGNOSIS · CHEK2-NOT-HRD · CDKN2A-CDK4-6i-INACTIVE · RB1-VENETOCLAX-ACTIVE
        </div>
      </div>

      {/* Gene colour legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 16 }}>
        {genes.map(g => (
          <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '3px 10px', borderRadius: 4, fontSize: 12, fontWeight: 600 }}>{g}</span>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1a237e' : '#e0e0e0', color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400 }}>
            {t}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#666', padding: 20 }}>Loading…</div>}
      {error && <div style={{ color: 'red', padding: 10 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 20 }}>
            {[
              ['Total Patients', overview.total_patients],
              ['Genes', overview.n_genes],
              ['Mean Age', `${overview.mean_age_overall} yr`],
              ['Seeds', overview.seeds],
            ].map(([label, val]) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#1a237e' }}>{val}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Stage / cytogenetics / treatment */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#fff8e1', borderRadius: 8, padding: 16 }}>
              <b>Binet Stage Distribution</b>
              {overview.stage_distribution && Object.entries(overview.stage_distribution).filter(([k]) => !k.includes('pct')).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 13 }}>
                  <span>{k.toUpperCase()}</span><span style={{ fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>
            <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16 }}>
              <b>Cytogenetics (FISH)</b>
              {overview.cytogenetic_distribution && Object.entries(overview.cytogenetic_distribution).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 13 }}>
                  <span>{k}</span><span style={{ fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>
            <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16 }}>
              <b>First-line Treatment</b>
              {overview.treatment_distribution && Object.entries(overview.treatment_distribution).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 13 }}>
                  <span>{k.replace(/_/g, ' ')}</span><span style={{ fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Outcomes */}
          {overview.outcomes && (
            <div style={{ background: '#f3e5f5', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <b>Aggregate Outcomes (320 patients)</b>
              <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginTop: 8, fontSize: 13 }}>
                <span>Avg CR: <b>{overview.outcomes.avg_cr_pct}%</b></span>
                <span>Avg 2-yr OS: <b>{overview.outcomes.avg_os_2yr_pct}%</b></span>
                <span>Richter transforms: <b>{overview.outcomes.total_richter}</b> ({overview.outcomes.richter_pct}%)</span>
              </div>
            </div>
          )}

          {/* Per-gene summary */}
          <h3>Per-Gene Summary</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Locus', 'n', 'Mean Age', 'Binet A%', 'Binet C%', 'IGHV Unmut%', 'BTKi%', 'CR%', 'OS-2yr%', 'Richter', 'Key Rule'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview.per_gene?.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #ddd' }}>
                  <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                  <td style={{ padding: '7px 10px' }}>{row.locus}</td>
                  <td style={{ padding: '7px 10px' }}>{row.n}</td>
                  <td style={{ padding: '7px 10px' }}>{row.mean_age}</td>
                  <td style={{ padding: '7px 10px' }}>{row.binet_a_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.binet_c_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.ighv_unmut_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.btki_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.cr_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.os_2yr_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.richter_count}</td>
                  <td style={{ padding: '7px 10px', fontSize: 11 }}>{row.key_rule?.split(';')[0]}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Key distinctions */}
          {overview.key_clinical_distinctions && (
            <div style={{ marginTop: 24 }}>
              <h3>Key Clinical Distinctions</h3>
              {overview.key_clinical_distinctions.map((d, i) => {
                const geneName = genes[i] || '';
                return (
                  <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
                    <span style={{ background: GENE_COLORS[geneName] || '#666', color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 11, minWidth: 60, textAlign: 'center' }}>{geneName}</span>
                    <span style={{ fontSize: 12, lineHeight: 1.5 }}>{d}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h3>Complete 8-Gene Reference Table</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#004d40', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Syndrome', 'Inheritance', 'CLL Risk', 'Pathognomonic', 'Key Rule', 'Targeted Rx'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {breakdown.atlas_genes?.map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #ddd', verticalAlign: 'top' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                  <td style={{ padding: '8px 10px' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{GENE_INFO[g.gene]?.size}</td>
                  <td style={{ padding: '8px 10px' }}>{g.syndrome}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{g.inheritance}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{g.cll_risk}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, color: '#b71c1c' }}>{g.pathognomonic}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11, color: '#1b5e20', fontWeight: 600 }}>{g.key_rule}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{g.targeted_rx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h3>Clinical Atlas — 8-Gene CLL Predisposition Cards</h3>
          {genes.map(g => (
            <GeneCard key={g} gene={g} color={GENE_COLORS[g]} info={GENE_INFO[g]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3>Clinical Reference Definitions</h3>
          {definitions.definitions?.map((d, i) => (
            <div key={i} style={{ marginBottom: 16, background: i % 2 === 0 ? '#f5f5f5' : '#fff', padding: '12px 16px', borderRadius: 6, borderLeft: '4px solid #1a237e' }}>
              <div style={{ fontWeight: 700, marginBottom: 4, color: '#1a237e' }}>{d.term}</div>
              <div style={{ fontSize: 12, lineHeight: 1.6 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
