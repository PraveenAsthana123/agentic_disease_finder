'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-aml-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'CEBPA':   '#1a237e',  // deep indigo       -- CEBPA-FPD; biallelic N+C PATHOGNOMONIC; AML 40-60% HIGHEST; NOT-allograft-CR1
  'DDX41':   '#880e4f',  // deep crimson       -- late onset 55-65yr HALLMARK; R525H 50%; DONOR-TESTING-MANDATORY; HMA-venetoclax
  'RUNX1':   '#004d40',  // deep teal          -- FPD-AML; thrombocytopenia PATHOGNOMONIC; SIBLING-DONORS-CI; azacitidine
  'GATA2':   '#bf360c',  // deep burnt orange  -- MonoMAC; AML 30-70x HIGHEST; NK-lymphopenia PATHOGNOMONIC; HSCT-CURATIVE
  'TP53':    '#006064',  // dark cyan          -- LFS; complex-karyotype PATHOGNOMONIC; AVOID-RADIATION-ABSOLUTELY; venetoclax-decitabine
  'ANKRD26': '#4a148c',  // deep purple        -- THC2; 5UTR-PANEL-MANDATORY; standard-sequencing-MISSES; AML/MDS 5-10%
  'ETV6':    '#33691e',  // olive green        -- THC5; AVOID-NSAIDs; AVOID-high-dose-MTX; ETV6-RUNX1-fusion-ABSENT-germline
  'SAMD9L':  '#e65100',  // deep amber         -- ATAD; monosomy-7 80% PATHOGNOMONIC; cerebellar-ataxia PATHOGNOMONIC; HSCT-corrects-AML-NOT-ataxia
};

const GENE_INFO = {
  'CEBPA':   { full: 'CEBPA-FPD / biallelic-N+C-terminal-PATHOGNOMONIC / AML-40-60pct-HIGHEST-PENETRANCE / NOT-allograft-CR1-unless-biallelic / sibling-donor-CEBPA-test-MANDATORY / aza-venetoclax-R/R / menin-inhibitors-emerging', locus: '19q13.11', size: '358 aa / 42 kDa', inh: 'AD LOF (biallelic N+C hits)' },
  'DDX41':   { full: 'Familial-AML-MDS / late-onset-55-65yr-HALLMARK / R525H-somatic-second-hit-50pct / DONOR-DDX41-TESTING-MANDATORY / HMA-venetoclax-preferred / menin-inhibitors-emerging', locus: '5q35.3', size: '622 aa / 70 kDa', inh: 'AD LOF heterozygous' },
  'RUNX1':   { full: 'FPD-AML / thrombocytopenia-PATHOGNOMONIC / AML-30-50pct-lifetime / SIBLING-DONORS-CI-ABSOLUTELY / azacitidine-preferred / menin-inhibitors-active / cascade-test-MANDATORY', locus: '21q22.12', size: '453 aa / 51 kDa', inh: 'AD LOF heterozygous' },
  'GATA2':   { full: 'MonoMAC-Emberger / AML-MDS-30-70x-HIGHEST / NK-lymphopenia-PATHOGNOMONIC / HSCT-CURATIVE-DO-NOT-DELAY / EBV-CMV-Q3M-MANDATORY / NTM-azithromycin', locus: '3q21.3', size: '480 aa / 50 kDa', inh: 'AD LOF heterozygous' },
  'TP53':    { full: 'LFS / AML-10-20x / complex-karyotype-PATHOGNOMONIC / AVOID-RADIATION-ABSOLUTELY / venetoclax-decitabine-preferred / WB-MRI-Toronto-annually / eprenetapopt-APR246-trials', locus: '17p13.1', size: '393 aa / 43 kDa', inh: 'AD LOF heterozygous' },
  'ANKRD26': { full: 'THC2 / 5UTR-PANEL-MANDATORY / standard-sequencing-MISSES-5UTR / AML-MDS-5-10pct / thrombocytopenia-40-100x / aza-venetoclax / avoid-eltrombopag-MDS-features', locus: '10p12.1', size: '1710 aa / 190 kDa', inh: 'AD LOF (5UTR promoter)' },
  'ETV6':    { full: 'THC5 / AVOID-NSAIDs / AVOID-HIGH-DOSE-MTX / AML-MDS-10-15pct / ETV6-RUNX1-fusion-ABSENT-germline / thrombocytopenia / aza-venetoclax', locus: '12p13.2', size: '452 aa / 57 kDa', inh: 'AD LOF heterozygous' },
  'SAMD9L':  { full: 'ATAD / monosomy-7-80pct-PATHOGNOMONIC / cerebellar-ataxia-PATHOGNOMONIC / HSCT-CORRECTS-AML-NOT-ATAXIA / adrenal-sick-day-rule-10x / reversion-mosaicism / azacitidine-bridge', locus: '7q21.2', size: '1589 aa / 175 kDa', inh: 'AD GOF heterozygous' },
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
        <Badge text={`AML risk: ${geneAtlas.aml_risk?.split(';')[0] || ''}`} color={bg} />
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
          <span>ELN Fav: {cohort.eln_favourable} | Int: {cohort.eln_intermediate} | Adv: {cohort.eln_adverse}</span>
          <span>Intensive: {cohort.tx_intensive} | HMA+Ven: {cohort.tx_hma_ven}</span>
          <span>CR: {Math.round(cohort.cr_rate * 100)}%</span>
          <span>OS-2yr: {Math.round(cohort.os_2yr * 100)}%</span>
          {cohort.monosomy7_count > 0 && <span style={{ color: '#bf360c', fontWeight: 700 }}>Mono-7: {cohort.monosomy7_count}</span>}
          {cohort.complex_karyotype_count > 0 && <span style={{ color: '#006064', fontWeight: 700 }}>Complex kar: {cohort.complex_karyotype_count}</span>}
        </div>
      )}
    </div>
  );
}

export default function HereditaryAMLPredispositionAtlasPage() {
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
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#bf360c 100%)', color: '#fff', borderRadius: 8, padding: '20px 24px', marginBottom: 20 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>🧬 Hereditary AML Predisposition Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.9 }}>
          Complete 8-Gene Reference · CEBPA · DDX41 · RUNX1 · GATA2 · TP53 · ANKRD26 · ETV6 · SAMD9L · 320-Patient Aggregate · Seeds 3662–3669
        </div>
        <div style={{ fontSize: 11, marginTop: 4, opacity: 0.75 }}>
          CEBPA-biallelic-N+C-PATHOGNOMONIC · DDX41-late-onset-55-65yr-HALLMARK · RUNX1-thrombocytopenia-PATHOGNOMONIC · GATA2-NK-lymphopenia-PATHOGNOMONIC · TP53-complex-karyotype-PATHOGNOMONIC · ANKRD26-5UTR-PANEL-MANDATORY · ETV6-RUNX1-fusion-ABSENT-germline · SAMD9L-monosomy7-80pct-PATHOGNOMONIC
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

          {/* ELN / cytogenetics / treatment */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#fff8e1', borderRadius: 8, padding: 16 }}>
              <b>ELN 2022 Risk Distribution</b>
              {overview.eln_distribution && Object.entries(overview.eln_distribution).filter(([k]) => !k.includes('pct')).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 13 }}>
                  <span>{k.replace(/_/g, ' ')}</span><span style={{ fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>
            <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16 }}>
              <b>Cytogenetics (AML-specific)</b>
              {overview.cytogenetic_distribution && Object.entries(overview.cytogenetic_distribution).map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', marginTop: 6, fontSize: 13 }}>
                  <span>{k.replace(/_/g, ' ')}</span><span style={{ fontWeight: 700 }}>{v}</span>
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
              </div>
            </div>
          )}

          {/* Per-gene summary */}
          <h3>Per-Gene Summary</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Locus', 'n', 'Mean Age', 'ELN Fav%', 'ELN Adv%', 'Intensive%', 'HMA+Ven%', 'CR%', 'OS-2yr%', 'Mono-7', 'Complex kar', 'Key Rule'].map(h => (
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
                  <td style={{ padding: '7px 10px' }}>{row.eln_fav_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.eln_adv_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.intensive_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.hma_ven_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.cr_pct}%</td>
                  <td style={{ padding: '7px 10px' }}>{row.os_2yr_pct}%</td>
                  <td style={{ padding: '7px 10px', color: row.monosomy7_count > 0 ? '#bf360c' : '#999', fontWeight: row.monosomy7_count > 0 ? 700 : 400 }}>{row.monosomy7_count}</td>
                  <td style={{ padding: '7px 10px', color: row.complex_karyotype_count > 0 ? '#006064' : '#999', fontWeight: row.complex_karyotype_count > 0 ? 700 : 400 }}>{row.complex_karyotype_count}</td>
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
                    <span style={{ background: GENE_COLORS[geneName] || '#666', color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 11, minWidth: 70, textAlign: 'center' }}>{geneName}</span>
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
              <tr style={{ background: '#bf360c', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Syndrome', 'Inheritance', 'AML Risk', 'Pathognomonic', 'Key Rule', 'Targeted Rx'].map(h => (
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
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{g.aml_risk}</td>
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
          <h3>Clinical Atlas — 8-Gene AML Predisposition Cards</h3>
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
