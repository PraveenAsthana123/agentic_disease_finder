'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mds-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GATA2':   '#1a237e',  // deep indigo           — MonoMAC; MDS 30-70x HIGHEST; HSCT curative; NK lymphopenia PATHOGNOMONIC
  'DDX41':   '#004d40',  // deep teal             — Familial MDS/AML; late onset 55-65yr HALLMARK; HMA+venetoclax
  'SAMD9L':  '#e65100',  // deep orange           — ATAD; monosomy 7; cerebellar ataxia PATHOGNOMONIC; reversion mosaicism
  'RUNX1':   '#880e4f',  // deep pink             — FPD; thrombocytopenia PATHOGNOMONIC; sibling donors CI; azacitidine
  'ETV6':    '#006064',  // dark cyan             — THC5; thrombocytopenia + MDS 10-15%; avoid NSAIDs; no ETV6-RUNX1 germline
  'ANKRD26': '#4a148c',  // deep purple           — THC2; 5-UTR promoter; standard sequencing MISSES; MDS 5-10%
  'TERT':    '#bf360c',  // deep orange-red       — TBD/DC; AVOID alkylating ABSOLUTELY; danazol 70%; fludarabine RIC
  'TP53':    '#33691e',  // olive green           — LFS; AVOID RADIATION ABSOLUTELY; venetoclax+decitabine; WB-MRI Toronto
};

const GENE_INFO = {
  'GATA2':   { full: 'MonoMAC-Emberger / MDS-30-70x-HIGHEST / NK-lymphopenia-PATHOGNOMONIC / monosomy7-del7q-PATHOGNOMONIC / HSCT-curative-early / HPV-vaccine / EBV-CMV-NTM-Q3M', locus: '3q21.3', size: '480 aa / 50 kDa', inh: 'AD LOF heterozygous' },
  'DDX41':   { full: 'Familial-MDS-AML / late-onset-55-65yr-HALLMARK / R525H-50pct-Ashkenazi / HMA-venetoclax-preferred / donor-DDX41-MANDATORY / family-cascade->40yr', locus: '5q35.3', size: '622 aa / 70 kDa', inh: 'AD LOF heterozygous' },
  'SAMD9L':  { full: 'ATAD-Syndrome / monosomy7->80pct / cerebellar-ataxia-PATHOGNOMONIC / reversion-mosaicism / HSCT-corrects-MDS-NOT-ataxia / adrenal-Sick-Day-Rule-10x / G-CSF-CAUTION', locus: '7q21.2', size: '1589 aa / 175 kDa', inh: 'AD GOF heterozygous' },
  'RUNX1':   { full: 'FPD / thrombocytopenia-PATHOGNOMONIC / MDS-AML-30-50pct / sibling-donors-ABSOLUTELY-CI / azacitidine-preferred / avoid-aspirin-NSAIDs / dense-granule-deficiency', locus: '21q22.12', size: '453 aa / 51 kDa', inh: 'AD LOF heterozygous' },
  'ETV6':    { full: 'THC5 / thrombocytopenia-plt30-150k / MDS-10-15pct / ETV6-RUNX1-ABSENT-germline-ETV6 / avoid-NSAIDs / avoid-high-dose-MTX / annual-CBC / HSCT-MDS-progression', locus: '12p13.2', size: '452 aa / 57 kDa', inh: 'AD LOF heterozygous' },
  'ANKRD26': { full: 'THC2 / 5-UTR-promoter-ALL-variants / standard-WES-MISSES / targeted-5UTR-panel-MANDATORY / MDS-AML-5-10pct / c.-127A>T-c.-128A>T-c.-126A>G / IPSS-R-risk', locus: '10p12.1', size: '1710 aa / 190 kDa', inh: 'AD LOF — 5-UTR promoter only' },
  'TERT':    { full: 'TBD-DC / aplastic-anaemia-MDS / PF-20-30pct / short-telomeres-flow-FISH-PATHOGNOMONIC / AVOID-alkylating-ABSOLUTELY / fludarabine-RIC / danazol-70pct / eltrombopag', locus: '5p15.33', size: '1132 aa / 127 kDa', inh: 'AD LOF (TBD); AR biallelic = Hoyeraal-Hreidarsson' },
  'TP53':    { full: 'LFS / MDS-10-20x / complex-karyotype / AVOID-RADIATION-ABSOLUTELY / WB-MRI-Toronto-annually / venetoclax-decitabine-preferred / APR-246-Phase-III / fludarabine-busulfan-HSCT', locus: '17p13.1', size: '393 aa / 43 kDa', inh: 'AD LOF heterozygous' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 130, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryMDSPredispositionAtlasPage() {
  const [tab, setTab]                  = useState('Overview');
  const [overview, setOverview]        = useState(null);
  const [breakdown, setBreakdown]      = useState(null);
  const [definitions, setDefinitions]  = useState(null);
  const [loading, setLoading]          = useState(false);
  const [error, setError]              = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError('');
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '8px 18px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, background: tab === t ? '#1a237e' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary MDS Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ padding: '28px 32px', fontFamily: 'Inter, sans-serif', background: '#f8f9fa', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1a237e', margin: 0 }}>
          🧬 Hereditary MDS Predisposition Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555', marginTop: 4 }}>
          Complete 8-Gene Reference — GATA2 · DDX41 · SAMD9L · RUNX1 · ETV6 · ANKRD26 · TERT · TP53
          &nbsp;|&nbsp; 320-patient aggregate · seeds 3638–3645
        </div>
        <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
          Myelodysplastic Syndrome (MDS) germline predisposition · HSCT · HMA · Venetoclax · Danazol · Azacitidine
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="Total Patients"    value={overview.total_patients}       sub="8 genes × 40"          color="#1a237e" />
            <StatCard label="CR Rate"           value={`${overview.cr_pct}%`}         sub={`${overview.cr_n} pts`} color="#004d40" />
            <StatCard label="2-yr OS"           value={`${overview.os2_pct}%`}        sub={`${overview.os2_n} pts`} color="#880e4f" />
            <StatCard label="Monosomy 7"        value={`${overview.monosomy7_pct}%`}  sub={`${overview.monosomy7_n} pts`} color="#e65100" />
            <StatCard label="Complex Karyotype" value={`${overview.complex_karyotype_pct}%`} sub={`${overview.complex_karyotype_n} pts`} color="#bf360c" />
            <StatCard label="HSCT"              value={`${overview.hsct_pct}%`}       sub={`${overview.hsct_n} pts`} color="#006064" />
            <StatCard label="HMA Therapy"       value={`${overview.hma_pct}%`}        sub={`${overview.hma_n} pts`}  color="#4a148c" />
            <StatCard label="Venetoclax"        value={`${overview.venetoclax_pct}%`} sub={`${overview.venetoclax_n} pts`} color="#33691e" />
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="IPSS-R High/VHigh" value={`${overview.ipss_high_vh_pct}%`} sub={`${overview.ipss_high_vh_n} pts`} color="#1a237e" />
            <StatCard label="AML Progression"   value={`${overview.aml_prog_pct}%`}     sub={`${overview.aml_prog_n} pts`}      color="#bf360c" />
            <StatCard label="Thrombocytopenia"  value={`${overview.thrombocytopenia_pct}%`} sub={`${overview.thrombocytopenia_n} pts`} color="#880e4f" />
          </div>

          {/* Per-gene summary */}
          <div style={{ background: '#fff', borderRadius: 10, padding: 20, boxShadow: '0 1px 4px #0001' }}>
            <div style={{ fontWeight: 700, marginBottom: 14, color: '#1a237e' }}>Per-Gene Summary (8 genes × 40 patients)</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  {['Gene', 'Locus', 'Inheritance', 'N', 'HSCT', 'HMA', 'Venetoclax', 'Monosomy7', 'AML Prog', 'MDS Risk'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 700, color: '#444', borderBottom: '2px solid #eee' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(overview.gene_counts).map(([gene, c]) => (
                  <tr key={gene} style={{ borderBottom: '1px solid #f0f0f0' }}
                      onClick={() => setExpandedGene(expandedGene === gene ? null : gene)}
                      style={{ borderBottom: '1px solid #f0f0f0', cursor: 'pointer', background: expandedGene === gene ? `${GENE_COLORS[gene]}08` : 'transparent' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] }}>
                      {gene}
                      <div style={{ fontSize: 10, fontWeight: 400, color: '#888' }}>{GENE_INFO[gene]?.locus}</div>
                    </td>
                    <td style={{ padding: '8px 10px', color: '#666', fontSize: 11 }}>{GENE_INFO[gene]?.locus}</td>
                    <td style={{ padding: '8px 10px', color: '#666', fontSize: 11 }}>{GENE_INFO[gene]?.inh}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 600 }}>{c.n}</td>
                    <td style={{ padding: '8px 10px' }}>{c.hsct_n}</td>
                    <td style={{ padding: '8px 10px' }}>{c.hma_n}</td>
                    <td style={{ padding: '8px 10px' }}>{c.venetoclax_n}</td>
                    <td style={{ padding: '8px 10px' }}>{c.monosomy7_n}</td>
                    <td style={{ padding: '8px 10px' }}>{c.aml_prog_n}</td>
                    <td style={{ padding: '8px 10px', fontSize: 11, color: '#555', maxWidth: 200 }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200 }}>{c.mds_risk?.slice(0, 60)}…</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {Object.entries(breakdown.gene_data).map(([gene, c]) => (
            <div key={gene} style={{ background: '#fff', borderRadius: 10, padding: 18, marginBottom: 14,
              boxShadow: '0 1px 4px #0001', borderLeft: `4px solid ${GENE_COLORS[gene]}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                <div>
                  <span style={{ fontWeight: 800, fontSize: 16, color: GENE_COLORS[gene] }}>{gene}</span>
                  <Badge text={GENE_INFO[gene]?.locus} color={GENE_COLORS[gene]} />
                  <Badge text={GENE_INFO[gene]?.size}  color="#666" />
                  <Badge text={c.inheritance}           color="#555" />
                </div>
                <button onClick={() => setExpandedGene(expandedGene === gene ? null : gene)}
                  style={{ background: GENE_COLORS[gene], color: '#fff', border: 'none', borderRadius: 6,
                    padding: '4px 12px', fontSize: 11, cursor: 'pointer' }}>
                  {expandedGene === gene ? 'Collapse' : 'Expand'}
                </button>
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', fontSize: 12 }}>
                {[
                  ['Patients', c.n],
                  ['Age Range', `${c.age_min}–${c.age_max}yr`],
                  ['Median Age', `${c.age_median}yr`],
                  ['HSCT', c.hsct_n],
                  ['HMA', c.hma_n],
                  ['Venetoclax', c.venetoclax_n],
                  ['Monosomy 7', c.monosomy7_n],
                  ['Complex Kary', c.complex_karyotype_n],
                  ['AML Prog', c.aml_prog_n],
                  ['IPSS High/VH', c.ipss_high_vh_n],
                  ['CR', c.cr_n],
                  ['2yr-OS', c.os2_n],
                  ...(c.danazol_n > 0 ? [['Danazol', c.danazol_n]] : []),
                ].map(([k, v]) => (
                  <div key={k} style={{ background: `${GENE_COLORS[gene]}0d`, borderRadius: 6, padding: '4px 10px' }}>
                    <span style={{ color: '#888' }}>{k}: </span>
                    <span style={{ fontWeight: 700, color: GENE_COLORS[gene] }}>{v}</span>
                  </div>
                ))}
              </div>
              {expandedGene === gene && (
                <div style={{ marginTop: 12, padding: 12, background: `${GENE_COLORS[gene]}08`, borderRadius: 8, fontSize: 12 }}>
                  <div style={{ marginBottom: 6 }}><strong>MDS Risk:</strong> {c.mds_risk}</div>
                  <div style={{ marginBottom: 6 }}><strong>Pathognomonic:</strong> {c.pathognomonic}</div>
                  <div style={{ marginBottom: 6 }}><strong>Key Rule:</strong> <span style={{ color: GENE_COLORS[gene], fontWeight: 600 }}>{c.key_rule}</span></div>
                  <div style={{ marginBottom: 6 }}><strong>Key Avoid:</strong> <span style={{ color: '#c00', fontWeight: 600 }}>{c.key_avoid}</span></div>
                  <div style={{ marginBottom: 6 }}><strong>Surveillance:</strong> {c.surveillance}</div>
                  <div><strong>Targeted Rx:</strong> {c.targeted_rx}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && definitions && (
        <div>
          <div style={{ marginBottom: 16, fontSize: 13, color: '#666' }}>
            8 Key Clinical Distinctions — Hereditary MDS Predisposition
          </div>
          {definitions.key_clinical_distinctions?.map((d, i) => (
            <div key={i} style={{ background: '#fff', borderRadius: 8, padding: '12px 16px', marginBottom: 10,
              boxShadow: '0 1px 3px #0001', borderLeft: `4px solid ${Object.values(GENE_COLORS)[i % 8]}` }}>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.6 }}>{d}</div>
            </div>
          ))}

          <div style={{ marginTop: 24, fontWeight: 700, color: '#1a237e', marginBottom: 10 }}>Gene Molecular Profiles</div>
          {Object.entries(definitions.gene_defs || {}).map(([gene, profile]) => (
            <div key={gene} style={{ background: '#fff', borderRadius: 8, padding: '12px 16px', marginBottom: 8,
              boxShadow: '0 1px 3px #0001' }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 6 }}>{gene}</div>
              <div style={{ fontSize: 11, color: '#555', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                {expandedGene === gene ? profile : profile.slice(0, 350) + '…'}
              </div>
              <button onClick={() => setExpandedGene(expandedGene === gene ? null : gene)}
                style={{ marginTop: 6, background: 'none', border: `1px solid ${GENE_COLORS[gene]}55`,
                  color: GENE_COLORS[gene], borderRadius: 4, padding: '2px 8px', fontSize: 11, cursor: 'pointer' }}>
                {expandedGene === gene ? 'Less' : 'More'}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          {definitions.definitions?.map((d, i) => {
            const gene = d.term?.split(' — ')[0];
            return (
              <div key={i} style={{ background: '#fff', borderRadius: 10, padding: 16, marginBottom: 10,
                boxShadow: '0 1px 4px #0001', borderLeft: `4px solid ${GENE_COLORS[gene] || '#888'}` }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#444', marginBottom: 8, fontSize: 14 }}
                     onClick={() => setExpandedDef(expandedDef === i ? null : i)}
                     style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#444', marginBottom: 8, fontSize: 14, cursor: 'pointer' }}>
                  {d.term}
                </div>
                {(expandedDef === i || true) && (
                  <pre style={{ fontSize: 11, color: '#444', whiteSpace: 'pre-wrap', lineHeight: 1.7, margin: 0 }}>
                    {d.definition}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
