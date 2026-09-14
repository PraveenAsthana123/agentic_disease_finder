'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-leukodystrophy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ABCD1':  '#1565c0',  // deep blue — X-ALD; VLCFA; CALD posterior advancing; Skysona FDA2022
  'ARSA':   '#4a148c',  // deep purple — MLD; sulfatide; Libmeldy EMA2020; metachromatic PATHOGNOMONIC
  'GALC':   '#b71c1c',  // deep red — Krabbe; psychosine; globoid cells PATHOGNOMONIC; extreme irritability
  'PLP1':   '#006064',  // dark teal — PMD; nystagmus at birth PATHOGNOMONIC; XLR; duplication
  'GJC2':   '#1b5e20',  // dark green — PMLD; connexin 47; AR; milder than PLP1; SPG44 AD
  'POLR3A': '#e65100',  // burnt orange — 4H; Pol-III; 4H-TRIAD PATHOGNOMONIC; HRT mandatory
  'EIF2B5': '#880e4f',  // dark magenta — VWM; stress-triggered crises PATHOGNOMONIC; ISRIB
  'ADAR':   '#4e342e',  // brown — AGS6; pseudo-TORCH PATHOGNOMONIC; IFN-alpha; JAK inhibitors
};

const GENE_INFO = {
  'ABCD1':  { full: 'ABCD1 / ALDP / 745aa', locus: 'Xq28', size: '745 aa / 84 kDa', inh: 'XLR', disease: 'X-ALD (X-linked Adrenoleukodystrophy); VLCFA accumulate; CALD posterior-advancing MRI → HSCT/Skysona-FDA2022 if Loes≤9+gadolinium; AMN adults; Addison 70% males; NBS RUSP 2016' },
  'ARSA':   { full: 'ARSA / Arylsulfatase-A / 507aa', locus: '22q13.33', size: '507 aa / 62 kDa', inh: 'AR', disease: 'MLD (Metachromatic Leukodystrophy); sulfatide accumulates; metachromatic granules nerve PATHOGNOMONIC; Libmeldy EMA2020 pre-symptomatic; adult MLD = schizophrenia mimicry; pseudodeficiency 10% population' },
  'GALC':   { full: 'GALC / Galactocerebrosidase / 669aa', locus: '14q31.3', size: '669 aa / 74 kDa', inh: 'AR', disease: 'Krabbe (GLD); psychosine cytotoxic; globoid cells PATHOGNOMONIC; extreme irritability infantile (touch→screaming); HSCT pre-symptomatic late-onset; NBS RUSP 2016' },
  'PLP1':   { full: 'PLP1 / Proteolipid Protein 1 / 276aa', locus: 'Xq22.2', size: '276 aa / 30 kDa', inh: 'XLR', disease: 'PMD (Pelizaeus-Merzbacher); nystagmus at birth PATHOGNOMONIC; hypomyelination static; duplication 60-70%; NO disease-modifying therapy; ASO Phase I' },
  'GJC2':   { full: 'GJC2 / Connexin-47 / 436aa', locus: '1q42.13', size: '436 aa / 46 kDa', inh: 'AR/AD', disease: 'PMLD (AR biallelic); SPG44 (AD monoallelic adult); connexin 47 K+ buffering failure; milder hypomyelination than PLP1; nystagmus less prominent; both sexes AR' },
  'POLR3A': { full: 'POLR3A / RNA Pol III subunit A / 1390aa', locus: '10q22.3', size: '1390 aa / 155 kDa', inh: 'AR', disease: '4H syndrome (POLR3-HLD); Pol-III tRNA transcription failure; 4H TRIAD PATHOGNOMONIC: Hypomyelination + Hypodontia + Hypogonadotropic Hypogonadism; cerebellar atrophy; HRT mandatory' },
  'EIF2B5': { full: 'EIF2B5 / eIF2B-epsilon / 721aa', locus: '3q27.1', size: '721 aa / 80 kDa', inh: 'AR', disease: 'VWM/CACH; ISR GEF failure; stress-triggered crises (fever/trauma) PATHOGNOMONIC; white matter vanishes; ovarian failure females; contact sports PROHIBITED; ISRIB preclinical' },
  'ADAR':   { full: 'ADAR / ADAR1 / 1226aa', locus: '1q21.3', size: '1226 aa / 136 kDa', inh: 'AD GOF/AR', disease: 'AGS6 interferonopathy; pseudo-TORCH PATHOGNOMONIC (calcifications+leukodystrophy+negative-TORCH); CT calcifications; IFN-alpha CSF >2 IU/mL; JAK inhibitors baricitinib emerging' },
};

function GeneChip({ gene }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span style={{ background: col, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 12, fontWeight: 700, margin: '0 2px' }}>
      {gene}
    </span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 120 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8' }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryLeukodystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' ? 'breakdown'
      : tab === 'Clinical Atlas' ? 'breakdown'
      : 'overview';
    setLoading(true); setErr(null);
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setErr(String(e)); setLoading(false); });
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#38bdf8';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <h1 style={{ color: accent, fontSize: 20, marginBottom: 4 }}>
        🧬 Hereditary-Leukodystrophy-Atlas
      </h1>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 16 }}>
        Complete 8-Gene Reference · ABCD1 · ARSA · GALC · PLP1 · GJC2 · POLR3A · EIF2B5 · ADAR · 320 Patients · Seeds 2614-2621
      </div>

      {/* Gene chips */}
      <div style={{ marginBottom: 16, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ background: tab === t ? accent : '#1e293b', color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: 700, fontSize: 13 }}>
            {t}
          </button>
        ))}
      </div>

      {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
      {err && <div style={{ color: '#ef4444' }}>Error: {err}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <MetricCard label="Total Genes" value={overview.n_genes} />
            <MetricCard label="Total Patients" value={overview.total_patients} />
            <MetricCard label="Seeds" value={overview.seeds} />
            <MetricCard label="WM Lesion %" value={`${overview.aggregate_stats?.overall_wm_lesion_pct}%`} />
            <MetricCard label="Seizure %" value={`${overview.aggregate_stats?.overall_seizure_pct}%`} />
            <MetricCard label="Spastic %" value={`${overview.aggregate_stats?.overall_spastic_pct}%`} />
          </div>

          <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Key Clinical Distinctions</div>
            {(overview.key_clinical_distinctions || []).map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 6, borderLeft: '3px solid #334155', paddingLeft: 10 }}>
                {d}
              </div>
            ))}
          </div>

          <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Disease Classes</div>
            {(overview.disease_classes || []).map((d, i) => (
              <div key={i} style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>• {d}</div>
            ))}
          </div>

          <div style={{ background: card, borderRadius: 8, padding: 16 }}>
            <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Gene Summary Table</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ color: '#64748b' }}>
                  {['Gene', 'Locus', 'Patients', 'Avg Onset (yr)', 'WM Lesion %', 'Seizure %', 'Spastic %'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '4px 8px', borderBottom: '1px solid #334155' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.gene_summaries || []).map(g => (
                  <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '4px 8px' }}><GeneChip gene={g.gene} /></td>
                    <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{g.locus}</td>
                    <td style={{ padding: '4px 8px', color: '#38bdf8' }}>{g.n_patients}</td>
                    <td style={{ padding: '4px 8px', color: '#f8fafc' }}>{g.avg_onset_age}</td>
                    <td style={{ padding: '4px 8px', color: '#a3e635' }}>{g.wm_lesion_pct}%</td>
                    <td style={{ padding: '4px 8px', color: '#fb923c' }}>{g.seizure_pct}%</td>
                    <td style={{ padding: '4px 8px', color: '#c084fc' }}>{g.spastic_pct}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
            {Object.entries(GENE_INFO).map(([g, info]) => (
              <button key={g} onClick={() => setSelGene(selGene === g ? null : g)}
                style={{ background: selGene === g ? GENE_COLORS[g] : '#1e293b', color: '#fff',
                  border: `1px solid ${GENE_COLORS[g]}`, borderRadius: 6, padding: '4px 12px',
                  cursor: 'pointer', fontSize: 12, fontWeight: 700 }}>
                {g}
              </button>
            ))}
          </div>
          {selGene && GENE_INFO[selGene] && (
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[selGene]}` }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[selGene], fontSize: 15, marginBottom: 8 }}>{GENE_INFO[selGene].full}</div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>Locus: {GENE_INFO[selGene].locus} · Size: {GENE_INFO[selGene].size} · Inheritance: {GENE_INFO[selGene].inh}</div>
              <div style={{ fontSize: 12, color: '#cbd5e1' }}>{GENE_INFO[selGene].disease}</div>
            </div>
          )}
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ color: '#64748b' }}>
                {['Gene', 'Locus', 'Protein', 'Inheritance', 'Disease / Mechanism', 'Patients', 'Onset (yr)'].map(h => (
                  <th key={h} style={{ textAlign: 'left', padding: '6px 8px', borderBottom: '1px solid #334155' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(breakdown.gene_breakdowns || []).map(g => (
                <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b', cursor: 'pointer' }}
                  onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}>
                  <td style={{ padding: '6px 8px' }}><GeneChip gene={g.gene} /></td>
                  <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '6px 8px', color: '#e2e8f0', fontSize: 11 }}>{g.protein_size}</td>
                  <td style={{ padding: '6px 8px', color: '#a3e635' }}>{g.inheritance?.split(';')[0]}</td>
                  <td style={{ padding: '6px 8px', color: '#cbd5e1', maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {g.disease_category?.split(';')[0]}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#38bdf8' }}>{g.n_patients}</td>
                  <td style={{ padding: '6px 8px', color: '#f8fafc' }}>{g.avg_onset_age}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {(breakdown.gene_breakdowns || []).map(g => (
            <div key={g.gene} style={{ background: card, borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <GeneChip gene={g.gene} />
                <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</span>
              </div>
              <div style={{ fontSize: 13, color: '#fbbf24', fontWeight: 600, marginBottom: 6 }}>
                {g.disease_category?.split(';')[0]}
              </div>
              <div style={{ fontSize: 12, color: '#ef4444', marginBottom: 8, background: '#1e1010', borderRadius: 4, padding: '4px 8px' }}>
                ⚠ PATHOGNOMONIC: {g.pathognomonic?.split(';')[0]}
              </div>
              <div style={{ fontSize: 12, color: '#a3e635', marginBottom: 6 }}>
                Rx: {g.treatment?.split(';')[0]}
              </div>
              <div style={{ marginBottom: 6 }}>
                {(g.key_features || []).slice(0, 4).map((f, i) => (
                  <div key={i} style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 2 }}>• {f}</div>
                ))}
              </div>
              <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#64748b' }}>
                <span>n={g.n_patients}</span>
                <span>Onset: {g.avg_onset_age}yr</span>
                <span>WM: {g.wm_lesion_pct}%</span>
                <span>Sz: {g.seizure_pct}%</span>
                <span>Sp: {g.spastic_pct}%</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 24 }}>
            {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
              <div key={gene} style={{ background: card, borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GeneChip gene={gene} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</span>
                </div>
                <div style={{ fontSize: 13, color: '#fbbf24', fontWeight: 600, marginBottom: 6 }}>{entry.disease_name}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 6 }}>{entry.disease_pathway}</div>
                <div style={{ fontSize: 12, color: '#ef4444', marginBottom: 6, background: '#1e1010', borderRadius: 4, padding: '4px 8px' }}>
                  ⚠ {entry.pathognomonic}
                </div>
                <div style={{ fontSize: 12, color: '#a3e635', marginBottom: 6 }}>Rx: {entry.treatment_summary}</div>
                <div style={{ marginTop: 6 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>KEY DDx:</div>
                  {(entry.key_ddx || []).map((d, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>• {d}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {definitions.leukodystrophy_glossary && (
            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 12 }}>Leukodystrophy Glossary</div>
              {Object.entries(definitions.leukodystrophy_glossary).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 14 }}>
                  <div style={{ color: '#fbbf24', fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8' }}>{def}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
