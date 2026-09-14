'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-sphingolipidosis-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GBA1':  '#1a237e',  // deep indigo — Gaucher; glucosylceramide; crinkled Gaucher cells; ERT pioneer
  'GLA':   '#880e4f',  // deep pink — Fabry; Gb3; angiokeratoma; XLR; agalsidase; migalastat
  'HEXA':  '#b71c1c',  // deep red — Tay-Sachs; GM2; cherry-red spot; Ashkenazi founder; no ERT
  'HEXB':  '#bf360c',  // deep orange-red — Sandhoff; GM2+GA2; both HexA+HexB low; no ethnic predilection
  'GLB1':  '#1b5e20',  // dark green — GM1; facial coarsening at birth; hepatomegaly; Morquio B same gene
  'SMPD1': '#4a148c',  // deep purple — NPD-A/B; sphingomyelin; foam cells; olipudase ERT 2022
  'NPC1':  '#006064',  // dark teal — NPC; cholesterol trafficking; VSGP; cataplexy; filipin; miglustat
  'ASAH1': '#3e2723',  // dark brown — Farber; ceramide; nodules+joints+hoarseness triad; rarest
};

const GENE_INFO = {
  'GBA1':  { full: 'GBA1 / Glucocerebrosidase / 497aa', locus: '1q22', size: '497 aa / 62 kDa', inh: 'AR', disease: 'Gaucher Disease; glucosylceramide accumulates in macrophages; Gaucher cells (crinkled-tissue-paper) PATHOGNOMONIC; hepatosplenomegaly + bone disease + cytopenias; Imiglucerase ERT FDA1994; Eliglustat SRT FDA2014; GBA1 heterozygous = 5× PD risk' },
  'GLA':   { full: 'GLA / Alpha-Galactosidase-A / 429aa', locus: 'Xq22.1', size: '429 aa / 50 kDa', inh: 'XLR', disease: 'Fabry Disease; Gb3 accumulates in endothelium/neurons/podocytes; angiokeratoma + cornea verticillata + zebra bodies PATHOGNOMONIC; neuropathic pain; Agalsidase beta ERT FDA2003; Migalastat FDA2018 (amenable variants only); GLA unreliable in females — sequence directly' },
  'HEXA':  { full: 'HEXA / Hexosaminidase-A-alpha / 529aa', locus: '15q23', size: '529 aa / 60 kDa', inh: 'AR', disease: 'Tay-Sachs Disease (GM2 gangliosidosis Type I); GM2 accumulates in neurons; cherry-red spot + hyperacusis PATHOGNOMONIC; Ashkenazi Jewish carrier 1:30; NO ERT; carrier screening is primary prevention; adult TSD = motor neuron disease + psychosis' },
  'HEXB':  { full: 'HEXB / Hexosaminidase-B-beta / 556aa', locus: '5q13.3', size: '556 aa / 63 kDa', inh: 'AR', disease: 'Sandhoff Disease (GM2 gangliosidosis Type II); GM2+GA2 accumulate; BOTH HexA AND HexB low (vs TSD HexA only); cherry-red spot; mild hepatosplenomegaly; NO ethnic founder effect; NO ERT; clinically indistinguishable from TSD' },
  'GLB1':  { full: 'GLB1 / Beta-Galactosidase / 677aa', locus: '3p22.3', size: '677 aa / 76 kDa', inh: 'AR', disease: 'GM1 Gangliosidosis / Morquio B (same gene); facial coarsening AT BIRTH PATHOGNOMONIC; cherry-red spot ~50%; hepatosplenomegaly; skeletal dysplasia; Morquio B = residual activity → skeletal only; odontoid hypoplasia mandatory screening; NO approved ERT' },
  'SMPD1': { full: 'SMPD1 / Acid-Sphingomyelinase / 629aa', locus: '11p15.4', size: '629 aa / 70 kDa', inh: 'AR', disease: 'Niemann-Pick A/B (ASMD); sphingomyelin accumulates in macrophages; foam cells PATHOGNOMONIC; NPD-A: severe neuronopathic, no therapy; NPD-B: Olipudase alfa ERT FDA/EMA 2022 (LOW initial dose 0.03 mg/kg CRITICAL); NOT NPC (different disease/gene)' },
  'NPC1':  { full: 'NPC1 / Niemann-Pick-C1 / 1278aa', locus: '18q11.2', size: '1278 aa / 145 kDa', inh: 'AR', disease: 'Niemann-Pick C; CHOLESTEROL TRAFFICKING (not enzyme deficiency); VSGP + gelastic cataplexy PATHOGNOMONIC; Filipin test gold standard; Plasma oxysterols preferred screening; Miglustat EU/Canada approved; Cyclodextrin (HPbetaCD) investigational; I1061T most common allele' },
  'ASAH1': { full: 'ASAH1 / Acid-Ceramidase / 395aa', locus: '8p22', size: '395 aa / 53 kDa', inh: 'AR', disease: 'Farber Disease; ceramide accumulates; Farber TRIAD PATHOGNOMONIC (periarticular nodules + joint deformity + hoarse voice); hoarse cry first months = earliest clue; HSCT improves visceral/joints NOT neurological; SMA-PME = same gene hypomorphic; rarest sphingolipidosis' },
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

export default function HereditarySphingolipidosisAtlasPage() {
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
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: accent, marginBottom: 6 }}>
            🧬 Hereditary Sphingolipidosis Atlas
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>
            8-Gene Reference · GBA1 · GLA · HEXA · HEXB · GLB1 · SMPD1 · NPC1 · ASAH1 · 320 patients (8×40) · seeds 2622–2629
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8 }}>
            {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} />)}
          </div>
          {/* Tabs */}
          <div style={{ display: 'flex', gap: 8, borderBottom: '1px solid #334155', paddingBottom: 0, marginTop: 12 }}>
            {TABS.map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                background: tab === t ? accent : 'transparent',
                color: tab === t ? '#0f172a' : '#94a3b8',
                border: 'none', padding: '6px 16px', borderRadius: '4px 4px 0 0',
                fontWeight: tab === t ? 700 : 400, cursor: 'pointer', fontSize: 13,
              }}>{t}</button>
            ))}
          </div>
        </div>

        {loading && <div style={{ color: '#94a3b8', padding: 24 }}>Loading...</div>}
        {err && <div style={{ color: '#ef4444', padding: 12 }}>Error: {err}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" />
              <MetricCard label="Genes" value={overview.n_genes} sub="GBA1–ASAH1" />
              <MetricCard label="Overall Hepatosplenomegaly" value={`${overview.aggregate_stats?.overall_hepatosplenomegaly_pct}%`} />
              <MetricCard label="Overall Cherry-Red Spot" value={`${overview.aggregate_stats?.overall_cherry_red_spot_pct}%`} />
              <MetricCard label="Seeds" value={overview.seeds} sub="sequential" />
            </div>

            {/* Disease Classes */}
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Disease Classes by Gene</div>
              {overview.disease_classes?.map((dc, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderBottom: '1px solid #1e293b', padding: '5px 0' }}>
                  {dc}
                </div>
              ))}
            </div>

            {/* Key Clinical Distinctions */}
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Key Clinical Distinctions</div>
              {overview.key_clinical_distinctions?.map((kd, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderBottom: '1px solid #1e293b', padding: '6px 0', lineHeight: 1.5 }}>
                  {kd}
                </div>
              ))}
            </div>

            {/* Gene Summary Table */}
            <div style={{ background: card, borderRadius: 8, padding: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Gene Summary — 40 Patients Each</div>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: '#64748b' }}>
                    <th style={{ textAlign: 'left', padding: '4px 8px' }}>Gene</th>
                    <th style={{ textAlign: 'left', padding: '4px 8px' }}>Locus</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>N</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Avg Onset (yr)</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Hepatosplenomegaly%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Seizure%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Cherry-Red%</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((gs, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : card }}>
                      <td style={{ padding: '4px 8px' }}><GeneChip gene={gs.gene} /></td>
                      <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{gs.locus}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.n_patients}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.avg_onset_age}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.hepatosplenomegaly_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.seizure_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.cherry_red_spot_pct}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
            {breakdown.gene_breakdowns?.map((gb, i) => {
              const info = GENE_INFO[gb.gene] || {};
              return (
                <div key={i} style={{ background: card, borderRadius: 8, padding: 16, borderLeft: `4px solid ${GENE_COLORS[gb.gene] || '#555'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gb.gene] || accent, fontSize: 15, marginBottom: 4 }}>
                    {gb.gene}
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>
                    {info.locus} · {info.size} · {info.inh}
                  </div>
                  <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 8, lineHeight: 1.5 }}>
                    {info.disease}
                  </div>
                  <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#94a3b8' }}>
                    <span>Onset: {gb.avg_onset_age}yr</span>
                    <span>Hepato: {gb.hepatosplenomegaly_pct}%</span>
                    <span>Seizure: {gb.seizure_pct}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {/* Gene selector */}
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
              {breakdown.gene_breakdowns?.map(gb => (
                <button key={gb.gene} onClick={() => setSelGene(gb.gene === selGene ? null : gb.gene)}
                  style={{
                    background: selGene === gb.gene ? GENE_COLORS[gb.gene] : 'transparent',
                    color: selGene === gb.gene ? '#fff' : '#94a3b8',
                    border: `1px solid ${GENE_COLORS[gb.gene] || '#334155'}`,
                    padding: '4px 12px', borderRadius: 4, cursor: 'pointer', fontSize: 12, fontWeight: 600,
                  }}>{gb.gene}</button>
              ))}
            </div>

            {breakdown.gene_breakdowns?.filter(gb => !selGene || gb.gene === selGene).map((gb, i) => (
              <div key={i} style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[gb.gene] || '#555'}` }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gb.gene] || accent, fontSize: 16, marginBottom: 8 }}>
                  {gb.gene} — {gb.disease_category?.split(';')[0]}
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>INHERITANCE / PREVALENCE</div>
                    <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.5 }}>{gb.inheritance}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>PATHOGNOMONIC FEATURES</div>
                    <div style={{ fontSize: 12, color: '#fbbf24', lineHeight: 1.5 }}>{gb.pathognomonic?.substring(0, 400)}...</div>
                  </div>
                </div>
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>TREATMENT</div>
                  <div style={{ fontSize: 12, color: '#86efac', lineHeight: 1.5 }}>{gb.treatment?.substring(0, 500)}...</div>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>KEY FEATURES</div>
                  {gb.key_features?.map((kf, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#e2e8f0', padding: '2px 0' }}>• {kf}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>KEY DDx</div>
                  {gb.key_ddx?.map((kd, j) => (
                    <div key={j} style={{ fontSize: 12, color: '#f87171', padding: '2px 0' }}>• {kd}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* Gene entries */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontWeight: 700, color: accent, fontSize: 15, marginBottom: 12 }}>Gene Reference Entries</div>
              {Object.entries(definitions.gene_entries || {}).map(([gene, entry]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || accent, marginBottom: 4 }}>{gene} — {entry.disease_name}</div>
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>{entry.locus} · {entry.protein_size} · {entry.inheritance}</div>
                  <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, marginBottom: 6 }}>{entry.disease_pathway?.substring(0, 300)}...</div>
                  <div style={{ fontSize: 12, color: '#fbbf24', lineHeight: 1.5 }}>{entry.pathognomonic?.substring(0, 250)}...</div>
                </div>
              ))}
            </div>

            {/* Glossary */}
            <div style={{ fontWeight: 700, color: accent, fontSize: 15, marginBottom: 12 }}>Sphingolipidosis Glossary</div>
            {Object.entries(definitions.sphingolipidosis_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: card, borderRadius: 8, padding: 14, marginBottom: 12 }}>
                <div style={{ fontWeight: 700, color: '#7dd3fc', marginBottom: 6 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{def}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
