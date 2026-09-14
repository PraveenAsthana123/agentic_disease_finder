'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-glycoprotein-storage-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MAN2B1': '#1a237e',  // deep indigo — Alpha-mannosidosis; vacuolated lymphocytes; recurrent infections; velmanase alfa EMA2018
  'MANBA':  '#880e4f',  // deep magenta — Beta-mannosidosis; Man-GlcNAc disaccharide; neonatal severe or adult angiokeratoma
  'FUCA1':  '#b71c1c',  // deep red — Fucosidosis; angiokeratoma; globus pallidus T2; HSCT; pseudo-deficiency pitfall
  'NEU1':   '#e65100',  // deep orange — Sialidosis; cherry-red spot + myoclonus + NORMAL IQ (Type 1); CTSA complex
  'AGA':    '#1b5e20',  // dark green — Aspartylglucosaminuria; Finnish founder; biphasic course; GlcNAc-Asn dipeptide
  'NAGA':   '#4a148c',  // deep purple — Schindler/Kanzaki; GalNAc-glycopeptides; Type I infantile / Type II adult angiokeratoma
  'GNPTAB': '#006064',  // dark teal — ML II I-cell disease; MULTIPLE plasma lysosomal enzymes 10-40× PATHOGNOMONIC
  'MCOLN1': '#3e2723',  // dark brown — ML IV; TRPML1 channel; corneal clouding from birth; gastrin >1000 PATHOGNOMONIC; Ashkenazi
};

const GENE_INFO = {
  'MAN2B1': { full: 'MAN2B1 / Lysosomal Alpha-D-Mannosidase / 867aa', locus: '19p13.13', size: '867 aa / 114 kDa', inh: 'AR', disease: 'Alpha-mannosidosis; lysosomal alpha-mannosidase deficiency; 1:500,000; intellectual disability + recurrent bacterial infections + coarse facies; VACUOLATED LYMPHOCYTES on blood smear; urine mannose-rich oligosaccharides (Man2-Man6GlcNAc2); SNHL >75% by adolescence; psychiatric symptoms ~25%; Velmanase alfa (Lamzede EMA 2018) IV every 2 weeks; HSCT for young severe cases; p.Arg750Trp common European allele' },
  'MANBA':  { full: 'MANBA / Lysosomal Beta-D-Mannosidase / 879aa', locus: '4q22-4q25', size: '879 aa / 100 kDa', inh: 'AR', disease: 'Beta-mannosidosis; very rare (<100 families); highly variable: severe neonatal (respiratory failure, profound hypotonia) to mild adult (angiokeratoma + SNHL + ID); urine Man-beta-1,4-GlcNAc DISACCHARIDE — pathognomonic small band on TLC; NO approved ERT; caprine model well-characterized; supportive care only' },
  'FUCA1':  { full: 'FUCA1 / Lysosomal Alpha-L-Fucosidase / 466aa', locus: '1p36.11', size: '466 aa / 53 kDa', inh: 'AR', disease: 'Fucosidosis; 1:200,000 (higher in Calabria/Cuba/Spain); Type 1 severe / Type 2 milder; ANGIOKERATOMA CORPORIS DIFFUSUM (Type 2); urine fucose-containing oligosaccharides; MRI T2 HYPERINTENSITY GLOBUS PALLIDUS; elevated sweat chloride clue; PSEUDO-DEFICIENCY ALLELES (Ala287Thr — low enzyme but NO disease; confirm with urine oligos); HSCT standard of care for severe early onset' },
  'NEU1':   { full: 'NEU1 / Lysosomal Sialidase (Neuraminidase-1) / 415aa', locus: '6p21.33', size: '415 aa / 45 kDa', inh: 'AR', disease: 'Sialidosis (Mucolipidosis I); requires CTSA (protective protein) for activation; Type 1 (adult): CHERRY-RED SPOT + ACTION MYOCLONUS + NORMAL INTELLECT — cardinal triad; Type 2 (childhood): coarse facies + ID + cherry-red; urine sialyloligosaccharides (NeuAc-containing); if BOTH NEU1+GLB1 low → CTSA deficiency (galactosialidosis); myoclonus: clonazepam + levetiracetam; no ERT' },
  'AGA':    { full: 'AGA / Lysosomal Aspartylglucosaminidase / 346aa', locus: '4q34.3', size: '346 aa / 24 kDa (alpha2beta2)', inh: 'AR', disease: 'Aspartylglucosaminuria (AGU); 1:18,000 in Finland (Finnish founder p.Cys163Ser); very rare outside Finland; BIPHASIC COURSE: apparently normal infancy → progressive REGRESSION from 5-10yr; urine GlcNAc-Asn (aspartylglucosamine) dipeptide by GC-MS/LC-MS; coarse features DEVELOP progressively NOT prominent in infancy; progressive ID to profound; survival to 50-60yr; no approved ERT' },
  'NAGA':   { full: 'NAGA / Lysosomal Alpha-N-Acetylgalactosaminidase / 411aa', locus: '22q13.2', size: '411 aa / 48 kDa', inh: 'AR', disease: 'Schindler disease (Type I) / Kanzaki disease (Type II); very rare; Type I: infantile neurodegeneration (regression 1-3yr, myoclonus, cortical blindness, autistic features); Type II: adult ANGIOKERATOMA + SNHL + peripheral neuropathy + lymphedema; urine GalNAc-sialyl-glycopeptides; NAGA enzyme use GalNAc-specific substrate (not GLA Gal-substrate); p.Glu325Lys Dutch/Type I; p.Arg329Gln Japanese/Type II; no approved ERT' },
  'GNPTAB': { full: 'GNPTAB / GlcNAc-1-Phosphotransferase Alpha/Beta / 1256aa', locus: '12q23.2', size: '1256 aa / 277 kDa', inh: 'AR', disease: 'Mucolipidosis II (I-cell disease) + ML IIIab; GlcNAc-1-phosphotransferase deficiency; NO mannose-6-phosphate tags on lysosomal enzymes → all 60+ enzymes secreted extracellularly; MULTIPLE PLASMA LYSOSOMAL ENZYMES 10-40× ELEVATED — PATHOGNOMONIC; I-CELLS: phase-dense inclusions in fibroblasts; ML II: Hurler-like coarse facies AT BIRTH (not after months), NO corneal clouding; ML III: milder, childhood joint stiffness; NO effective ERT; ML II fatal <10yr' },
  'MCOLN1': { full: 'MCOLN1 / Mucolipin-1 (TRPML1) / 580aa', locus: '19p13.2', size: '580 aa / 65 kDa', inh: 'AR', disease: 'Mucolipidosis IV; Ashkenazi enriched (1:40,000); TRPML1 lysosomal Ca2+-channel; CORNEAL CLOUDING from birth/early infancy >90%; SERUM GASTRIN >1000 pg/mL PATHOGNOMONIC (achlorhydria from parietal cell TRPML1 loss); NORMAL urine metabolites + NORMAL plasma lysosomal enzymes — key differentiators; psychomotor delay from infancy; retinal degeneration 3-5yr; non-progressive CNS; p.Arg403Cys + del6.4kb Ashkenazi founders; no approved ERT; gene therapy preclinical' },
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

export default function HereditaryGlycoproteinStorageAtlasPage() {
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
            🧬 Hereditary Glycoprotein Storage Atlas
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>
            8-Gene Reference · MAN2B1 · MANBA · FUCA1 · NEU1 · AGA · NAGA · GNPTAB · MCOLN1 · 320 patients (8×40) · seeds 2646–2653
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
              <MetricCard label="Genes" value={overview.n_genes} sub="MAN2B1–MCOLN1" />
              <MetricCard label="Hearing Loss%" value={`${overview.aggregate_stats?.overall_hearing_loss_pct}%`} />
              <MetricCard label="Corneal Clouding%" value={`${overview.aggregate_stats?.overall_corneal_clouding_pct}%`} />
              <MetricCard label="Angiokeratoma%" value={`${overview.aggregate_stats?.overall_angiokeratoma_pct}%`} />
              <MetricCard label="Seeds" value={overview.seeds} sub="sequential" />
            </div>

            {/* Disease Classes */}
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Disease Classes by Gene</div>
              {overview.disease_classes?.map((dc, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderBottom: '1px solid #0f172a', padding: '5px 0', lineHeight: 1.5 }}>
                  {dc}
                </div>
              ))}
            </div>

            {/* Key Clinical Distinctions */}
            <div style={{ background: card, borderRadius: 8, padding: 16, marginBottom: 16 }}>
              <div style={{ fontWeight: 700, color: accent, marginBottom: 10 }}>Key Clinical Distinctions</div>
              {overview.key_clinical_distinctions?.map((kd, i) => (
                <div key={i} style={{ fontSize: 12, color: '#cbd5e1', borderBottom: '1px solid #0f172a', padding: '6px 0', lineHeight: 1.5 }}>
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
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Hearing Loss%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Corneal Clouding%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Angiokeratoma%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>ID%</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((gs, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : card }}>
                      <td style={{ padding: '4px 8px' }}><GeneChip gene={gs.gene} /></td>
                      <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{gs.locus}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.n_patients}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.avg_onset_age}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.hearing_loss_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right', color: gs.corneal_clouding_pct > 50 ? '#fbbf24' : '#e2e8f0' }}>{gs.corneal_clouding_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.angiokeratoma_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right', color: '#86efac' }}>{gs.intellectual_disability_pct}%</td>
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
                  <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#94a3b8', flexWrap: 'wrap' }}>
                    <span>Onset: {gb.avg_onset_age}yr</span>
                    <span>N: {gb.n_patients}</span>
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
                {/* Feature rates */}
                {gb.feature_rates && Object.keys(gb.feature_rates).length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>COHORT FEATURE RATES (N=40)</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {Object.entries(gb.feature_rates).map(([feat, pct]) => (
                        <span key={feat} style={{
                          background: pct > 70 ? '#1a3a2a' : pct > 40 ? '#1e2a1e' : '#1e293b',
                          border: `1px solid ${pct > 70 ? '#22c55e' : pct > 40 ? '#86efac' : '#334155'}`,
                          borderRadius: 4, padding: '2px 6px', fontSize: 11, color: '#cbd5e1'
                        }}>
                          {feat.replace(/_/g, ' ')}: {pct}%
                        </span>
                      ))}
                    </div>
                  </div>
                )}
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

            {/* Glycoprotein Storage Glossary */}
            <div style={{ fontWeight: 700, color: accent, fontSize: 15, marginBottom: 12 }}>Glycoprotein Storage Glossary</div>
            {Object.entries(definitions.glycoprotein_storage_glossary || {}).map(([term, def]) => (
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
