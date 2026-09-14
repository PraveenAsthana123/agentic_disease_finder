'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mps-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'IDUA':   '#1a237e',  // deep indigo — MPS I Hurler/Scheie; DS+HS; laronidase ERT; HSCT curative
  'IDS':    '#880e4f',  // deep pink — MPS II Hunter; XLR; NO corneal clouding PATHOGNOMONIC; pebbly skin
  'SGSH':   '#b71c1c',  // deep red — MPS IIIA Sanfilippo A; HS only; severe behaviour; no ERT
  'GALNS':  '#e65100',  // deep orange — MPS IVA Morquio A; KS+C6S; skeletal; normal intellect; elosulfase
  'ARSB':   '#1b5e20',  // dark green — MPS VI Maroteaux-Lamy; DS only; normal intellect; galsulfase
  'GUSB':   '#4a148c',  // deep purple — MPS VII Sly; DS+HS+CS; hydrops fetalis PATHOGNOMONIC; vestronidase
  'NAGLU':  '#006064',  // dark teal — MPS IIIB Sanfilippo B; HS only; BMN-250 trials
  'HGSNAT': '#3e2723',  // dark brown — MPS IIIC Sanfilippo C; transmembrane enzyme; gene therapy
};

const GENE_INFO = {
  'IDUA':   { full: 'IDUA / Alpha-L-Iduronidase / 653aa', locus: '4p16.3', size: '653 aa / 74 kDa', inh: 'AR', disease: 'MPS I (Hurler / Hurler-Scheie / Scheie); DS + HS accumulate; coarse facies + corneal clouding + hepatosplenomegaly; Laronidase ERT FDA2003; HSCT curative in Hurler IF done <2yr; Scheie/HS = attenuated with near-normal intellect; universal newborn screening in ~25 US states' },
  'IDS':    { full: 'IDS / Iduronate-2-Sulfatase / 550aa', locus: 'Xq28', size: '550 aa / 62 kDa', inh: 'XLR', disease: 'MPS II (Hunter Syndrome); XLR — males primarily affected; DS + HS accumulate; ABSENCE OF CORNEAL CLOUDING PATHOGNOMONIC (distinguishes from MPS I); pebbly ivory skin nodules PATHOGNOMONIC; Idursulfase ERT FDA2006; CNS form: severe cognitive decline; attenuated: normal/near-normal cognition' },
  'SGSH':   { full: 'SGSH / Heparan-N-Sulfatase / 502aa', locus: '17q25.3', size: '502 aa / 56 kDa', inh: 'AR', disease: 'MPS IIIA (Sanfilippo A); HS only substrate; severe behavioural disorder + sleep disturbance (melatonin often needed) CARDINAL; cognitive regression after 3-6yr; NO approved ERT (CNS barrier); gene therapy clinical trials; hyperactivity, aggression, sleep inversion are diagnostic clues; MPS IIIA most severe Sanfilippo subtype' },
  'GALNS':  { full: 'GALNS / N-Acetylgalactosamine-6-Sulfatase / 552aa', locus: '16q24.3', size: '552 aa / 62 kDa', inh: 'AR', disease: 'MPS IVA (Morquio A); KS + C6S accumulate; SEVERE SKELETAL DYSPLASIA with NORMAL INTELLECT PATHOGNOMONIC; odontoid hypoplasia = cervical spine instability → anaesthetic risk → MANDATORY flexion/extension X-ray; Elosulfase alfa ERT FDA2014; short stature + joint laxity + pectus carinatum; no CNS involvement' },
  'ARSB':   { full: 'ARSB / Arylsulfatase-B / 533aa', locus: '5q14.1', size: '533 aa / 59 kDa', inh: 'AR', disease: 'MPS VI (Maroteaux-Lamy Syndrome); DS only substrate (unlike MPS I); NORMAL INTELLECT — key distinguisher from MPS I/II; coarse facies + hepatosplenomegaly + skeletal dysplasia + corneal clouding; Galsulfase ERT FDA2005; severity varies; joint disease + valve disease progressive without ERT' },
  'GUSB':   { full: 'GUSB / Beta-Glucuronidase / 651aa', locus: '7q11.21', size: '651 aa / 75 kDa', inh: 'AR', disease: 'MPS VII (Sly Syndrome); DS + HS + CS accumulate; HYDROPS FETALIS PATHOGNOMONIC (only MPS with this presentation); rarest MPS; Vestronidase alfa ERT FDA2017; NBS expanded to include MPS VII 2024; range from severe neonatal hydrops to mild attenuated; eosinophilic granules in leukocytes on blood smear' },
  'NAGLU':  { full: 'NAGLU / Alpha-N-Acetylglucosaminidase / 743aa', locus: '17q21.2', size: '743 aa / 83 kDa', inh: 'AR', disease: 'MPS IIIB (Sanfilippo B); HS only substrate; clinically IDENTICAL to MPS IIIA; severe behaviour + sleep disorder; NO approved ERT; BMN-250 (CNS-delivered enzyme conjugate) Phase II/III trials; distinguish from IIIA by enzyme assay (NAGLU vs SGSH); milder course possible in some late-onset' },
  'HGSNAT': { full: 'HGSNAT / Heparan-Acetyl-CoA-Glucosaminide-Acetyltransferase / 635aa', locus: '8p11.21', size: '635 aa / 73 kDa', inh: 'AR', disease: 'MPS IIIC (Sanfilippo C); HS only substrate; TRANSMEMBRANE ENZYME (not soluble) — ERT INFEASIBLE, gene therapy is primary treatment strategy; clinically identical to IIIA/IIIB; severe behaviour + cognitive regression + sleep disorder; late-onset HGSNAT = attenuated with retinitis pigmentosa; gene therapy AAV clinical trials ongoing' },
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

export default function HereditaryMpsAtlasPage() {
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
            🧬 Hereditary MPS Atlas
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>
            8-Gene Reference · IDUA · IDS · SGSH · GALNS · ARSB · GUSB · NAGLU · HGSNAT · 320 patients (8×40) · seeds 2630–2637
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
              <MetricCard label="Genes" value={overview.n_genes} sub="IDUA–HGSNAT" />
              <MetricCard label="Overall Coarse Facies" value={`${overview.aggregate_stats?.overall_coarse_facies_pct}%`} />
              <MetricCard label="Overall Skeletal Dysplasia" value={`${overview.aggregate_stats?.overall_skeletal_dysplasia_pct}%`} />
              <MetricCard label="Overall Odontoid Risk" value={`${overview.aggregate_stats?.overall_odontoid_risk_pct}%`} warn={true} />
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
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Coarse Facies%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Skeletal%</th>
                    <th style={{ textAlign: 'right', padding: '4px 8px' }}>Odontoid Risk%</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_summaries?.map((gs, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : card }}>
                      <td style={{ padding: '4px 8px' }}><GeneChip gene={gs.gene} /></td>
                      <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{gs.locus}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.n_patients}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.avg_onset_age}</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.coarse_facies_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right' }}>{gs.skeletal_dysplasia_pct}%</td>
                      <td style={{ padding: '4px 8px', textAlign: 'right', color: gs.odontoid_risk_pct > 0 ? '#ef4444' : 'inherit' }}>{gs.odontoid_risk_pct}%</td>
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
                    <span>CoarseFacies: {gb.coarse_facies_pct}%</span>
                    <span>Skeletal: {gb.skeletal_dysplasia_pct}%</span>
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

            {/* MPS Glossary */}
            <div style={{ fontWeight: 700, color: accent, fontSize: 15, marginBottom: 12 }}>MPS Glossary</div>
            {Object.entries(definitions.mps_glossary || {}).map(([term, def]) => (
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
