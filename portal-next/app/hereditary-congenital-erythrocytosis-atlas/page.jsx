'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-congenital-erythrocytosis-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'EPOR':  '#b71c1c',  // deep red      — Primary, EPO suppressed, GOF receptor
  'VHL':   '#1565c0',  // deep blue     — Chuvash/CE2, HIF axis, thrombosis
  'EGLN1': '#2e7d32',  // dark green    — PHD2/CE3, HIF pathway, paraganglioma
  'EPAS1': '#6a1b9a',  // deep purple   — HIF-2α/CE4, belzutifan target
  'HBB':   '#e65100',  // deep orange   — High-affinity Hb variants, left-shifted ODC
  'HBA1':  '#00695c',  // dark teal     — High-affinity alpha chain variants
  'BPGM':  '#c62828',  // crimson       — 2,3-BPG deficiency, extreme left-shift
  'EPO':   '#4527a0',  // deep indigo   — Rare EPO GOF, markedly elevated EPO
};

const GENE_INFO = {
  'EPOR':  { full: 'EPOR / Erythropoietin Receptor / 508aa', locus: '19p13.2', size: '508 aa / 56 kDa (C-terminal truncation removes SHP-1/SOCS3 negative-regulatory docking → constitutive JAK2-STAT5 → autonomous erythropoiesis; EPO SUPPRESSED <5 IU/L — key DDx from all other hereditary erythrocytoses; CE1; AD)', inh: 'AD GOF' },
  'VHL':   { full: 'VHL / Von Hippel-Lindau / 213aa', locus: '3p25.3', size: '213 aa / 24 kDa (E3 ubiquitin ligase subunit targeting HIF1α/2α for proteasomal degradation; biallelic p.Arg200Trp = Chuvash/CE2 — erythrocytosis + elevated EPO WITHOUT tumour syndrome; heterozygous = VHL syndrome: RCC + haemangioblastoma + phaeochromocytoma; belzutifan FDA 2021)', inh: 'AR (Chuvash) / AD (VHL)' },
  'EGLN1': { full: 'EGLN1 / PHD2 / 426aa', locus: '1q42.2', size: '426 aa / 46 kDa (Prolyl hydroxylase domain 2; main HIF-α prolyl hydroxylase; LOF → HIF1α/2α not hydroxylated → VHL-targeting fails → HIF stable → EPO elevated; CE3; paraganglioma in 5-10% — annual plasma metanephrines mandatory; AD)', inh: 'AD LOF' },
  'EPAS1': { full: 'EPAS1 / HIF-2α / 870aa', locus: '2p21', size: '870 aa / 97 kDa (Hypoxia-Inducible Factor 2α; GOF → resistance to PHD2 hydroxylation/VHL degradation → HIF-2α constitutively active → elevated EPO, VEGF, other HIF targets; CE4; polycythaemia-paraganglioma-PAH TRIAD; somatic mosaic variants; belzutifan direct HIF-2α inhibitor, FDA 2021)', inh: 'AD GOF' },
  'HBB':   { full: 'HBB / Haemoglobin Beta / 147aa', locus: '11p15.4', size: '147 aa / 16 kDa (β-chain of adult HbA; high-affinity variants — Hb Chesapeake α92Arg→Leu, Hb Hiroshima β146His→Asp, Hb Malmö, Hb Rainier; left-shifted ODC (low p50) → O₂ not released → tissue hypoxia → EPO elevated → compensatory erythrocytosis; CE6; treatment NOT needed unless Hct >0.56)', inh: 'AD' },
  'HBA1':  { full: 'HBA1 / Haemoglobin Alpha-1 / 142aa', locus: '16p13.3', size: '142 aa / 15 kDa (α1-chain of HbA; high-affinity alpha variants — Hb Suresnes, Hb Evanston, Hb Torino, Hb Creteil; less common than HBB high-affinity variants; 4-gene alpha deletion analysis needed for both HBA1+HBA2; p50 O₂ measurement confirms diagnosis)', inh: 'AD' },
  'BPGM':  { full: 'BPGM / Bisphosphoglycerate Mutase / 258aa', locus: '7q33', size: '258 aa / 29 kDa (2,3-BPG synthase/mutase; sole enzyme for 2,3-bisphosphoglycerate synthesis; BPGM deficiency → 2,3-BPG absent → Hb oxygen affinity extremely HIGH (most left-shifted ODC) → tissue hypoxia → EPO elevated → erythrocytosis; CE8; AR; p50 very low; same outcome as high-affinity Hb but enzymatic)', inh: 'AR' },
  'EPO':   { full: 'EPO / Erythropoietin / 193aa', locus: '7q22.3', size: '193 aa / 21 kDa (kidney/liver-secreted glycoprotein; very rare GOF variant → constitutively elevated EPO → secondary erythrocytosis; CE5; EPO markedly elevated; regulatory region variants — standard exon sequencing MISSES promoter/enhancer; distinguish from PV (JAK2 V617F somatic) and reactive erythrocytosis)', inh: 'AD GOF' },
};

function GeneChip({ gene, active, onClick }) {
  const col = GENE_COLORS[gene] || '#555';
  return (
    <span
      onClick={() => onClick && onClick(gene)}
      style={{
        background: col, color: '#fff', borderRadius: 4,
        padding: '3px 10px', fontSize: 12, fontWeight: 700,
        margin: '0 3px 4px 0', cursor: onClick ? 'pointer' : 'default',
        opacity: active === null || active === gene ? 1 : 0.45,
        border: active === gene ? '2px solid #fff' : '2px solid transparent',
        display: 'inline-block',
      }}
    >{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', border: `1px solid ${warn ? '#ef4444' : '#334155'}`, borderRadius: 8, padding: '12px 16px', minWidth: 130 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: warn ? '#ef4444' : '#38bdf8' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryCongenitalErythrocytosisAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [activeGene, setActiveGene] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [ovRes, bkRes, dfRes] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`),
          fetch(`${API}/api/${SLUG}/breakdown`),
          fetch(`${API}/api/${SLUG}/definitions`),
        ]);
        setOverview(await ovRes.json());
        setBreakdown(await bkRes.json());
        setDefs(await dfRes.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const geneList = Object.keys(GENE_COLORS);

  if (loading) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#94a3b8', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 18 }}>
      Loading Congenital Erythrocytosis Atlas…
    </div>
  );
  if (error) return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#ef4444', padding: 32, fontSize: 16 }}>
      Error: {error}
    </div>
  );

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #200010 0%, #0f172a 100%)', borderBottom: '1px solid #b71c1c', padding: '20px 28px 14px' }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#f87171', letterSpacing: 1 }}>
          🧬 Hereditary Congenital Erythrocytosis Atlas
        </div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 4 }}>
          Complete 8-Gene Reference — EPOR · VHL · EGLN1 · EPAS1 · HBB · HBA1 · BPGM · EPO
        </div>
        <div style={{ marginTop: 6, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <div style={{ padding: '5px 10px', background: '#7c1a1a', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#fca5a5', fontWeight: 700 }}>
            ⚡ EPOR (CE1): ONLY hereditary erythrocytosis with SUPPRESSED EPO — all others have elevated EPO
          </div>
          <div style={{ padding: '5px 10px', background: '#1e3a5f', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#7dd3fc', fontWeight: 700 }}>
            ⚠ VHL (Chuvash): biallelic p.Arg200Trp — THROMBOSIS leading cause of death (portal/Budd-Chiari)
          </div>
          <div style={{ padding: '5px 10px', background: '#1e1a3f', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#c4b5fd', fontWeight: 700 }}>
            ⚠ EPAS1 (CE4): polycythaemia-paraganglioma-PAH TRIAD — belzutifan direct HIF-2α target
          </div>
          <div style={{ padding: '5px 10px', background: '#1a2e1a', borderRadius: 4, display: 'inline-block', fontSize: 11, color: '#86efac', fontWeight: 700 }}>
            ⚠ HBB/HBA1/BPGM: HIGH-AFFINITY Hb — low p50; treatment NOT needed unless Hct &gt;0.56
          </div>
        </div>
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {geneList.map(g => (
            <GeneChip key={g} gene={g} active={activeGene} onClick={g => setActiveGene(activeGene === g ? null : g)} />
          ))}
        </div>
        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginTop: 14 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? '#f87171' : '#1e293b',
              color: tab === t ? '#0f172a' : '#94a3b8',
              border: 'none', borderRadius: 4, padding: '5px 14px', fontSize: 12,
              fontWeight: tab === t ? 700 : 400, cursor: 'pointer',
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '20px 28px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40, seeds 2886-2893" />
              <MetricCard label="Genes Covered" value={overview.total_genes} sub="Primary / HIF / High-Affinity Hb / 2,3-BPG / EPO-GOF" />
              <MetricCard label="Disorder Classes" value="4" sub="CE1 (primary) / CE2-4 (HIF) / CE6/8 (Hb/BPG) / CE5 (EPO)" />
              <MetricCard label="EPO Suppressed" value="EPOR only" sub="<5 IU/L — JAK2 negative; all others elevated" warn={false} />
              <MetricCard label="HIF-2α Inhibitor" value="Belzutifan" sub="FDA 2021 — VHL/EPAS1-GOF direct target" warn={false} />
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171', marginBottom: 10 }}>
              Gene Summary — Avg Onset, Hb (g/dL), EPO Pattern
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e293b' }}>
                    {['Gene','Locus','Inh','Pts','Avg Onset (yr)','Avg Hb (g/dL)','EPO Pattern'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#f87171', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {overview.gene_rows?.filter(g => !activeGene || g.gene === activeGene).map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1a2332', cursor: 'pointer' }}
                        onClick={() => setActiveGene(activeGene === g.gene ? null : g.gene)}>
                      <td style={{ padding: '5px 8px' }}>
                        <GeneChip gene={g.gene} active={activeGene} />
                      </td>
                      <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{g.locus}</td>
                      <td style={{ padding: '5px 8px', color: '#fbbf24', fontSize: 10 }}>{GENE_INFO[g.gene]?.inh}</td>
                      <td style={{ padding: '5px 8px' }}>{g.patients}</td>
                      <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{g.avg_onset}</td>
                      <td style={{ padding: '5px 8px', color: g.avg_hgb_g_dl > 19 ? '#ef4444' : '#fbbf24' }}>
                        {g.avg_hgb_g_dl} {g.avg_hgb_g_dl > 19 ? '↑↑' : '↑'}
                      </td>
                      <td style={{ padding: '5px 8px' }}>
                        <span style={{
                          background: g.epo_pattern === 'suppressed' ? '#7c1a1a' : '#14532d',
                          color: g.epo_pattern === 'suppressed' ? '#fca5a5' : '#86efac',
                          padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700
                        }}>
                          {g.epo_pattern === 'suppressed' ? '⬇ SUPPRESSED' : '⬆ ELEVATED'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171', marginTop: 24, marginBottom: 10 }}>
              Mechanistic Categories
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
              {overview.categories && Object.entries(overview.categories).map(([cat, genes], i) => (
                <div key={i} style={{ background: '#1e293b', border: `1px solid ${i === 0 ? '#7c1a1a' : i === 1 ? '#1e3a5f' : i === 2 ? '#14532d' : '#4c1d95'}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ fontWeight: 700, color: i === 0 ? '#fca5a5' : i === 1 ? '#7dd3fc' : i === 2 ? '#86efac' : '#c4b5fd', fontSize: 12, marginBottom: 6 }}>{cat}</div>
                  <div style={{ marginBottom: 4, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                    {genes?.map(g => <GeneChip key={g} gene={g} active={null} />)}
                  </div>
                </div>
              ))}
            </div>

            <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171', marginTop: 24, marginBottom: 10 }}>
              Key Facts &amp; Diagnostic Algorithm
            </div>
            {overview.key_facts?.map((d, i) => (
              <div key={i} style={{
                background: '#1e293b',
                border: `1px solid ${d.includes('SUPPRESSED') ? '#7c1a1a' : d.includes('THROMBOSIS') || d.includes('Budd') ? '#1e3a5f' : d.includes('belzutifan') || d.includes('TRIAD') ? '#4c1d95' : '#334155'}`,
                borderRadius: 6, padding: '8px 12px', marginBottom: 6,
                fontSize: 11, color: '#cbd5e1', lineHeight: 1.6
              }}>
                {d.includes('SUPPRESSED') ? '⚡ ' : d.includes('THROMBOSIS') ? '⚠ ' : d.includes('belzutifan') ? '💊 ' : '▶ '}{d}
              </div>
            ))}
            {overview.diagnostic_algorithm && (
              <div style={{ background: '#1e2a1e', border: '1px solid #14532d', borderRadius: 8, padding: '14px 16px', marginTop: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#86efac', marginBottom: 8 }}>DIAGNOSTIC ALGORITHM</div>
                <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{overview.diagnostic_algorithm}</div>
              </div>
            )}
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171', marginBottom: 14 }}>
              Gene Reference — Protein, Locus, Inheritance, Clinical Role
            </div>
            {geneList.filter(g => !activeGene || g === activeGene).map(g => (
              <div key={g} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g]}44`, borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
                  <GeneChip gene={g} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{GENE_INFO[g]?.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 12, fontWeight: 700 }}>{GENE_INFO[g]?.inh}</span>
                  {g === 'EPOR' && (
                    <span style={{ background: '#7c1a1a', color: '#fca5a5', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      EPO SUPPRESSED — ONLY CE with low EPO
                    </span>
                  )}
                  {g === 'VHL' && (
                    <span style={{ background: '#1e3a5f', color: '#7dd3fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      CHUVASH: THROMBOSIS — PORTAL/BUDD-CHIARI
                    </span>
                  )}
                  {g === 'EGLN1' && (
                    <span style={{ background: '#14532d', color: '#86efac', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      PHD2 — PARAGANGLIOMA 5-10%
                    </span>
                  )}
                  {g === 'EPAS1' && (
                    <span style={{ background: '#4c1d95', color: '#c4b5fd', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      HIF-2α — BELZUTIFAN DIRECT TARGET
                    </span>
                  )}
                  {(g === 'HBB' || g === 'HBA1' || g === 'BPGM') && (
                    <span style={{ background: '#451a03', color: '#fdba74', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      HIGH-AFFINITY Hb — LOW p50 — NO Rx unless Hct &gt;0.56
                    </span>
                  )}
                  {g === 'EPO' && (
                    <span style={{ background: '#1e1a3f', color: '#c4b5fd', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      RARE GOF — PROMOTER REGION MISSES EXON SEQ
                    </span>
                  )}
                </div>
                <div style={{ fontSize: 12, color: '#e2e8f0', marginBottom: 4 }}>{GENE_INFO[g]?.full}</div>
                <div style={{ fontSize: 11, color: '#64748b' }}>{GENE_INFO[g]?.size}</div>
              </div>
            ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.genes?.filter(g => !activeGene || g.gene === activeGene).map(g => (
              <div key={g.gene} style={{ background: '#1e293b', border: `1px solid ${GENE_COLORS[g.gene]}55`, borderRadius: 10, padding: 18, marginBottom: 18 }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
                  <GeneChip gene={g.gene} active={null} />
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus}</span>
                  <span style={{ color: '#fbbf24', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh}</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>{g.patient_count} patients</span>
                  {g.gene === 'EPOR' && (
                    <span style={{ background: '#7c1a1a', color: '#fca5a5', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚡ EPO SUPPRESSED — JAK2 NEGATIVE — PRIMARY
                    </span>
                  )}
                  {g.gene === 'VHL' && (
                    <span style={{ background: '#1e3a5f', color: '#7dd3fc', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ CHUVASH: THROMBOSIS RISK — ANTICOAGULATION
                    </span>
                  )}
                  {g.gene === 'EPAS1' && (
                    <span style={{ background: '#4c1d95', color: '#c4b5fd', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ PARAGANGLIOMA-PAH TRIAD — BELZUTIFAN
                    </span>
                  )}
                  {g.gene === 'EGLN1' && (
                    <span style={{ background: '#14532d', color: '#86efac', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ PARAGANGLIOMA — ANNUAL METANEPHRINES
                    </span>
                  )}
                  {(g.gene === 'HBB' || g.gene === 'HBA1' || g.gene === 'BPGM') && (
                    <span style={{ background: '#451a03', color: '#fdba74', fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4 }}>
                      ⚠ HIGH-AFFINITY Hb — p50 LOW — AVOID over-phlebotomy
                    </span>
                  )}
                </div>
                {[['Protein', g.protein], ['Inheritance', g.inheritance], ['Disease', g.disease_category],
                  ['Pathway', g.disease_pathway], ['Pathognomonic', g.pathognomonic], ['Treatment', g.treatment]].map(([label, val]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, color: '#f87171', fontWeight: 700, marginBottom: 2 }}>{label.toUpperCase()}</div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{val}</div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && defs && (
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171', marginBottom: 14 }}>
              Glossary — Erythrocytosis / EPO-Suppressed-vs-Elevated / p50-O₂ / Chuvash / PV-vs-Hereditary / 2,3-BPG / Belzutifan / EGLN1-Paraganglioma / High-Affinity-Hb
            </div>
            {defs.definitions?.map((d, i) => (
              <div key={i} style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: '#fbbf24', fontSize: 13, marginBottom: 6 }}>{d.term}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{d.definition}</div>
              </div>
            ))}
            {defs.standards && defs.standards.length > 0 && (
              <>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#f87171', marginTop: 20, marginBottom: 10 }}>
                  Standards &amp; References
                </div>
                {defs.standards.map((s, i) => (
                  <div key={i} style={{ background: '#1e293b', border: '1px solid #7c1a1a', borderRadius: 6, padding: '6px 12px', marginBottom: 6, fontSize: 11, color: '#94a3b8' }}>
                    📋 {s}
                  </div>
                ))}
              </>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
