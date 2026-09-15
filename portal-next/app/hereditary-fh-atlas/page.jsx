'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-fh-atlas';
const TABS = ['Overview', 'Gene Breakdown', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'LDLR':    '#b71c1c',  // deep red — most common monogenic FH; 1:200-500 HeFH; tendon xanthomas PATHOGNOMONIC
  'APOB':    '#e65100',  // deep orange — FDB Arg3500Gln; 1:700 European; milder LDL; PCSK9i most effective
  'PCSK9':   '#f57f17',  // amber — GOF D374Y; Norwegian founder; inclisiran/evolocumab mechanism-specific
  'LDLRAP1': '#4e342e',  // dark brown — ARH; fibroblast LDLR NORMAL; apheresis + evinacumab
  'ABCG5':   '#1b5e20',  // dark forest green — sitosterolemia type A; childhood xanthomas + NORMAL LDL-C
  'ABCG8':   '#004d40',  // dark teal — sitosterolemia type B; D19H Asian founder; stomatocytes
  'LIPA':    '#4a148c',  // deep purple — LAL-D; Wolman bilateral adrenal calcification; CESD; sebelipase alfa
  'LPA':     '#37474f',  // dark slate — Lp(a) dose-effect; statins may RAISE; pelacarsen/olpasiran Phase 3
};

const GENE_INFO = {
  'LDLR':    { full: 'LDLR / Low-Density Lipoprotein Receptor / 860aa', locus: '19p13.2', size: '860 aa / 95 kDa (type-I TM glycoprotein; clathrin-coated pit endocytosis; EGF-precursor homology domain)', inh: 'AD (AR for HoFH)' },
  'APOB':    { full: 'APOB / Apolipoprotein B-100 / 4563aa', locus: '2p24.1', size: '4563 aa / 512 kDa (LDLR-binding domain Arg3500Gln; FDB if receptor-binding site mutated)', inh: 'AD' },
  'PCSK9':   { full: 'PCSK9 / Proprotein Convertase Subtilisin/Kexin Type 9 / 692aa', locus: '1p32.3', size: '692 aa / 73 kDa (serine protease; targets LDLR for lysosomal degradation; GOF → FH3)', inh: 'AD (GOF) / Protective (LOF)' },
  'LDLRAP1': { full: 'LDLRAP1 / LDL Receptor Adaptor Protein 1 / 308aa', locus: '1p36.11', size: '308 aa / 35 kDa (ARH adaptor protein; bridges LDLR NPxY motif to clathrin; liver-specific critical role)', inh: 'AR' },
  'ABCG5':   { full: 'ABCG5 / Sterolin-1 / 651aa', locus: '2p21', size: '651 aa / 75 kDa (half-transporter; ABCG5:ABCG8 heterodimer; intestinal+biliary plant sterol efflux)', inh: 'AR' },
  'ABCG8':   { full: 'ABCG8 / Sterolin-2 / 673aa', locus: '2p21', size: '673 aa / 77 kDa (half-transporter; ABCG5:ABCG8 obligate heterodimer; D19H Asian founder 1:1000)', inh: 'AR' },
  'LIPA':    { full: 'LIPA / Lysosomal Acid Lipase / 399aa', locus: '10q23.31', size: '399 aa / 45 kDa (lysosomal serine esterase; hydrolyses CE and TG from endocytosed LDL)', inh: 'AR' },
  'LPA':     { full: 'LPA / Apolipoprotein(a) / ~5927aa (KIV-2 variable)', locus: '6q27', size: '~5927 aa / 500–700 kDa (KIV-2 repeats determine Lp(a) mass; disulfide bond to ApoB-100)', inh: 'AD dose-effect' },
};

function GeneChip({ gene, active, onClick }) {
  return (
    <button
      onClick={() => onClick(gene)}
      style={{
        background: active ? GENE_COLORS[gene] : '#263238',
        color: '#fff',
        border: `2px solid ${GENE_COLORS[gene]}`,
        borderRadius: 8,
        padding: '6px 14px',
        margin: 4,
        cursor: 'pointer',
        fontWeight: active ? 700 : 400,
        fontSize: 13,
        transition: 'all 0.15s',
      }}
    >
      {gene}
    </button>
  );
}

function StatBadge({ label, value, color }) {
  return (
    <div style={{ background: '#1e2a31', border: `1px solid ${color || '#37474f'}`, borderRadius: 8, padding: '10px 14px', minWidth: 100, textAlign: 'center', margin: 4 }}>
      <div style={{ color: color || '#90a4ae', fontSize: 10, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#fff', fontSize: 20, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function Section({ title, children, color }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ color: color || '#90a4ae', fontWeight: 700, fontSize: 13, marginBottom: 6, textTransform: 'uppercase', letterSpacing: 1 }}>{title}</div>
      {children}
    </div>
  );
}

function ClinicalText({ text }) {
  if (!text) return null;
  return (
    <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12, fontSize: 12, color: '#cfd8dc', lineHeight: 1.7, whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
      {text}
    </div>
  );
}

export default function HFHAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState('LDLR');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      setError(null);
      try {
        const [ovRes, bkRes, defRes] = await Promise.all([
          fetch(`${API}/api/${SLUG}/overview`),
          fetch(`${API}/api/${SLUG}/breakdown`),
          fetch(`${API}/api/${SLUG}/definitions`),
        ]);
        const ovData = await ovRes.json();
        const bkData = await bkRes.json();
        setOverview(ovData);
        // Index breakdown by gene
        const bkMap = {};
        (bkData.breakdown || []).forEach(g => { bkMap[g.gene] = g; });
        setBreakdown(bkMap);
        setDefinitions(await defRes.json());
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  const bg = '#0d1921';
  const card = '#152232';
  const border = '#263238';

  if (loading) return <div style={{ background: bg, minHeight: '100vh', color: '#cfd8dc', padding: 32 }}>Loading Hereditary FH Atlas…</div>;
  if (error) return <div style={{ background: bg, minHeight: '100vh', color: '#ef5350', padding: 32 }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = overview.genes_covered || [];

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#cfd8dc', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1a0a0a 0%, #1a0f00 50%, #0d2137 100%)', padding: '28px 32px 20px', borderBottom: `1px solid ${border}` }}>
        <div style={{ fontSize: 11, color: '#78909c', marginBottom: 6 }}>🧬 HEREDITARY DISEASE ATLAS / FAMILIAL HYPERCHOLESTEROLAEMIA / LIPID METABOLISM</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#fff' }}>Hereditary FH &amp; Hyperlipidaemia Atlas</h1>
        <div style={{ color: '#90a4ae', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Reference — LDLR · APOB · PCSK9 · LDLRAP1 · ABCG5 · ABCG8 · LIPA · LPA
        </div>
        <div style={{ color: '#78909c', fontSize: 11, marginTop: 4 }}>
          FH1 / FDB / FH3 / ARH / Sitosterolemia / LAL-D / Lp(a) Spectrum · 320 Patients (8×40) · Seeds 2766–2773
        </div>
        <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap' }}>
          {genes.map(g => (
            <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: card, borderBottom: `1px solid ${border}`, padding: '0 32px', display: 'flex', gap: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', color: tab === t ? '#ef9a9a' : '#78909c',
            borderBottom: tab === t ? '2px solid #ef9a9a' : '2px solid transparent',
            padding: '10px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
              <StatBadge label="Genes" value={overview.gene_count} color="#ef9a9a" />
              <StatBadge label="Patients" value={overview.total_patients} color="#a5d6a7" />
              <StatBadge label="Seeds" value={overview.seed_range} color="#ce93d8" />
            </div>

            {/* FH Classification */}
            <Section title="FH Spectrum Classification" color="#ef9a9a">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 10 }}>
                {(overview.fh_classification || []).map((cls, i) => {
                  const gene = ['LDLR','APOB','PCSK9','LDLRAP1','ABCG5','ABCG8','LIPA','LPA'][i];
                  return (
                    <div key={i} style={{ background: card, border: `1px solid ${GENE_COLORS[gene] || border}`, borderRadius: 10, padding: 14 }}>
                      <div style={{ color: GENE_COLORS[gene] || '#90a4ae', fontWeight: 700, fontSize: 12, marginBottom: 6 }}>{gene}</div>
                      <div style={{ color: '#cfd8dc', fontSize: 11, lineHeight: 1.6 }}>{cls}</div>
                    </div>
                  );
                })}
              </div>
            </Section>

            {/* Critical distinctions */}
            <Section title="Critical Clinical Distinctions" color="#ff8a65">
              <div style={{ background: card, border: `1px solid ${border}`, borderRadius: 10, padding: 14 }}>
                {(overview.critical_distinctions || []).map((d, i) => (
                  <div key={i} style={{ borderLeft: '3px solid #ef9a9a', paddingLeft: 10, marginBottom: 8, fontSize: 12, lineHeight: 1.6, color: '#cfd8dc' }}>
                    {d}
                  </div>
                ))}
              </div>
            </Section>

            {/* Key drug classes */}
            <Section title="Key Drug Classes" color="#a5d6a7">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 10 }}>
                {(overview.key_drug_classes || []).map((drug, i) => (
                  <div key={i} style={{ background: card, border: `1px solid ${border}`, borderRadius: 8, padding: 12, fontSize: 11, color: '#cfd8dc', lineHeight: 1.6 }}>
                    <span style={{ color: '#a5d6a7', fontWeight: 700 }}>{drug.split(':')[0]}</span>
                    {drug.includes(':') ? ': ' + drug.split(':').slice(1).join(':') : ''}
                  </div>
                ))}
              </div>
            </Section>

            {/* Gene summary table */}
            <Section title="Gene Summary Table" color="#90caf9">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#1e2a31' }}>
                      {['Gene', 'N', 'Mean LDL (mg/dL)', 'Tendon Xanth %', 'CVD Event %', 'Dutch ≥8 %', 'PCSK9i %', 'Apheresis %'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#90a4ae', borderBottom: `1px solid ${border}`, whiteSpace: 'nowrap' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(overview.gene_summaries || []).map(g => (
                      <tr key={g.gene} onClick={() => { setActiveGene(g.gene); setTab('Clinical Atlas'); }}
                        style={{ cursor: 'pointer', borderBottom: `1px solid ${border}`, background: activeGene === g.gene ? '#1e2a31' : 'transparent' }}>
                        <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#fff' }}>{g.gene}</td>
                        <td style={{ padding: '7px 10px', color: '#a5d6a7' }}>{g.n}</td>
                        <td style={{ padding: '7px 10px', color: '#ef9a9a', fontWeight: 700 }}>{g.mean_ldl_c_mg_dL}</td>
                        <td style={{ padding: '7px 10px', color: '#80cbc4' }}>{g.tendon_xanthoma_pct}%</td>
                        <td style={{ padding: '7px 10px', color: '#ff8a65' }}>{g.cvd_event_pct}%</td>
                        <td style={{ padding: '7px 10px', color: '#ce93d8' }}>{g.dutch_score_ge8_pct}%</td>
                        <td style={{ padding: '7px 10px', color: '#90caf9' }}>{g.pcsk9i_candidate_pct}%</td>
                        <td style={{ padding: '7px 10px', color: '#f48fb1' }}>{g.apheresis_needed_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        )}

        {/* ── GENE BREAKDOWN ── */}
        {tab === 'Gene Breakdown' && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
              {genes.map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
            </div>
            {activeGene && GENE_INFO[activeGene] && (
              <div style={{ background: card, border: `2px solid ${GENE_COLORS[activeGene]}`, borderRadius: 12, padding: 20 }}>
                <div style={{ color: GENE_COLORS[activeGene], fontSize: 20, fontWeight: 800, marginBottom: 4 }}>{activeGene}</div>
                <div style={{ color: '#fff', fontSize: 14, marginBottom: 8 }}>{GENE_INFO[activeGene].full}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
                  <span style={{ background: '#1e2a31', borderRadius: 6, padding: '4px 10px', fontSize: 12, color: '#90a4ae' }}>Locus: {GENE_INFO[activeGene].locus}</span>
                  <span style={{ background: '#1e2a31', borderRadius: 6, padding: '4px 10px', fontSize: 12, color: '#ce93d8' }}>Inheritance: {GENE_INFO[activeGene].inh}</span>
                </div>
                <div style={{ color: '#90a4ae', fontSize: 12, marginBottom: 12 }}>{GENE_INFO[activeGene].size}</div>
                {breakdown && breakdown[activeGene] && (
                  <div>
                    <Section title="Stats (40 patients)" color={GENE_COLORS[activeGene]}>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        <StatBadge label="Mean LDL-C" value={`${breakdown[activeGene].mean_ldl_c_mg_dL} mg/dL`} color="#ef9a9a" />
                        <StatBadge label="Tendon Xanth %" value={`${breakdown[activeGene].tendon_xanthoma_pct}%`} color="#80cbc4" />
                        <StatBadge label="CVD Event %" value={`${breakdown[activeGene].cvd_event_pct}%`} color="#ff8a65" />
                        <StatBadge label="Dutch ≥8 %" value={`${breakdown[activeGene].dutch_score_ge8_pct}%`} color="#ce93d8" />
                        <StatBadge label="PCSK9i %" value={`${breakdown[activeGene].pcsk9i_candidate_pct}%`} color="#90caf9" />
                        <StatBadge label="Apheresis %" value={`${breakdown[activeGene].apheresis_needed_pct}%`} color="#f48fb1" />
                      </div>
                    </Section>
                    <Section title="Disease Category" color={GENE_COLORS[activeGene]}>
                      <ClinicalText text={breakdown[activeGene].disease_category} />
                    </Section>
                    <Section title="Pathognomonic Features" color="#80cbc4">
                      <ClinicalText text={breakdown[activeGene].pathognomonic} />
                    </Section>
                    <Section title="Treatment Summary" color="#a5d6a7">
                      <ClinicalText text={breakdown[activeGene].treatment_summary} />
                    </Section>
                    <Section title="Key Facts" color="#ce93d8">
                      <div style={{ background: card, border: `1px solid ${border}`, borderRadius: 8, padding: 12 }}>
                        {(breakdown[activeGene].key_facts || []).map((f, i) => (
                          <div key={i} style={{ borderLeft: `3px solid ${GENE_COLORS[activeGene]}`, paddingLeft: 10, marginBottom: 6, fontSize: 12, color: '#cfd8dc', lineHeight: 1.6 }}>{f}</div>
                        ))}
                      </div>
                    </Section>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* ── CLINICAL ATLAS ── */}
        {tab === 'Clinical Atlas' && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
              {genes.map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
            </div>
            {breakdown && breakdown[activeGene] && (
              <div>
                <div style={{ background: card, border: `2px solid ${GENE_COLORS[activeGene]}`, borderRadius: 12, padding: 20, marginBottom: 16 }}>
                  <div style={{ color: GENE_COLORS[activeGene], fontSize: 18, fontWeight: 800 }}>{activeGene} — Inheritance &amp; Disease Overview</div>
                  <div style={{ marginTop: 12 }}>
                    <ClinicalText text={breakdown[activeGene].inheritance} />
                  </div>
                </div>
                <Section title="Sample Patient Cohort — first 5 patients" color="#a5d6a7">
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                      <thead>
                        <tr style={{ background: '#1e2a31' }}>
                          {['ID', 'Sex', 'Onset (y)', 'LDL-C mg/dL', 'HDL-C', 'TG', 'Lp(a)', 'Tendon Xanth', 'CVD', 'Dutch ≥8', 'PCSK9i', 'Apheresis'].map(h => (
                            <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#90a4ae', borderBottom: `1px solid ${border}`, whiteSpace: 'nowrap' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(breakdown[activeGene].sample_patients || []).map(p => (
                          <tr key={p.patient_id} style={{ borderBottom: `1px solid ${border}` }}>
                            <td style={{ padding: '5px 8px', color: GENE_COLORS[activeGene], fontWeight: 700 }}>{p.patient_id}</td>
                            <td style={{ padding: '5px 8px', color: p.sex === 'M' ? '#90caf9' : '#f48fb1' }}>{p.sex}</td>
                            <td style={{ padding: '5px 8px', color: '#cfd8dc' }}>{p.onset_years}</td>
                            <td style={{ padding: '5px 8px', color: '#ef9a9a', fontWeight: 700 }}>{p.ldl_c_mg_dL}</td>
                            <td style={{ padding: '5px 8px', color: '#a5d6a7' }}>{p.hdl_c_mg_dL}</td>
                            <td style={{ padding: '5px 8px', color: '#80cbc4' }}>{p.tg_mg_dL}</td>
                            <td style={{ padding: '5px 8px', color: '#ce93d8' }}>{p.lpa_mg_dL}</td>
                            <td style={{ padding: '5px 8px', color: p.tendon_xanthoma ? '#ff8a65' : '#546e7a' }}>{p.tendon_xanthoma ? '✓' : '—'}</td>
                            <td style={{ padding: '5px 8px', color: p.cvd_event ? '#ef5350' : '#546e7a' }}>{p.cvd_event ? '✓' : '—'}</td>
                            <td style={{ padding: '5px 8px', color: p.dutch_score_ge8 ? '#ce93d8' : '#546e7a' }}>{p.dutch_score_ge8 ? '✓' : '—'}</td>
                            <td style={{ padding: '5px 8px', color: p.pcsk9i_candidate ? '#90caf9' : '#546e7a' }}>{p.pcsk9i_candidate ? '✓' : '—'}</td>
                            <td style={{ padding: '5px 8px', color: p.apheresis_needed ? '#f48fb1' : '#546e7a' }}>{p.apheresis_needed ? '✓' : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Section>
              </div>
            )}
          </div>
        )}

        {/* ── DEFINITIONS ── */}
        {tab === 'Definitions' && definitions && (
          <div>
            <div style={{ color: '#90a4ae', fontSize: 12, marginBottom: 12 }}>
              {definitions.n_entries} entries — FH spectrum, Dutch score, tendon xanthomas, PCSK9i, evinacumab, sitosterolemia, LAL-D, Lp(a)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
              {Object.entries(definitions.definitions || {}).map(([term, def]) => {
                const gColor = Object.entries(GENE_COLORS).find(([g]) => term.includes(g))?.[1] || '#37474f';
                return (
                  <div key={term} style={{ background: card, border: `1px solid ${gColor}`, borderRadius: 10, padding: 14 }}>
                    <div style={{ color: gColor, fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{term}</div>
                    <div style={{ color: '#cfd8dc', fontSize: 11, lineHeight: 1.6 }}>{def}</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
