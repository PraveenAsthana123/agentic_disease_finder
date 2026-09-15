'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-progeroid-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'WRN':    '#4e342e',  // dark brown — Werner Syndrome adult-onset; bilateral cataracts; mesenchymal cancer; DM; VTM
  'BLM':    '#1a237e',  // deep navy — Bloom Syndrome; SCE 10x PATHOGNOMONIC; BLMAsh Ashkenazi; azoospermia
  'RECQL4': '#880e4f',  // deep magenta — RTS2/Rapadilino/Baller-Gerold; osteosarcoma 30-50%; poikiloderma infantile
  'ERCC6':  '#1b5e20',  // dark forest green — CS-B most severe; TC-NER; NO skin cancer; cachectic dwarfism
  'ERCC8':  '#004d40',  // dark teal — CS-A milder; same assay as CS-B; gene sequencing distinguishes
  'BANF1':  '#37474f',  // dark slate — NGPS ultra-rare; LMNA normal; clavicular resorption; Ala12Thr
  'TINF2':  '#4a148c',  // deep purple — DC2/Revesz; bilateral exudative retinopathy PATHOGNOMONIC; shortest telomeres; de novo
  'RTEL1':  '#b71c1c',  // deep red — HHS most severe DC; cerebellar hypoplasia + SCID + BMF; D-loop disassembly; IPF monoallelic
};

const GENE_INFO = {
  'WRN':    { full: 'WRN / Werner Syndrome Protein / 1432aa', locus: '8p12', size: '1432 aa / 162 kDa (RecQ helicase + 3\'→5\' exonuclease; T-loop unwinding; telomere maintenance)', inh: 'AR' },
  'BLM':    { full: 'BLM / Bloom Syndrome Helicase / 1417aa', locus: '15q26.1', size: '1417 aa / 159 kDa (RecQ helicase; BTR complex; dHJ dissolution; SCE suppressor; anti-crossover)', inh: 'AR' },
  'RECQL4': { full: 'RECQL4 / RecQ Helicase 4 / 1208aa', locus: '8q24.12', size: '1208 aa / 133 kDa (Sld2-homologous N-terminus initiates replication; 3\'→5\' helicase; mitochondrial function)', inh: 'AR' },
  'ERCC6':  { full: 'ERCC6 / CSB / Cockayne Syndrome B Protein / 1493aa', locus: '10q11.23', size: '1493 aa / 168 kDa (SWI2/SNF2 ATPase; TC-NER coupling factor; recruits XPA/TFIIH to stalled RNAPII)', inh: 'AR' },
  'ERCC8':  { full: 'ERCC8 / CSA / CRL4 WD40 Substrate Adaptor / 396aa', locus: '5q12.1', size: '396 aa / 44 kDa (WD40 DCAF receptor; CRL4-DDB1 E3 ligase; ubiquitylates CSB+RNAPII for TC-NER)', inh: 'AR' },
  'BANF1':  { full: 'BANF1 / BAF / Barrier-to-Autointegration Factor / 89aa', locus: '11q13.1', size: '89 aa / 10 kDa (homodimer; bridges LEM-domain proteins to chromatin; nuclear lamina reassembly)', inh: 'AR' },
  'TINF2':  { full: 'TINF2 / TIN2 / Shelterin Core / 354aa', locus: '14q12', size: '354 aa / 40 kDa (shelterin bridge: TRF1-TRF2 dsDNA ↔ TPP1-POT1 ssDNA; telomere length regulator)', inh: 'AD (de novo)' },
  'RTEL1':  { full: 'RTEL1 / Regulator of Telomere Elongation Helicase 1 / 1219aa', locus: '20q13.33', size: '1219 aa / 128 kDa (DEAH helicase; T-loop disassembly; G4 resolution; PCNA-PIP interaction)', inh: 'AR / AD' },
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

export default function HProgeroidAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState('WRN');
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
        setOverview(await ovRes.json());
        setBreakdown(await bkRes.json());
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

  if (loading) return <div style={{ background: bg, minHeight: '100vh', color: '#cfd8dc', padding: 32 }}>Loading Hereditary Progeroid Atlas…</div>;
  if (error) return <div style={{ background: bg, minHeight: '100vh', color: '#ef5350', padding: 32 }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = overview.genes || [];

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#cfd8dc', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #1a0a2e 0%, #0d2137 50%, #0d3320 100%)', padding: '28px 32px 20px', borderBottom: `1px solid ${border}` }}>
        <div style={{ fontSize: 11, color: '#78909c', marginBottom: 6 }}>🧬 HEREDITARY DISEASE ATLAS / PROGEROID SYNDROMES / DNA STABILITY</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#fff' }}>Hereditary Progeroid &amp; Premature-Aging Atlas</h1>
        <div style={{ color: '#90a4ae', fontSize: 13, marginTop: 6 }}>
          Complete 8-Gene Segmental Progeroid Syndrome Reference — WRN · BLM · RECQL4 · ERCC6 · ERCC8 · BANF1 · TINF2 · RTEL1
        </div>
        <div style={{ color: '#78909c', fontSize: 11, marginTop: 4 }}>
          Werner / Bloom / RTS2-Rapadilino / CS-B / CS-A / NGPS / DC2-Revesz / HHS · 320 Patients (8×40) · Seeds 2758–2765
        </div>
        {/* Gene chips */}
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
            background: 'none', border: 'none', color: tab === t ? '#80cbc4' : '#78909c',
            borderBottom: tab === t ? '2px solid #80cbc4' : '2px solid transparent',
            padding: '10px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* ── OVERVIEW ── */}
        {tab === 'Overview' && (
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
              <StatBadge label="Genes" value={overview.n_genes} color="#80cbc4" />
              <StatBadge label="Patients" value={overview.n_patients} color="#a5d6a7" />
              <StatBadge label="Seeds" value={overview.seeds} color="#ce93d8" />
              <StatBadge label="Avg Cancer Risk %" value={`${overview.avg_cancer_risk_pct}%`} color="#ef9a9a" />
              <StatBadge label="Male" value={overview.sex_distribution?.M} color="#90caf9" />
              <StatBadge label="Female" value={overview.sex_distribution?.F} color="#f48fb1" />
            </div>

            {/* Pathway groups */}
            <Section title="Pathway Groups" color="#80cbc4">
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 12 }}>
                {Object.entries(overview.pathway_groups || {}).map(([group, geneList]) => (
                  <div key={group} style={{ background: card, border: `1px solid ${border}`, borderRadius: 10, padding: 14 }}>
                    <div style={{ color: '#80cbc4', fontWeight: 700, fontSize: 12, marginBottom: 8 }}>{group}</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {geneList.map(g => (
                        <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', borderRadius: 6, padding: '3px 10px', fontSize: 12, fontWeight: 700 }}>{g}</span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            {/* Key distinctions */}
            <Section title="Key Clinical Distinctions" color="#ef9a9a">
              <div style={{ background: card, border: `1px solid ${border}`, borderRadius: 10, padding: 14 }}>
                {(overview.key_clinical_distinctions || []).map((d, i) => (
                  <div key={i} style={{ borderLeft: `3px solid ${GENE_COLORS[d.split(':')[0].trim()] || '#37474f'}`, paddingLeft: 10, marginBottom: 8, fontSize: 12, lineHeight: 1.6, color: '#cfd8dc' }}>
                    {d}
                  </div>
                ))}
              </div>
            </Section>

            {/* Contraindications */}
            <Section title="Absolutely Contraindicated / Critical Drug Warnings" color="#ff8a65">
              <div style={{ background: '#1a0a0a', border: '1px solid #b71c1c', borderRadius: 10, padding: 14 }}>
                {(overview.absolutely_contraindicated || []).map((w, i) => (
                  <div key={i} style={{ color: '#ef9a9a', fontSize: 12, marginBottom: 6, paddingLeft: 8, borderLeft: '3px solid #b71c1c' }}>⚠️ {w}</div>
                ))}
              </div>
            </Section>

            {/* Gene summary table */}
            <Section title="Gene Summary Table" color="#a5d6a7">
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#1e2a31' }}>
                      {['Gene', 'Locus', 'Protein', 'N', 'Inheritance', 'Onset', 'Avg Cancer %'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#90a4ae', borderBottom: `1px solid ${border}`, whiteSpace: 'nowrap' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(overview.gene_summaries || []).map(g => (
                      <tr key={g.gene} onClick={() => { setActiveGene(g.gene); setTab('Clinical Atlas'); }}
                        style={{ cursor: 'pointer', borderBottom: `1px solid ${border}`, background: activeGene === g.gene ? '#1e2a31' : 'transparent' }}>
                        <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#fff' }}>{g.gene}</td>
                        <td style={{ padding: '7px 10px', color: '#90a4ae' }}>{g.locus}</td>
                        <td style={{ padding: '7px 10px', color: '#cfd8dc', maxWidth: 200, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{g.protein_size}</td>
                        <td style={{ padding: '7px 10px', color: '#a5d6a7' }}>{g.n_patients}</td>
                        <td style={{ padding: '7px 10px', color: '#ce93d8' }}>{g.inheritance}</td>
                        <td style={{ padding: '7px 10px', color: '#80cbc4' }}>{g.onset}</td>
                        <td style={{ padding: '7px 10px', color: '#ef9a9a', fontWeight: 700 }}>{g.avg_cancer_risk_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          </div>
        )}

        {/* ── GENE TABLE ── */}
        {tab === 'Gene Table' && (
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
                    <Section title="Disease Category" color={GENE_COLORS[activeGene]}>
                      <ClinicalText text={breakdown[activeGene].disease_category} />
                    </Section>
                    <Section title="Molecular Pathway" color="#90a4ae">
                      <ClinicalText text={breakdown[activeGene].disease_pathway} />
                    </Section>
                    <Section title="Clinical Variables" color="#a5d6a7">
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {Object.entries(breakdown[activeGene].clinical_variables || {}).map(([k, [lo, hi, desc]]) => (
                          <div key={k} style={{ background: '#1e2a31', borderRadius: 8, padding: '8px 12px', minWidth: 140 }}>
                            <div style={{ color: '#90a4ae', fontSize: 10 }}>{k}</div>
                            <div style={{ color: '#fff', fontWeight: 700, fontSize: 16 }}>{lo}–{hi}</div>
                            <div style={{ color: '#78909c', fontSize: 10, marginTop: 2 }}>{desc}</div>
                          </div>
                        ))}
                      </div>
                    </Section>
                    <Section title="Summary Stats (40 patients)" color="#ce93d8">
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                        {Object.entries(breakdown[activeGene].summary_stats || {}).map(([k, v]) => (
                          <StatBadge key={k} label={k.replace(/_/g, ' ')} value={v} color={GENE_COLORS[activeGene]} />
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
                <Section title="Pathognomonic / Diagnostic Approach" color={GENE_COLORS[activeGene]}>
                  <ClinicalText text={breakdown[activeGene].pathognomonic_notes} />
                </Section>
                {/* Patient cohort sample */}
                <Section title={`Patient Cohort — ${activeGene} (40 patients)`} color="#a5d6a7">
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                      <thead>
                        <tr style={{ background: '#1e2a31' }}>
                          {['ID', 'Sex', 'Age at Onset (y)', 'Onset Type', 'Inheritance', 'Primary Measure', 'Value', 'Cancer Risk %', 'Treatment'].map(h => (
                            <th key={h} style={{ padding: '6px 8px', textAlign: 'left', color: '#90a4ae', borderBottom: `1px solid ${border}`, whiteSpace: 'nowrap' }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(breakdown[activeGene].patients || []).slice(0, 20).map(p => (
                          <tr key={p.id} style={{ borderBottom: `1px solid ${border}` }}>
                            <td style={{ padding: '5px 8px', color: GENE_COLORS[activeGene], fontWeight: 700 }}>{p.id}</td>
                            <td style={{ padding: '5px 8px', color: p.sex === 'M' ? '#90caf9' : '#f48fb1' }}>{p.sex}</td>
                            <td style={{ padding: '5px 8px', color: '#cfd8dc' }}>{p.age_at_presentation_y}</td>
                            <td style={{ padding: '5px 8px', color: '#80cbc4', fontSize: 10 }}>{p.onset_label}</td>
                            <td style={{ padding: '5px 8px', color: '#ce93d8' }}>{p.inheritance}</td>
                            <td style={{ padding: '5px 8px', color: '#90a4ae', fontSize: 10 }}>{p.primary_measure_label}</td>
                            <td style={{ padding: '5px 8px', color: '#fff', fontWeight: 700 }}>{p.primary_measure_value}</td>
                            <td style={{ padding: '5px 8px', color: '#ef9a9a', fontWeight: 700 }}>{p.cancer_risk_pct}%</td>
                            <td style={{ padding: '5px 8px', color: '#78909c', fontSize: 10, maxWidth: 200 }}>{p.treatment}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {(breakdown[activeGene].patients || []).length > 20 && (
                      <div style={{ color: '#78909c', fontSize: 11, padding: 8 }}>Showing 20 of {breakdown[activeGene].patients.length} patients</div>
                    )}
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
              {definitions.n_entries} entries — progeroid syndromes, RecQ helicases, Cockayne syndrome, nuclear lamina, DC spectrum, shelterin
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: 12 }}>
              {Object.entries(definitions.definitions || {}).map(([term, def]) => {
                const gColor = GENE_COLORS[term] || '#37474f';
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
