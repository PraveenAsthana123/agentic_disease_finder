'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-sterol-biosynthesis-atlas';
const TABS = ['Overview', 'Gene Breakdown', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'DHCR7': '#b71c1c',  // deep red — Smith-Lemli-Opitz; 2,3-toe syndactyly PATHOGNOMONIC; 7-DHC elevated; most common
  'EBP':   '#4a148c',  // deep purple — CDPX2; stippled epiphyses PATHOGNOMONIC; Blaschko-line females; 8-DHC elevated
  'NSDHL': '#006064',  // dark cyan — CHILD syndrome; unilateral midline PATHOGNOMONIC; CK syndrome in males
  'SC5D':  '#1b5e20',  // dark forest green — Lathosterolosis; lathosterol elevated; liver disease; penultimate step
  'DHCR24':'#e65100',  // deep orange — Desmosterolosis; desmosterol elevated; statins ABSOLUTELY CONTRAINDICATED
  'LSS':   '#37474f',  // dark slate — Lanosterol synthase; cataracts + alopecia; ring-cyclisation first step
  'MVK':   '#f57f17',  // amber — HIDS/mevalonic aciduria; periodic fever; IL-1β; canakinumab FDA-approved
  'SQLE':  '#880e4f',  // deep pink — Squalene epoxidase; congenital alopecia; terbinafine interaction; cancer GOF
};

const GENE_INFO = {
  'DHCR7': { full: 'DHCR7 / 7-Dehydrocholesterol Reductase / 475aa', locus: '11q13.4', size: '475 aa / 54 kDa (8-TM ER sterol reductase; final step Kandutsch-Russell: 7-DHC → cholesterol)', inh: 'AR' },
  'EBP':   { full: 'EBP / Emopamil-Binding Protein / 230aa',         locus: 'Xp11.23',  size: '230 aa / 25 kDa (5-TM ER Δ8→Δ7 sterol isomerase; 8-DHC → 7-DHC direction)',                   inh: 'XLD (females mosaic; males lethal)' },
  'NSDHL': { full: 'NSDHL / NAD(P)H Steroid Dehydrogenase-Like / 374aa', locus: 'Xq28',  size: '374 aa / 41 kDa (ER C-4 decarboxylase; C-4 demethylation complex with SC4MOL)',                inh: 'XLD (CHILD females; CK males hypomorphic)' },
  'SC5D':  { full: 'SC5D / Sterol C5-Desaturase / 299aa',             locus: '11q23.3', size: '299 aa / 33 kDa (ER Δ5-desaturase; lathosterol → 7-DHC; penultimate K-R step)',                  inh: 'AR' },
  'DHCR24':{ full: 'DHCR24 / 24-Dehydrocholesterol Reductase / 516aa',locus: '1p32.3',  size: '516 aa / 60 kDa (ER FAD-dependent; Bloch final step: desmosterol → cholesterol; seladin-1)',     inh: 'AR' },
  'LSS':   { full: 'LSS / Lanosterol Synthase / 733aa',                locus: '21q22.3', size: '733 aa / 83 kDa (ER oxidosqualene cyclase; first ring-closure: 2,3-oxidosqualene → lanosterol)', inh: 'AR (hypomorphic viable)' },
  'MVK':   { full: 'MVK / Mevalonate Kinase / 396aa',                  locus: '12q24.11',size: '396 aa / 41 kDa (GHMP kinase; mevalonate → mevalonate-5-P; upstream of squalene/cholesterol)',   inh: 'AR' },
  'SQLE':  { full: 'SQLE / Squalene Epoxidase / 574aa',                locus: '8q24.13', size: '574 aa / 64 kDa (ER FAD-dependent; squalene → 2,3-oxidosqualene; terbinafine target)',           inh: 'AR (LOF→alopecia) / AD GOF (cancer amplification)' },
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

export default function HSBAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [activeGene, setActiveGene] = useState('DHCR7');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, []);

  const genes = Object.keys(GENE_COLORS);

  if (loading) return <div style={{ background: '#102027', minHeight: '100vh', color: '#90a4ae', padding: 40, fontFamily: 'monospace' }}>Loading Hereditary-Sterol-Biosynthesis-Atlas…</div>;
  if (error) return <div style={{ background: '#102027', minHeight: '100vh', color: '#ef5350', padding: 40, fontFamily: 'monospace' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#102027', minHeight: '100vh', color: '#eceff1', fontFamily: 'monospace', padding: '24px 32px' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ fontSize: 22, fontWeight: 700, color: '#b71c1c', marginBottom: 4 }}>
          🧬 Hereditary-Sterol-Biosynthesis-Atlas
        </div>
        <div style={{ fontSize: 13, color: '#90a4ae' }}>
          Complete 8-Gene Post-Squalene Cholesterol Biosynthesis & Isoprenoid Pathway Reference · 320 patients · seeds 2782-2789
        </div>
        <div style={{ fontSize: 11, color: '#546e7a', marginTop: 4 }}>
          DHCR7 · EBP · NSDHL · SC5D · DHCR24 · LSS · MVK · SQLE
        </div>
      </div>

      {/* Gene chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', marginBottom: 20 }}>
        {genes.map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#b71c1c' : '#1e2a31',
            color: '#fff', border: '1px solid #37474f', borderRadius: 6,
            padding: '6px 16px', cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', flexWrap: 'wrap', marginBottom: 20 }}>
            <StatBadge label="Total Patients" value={overview.total_patients} color="#b71c1c" />
            <StatBadge label="Genes" value={overview.genes?.length} color="#4a148c" />
            <StatBadge label="Avg Cholesterol" value={`${overview.summary?.avg_plasma_cholesterol_mgdl} mg/dL`} color="#006064" />
            <StatBadge label="Seeds" value={overview.seeds} color="#37474f" />
          </div>

          <Section title="Pathway Steps (Upstream → Cholesterol)" color="#b71c1c">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.biosynthesis_pathway_steps && Object.entries(overview.biosynthesis_pathway_steps).map(([step, desc]) => (
                <div key={step} style={{ marginBottom: 8 }}>
                  <span style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12 }}>{step.replace(/_/g,' ')}: </span>
                  <span style={{ color: '#cfd8dc', fontSize: 11 }}>{desc}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Pathognomonic Signs by Gene" color="#4a148c">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.pathognomonic_signs && Object.entries(overview.pathognomonic_signs).map(([gene, sign]) => (
                <div key={gene} style={{ marginBottom: 8, display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ color: GENE_COLORS[gene] || '#90a4ae', fontWeight: 700, fontSize: 12, minWidth: 60 }}>{gene}</span>
                  <span style={{ color: '#cfd8dc', fontSize: 11 }}>{sign}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Critical Statin Rules" color="#e65100">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.critical_statin_rules && Object.entries(overview.critical_statin_rules).map(([gene, rule]) => (
                <div key={gene} style={{ marginBottom: 8 }}>
                  <span style={{ color: GENE_COLORS[gene.split('_')[0]] || '#e65100', fontWeight: 700, fontSize: 12 }}>{gene.replace(/_/g,' ')}: </span>
                  <span style={{ color: '#cfd8dc', fontSize: 11 }}>{rule}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Per-Gene Summary" color="#1b5e20">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {overview.summary?.per_gene && Object.entries(overview.summary.per_gene).map(([gene, info]) => (
                <div key={gene} style={{
                  background: '#1e2a31', border: `1px solid ${GENE_COLORS[gene] || '#37474f'}`,
                  borderRadius: 8, padding: '10px 14px', minWidth: 200
                }}>
                  <div style={{ color: GENE_COLORS[gene], fontWeight: 700, fontSize: 14, marginBottom: 4 }}>{gene}</div>
                  <div style={{ color: '#90a4ae', fontSize: 11 }}>{info.locus} · {info.protein_size}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11, marginTop: 4 }}>n={info.n} · avg chol={info.avg_cholesterol} mg/dL</div>
                  <div style={{ color: '#78909c', fontSize: 10, marginTop: 4 }}>{info.disease}</div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Cascade Testing" color="#f57f17">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 10, color: '#cfd8dc', fontSize: 12 }}>
              {overview.cascade_testing}
            </div>
          </Section>
        </div>
      )}

      {/* Gene Breakdown Tab */}
      {tab === 'Gene Breakdown' && breakdown && breakdown[activeGene] && (
        <div>
          <div style={{ marginBottom: 12 }}>
            <span style={{ color: GENE_COLORS[activeGene], fontSize: 18, fontWeight: 700 }}>{activeGene}</span>
            <span style={{ color: '#546e7a', fontSize: 12, marginLeft: 12 }}>{GENE_INFO[activeGene]?.locus} · {GENE_INFO[activeGene]?.inh}</span>
          </div>
          <div style={{ color: '#90a4ae', fontSize: 12, marginBottom: 16 }}>{GENE_INFO[activeGene]?.size}</div>

          <Section title="Key Facts" color={GENE_COLORS[activeGene]}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              {breakdown[activeGene].key_facts?.map(f => (
                <span key={f} style={{ background: '#1e2a31', border: `1px solid ${GENE_COLORS[activeGene]}`, borderRadius: 4, padding: '3px 8px', fontSize: 10, color: '#cfd8dc' }}>{f}</span>
              ))}
            </div>
          </Section>

          <Section title="Inheritance & Gene Function" color="#90a4ae">
            <ClinicalText text={breakdown[activeGene].inheritance} />
          </Section>

          <Section title="Disease Category & Clinical Features" color={GENE_COLORS[activeGene]}>
            <ClinicalText text={breakdown[activeGene].disease_category} />
          </Section>

          <Section title="Disease Pathway & Mechanism" color="#4a148c">
            <ClinicalText text={breakdown[activeGene].disease_pathway} />
          </Section>

          <Section title="Pathognomonic Features & DDx" color="#b71c1c">
            <ClinicalText text={breakdown[activeGene].pathognomonic} />
          </Section>

          <Section title="Treatment" color="#1b5e20">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12, fontSize: 12, color: '#cfd8dc', lineHeight: 1.7 }}>
              {breakdown[activeGene].treatment}
            </div>
          </Section>

          <Section title="Sample Patients" color="#37474f">
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#1e2a31' }}>
                    {['Patient ID', 'Age', 'Sex', 'Cholesterol (mg/dL)', 'Key Phenotype'].map(h => (
                      <th key={h} style={{ padding: '6px 10px', color: '#90a4ae', textAlign: 'left', borderBottom: '1px solid #37474f' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown[activeGene].sample_patients?.map((p, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#1a2730' : '#1e2a31' }}>
                      <td style={{ padding: '5px 10px', color: '#cfd8dc' }}>{p.patient_id}</td>
                      <td style={{ padding: '5px 10px', color: '#90a4ae' }}>{p.age}</td>
                      <td style={{ padding: '5px 10px', color: '#90a4ae' }}>{p.sex}</td>
                      <td style={{ padding: '5px 10px', color: '#cfd8dc' }}>{p.plasma_cholesterol_mgdl}</td>
                      <td style={{ padding: '5px 10px', color: '#78909c', fontSize: 10 }}>{p.key_phenotype}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Section>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && overview && (
        <div>
          <Section title="Biosynthesis Pathway — Full 8-Gene Spectrum" color="#b71c1c">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 14 }}>
              <div style={{ color: '#78909c', fontSize: 11, marginBottom: 10, fontStyle: 'italic' }}>
                Acetyl-CoA → HMG-CoA → (HMGCR/statin target) → mevalonate → (MVK) → farnesyl-PP → (FDFT1) → squalene → (SQLE) → 2,3-oxidosqualene → (LSS) → lanosterol → (NSDHL/C4-demethylation) → (EBP/Δ8→Δ7) → (SC5D/lathosterol→7-DHC) → 7-DHC → (DHCR7) → CHOLESTEROL; [Bloch: ...→desmosterol→(DHCR24)→cholesterol]
              </div>
              {genes.map(gene => (
                <div key={gene} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid #263238' }}>
                  <div style={{ minWidth: 70, color: GENE_COLORS[gene], fontWeight: 700, fontSize: 14 }}>{gene}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: '#90a4ae', fontSize: 11 }}>{GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.size?.split('(')[0]?.trim()}</div>
                    <div style={{ color: '#cfd8dc', fontSize: 12, marginTop: 4 }}>{overview.pathognomonic_signs?.[gene]}</div>
                    <div style={{ color: '#546e7a', fontSize: 10, marginTop: 4 }}>{GENE_INFO[gene]?.inh}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Statin Rule Matrix (CRITICAL — gene-dependent)" color="#e65100">
            <div style={{ background: '#1e2a31', borderRadius: 8, padding: 12 }}>
              {overview.critical_statin_rules && Object.entries(overview.critical_statin_rules).map(([key, rule]) => {
                const gene = key.split('_')[0];
                const isCI = rule.includes('CONTRAINDICATED');
                const isHelp = rule.includes('HELP') || rule.includes('PARADOX');
                return (
                  <div key={key} style={{ display: 'flex', gap: 10, marginBottom: 8, alignItems: 'flex-start' }}>
                    <span style={{
                      background: isCI ? '#b71c1c' : isHelp ? '#1b5e20' : '#37474f',
                      color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 10, fontWeight: 700, minWidth: 80, textAlign: 'center'
                    }}>{isCI ? 'CONTRAIND' : isHelp ? 'HELPS' : 'CAUTION'}</span>
                    <span style={{ color: GENE_COLORS[gene] || '#90a4ae', fontWeight: 700, fontSize: 12, minWidth: 60 }}>{gene}</span>
                    <span style={{ color: '#cfd8dc', fontSize: 11 }}>{rule}</span>
                  </div>
                );
              })}
            </div>
          </Section>

          <Section title="Biochemical Fingerprints" color="#37474f">
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
              {[
                { gene: 'DHCR7', marker: '7-DHC elevated', normal: 'desmosterol/lathosterol', color: '#b71c1c' },
                { gene: 'EBP', marker: '8-DHC / 8(9)-cholestenol elevated', normal: '7-DHC absent', color: '#4a148c' },
                { gene: 'NSDHL', marker: 'C4-methylated sterols elevated', normal: 'skin sterol analysis', color: '#006064' },
                { gene: 'SC5D', marker: 'Lathosterol elevated; 7-DHC absent', normal: 'lathocholate in bile', color: '#1b5e20' },
                { gene: 'DHCR24', marker: 'Desmosterol elevated; no 7-DHC', normal: 'thick calvariae', color: '#e65100' },
                { gene: 'LSS', marker: '2,3-oxidosqualene elevated; lanosterol absent', normal: 'hair GC-MS', color: '#37474f' },
                { gene: 'MVK', marker: 'Mevalonic acid elevated (urine during attack)', normal: 'IgD >100 IU/mL', color: '#f57f17' },
                { gene: 'SQLE', marker: 'Squalene elevated in sebum/skin', normal: 'GC-MS sebum lipids', color: '#880e4f' },
              ].map(({ gene, marker, normal, color }) => (
                <div key={gene} style={{ background: '#1e2a31', border: `1px solid ${color}`, borderRadius: 8, padding: '10px 14px', minWidth: 200 }}>
                  <div style={{ color, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{gene}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11 }}>↑ {marker}</div>
                  <div style={{ color: '#546e7a', fontSize: 10, marginTop: 4 }}>Also: {normal}</div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <Section title="Glossary" color="#b71c1c">
            <div style={{ columns: 1, gap: 20 }}>
              {Object.entries(definitions.terms || {}).map(([term, def]) => (
                <div key={term} style={{ background: '#1e2a31', borderRadius: 8, padding: '10px 14px', marginBottom: 10, breakInside: 'avoid' }}>
                  <div style={{ color: '#b71c1c', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{term.replace(/_/g, ' ')}</div>
                  <div style={{ color: '#cfd8dc', fontSize: 11, lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 40, color: '#37474f', fontSize: 10, borderTop: '1px solid #1e2a31', paddingTop: 12 }}>
        Hereditary-Sterol-Biosynthesis-Atlas · 8-gene · 320 patients · seeds 2782-2789 · DHCR7-EBP-NSDHL-SC5D-DHCR24-LSS-MVK-SQLE · Registered 2026-09-15
      </div>
    </div>
  );
}
