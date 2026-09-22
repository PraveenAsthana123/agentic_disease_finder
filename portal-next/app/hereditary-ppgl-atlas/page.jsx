'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ppgl-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'VHL':    '#1565c0',  // deep blue     — VHL disease, clear cell RCC
  'SDHB':   '#b71c1c',  // deep red      — PPGL2, highest malignancy 40%
  'SDHD':   '#4a148c',  // deep purple   — PPGL1, head-neck, paternal imprinting
  'SDHA':   '#e65100',  // deep orange   — PPGL5, GIST, imatinib-resistant
  'SDHC':   '#00695c',  // dark teal     — PPGL3, head-neck, no imprinting
  'SDHAF2': '#2e7d32',  // dark green    — PGL2, exclusively head-neck, ultra-rare
  'RET':    '#f57f17',  // amber         — MEN2A/2B, MTC + PHEO + PHPT
  'MAX':    '#6a1b9a',  // purple        — bilateral adrenal, paternal imprinting
};

const GENE_INFO = {
  'VHL': {
    full: 'VHL / Von Hippel-Lindau Tumour Suppressor / 213aa',
    locus: '3p25.3',
    size: '213 aa / 24 kDa (E3 ubiquitin ligase adaptor; targets HIF-1α for degradation under normoxia; LOF → HIF-1α accumulates → VEGF/EPO/GLUT1 → pseudohypoxia; VHL disease: PPGL + clear cell RCC + hemangioblastoma; Type 2 missense = PHEO; MLPA mandatory 20% deletions; biallelic loss required; AD)',
    inh: 'AD LOF',
    imprinting: null,
  },
  'SDHB': {
    full: 'SDHB / Succinate Dehydrogenase Iron-Sulphur Subunit B / 280aa',
    locus: '1p36.13',
    size: '280 aa / 32 kDa (3 Fe-S clusters [2Fe-2S][4Fe-4S][3Fe-4S]; electron relay SDHA→ubiquinone; SDH LOF → succinate → PHD inhibition → HIF pseudohypoxia + TET inhibition → CIMP; PPGL2 highest malignancy 40%; extra-adrenal predilection; SDHB IHC absent = any SDHx mutation; AD)',
    inh: 'AD LOF',
    imprinting: null,
  },
  'SDHD': {
    full: 'SDHD / Succinate Dehydrogenase Subunit D / 159aa',
    locus: '11q23.1',
    size: '159 aa / 17 kDa (2 TM helices; ubiquinone-binding pocket anchor; maternal imprinting = only PATERNAL SDHD mutation causes PPGL1; head-neck PGL predominantly; carotid body/jugulotympanic; multifocal 50%; malignancy 5%; methoxytyramine biomarker; AD paternal imprinting)',
    inh: 'AD LOF (maternal imprinting)',
    imprinting: 'PATERNAL',
  },
  'SDHA': {
    full: 'SDHA / Succinate Dehydrogenase Flavoprotein Subunit A / 664aa',
    locus: '5p15.33',
    size: '664 aa / 73 kDa (catalytic subunit; FAD covalently at His99; oxidises succinate→fumarate; SDHA IHC absent = SDHA mutation specific; PPGL5 + SDH-deficient GIST + pituitary adenoma; lowest penetrance SDHx 10-20%; imatinib-resistant GIST diagnostic pearl; AD)',
    inh: 'AD LOF',
    imprinting: null,
  },
  'SDHC': {
    full: 'SDHC / Succinate Dehydrogenase Subunit C / 169aa',
    locus: '1q23.3',
    size: '169 aa / 18 kDa (1 TM helix; haem b coordination; anchors SDH with SDHD; PPGL3 head-neck predominantly; NO imprinting (unlike SDHD); carotid body/jugulotympanic; malignancy 2-5% lowest SDHx; AD)',
    inh: 'AD LOF',
    imprinting: null,
  },
  'SDHAF2': {
    full: 'SDHAF2 / SDH Assembly Factor 2 / 166aa',
    locus: '11q13.1',
    size: '166 aa / 18 kDa (assembly factor — covalent FAD attachment to SDHA His99; not present in mature complex; paternal imprinting same as SDHD; PGL2 EXCLUSIVELY head-neck (no adrenal pheo); ultra-rare; Dutch founder Gly78Val c.232G>T; AD paternal imprinting)',
    inh: 'AD LOF (paternal imprinting)',
    imprinting: 'PATERNAL',
  },
  'RET': {
    full: 'RET / RET Proto-Oncogene RTK / 1114aa',
    locus: '10q11.21',
    size: '1114 aa / 120 kDa (receptor tyrosine kinase; GDNF family ligands; GOF = constitutive kinase; MEN2A: MTC+PHEO+PHPT; MEN2B: M918T neonatal; C634 highest risk; EXCLUDE PHEO before any neck surgery; prophylactic thyroidectomy by codon (ATA-A→D); vandetanib/cabozantinib; AD GOF)',
    inh: 'AD GOF',
    imprinting: null,
  },
  'MAX': {
    full: 'MAX / MYC Associated Factor X / 160aa',
    locus: '14q23.3',
    size: '160 aa / 18 kDa (bHLHLZ; MYC network dimerisation partner; LOF → MYC/MAD balance disrupted → chromaffin proliferation; bilateral adrenal PHEO in young males; paternal imprinting same as SDHD/SDHAF2; cortical-sparing surgery preferred; AD paternal imprinting)',
    inh: 'AD LOF (paternal imprinting)',
    imprinting: 'PATERNAL',
  },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22',
      color,
      border: `1px solid ${color}55`,
      borderRadius: 4,
      padding: '2px 7px',
      fontSize: 11,
      fontWeight: 600,
      marginRight: 4,
    }}>{text}</span>
  );
}

function ImprintingBadge({ gene }) {
  const info = GENE_INFO[gene];
  if (!info?.imprinting) return null;
  return (
    <span style={{
      background: '#ff6f001a',
      color: '#e65100',
      border: '1px solid #ff6f0055',
      borderRadius: 4,
      padding: '2px 7px',
      fontSize: 11,
      fontWeight: 700,
      marginLeft: 4,
    }}>⚠ PATERNAL IMPRINTING</span>
  );
}

export default function HreditaryPPGLAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = {
      'Overview': 'overview',
      'Gene Table': 'breakdown',
      'Clinical Atlas': 'breakdown',
      'Definitions': 'definitions',
    };
    const ep = endpoints[tab] || 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(data => {
        if (tab === 'Overview') setOverview(data);
        else if (tab === 'Gene Table' || tab === 'Clinical Atlas') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const cardStyle = {
    background: '#fff',
    border: '1px solid #e0e0e0',
    borderRadius: 8,
    padding: 16,
    marginBottom: 14,
    boxShadow: '0 1px 3px rgba(0,0,0,0.07)',
  };

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: '20px 16px' }}>
      <div style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1a1a2e', margin: '0 0 4px' }}>
          🧬 Hereditary Pheochromocytoma-Paraganglioma Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>
          Complete 8-Gene PPGL Reference · VHL-SDHB-SDHD-SDHA-SDHC-SDHAF2-RET-MAX · 320 Patients · Seeds 2982-2989
        </p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 16px', border: 'none', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
            background: tab === t ? '#1565c0' : 'transparent',
            color: tab === t ? '#fff' : '#555',
            borderRadius: '6px 6px 0 0', fontSize: 13,
            borderBottom: tab === t ? '2px solid #1565c0' : 'none',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888', padding: 20 }}>Loading…</div>}
      {error && <div style={{ color: '#c00', padding: 20 }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Genes', value: overview.total_genes },
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Seed Range', value: overview.seed_range },
              { label: 'Imprinted Genes', value: '3 (SDHD, SDHAF2, MAX)' },
              { label: 'GOF Oncogene', value: 'RET (MEN2)' },
              { label: 'Highest Malignancy', value: 'SDHB ~40%' },
            ].map(k => (
              <div key={k.label} style={{ ...cardStyle, textAlign: 'center', padding: 14 }}>
                <div style={{ fontSize: 20, fontWeight: 800, color: '#1565c0' }}>{k.value}</div>
                <div style={{ fontSize: 11, color: '#777', marginTop: 2 }}>{k.label}</div>
              </div>
            ))}
          </div>

          {/* Gene chips */}
          <div style={{ ...cardStyle }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>8 PPGL Genes — Loci &amp; Inheritance</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {overview.genes.map(g => (
                <div key={g} style={{
                  background: GENE_COLORS[g] + '15',
                  border: `1.5px solid ${GENE_COLORS[g]}55`,
                  borderRadius: 8, padding: '8px 12px', minWidth: 140,
                }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[g], fontSize: 15 }}>{g}</div>
                  <div style={{ fontSize: 10, color: '#555', marginTop: 2 }}>{overview.gene_loci[g]}</div>
                  <div style={{ fontSize: 10, color: '#777' }}>{overview.inheritance_modes[g]}</div>
                  {GENE_INFO[g]?.imprinting && (
                    <div style={{ fontSize: 9, color: '#e65100', fontWeight: 700, marginTop: 2 }}>⚠ PATERNAL IMPRINTING</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Key clinical rules */}
          <div style={cardStyle}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 10, color: '#b71c1c' }}>
              🔑 Key Clinical Rules
            </h3>
            {(overview.key_clinical_rules || []).map((rule, i) => {
              const [head, ...rest] = rule.split(':');
              return (
                <div key={i} style={{ borderLeft: '3px solid #1565c0', paddingLeft: 10, marginBottom: 8, fontSize: 12 }}>
                  <strong style={{ color: '#1565c0' }}>{head}:</strong>{' '}
                  <span style={{ color: '#444' }}>{rest.join(':')}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'Locus', 'Inheritance', 'Adrenal PHEO%', 'Head-Neck PGL%', 'Extra-Adrenal%', 'Malignant%', 'Bilateral%', 'Mean Age Dx', 'n'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {breakdown.genes.map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff', cursor: 'pointer' }}
                    onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>
                      {g.gene}
                      {GENE_INFO[g.gene]?.imprinting && <span style={{ fontSize: 9, color: '#e65100', marginLeft: 4 }}>⚠ P.IMP</span>}
                    </td>
                    <td style={{ padding: '7px 10px', color: '#555' }}>{g.locus}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh || g.inheritance?.slice(0, 30)}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.adrenal_pheo_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.head_neck_pgl_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.extra_adrenal_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', color: g.malignant_pct >= 20 ? '#b71c1c' : g.malignant_pct >= 10 ? '#e65100' : '#2e7d32', fontWeight: g.malignant_pct >= 20 ? 700 : 400 }}>
                      {g.malignant_pct}%{g.malignant_pct >= 20 ? ' ⚠' : ''}
                    </td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.bilateral_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{g.mean_age_dx}y</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right', color: '#888' }}>{g.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {expandedGene && (() => {
            const g = breakdown.genes.find(x => x.gene === expandedGene);
            if (!g) return null;
            return (
              <div style={{ ...cardStyle, marginTop: 16, borderLeft: `4px solid ${GENE_COLORS[expandedGene]}` }}>
                <h3 style={{ color: GENE_COLORS[expandedGene], marginTop: 0, fontSize: 15 }}>
                  {expandedGene} — {GENE_INFO[expandedGene]?.full}
                </h3>
                <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, marginBottom: 8, whiteSpace: 'pre-wrap' }}>
                  <strong>Protein/Function:</strong> {g.protein_size}
                </div>
                <div style={{ fontSize: 11, color: '#444', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  <strong>Disease Category:</strong> {g.disease_category}
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {breakdown.genes.map(g => (
            <div key={g.gene} style={{ ...cardStyle, borderLeft: `4px solid ${GENE_COLORS[g.gene]}` }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <h3 style={{ color: GENE_COLORS[g.gene], margin: 0, fontSize: 15 }}>
                  {g.gene}
                  <ImprintingBadge gene={g.gene} />
                  <span style={{ fontWeight: 400, fontSize: 12, color: '#555', marginLeft: 8 }}>
                    {g.locus} · {GENE_INFO[g.gene]?.inh}
                  </span>
                </h3>
                <div style={{ fontSize: 11, color: '#888' }}>n={g.n_patients} · mean age {g.mean_age_dx}y</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(120px,1fr))', gap: 8, marginBottom: 10 }}>
                {[
                  { label: 'Adrenal PHEO', val: g.adrenal_pheo_pct + '%' },
                  { label: 'Head-Neck PGL', val: g.head_neck_pgl_pct + '%' },
                  { label: 'Extra-Adrenal', val: g.extra_adrenal_pct + '%' },
                  { label: 'Malignant', val: g.malignant_pct + '%', warn: g.malignant_pct >= 20 },
                  { label: 'Bilateral', val: g.bilateral_pct + '%' },
                  { label: 'Tumour Size', val: g.mean_tumour_size_cm + ' cm' },
                ].map(m => (
                  <div key={m.label} style={{
                    background: m.warn ? '#ffebee' : '#f5f5f5',
                    borderRadius: 6, padding: '6px 10px', textAlign: 'center',
                    border: m.warn ? '1px solid #ef9a9a' : '1px solid #e0e0e0',
                  }}>
                    <div style={{ fontWeight: 700, color: m.warn ? '#b71c1c' : '#333', fontSize: 14 }}>{m.val}</div>
                    <div style={{ fontSize: 10, color: '#777' }}>{m.label}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#555', lineHeight: 1.6, whiteSpace: 'pre-wrap', borderTop: '1px solid #f0f0f0', paddingTop: 8 }}>
                {g.inheritance?.slice(0, 900)}
              </div>
              <div style={{ marginTop: 8 }}>
                <strong style={{ fontSize: 11 }}>Common mutations: </strong>
                {Object.keys(g.mutation_breakdown || {}).slice(0, 4).map(m => (
                  <Badge key={m} text={m} color={GENE_COLORS[g.gene]} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ ...cardStyle }}>
              <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginTop: 0, marginBottom: 8 }}>
                {d.term}
              </h3>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                {(d.genes || []).map(g => (
                  <span key={g} style={{
                    background: GENE_COLORS[g] + '22',
                    color: GENE_COLORS[g],
                    border: `1px solid ${GENE_COLORS[g]}55`,
                    borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 700,
                  }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
