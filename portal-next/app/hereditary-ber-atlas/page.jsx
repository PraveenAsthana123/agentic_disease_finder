'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ber-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MUTYH': '#b71c1c',  // deep red — MAP most common BER polyposis, 40-100 adenomas, CRC 50-80%
  'OGG1':  '#e65100',  // deep orange — 8-oxoG bifunctional glycosylase/AP-lyase, Lys326Gln controversy
  'NTHL1': '#1b5e20',  // dark green — NTHL1-NAP polyposis + breast + endometrial + urothelial
  'NEIL1': '#0d47a1',  // deep blue — formamidopyrimidine/ring-opened purine; metabolic syndrome mice
  'NEIL2': '#4a148c',  // deep purple — transcribed-strand-preferring 5-OHU TC-BER
  'NEIL3': '#006064',  // deep teal — G-quadruplex unhooking; ICL-backup FA pathway; meiotic SSB
  'UNG':   '#37474f',  // dark slate — HIGM5 class-switch recombination; AID-uracil substrate; IVIG
  'MPG':   '#bf360c',  // burnt sienna — alkylated base 3-MeA/7-MeG; TMZ pharmacogenomics
};

const GENE_INFO = {
  'MUTYH': { full: 'MUTYH / MutY DNA Glycosylase / 546aa', locus: '1p34.1', size: '546 aa / 60 kDa (Y-family; adenine opposite 8-oxoG; MAP biallelic)', inh: 'AR' },
  'OGG1':  { full: 'OGG1 / 8-Oxoguanine DNA Glycosylase 1 / 345aa', locus: '3p25.3', size: '345 aa / 39 kDa (bifunctional glycosylase/AP-lyase; OG:C repair)', inh: 'AR' },
  'NTHL1': { full: 'NTHL1 / Endonuclease III-Like 1 / 312aa', locus: '16p13.3', size: '312 aa / 35 kDa (HhH-GPD; formamidopyrimidine + oxidised pyrimidine; NAP)', inh: 'AR' },
  'NEIL1': { full: 'NEIL1 / Nei Endonuclease VIII-Like 1 / 390aa', locus: '15q24.2', size: '390 aa / 44 kDa (Fpg/Nei fold; FapyAde; β-δ lyase; TC-BER)', inh: 'AR' },
  'NEIL2': { full: 'NEIL2 / Nei Endonuclease VIII-Like 2 / 342aa', locus: '8p21.3', size: '342 aa / 38 kDa (transcribed-strand-preferring; 5-OHU; β-δ lyase)', inh: 'AR' },
  'NEIL3': { full: 'NEIL3 / Nei Endonuclease VIII-Like 3 / 605aa', locus: '4q24', size: '605 aa / 68 kDa (G-quadruplex Sp/Gh; ICL-backup; meiotic SSB)', inh: 'AR' },
  'UNG':   { full: 'UNG / Uracil-DNA Glycosylase / 304aa', locus: '12q24.11', size: '304 aa / 35 kDa (UNG2 nuclear + UNG1 mito; CSR; HIGM5)', inh: 'AR' },
  'MPG':   { full: 'MPG / Methylpurine-DNA Glycosylase / 298aa', locus: '16p13.3', size: '298 aa / 33 kDa (monofunctional; 3-MeA, 7-MeG, εA; TMZ pharmacogenomics)', inh: 'AR' },
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
    <div style={{ background: '#1e2a31', border: `1px solid ${color || '#37474f'}`, borderRadius: 8, padding: '10px 14px', minWidth: 110, textAlign: 'center', margin: 4 }}>
      <div style={{ color: color || '#90a4ae', fontSize: 11, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#fff', fontSize: 22, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

function OverviewPanel({ data }) {
  if (!data) return null;
  const { summary = [], pathway_categories = [], critical_distinctions = [] } = data;
  return (
    <div>
      <h2 style={{ color: '#ef5350', marginBottom: 8 }}>Hereditary BER Atlas — 8-Gene Base Excision Repair Reference</h2>
      <p style={{ color: '#90a4ae', marginBottom: 16 }}>
        MUTYH · OGG1 · NTHL1 · NEIL1 · NEIL2 · NEIL3 · UNG · MPG &mdash; {data.total_patients} patients (8 × 40), seeds {data.seeds}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 24 }}>
        {summary.map(g => (
          <div key={g.gene} style={{ background: '#1e2a31', border: `2px solid ${GENE_COLORS[g.gene]}`, borderRadius: 10, padding: 12, minWidth: 170 }}>
            <div style={{ color: GENE_COLORS[g.gene], fontWeight: 700, fontSize: 16 }}>{g.gene}</div>
            <div style={{ color: '#90a4ae', fontSize: 11 }}>{g.locus} · {g.inheritance.split(';')[0]}</div>
            <div style={{ color: '#cfd8dc', fontSize: 11, marginTop: 4 }}>{g.protein_size.split('(')[0].trim()}</div>
            <div style={{ marginTop: 8, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, fontSize: 11 }}>
              {g.pct_colorectal_cancer > 0 && <span style={{ color: '#ef9a9a' }}>CRC {g.pct_colorectal_cancer}%</span>}
              {g.pct_breast_cancer > 0 && <span style={{ color: '#f48fb1' }}>Breast {g.pct_breast_cancer}%</span>}
              {g.pct_endometrial_cancer > 0 && <span style={{ color: '#ce93d8' }}>Endo {g.pct_endometrial_cancer}%</span>}
              {g.pct_urothelial_cancer > 0 && <span style={{ color: '#80cbc4' }}>Uro {g.pct_urothelial_cancer}%</span>}
              {g.pct_hyper_igm > 0 && <span style={{ color: '#80deea' }}>HIGM {g.pct_hyper_igm}%</span>}
              {g.pct_alkylated_sensitivity > 0 && <span style={{ color: '#ffe082' }}>AlkSens {g.pct_alkylated_sensitivity}%</span>}
              {g.avg_polyp_burden > 1 && <span style={{ color: '#a5d6a7' }}>Polyps ~{g.avg_polyp_burden}</span>}
            </div>
          </div>
        ))}
      </div>

      <h3 style={{ color: '#90a4ae', marginBottom: 8 }}>Pathway Categories</h3>
      {pathway_categories.map((cat, i) => (
        <div key={i} style={{ background: '#1e2a31', border: '1px solid #37474f', borderRadius: 8, padding: 12, marginBottom: 10 }}>
          <div style={{ color: '#ef5350', fontWeight: 600, marginBottom: 4 }}>{cat.pathway}</div>
          <div style={{ display: 'flex', gap: 6, marginBottom: 6, flexWrap: 'wrap' }}>
            {cat.genes.map(g => <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600 }}>{g}</span>)}
          </div>
          <div style={{ color: '#90a4ae', fontSize: 12 }}>{cat.note}</div>
        </div>
      ))}

      <h3 style={{ color: '#90a4ae', margin: '20px 0 8px' }}>Critical Clinical Distinctions</h3>
      {critical_distinctions.map((d, i) => (
        <div key={i} style={{ background: '#1e2a31', border: '1px solid #b71c1c', borderLeft: '4px solid #ef5350', borderRadius: 6, padding: '8px 12px', marginBottom: 6, color: '#cfd8dc', fontSize: 12 }}>
          {d}
        </div>
      ))}
    </div>
  );
}

function GeneTablePanel({ data }) {
  if (!data?.genes) return null;
  const cols = ['Gene', 'Locus', 'Size', 'Inh.', 'CRC%', 'Breast%', 'Endo%', 'Uro%', 'HIGM%', 'AlkSens%', 'Colectomy%', 'Aspirin%', 'Avg Polyps'];
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1e2a31' }}>
            {cols.map(c => <th key={c} style={{ padding: '8px 10px', color: '#90a4ae', textAlign: 'left', borderBottom: '2px solid #37474f', whiteSpace: 'nowrap' }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {data.genes.map(g => (
            <tr key={g.gene} style={{ borderBottom: '1px solid #263238', background: 'transparent' }}>
              <td style={{ padding: '8px 10px', color: GENE_COLORS[g.gene], fontWeight: 700 }}>{g.gene}</td>
              <td style={{ padding: '8px 10px', color: '#cfd8dc' }}>{g.locus}</td>
              <td style={{ padding: '8px 10px', color: '#90a4ae', whiteSpace: 'nowrap' }}>{g.protein_size.split('(')[0].trim()}</td>
              <td style={{ padding: '8px 10px', color: '#cfd8dc' }}>{GENE_INFO[g.gene]?.inh}</td>
              <td style={{ padding: '8px 10px', color: g.pct_colorectal_cancer > 40 ? '#ef9a9a' : '#cfd8dc' }}>{g.pct_colorectal_cancer}%</td>
              <td style={{ padding: '8px 10px', color: g.pct_breast_cancer > 20 ? '#f48fb1' : '#cfd8dc' }}>{g.pct_breast_cancer}%</td>
              <td style={{ padding: '8px 10px', color: g.pct_endometrial_cancer > 20 ? '#ce93d8' : '#cfd8dc' }}>{g.pct_endometrial_cancer}%</td>
              <td style={{ padding: '8px 10px', color: g.pct_urothelial_cancer > 10 ? '#80cbc4' : '#cfd8dc' }}>{g.pct_urothelial_cancer}%</td>
              <td style={{ padding: '8px 10px', color: g.pct_hyper_igm > 50 ? '#80deea' : '#cfd8dc' }}>{g.pct_hyper_igm}%</td>
              <td style={{ padding: '8px 10px', color: g.pct_alkylated_sensitivity > 30 ? '#ffe082' : '#cfd8dc' }}>{g.pct_alkylated_sensitivity}%</td>
              <td style={{ padding: '8px 10px', color: '#cfd8dc' }}>{g.pct_colectomy}%</td>
              <td style={{ padding: '8px 10px', color: '#cfd8dc' }}>{g.pct_aspirin}%</td>
              <td style={{ padding: '8px 10px', color: g.avg_polyp_burden > 10 ? '#a5d6a7' : '#cfd8dc' }}>{g.avg_polyp_burden}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasPanel({ data }) {
  const [activeGene, setActiveGene] = useState('MUTYH');
  if (!data?.genes) return null;
  const geneData = data.genes.find(g => g.gene === activeGene);
  if (!geneData) return null;
  const sections = [
    { label: 'Inheritance & Penetrance', key: 'inheritance', color: '#ef5350' },
    { label: 'Disease Category', key: 'disease_category', color: '#9c27b0' },
    { label: 'Disease Pathway', key: 'disease_pathway', color: '#1976d2' },
    { label: 'Pathognomonic / Diagnosis', key: 'pathognomonic', color: '#00897b' },
    { label: 'Treatment', key: 'treatment', color: '#f57c00' },
  ];
  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', flexWrap: 'wrap' }}>
        {Object.keys(GENE_COLORS).map(g => <GeneChip key={g} gene={g} active={activeGene === g} onClick={setActiveGene} />)}
      </div>
      <div style={{ background: `${GENE_COLORS[activeGene]}22`, border: `2px solid ${GENE_COLORS[activeGene]}`, borderRadius: 10, padding: 16, marginBottom: 16 }}>
        <div style={{ color: GENE_COLORS[activeGene], fontSize: 22, fontWeight: 700 }}>{geneData.gene}</div>
        <div style={{ color: '#90a4ae', fontSize: 13 }}>{geneData.locus} · {geneData.protein_size}</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 12 }}>
          <StatBadge label="CRC" value={`${geneData.pct_colorectal_cancer}%`} color="#ef9a9a" />
          <StatBadge label="Breast" value={`${geneData.pct_breast_cancer}%`} color="#f48fb1" />
          <StatBadge label="Endo" value={`${geneData.pct_endometrial_cancer}%`} color="#ce93d8" />
          <StatBadge label="Uro" value={`${geneData.pct_urothelial_cancer}%`} color="#80cbc4" />
          <StatBadge label="HIGM" value={`${geneData.pct_hyper_igm}%`} color="#80deea" />
          <StatBadge label="AlkSens" value={`${geneData.pct_alkylated_sensitivity}%`} color="#ffe082" />
          <StatBadge label="Colectomy" value={`${geneData.pct_colectomy}%`} color="#a5d6a7" />
          <StatBadge label="Aspirin" value={`${geneData.pct_aspirin}%`} color="#ffcc02" />
          <StatBadge label="Avg Polyps" value={geneData.avg_polyp_burden} color="#a5d6a7" />
          <StatBadge label="n" value={geneData.n_patients} color="#90a4ae" />
        </div>
      </div>
      {sections.map(s => (
        <div key={s.key} style={{ background: '#1e2a31', border: `1px solid ${s.color}44`, borderLeft: `4px solid ${s.color}`, borderRadius: 8, padding: 14, marginBottom: 10 }}>
          <div style={{ color: s.color, fontWeight: 600, marginBottom: 6 }}>{s.label}</div>
          <pre style={{ color: '#cfd8dc', fontSize: 11, whiteSpace: 'pre-wrap', margin: 0, fontFamily: 'inherit', lineHeight: 1.6 }}>{geneData[s.key]}</pre>
        </div>
      ))}
      <div style={{ background: '#1e2a31', border: '1px solid #37474f', borderRadius: 8, padding: 14, marginTop: 12 }}>
        <div style={{ color: '#90a4ae', fontWeight: 600, marginBottom: 8 }}>Patient Sample ({geneData.patients?.length} patients)</div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#263238' }}>
                {['ID', 'Age', 'Sex', 'CRC', 'Breast', 'Endo', 'Uro', 'HIGM', 'Immuno', 'AlkSens', 'Colect', 'Aspirin', 'Polyps'].map(h => (
                  <th key={h} style={{ padding: '4px 8px', color: '#90a4ae', textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(geneData.patients || []).slice(0, 20).map(p => (
                <tr key={p.id} style={{ borderBottom: '1px solid #1e2a31' }}>
                  <td style={{ padding: '3px 8px', color: GENE_COLORS[geneData.gene] }}>{p.id}</td>
                  <td style={{ padding: '3px 8px', color: '#cfd8dc' }}>{p.age}</td>
                  <td style={{ padding: '3px 8px', color: '#cfd8dc' }}>{p.sex}</td>
                  <td style={{ padding: '3px 8px', color: p.colorectal_cancer ? '#ef9a9a' : '#546e7a' }}>{p.colorectal_cancer ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.breast_cancer ? '#f48fb1' : '#546e7a' }}>{p.breast_cancer ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.endometrial_cancer ? '#ce93d8' : '#546e7a' }}>{p.endometrial_cancer ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.urothelial_cancer ? '#80cbc4' : '#546e7a' }}>{p.urothelial_cancer ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.hyper_igm_syndrome ? '#80deea' : '#546e7a' }}>{p.hyper_igm_syndrome ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.immunodeficiency ? '#80deea' : '#546e7a' }}>{p.immunodeficiency ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.alkylated_base_sensitivity ? '#ffe082' : '#546e7a' }}>{p.alkylated_base_sensitivity ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.colectomy_performed ? '#a5d6a7' : '#546e7a' }}>{p.colectomy_performed ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.aspirin_use ? '#ffcc02' : '#546e7a' }}>{p.aspirin_use ? 'Y' : '-'}</td>
                  <td style={{ padding: '3px 8px', color: p.polyp_burden > 10 ? '#a5d6a7' : '#cfd8dc' }}>{p.polyp_burden}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsPanel({ data }) {
  if (!data?.glossary) return null;
  return (
    <div>
      <h3 style={{ color: '#ef5350', marginBottom: 12 }}>Glossary — BER Atlas</h3>
      {Object.entries(data.glossary).map(([term, def]) => (
        <div key={term} style={{ background: '#1e2a31', border: '1px solid #37474f', borderRadius: 8, padding: 12, marginBottom: 8 }}>
          <div style={{ color: '#ef5350', fontWeight: 600, marginBottom: 4 }}>{term}</div>
          <div style={{ color: '#cfd8dc', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
        </div>
      ))}
      {data.standards && (
        <div style={{ marginTop: 20 }}>
          <h3 style={{ color: '#90a4ae', marginBottom: 8 }}>Standards &amp; Key References</h3>
          {data.standards.map((s, i) => (
            <div key={i} style={{ color: '#78909c', fontSize: 12, padding: '4px 0', borderBottom: '1px solid #263238' }}>{s}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function BERAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const base = `${API}/api/${SLUG}`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#fff', padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <div style={{ marginBottom: 8, fontSize: 12, color: '#546e7a' }}>
          Hereditary Cancer Genetics &rsaquo; DNA Repair Disorders &rsaquo; <span style={{ color: '#ef5350' }}>BER Atlas</span>
        </div>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: '#ef5350', marginBottom: 4 }}>
          &#x1f9ec; Hereditary BER Atlas — 8-Gene Base Excision Repair Reference
        </h1>
        <p style={{ color: '#78909c', marginBottom: 20, fontSize: 14 }}>
          MUTYH · OGG1 · NTHL1 · NEIL1 · NEIL2 · NEIL3 · UNG · MPG — Complete Base Excision Repair Clinical Reference
          · 320 patients (8 × 40) · Seeds 2742–2749
        </p>

        <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #1e2a31', paddingBottom: 0 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: 'none', border: 'none', color: tab === t ? '#ef5350' : '#546e7a',
              borderBottom: tab === t ? '2px solid #ef5350' : '2px solid transparent',
              padding: '10px 18px', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
              fontSize: 14, marginBottom: -2,
            }}>{t}</button>
          ))}
        </div>

        {loading && <div style={{ color: '#90a4ae', padding: 40, textAlign: 'center' }}>Loading BER Atlas…</div>}
        {error && <div style={{ color: '#f44336', padding: 20 }}>Error: {error}</div>}
        {!loading && !error && (
          <>
            {tab === 'Overview' && <OverviewPanel data={overview} />}
            {tab === 'Gene Table' && <GeneTablePanel data={breakdown} />}
            {tab === 'Clinical Atlas' && <ClinicalAtlasPanel data={breakdown} />}
            {tab === 'Definitions' && <DefinitionsPanel data={definitions} />}
          </>
        )}
      </div>
    </div>
  );
}
