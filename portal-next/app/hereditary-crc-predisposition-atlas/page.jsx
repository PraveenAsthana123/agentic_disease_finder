'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-crc-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Gene Breakdown', 'Definitions'];

const GENE_COLORS = {
  'APC':   '#b71c1c',  // deep red        — AD LOF; FAP; 100% CRC lifetime; WNT gatekeeper
  'MUTYH': '#e65100',  // deep orange     — AR LOF; MAP; biallelic; base-excision repair
  'MLH1':  '#1565c0',  // deep blue       — AD LOF; Lynch1; MSI-H; BRAF gate
  'MSH2':  '#1b5e20',  // dark green      — AD LOF; Lynch2; Muir-Torre; urothelial 14%
  'MSH6':  '#4a148c',  // dark purple     — AD LOF; Lynch3; endometrial 71%; MSI-L pitfall
  'PMS2':  '#006064',  // dark teal       — AD LOF; Lynch4; lowest penetrance; CMMRD biallelic
  'STK11': '#880e4f',  // dark magenta    — AD LOF; PJS; perioral pigmentation; SCTAT
  'SMAD4': '#33691e',  // dark lime       — AD LOF; JPS-HHT overlap; juvenile polyposis; aortic
};

const GENE_INFO = {
  'APC':   { full: 'APC / WNT Signalling Regulator / FAP / 2843aa', locus: '5q22.2',  inh: 'AD LOF',
             note: '2843aa/310kDa; WNT gatekeeper β-catenin destruction complex; 100% CRC lifetime; CHRPE PATHOGNOMONIC; prophylactic colectomy 20-25yr MANDATORY; codon 1309 = dense polyposis; AFAP codons <200 or >1400' },
  'MUTYH': { full: 'MUTYH / MutY DNA Glycosylase / MAP / 546aa',   locus: '1p34.1',  inh: 'AR LOF',
             note: '546aa/60kDa; base-excision repair oxoG lesions; BIALLELIC required for MAP; Y179C+G396D Northern-European founders; MSS (NOT MSI-H unlike Lynch); APC test first if dense polyposis; partner MANDATORY' },
  'MLH1':  { full: 'MLH1 / MutL Homolog 1 / Lynch1 / 756aa',       locus: '3p22.2',  inh: 'AD LOF',
             note: '756aa/85kDa; MMR endonuclease; MSI-H PATHOGNOMONIC; BRAF V600E absent = confirms germline (90% sporadic MLH1 = somatic methylation); aspirin CAPP2 50% risk reduction Level A; 40-50% CRC lifetime' },
  'MSH2':  { full: 'MSH2 / MutS Homolog 2 / Lynch2 / 934aa',       locus: '2p21',    inh: 'AD LOF',
             note: '934aa/100kDa; MMR mismatch recognition; Muir-Torre sebaceous gland neoplasms PATHOGNOMONIC; urothelial 14% HIGHEST Lynch gene; EPCAM 3\'-deletion MLPA MANDATORY if sequencing negative; 40-50% CRC' },
  'MSH6':  { full: 'MSH6 / MutS Homolog 6 / Lynch3 / 1360aa',      locus: '2p16.3',  inh: 'AD LOF',
             note: '1360aa/161kDa; MMR mismatch recognition G:T mismatches; endometrial 71% ABSOLUTE HIGHEST single MMR gene; MSI-L 30% false-negative (IHC primary test NOT MSI); attenuated CRC 10-22%; later onset' },
  'PMS2':  { full: 'PMS2 / PMS1 Homolog 2 / Lynch4 / 862aa',       locus: '7p22.1',  inh: 'AD LOF',
             note: '862aa/96kDa; MMR endonuclease accessory; LOWEST penetrance Lynch (CRC 15-20%); 4 PSEUDOGENES — MLPA MANDATORY (standard seq gives false positives); biallelic = CMMRD childhood café-au-lait + brain + haematologic' },
  'STK11': { full: 'STK11 / LKB1 / Serine-Threonine Kinase 11 / PJS / 433aa', locus: '19p13.3', inh: 'AD LOF',
             note: '433aa/48kDa; LKB1 master energy sensor AMPK pathway; perioral mucocutaneous pigmentation PATHOGNOMONIC; SCTAT ovarian PATHOGNOMONIC; GI intussusception EMERGENCY; gastric 29%; pancreatic 36%; HIGHEST extra-colonic risk Lynch' },
  'SMAD4': { full: 'SMAD4 / TGF-β Effector / JPS-HHT Overlap / 552aa', locus: '18q21.2', inh: 'AD LOF',
             note: '552aa/60kDa; TGF-β/BMP SMAD R-SMAD co-SMAD signalling; JPS-HHT OVERLAP = juvenile polyposis + HHT telangiectasia + AORTIC DILATATION (echo MANDATORY); protein-losing enteropathy; gastrectomy may be required' },
};

const PATHOGNOMONIC = {
  'APC':   'FAP: 100% CRC lifetime if unscreened — prophylactic colectomy 20-25yr MANDATORY, cannot defer. CHRPE (bilateral >4 lesions = >95% PPV for APC). Codon 1309 hotspot = dense polyposis + early-onset CRC. AFAP (codons <200 or >1400): attenuated polyposis, colonoscopy may suffice. Desmoid: codon >1310 highest risk. Spigelman stage guides duodenoscopy interval.',
  'MUTYH': 'MAP: BIALLELIC required — monoallelic alone not disease-causing. Y179C and G396D = Northern European founders; single-variant test INSUFFICIENT (must sequence full gene). CRITICAL DDx from Lynch: MAP tumours are MSS (NOT MSI-H) — IHC negative for MMR does NOT exclude MAP. Partner testing MANDATORY for carrier couples. APC test first if >1000 polyps.',
  'MLH1':  'Lynch1 — BRAF V600E GATE: 90% MLH1-deficient sporadic CRC: BRAF V600E present + somatic methylation = NOT germline Lynch. Germline = BRAF V600E absent. Aspirin 600mg/day CAPP2 trial: 50% CRC risk reduction (Level A evidence — only Lynch prevention with Level A). MSI-H = pembrolizumab-eligible (FDA 2017 tumour-agnostic). Lifetime CRC risk 40-50%.',
  'MSH2':  'Lynch2 — UROTHELIAL 14% HIGHEST of all Lynch genes (bladder + ureteral surveillance MANDATORY). Muir-Torre syndrome (sebaceous carcinomas/adenomas ± keratoacanthomas) PATHOGNOMONIC for MSH2/MSH6. EPCAM 3\'-deletion causes MSH2 silencing — MLPA test MANDATORY if MSH2 sequencing normal but IHC shows MSH2/MSH6 loss. Uterine surveillance from 30-35yr.',
  'MSH6':  'Lynch3 — MSI-L PITFALL: 30% MSH6-deficient CRC = MSI-L (not MSI-H). IHC ALWAYS primary test (MSH6 loss alone diagnostic). Endometrial 71% HIGHEST risk — hysterectomy/BSO counselling from 40yr. Later CRC onset (age 50-60s) vs MLH1/MSH2. Attenuated CRC 10-22%. Aspirin benefit as per CAPP2 (MLH1 data, extrapolated). Urology surveillance (urothelial 10%).',
  'PMS2':  'Lynch4 — LOWEST penetrance: CRC 15-20% lifetime; endometrial 15%; other cancers rare. 4 PSEUDOGENES make standard Sanger/NGS unreliable — MLPA MANDATORY for large deletions; specialized labs required. Biallelic PMS2 = CMMRD: childhood onset, café-au-lait macules (NF1-like), brain tumours, haematologic malignancy, MSI-H in ALL cancers. Pembrolizumab response in CMMRD reported.',
  'STK11': 'PJS — perioral mucocutaneous pigmented macules PATHOGNOMONIC (lips/buccal/digits from infancy). SCTAT (sex cord tumour with annular tubules) ovarian pathognomonic marker. GI INTUSSUSCEPTION: paediatric emergency — bowel obstruction; polypectomy to reduce risk. Highest extra-colonic risks of all CRC syndromes: pancreatic 36%, gastric 29%, breast 32-54%. Annual multi-organ surveillance.',
  'SMAD4': 'JPS-HHT OVERLAP (30-40% SMAD4 carriers): juvenile GI polyposis + HHT telangiectasia (epistaxis, pulmonary AVMs) + AORTIC DILATATION — echo MANDATORY at diagnosis and annually. Protein-losing enteropathy (hypoalbuminaemia, oedema) = severe JPS. Gastrectomy/colectomy may be required for unmanageable polyposis. DDx: BMPR1A = JPS without HHT (no aortic; no epistaxis).',
};

export default function HereditaryCRCPredispositionAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDef, setExpandedDef] = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#0d1b2a', minHeight: '100vh', fontSize: 18 }}>Loading Hereditary-CRC-Predisposition-Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#0d1b2a', minHeight: '100vh' }}>Error: {error}</div>;

  return (
    <div style={{ background: '#0d1117', minHeight: '100vh', color: '#e6edf3', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#1565c0 25%,#1b5e20 50%,#4a148c 75%,#880e4f 100%)', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 13, color: 'rgba(255,255,255,0.7)', marginBottom: 6 }}>Hereditary Colorectal Cancer · Lynch Syndrome · FAP · MAP · Peutz-Jeghers · JPS-HHT · Germline MMR</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: '#fff', lineHeight: 1.3 }}>
          🧬 Hereditary-CRC-Predisposition-Atlas
        </h1>
        <div style={{ marginTop: 8, fontSize: 13, color: 'rgba(255,255,255,0.85)' }}>
          Complete 8-Gene Reference · APC · MUTYH · MLH1 · MSH2 · MSH6 · PMS2 · STK11 · SMAD4 ·
          320 Patients (8×40) · Seeds 3398–3405
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Total Patients', val: overview?.total_patients },
            { label: 'Genes', val: overview?.genes?.length },
            { label: 'MSI-H Rate', val: overview?.msi_h_rate_pct != null ? `${overview.msi_h_rate_pct}%` : null },
            { label: 'Immunotherapy Eligible', val: overview?.immunotherapy_eligible_pct != null ? `${overview.immunotherapy_eligible_pct}%` : null },
            { label: 'Mean Age Dx', val: overview?.mean_age_at_dx != null ? `${overview.mean_age_at_dx} yr` : null },
            { label: 'Prophylactic Procedures', val: overview?.prophylactic_procedure_pct != null ? `${overview.prophylactic_procedure_pct}%` : null },
          ].map(({ label, val }) => (
            <div key={label} style={{ background: 'rgba(255,255,255,0.15)', borderRadius: 8, padding: '6px 14px' }}>
              <div style={{ fontSize: 11, opacity: 0.8 }}>{label}</div>
              <div style={{ fontWeight: 700, fontSize: 15 }}>{val ?? '…'}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#161b22', borderBottom: '1px solid #30363d', display: 'flex', gap: 0, overflowX: 'auto' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '10px 20px', background: activeTab === t ? '#1f2d3d' : 'transparent',
              color: activeTab === t ? '#58a6ff' : '#8b949e', border: 'none', borderBottom: activeTab === t ? '2px solid #58a6ff' : '2px solid transparent',
              cursor: 'pointer', fontSize: 14, whiteSpace: 'nowrap' }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ padding: '24px 32px', maxWidth: 1400, margin: '0 auto' }}>

        {/* OVERVIEW TAB */}
        {activeTab === 'Overview' && overview && (
          <div>
            {/* Atlas-wide stats bar */}
            <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 16 }}>📊 Atlas-Wide Cohort Statistics (n=320)</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 10 }}>
                {[
                  { label: 'MSI-H Rate', val: `${overview.msi_h_rate_pct}%`, note: 'dMMR tumours' },
                  { label: 'Immunotherapy Eligible', val: `${overview.immunotherapy_eligible_pct}%`, note: 'pembrolizumab / nivolumab' },
                  { label: 'Prophylactic Procedures', val: `${overview.prophylactic_procedure_pct}%`, note: 'colectomy / hysterectomy / BSO' },
                  { label: 'Aspirin Prescribed', val: `${overview.aspirin_prescribed_pct}%`, note: 'Lynch CAPP2 protocol' },
                  { label: 'Mean Age at Dx', val: `${overview.mean_age_at_dx} yr`, note: 'all genes combined' },
                  { label: 'Metastatic at Dx', val: `${overview.metastatic_pct}%`, note: 'stage IV presentation' },
                ].map(({ label, val, note }) => (
                  <div key={label} style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                    <div style={{ fontSize: 22, fontWeight: 700, color: '#58a6ff' }}>{val}</div>
                    <div style={{ fontSize: 12, color: '#e6edf3', fontWeight: 600, marginTop: 2 }}>{label}</div>
                    <div style={{ fontSize: 11, color: '#8b949e', marginTop: 2 }}>{note}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Key clinical pearls */}
            <div style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 14px', color: '#58a6ff', fontSize: 16 }}>⚡ Key Clinical Pearls — Hereditary CRC</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 10 }}>
                {(Array.isArray(overview.key_facts) ? overview.key_facts : Object.entries(overview.key_facts || {}).map(([k, v]) => `${k}: ${v}`)).map((fact, i) => {
                  const gene = Object.keys(GENE_COLORS).find(g => fact.startsWith(g));
                  const color = gene ? GENE_COLORS[gene] : '#444';
                  return (
                    <div key={i} style={{ background: '#0d1117', border: `1px solid ${color}44`, borderLeft: `3px solid ${color}`, borderRadius: 8, padding: '10px 14px' }}>
                      <div style={{ fontSize: 13, color: '#cdd9e5', lineHeight: 1.5 }}>{fact}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Gene cards */}
            <h3 style={{ color: '#58a6ff', fontSize: 16, marginBottom: 16 }}>Gene-Specific Reference Cards (8 Genes × 40 Patients)</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 16 }}>
              {(overview.genes || []).map(gene => {
                const info = GENE_INFO[gene];
                const color = GENE_COLORS[gene] || '#444';
                const cnt = overview.gene_counts?.[gene];
                return (
                  <div key={gene} style={{ background: '#161b22', border: `1px solid ${color}44`, borderRadius: 10, overflow: 'hidden' }}>
                    <div style={{ background: color, padding: '10px 14px' }}>
                      <div style={{ fontWeight: 700, fontSize: 15, color: '#fff' }}>{gene}</div>
                      <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.8)', marginTop: 2 }}>{info?.inh} · {info?.locus} · n={cnt}</div>
                    </div>
                    <div style={{ padding: '12px 14px' }}>
                      <div style={{ fontSize: 12, color: '#8b949e', marginBottom: 8 }}>{info?.full}</div>
                      <div style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5, marginBottom: 10 }}>{info?.note}</div>
                      <div style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#e6edf3', lineHeight: 1.5, borderLeft: `3px solid ${color}` }}>
                        {PATHOGNOMONIC[gene]}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {activeTab === 'Gene Table' && breakdown && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#161b22' }}>
                  {['Gene', 'Size', 'Locus', 'Inheritance', 'Syndrome', 'MSI-H %', 'Immunotherapy %', 'Prophylactic %', 'Aspirin %', 'Relapse %', 'Mean Age (yr)'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: '#8b949e', fontWeight: 600, fontSize: 12, borderBottom: '1px solid #30363d', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview?.genes || []).map((gene, i) => {
                  const bd = breakdown?.breakdown?.[gene] || {};
                  const info = GENE_INFO[gene];
                  const color = GENE_COLORS[gene] || '#444';
                  return (
                    <tr key={gene} style={{ background: i % 2 === 0 ? '#0d1117' : '#161b22', borderBottom: '1px solid #21262d' }}>
                      <td style={{ padding: '9px 12px', fontWeight: 700, color }}>{gene}</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{bd.gene_info?.size || info?.full?.match(/\d+aa/)?.[0] || '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#8b949e', fontSize: 12 }}>{info?.locus}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3', fontSize: 12 }}>{info?.inh}</td>
                      <td style={{ padding: '9px 12px', color: '#cdd9e5', fontSize: 11 }}>{bd.gene_info?.syndrome?.split('/')[0] || '—'}</td>
                      <td style={{ padding: '9px 12px', color: bd.msi_h_pct > 50 ? '#3fb950' : '#e6edf3', fontWeight: bd.msi_h_pct > 50 ? 700 : 400 }}>{bd.msi_h_pct != null ? `${bd.msi_h_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: bd.immunotherapy_pct > 30 ? '#58a6ff' : '#e6edf3', fontWeight: bd.immunotherapy_pct > 30 ? 700 : 400 }}>{bd.immunotherapy_pct != null ? `${bd.immunotherapy_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: bd.prophylactic_pct > 50 ? '#f0883e' : '#e6edf3', fontWeight: bd.prophylactic_pct > 50 ? 700 : 400 }}>{bd.prophylactic_pct != null ? `${bd.prophylactic_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3' }}>{bd.aspirin_pct != null ? `${bd.aspirin_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: bd.relapse_pct > 30 ? '#f85149' : '#e6edf3' }}>{bd.relapse_pct != null ? `${bd.relapse_pct}%` : '—'}</td>
                      <td style={{ padding: '9px 12px', color: '#e6edf3' }}>{bd.mean_age != null ? bd.mean_age : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {/* Protein reference cards */}
            <h3 style={{ color: '#58a6ff', fontSize: 15, margin: '28px 0 12px' }}>Gene Detail — Protein & Pathway Reference</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 12 }}>
              {Object.entries(GENE_INFO).map(([gene, info]) => (
                <div key={gene} style={{ background: '#161b22', border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: '12px 14px' }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 4 }}>{gene} — {info.locus} — {info.inh}</div>
                  <div style={{ fontSize: 12, color: '#cdd9e5', lineHeight: 1.5 }}>{info.note}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE BREAKDOWN TAB */}
        {activeTab === 'Gene Breakdown' && breakdown && (
          <div>
            <div style={{ display: 'grid', gap: 16 }}>
              {(overview?.genes || []).map(gene => {
                const bd = breakdown?.breakdown?.[gene] || {};
                const color = GENE_COLORS[gene] || '#444';
                const isExpanded = expandedGene === gene;
                return (
                  <div key={gene} style={{ background: '#161b22', border: `1px solid ${color}44`, borderRadius: 10, overflow: 'hidden' }}>
                    <button onClick={() => setExpandedGene(isExpanded ? null : gene)}
                      style={{ width: '100%', textAlign: 'left', padding: '14px 18px', background: color, border: 'none', cursor: 'pointer',
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <span style={{ fontWeight: 700, fontSize: 16, color: '#fff' }}>{gene}</span>
                        <span style={{ fontSize: 12, color: 'rgba(255,255,255,0.8)', marginLeft: 12 }}>{GENE_INFO[gene]?.inh} · {GENE_INFO[gene]?.locus} · n={bd.n}</span>
                      </div>
                      <span style={{ color: 'rgba(255,255,255,0.8)', fontSize: 18 }}>{isExpanded ? '▲' : '▼'}</span>
                    </button>
                    {isExpanded && (
                      <div style={{ padding: '16px 18px' }}>
                        {/* Stats row */}
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 8, marginBottom: 16 }}>
                          {[
                            { label: 'MSI-H', val: `${bd.msi_h_pct}%` },
                            { label: 'Immunotherapy', val: `${bd.immunotherapy_pct}%` },
                            { label: 'Prophylactic Proc', val: `${bd.prophylactic_pct}%` },
                            { label: 'Aspirin Rx', val: `${bd.aspirin_pct}%` },
                            { label: 'Relapse', val: `${bd.relapse_pct}%` },
                            { label: 'Mean Age', val: `${bd.mean_age} yr` },
                          ].map(({ label, val }) => (
                            <div key={label} style={{ background: '#0d1117', borderRadius: 6, padding: '8px 10px', textAlign: 'center' }}>
                              <div style={{ fontSize: 18, fontWeight: 700, color: '#58a6ff' }}>{val}</div>
                              <div style={{ fontSize: 11, color: '#8b949e' }}>{label}</div>
                            </div>
                          ))}
                        </div>
                        {/* Top tumour types */}
                        {bd.top_tumour_types?.length > 0 && (
                          <div style={{ marginBottom: 14 }}>
                            <div style={{ fontSize: 12, color: '#f0883e', fontWeight: 700, marginBottom: 6 }}>TOP TUMOUR TYPES</div>
                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              {bd.top_tumour_types.map((t, i) => (
                                <span key={i} style={{ background: '#0d1117', border: '1px solid #30363d', borderRadius: 5, padding: '3px 10px', fontSize: 12, color: '#cdd9e5' }}>
                                  {t.type} <b style={{ color: '#58a6ff' }}>({t.count})</b>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                        {/* Top variants */}
                        {bd.top_variants?.length > 0 && (
                          <div style={{ marginBottom: 14 }}>
                            <div style={{ fontSize: 12, color: '#f0883e', fontWeight: 700, marginBottom: 6 }}>KEY VARIANTS</div>
                            {bd.top_variants.map((v, i) => (
                              <div key={i} style={{ background: '#0d1117', borderRadius: 5, padding: '5px 10px', fontSize: 12, color: '#cdd9e5', marginBottom: 4 }}>
                                <b style={{ color: '#3fb950' }}>{v.variant}</b> — n={v.count}
                              </div>
                            ))}
                          </div>
                        )}
                        {/* Surveillance protocols */}
                        {bd.surveillance_protocols?.length > 0 && (
                          <div style={{ marginBottom: 14 }}>
                            <div style={{ fontSize: 12, color: '#58a6ff', fontWeight: 700, marginBottom: 6 }}>SURVEILLANCE PROTOCOLS</div>
                            {bd.surveillance_protocols.map((s, i) => (
                              <div key={i} style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 5, padding: '5px 10px', fontSize: 12, color: '#cdd9e5', marginBottom: 4 }}>
                                {s}
                              </div>
                            ))}
                          </div>
                        )}
                        {/* Treatment protocols */}
                        {bd.treatment_protocols?.length > 0 && (
                          <div>
                            <div style={{ fontSize: 12, color: '#3fb950', fontWeight: 700, marginBottom: 6 }}>TREATMENT PROTOCOLS</div>
                            {bd.treatment_protocols.map((t, i) => (
                              <div key={i} style={{ background: '#0d1117', border: '1px solid #21262d', borderRadius: 5, padding: '5px 10px', fontSize: 12, color: '#cdd9e5', marginBottom: 4 }}>
                                {t}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {activeTab === 'Definitions' && definitions && (
          <div>
            {/* Key clinical distinctions */}
            {definitions.key_clinical_distinctions?.length > 0 && (
              <div style={{ background: '#161b22', border: '1px solid #f0883e44', borderRadius: 10, padding: 20, marginBottom: 24 }}>
                <h3 style={{ margin: '0 0 14px', color: '#f0883e', fontSize: 15 }}>🚨 Key Clinical Distinctions — Must-Know DDx</h3>
                <div style={{ display: 'grid', gap: 8 }}>
                  {definitions.key_clinical_distinctions.map((item, i) => {
                    const gene = Object.keys(GENE_COLORS).find(g => item.startsWith(g));
                    const color = gene ? GENE_COLORS[gene] : '#444';
                    return (
                      <div key={i} style={{ background: '#0d1117', border: `1px solid ${color}44`, borderLeft: `3px solid ${color}`, borderRadius: 6, padding: '10px 14px', fontSize: 13, color: '#cdd9e5', lineHeight: 1.5 }}>
                        {item}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {/* Accordion definitions */}
            <h3 style={{ color: '#58a6ff', fontSize: 15, marginBottom: 12 }}>Reference Definitions</h3>
            <div style={{ display: 'grid', gap: 10 }}>
              {Object.entries(definitions.definitions || {}).map(([key, val], i) => (
                <div key={key} style={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 10, overflow: 'hidden' }}>
                  <button onClick={() => setExpandedDef(expandedDef === key ? null : key)}
                    style={{ width: '100%', textAlign: 'left', padding: '14px 18px', background: 'transparent', border: 'none', cursor: 'pointer',
                      color: '#58a6ff', fontSize: 14, fontWeight: 600, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                    <span style={{ color: '#8b949e', fontSize: 18, marginLeft: 12 }}>{expandedDef === key ? '▲' : '▼'}</span>
                  </button>
                  {expandedDef === key && (
                    <div style={{ padding: '0 18px 16px', fontSize: 13, color: '#cdd9e5', lineHeight: 1.75, borderTop: '1px solid #21262d' }}>
                      <div style={{ marginTop: 12 }}>{val}</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
