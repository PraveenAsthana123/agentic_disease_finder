'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-b-all-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ETV6':   '#b71c1c',  // deep red         — THC5; B-ALL 10-15x MOST COMMON hereditary ALL gene; ETV6::RUNX1
  'PAX5':   '#004d40',  // deep teal        — autosomal dominant B-ALL; P80R; 40-80% lifetime risk MOST SPECIFIC
  'IKZF1':  '#1a237e',  // deep navy        — Ikaros; Ph-like ALL 60%; del7p12.2 somatic 30% poor prognosis
  'TP53':   '#e65100',  // deep orange      — LFS; B-ALL 15x; AVOID CRANIAL RT ABSOLUTELY; WBMRI Toronto
  'ATM':    '#4a148c',  // deep purple      — A-T biallelic T-ALL 75x; B-ALL 2-3x monoallelic; RADIOSENSITIVITY ABSOLUTE
  'CDKN2A': '#006064',  // dark cyan        — del9p21 ALL 5-10x; p16/p14ARF dual; CDK4/6i palbociclib INACTIVE
  'NBN':    '#880e4f',  // deep magenta     — NBS; 657del5 Slavic; ALL 60x biallelic; cranial RT ABSOLUTELY CI
  'BRCA2':  '#1b5e20',  // deep green       — FA-D1 biallelic; ALL/AML HIGHEST; AVOID ALKYLATING ABSOLUTELY
};

const GENE_INFO = {
  'ETV6':   { full: 'THC5 / B-ALL-10-15x-MOST-COMMON-Hereditary-ALL / ETV6::RUNX1-somatic / PAX5-fusion / Early-Childhood-Onset',            locus: '12p13.2', size: '1236 aa / 57 kDa',  inh: 'AD LOF' },
  'PAX5':   { full: 'Autosomal-Dominant-B-ALL / P80R-Founder-40-80pct-MOST-SPECIFIC / B-Cell-Differentiation-Block / Familial-B-ALL',         locus: '9p13.2',  size: '391 aa / 48 kDa',   inh: 'AD LOF' },
  'IKZF1':  { full: 'Ikaros-Haploinsufficiency / Ph-like-ALL-60pct / del7p12.2-somatic-30pct-Poor-Prognosis / Emerging-Germline-Predisposition', locus: '7p12.2',  size: '519 aa / 57 kDa',   inh: 'AD haploinsufficiency' },
  'TP53':   { full: 'LFS / B-ALL-15x / AVOID-CRANIAL-RT-ABSOLUTELY / WBMRI-Toronto / Cranial-RT-CI-Second-Brain-Tumour',                       locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'ATM':    { full: 'A-T-Biallelic-T-ALL-75x-HIGHEST / B-ALL-2-3x-Monoallelic / IgA-Deficiency-PATHOGNOMONIC / RADIOSENSITIVITY-ABSOLUTE',    locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD/AR LOF' },
  'CDKN2A': { full: 'FAMM / del9p21-ALL-5-10x / p16INK4a-p14ARF-Dual / CDK4-6i-Palbociclib-INACTIVE / Melanoma-28-67x / Pancreatic-17-39x',  locus: '9p21.3',  size: '156 aa / 17 kDa',   inh: 'AD LOF' },
  'NBN':    { full: 'NBS / 657del5-Slavic-1:150 / ALL-60x-Biallelic-HIGHEST / Cranial-RT-ABSOLUTELY-CI / RADIOSENSITIVITY-ABSOLUTE-Biallelic', locus: '8q21.3',  size: '754 aa / 85 kDa',   inh: 'AD/AR LOF' },
  'BRCA2':  { full: 'HBOC / FA-D1-Biallelic-ALL-AML-HIGHEST / AVOID-ALKYLATING-ABSOLUTELY / PARP-Olaparib-Sensitivity / Youngest-Onset',      locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function card(label, value, sub, color) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 140, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryBALLPredispositionAtlasPage() {
  const [tab, setTab]                 = useState('Overview');
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError('');
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '8px 18px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, background: tab === t ? '#b71c1c' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary B-ALL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ padding: '28px 32px', background: '#f8f9fa', minHeight: '100vh', fontFamily: 'system-ui,sans-serif' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#b71c1c', marginBottom: 6 }}>
          🧬 Hereditary B-ALL Predisposition Atlas
        </h1>
        <p style={{ fontSize: 13, color: '#555', maxWidth: 900 }}>
          Complete 8-Gene Reference — ETV6 · PAX5 · IKZF1 · TP53 · ATM · CDKN2A · NBN · BRCA2
          · 320-patient aggregate cohort (8×40, seeds 3550–3557)
          · Hereditary Acute Lymphoblastic Leukaemia (B-ALL) predisposition, cranial-RT safety rules,
          alkylating agent contraindications, and targeted therapy guidance
        </p>
      </div>

      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ─── OVERVIEW ─── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {card('Total Patients', overview.total_patients, `${overview.genes_n} genes × 40`, '#b71c1c')}
            {card('B-ALL Events', `${overview.all_total_n} (${overview.all_total_pct}%)`, 'confirmed ALL', '#004d40')}
            {card('Mean Age at Dx', `${overview.mean_age_onset}yr`, 'weighted average', '#1a237e')}
            {card('del9p21 (CDKN2A)', `${overview.del9p21_n} (${overview.del9p21_pct}%)`, 'somatic del', '#e65100')}
            {card('Ph-like ALL', `${overview.ph_like_n} (${overview.ph_like_pct}%)`, 'IKZF1-driven', '#4a148c')}
            {card('t(12;21) ETV6::RUNX1', `${overview.t_12_21_n} (${overview.t_12_21_pct}%)`, 'somatic transloc', '#006064')}
            {card('Craniospinal RT', `${overview.craniospinal_rt_n} (${overview.craniospinal_rt_pct}%)`, 'received CRT', '#880e4f')}
            {card('HRD Positive', `${overview.hrd_positive_n} (${overview.hrd_positive_pct}%)`, 'BRCA2/NBN/ATM', '#1b5e20')}
          </div>

          <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 10, padding: '14px 18px', marginBottom: 20 }}>
            <strong style={{ color: '#e65100', fontSize: 13 }}>⚠ CRITICAL SAFETY RULES — HEREDITARY B-ALL</strong>
            <ul style={{ fontSize: 12, color: '#555', marginTop: 8, paddingLeft: 18 }}>
              <li><strong>CRANIAL RT ABSOLUTELY CONTRAINDICATED:</strong> TP53 germline (LFS — catastrophic 2nd brain tumours), ATM biallelic (A-T — RADIOSENSITIVITY ABSOLUTE), NBN biallelic (NBS — RADIOSENSITIVITY ABSOLUTE + neurotoxicity)</li>
              <li><strong>ALKYLATING AGENTS ABSOLUTELY CONTRAINDICATED:</strong> BRCA2 biallelic (FA-D1 — DNA crosslinks catastrophic), NBN biallelic (NBS — alkylator hypersensitivity); reduce doses monoallelic BRCA2/NBN</li>
              <li><strong>CDK4/6 INHIBITOR (palbociclib) INACTIVE in CDKN2A-null ALL:</strong> p16INK4a absent → CDK4/6 axis unregulated at protein level even with drug; venetoclax ACTIVE</li>
              <li><strong>TBI CONDITIONING CI:</strong> ATM biallelic (A-T), NBN biallelic (NBS) — use busulfan-based non-TBI conditioning; TP53 (LFS) — TBI relative CI (WBMRI monitoring MANDATORY)</li>
              <li><strong>PARP inhibitor sensitivity:</strong> BRCA2 (FA-D1/HRD), ATM (HRD), NBN (MRN complex HRD) — olaparib/niraparib ALL investigational but HRD-guided</li>
            </ul>
          </div>

          <div style={{ background: '#fff', borderRadius: 10, border: '1px solid #e0e0e0', padding: 20, marginBottom: 20 }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 14, color: '#333' }}>Gene Summary — 8 Hereditary B-ALL Predisposition Genes</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  {['Gene', 'Locus', 'Inheritance', 'B-ALL Risk', 'N', 'B-ALL %', 'Age (yr)'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e0e0e0', color: '#333' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.gene_summary || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>
                      {g.gene}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{GENE_INFO[g.gene]?.locus || g.locus || '—'}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{GENE_INFO[g.gene]?.inh || '—'}</td>
                    <td style={{ padding: '8px 10px', color: '#666', fontSize: 11 }}>{(GENE_INFO[g.gene]?.full || '').substring(0, 60)}…</td>
                    <td style={{ padding: '8px 10px', color: '#333' }}>{g.n}</td>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[g.gene], fontWeight: 700 }}>{g.all_pct}%</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{g.mean_age_onset}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ─── GENE TABLE ─── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: '#333' }}>Per-Gene Breakdown — 8 Genes × 40 Patients</h3>
          {(breakdown.breakdown || []).map((g) => (
            <div key={g.gene} style={{
              background: '#fff', border: `2px solid ${GENE_COLORS[g.gene] || '#ccc'}33`,
              borderRadius: 10, padding: 18, marginBottom: 14,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10, cursor: 'pointer' }}
                   onClick={() => setExpandedGene(expandedGene === g.gene ? null : g.gene)}>
                <span style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</span>
                <span style={{ marginLeft: 10, fontSize: 12, color: '#666' }}>{GENE_INFO[g.gene]?.locus} · {GENE_INFO[g.gene]?.inh}</span>
                <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>{GENE_INFO[g.gene]?.size}</span>
                <span style={{ marginLeft: 'auto', fontSize: 12, color: '#aaa' }}>{expandedGene === g.gene ? '▲' : '▼'}</span>
              </div>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                <Badge text={`n=${g.n}`} color="#555" />
                <Badge text={`B-ALL ${g.all_pct}%`} color={GENE_COLORS[g.gene] || '#333'} />
                <Badge text={`Age ${g.mean_age_onset}yr`} color="#1a237e" />
                <Badge text={`del9p21 ${g.del9p21_n}`} color="#e65100" />
                <Badge text={`Ph-like ${g.ph_like_n}`} color="#4a148c" />
                <Badge text={`CRT ${g.craniospinal_rt_n}`} color="#880e4f" />
                <Badge text={`HRD ${g.hrd_positive_n}`} color="#006064" />
                <Badge text={`seed ${g.seed}`} color="#999" />
              </div>
              {expandedGene === g.gene && (
                <div style={{ marginTop: 12, borderTop: '1px solid #f0f0f0', paddingTop: 12 }}>
                  <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}>
                    <strong>Full Gene Descriptor:</strong> {GENE_INFO[g.gene]?.full}
                  </div>
                  <div style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>
                    <strong>Pathognomonic:</strong> {g.pathognomonic}
                  </div>
                  <div style={{ fontSize: 12, color: '#c62828', marginBottom: 6 }}>
                    <strong>Surveillance Key:</strong> {g.surveillance_key}
                  </div>
                  <div style={{ fontSize: 12, color: '#1b5e20' }}>
                    <strong>Key Rule:</strong> {g.key_rule}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ─── CLINICAL ATLAS ─── */}
      {tab === 'Clinical Atlas' && overview && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: '#333' }}>
            Clinical Atlas — Hereditary B-ALL Predisposition
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: 16 }}>
            {(overview.genes_detail || []).map((g) => (
              <div key={g.gene} style={{
                background: '#fff', border: `2px solid ${GENE_COLORS[g.gene] || '#ccc'}44`,
                borderRadius: 10, padding: 18,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                  <span style={{ fontSize: 15, fontWeight: 800, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</span>
                  <Badge text={g.inheritance} color={GENE_COLORS[g.gene] || '#555'} />
                </div>
                <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>{g.locus} · {g.protein_size?.substring(0, 80)}…</div>
                <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>
                  <strong>Syndrome:</strong> {g.syndrome}
                </div>
                <div style={{ fontSize: 11, color: '#333', marginBottom: 6 }}>
                  <strong>B-ALL Risk:</strong> {g.cancer_risk?.substring(0, 150)}…
                </div>
                <div style={{ fontSize: 11, color: '#666' }}>
                  <strong>Surveillance:</strong> {g.surveillance_key?.substring(0, 120)}…
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─── DEFINITIONS ─── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8, color: '#333' }}>Clinical Definitions & Key Distinctions</h3>
          <p style={{ fontSize: 12, color: '#888', marginBottom: 16 }}>Seed range: {definitions.seed_range} · 8 hereditary B-ALL predisposition genes</p>

          <div style={{ background: '#e8f5e9', border: '1px solid #43a047', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <strong style={{ color: '#1b5e20', fontSize: 13 }}>Key Clinical Distinctions</strong>
            <ol style={{ fontSize: 12, color: '#333', marginTop: 8, paddingLeft: 18 }}>
              {(definitions.key_clinical_distinctions || []).map((d, i) => (
                <li key={i} style={{ marginBottom: 6 }}>{d}</li>
              ))}
            </ol>
          </div>

          {(definitions.definitions || []).map((d) => {
            const gene = d.term?.split(' — ')[0];
            return (
              <div key={d.term} style={{
                background: '#fff', border: `1px solid ${GENE_COLORS[gene] || '#ccc'}44`,
                borderRadius: 8, marginBottom: 10, overflow: 'hidden',
              }}>
                <div
                  style={{
                    background: `${GENE_COLORS[gene] || '#ccc'}11`, padding: '10px 16px',
                    cursor: 'pointer', display: 'flex', justifyContent: 'space-between',
                  }}
                  onClick={() => setExpandedDef(expandedDef === d.term ? null : d.term)}
                >
                  <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#333', fontSize: 13 }}>{d.term}</span>
                  <span style={{ color: '#aaa', fontSize: 12 }}>{expandedDef === d.term ? '▲' : '▼'}</span>
                </div>
                {expandedDef === d.term && (
                  <pre style={{
                    padding: 16, fontSize: 11, color: '#333', whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word', margin: 0, background: '#fafafa',
                  }}>{d.definition}</pre>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
