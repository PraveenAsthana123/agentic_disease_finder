'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-t-all-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':    '#4a148c',  // deep purple      — A-T biallelic T-ALL 75x HIGHEST; RADIOSENSITIVITY ABSOLUTE; IgA deficiency PATHOGNOMONIC
  'TP53':   '#e65100',  // deep orange      — LFS; T-ALL elevated; AVOID RADIATION ABSOLUTELY; WBMRI Toronto
  'CDKN2A': '#006064',  // dark cyan        — del9p21 T-ALL; p16/p14ARF dual; CDK4/6i palbociclib INACTIVE; NOTCH1 co-deletion
  'PTEN':   '#1a237e',  // deep navy        — PHTS/Cowden; T-ALL 5x mTOR; macrocephaly PATHOGNOMONIC; everolimus active
  'NF1':    '#1b5e20',  // deep green       — NF1 syndrome; T-ALL/T-LBL 5-10x Ras; café-au-lait PATHOGNOMONIC; selumetinib
  'NBN':    '#880e4f',  // deep magenta     — NBS; 657del5 Slavic; ALL 60x T-ALL biallelic; cranial RT ABSOLUTELY CI
  'RB1':    '#bf360c',  // deep brown-red   — Hereditary Rb; bilateral Rb PATHOGNOMONIC; secondary T-ALL post-RT; CDK4/6i INACTIVE
  'ETV6':   '#b71c1c',  // deep red         — THC5; T-ALL also elevated; thrombocytopenia co-phenotype; avoid NSAIDs
};

const GENE_INFO = {
  'ATM':    { full: 'A-T-Biallelic-T-ALL-75x-HIGHEST / RADIOSENSITIVITY-ABSOLUTE-Biallelic / IgA-Deficiency-PATHOGNOMONIC / AFP-Elevated-Screening / TBI-CI-Biallelic', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD/AR LOF' },
  'TP53':   { full: 'LFS / T-ALL-Elevated / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto / Cranial-RT-ABSOLUTELY-CONTRAINDICATED-LFS',                                   locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'CDKN2A': { full: 'FAMM / del9p21-T-ALL-5-10x / NOTCH1-Somatic-Co-Deletion / p16INK4a-p14ARF-Dual / CDK4-6i-Palbociclib-INACTIVE-T-ALL',                             locus: '9p21.3',  size: '156 aa / 17 kDa',   inh: 'AD LOF' },
  'PTEN':   { full: 'PHTS-Cowden / T-ALL-5x-PI3K-mTOR / Macrocephaly-PATHOGNOMONIC / Lhermitte-Duclos-PATHOGNOMONIC / Everolimus-mTOR-Active',                         locus: '10q23.31',size: '403 aa / 47 kDa',   inh: 'AD LOF' },
  'NF1':    { full: 'NF1-Syndrome / T-ALL-T-LBL-5-10x-Ras-MAPK / Cafe-au-Lait-6-PATHOGNOMONIC / MPNST-8-13pct / Selumetinib-FDA2020 / Mediastinal-T-LBL',             locus: '17q11.2', size: '2839 aa / 327 kDa', inh: 'AD LOF' },
  'NBN':    { full: 'NBS / 657del5-Slavic-1:150 / ALL-60x-Biallelic-T-ALL-PREDOMINANT / Cranial-RT-ABSOLUTELY-CI / RADIOSENSITIVITY-ABSOLUTE-Biallelic',                locus: '8q21.3',  size: '754 aa / 85 kDa',   inh: 'AD/AR LOF' },
  'RB1':    { full: 'Hereditary-Retinoblastoma / Bilateral-Rb-PATHOGNOMONIC / Secondary-T-ALL-Post-RT-5x / CDK4-6i-INACTIVE-RB1-Null / WBMRI-Secondary-Malignancy',     locus: '13q14.2', size: '928 aa / 110 kDa',  inh: 'AD LOF' },
  'ETV6':   { full: 'THC5 / T-ALL-Also-Elevated / Thrombocytopenia-50-120-Lifelong / Avoid-NSAIDs-Aspirin / ETV6-RUNX1-Somatic-DISTINCT-From-Germline',                locus: '12p13.2', size: '1236 aa / 57 kDa',  inh: 'AD LOF' },
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

export default function HereditaryTALLPredispositionAtlasPage() {
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
    fontSize: 13, background: tab === t ? '#4a148c' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary T-ALL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ padding: '28px 32px', background: '#f8f9fa', minHeight: '100vh', fontFamily: 'system-ui,sans-serif' }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#4a148c', marginBottom: 6 }}>
          🧬 Hereditary T-ALL Predisposition Atlas
        </h1>
        <p style={{ fontSize: 13, color: '#555', maxWidth: 900 }}>
          Complete 8-Gene Reference — ATM · TP53 · CDKN2A · PTEN · NF1 · NBN · RB1 · ETV6
          · 320-patient aggregate cohort (8×40, seeds 3558–3565)
          · Hereditary T-cell Acute Lymphoblastic Leukaemia / Lymphoblastic Lymphoma predisposition,
          radiation safety rules, nelarabine T-lineage therapy, and targeted treatment guidance
        </p>
      </div>

      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ─── OVERVIEW ─── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {card('Total Patients', overview.total_patients, `${overview.genes_n} genes × 40`, '#4a148c')}
            {card('T-ALL Events', `${overview.t_all_total_n} (${overview.t_all_total_pct}%)`, 'confirmed T-ALL', '#e65100')}
            {card('Mean Age at Dx', `${overview.mean_age_onset}yr`, 'weighted average', '#1a237e')}
            {card('del9p21 (CDKN2A)', `${overview.del9p21_n} (${overview.del9p21_pct}%)`, 'somatic del', '#006064')}
            {card('T-LBL', `${overview.t_lbl_n} (${overview.t_lbl_pct}%)`, '<25% BM blasts', '#880e4f')}
            {card('Nelarabine-Eligible', `${overview.nelarabine_eligible_n} (${overview.nelarabine_eligible_pct}%)`, 'T-lineage specific', '#1b5e20')}
            {card('Craniospinal RT', `${overview.craniospinal_rt_n} (${overview.craniospinal_rt_pct}%)`, 'received CRT', '#bf360c')}
            {card('HRD Positive', `${overview.hrd_positive_n} (${overview.hrd_positive_pct}%)`, 'ATM/NBN/PTEN', '#b71c1c')}
          </div>

          <div style={{ background: '#f3e5f5', border: '2px solid #4a148c', borderRadius: 10, padding: '14px 18px', marginBottom: 20 }}>
            <strong style={{ color: '#4a148c', fontSize: 13 }}>⚠ CRITICAL SAFETY RULES — HEREDITARY T-ALL</strong>
            <ul style={{ fontSize: 12, color: '#555', marginTop: 8, paddingLeft: 18 }}>
              <li><strong>CRANIAL/CRANIOSPINAL RT ABSOLUTELY CONTRAINDICATED:</strong> TP53 germline (LFS — catastrophic second brain tumours), ATM biallelic (A-T — RADIOSENSITIVITY ABSOLUTE), NBN biallelic (NBS — RADIOSENSITIVITY ABSOLUTE + profound neurotoxicity)</li>
              <li><strong>TBI CONDITIONING CI:</strong> ATM biallelic (A-T), NBN biallelic (NBS) — use busulfan/fludarabine-based non-TBI conditioning; RB1 germline — TBI relative CI (secondary solid tumour risk); TP53 LFS — TBI absolutely CI</li>
              <li><strong>CDK4/6 INHIBITOR (palbociclib) INACTIVE in CDKN2A-null T-ALL:</strong> p16INK4a absent → CDK4/6 unregulated at protein level even with drug; venetoclax ACTIVE CDKN2A-null T-ALL</li>
              <li><strong>NELARABINE T-LINEAGE SPECIFIC:</strong> FDA-approved T-ALL/T-LBL; do NOT use in B-ALL; avoid nelarabine + intrathecal cytarabine combination (additive neurotoxicity); prior cranial RT + nelarabine = high neurotoxicity risk</li>
              <li><strong>ALKYLATING AGENTS CI:</strong> NBN biallelic (NBS — alkylator hypersensitivity); RB1 germline — avoid high-dose alkylating at T-ALL site (secondary osteosarcoma risk post-alkylator-RT combination)</li>
              <li><strong>mTOR (everolimus) ACTIVE in PTEN-null T-ALL:</strong> PTEN LOF → PI3K/mTOR hyperactivation → everolimus (FDA-approved Cowden) active; standard chemotherapy backbone + mTOR inhibitor emerging in PTEN-null T-ALL</li>
            </ul>
          </div>

          <div style={{ background: '#fff', borderRadius: 10, border: '1px solid #e0e0e0', padding: 20, marginBottom: 20 }}>
            <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 14, color: '#333' }}>Gene Summary — 8 Hereditary T-ALL Predisposition Genes</h3>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#f5f5f5' }}>
                  {['Gene', 'Locus', 'Inheritance', 'T-ALL Risk', 'N', 'T-ALL %', 'Age (yr)'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', borderBottom: '2px solid #e0e0e0', color: '#333' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.gene_summaries || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>
                      {g.gene}
                    </td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{GENE_INFO[g.gene]?.locus || g.locus || '—'}</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{GENE_INFO[g.gene]?.inh || '—'}</td>
                    <td style={{ padding: '8px 10px', color: '#666', fontSize: 11 }}>{(GENE_INFO[g.gene]?.full || '').substring(0, 60)}…</td>
                    <td style={{ padding: '8px 10px', color: '#333' }}>{g.n_patients}</td>
                    <td style={{ padding: '8px 10px', color: GENE_COLORS[g.gene], fontWeight: 700 }}>{g.pct_t_all}%</td>
                    <td style={{ padding: '8px 10px', color: '#555' }}>{g.mean_onset_years}</td>
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
                <Badge text={`T-ALL ${g.all_pct}%`} color={GENE_COLORS[g.gene] || '#333'} />
                <Badge text={`Age ${g.mean_age_onset}yr`} color="#1a237e" />
                <Badge text={`del9p21 ${g.del9p21_n}`} color="#006064" />
                <Badge text={`T-LBL ${g.t_lbl_n}`} color="#880e4f" />
                <Badge text={`CRT ${g.craniospinal_rt_n}`} color="#bf360c" />
                <Badge text={`HRD ${g.hrd_positive_n}`} color="#4a148c" />
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
            Clinical Atlas — Hereditary T-ALL Predisposition
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
                <div style={{ fontSize: 11, color: '#888', marginBottom: 8 }}>{g.locus} · {String(g.protein_size || '').substring(0, 80)}…</div>
                <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>
                  <strong>Syndrome:</strong> {g.syndrome}
                </div>
                <div style={{ fontSize: 11, color: '#333', marginBottom: 6 }}>
                  <strong>T-ALL Risk:</strong> {String(g.cancer_risk || '').substring(0, 150)}…
                </div>
                <div style={{ fontSize: 11, color: '#666' }}>
                  <strong>Surveillance:</strong> {String(g.surveillance_key || '').substring(0, 120)}…
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
          <p style={{ fontSize: 12, color: '#888', marginBottom: 16 }}>Seed range: {definitions.seed_range} · 8 hereditary T-ALL predisposition genes</p>

          <div style={{ background: '#f3e5f5', border: '1px solid #4a148c', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <strong style={{ color: '#4a148c', fontSize: 13 }}>Key Clinical Distinctions</strong>
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
