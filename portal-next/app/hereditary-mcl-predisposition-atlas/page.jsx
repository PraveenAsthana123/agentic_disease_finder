'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mcl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':    '#1a237e',  // deep navy       — RADIOSENSITIVITY ABSOLUTE; del11q 40-75%; ceralasertib; ibrutinib
  'TP53':   '#b71c1c',  // deep red        — LFS; blastoid 30-35%; AVOID RADIATION ABSOLUTELY; venetoclax
  'CDKN2A': '#4a148c',  // deep purple     — FAMM; del9p21 blastoid 30-60%; CDK4/6i INACTIVE MONARCL
  'RB1':    '#006064',  // dark teal       — Hereditary Rb; del13q14 25-30%; secondary MCL post-RT; CDK4/6i INACTIVE
  'NBN':    '#1b5e20',  // deep green      — NBS; NHL 60x; 657del5 Slavic; radiosensitivity biallelic
  'BRCA2':  '#880e4f',  // deep magenta    — HBOC; HRD 60%; cisplatin preferred; olaparib PARP
  'CHEK2':  '#e65100',  // deep orange     — 1100delC European 1:200; NHL 2x; NOT HRD; no PARPi
  'MAP2K1': '#bf360c',  // deep red-brown  — CFC; MEK1 GOF; cobimetinib; BRAF V600E co-test
};

const GENE_INFO = {
  'ATM':    { full: 'AT-heterozygous / del11q-40-75pct-MCL / RADIOSENSITIVITY-ABSOLUTE / ibrutinib-active / ceralasertib-ATRi-olaparib / RIC-HSCT-no-TBI', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF heterozygous' },
  'TP53':   { full: 'LFS / MCL-blastoid-30-35pct / complex-karyotype-60-80pct / AVOID-RADIATION-ABSOLUTELY / venetoclax-backbone / CAR-T-brexucabtagene / WB-MRI-Toronto', locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous' },
  'CDKN2A': { full: 'FAMM / del9p21-blastoid-30-60pct / CDK4/6i-INACTIVE-MONARCL / pancreatic-EUS-40yr / melanoma-28-67x / allo-SCT-CR1-blastoid', locus: '9p21.3',  size: '156 aa / 16 kDa',   inh: 'AD LOF heterozygous' },
  'RB1':    { full: 'Hereditary-Rb / del13q14-25-30pct / secondary-MCL-post-RT / CDK4/6i-INACTIVE / shortened-PFS-BTKi-monotherapy / BTKi-venetoclax-doublet', locus: '13q14.2', size: '928 aa / 110 kDa',  inh: 'AD LOF heterozygous' },
  'NBN':    { full: 'NBS / NHL-60x-HIGHEST / MCL-3-5x / 657del5-Slavic-1in150 / radiosensitivity-biallelic-ABSOLUTE / het-reduce-RT-20-30pct / cranial-RT-CI', locus: '8q21.3',  size: '754 aa / 95 kDa',   inh: 'AD het / AR biallelic NBS' },
  'BRCA2':  { full: 'HBOC / NHL-MCL-2-3x / HRD-60pct / cisplatin-preferred / olaparib-maintenance / FA-D1-biallelic-AVOID-ALKYLATING / annual-breast-MRI', locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF heterozygous' },
  'CHEK2':  { full: '1100delC-European-1in200 / NHL-2x / ATM-CHEK2-additive-8-10x / NOT-HRD-NO-PARPi / standard-BTKi-venetoclax / breast-25-30pct', locus: '22q12.1', size: '543 aa / 60 kDa',   inh: 'AD LOF heterozygous' },
  'MAP2K1': { full: 'CFC-Syndrome / MEK1-GOF / MCL-relapsed-MAPK-10-15pct / cobimetinib-MEKi / trametinib / BRAF-V600E-co-test / p-ERK-IHC-biomarker / ibrutinib-resistance', locus: '15q22.31', size: '393 aa / 44 kDa',   inh: 'AD GOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 130, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryMCLPredispositionAtlasPage() {
  const [tab, setTab]                  = useState('Overview');
  const [overview, setOverview]        = useState(null);
  const [breakdown, setBreakdown]      = useState(null);
  const [definitions, setDefinitions]  = useState(null);
  const [loading, setLoading]          = useState(false);
  const [error, setError]              = useState('');
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
    fontSize: 13, background: tab === t ? '#1a237e' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary MCL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: '28px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#4a148c 50%,#b71c1c 100%)',
        borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
          🧬 Hereditary Mantle Cell Lymphoma Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, opacity: 0.88 }}>
          Complete 8-Gene Reference · ATM · TP53 · CDKN2A · RB1 · NBN · BRCA2 · CHEK2 · MAP2K1
        </div>
        <div style={{ fontSize: 11, opacity: 0.70, marginTop: 4 }}>
          320-Patient Aggregate · 8×40 · Seeds 3598–3605 · MCL Germline Predisposition
        </div>
      </div>

      {/* Safety Rules Banner */}
      <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 10,
        padding: '12px 20px', marginBottom: 18, fontSize: 12 }}>
        <strong style={{ color: '#b71c1c' }}>⚠ MCL SAFETY RULES:</strong>
        <span style={{ color: '#555', marginLeft: 8 }}>
          ATM/NBN biallelic: NO RT EVER · TP53/LFS: AVOID RADIATION ABSOLUTELY · CDK4/6i INACTIVE in ALL MCL (MONARCL) ·
          CHEK2: NOT HRD — NO PARPi · BRCA2: cisplatin preferred over alkylating · MAP2K1 GOF: BRAF V600E co-test mandatory
        </span>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && overview && (
        <div>
          {/* KPI row 1 */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
            <StatCard label="Total Patients" value={overview.total_patients} sub="8 genes × 40" color="#1a237e" />
            <StatCard label="CR (1st-line)"  value={overview.cr_n} sub={`${overview.cr_pct}% complete response`} color="#1b5e20" />
            <StatCard label="Blastoid MCL"   value={overview.blastoid_n} sub={`${overview.blastoid_pct}% blastoid/pleomorphic`} color="#b71c1c" />
            <StatCard label="Ki-67 ≥70%"     value={overview.ki67_high_n} sub={`${overview.ki67_high_pct}% high proliferation`} color="#880e4f" />
            <StatCard label="3-yr Alive"      value={overview.alive_3yr_n} sub={`${overview.alive_3yr_pct}% 3-year OS`} color="#1b5e20" />
            <StatCard label="MCL-IPI High"    value={overview.mcl_ipi_high_n} sub={`${overview.mcl_ipi_high_pct}% high-risk score`} color="#4a148c" />
          </div>
          {/* KPI row 2 — chromosomal */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="del11q (ATM)"  value={overview.del11q_n}  sub={`${overview.del11q_pct}% — ibrutinib-active; radiosens`} color="#1a237e" />
            <StatCard label="del17p (TP53)" value={overview.del17p_n}  sub={`${overview.del17p_pct}% — venetoclax backbone`} color="#b71c1c" />
            <StatCard label="del9p21"       value={overview.del9p21_n} sub={`${overview.del9p21_pct}% — CDK4/6i INACTIVE`}  color="#4a148c" />
            <StatCard label="del13q14 (RB1)"value={overview.del13q14_n}sub={`${overview.del13q14_pct}% — short PFS BTKi mono`} color="#006064" />
            <StatCard label="Complex Karyotype" value={overview.complex_karyotype_n} sub={`${overview.complex_karyotype_pct}% ≥3 abnormalities`} color="#880e4f" />
            <StatCard label="IGHV Unmutated"    value={overview.ighv_unmutated_n}    sub={`${overview.ighv_unmutated_pct}% poor prognosis`}   color="#bf360c" />
          </div>

          {/* Per-gene summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
            {Object.entries(overview.gene_counts || {}).map(([gene, d]) => {
              const color = GENE_COLORS[gene] || '#555';
              const info  = GENE_INFO[gene] || {};
              const expanded = expandedGene === gene;
              return (
                <div key={gene} style={{ background: '#fff', border: `2px solid ${color}33`,
                  borderRadius: 12, padding: '16px 18px', cursor: 'pointer' }}
                  onClick={() => setExpandedGene(expanded ? null : gene)}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 17, fontWeight: 800, color }}>{gene}</span>
                    <Badge text={info.locus || ''} color={color} />
                    <span style={{ marginLeft: 'auto', fontSize: 11, color: '#aaa' }}>{info.size}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>
                    {info.inh} · n={d.n}
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginBottom: 8 }}>
                    {[
                      ['Blastoid', d.blastoid_n],
                      ['Ki-67 ≥70%', d.ki67_high_n],
                      ['del11q', d.del11q_n],
                      ['del17p', d.del17p_n],
                      ['Complex Karyotype', d.complex_n],
                      ['MCL-IPI High', d.ipi_high_n],
                      ['CR (1st-line)', d.cr_n],
                      ['3yr Alive', d.alive_3yr_n],
                    ].map(([k, v]) => (
                      <div key={k} style={{ background: '#f8f9fa', borderRadius: 5,
                        padding: '4px 7px', fontSize: 10 }}>
                        <span style={{ color: '#aaa' }}>{k}: </span>
                        <span style={{ fontWeight: 700, color }}>{v}</span>
                      </div>
                    ))}
                  </div>
                  <div style={{ fontSize: 11, background: `${color}0d`, borderRadius: 6,
                    padding: '5px 8px', color }}>
                    {d.mcl_risk?.split(';')[0]}
                  </div>
                  {expanded && (
                    <div style={{ marginTop: 10, fontSize: 11, color: '#555' }}>
                      <strong>Full profile:</strong> {info.full}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fff',
            borderRadius: 10, overflow: 'hidden', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene','Locus','Inh','n','Avg Age','CR','Blastoid','Ki67≥70%','del11q','del17p','del9p21','del13q14','Complex','IPI High','3yr Alive','Ibrutinib Resp','Venetoclax Resp'].map(h => (
                  <th key={h} style={{ padding: '9px 10px', textAlign: 'left', fontWeight: 700, fontSize: 10, whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown.gene_data || {}).map(([gene, d], idx) => {
                const color = GENE_COLORS[gene] || '#555';
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#fff' : '#fafafa',
                    borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 800, color }}>{gene}</td>
                    <td style={{ padding: '8px 10px' }}>{d.locus}</td>
                    <td style={{ padding: '8px 10px', fontSize: 10, maxWidth: 100, whiteSpace: 'nowrap', overflow: 'hidden' }}>{d.inheritance?.split(' ').slice(0,2).join(' ')}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700 }}>{d.n}</td>
                    <td style={{ padding: '8px 10px', color }}>{d.avg_age_dx}yr</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: '#1b5e20' }}>{d.cr_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.blastoid_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ki67_hi_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del11q_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del17p_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del9p21_n ?? '—'}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del13q14_n ?? '—'}</td>
                    <td style={{ padding: '8px 10px' }}>{d.complex_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ipi_hi_n}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: '#1b5e20' }}>{d.alive_3yr_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ibrutinib_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.venetoclax_n}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Clinical Atlas Tab ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 16 }}>
          {Object.entries(breakdown.gene_data || {}).map(([gene, d]) => {
            const color = GENE_COLORS[gene] || '#555';
            return (
              <div key={gene} style={{ background: '#fff', border: `2px solid ${color}33`,
                borderRadius: 12, padding: '16px 18px' }}>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                  <span style={{ fontSize: 18, fontWeight: 800, color }}>{gene}</span>
                  <span style={{ marginLeft: 8, fontSize: 11, color: '#888' }}>{d.locus}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 11, background: `${color}18`,
                    color, borderRadius: 4, padding: '2px 8px' }}>n={d.n}</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5, marginBottom: 10 }}>
                  {[
                    ['Avg Age Dx', `${d.avg_age_dx} yr`],
                    ['CR (1st-line)', d.cr_n],
                    ['Blastoid', d.blastoid_n],
                    ['Ki-67 ≥70%', d.ki67_hi_n],
                    ['del11q', d.del11q_n],
                    ['del17p', d.del17p_n],
                    ['Complex Karyotype', d.complex_n],
                    ['MCL-IPI High', d.ipi_hi_n],
                    ['Stage 4', d.stage4_n],
                    ['3yr Alive', d.alive_3yr_n],
                    ['Ibrutinib Resp', d.ibrutinib_n],
                    ['Venetoclax Resp', d.venetoclax_n],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: '#f8f9fa', borderRadius: 6,
                      padding: '5px 8px', fontSize: 11 }}>
                      <div style={{ color: '#aaa', fontSize: 10 }}>{k}</div>
                      <div style={{ fontWeight: 700, color }}>{v}</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11, background: '#ffebee', borderRadius: 6,
                  padding: '6px 10px', color: '#b71c1c', marginBottom: 6 }}>
                  <strong>KEY AVOID:</strong> {d.key_avoid?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e8f5e9', borderRadius: 6,
                  padding: '6px 10px', color: '#1b5e20', marginBottom: 6 }}>
                  <strong>KEY RULE:</strong> {d.key_rule?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e3f2fd', borderRadius: 6,
                  padding: '6px 10px', color: '#1a237e' }}>
                  <strong>Rx:</strong> {d.targeted_rx?.split(';')[0]}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          {(definitions.key_clinical_distinctions || []).length > 0 && (
            <div style={{ background: '#fff', borderRadius: 12, padding: '18px 22px', marginBottom: 20 }}>
              <div style={{ fontSize: 15, fontWeight: 800, color: '#1a237e', marginBottom: 12 }}>
                Key Clinical Distinctions — Hereditary MCL Predisposition
              </div>
              {definitions.key_clinical_distinctions.map((d, i) => (
                <div key={i} style={{ background: '#e3f2fd', borderLeft: `4px solid ${Object.values(GENE_COLORS)[i % 8]}`,
                  borderRadius: 6, padding: '10px 14px', marginBottom: 8, fontSize: 12, color: '#333' }}>
                  {d}
                </div>
              ))}
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 14 }}>
            {(definitions.definitions || []).map((def, i) => {
              const gene = def.term?.split(' — ')[0];
              const color = GENE_COLORS[gene] || '#555';
              const expanded = expandedDef === i;
              return (
                <div key={i} style={{ background: '#fff', border: `2px solid ${color}22`,
                  borderRadius: 10, padding: '14px 18px', cursor: 'pointer' }}
                  onClick={() => setExpandedDef(expanded ? null : i)}>
                  <div style={{ fontWeight: 700, color, marginBottom: 6, fontSize: 13 }}>
                    {def.term}
                  </div>
                  {expanded && (
                    <pre style={{ fontSize: 11, color: '#555', whiteSpace: 'pre-wrap',
                      background: '#f8f9fa', borderRadius: 6, padding: '10px', marginTop: 8 }}>
                      {def.definition}
                    </pre>
                  )}
                  {!expanded && (
                    <div style={{ fontSize: 11, color: '#aaa' }}>Click to expand definition…</div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
