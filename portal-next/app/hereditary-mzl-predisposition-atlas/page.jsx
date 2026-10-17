'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-mzl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MYD88':     '#1a237e',  // deep navy        — L265P HP-eradication-resistant; NF-κB constitutive; ibrutinib
  'TNFRSF13B': '#b71c1c',  // deep red         — CVID2; NHL 10-12x HIGHEST; IVIG mandatory IgG>7g/L
  'ATM':       '#4a148c',  // deep purple      — RADIOSENSITIVITY ABSOLUTE; del11q; ceralasertib ATRi
  'TP53':      '#880e4f',  // deep magenta     — LFS; AVOID RADIATION ABSOLUTELY; venetoclax; WB-MRI Toronto
  'CDKN2A':    '#006064',  // dark teal        — FAMM; gastric MALT 5-10x; H.pylori synergy; CDK4/6i INACTIVE
  'BRCA2':     '#1b5e20',  // deep green       — HBOC; HRD 60%; cisplatin preferred; olaparib PARP
  'CARD11':    '#e65100',  // deep orange      — BENTA GOF; NF-κB; venetoclax; MALT1 inhibitor trial
  'PIK3CD':    '#bf360c',  // deep red-brown   — APDS1; EBV+ MZL/MALT; leniolisib FDA2023; sirolimus
};

const GENE_INFO = {
  'MYD88':     { full: 'L265P-HP-eradication-resistant-MALT / NF-κB-constitutive / rituximab-mandatory-L265P+ / ibrutinib-active / BRAF-V600E-co-test / WM-conversion-monitor', locus: '3p22.2',  size: '296 aa / 33 kDa',   inh: 'AD GOF (somatic L265P); AR LOF rare' },
  'TNFRSF13B': { full: 'CVID2-NHL-10-12x-HIGHEST / IVIG-mandatory-IgG>7g/L / rituximab+IVIG-co-mandatory / splenic-MZL-aggressive / EBV-risk / C104R-European-1in500 / HSCT-curative', locus: '17p11.2', size: '293 aa / 31 kDa',   inh: 'AD het (C104R); AR homozygous severe' },
  'ATM':       { full: 'RADIOSENSITIVITY-ABSOLUTE-NO-RT / del11q22.3-15-25pct-MZL / NHL-4-5x / MZL-splenic-3-4x / ceralasertib-ATRi / RIC-HSCT-no-TBI / PET-CT-acceptable', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF heterozygous' },
  'TP53':      { full: 'LFS / MZL-transformation-20-30pct / del17p / AVOID-RADIATION-ABSOLUTELY / WB-MRI-Toronto / venetoclax-backbone / CAR-T-brexucabtagene / cascade-testing', locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous' },
  'CDKN2A':   { full: 'FAMM / gastric-MALT-5-10x / H-pylori-synergy / CDK4-6i-INACTIVE-MZL / del9p21-15-20pct-transformation / EUS-40yr / melanoma-25-36x / H-pylori-eradication-mandatory', locus: '9p21.3',  size: '156 aa / 16 kDa',   inh: 'AD LOF heterozygous' },
  'BRCA2':     { full: 'HBOC / NHL-MZL-1-5-2x / HRD-60pct-SBS3 / cisplatin-preferred / olaparib-maintenance / FA-D1-biallelic-AVOID-ALKYLATING-ABSOLUTELY / annual-breast-MRI', locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF heterozygous' },
  'CARD11':    { full: 'BENTA-GOF-NF-kB / venetoclax-preferred / MALT1-inhibitor-trial / extranodal-MZL-MALT / ibrutinib-active / LOF-CID-EBV-NHL-HSCT-URGENT', locus: '7p22.2',  size: '1154 aa / 133 kDa', inh: 'AD GOF (BENTA); AD/AR LOF (CID)' },
  'PIK3CD':    { full: 'APDS1-EBV-MZL-MALT-10-15pct / leniolisib-FDA2023 / sirolimus-bridge / EBV-viral-load-q3M / salivary-parotid-MALT / HSCT-curative-refractory', locus: '1p36.22', size: '1044 aa / 119 kDa', inh: 'AD GOF heterozygous' },
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

export default function HereditaryMZLPredispositionAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary MZL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: '28px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#4a148c 50%,#b71c1c 100%)',
        borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
          🧬 Hereditary Marginal Zone Lymphoma Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, opacity: 0.88 }}>
          Complete 8-Gene Reference · MYD88 · TNFRSF13B · ATM · TP53 · CDKN2A · BRCA2 · CARD11 · PIK3CD
        </div>
        <div style={{ fontSize: 11, opacity: 0.70, marginTop: 4 }}>
          320-Patient Aggregate · 8×40 · Seeds 3606–3613 · MALT / Nodal MZL / Splenic MZL Germline Predisposition
        </div>
      </div>

      {/* Safety Rules Banner */}
      <div style={{ background: '#ffebee', border: '2px solid #b71c1c', borderRadius: 10,
        padding: '12px 20px', marginBottom: 18, fontSize: 12 }}>
        <strong style={{ color: '#b71c1c' }}>⚠ MZL SAFETY RULES:</strong>
        <span style={{ color: '#555', marginLeft: 8 }}>
          MYD88 L265P+: NO H. pylori monotherapy (fails — rituximab MANDATORY) ·
          TNFRSF13B CVID2: IVIG NEVER STOP (IgG {'>'} 7 g/L), rituximab + IVIG co-mandatory ·
          ATM/NBN: NO RT EVER (RADIOSENSITIVITY ABSOLUTE) ·
          TP53 LFS: AVOID RADIATION ABSOLUTELY (WB-MRI Toronto) ·
          CDKN2A: CDK4/6i INACTIVE in MZL · BRCA2: cisplatin over alkylating (HRD) ·
          PIK3CD APDS1: leniolisib FDA2023 first-line, EBV viral load q3M
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
            <StatCard label="Total Patients"   value={overview.total_patients}  sub="8 genes × 40"                      color="#1a237e" />
            <StatCard label="CR (1st-line)"    value={overview.cr_n}            sub={`${overview.cr_pct}% complete response`} color="#1b5e20" />
            <StatCard label="5yr OS"           value={overview.os5_n}           sub={`${overview.os5_pct}% alive at 5yr`}    color="#4a148c" />
            <StatCard label="Transformed DLBCL" value={overview.transformed_n}  sub={`${overview.transformed_pct}% transformation`} color="#b71c1c" />
            <StatCard label="MYD88 L265P"      value={overview.l265p_n}         sub={`${overview.l265p_pct}% HP-eradication resistant`} color="#1a237e" />
            <StatCard label="EBV+"             value={overview.ebv_pos_n}       sub={`${overview.ebv_pos_pct}% EBV-driven`}  color="#880e4f" />
          </div>
          {/* KPI row 2 — subtype */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="Extranodal MALT"  value={overview.malt_n}          sub={`${overview.malt_pct}% MALT (gastric/ocular/salivary)`} color="#006064" />
            <StatCard label="Nodal MZL"        value={overview.nodal_mzl_n}     sub={`${overview.nodal_mzl_pct}% nodal MZL`}   color="#1b5e20" />
            <StatCard label="Splenic MZL"      value={overview.splenic_mzl_n}   sub={`${overview.splenic_mzl_pct}% splenic MZL`} color="#e65100" />
            <StatCard label="H. pylori+"       value={overview.hp_pos_n}        sub={`${overview.hp_pos_pct}% H. pylori co-exposure`} color="#bf360c" />
            <StatCard label="del11q (ATM)"     value={overview.del11q_n}        sub={`${overview.del11q_pct}% somatic del11q`} color="#4a148c" />
          </div>

          {/* Per-gene summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
            {Object.entries(overview.gene_counts || {}).map(([gene, d]) => {
              const color = GENE_COLORS[gene] || '#555';
              const info  = GENE_INFO[gene]  || {};
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
                      ['MALT', d.malt_n],
                      ['Splenic MZL', d.splenic_n],
                      ['H. pylori+', d.hp_pos_n],
                      ['EBV+', d.ebv_pos_n],
                      ['L265P', d.l265p_n],
                      ['del11q', d.del11q_n],
                      ['Transformed', d.transformed_n],
                      ['CR', d.cr_n],
                      ['5yr OS', d.os5_n],
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
                    {d.mzl_risk?.split(';')[0]}
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
                {['Gene','Locus','Inh','n','Avg Age','CR','5yr OS','MALT','Splenic','Nodal','H.pylori+','EBV+','L265P','del11q','Transformed','Key Avoid'].map(h => (
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
                    <td style={{ padding: '8px 10px', fontSize: 10, maxWidth: 90, whiteSpace: 'nowrap', overflow: 'hidden' }}>{d.inheritance?.split(' ').slice(0,2).join(' ')}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700 }}>{d.n}</td>
                    <td style={{ padding: '8px 10px', color }}>{d.avg_age_dx}yr</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: '#1b5e20' }}>{d.cr_n}</td>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: '#1b5e20' }}>{d.os5_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.subtype_malt_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.subtype_splenic_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.subtype_nodal_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.hp_pos_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ebv_pos_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.l265p_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del11q_n}</td>
                    <td style={{ padding: '8px 10px', color: '#b71c1c', fontWeight: 600 }}>{d.transformed_n}</td>
                    <td style={{ padding: '8px 10px', fontSize: 10, maxWidth: 160, color: '#b71c1c' }}>
                      {d.key_avoid?.split(';')[0]?.slice(0, 60)}…
                    </td>
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
                    ['5yr OS', d.os5_n],
                    ['Transformed', d.transformed_n],
                    ['MALT', d.subtype_malt_n],
                    ['Splenic MZL', d.subtype_splenic_n],
                    ['Nodal MZL', d.subtype_nodal_n],
                    ['H. pylori+', d.hp_pos_n],
                    ['EBV+', d.ebv_pos_n],
                    ['L265P somatic', d.l265p_n],
                    ['del11q', d.del11q_n],
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
                Key Clinical Distinctions — Hereditary MZL Predisposition
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
