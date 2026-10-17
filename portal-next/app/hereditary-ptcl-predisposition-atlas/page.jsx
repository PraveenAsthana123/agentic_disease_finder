'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ptcl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':     '#006064',  // dark cyan          — biallelic A-T T-cell lymphoma 75x; TBI ABSOLUTELY CI; ceralasertib ATRi
  'TP53':    '#bf360c',  // deep orange-red     — LFS PTCL 3-4x; AVOID RADIATION ABSOLUTELY; venetoclax; pralatrexate FOLR1
  'CD27':    '#1b5e20',  // forest green        — CD27 deficiency; EBV+ T/NK LPD 60-70x HIGHEST; HSCT MANDATORY; Rituximab WRONG
  'ITK':     '#4a148c',  // deep purple         — ITK deficiency; EBV CD4+ T-LPD 90%+; HSCT before EBV; HLH 50%
  'GATA2':   '#0d47a1',  // deep blue           — Emberger; NK-lymphopenia; NK/T-cell lymphoma; HSCT curative; EBV Q3M
  'TET2':    '#880e4f',  // deep pink           — AITL 65%; IDH2 R172K co-mutation MANDATORY; enasidenib; azacitidine HMA
  'STAT3':   '#33691e',  // olive green         — T-LGL; cytopenia HALLMARK; ruxolitinib JAKi FDA2023; methotrexate AVOID
  'PIK3CD':  '#f57f17',  // amber dark          — APDS1; EBV/CMV T-LPD 10-15%; leniolisib FDA2023; acyclovir MANDATORY
};

const GENE_INFO = {
  'ATM':    { full: 'RADIOSENSITIVITY-ABSOLUTE / biallelic-A-T-T-cell-lymphoma-75x / monoallelic-PTCL-3-5x / TBI-conditioning-CI / RIC-HSCT-only / BV-CHP-preferred / ceralasertib-ATRi / EBV-Q3M', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF (monoallelic); AR biallelic = A-T' },
  'TP53':   { full: 'LFS / PTCL-3-4x / AVOID-RADIATION-ABSOLUTELY / WB-MRI-Toronto-annually / venetoclax-backbone / pralatrexate-FOLR1-FDA / complex-karyotype-50-60pct', locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous' },
  'CD27':   { full: 'CD27-Deficiency / EBV+T-NK-LPD-60-70x-HIGHEST / HLH-45pct / HSCT-MANDATORY / RITUXIMAB-NOT-APPROPRIATE-T-cell-CD20neg / EBV-IGIV-bridge / NK-T-cell-60pct', locus: '12p13.31', size: '260 aa / 29 kDa',  inh: 'AR biallelic LOF' },
  'ITK':    { full: 'ITK-Deficiency / EBV-CD4+T-LPD-90pct / HLH-50pct / HSCT-MANDATORY-before-EBV / RITUXIMAB-INEFFECTIVE-T-cell / EBV-IGIV-bridge / etoposide-HLH', locus: '5q33.3',   size: '620 aa / 72 kDa',  inh: 'AR biallelic LOF' },
  'GATA2':  { full: 'GATA2-Deficiency / Emberger-lymphedema-PATHOGNOMONIC / MonoMAC / NK-lymphopenia / NK-T-cell-50pct / EBV-CMV-Q3M-MANDATORY / HSCT-curative / SMILE-AspaMetDex', locus: '3q21.3',   size: '480 aa / 50 kDa',  inh: 'AD LOF heterozygous' },
  'TET2':   { full: 'Germline-TET2-CHIP / AITL-65pct / IDH2-R172K-co-mutation-MANDATORY-test / enasidenib-if-IDH2+ / azacitidine-decitabine-HMA-sensitive / PTCL-3-5x', locus: '4q24',      size: '2002 aa / 223 kDa', inh: 'AD LOF heterozygous (CHIP-associated)' },
  'STAT3':  { full: 'STAT3-GOF / T-LGL-leukemia / cytopenia-HALLMARK-neutropenia-65pct-anaemia-50pct / ruxolitinib-JAKi-FDA2023 / METHOTREXATE-AVOID-LGL / cyclosporine-secondline', locus: '17q21.2',  size: '770 aa / 92 kDa',  inh: 'AD GOF heterozygous' },
  'PIK3CD': { full: 'APDS1 / PI3Kdelta-GOF-E1021K / EBV-CMV-T-LPD-10-15pct / leniolisib-FDA2023 / acyclovir-MANDATORY / sirolimus-bridge / live-vaccines-ABSOLUTELY-CI / EBV-CMV-Q3M', locus: '1p36.22',  size: '1044 aa / 119 kDa', inh: 'AD GOF heterozygous' },
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

export default function HereditaryPTCLPredispositionAtlasPage() {
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
    fontSize: 13, background: tab === t ? '#006064' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary PTCL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: '28px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#006064 0%,#0d47a1 50%,#880e4f 100%)',
        borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
          🧬 Hereditary Peripheral T-Cell Lymphoma Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, opacity: 0.88 }}>
          Complete 8-Gene Reference · ATM · TP53 · CD27 · ITK · GATA2 · TET2 · STAT3 · PIK3CD
        </div>
        <div style={{ fontSize: 11, opacity: 0.70, marginTop: 4 }}>
          320-Patient Aggregate · 8×40 · Seeds 3622–3629 · PTCL / T-cell Lymphoma Germline Predisposition
        </div>
      </div>

      {/* Safety Rules Banner */}
      <div style={{ background: '#e0f2f1', border: '2px solid #006064', borderRadius: 10,
        padding: '12px 20px', marginBottom: 18, fontSize: 12 }}>
        <strong style={{ color: '#006064' }}>⚠ PTCL SAFETY RULES:</strong>
        <span style={{ color: '#555', marginLeft: 8 }}>
          ATM biallelic (A-T): T-cell lymphoma 75x — TBI conditioning ABSOLUTELY CI ·
          ATM monoallelic: PTCL 3-5x — RADIOSENSITIVITY reduced-intensity conditioning only ·
          TP53 LFS: AVOID RADIATION ABSOLUTELY — WB-MRI Toronto protocol ·
          CD27/ITK deficiency: RITUXIMAB INEFFECTIVE (T-cell disease) — HSCT MANDATORY ·
          GATA2 Emberger: NK/T lymphoma — EBV viral load Q3M — HSCT before decompensation ·
          TET2 germline: test IDH2 R172K co-mutation — enasidenib if co-IDH2 positive ·
          STAT3 GOF: METHOTREXATE AVOID in LGL — ruxolitinib JAKi FDA2023 first ·
          PIK3CD APDS1: acyclovir MANDATORY prophylaxis — live vaccines ABSOLUTELY CI
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
            <StatCard label="Total Patients"  value={overview.total_patients}  sub="8 genes × 40"                          color="#006064" />
            <StatCard label="CR (1st-line)"   value={overview.cr_n}            sub={`${overview.cr_pct}% complete response`}   color="#1b5e20" />
            <StatCard label="5yr OS"          value={overview.os5_n}           sub={`${overview.os5_pct}% alive at 5yr`}        color="#0d47a1" />
            <StatCard label="EBV+"            value={overview.ebv_pos_n}       sub={`${overview.ebv_pos_pct}% EBV-associated`}   color="#bf360c" />
            <StatCard label="HLH Triggered"   value={overview.hlh_n}           sub={`${overview.hlh_pct}% HLH trigger`}          color="#880e4f" />
          </div>
          {/* KPI row 2 */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="HSCT (curative)" value={overview.hsct_n}          sub={`${overview.hsct_pct}% received HSCT`}       color="#4a148c" />
            <StatCard label="AITL subtype"    value={overview.aitl_n}          sub={`${overview.aitl_pct}% AITL`}                 color="#880e4f" />
            <StatCard label="NK/T-cell"       value={overview.nkt_n}           sub={`${overview.nkt_pct}% NK/T-cell subtype`}     color="#006064" />
            <StatCard label="CHOP/CHOEP"      value={overview.chop_n}          sub={`${overview.chop_pct}% front-line CHOP`}      color="#33691e" />
            <StatCard label="Targeted Rx"     value={overview.targeted_n}      sub={`${overview.targeted_pct}% targeted agent`}   color="#f57f17" />
            <StatCard label="Brentuximab"     value={overview.bv_n}            sub={`${overview.bv_pct}% BV received`}            color="#0d47a1" />
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
                      ['AITL', d.aitl_n],
                      ['NK/T-cell', d.nkt_n],
                      ['EBV+', d.ebv_pos_n],
                      ['HLH', d.hlh_n],
                      ['HSCT', d.hsct_n],
                      ['CHOP/CHOEP', d.tx_chop_choep],
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
                    {d.ptcl_risk?.split(';')[0]}
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
              <tr style={{ background: '#006064', color: '#fff' }}>
                {['Gene','Locus','Inh','n','Avg Age','CR','5yr OS','AITL','NK/T','EBV+','HLH','HSCT','CHOP','Targeted','Key Avoid'].map(h => (
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
                    <td style={{ padding: '8px 10px' }}>{d.aitl_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.nkt_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ebv_pos_n}</td>
                    <td style={{ padding: '8px 10px', color: '#880e4f', fontWeight: 600 }}>{d.hlh_n}</td>
                    <td style={{ padding: '8px 10px', color: '#4a148c', fontWeight: 600 }}>{d.hsct_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.tx_chop_choep}</td>
                    <td style={{ padding: '8px 10px' }}>{d.tx_targeted}</td>
                    <td style={{ padding: '8px 10px', fontSize: 10, maxWidth: 160, color: '#bf360c' }}>
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
                    ['AITL', d.aitl_n],
                    ['NK/T-cell', d.nkt_n],
                    ['EBV+', d.ebv_pos_n],
                    ['HLH Triggered', d.hlh_n],
                    ['HSCT (curative)', d.hsct_n],
                    ['CHOP/CHOEP', d.tx_chop_choep],
                    ['Targeted Rx', d.tx_targeted],
                    ['Brentuximab', d.tx_brentuximab],
                  ].map(([k, v]) => (
                    <div key={k} style={{ background: '#f8f9fa', borderRadius: 6,
                      padding: '5px 8px', fontSize: 11 }}>
                      <div style={{ color: '#aaa', fontSize: 10 }}>{k}</div>
                      <div style={{ fontWeight: 700, color }}>{v}</div>
                    </div>
                  ))}
                </div>
                <div style={{ fontSize: 11, background: '#ffebee', borderRadius: 6,
                  padding: '6px 10px', color: '#bf360c', marginBottom: 6 }}>
                  <strong>KEY AVOID:</strong> {d.key_avoid?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e0f2f1', borderRadius: 6,
                  padding: '6px 10px', color: '#006064', marginBottom: 6 }}>
                  <strong>KEY RULE:</strong> {d.key_rule?.split(';')[0]}
                </div>
                <div style={{ fontSize: 11, background: '#e8eaf6', borderRadius: 6,
                  padding: '6px 10px', color: '#4a148c' }}>
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
              <div style={{ fontSize: 15, fontWeight: 800, color: '#006064', marginBottom: 12 }}>
                Key Clinical Distinctions — Hereditary PTCL Predisposition
              </div>
              {definitions.key_clinical_distinctions.map((d, i) => (
                <div key={i} style={{ background: '#e0f2f1', borderLeft: `4px solid ${Object.values(GENE_COLORS)[i % 8]}`,
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
