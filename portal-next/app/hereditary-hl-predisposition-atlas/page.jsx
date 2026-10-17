'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hl-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'ATM':      '#4a148c',  // deep purple       — RADIOSENSITIVITY ABSOLUTE; HL 3-4x; ceralasertib ATRi
  'NBN':      '#1a237e',  // deep navy         — NHL 60x HIGHEST biallelic; 657del5 Slavic; cranial RT CI
  'BRCA2':    '#1b5e20',  // deep green        — HBOC; HRD 60%; cisplatin preferred; olaparib PARP
  'TP53':     '#880e4f',  // deep magenta      — LFS; AVOID RADIATION ABSOLUTELY; WB-MRI Toronto; venetoclax
  'CHEK2':    '#006064',  // dark teal         — 1100delC European 1in200; NOT HRD — NO PARPi; ATM+CHEK2 8-10x
  'BLM':      '#e65100',  // deep orange       — Bloom syndrome; SCE PATHOGNOMONIC; HL 150-300x biallelic
  'KLHDC8B':  '#5d4037',  // deep brown        — NLPHL-specific; EBV-negative; rituximab preferred; young onset
  'RB1':      '#b71c1c',  // deep red          — CDK4/6i INACTIVE; secondary HL post-Rb RT 5-10x
};

const GENE_INFO = {
  'ATM':     { full: 'RADIOSENSITIVITY-ABSOLUTE / HL-3-4x-monoallelic / A-T-biallelic-40-60x / BV-AVD-preferred-over-ABVD+RT / ceralasertib-ATRi / del11q-15-25pct / IgA-deficiency-A-T', locus: '11q22.3', size: '3056 aa / 350 kDa', inh: 'AD LOF (monoallelic); AR biallelic = A-T' },
  'NBN':     { full: 'NHL-60x-HIGHEST-biallelic / 657del5-Slavic-1in150 / cranial-RT-absolutely-CI / microcephaly-PATHOGNOMONIC / RIC-HSCT-no-TBI / NBS-biallelic / cascade-testing', locus: '8q21.3',  size: '754 aa / 95 kDa',   inh: 'AR biallelic = NBS; AD heterozygous carrier = mild risk' },
  'BRCA2':   { full: 'HBOC / HL-2-3x / HRD-60pct-SBS3 / cisplatin-preferred / olaparib-maintenance / FA-D1-biallelic-AVOID-ALKYLATING-ABSOLUTELY / annual-breast-MRI', locus: '13q12.3', size: '3418 aa / 384 kDa', inh: 'AD LOF (HBOC); AR biallelic = FA-D1' },
  'TP53':    { full: 'LFS / HL-3-5x / AVOID-RADIATION-ABSOLUTELY / WB-MRI-Toronto-annually / venetoclax-backbone / chemotherapy-only-no-RT / cascade-TP53-testing / secondary-HL', locus: '17p13.1', size: '393 aa / 43 kDa',   inh: 'AD LOF heterozygous' },
  'CHEK2':   { full: '1100delC-European-1in200 / HL-2-3x / NOT-HRD-NO-PARPi / ATM+CHEK2-additive-8-10x / IVS2-Ashkenazi / standard-RT-safe / BV-AVD-first-line', locus: '22q12.1', size: '543 aa / 60 kDa',   inh: 'AD LOF heterozygous' },
  'BLM':     { full: 'Bloom-syndrome-biallelic / SCE-10x-PATHOGNOMONIC / HL-150-300x-biallelic / 2281del6ins7-Ashkenazi-1in107 / AVOID-ALKYLATING-ABSOLUTELY / AVOID-RT / gemcitabine-based-protocol', locus: '15q26.1', size: '1417 aa / 159 kDa', inh: 'AR biallelic = Bloom syndrome; AD carrier = mild HL risk' },
  'KLHDC8B': { full: 'NLPHL-specific-familial / EBV-negative / LP-cells-CD20+-CD15--CD30- / young-onset-20-40yr / rituximab-monotherapy-early / R-CHOP-advanced / ABVD-WRONG-for-NLPHL', locus: '3q22.3',  size: '560 aa / 63 kDa',   inh: 'AD LOF heterozygous (familial NLPHL)' },
  'RB1':     { full: 'CDK4-6i-INACTIVE-pRb-bypassed / secondary-HL-post-Rb-RT-5-10x / del13q14-somatic / BV-AVD-no-RT / venetoclax-BCL2-active / avoid-RT-Rb-survivors / bilateral-Rb-PATHOGNOMONIC', locus: '13q14.2', size: '928 aa / 110 kDa',  inh: 'AD LOF heterozygous (hereditary Rb)' },
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

export default function HereditaryHLPredispositionAtlasPage() {
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
    fontSize: 13, background: tab === t ? '#4a148c' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary HL Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: '28px 24px' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#4a148c 0%,#1a237e 50%,#b71c1c 100%)',
        borderRadius: 14, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 22, fontWeight: 800, marginBottom: 6 }}>
          🧬 Hereditary Hodgkin Lymphoma Predisposition Atlas
        </div>
        <div style={{ fontSize: 13, opacity: 0.88 }}>
          Complete 8-Gene Reference · ATM · NBN · BRCA2 · TP53 · CHEK2 · BLM · KLHDC8B · RB1
        </div>
        <div style={{ fontSize: 11, opacity: 0.70, marginTop: 4 }}>
          320-Patient Aggregate · 8×40 · Seeds 3614–3621 · cHL / NLPHL Germline Predisposition
        </div>
      </div>

      {/* Safety Rules Banner */}
      <div style={{ background: '#f3e5f5', border: '2px solid #4a148c', borderRadius: 10,
        padding: '12px 20px', marginBottom: 18, fontSize: 12 }}>
        <strong style={{ color: '#4a148c' }}>⚠ HL SAFETY RULES:</strong>
        <span style={{ color: '#555', marginLeft: 8 }}>
          ATM/NBN monoallelic: RADIOSENSITIVITY — dose-reduce RT 20-30% or substitute BV-AVD chemotherapy-only ·
          ATM biallelic (A-T) / NBN biallelic (NBS): RT ABSOLUTELY CI — fatal toxicity ·
          TP53 LFS: AVOID RADIATION ABSOLUTELY (WB-MRI Toronto annually) ·
          BLM biallelic: AVOID ALKYLATING ABSOLUTELY + AVOID RT (SCE diagnostic) ·
          KLHDC8B NLPHL: NOT cHL — ABVD WRONG; rituximab/R-CHOP correct (LP cells CD20+) ·
          CHEK2: NOT HRD — NO PARPi ·
          RB1: CDK4/6i INACTIVE (pRb bypassed)
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
            <StatCard label="Total Patients"    value={overview.total_patients}  sub="8 genes × 40"                         color="#4a148c" />
            <StatCard label="CR (1st-line)"     value={overview.cr_n}            sub={`${overview.cr_pct}% complete response`}  color="#1b5e20" />
            <StatCard label="5yr OS"            value={overview.os5_n}           sub={`${overview.os5_pct}% alive at 5yr`}       color="#1a237e" />
            <StatCard label="Transformed DLBCL" value={overview.transformed_n}   sub={`${overview.transformed_pct}% transformation`} color="#b71c1c" />
            <StatCard label="EBV+"              value={overview.ebv_pos_n}        sub={`${overview.ebv_pos_pct}% EBV-associated`}   color="#880e4f" />
          </div>
          {/* KPI row 2 */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="Classical HL (cHL)" value={overview.chl_n}          sub={`${overview.chl_pct}% RS cells (cHL)`}       color="#4a148c" />
            <StatCard label="NLPHL"              value={overview.nlphl_n}         sub={`${overview.nlphl_pct}% LP cells (NLPHL)`}   color="#5d4037" />
            <StatCard label="del11q (ATM)"       value={overview.del11q_n}        sub={`${overview.del11q_pct}% somatic del11q`}    color="#4a148c" />
            <StatCard label="del13q (RB1)"       value={overview.del13q_n}        sub={`${overview.del13q_pct}% somatic del13q14`}  color="#b71c1c" />
            <StatCard label="BV-AVD regimen"     value={overview.bv_avd_n}        sub={`${overview.bv_avd_pct}% brentuximab+AVD`}  color="#006064" />
            <StatCard label="Rituximab regimen"  value={overview.rituximab_n}     sub={`${overview.rituximab_pct}% anti-CD20`}      color="#5d4037" />
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
                      ['cHL', d.chl_n],
                      ['NLPHL', d.nlphl_n],
                      ['EBV+', d.ebv_pos_n],
                      ['del11q', d.del11q_n],
                      ['del13q', d.del13q_n],
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
                    {d.hl_risk?.split(';')[0]}
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
              <tr style={{ background: '#4a148c', color: '#fff' }}>
                {['Gene','Locus','Inh','n','Avg Age','CR','5yr OS','cHL','NLPHL','EBV+','del11q','del13q','Transf','BV-AVD','Rituximab','Key Avoid'].map(h => (
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
                    <td style={{ padding: '8px 10px' }}>{d.chl_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.nlphl_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.ebv_pos_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del11q_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.del13q_n}</td>
                    <td style={{ padding: '8px 10px', color: '#b71c1c', fontWeight: 600 }}>{d.transformed_n}</td>
                    <td style={{ padding: '8px 10px' }}>{d.tx_bv_avd}</td>
                    <td style={{ padding: '8px 10px' }}>{d.tx_rituximab}</td>
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
                    ['cHL', d.chl_n],
                    ['NLPHL', d.nlphl_n],
                    ['EBV+', d.ebv_pos_n],
                    ['del11q', d.del11q_n],
                    ['del13q14', d.del13q_n],
                    ['BV-AVD', d.tx_bv_avd],
                    ['Rituximab', d.tx_rituximab],
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
                <div style={{ fontSize: 11, background: '#ede7f6', borderRadius: 6,
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
              <div style={{ fontSize: 15, fontWeight: 800, color: '#4a148c', marginBottom: 12 }}>
                Key Clinical Distinctions — Hereditary HL Predisposition
              </div>
              {definitions.key_clinical_distinctions.map((d, i) => (
                <div key={i} style={{ background: '#ede7f6', borderLeft: `4px solid ${Object.values(GENE_COLORS)[i % 8]}`,
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
