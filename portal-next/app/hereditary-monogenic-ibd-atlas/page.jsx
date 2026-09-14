'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-monogenic-ibd-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'IL10':   '#1565c0',  // deep blue    — IL-10 cytokine deficiency, HSCT mandatory
  'IL10RA': '#0277bd',  // medium blue  — IL-10 receptor alpha, ethnic founders
  'IL10RB': '#01579b',  // navy blue    — shared receptor, higher HLH risk
  'XIAP':   '#b71c1c',  // deep red     — XLP-2 X-linked, males only, HLH + IBD
  'LRBA':   '#6a1b9a',  // deep purple  — CVID + IBD + autoimmunity, abatacept
  'CYBB':   '#e65100',  // burnt orange — X-CGD gp91phox, antifungal mandatory
  'CYBA':   '#f57f17',  // amber        — AR-CGD p22phox, both sexes
  'NCF2':   '#2e7d32',  // dark green   — AR-CGD p67phox, perianal prominent
};

const GENE_INFO = {
  'IL10':   { full: 'IL10 / Interleukin-10 / 178aa', locus: '1q32.1', size: '178 aa / 19 kDa', inh: 'AR', disease: 'IL-10 deficiency — onset <3 months; pan-colitis + perianal disease; serum IL-10 ABSENT; HSCT MANDATORY + CURATIVE; conventional IBD therapy FAILS; every month delay = worse outcome' },
  'IL10RA': { full: 'IL10RA / IL-10 Receptor Alpha / 576aa', locus: '11q23.3', size: '576 aa / 66 kDa', inh: 'AR', disease: 'IL-10 receptor alpha deficiency — identical to IL10; serum IL-10 ELEVATED (cytokine present, cannot signal); p.Cys171Tyr Chinese founder >60%; HSCT MANDATORY' },
  'IL10RB': { full: 'IL10RB / IL-10 Receptor Beta (shared) / 325aa', locus: '21q22.11', size: '325 aa / 37 kDa', inh: 'AR', disease: 'IL-10Rb shared receptor (IL-10 + IFN-λ + IL-22) — COMBINED cytokine signalling loss; HLH RISK HIGHER than IL10RA; neonatal onset; antiviral surveillance mandatory' },
  'XIAP':   { full: 'XIAP / BIRC4 / 497aa', locus: 'Xq25', size: '497 aa / 57 kDa', inh: 'XLR', disease: 'XLP-2 — MALES ONLY; Crohn-like ileocolitis + perianal + RECURRENT HLH; XIAP protein absent on intracellular flow cytometry; anti-TNF controls IBD but NOT HLH; HSCT curative both' },
  'LRBA':   { full: 'LRBA / LPS-Responsive Beige Anchor / 2863aa', locus: '4q31.3', size: '2863 aa / 319 kDa', inh: 'AR', disease: 'LRBA deficiency — CVID + Crohn-like IBD + AIHA/ITP + lymphadenopathy; ABATACEPT highly effective (mechanism-targeted — CTLA4 replacement); abatacept response within 2-4 weeks is pathognomonic' },
  'CYBB':   { full: 'CYBB / gp91phox / 570aa', locus: 'Xp21.1', size: '570 aa / 91 kDa', inh: 'XLR', disease: 'X-CGD — MALES ONLY; granulomatous colitis mimicking Crohn\'s; NBT test + DHR flow cytometry DIAGNOSTIC before WES; ANTIFUNGAL (itraconazole) MANDATORY lifelong; Aspergillus lung abscess #1 killer' },
  'CYBA':   { full: 'CYBA / p22phox / 195aa', locus: '16q24.2', size: '195 aa / 22 kDa', inh: 'AR', disease: 'AR-CGD p22phox — BOTH SEXES; identical phenotype to X-CGD; gp91phox ABSENT on Western blot (p22phox stabilises gp91phox); antifungal + antibacterial prophylaxis mandatory' },
  'NCF2':   { full: 'NCF2 / p67phox / 526aa', locus: '1q25.3', size: '526 aa / 67 kDa', inh: 'AR', disease: 'AR-CGD p67phox — PERIANAL DISEASE PROMINENT (abscesses + complex anorectal fistulae); NBT/DHR absent; gp91phox PRESENT on Western blot (cytosolic activator defect); rarest classic CGD gene' },
};

function GeneChip({ gene }) {
  return (
    <span style={{
      background: GENE_COLORS[gene] || '#555',
      color: '#fff',
      borderRadius: 4,
      padding: '2px 8px',
      fontSize: 12,
      fontWeight: 700,
      marginRight: 4,
      display: 'inline-block',
    }}>{gene}</span>
  );
}

function MetricCard({ label, value, sub, warn }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 130, flex: '1 1 130px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: warn ? '#f87171' : '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredMonogenicIBDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [selGene, setSelGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    const ep = tab === 'Definitions' ? 'definitions' : tab === 'Gene Table' ? 'breakdown' : tab === 'Clinical Atlas' ? 'breakdown' : 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (tab === 'Overview') setOverview(d);
        else if (tab === 'Gene Table' || tab === 'Clinical Atlas') setBreakdown(d);
        else setDefinitions(d);
      })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false));
  }, [tab]);

  const bg = '#0f172a';
  const card = '#1e293b';
  const accent = '#1565c0';

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '0 0 40px' }}>
      {/* Header */}
      <div style={{ background: '#1e293b', borderBottom: '1px solid #334155', padding: '18px 28px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>🧬 Hereditary Disease Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          Hereditary Monogenic IBD Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>
          Complete 8-Gene Very Early Onset &amp; Monogenic IBD Reference ·{' '}
          {['IL10','IL10RA','IL10RB','XIAP','LRBA','CYBB','CYBA','NCF2'].map(g => (
            <GeneChip key={g} gene={g} />
          ))}
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#475569' }}>
          320 patients · 8 × 40 · seeds 2558-2565 · IL10/IL10RA/IL10RB (IL-10 pathway — curative HSCT) · XIAP (XLP-2 X-linked Crohn+HLH) · LRBA (CVID+IBD+autoimmunity — abatacept) · CYBB/CYBA/NCF2 (CGD — granulomatous colitis)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, padding: '14px 28px 0', borderBottom: '1px solid #334155' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? accent : 'transparent',
            color: tab === t ? '#fff' : '#94a3b8',
            border: 'none', borderRadius: '6px 6px 0 0', padding: '8px 18px',
            cursor: 'pointer', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 28px' }}>
        {loading && <div style={{ color: '#64748b' }}>Loading…</div>}
        {err && <div style={{ color: '#f87171' }}>Error: {err}</div>}

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
              <MetricCard label="Total Patients" value={overview.total_patients} sub="8 × 40" />
              <MetricCard label="Genes Covered" value={overview.n_genes} sub="seeds 2558-2565" />
              <MetricCard label="Perianal Disease" value={`${overview.aggregate_metrics?.perianal_disease_pct}%`} sub="abscesses + fistulae" warn />
              <MetricCard label="HLH Episodes" value={`${overview.aggregate_metrics?.hlh_episode_pct}%`} sub="XIAP/IL10RB mainly" warn />
              <MetricCard label="Underwent HSCT" value={`${overview.aggregate_metrics?.hsct_pct}%`} sub="curative for IL10/XIAP/CGD" />
              <MetricCard label="On Biologics" value={`${overview.aggregate_metrics?.on_biologic_pct}%`} sub="anti-TNF / abatacept" />
              <MetricCard label="CGD Confirmed" value={`${overview.aggregate_metrics?.cgd_pct}%`} sub="CYBB/CYBA/NCF2" />
              <MetricCard label="Hypogammaglobulinaemia" value={`${overview.aggregate_metrics?.hypogammaglobulinaemia_pct}%`} sub="LRBA mainly" />
              <MetricCard label="Pre-HSCT Mortality" value={`${overview.aggregate_metrics?.mortality_pre_hsct_pct}%`} sub="IL10RB highest risk" warn />
              <MetricCard label="Hospitalisations/yr" value={overview.aggregate_metrics?.avg_hospitalizations_per_year} sub="median across all genes" warn />
            </div>

            {/* Disease class grid */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>8 Disease Classes</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 10 }}>
                {overview.disease_classes?.map((cls, i) => {
                  const gene = overview.genes?.[i];
                  return (
                    <div key={cls} style={{ background: card, borderRadius: 8, padding: '12px 14px', borderLeft: `4px solid ${GENE_COLORS[gene] || '#555'}` }}>
                      <div style={{ color: '#f1f5f9', fontSize: 13, fontWeight: 600 }}>{cls}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Gene summary table */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Gene Summary</div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#334155', color: '#94a3b8' }}>
                      {['Gene','Locus','Inh.','Disease','Onset (mo)','Perianal%','HLH%','HSCT%','Biologic%','Hypogam%','Mortality%'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_summary?.map(g => (
                      <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                        <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                        <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                        <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance?.split('(')[0]?.trim()}</td>
                        <td style={{ padding: '8px 10px', color: '#f1f5f9', maxWidth: 160 }}>{g.disease_name}</td>
                        <td style={{ padding: '8px 10px', color: g.onset_months_median <= 3 ? '#f87171' : '#94a3b8' }}>{g.onset_months_median}</td>
                        <td style={{ padding: '8px 10px', color: g.perianal_disease_pct > 50 ? '#fb923c' : '#cbd5e1' }}>{g.perianal_disease_pct}%</td>
                        <td style={{ padding: '8px 10px', color: g.hlh_risk_pct > 20 ? '#f87171' : '#94a3b8' }}>{g.hlh_risk_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.hsct_rate_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.on_biologic_pct}%</td>
                        <td style={{ padding: '8px 10px', color: g.hypogammaglobulinaemia_pct > 50 ? '#a78bfa' : '#94a3b8' }}>{g.hypogammaglobulinaemia_pct}%</td>
                        <td style={{ padding: '8px 10px', color: g.mortality_pre_hsct_pct > 10 ? '#f87171' : '#94a3b8' }}>{g.mortality_pre_hsct_pct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Clinical pearls */}
            <div style={{ background: card, borderRadius: 8, padding: '16px 20px' }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 10, textTransform: 'uppercase', letterSpacing: 1 }}>Clinical Pearls</div>
              {overview.clinical_pearls?.map((p, i) => (
                <div key={i} style={{ marginBottom: 8, fontSize: 13, color: '#e2e8f0', display: 'flex', gap: 8 }}>
                  <span style={{ color: accent, fontWeight: 800 }}>⬥</span>
                  <span>{p}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 8, marginBottom: 20 }}>
              {breakdown.gene_breakdowns?.map(g => (
                <div key={g.gene}
                  onClick={() => setSelGene(selGene === g.gene ? null : g.gene)}
                  style={{ background: selGene === g.gene ? GENE_COLORS[g.gene] + '33' : card, borderRadius: 8, padding: '12px 14px', cursor: 'pointer', border: `1px solid ${selGene === g.gene ? GENE_COLORS[g.gene] : '#334155'}` }}>
                  <GeneChip gene={g.gene} />
                  <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>{g.locus} · {g.protein_size}</div>
                  <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{g.inheritance?.split(';')[0]}</div>
                </div>
              ))}
            </div>

            {selGene && (() => {
              const g = breakdown.gene_breakdowns?.find(x => x.gene === selGene);
              if (!g) return null;
              const info = GENE_INFO[selGene];
              return (
                <div style={{ background: card, borderRadius: 10, padding: '20px 24px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                    <GeneChip gene={g.gene} />
                    <div>
                      <div style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 16 }}>{info?.full}</div>
                      <div style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance?.split(';')[0]}</div>
                    </div>
                  </div>

                  {/* Key metrics */}
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
                    {[
                      { label: 'Patients', value: g.n_patients },
                      { label: 'Onset (mo avg)', value: g.avg_onset_months },
                      { label: 'Perianal %', value: `${g.perianal_disease_pct}%` },
                      { label: 'HLH %', value: `${g.hlh_episode_pct}%` },
                      { label: 'HSCT %', value: `${g.hsct_pct}%` },
                      { label: 'Biologic %', value: `${g.on_biologic_pct}%` },
                      { label: 'Hypogam %', value: `${g.hypogammaglobulinaemia_pct}%` },
                      { label: 'CGD %', value: `${g.cgd_confirmed_pct}%` },
                      { label: 'Mortality pre-HSCT', value: `${g.mortality_pre_hsct_pct}%` },
                      { label: 'Hosp/yr avg', value: g.avg_hospitalizations },
                      { label: 'Colonoscopy score', value: g.avg_colonoscopy_score },
                    ].map(m => (
                      <div key={m.label} style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 90 }}>
                        <div style={{ color: '#64748b', fontSize: 10 }}>{m.label}</div>
                        <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{m.value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Pathognomonic */}
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', marginBottom: 12, borderLeft: `3px solid ${GENE_COLORS[selGene]}` }}>
                    <div style={{ color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Pathognomonic</div>
                    <div style={{ color: '#fbbf24', fontSize: 12, fontWeight: 600 }}>{g.pathognomonic}</div>
                  </div>

                  {/* Key features */}
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Key Features</div>
                    {g.key_features?.map((f, i) => (
                      <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 12, color: '#e2e8f0' }}>
                        <span style={{ color: GENE_COLORS[selGene] }}>▸</span><span>{f}</span>
                      </div>
                    ))}
                  </div>

                  {/* Treatment */}
                  <div style={{ background: '#0f172a', borderRadius: 6, padding: '10px 14px', marginBottom: 12 }}>
                    <div style={{ color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Treatment</div>
                    <div style={{ color: '#e2e8f0', fontSize: 12 }}>{g.treatment}</div>
                  </div>

                  {/* DDx */}
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Key DDx</div>
                    {g.key_ddx?.map((d, i) => (
                      <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 4, fontSize: 12, color: '#94a3b8' }}>
                        <span style={{ color: '#f87171' }}>⚠</span><span>{d}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            {breakdown.gene_breakdowns?.map(g => (
              <div key={g.gene} style={{ background: card, borderRadius: 10, padding: '18px 22px', marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#555'}` }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <GeneChip gene={g.gene} />
                  <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{g.disease_category?.split(' —')[0]}</div>
                  <div style={{ marginLeft: 'auto', color: '#64748b', fontSize: 11 }}>{g.locus} · {g.inheritance?.split(';')[0]}</div>
                </div>
                <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 8, fontStyle: 'italic' }}>{g.pathognomonic?.substring(0, 200)}…</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: '#94a3b8' }}>
                  <span>Onset: <b style={{ color: g.avg_onset_months <= 3 ? '#f87171' : '#f1f5f9' }}>{g.avg_onset_months} mo</b></span>
                  <span>Perianal: <b style={{ color: g.perianal_disease_pct > 50 ? '#fb923c' : '#f1f5f9' }}>{g.perianal_disease_pct}%</b></span>
                  <span>HLH: <b style={{ color: g.hlh_episode_pct > 20 ? '#f87171' : '#f1f5f9' }}>{g.hlh_episode_pct}%</b></span>
                  <span>HSCT: <b style={{ color: '#f1f5f9' }}>{g.hsct_pct}%</b></span>
                  <span>Biologic: <b style={{ color: '#f1f5f9' }}>{g.on_biologic_pct}%</b></span>
                  <span>Hypogam: <b style={{ color: g.hypogammaglobulinaemia_pct > 50 ? '#a78bfa' : '#f1f5f9' }}>{g.hypogammaglobulinaemia_pct}%</b></span>
                  <span>Mortality pre-HSCT: <b style={{ color: g.mortality_pre_hsct_pct > 10 ? '#f87171' : '#f1f5f9' }}>{g.mortality_pre_hsct_pct}%</b></span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* Gene entries */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Gene Definitions</div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 14 }}>
                {Object.keys(definitions.gene_entries || {}).map(g => (
                  <button key={g} onClick={() => setSelGene(selGene === g ? null : g)} style={{
                    background: GENE_COLORS[g] || '#334155',
                    color: '#fff', border: 'none', borderRadius: 4,
                    padding: '4px 12px', cursor: 'pointer', fontWeight: 700, fontSize: 12,
                    opacity: selGene && selGene !== g ? 0.5 : 1,
                  }}>{g}</button>
                ))}
              </div>

              {Object.entries(definitions.gene_entries || {}).filter(([g]) => !selGene || selGene === g).map(([gene, d]) => (
                <div key={gene} style={{ background: card, borderRadius: 8, padding: '16px 20px', marginBottom: 10, borderLeft: `3px solid ${GENE_COLORS[gene] || '#555'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <GeneChip gene={gene} />
                    <div style={{ color: '#f1f5f9', fontWeight: 700 }}>{d.disease_name?.split(' —')[0] || d.full_name}</div>
                    <div style={{ marginLeft: 'auto', color: '#64748b', fontSize: 11 }}>{d.locus} · {d.protein_size} · {d.inheritance}</div>
                  </div>
                  <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 8 }}>{d.pathognomonic}</div>
                  <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8 }}><b style={{ color: '#e2e8f0' }}>Treatment:</b> {d.treatment}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 8, fontSize: 11, marginBottom: 8 }}>
                    <div><span style={{ color: '#64748b' }}>Perianal:</span> <span style={{ color: d.perianal_disease_pct > 50 ? '#fb923c' : '#cbd5e1' }}>{d.perianal_disease_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>HLH risk:</span> <span style={{ color: d.hlh_risk_pct > 20 ? '#f87171' : '#cbd5e1' }}>{d.hlh_risk_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>HSCT rate:</span> <span style={{ color: '#cbd5e1' }}>{d.hsct_rate_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>CGD:</span> <span style={{ color: d.cgd_pct === 100 ? '#fb923c' : '#cbd5e1' }}>{d.cgd_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Hypogam:</span> <span style={{ color: d.on_biologic_pct > 50 ? '#a78bfa' : '#cbd5e1' }}>{d.on_biologic_pct}%</span></div>
                    <div><span style={{ color: '#64748b' }}>Mortality pre-HSCT:</span> <span style={{ color: d.mortality_pre_hsct_pct > 10 ? '#f87171' : '#cbd5e1' }}>{d.mortality_pre_hsct_pct}%</span></div>
                  </div>
                  <div>
                    <div style={{ color: '#94a3b8', fontSize: 10, textTransform: 'uppercase', marginBottom: 4 }}>Key DDx</div>
                    {d.key_ddx?.map((ddx, i) => (
                      <div key={i} style={{ fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>⚠ {ddx}</div>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Monogenic IBD Glossary */}
            <div>
              <div style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Monogenic IBD Glossary</div>
              {Object.entries(definitions.mibd_glossary || {}).map(([term, def]) => (
                <div key={term} style={{ background: '#0f172a', borderRadius: 8, padding: '14px 18px', marginBottom: 8 }}>
                  <div style={{ color: '#fbbf24', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>{term}</div>
                  <div style={{ color: '#e2e8f0', fontSize: 12, lineHeight: 1.6 }}>{def}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
