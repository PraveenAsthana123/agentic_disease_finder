'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-periodic-fever-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'MEFV':     '#b71c1c',  // deep red        — FMF, pyrin inflammasome, colchicine mandatory
  'MVK':      '#e65100',  // burnt orange     — HIDS, vaccination trigger, canakinumab
  'TNFRSF1A': '#f57f17',  // amber-orange     — TRAPS, migratory myalgia, etanercept
  'NLRP3':    '#1565c0',  // deep blue        — CAPS, cold urticaria, canakinumab
  'NOD2':     '#2e7d32',  // dark green       — Blau, granuloma triad, infliximab
  'PSTPIP1':  '#6a1b9a',  // deep purple      — PAPA, pyoderma gangrenosum, pathergy
  'IL1RN':    '#00695c',  // teal-green       — DIRA, neonatal, anakinra curative
  'NLRP12':   '#37474f',  // blue-grey        — FCAS2, cold triggered, colchicine partial
};

const GENE_INFO = {
  'MEFV':     { full: 'MEFV / Pyrin / 781aa', locus: '16p13.3', size: '781 aa / 86 kDa', inh: 'AR (AD-modifier)', disease: 'Familial Mediterranean Fever (FMF) — COLCHICINE MANDATORY AND LIFELONG (prevents AA amyloidosis); attacks 12-72h sterile peritonitis; M694V most severe; anakinra/canakinumab for colchicine-resistant' },
  'MVK':      { full: 'MVK / Mevalonate Kinase / 396aa', locus: '12q24.11', size: '396 aa / 43 kDa', inh: 'AR', disease: 'HIDS/MKD — VACCINATION-TRIGGERED ATTACKS PATHOGNOMONIC; cervical lymphadenopathy KEY DDx from FMF; urinary mevalonic acid diagnostic; canakinumab FDA-approved; colchicine NOT effective' },
  'TNFRSF1A': { full: 'TNFRSF1A / TNFR1 / 455aa', locus: '12p13.31', size: '455 aa / 51 kDa', inh: 'AD', disease: 'TRAPS — ATTACKS >7 DAYS + MIGRATORY MYALGIA + PERIORBITAL EDEMA PATHOGNOMONIC; ETANERCEPT preferred (NOT infliximab — anti-drug antibodies); R92Q/P46L low penetrance — uncertain significance; canakinumab FDA-approved' },
  'NLRP3':    { full: 'NLRP3 / Cryopyrin / 1036aa', locus: '1q44', size: '1036 aa / 118 kDa', inh: 'AD GOF', disease: 'CAPS (FCAS → Muckle-Wells → NOMID) — NON-PRURITIC URTICARIA PATHOGNOMONIC (antihistamines INEFFECTIVE); COLD TRIGGER = FCAS; NOMID treat URGENTLY; canakinumab + rilonacept FDA-approved; colchicine NOT effective' },
  'NOD2':     { full: 'NOD2 / CARD15 / 1040aa', locus: '16q12.1', size: '1040 aa / 114 kDa', inh: 'AD GOF', disease: 'Blau Syndrome/EOS — GRANULOMATOUS ARTHRITIS + UVEITIS + SKIN GRANULOMA TRIAD PATHOGNOMONIC; onset <4 years; R334W/R334Q hotspot; infliximab best evidence; NOT Crohn NOD2 variants (those are LOF common alleles)' },
  'PSTPIP1':  { full: 'PSTPIP1 / CD2BP1 / 416aa', locus: '15q24.3', size: '416 aa / 47 kDa', inh: 'AD', disease: 'PAPA Syndrome — PYOGENIC ARTHRITIS + PYODERMA GANGRENOSUM + ACNE TRIAD PATHOGNOMONIC; PG PATHERGY: AVOID DEBRIDEMENT (worsens); culture-negative purulent arthritis (sterile); anakinra/canakinumab effective' },
  'IL1RN':    { full: 'IL1RN / IL-1Ra / 177aa', locus: '2q14.2', size: '177 aa / 25 kDa', inh: 'AR', disease: 'DIRA — NEONATAL ONSET multifocal sterile osteomyelitis + pustulosis + periostitis; ANAKINRA CURATIVE AND LIFE-SAVING (physiological IL-1Ra replacement); start URGENTLY — delay causes irreversible bone destruction; life-long treatment' },
  'NLRP12':   { full: 'NLRP12 / Monarch-1 / 1062aa', locus: '19q13.42', size: '1062 aa / 119 kDa', inh: 'AD LOF', disease: 'FCAS2/NLRP12-AU — COLD-TRIGGERED attacks (like FCAS/NLRP3 but distinct); colchicine PARTIALLY effective (unlike CAPS); less deafness; often misdiagnosed as FCAS or TRAPS; anakinra/canakinumab second-line' },
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

function MetricCard({ label, value, sub }) {
  return (
    <div style={{ background: '#1e293b', borderRadius: 8, padding: '14px 18px', minWidth: 140, flex: '1 1 140px' }}>
      <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 4 }}>{label}</div>
      <div style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function HeredPeriodicFeverAtlasPage() {
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

  return (
    <div style={{ background: bg, minHeight: '100vh', color: '#f1f5f9', fontFamily: 'system-ui,sans-serif', padding: '0 0 40px' }}>
      {/* Header */}
      <div style={{ background: '#1e293b', borderBottom: '1px solid #334155', padding: '18px 28px' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>🧬 Hereditary Disease Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>
          Hereditary Periodic Fever Syndrome Atlas
        </h1>
        <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>
          Complete 8-Gene Hereditary Periodic Fever &amp; Autoinflammatory Atlas ·{' '}
          {['MEFV','MVK','TNFRSF1A','NLRP3','NOD2','PSTPIP1','IL1RN','NLRP12'].map(g => (
            <GeneChip key={g} gene={g} />
          ))}
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#475569' }}>
          320 patients · 8 × 40 · seeds 2542-2549 · FMF · HIDS · TRAPS · CAPS · Blau · PAPA · DIRA · FCAS2
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, padding: '14px 28px 0', borderBottom: '1px solid #334155' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#b71c1c' : 'transparent',
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
              <MetricCard label="Genes Covered" value={overview.n_genes} sub="seeds 2542-2549" />
              <MetricCard label="On Colchicine" value={`${overview.aggregate_metrics?.on_colchicine_pct}%`} sub="FMF/MEFV mainly" />
              <MetricCard label="On Anakinra" value={`${overview.aggregate_metrics?.on_anakinra_pct}%`} sub="IL-1 blockade" />
              <MetricCard label="On Canakinumab" value={`${overview.aggregate_metrics?.on_canakinumab_pct}%`} sub="FDA-approved" />
              <MetricCard label="AA Amyloidosis" value={`${overview.aggregate_metrics?.amyloid_pct}%`} sub="preventable" />
              <MetricCard label="SNHL" value={`${overview.aggregate_metrics?.snhl_pct}%`} sub="esp. Muckle-Wells" />
              <MetricCard label="Uveitis" value={`${overview.aggregate_metrics?.uveitis_pct}%`} sub="esp. Blau/NOD2" />
            </div>

            {/* Disease class grid */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>8 Disease Classes</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(280px,1fr))', gap: 10 }}>
                {overview.disease_classes?.map((cls, i) => {
                  const gene = Object.keys(GENE_COLORS)[i];
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
                      {['Gene','Locus','Inh.','Disease','Attack Dur.','Trigger','Amyloid','Colchicine%','Anakinra%','Canakinumab%'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.gene_summary?.map(g => (
                      <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                        <td style={{ padding: '8px 10px' }}><GeneChip gene={g.gene} /></td>
                        <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                        <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                        <td style={{ padding: '8px 10px', color: '#f1f5f9', maxWidth: 200 }}>{g.disease_name}</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.attack_duration_h}</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.dominant_trigger}</td>
                        <td style={{ padding: '8px 10px', color: g.amyloid_risk.includes('High') ? '#f87171' : '#94a3b8' }}>{g.amyloid_risk.split('—')[0]}</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.on_colchicine_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.on_anakinra_pct}%</td>
                        <td style={{ padding: '8px 10px', color: '#cbd5e1' }}>{g.on_canakinumab_pct}%</td>
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
                  <span style={{ color: '#b71c1c', fontWeight: 800 }}>⬥</span>
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
                  <div style={{ color: '#64748b', fontSize: 11, marginTop: 2 }}>{g.inheritance.split(';')[0]}</div>
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
                      <div style={{ color: '#94a3b8', fontSize: 12 }}>{g.locus} · {g.protein_size} · {g.inheritance.split(';')[0]}</div>
                    </div>
                  </div>

                  {/* Key metrics row */}
                  <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
                    {[
                      { label: 'Patients', value: g.n_patients },
                      { label: 'Avg Peak Temp', value: `${g.avg_peak_temp}°C` },
                      { label: 'Avg CRP (attack)', value: `${g.avg_crp_attack} mg/L` },
                      { label: 'Colchicine', value: `${g.on_colchicine_pct}%` },
                      { label: 'Anakinra', value: `${g.on_anakinra_pct}%` },
                      { label: 'Canakinumab', value: `${g.on_canakinumab_pct}%` },
                      { label: 'Amyloid', value: `${g.amyloid_pct}%` },
                      { label: 'SNHL', value: `${g.snhl_pct}%` },
                      { label: 'Uveitis', value: `${g.uveitis_pct}%` },
                    ].map(m => (
                      <div key={m.label} style={{ background: '#0f172a', borderRadius: 6, padding: '8px 12px', minWidth: 100 }}>
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
                  <div style={{ marginLeft: 'auto', color: '#64748b', fontSize: 11 }}>{g.locus} · {g.inheritance.split(';')[0]}</div>
                </div>
                <div style={{ color: '#fbbf24', fontSize: 12, marginBottom: 8, fontStyle: 'italic' }}>{g.pathognomonic?.substring(0, 160)}…</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 11, color: '#94a3b8' }}>
                  <span>Colchicine: <b style={{ color: '#f1f5f9' }}>{g.on_colchicine_pct}%</b></span>
                  <span>Anakinra: <b style={{ color: '#f1f5f9' }}>{g.on_anakinra_pct}%</b></span>
                  <span>Canakinumab: <b style={{ color: '#f1f5f9' }}>{g.on_canakinumab_pct}%</b></span>
                  <span>Amyloid: <b style={{ color: '#f87171' }}>{g.amyloid_pct}%</b></span>
                  <span>Attack: <b style={{ color: '#f1f5f9' }}>{g.attack_duration_h}</b></span>
                  <span>Trigger: <b style={{ color: '#f1f5f9' }}>{g.dominant_trigger}</b></span>
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
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 11 }}>
                    <div><span style={{ color: '#64748b' }}>Attack:</span> <span style={{ color: '#cbd5e1' }}>{d.attack_duration_h}</span></div>
                    <div><span style={{ color: '#64748b' }}>Trigger:</span> <span style={{ color: '#cbd5e1' }}>{d.dominant_trigger}</span></div>
                    <div><span style={{ color: '#64748b' }}>Amyloid risk:</span> <span style={{ color: '#cbd5e1' }}>{d.amyloid_risk}</span></div>
                    <div><span style={{ color: '#64748b' }}>NBS indicated:</span> <span style={{ color: d.nbs_indicated ? '#4ade80' : '#64748b' }}>{d.nbs_indicated ? 'Yes' : 'No'}</span></div>
                  </div>
                </div>
              ))}
            </div>

            {/* Glossary */}
            <div>
              <div style={{ color: '#94a3b8', fontSize: 12, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>Fever Syndrome Glossary</div>
              {Object.entries(definitions.fever_glossary || {}).map(([term, def]) => (
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
