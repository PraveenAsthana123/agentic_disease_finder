'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-pigmentation-disorder-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TYR:     '#1565c0',  // deep blue    — OCA1, zero/reduced tyrosinase
  OCA2:    '#0d47a1',  // navy         — most common OCA worldwide
  TYRP1:   '#b05c00',  // rufous brown — OCA3, rufous albinism
  SLC45A2: '#2e7d32',  // deep green   — OCA4, Japan most common
  HPS1:    '#b71c1c',  // deep red     — Hermansky-Pudlak, pulmonary fibrosis
  LYST:    '#4a148c',  // deep purple  — Chédiak-Higashi, HLH emergency
  KIT:     '#004d40',  // dark teal    — Piebaldism, stable white patches
  MC1R:    '#e65100',  // burnt orange — Red hair, melanoma predisposition
};

const GENE_INFO = {
  TYR:     { full: 'TYR / 529aa',     locus: '11q14.3', size: '529 aa / 58 kDa',  inh: 'AR',          disease: 'OCA1A (zero tyrosinase — white hair/pink skin/pink eyes LIFELONG; complete melanin absence PATHOGNOMONIC) / OCA1B (residual activity — yellow-blonde hair accrues age 1-2yr); NYSTAGMUS + PHOTOPHOBIA + FOVEAL HYPOPLASIA ocular triad ALL OCA; VEP chiasmal misrouting PATHOGNOMONIC; SPF50 mandatory lifelong (no UV protection from melanin)' },
  OCA2:    { full: 'OCA2 / 838aa',    locus: '15q13.3', size: '838 aa / 110 kDa', inh: 'AR',          disease: 'OCA Type 2 — MOST COMMON OCA WORLDWIDE (1:3,900 sub-Saharan Africa); YELLOW-BLONDE HAIR + PALE CREAM SKIN in dark-skinned populations PATHOGNOMONIC; sky-blue to hazel eye colour (not pink); 15q11-q13 deletion → Prader-Willi/Angelman overlap; SPF50 + annual dermatology; equatorial SCC risk high' },
  TYRP1:   { full: 'TYRP1 / 537aa',   locus: '9p23',    size: '537 aa / 75 kDa',  inh: 'AR',          disease: 'OCA Type 3 (Rufous albinism) — RED-RUFOUS HAIR + BRONZE SKIN + BROWN EYES in Africans PATHOGNOMONIC; PIGMENT REDUCED NOT ABSENT = KEY DDx OCA1A; OCULAR FEATURES MILDER — nystagmus absent/minimal; UNDERDIAGNOSED worldwide; molecular panel mandatory; TYRP1 stabilises TYR; DHICA oxidase activity' },
  SLC45A2: { full: 'SLC45A2 / 530aa', locus: '5p13.2',  size: '530 aa / 58 kDa',  inh: 'AR',          disease: 'OCA Type 4 — MOST COMMON OCA JAPAN (>70% Japanese OCA); PALE CREAM SKIN + NYSTAGMUS PATHOGNOMONIC; variable expressivity (near-normal to OCA1B-like pale); MATP/melanosomal pH regulation; full OCA panel required (missed by TYR/OCA2-only panels); nystagmus universal in OCA4' },
  HPS1:    { full: 'HPS1 / 700aa',    locus: '10q24.2', size: '700 aa / 80 kDa',  inh: 'AR',          disease: 'Hermansky-Pudlak Syndrome Type 1 — OCA + PLATELET DENSE GRANULE DEFICIENCY (absent ADP/serotonin; prolonged bleeding time; NORMAL platelet count = falsely reassuring) + PULMONARY FIBROSIS (UIP pattern, lethal 3rd-4th decade) TRIAD PATHOGNOMONIC; PUERTO RICO FOUNDER (1:1800); NO ASPIRIN/NSAIDs ABSOLUTE; DDAVP pre-op; pirfenidone for ILD' },
  LYST:    { full: 'LYST / 3801aa',   locus: '1q42.3',  size: '3801 aa / 430 kDa', inh: 'AR',         disease: 'Chédiak-Higashi Syndrome — PARTIAL ALBINISM (SILVER-GREY hair metallic sheen) + GIANT PEROXIDASE-POSITIVE GRANULES IN NEUTROPHILS (peripheral smear MPO stain = PATHOGNOMONIC, IMMEDIATE) + NK cytotoxicity absent + recurrent pyogenic infections; ACCELERATED PHASE = HLH (fever+splenomegaly+pancytopenia) = LETHAL; HLH-2004 → HSCT (ONLY CURATIVE)' },
  KIT:     { full: 'KIT / 976aa',     locus: '4q12',    size: '976 aa / 145 kDa', inh: 'AD',          disease: 'Piebaldism — WHITE FORELOCK (triangular frontal scalp patch) + STABLE DEPIGMENTED PATCHES (ventral/abdominal/extremity, islands of pigmented skin within white patches) PATHOGNOMONIC; CONGENITAL STABLE NON-PROGRESSIVE = KEY DDx vitiligo (acquired/progressive); NO systemic features; KIT LOF → melanocyte migration failure embryogenesis; SPF50 depigmented patches' },
  MC1R:    { full: 'MC1R / 317aa',    locus: '16q24.3', size: '317 aa / 35 kDa',  inh: 'AD-incomplete', disease: 'Red Hair Color / Melanoma Predisposition — RED/AUBURN/STRAWBERRY-BLONDE HAIR + FAIR FRECKLED SKIN (phototype I-II) + EPHELIDES PATHOGNOMONIC; R-VARIANTS (Arg151Cys/Arg160Trp/Asp294His) = 4-10x melanoma risk; R/R compound heterozygotes can have BROWN hair yet FULL MELANOMA RISK; SPF50 LIFELONG + ANNUAL TOTAL-BODY DERMOSCOPY from age 18yr' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px', fontSize: 11, fontWeight: 700, marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff', border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10, padding: '14px 18px', minWidth: 120, textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

export default function HeredPigmentAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefinitions(df);
    }).catch(e => setError(String(e))).finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#1565c0', fontWeight: 700 }}>Loading Hereditary Pigmentation Disorder Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1565c0', marginBottom: 4 }}>
          🧬 Hereditary Pigmentation Disorder Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — TYR · OCA2 · TYRP1 · SLC45A2 · HPS1 · LYST · KIT · MC1R —
          320 patients (8 × 40, seeds 2302-2309)
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#1565c0" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2302-2309" color="#e65100" />
        <StatCard label="Melanoma Risk" value={overview.melanoma_risk_genes?.length ?? 5} sub="TYR/OCA2/TYRP1/SLC45A2/HPS1/MC1R" color="#b71c1c" />
        <StatCard label="HLH Emergency" value="1" sub="LYST (CHS accelerated phase)" color="#4a148c" />
        <StatCard label="Stable Leukoderma" value="1" sub="KIT (piebaldism — congenital)" color="#004d40" />
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            padding: '8px 18px', border: 'none', cursor: 'pointer', fontWeight: 700,
            fontSize: 13, borderRadius: '6px 6px 0 0',
            background: tab === i ? '#1565c0' : '#f5f5f5',
            color: tab === i ? '#fff' : '#555',
          }}>{t}</button>
        ))}
      </div>

      {/* Tab 0 — Overview */}
      {tab === 0 && (
        <div>
          {/* Gene colour legend */}
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 18 }}>
            {genes.map(g => (
              <span key={g} style={{
                background: GENE_COLORS[g] + '18', border: `2px solid ${GENE_COLORS[g]}`,
                borderRadius: 6, padding: '4px 10px', fontSize: 12, fontWeight: 700, color: GENE_COLORS[g],
              }}>{g}</span>
            ))}
          </div>

          {/* Category map */}
          <h3 style={{ color: '#333', marginBottom: 10, fontSize: 15 }}>Pigmentation Disorder Categories</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 20 }}>
            {overview.pigment_categories && Object.entries(overview.pigment_categories).map(([cat, desc]) => (
              <div key={cat} style={{ background: '#f8f9fa', border: '1px solid #dee2e6', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1565c0', marginBottom: 4 }}>{cat}</div>
                <div style={{ fontSize: 12, color: '#555' }}>{desc}</div>
              </div>
            ))}
          </div>

          {/* Clinical pearls */}
          <h3 style={{ color: '#333', marginBottom: 10, fontSize: 15 }}>Key Clinical Pearls</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 20 }}>
            {overview.key_clinical_pearls?.map((pearl, i) => {
              const gene = genes[i] || 'TYR';
              return (
                <div key={i} style={{
                  background: GENE_COLORS[gene] + '10', border: `1px solid ${GENE_COLORS[gene]}44`,
                  borderLeft: `4px solid ${GENE_COLORS[gene]}`, borderRadius: 6, padding: 12,
                }}>
                  <span style={{ fontSize: 11, fontWeight: 700, color: GENE_COLORS[gene], marginRight: 8 }}>{gene}</span>
                  <span style={{ fontSize: 12, color: '#333' }}>{pearl}</span>
                </div>
              );
            })}
          </div>

          {/* Diagnostic algorithm */}
          <h3 style={{ color: '#333', marginBottom: 10, fontSize: 15 }}>Diagnostic Algorithm</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {overview.diagnostic_algorithm && Object.entries(overview.diagnostic_algorithm).map(([step, desc]) => (
              <div key={step} style={{
                display: 'flex', gap: 10, background: '#f8f9fa',
                border: '1px solid #dee2e6', borderRadius: 6, padding: 10,
              }}>
                <span style={{ fontWeight: 800, color: '#1565c0', minWidth: 60, fontSize: 12 }}>{step.replace('_', ' ')}</span>
                <span style={{ fontSize: 12, color: '#333' }}>{desc}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Tab 1 — Gene Table */}
      {tab === 1 && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1565c0', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Phenotype', 'Melanoma Risk', 'Bleeding', 'HLH', 'Stable Pigment'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {genes.map((g, idx) => {
                const info = GENE_INFO[g];
                const gd = breakdown?.gene_breakdown?.[g];
                return (
                  <tr key={g} style={{ background: idx % 2 === 0 ? '#fff' : '#f8f9fa', borderBottom: '1px solid #e0e0e0' }}>
                    <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g] }}>{g}</td>
                    <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>{info.locus}</td>
                    <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>{info.size}</td>
                    <td style={{ padding: '8px 10px' }}>
                      <Badge text={info.inh} color={GENE_COLORS[g]} />
                    </td>
                    <td style={{ padding: '8px 10px', maxWidth: 340 }}>{info.disease.substring(0, 160)}…</td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      {gd?.melanoma_risk ? <Badge text="YES" color="#b71c1c" /> : <span style={{ color: '#aaa' }}>—</span>}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      {gd?.bleeding_risk ? <Badge text="YES" color="#e65100" /> : <span style={{ color: '#aaa' }}>—</span>}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      {gd?.hlh_risk ? <Badge text="HLH" color="#4a148c" /> : <span style={{ color: '#aaa' }}>—</span>}
                    </td>
                    <td style={{ padding: '8px 10px', textAlign: 'center' }}>
                      {gd?.stable_pigment ? <Badge text="STABLE" color="#004d40" /> : <span style={{ color: '#aaa' }}>—</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Tab 2 — Clinical Atlas */}
      {tab === 2 && breakdown && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {/* Emergency flags */}
          {breakdown.clinical_emergency_flags?.length > 0 && (
            <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 10, padding: 16, marginBottom: 8 }}>
              <div style={{ fontWeight: 800, color: '#b71c1c', marginBottom: 10, fontSize: 14 }}>⚠ Clinical Emergency Flags</div>
              {breakdown.clinical_emergency_flags.map((flag, i) => (
                <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 6, paddingLeft: 10, borderLeft: '3px solid #e65100' }}>
                  {flag}
                </div>
              ))}
            </div>
          )}

          {/* Per-gene cards */}
          {genes.map(g => {
            const gd = breakdown.gene_breakdown?.[g];
            if (!gd) return null;
            return (
              <div key={g} style={{
                border: `2px solid ${GENE_COLORS[g]}44`, borderLeft: `5px solid ${GENE_COLORS[g]}`,
                borderRadius: 10, padding: 16, background: GENE_COLORS[g] + '08',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                  <div>
                    <span style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[g] }}>{g}</span>
                    <span style={{ fontSize: 12, color: '#666', marginLeft: 10 }}>{gd.locus} · {gd.protein_size} · {gd.inheritance}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {gd.melanoma_risk && <Badge text="MELANOMA RISK" color="#b71c1c" />}
                    {gd.bleeding_risk && <Badge text="BLEEDING RISK" color="#e65100" />}
                    {gd.hlh_risk && <Badge text="HLH EMERGENCY" color="#4a148c" />}
                    {gd.stable_pigment && <Badge text="STABLE PIGMENT" color="#004d40" />}
                  </div>
                </div>

                <div style={{ fontSize: 12, color: '#333', marginBottom: 10, fontStyle: 'italic' }}>
                  {gd.pathognomonic}
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color: '#555', marginBottom: 4 }}>Key Features</div>
                    {gd.key_features?.map((f, i) => (
                      <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 2 }}>• {f}</div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color: '#555', marginBottom: 4 }}>Monitoring</div>
                    {gd.monitoring?.map((m, i) => (
                      <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 2 }}>• {m}</div>
                    ))}
                  </div>
                </div>

                <div style={{ marginTop: 10 }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#555', marginBottom: 4 }}>Key DDx</div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {gd.key_ddx?.map((d, i) => (
                      <span key={i} style={{
                        background: '#f5f5f5', border: '1px solid #ddd',
                        borderRadius: 4, padding: '2px 8px', fontSize: 11, color: '#333',
                      }}>{d.substring(0, 80)}</span>
                    ))}
                  </div>
                </div>

                <div style={{ marginTop: 10 }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#555', marginBottom: 4 }}>Treatment Highlights</div>
                  <div style={{ fontSize: 11, color: '#333' }}>{gd.treatment_highlight}</div>
                </div>

                <div style={{ marginTop: 8, display: 'flex', gap: 16 }}>
                  <span style={{ fontSize: 11, color: '#666' }}>n={gd.n_patients} · avg dx {gd.avg_age_at_dx_yrs}yr · avg f/u {gd.avg_follow_up_yrs}yr</span>
                  {gd.melanoma_event_pct > 0 && <span style={{ fontSize: 11, color: '#b71c1c' }}>Melanoma events: {gd.melanoma_event_pct}%</span>}
                  {gd.bleeding_event_pct > 0 && <span style={{ fontSize: 11, color: '#e65100' }}>Bleeding events: {gd.bleeding_event_pct}%</span>}
                  {gd.hlh_event_pct > 0 && <span style={{ fontSize: 11, color: '#4a148c' }}>HLH events: {gd.hlh_event_pct}%</span>}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Tab 3 — Definitions */}
      {tab === 3 && definitions && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Gene entries */}
          <h3 style={{ color: '#333', fontSize: 15, marginBottom: 6 }}>Gene Reference</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {genes.map(g => {
              const entry = definitions.gene_entries?.[g];
              if (!entry) return null;
              return (
                <div key={g} style={{
                  border: `1px solid ${GENE_COLORS[g]}44`, borderLeft: `4px solid ${GENE_COLORS[g]}`,
                  borderRadius: 6, padding: 12, background: GENE_COLORS[g] + '08',
                }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[g], marginBottom: 4 }}>{g}</div>
                  <div style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>
                    {entry.locus} · {entry.protein_size} · {entry.inheritance?.split(';')[0]}
                  </div>
                  <div style={{ fontSize: 12, color: '#333' }}><strong>Pathognomonic:</strong> {entry.pathognomonic}</div>
                </div>
              );
            })}
          </div>

          {/* Biology glossary */}
          <h3 style={{ color: '#333', fontSize: 15, marginBottom: 6 }}>Pigmentation Biology Glossary</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {definitions.pigment_biology_glossary && Object.entries(definitions.pigment_biology_glossary).map(([term, def]) => (
              <div key={term} style={{ background: '#f8f9fa', border: '1px solid #dee2e6', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1565c0', marginBottom: 4 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#555' }}>{def}</div>
              </div>
            ))}
          </div>

          {/* Treatment glossary */}
          <h3 style={{ color: '#333', fontSize: 15, marginBottom: 6 }}>Treatment Glossary</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {definitions.treatment_glossary && Object.entries(definitions.treatment_glossary).map(([term, def]) => (
              <div key={term} style={{ background: '#fff3e0', border: '1px solid #ffe0b2', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#e65100', marginBottom: 4 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#555' }}>{def}</div>
              </div>
            ))}
          </div>

          {/* Diagnostic tests */}
          <h3 style={{ color: '#333', fontSize: 15, marginBottom: 6 }}>Diagnostic Tests</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {definitions.diagnostic_tests && Object.entries(definitions.diagnostic_tests).map(([test, desc]) => (
              <div key={test} style={{ background: '#e8f5e9', border: '1px solid #c8e6c9', borderRadius: 6, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#2e7d32', marginBottom: 4 }}>{test}</div>
                <div style={{ fontSize: 12, color: '#555' }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
