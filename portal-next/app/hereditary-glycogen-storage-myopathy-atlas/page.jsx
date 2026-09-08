'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-glycogen-storage-myopathy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  GAA:   '#1565c0',  // deep blue    — Pompe, ERT-treatable, massive cardiomegaly infantile
  PYGM:  '#b71c1c',  // deep red     — McArdle, second wind, most pathognomonic
  PFKM:  '#e65100',  // deep orange  — Tarui, out-of-wind, hemolysis + gout
  LAMP2: '#880e4f',  // deep pink    — Danon, X-linked dominant, WPW + HCM, transplant
  AGL:   '#2e7d32',  // deep green   — Cori-Forbes, liver childhood + muscle adult
  GBE1:  '#4a148c',  // deep purple  — Andersen, most severe GSD, hepatic cirrhosis
  GYS1:  '#f57f17',  // amber        — GSD-0a, paradoxical low glycogen, SCD arrhythmia
  PGAM2: '#00695c',  // teal         — GSD-X, tubular aggregates, African American founder
};

const GENE_INFO = {
  GAA:   { full: 'GAA / 762aa',    locus: '17q25.3', size: '762 aa / 110 kDa', inh: 'AR',           disease: 'Pompe / GSD-II — CLASSIC INFANTILE: cardiomegaly+hypotonia+short PR PATHOGNOMONIC / LOPD: proximal myopathy+respiratory / ERT: alglucosidase→avalglucosidase→cipaglucosidase / CRIM-NEGATIVE: ITI mandatory before ERT' },
  PYGM:  { full: 'PYGM / 842aa',   locus: '11q13.1', size: '842 aa / 97 kDa',  inh: 'AR',           disease: 'McArdle / GSD-V — SECOND WIND PHENOMENON most pathognomonic in myopathology / NO LACTATE RISE forearm test / CK 10-100× baseline / p.W797X 65% European alleles / aerobic exercise improves' },
  PFKM:  { full: 'PFKM / 780aa',   locus: '12q13.11',size: '780 aa / 85 kDa',  inh: 'AR',           disease: 'Tarui / GSD-VII — OUT-OF-WIND carbs WORSEN (opposite McArdle) / HEMOLYTIC ANEMIA + GOUT + myopathy TRIAD / Ashkenazi founder p.Arg370Ter / NO LACTATE RISE forearm test' },
  LAMP2: { full: 'LAMP2 / 410aa',  locus: 'Xq24',    size: '410 aa / 45 kDa',  inh: 'X-linked Dom', disease: 'Danon — TRIAD males: MASSIVE HCM + myopathy + intellectual disability PATHOGNOMONIC / WPW pre-excitation + HCM ECG/echo PATHOGNOMONIC / MANDATORY ICD / transplant often <30y / retinitis 70% males' },
  AGL:   { full: 'AGL / 1532aa',   locus: '1p21.2',  size: '1532 aa / 175 kDa',inh: 'AR',           disease: 'Cori-Forbes / GSD-III — IIIa liver+muscle (85%) vs IIIb liver-only (15%) / LIVER childhood then MUSCLE adults / NO CIRRHOSIS key DDx / liver IMPROVES puberty / raw cornstarch standard' },
  GBE1:  { full: 'GBE1 / 702aa',   locus: '3p24.2',  size: '702 aa / 80 kDa',  inh: 'AR',           disease: 'Andersen / GSD-IV — MOST SEVERE GSD / PAS+ POLYGLUCOSAN BODIES biopsy PATHOGNOMONIC / classic infantile cirrhosis FATAL <5y without LT / APBD: adult UMN+LMN+dementia' },
  GYS1:  { full: 'GYS1 / 700aa',   locus: '19q13.33',size: '700 aa / 81 kDa',  inh: 'AR',           disease: 'GSD-0a — EXERCISE-INDUCED VENTRICULAR ARRHYTHMIA + SCD PATHOGNOMONIC / PARADOXICAL: LOW glycogen (cannot synthesize) / HCM mandatory / ICD mandatory / ABSOLUTE exercise restriction' },
  PGAM2: { full: 'PGAM2 / 254aa',  locus: '7p13',    size: '254 aa / 29 kDa',  inh: 'AR',           disease: 'GSD-X — TUBULAR AGGREGATES biopsy PATHOGNOMONIC / exercise intolerance + MYOGLOBINURIA / African American founder p.W78X / NO LACTATE RISE / no cardiac involvement' },
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

function GeneCard({ gene, color, info, data }) {
  return (
    <div style={{
      border: `2px solid ${color}`, borderRadius: 10, padding: 16, marginBottom: 12,
      background: color + '08',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{ fontWeight: 800, fontSize: 18, color }}>{gene}</span>
        <Badge text={info.locus} color={color} />
        <Badge text={info.inh} color={color} />
        <Badge text={info.size} color="#555" />
      </div>
      <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          {data.avg_onset !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Avg Onset: <b>{data.avg_onset}yr</b></span>}
          {data.avg_ck !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Avg CK: <b>{data.avg_ck} IU/L</b></span>}
          {data.cardiac_pct !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Cardiac HCM: <b>{data.cardiac_pct}%</b></span>}
          {data.myoglobinuria_pct !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Myoglobinuria: <b>{data.myoglobinuria_pct}%</b></span>}
        </div>
      )}
    </div>
  );
}

export default function HGSMAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefs(df);
    }).catch(e => setError(e.message)).finally(() => setLoading(false));
  }, []);

  const TITLE = 'Hereditary Glycogen Storage Myopathy Atlas';
  const SUBTITLE = 'Complete 8-Gene GSD Muscle Spectrum — GAA · PYGM · PFKM · LAMP2 · AGL · GBE1 · GYS1 · PGAM2';

  return (
    <div style={{ fontFamily: 'Inter,sans-serif', background: '#f5f5f5', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#4a148c 100%)', color: '#fff', padding: '28px 32px 18px' }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>🧬 {TITLE}</h1>
        <div style={{ fontSize: 13, opacity: 0.85, marginTop: 6 }}>{SUBTITLE}</div>
        <div style={{ fontSize: 12, opacity: 0.65, marginTop: 4 }}>320-patient aggregate cohort · 8 genes · Seeds 2246-2253 · 2026-09-08</div>
        {/* Tab bar */}
        <div style={{ display: 'flex', gap: 6, marginTop: 18 }}>
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} style={{
              background: tab === i ? '#fff' : 'rgba(255,255,255,0.15)',
              color: tab === i ? '#1a237e' : '#fff', border: 'none',
              borderRadius: 6, padding: '6px 16px', cursor: 'pointer', fontWeight: 600, fontSize: 13,
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: '24px 32px' }}>
        {loading && <div style={{ color: '#666', padding: 40, textAlign: 'center' }}>Loading atlas data…</div>}
        {error && <div style={{ color: '#c00', padding: 20, background: '#fff3f3', borderRadius: 8 }}>Error: {error}</div>}

        {/* TAB 0: OVERVIEW */}
        {tab === 0 && overview && (
          <div>
            <h2 style={{ fontSize: 18, color: '#1a237e', marginBottom: 16 }}>GSD Muscle Atlas — KPIs</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 28 }}>
              <StatCard label="Total Patients" value={overview.kpis.total_patients} color="#1a237e" />
              <StatCard label="Genes Covered" value={overview.kpis.genes_covered} color="#4a148c" />
              <StatCard label="Cardiac HCM" value={`${overview.kpis.cardiac_hcm_pct}%`} color="#880e4f" sub="LAMP2 + GYS1 dominant" />
              <StatCard label="Myoglobinuria" value={`${overview.kpis.myoglobinuria_pct}%`} color="#b71c1c" sub="PYGM + PFKM + PGAM2" />
              <StatCard label="Arrhythmia" value={`${overview.kpis.arrhythmia_pct}%`} color="#f57f17" sub="GYS1 VT/VF" />
              <StatCard label="ICD Implanted" value={`${overview.kpis.icd_pct}%`} color="#e65100" sub="LAMP2 + GYS1" />
              <StatCard label="On ERT" value={`${overview.kpis.on_ert_pct}%`} color="#1565c0" sub="GAA (Pompe)" />
              <StatCard label="Second Wind" value={`${overview.kpis.second_wind_pct}%`} color="#2e7d32" sub="PYGM McArdle" />
              <StatCard label="Hemolysis" value={`${overview.kpis.hemolytic_anemia_pct}%`} color="#e65100" sub="PFKM Tarui" />
              <StatCard label="Avg CK" value={`${overview.kpis.avg_ck_iul}`} sub="IU/L cohort avg" color="#4a148c" />
            </div>

            {/* Gene overview cards */}
            <h2 style={{ fontSize: 18, color: '#1a237e', marginBottom: 14 }}>Gene Profiles</h2>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={info} />
            ))}

            {/* Pathognomonic features */}
            <h2 style={{ fontSize: 16, color: '#1a237e', marginTop: 28, marginBottom: 12 }}>Pathognomonic Features by Gene</h2>
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, border: '1px solid #e0e0e0' }}>
              {overview.pathognomonic_features && Object.entries(overview.pathognomonic_features).map(([gene, feat]) => (
                <div key={gene} style={{ display: 'flex', gap: 14, marginBottom: 10, alignItems: 'flex-start' }}>
                  <span style={{
                    background: GENE_COLORS[gene] + '22', color: GENE_COLORS[gene],
                    borderRadius: 5, padding: '2px 10px', fontWeight: 800, fontSize: 13, minWidth: 60, textAlign: 'center',
                  }}>{gene}</span>
                  <span style={{ fontSize: 13, color: '#333', lineHeight: 1.5 }}>{feat}</span>
                </div>
              ))}
            </div>

            {/* Key distinctions */}
            {overview.key_distinctions && (
              <>
                <h2 style={{ fontSize: 16, color: '#1a237e', marginTop: 24, marginBottom: 10 }}>Key Clinical Distinctions</h2>
                <div style={{ background: '#fff', borderRadius: 10, padding: 20, border: '1px solid #e0e0e0' }}>
                  {overview.key_distinctions.map((d, i) => (
                    <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
                      <span style={{ color: '#1a237e', fontWeight: 700, minWidth: 20 }}>▸</span>
                      <span style={{ fontSize: 13, color: '#333' }}>{d}</span>
                    </div>
                  ))}
                </div>
              </>
            )}

            {/* Critical treatments */}
            {overview.critical_treatments && (
              <>
                <h2 style={{ fontSize: 16, color: '#1a237e', marginTop: 24, marginBottom: 10 }}>Critical Treatments by Gene</h2>
                <div style={{ background: '#fff', borderRadius: 10, padding: 20, border: '1px solid #e0e0e0' }}>
                  {Object.entries(overview.critical_treatments).map(([gene, tx]) => (
                    <div key={gene} style={{ display: 'flex', gap: 14, marginBottom: 10 }}>
                      <span style={{
                        background: GENE_COLORS[gene] + '22', color: GENE_COLORS[gene],
                        borderRadius: 5, padding: '2px 10px', fontWeight: 800, fontSize: 12, minWidth: 60, textAlign: 'center',
                      }}>{gene}</span>
                      <span style={{ fontSize: 13, color: '#444' }}>{tx}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* TAB 1: GENE TABLE */}
        {tab === 1 && breakdown && (
          <div>
            <h2 style={{ fontSize: 18, color: '#1a237e', marginBottom: 16 }}>Gene Reference Table</h2>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', background: '#fff', borderRadius: 10, overflow: 'hidden', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1a237e', color: '#fff' }}>
                    {['Gene', 'Disease', 'Locus', 'Protein', 'Inheritance', 'Pathognomonic', 'Key Treatment'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.gene_profiles && breakdown.gene_profiles.map((gp, i) => {
                    const info = GENE_INFO[gp.gene];
                    const color = GENE_COLORS[gp.gene];
                    return (
                      <tr key={gp.gene} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #eee' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 800, color }}>{gp.gene}</td>
                        <td style={{ padding: '10px 12px' }}>
                          {gp.gene === 'GAA' ? 'Pompe / GSD-II' :
                           gp.gene === 'PYGM' ? 'McArdle / GSD-V' :
                           gp.gene === 'PFKM' ? 'Tarui / GSD-VII' :
                           gp.gene === 'LAMP2' ? 'Danon Disease' :
                           gp.gene === 'AGL' ? 'Cori-Forbes / GSD-III' :
                           gp.gene === 'GBE1' ? 'Andersen / GSD-IV' :
                           gp.gene === 'GYS1' ? 'GSD-0a' : 'GSD-X'}
                        </td>
                        <td style={{ padding: '10px 12px' }}>{gp.locus}</td>
                        <td style={{ padding: '10px 12px' }}>{gp.protein_size}</td>
                        <td style={{ padding: '10px 12px' }}>{gp.gene === 'LAMP2' ? 'X-linked Dom' : 'AR'}</td>
                        <td style={{ padding: '10px 12px', fontSize: 11, maxWidth: 180 }}>
                          {gp.key_features && gp.key_features[0]}
                        </td>
                        <td style={{ padding: '10px 12px', fontSize: 11, maxWidth: 160 }}>
                          {gp.treatment && gp.treatment.substring(0, 120)}…
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Monitoring tables */}
            <h2 style={{ fontSize: 18, color: '#1a237e', marginTop: 28, marginBottom: 16 }}>Monitoring Protocols</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 16 }}>
              {breakdown.gene_profiles && breakdown.gene_profiles.map(gp => (
                <div key={gp.gene} style={{
                  background: '#fff', borderRadius: 10, padding: 16,
                  border: `2px solid ${GENE_COLORS[gp.gene]}44`,
                }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[gp.gene], fontSize: 15, marginBottom: 10 }}>
                    {gp.gene} — Monitoring
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 18, listStyle: 'disc' }}>
                    {gp.monitoring && gp.monitoring.map((m, i) => (
                      <li key={i} style={{ fontSize: 12, color: '#333', marginBottom: 5 }}>{m}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 2: CLINICAL ATLAS */}
        {tab === 2 && breakdown && (
          <div>
            <h2 style={{ fontSize: 18, color: '#1a237e', marginBottom: 16 }}>
              Clinical Atlas — {breakdown.total} Patients
            </h2>
            {Object.keys(GENE_COLORS).map(gene => {
              const pts = (breakdown.patients || []).filter(p => p.gene === gene);
              const color = GENE_COLORS[gene];
              if (!pts.length) return null;
              const avgOnset = (pts.reduce((s, p) => s + (p.onset_years || 0), 0) / pts.length).toFixed(1);
              const avgCK = Math.round(pts.reduce((s, p) => s + (p.ck_iul || 0), 0) / pts.length);
              const cardiacPct = Math.round(100 * pts.filter(p => p.cardiac_hcm).length / pts.length);
              const myoPct = Math.round(100 * pts.filter(p => p.myoglobinuria).length / pts.length);
              return (
                <div key={gene} style={{
                  background: '#fff', borderRadius: 12, padding: 20, marginBottom: 18,
                  border: `2px solid ${color}44`,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 14 }}>
                    <span style={{
                      background: color, color: '#fff', borderRadius: 8,
                      padding: '6px 18px', fontWeight: 800, fontSize: 16,
                    }}>{gene}</span>
                    <div>
                      <div style={{ fontWeight: 700, color: '#222', fontSize: 14 }}>
                        {GENE_INFO[gene]?.full} · {GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.inh}
                      </div>
                      <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>{pts.length} patients</div>
                    </div>
                    <div style={{ display: 'flex', gap: 14, marginLeft: 'auto', flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 12, color: '#555' }}>Avg Onset: <b>{avgOnset}yr</b></span>
                      <span style={{ fontSize: 12, color: '#555' }}>Avg CK: <b>{avgCK} IU/L</b></span>
                      {cardiacPct > 0 && <span style={{ fontSize: 12, color: '#880e4f' }}>Cardiac HCM: <b>{cardiacPct}%</b></span>}
                      {myoPct > 0 && <span style={{ fontSize: 12, color: '#b71c1c' }}>Myoglobinuria: <b>{myoPct}%</b></span>}
                    </div>
                  </div>
                  {/* Mini patient grid */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {pts.map(p => (
                      <div key={p.id} title={`ID: ${p.id} | Onset: ${p.onset_years}yr | CK: ${p.ck_iul}`}
                        style={{
                          background: p.cardiac_hcm ? color + '33' : '#f5f5f5',
                          border: `1px solid ${color}55`, borderRadius: 6,
                          padding: '4px 8px', fontSize: 10, cursor: 'default',
                          fontWeight: p.myoglobinuria ? 700 : 400,
                        }}>
                        {p.gene}-{String(p.id).split('-').pop()}
                        {p.second_wind ? ' ⚡' : ''}
                        {p.cardiac_hcm ? ' ❤' : ''}
                        {p.myoglobinuria ? ' 🔴' : ''}
                        {p.icd_implanted ? ' ⚡ICD' : ''}
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* TAB 3: DEFINITIONS */}
        {tab === 3 && defs && (
          <div>
            <h2 style={{ fontSize: 18, color: '#1a237e', marginBottom: 16 }}>Definitions & Diagnostic Algorithm</h2>

            {/* Glossary */}
            <div style={{ background: '#fff', borderRadius: 12, padding: 24, marginBottom: 24, border: '1px solid #e0e0e0' }}>
              <h3 style={{ color: '#1a237e', marginTop: 0, marginBottom: 16 }}>Glossary</h3>
              {defs.glossary && Object.entries(defs.glossary).map(([term, def]) => (
                <div key={term} style={{ marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid #f0f0f0' }}>
                  <div style={{ fontWeight: 700, color: '#333', fontSize: 13, marginBottom: 4 }}>{term}</div>
                  <div style={{ fontSize: 12, color: '#555', lineHeight: 1.7 }}>{def}</div>
                </div>
              ))}
            </div>

            {/* Diagnostic algorithm */}
            {defs.diagnostic_algorithm && (
              <div style={{ background: '#fff', borderRadius: 12, padding: 24, marginBottom: 24, border: '1px solid #e0e0e0' }}>
                <h3 style={{ color: '#1a237e', marginTop: 0, marginBottom: 14 }}>Diagnostic Algorithm</h3>
                {defs.diagnostic_algorithm.map((step, i) => (
                  <div key={i} style={{ display: 'flex', gap: 12, marginBottom: 10 }}>
                    <span style={{
                      background: '#1a237e', color: '#fff', borderRadius: 4,
                      padding: '2px 8px', fontSize: 11, fontWeight: 700, minWidth: 28, textAlign: 'center',
                    }}>{i + 1}</span>
                    <span style={{ fontSize: 13, color: '#333', lineHeight: 1.5 }}>{step.replace(/^\d+\.\s*/, '')}</span>
                  </div>
                ))}
              </div>
            )}

            {/* References */}
            {defs.references && (
              <div style={{ background: '#fff', borderRadius: 12, padding: 20, marginBottom: 20, border: '1px solid #e0e0e0' }}>
                <h3 style={{ color: '#1a237e', marginTop: 0, marginBottom: 12 }}>Key References</h3>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {defs.references.map((r, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>{r}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Standards */}
            {defs.standards && (
              <div style={{ background: '#fff', borderRadius: 12, padding: 20, border: '1px solid #e0e0e0' }}>
                <h3 style={{ color: '#1a237e', marginTop: 0, marginBottom: 12 }}>Clinical Standards</h3>
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {defs.standards.map((s, i) => (
                    <li key={i} style={{ fontSize: 12, color: '#555', marginBottom: 6 }}>{s}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
