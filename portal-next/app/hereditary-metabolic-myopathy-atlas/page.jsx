'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-metabolic-myopathy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PYGM:    '#1565c0',  // deep blue    — McArdle, most common metabolic myopathy
  CPT2:    '#b71c1c',  // deep red     — thermogenic triggers, rhabdomyolysis
  ACADVL:  '#2e7d32',  // deep green   — NBS detected, neonatal cardiomyopathy
  ETFDH:   '#6a1b9a',  // deep purple  — RIBOFLAVIN RESPONSIVE essentially curative
  HADHA:   '#e65100',  // deep orange  — LCHAD retinopathy + neuropathy, AFLP
  PFKM:    '#00695c',  // teal         — Tarui, HIGH-CARB PARADOX, hemolytic anemia
  AMPD1:   '#4a148c',  // dark purple  — ammonia fails forearm test, benign
  SLC22A5: '#880e4f',  // deep pink    — carnitine deficiency, curative with L-carnitine
};

const GENE_INFO = {
  PYGM:    { full: 'PYGM / 842aa',     locus: '11q13.1',  size: '842 aa / 97 kDa',   inh: 'AR', disease: 'McArdle Disease (GSD V) — SECOND WIND PHENOMENON PATHOGNOMONIC / forearm ischemic test: lactate FAILS TO RISE / ammonia rises normally / p.Arg50Stop 90% European / CK 1000–50000x crisis / glucose/sucrose trick 10-15g pre-exercise' },
  CPT2:    { full: 'CPT2 / 658aa',     locus: '1p32.3',   size: '658 aa / 74 kDa',   inh: 'AR', disease: 'CPT2 Deficiency — THERMOGENIC TRIGGERS PATHOGNOMONIC (fever/fasting/cold/prolonged exercise → rhabdomyolysis) / p.Ser113Leu mild common allele / HIGH-CARB DIET beneficial / triheptanoin FDA approved / L-carnitine limited benefit' },
  ACADVL:  { full: 'ACADVL / 655aa',   locus: '17p13.1',  size: '655 aa / 70 kDa',   inh: 'AR', disease: 'VLCAD Deficiency — NBS detected C14:1 acylcarnitine ELEVATED / neonatal cardiomyopathy severe form / mild adult-onset rhabdomyolysis / avoid fasting / MCT + triheptanoin / L-carnitine adjunct' },
  ETFDH:   { full: 'ETFDH / 617aa',    locus: '4q32.1',   size: '617 aa / 68 kDa',   inh: 'AR', disease: 'MADD/GA2 — RIBOFLAVIN 100-300mg/day ESSENTIALLY CURATIVE in riboflavin-responsive variant (80%) / FAD cofactor loss restored by riboflavin / CoQ10 adjunctive / avoid fasting / trial MANDATORY before declaring refractory' },
  HADHA:   { full: 'HADHA / 763aa',    locus: '2p23.3',   size: '763 aa / 79 kDa',   inh: 'AR', disease: 'LCHAD/TFP Deficiency — PERIPHERAL RETINOPATHY + PERIPHERAL NEUROPATHY PATHOGNOMONIC combination / AFLP acute fatty liver of pregnancy maternal heterozygote / G1528C p.Glu510Gln common / MCT + avoid LCT / ophthalmology 6-monthly' },
  PFKM:    { full: 'PFKM / 780aa',     locus: '12q13.3',  size: '780 aa / 85 kDa',   inh: 'AR', disease: 'Tarui Disease (GSD VII) — HEMOLYTIC ANEMIA + EXERCISE MYOPATHY UNIQUE / HIGH-CARB PARADOX: IV glucose ABSOLUTELY CONTRAINDICATED in crisis (blocks FFAs) / give fat + protein instead / p.Arg232His Ashkenazi founder' },
  AMPD1:   { full: 'AMPD1 / 747aa',    locus: '1p13.3',   size: '747 aa / 88 kDa',   inh: 'AR', disease: 'Myoadenylate Deaminase Deficiency — FOREARM TEST: AMMONIA FAILS TO RISE / lactate rises normally (OPPOSITE of McArdle) / p.Gln12Stop 34CT 2% European carrier / post-exercise myalgia / generally benign / no specific treatment' },
  SLC22A5: { full: 'SLC22A5 / 557aa',  locus: '5q31.1',   size: '557 aa / 63 kDa',   inh: 'AR', disease: 'Primary Carnitine Deficiency (CDSP) — CARNITINE SUPPLEMENTATION ESSENTIALLY CURATIVE / L-carnitine 100mg/kg/day PO lifelong / cardiomyopathy reversible with treatment / FATAL WITHOUT treatment / NBS detected / low plasma free carnitine <5μmol/L' },
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
          <span style={{ fontSize: 12, color: '#555' }}>Ambulant: <b>{data.ambulant_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Vent: <b>{data.ventilator_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Cardiac: <b>{data.cardiac_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Avg Onset: <b>{data.avg_onset}yr</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Avg CK: <b>{data.avg_ck_peak?.toLocaleString()} IU/L</b></span>
        </div>
      )}
    </div>
  );
}

export default function HMetabolicMyopathyAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setErr(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary Metabolic Myopathy Atlas…</div>;
  if (err) return <div style={{ padding: 40, color: 'red' }}>Error: {err}</div>;
  if (!overview) return null;

  const genes = overview.genes || [];
  const geneSum = overview.gene_summary || {};
  const pharma = overview.key_pharmacological_distinctions || [];
  const alerts = overview.critical_treatment_alerts || [];
  const pathoFeatures = overview.pathognomonic_features || [];

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: '24px 32px', maxWidth: 1200, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, color: '#1a237e', marginBottom: 4 }}>
          🧬 Hereditary Metabolic Myopathy Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Exercise Intolerance &amp; Rhabdomyolysis Spectrum Atlas — PYGM · CPT2 · ACADVL · ETFDH · HADHA · PFKM · AMPD1 · SLC22A5
        </div>
        <div style={{ fontSize: 12, color: '#777', marginTop: 4 }}>
          320-patient aggregate cohort (8 × 40, seeds 2230–2237)
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 20px', border: 'none', background: tab === i ? '#1a237e' : '#f5f5f5',
            color: tab === i ? '#fff' : '#333', fontWeight: 700, borderRadius: '6px 6px 0 0',
            cursor: 'pointer', fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {/* ── TAB 0: Overview ── */}
      {tab === 0 && (
        <div>
          {/* KPI row */}
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
            <StatCard label="Total Patients" value={overview.total_patients} color="#1a237e" />
            <StatCard label="Genes" value={overview.gene_count || 8} color="#1565c0" />
            <StatCard label="Seeds" value="2230–2237" sub="reproducible" color="#283593" />
            {Object.entries(overview.kpis || {}).slice(0, 5).map(([k, v]) => (
              <StatCard key={k} label={k.replace(/_/g, ' ')} value={typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(1)) : v} color="#37474f" />
            ))}
          </div>

          {/* Critical alerts */}
          {alerts.length > 0 && (
            <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 10, padding: 16, marginBottom: 20 }}>
              <div style={{ fontWeight: 800, color: '#e65100', marginBottom: 8 }}>🚨 Critical Treatment Alerts</div>
              {alerts.map((a, i) => <div key={i} style={{ fontSize: 13, color: '#bf360c', marginBottom: 4 }}>• {a}</div>)}
            </div>
          )}

          {/* Gene summary cards */}
          <h3 style={{ color: '#1a237e', marginBottom: 12 }}>8-Gene Summary</h3>
          {genes.map(g => (
            <GeneCard key={g} gene={g} color={GENE_COLORS[g] || '#555'} info={GENE_INFO[g] || {}} data={geneSum[g]} />
          ))}

          {/* Key pharmacological distinctions */}
          <h3 style={{ color: '#1a237e', marginTop: 24, marginBottom: 12 }}>Key Pharmacological Distinctions</h3>
          <div style={{ background: '#e8f5e9', border: '1px solid #2e7d32', borderRadius: 10, padding: 16 }}>
            {pharma.map((p, i) => <div key={i} style={{ fontSize: 13, color: '#1b5e20', marginBottom: 6 }}>• {p}</div>)}
          </div>

          {/* Pathognomonic features */}
          <h3 style={{ color: '#1a237e', marginTop: 24, marginBottom: 12 }}>Pathognomonic Features</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {pathoFeatures.map((f, i) => (
              <div key={i} style={{
                background: '#e3f2fd', border: '1px solid #1565c0', borderRadius: 8,
                padding: '8px 14px', fontSize: 12, color: '#0d47a1', maxWidth: 340,
              }}>{f}</div>
            ))}
          </div>
        </div>
      )}

      {/* ── TAB 1: Gene Table ── */}
      {tab === 1 && (
        <div>
          <h3 style={{ color: '#1a237e', marginBottom: 16 }}>Gene Reference Table</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  {['Gene', 'Protein Size', 'Locus', 'Inh.', 'Disease / Key Feature',
                    'Ambulant %', 'Cardiac %', 'Avg CK (IU/L)'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {genes.map((g, i) => {
                  const info = GENE_INFO[g] || {};
                  const gs = geneSum[g] || {};
                  return (
                    <tr key={g} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 800, color: GENE_COLORS[g] || '#333' }}>{g}</td>
                      <td style={{ padding: '10px 12px' }}>{info.size}</td>
                      <td style={{ padding: '10px 12px' }}>{info.locus}</td>
                      <td style={{ padding: '10px 12px' }}><Badge text={info.inh} color={GENE_COLORS[g] || '#555'} /></td>
                      <td style={{ padding: '10px 12px', fontSize: 11, maxWidth: 260 }}>{(info.disease || '').substring(0, 100)}…</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{gs.ambulant_pct}%</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{gs.cardiac_pct}%</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{gs.avg_ck_peak?.toLocaleString()}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Protein sizes chart */}
          <h3 style={{ color: '#1a237e', marginTop: 24, marginBottom: 12 }}>Protein Size Comparison</h3>
          {genes.map(g => {
            const sz = parseInt((GENE_INFO[g]?.size || '').split(' ')[0]) || 0;
            const maxSz = 842;
            return (
              <div key={g} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[g], width: 90 }}>{g}</span>
                <div style={{ flex: 1, background: '#e0e0e0', borderRadius: 4, height: 18, overflow: 'hidden' }}>
                  <div style={{ width: `${Math.round(sz / maxSz * 100)}%`, background: GENE_COLORS[g], height: '100%', borderRadius: 4 }} />
                </div>
                <span style={{ fontSize: 12, color: '#555', width: 90 }}>{GENE_INFO[g]?.size}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* ── TAB 2: Clinical Atlas ── */}
      {tab === 2 && breakdown && (
        <div>
          <h3 style={{ color: '#1a237e', marginBottom: 16 }}>Clinical Atlas — 320 Patients</h3>
          <div style={{ overflowX: 'auto', maxHeight: 600 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead style={{ position: 'sticky', top: 0, zIndex: 1 }}>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  {['ID', 'Gene', 'Sex', 'Age', 'Onset', 'CK Peak', 'Ambulant', 'Cardiac', 'Rhabdo #', 'Key Trigger', 'Flags'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 700 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.patients || []).slice(0, 120).map((p, i) => (
                  <tr key={p.id} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '6px 10px', fontSize: 10, color: '#888' }}>{p.id}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[p.gene] || '#333' }}>{p.gene}</td>
                    <td style={{ padding: '6px 10px' }}>{p.sex}</td>
                    <td style={{ padding: '6px 10px' }}>{p.age}yr</td>
                    <td style={{ padding: '6px 10px' }}>{p.onset_age}yr</td>
                    <td style={{ padding: '6px 10px' }}>{(p.ck_peak || 0).toLocaleString()}</td>
                    <td style={{ padding: '6px 10px', color: p.ambulant ? '#2e7d32' : '#b71c1c' }}>{p.ambulant ? '✓' : '✗'}</td>
                    <td style={{ padding: '6px 10px', color: p.cardiac_involvement ? '#e65100' : '#888' }}>{p.cardiac_involvement ? '✓' : '–'}</td>
                    <td style={{ padding: '6px 10px', textAlign: 'center' }}>{p.rhabdo_episodes}</td>
                    <td style={{ padding: '6px 10px', fontSize: 10 }}>{p.key_trigger}</td>
                    <td style={{ padding: '6px 10px', fontSize: 10 }}>
                      {p.second_wind && <Badge text="2nd-Wind" color="#1565c0" />}
                      {p.riboflavin_responsive && <Badge text="Riboflavin-R" color="#6a1b9a" />}
                      {p.retinopathy && <Badge text="Retinopathy" color="#e65100" />}
                      {p.hemolytic_anemia && <Badge text="Haemolysis" color="#00695c" />}
                      {p.nbs_detected && <Badge text="NBS" color="#2e7d32" />}
                      {p.carnitine_responsive && <Badge text="L-Carn" color="#880e4f" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 12, color: '#888', marginTop: 8 }}>
            Showing first 120 of 320 patients. Seeds 2230–2237 (reproducible).
          </div>
        </div>
      )}

      {/* ── TAB 3: Definitions ── */}
      {tab === 3 && defs && (
        <div>
          <h3 style={{ color: '#1a237e', marginBottom: 16 }}>Glossary &amp; Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 14, marginBottom: 28 }}>
            {Object.entries(defs.glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#f5f5f5', border: '1px solid #ddd', borderRadius: 8, padding: '12px 14px' }}>
                <div style={{ fontWeight: 800, color: '#1a237e', marginBottom: 4 }}>{term}</div>
                <div style={{ fontSize: 12, color: '#333', lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#1a237e', marginBottom: 12 }}>Diagnostic Algorithm</h3>
          <div style={{ background: '#e8f5e9', border: '1px solid #2e7d32', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {(defs.diagnostic_algorithm || []).map((step, i) => (
              <div key={i} style={{ fontSize: 12, color: '#1b5e20', marginBottom: 6 }}>{step}</div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            <div>
              <h3 style={{ color: '#1a237e', marginBottom: 10 }}>References</h3>
              {(defs.references || []).map((r, i) => (
                <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 6, paddingLeft: 8, borderLeft: '3px solid #1565c0' }}>{r}</div>
              ))}
            </div>
            <div>
              <h3 style={{ color: '#1a237e', marginBottom: 10 }}>Standards</h3>
              {(defs.standards || []).map((s, i) => (
                <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 6, paddingLeft: 8, borderLeft: '3px solid #2e7d32' }}>{s}</div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
