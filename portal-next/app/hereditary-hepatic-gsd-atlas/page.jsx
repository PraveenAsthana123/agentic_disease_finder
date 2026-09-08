'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hepatic-gsd-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  G6PC:    '#1565c0',  // deep blue    — Von Gierke, metabolic quartet
  SLC37A4: '#b71c1c',  // deep red     — GSD Ib, cyclic neutropenia
  GAA:     '#2e7d32',  // deep green   — Pompe, ERT available
  AGL:     '#6a1b9a',  // deep purple  — Cori/Forbes, liver+muscle
  GBE1:    '#e65100',  // deep orange  — Andersen, polyglucosan
  PYGL:    '#00695c',  // teal         — Hers, benign course
  PHKA2:   '#4a148c',  // dark purple  — GSD IXa, X-linked
  GYS2:    '#880e4f',  // deep pink    — GSD 0, no hepatomegaly
};

const GENE_INFO = {
  G6PC:    { full: 'G6PC / 357aa',    locus: '17q21.31', size: '357 aa / 36 kDa',    inh: 'AR',  disease: 'Von Gierke Disease (GSD Ia) — METABOLIC QUARTET PATHOGNOMONIC: fasting hypoglycemia + lactic acidosis + hyperuricemia + hyperlipidemia / hepatomegaly + renal enlargement PATHOGNOMONIC combination / glucagon test FAILS (no glucose rise) / platelet dysfunction bleeding diathesis / cornstarch therapy / hepatic adenomas ≥70% adults → HCC surveillance mandatory / OMIM 232200' },
  SLC37A4: { full: 'SLC37A4 / 429aa', locus: '11q23.3',  size: '429 aa / 46 kDa',   inh: 'AR',  disease: 'GSD Ib — IDENTICAL to GSD Ia metabolically PLUS CYCLIC NEUTROPENIA PATHOGNOMONIC (absent in GSD Ia) / Crohn\'s-like IBD / recurrent bacterial infections / G-CSF mandatory for neutropenia / Empagliflozin (SGLT2i) FDA 2023 for neutropenia / p.G149E most common European allele / OMIM 232220' },
  GAA:     { full: 'GAA / 952aa',     locus: '17q25.3',  size: '952 aa / 105 kDa',  inh: 'AR',  disease: 'Pompe Disease (GSD II) — IOPD: massive HCM + hypotonia → death <1yr without ERT / LOPD: progressive proximal myopathy + respiratory failure NO cardiomyopathy / CRIM status critical (high-titre antibody → poorer ERT response) / Avalglucosidase alfa (Nexviazyme) FDA 2021 SUPERIOR to alglucosidase alfa / c.-32-13T>G IVS1 late-onset Caucasian allele / NBS detected / OMIM 232300' },
  AGL:     { full: 'AGL / 1532aa',    locus: '1p21.2',   size: '1532 aa / 170 kDa', inh: 'AR',  disease: 'Cori/Forbes Disease (GSD IIIa/b) — LIMIT DEXTRINOSIS / IIIa (liver+muscle 85%): CK elevated, myopathy / IIIb (liver-only 15%): CK normal / hepatomegaly → cirrhosis risk in adulthood / HIGH-PROTEIN DIET beneficial for muscle (provides alanine/gluconeogenic substrate) / OMIM 232400' },
  GBE1:    { full: 'GBE1 / 702aa',    locus: '3p12.3',   size: '702 aa / 80 kDa',   inh: 'AR',  disease: 'Andersen Disease (GSD IV) — AMYLOPECTINOSIS (polyglucosan deposits) / CLASSIC: neonatal hepatic failure → cirrhosis → liver transplant only curative / NON-PROGRESSIVE HEPATIC: p.Y329S Ashkenazi Jewish founder allele → survives without transplant / ADULT APBD: polyglucosan body disease ≥40yr neurodegeneration / OMIM 232500' },
  PYGL:    { full: 'PYGL / 846aa',    locus: '14q22.1',  size: '846 aa / 97 kDa',   inh: 'AR',  disease: 'Hers Disease (GSD VI) — BENIGN COURSE: hepatomegaly resolves with age / fasting hypoglycemia mild / hyperketonemia prominent / NO myopathy NO cardiac involvement / most adults ASYMPTOMATIC / cornstarch + frequent feeds for symptomatic children / OMIM 232700' },
  PHKA2:   { full: 'PHKA2 / 1235aa',  locus: 'Xp22.13',  size: '1235 aa / 135 kDa', inh: 'XLR', disease: 'GSD IXa (Liver Phosphorylase Kinase Deficiency) — X-LINKED RECESSIVE: males affected / MOST COMMON CHILDHOOD GSD after GSD III / hepatomegaly + growth retardation + fasting ketosis / NO MUSCLE INVOLVEMENT (liver-specific alpha2 subunit) / TRANSIENT: symptoms often resolve by puberty / high-protein diet + cornstarch mild cases / OMIM 306000' },
  GYS2:    { full: 'GYS2 / 703aa',    locus: '12p12.1',  size: '703 aa / 81 kDa',   inh: 'AR',  disease: 'GSD 0 (Glycogen Synthase Deficiency) — FASTING HYPOGLYCEMIA + HYPERKETONAEMIA WITHOUT HEPATOMEGALY PATHOGNOMONIC / NO HEPATOMEGALY (cannot synthesize glycogen → no storage → no liver enlargement) / postprandial HYPERGLYCEMIA (glucose shunted to blood not glycogen) / NO lactic acidosis (NOT a glycogenolytic block) / high-protein + frequent feeds / OMIM 240600' },
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
          <span style={{ fontSize: 12, color: '#555' }}>Hypoglycemia: <b>{data.hypoglycemia_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Hepatomegaly: <b>{data.hepatomegaly_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Cirrhosis: <b>{data.cirrhosis_pct}%</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Avg Onset: <b>{data.avg_onset}yr</b></span>
          <span style={{ fontSize: 12, color: '#555' }}>Avg Glucose: <b>{data.avg_glucose_mmol?.toFixed(1)} mmol/L</b></span>
        </div>
      )}
    </div>
  );
}

export default function HHepaticGsdAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary Hepatic GSD Atlas…</div>;
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
          🧬 Hereditary Hepatic Glycogen Storage Disease Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Hepatic GSD Spectrum Atlas — G6PC · SLC37A4 · GAA · AGL · GBE1 · PYGL · PHKA2 · GYS2
        </div>
        <div style={{ fontSize: 12, color: '#777', marginTop: 4 }}>
          320-patient aggregate cohort (8 × 40, seeds 2238–2245)
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
            <StatCard label="Seeds" value="2238–2245" sub="reproducible" color="#283593" />
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
                    'Hepatomegaly %', 'Cirrhosis %', 'Avg Glucose (mmol/L)'].map(h => (
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
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{gs.hepatomegaly_pct}%</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{gs.cirrhosis_pct}%</td>
                      <td style={{ padding: '10px 12px', textAlign: 'center' }}>{gs.avg_glucose_mmol?.toFixed(1)}</td>
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
            const maxSz = 1532;
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
                  {['ID', 'Gene', 'Sex', 'Age', 'Onset', 'Glucose (mmol/L)', 'Hepatomegaly', 'Cirrhosis', 'ERT/Tx', 'Key Feature', 'Flags'].map(h => (
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
                    <td style={{ padding: '6px 10px' }}>{(p.glucose_mmol || 0).toFixed(1)}</td>
                    <td style={{ padding: '6px 10px', color: p.hepatomegaly ? '#b71c1c' : '#888' }}>{p.hepatomegaly ? '✓' : '–'}</td>
                    <td style={{ padding: '6px 10px', color: p.cirrhosis ? '#e65100' : '#888' }}>{p.cirrhosis ? '✓' : '–'}</td>
                    <td style={{ padding: '6px 10px', color: p.ert_or_transplant ? '#2e7d32' : '#888' }}>{p.ert_or_transplant ? '✓' : '–'}</td>
                    <td style={{ padding: '6px 10px', fontSize: 10 }}>{p.key_feature}</td>
                    <td style={{ padding: '6px 10px', fontSize: 10 }}>
                      {p.cyclic_neutropenia && <Badge text="Neutropenia" color="#b71c1c" />}
                      {p.nbs_detected && <Badge text="NBS" color="#2e7d32" />}
                      {p.crim_positive && <Badge text="CRIM+" color="#6a1b9a" />}
                      {p.polyglucosan && <Badge text="Polyglucosan" color="#e65100" />}
                      {p.x_linked && <Badge text="X-linked" color="#4a148c" />}
                      {p.hcc_surveillance && <Badge text="HCC-Surv" color="#1565c0" />}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 12, color: '#888', marginTop: 8 }}>
            Showing first 120 of 320 patients. Seeds 2238–2245 (reproducible).
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
