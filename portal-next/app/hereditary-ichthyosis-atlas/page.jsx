'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ichthyosis-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  TGM1:    '#1565c0',  // deep blue   — ARCI1, most common, collodion baby, TGase-1
  ABCA12:  '#b71c1c',  // deep red    — Harlequin Ichthyosis, armor-plate scale, acitretin emergency
  CYP4F22: '#e65100',  // deep orange — ARCI6, non-erythrodermic LI, PPK
  NIPAL4:  '#4a148c',  // deep purple — ARCI4, pruritus prominent, ichthyin
  STS:     '#2e7d32',  // deep green  — X-linked, corneal opacity, cryptorchidism
  KRT1:    '#880e4f',  // deep pink   — EI, EHK pathognomonic, S. aureus
  GJB3:    '#f57f17',  // amber       — EKV1, figurate erythema, connexin
  ALOX12B: '#00695c',  // teal        — ARCI8, palmar fissuring, 12R-LOX
};

const GENE_INFO = {
  TGM1:    { full: 'TGM1 / 817aa',    locus: '14q12',    size: '817 aa / 90 kDa',   inh: 'AR',  disease: 'ARCI1 — Lamellar Ichthyosis type 1 — COLLODION BABY (ectropion + eclabium + tight shiny membrane at birth) PATHOGNOMONIC; most common ARCI worldwide (30-35%); plate-like dark-brown lamellar scale post-shedding; ANHIDROSIS → heat intolerance; TGase-1 enzyme activity assay confirms (absent = TGM1); acitretin most effective systemic therapy' },
  ABCA12:  { full: 'ABCA12 / 2595aa', locus: '2q35',     size: '2595 aa / 260 kDa', inh: 'AR',  disease: 'Harlequin Ichthyosis (HI) — most severe ARCI — ARMOR-PLATE DIAMOND-SHAPED SCALE PLATES SEPARATING AT BIRTH PATHOGNOMONIC; ectropion + eclabium + flattened nose/ears = neonatal emergency; ACITRETIN WITHIN 24h = LIFE-SAVING; absent lamellar granules on EM; NICU + ophthalmology + ENT mandatory; survival improved from 40% to 80% with modern management' },
  CYP4F22: { full: 'CYP4F22 / 524aa', locus: '19p13.12', size: '524 aa / 60 kDa',   inh: 'AR',  disease: 'ARCI6 — Lamellar Ichthyosis type 6 — omega-hydroxylase deficiency → ultra-long chain fatty acid barrier defect; FINE BROWN LAMELLAR SCALE + PALMOPLANTAR KERATODERMA PATHOGNOMONIC; NON-ERYTHRODERMIC (minimal erythema) — key DDx from TGM1 erythrodermic form; no/mild collodion baby; TGase-1 assay negative (excludes TGM1); gene panel mandatory' },
  NIPAL4:  { full: 'NIPAL4 / 399aa',  locus: '5q33.3',   size: '399 aa / 44 kDa',   inh: 'AR',  disease: 'ARCI4 — Lamellar Ichthyosis type 4 (Ichthyin) — PRURITUS + PLATE-LIKE LAMELLAR SCALE + ANHIDROSIS PATHOGNOMONIC; significant pruritus distinguishes from TGM1-LI (less pruritic); collodion baby 30-50%; heat intolerance; may improve with age; dupilumab under investigation for itch; TGase-1 negative (excludes TGM1)' },
  STS:     { full: 'STS / 583aa',     locus: 'Xp22.31',  size: '583 aa / 62 kDa',   inh: 'XLR', disease: 'X-linked Ichthyosis (XLI) — steroid sulfatase deficiency → cholesterol sulfate accumulation in SC; LARGE DARK BROWN POLYGONAL SCALE NECK/EXTENSOR + POSTERIOR CORNEAL OPACITY (asymptomatic, slit-lamp mandatory) PATHOGNOMONIC; CRYPTORCHIDISM 20% males (orchidopexy mandatory — malignancy risk 3-5×); NO COLLODION BABY; palms/soles spared; Kallmann overlap if Xp22.3 contiguous deletion' },
  KRT1:    { full: 'KRT1 / 644aa',    locus: '12q13.13', size: '644 aa / 67 kDa',   inh: 'AD',  disease: 'Epidermolytic Ichthyosis (EI) — EPIDERMOLYTIC HYPERKERATOSIS (EHK) ON BIOPSY PATHOGNOMONIC (suprabasal vacuolation + granular epidermolysis + compact hyperkeratosis); blistering at birth → dark verrucous scale in childhood; PALMOPLANTAR KERATODERMA prominent (KRT1 expressed in palms/soles); SECONDARY S. AUREUS SUPERINFECTION = most common complication; antiseptic washes daily mandatory' },
  GJB3:    { full: 'GJB3 / 270aa',    locus: '1p34.3',   size: '270 aa / 31 kDa',   inh: 'AD',  disease: 'Erythrokeratoderma Variabilis (EKV) type 1 — connexin-31 gap junction defect; TRANSIENT MIGRATORY FIGURATE ERYTHEMATOUS PATCHES + FIXED HYPERKERATOTIC PLAQUES PATHOGNOMONIC; erythema changes shape/location daily — UNIQUE among ichthyoses; triggered by emotional stress/temperature; palmoplantar keratoderma; pure skin disorder; GJB4 DDx (EKV type 2); hearing screen annually' },
  ALOX12B: { full: 'ALOX12B / 701aa', locus: '17p13.1',  size: '701 aa / 77 kDa',   inh: 'AR',  disease: 'ARCI8 — Lamellar Ichthyosis type 8 — 12R-lipoxygenase epidermal ceramide barrier defect; BROWN PLATE-LIKE SCALE + PALMAR FISSURING PATHOGNOMONIC; no blistering (distinguishes from EI/KRT1); heat intolerance + anhidrosis; collodion baby 40-60%; clinically identical to TGM1/NIPAL4/CYP4F22 — gene panel mandatory; check ALOXE3 (tandem pathway partner); TGase-1 negative excludes TGM1' },
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
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
        <span style={{ fontWeight: 800, fontSize: 18, color }}>{gene}</span>
        <Badge text={info.locus} color={color} />
        <Badge text={info.inh} color={color} />
        <Badge text={info.size} color="#555" />
      </div>
      <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap' }}>
          {data.avg_age_at_dx_yrs !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Avg Age Dx: <b>{data.avg_age_at_dx_yrs}yr</b></span>}
          {data.ichthyosis_type && <span style={{ fontSize: 12, color: '#555' }}>Type: <b>{data.ichthyosis_type.split(' — ')[0]}</b></span>}
          {data.n_patients !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Patients: <b>{data.n_patients}</b></span>}
        </div>
      )}
    </div>
  );
}

export default function HeredIchthyosisAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#1565c0', fontWeight: 700 }}>Loading Hereditary Ichthyosis Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1565c0', marginBottom: 4 }}>
          🧬 Hereditary Ichthyosis Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — TGM1 · ABCA12 · CYP4F22 · NIPAL4 · STS · KRT1 · GJB3 · ALOX12B —
          320 patients (8 × 40, seeds 2278-2285)
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#1565c0" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2278-2285" color="#e65100" />
        <StatCard label="ARCI Genes" value="5" sub="TGM1/ABCA12/CYP4F22/NIPAL4/ALOX12B" color="#4a148c" />
        <StatCard label="Non-ARCI" value="3" sub="STS (XLR) / KRT1 (AD) / GJB3 (AD)" color="#880e4f" />
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 22, borderBottom: '2px solid #e3e8f0' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', fontWeight: 700, fontSize: 13,
            border: 'none', cursor: 'pointer', borderRadius: '6px 6px 0 0',
            background: tab === i ? '#1565c0' : '#f5f7fa',
            color: tab === i ? '#fff' : '#555',
          }}>{t}</button>
        ))}
      </div>

      {/* TAB 0 — Overview */}
      {tab === 0 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 14 }}>Key Clinical Pearls</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 28 }}>
            {overview.key_clinical_pearls?.map((p, i) => (
              <div key={i} style={{
                background: '#f0f4ff', borderLeft: `4px solid ${Object.values(GENE_COLORS)[i] || '#1565c0'}`,
                padding: '10px 14px', borderRadius: 6, fontSize: 13, lineHeight: 1.6,
              }}>{p}</div>
            ))}
          </div>

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 14 }}>Ichthyosis Categories</h2>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            {Object.entries(overview.ichthyosis_categories || {}).map(([cat, genes_]) => (
              <div key={cat} style={{
                border: '2px solid #1565c020', borderRadius: 8, padding: '10px 14px',
                background: '#f8faff', minWidth: 200,
              }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1565c0', marginBottom: 4 }}>{cat}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{genes_}</div>
              </div>
            ))}
          </div>

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 12 }}>Diagnostic Algorithm</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 24 }}>
            {Object.entries(overview.diagnostic_algorithm || {}).map(([step, desc]) => (
              <div key={step} style={{
                background: '#fff', border: '1px solid #e0e0e0', borderRadius: 8,
                padding: '10px 14px', fontSize: 13,
              }}>
                <span style={{ fontWeight: 700, color: '#1565c0', marginRight: 8 }}>{step.replace('_', ' ')}:</span>
                {desc}
              </div>
            ))}
          </div>

          <h2 style={{ fontSize: 15, fontWeight: 700, color: '#1565c0', marginBottom: 10 }}>Collodion Baby Genes</h2>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
            {overview.collodion_baby_genes?.map(g => (
              <Badge key={g} text={g} color="#1565c0" />
            ))}
          </div>
        </div>
      )}

      {/* TAB 1 — Gene Table */}
      {tab === 1 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 14 }}>Gene Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene', 'Locus', 'Size', 'Inh.', 'Ichthyosis Type', 'Scale', 'Pathognomonic', 'Avg Dx (yr)', 'N'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(overview.gene_summary || []).map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f8faff' : '#fff' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '7px 10px' }}>{row.locus}</td>
                    <td style={{ padding: '7px 10px', whiteSpace: 'nowrap' }}>{row.protein_size}</td>
                    <td style={{ padding: '7px 10px' }}><Badge text={row.inheritance} color={GENE_COLORS[row.gene]} /></td>
                    <td style={{ padding: '7px 10px', maxWidth: 160 }}>{row.ichthyosis_type}</td>
                    <td style={{ padding: '7px 10px', maxWidth: 140, fontSize: 11 }}>{row.scale_morphology?.split(';')[0]}</td>
                    <td style={{ padding: '7px 10px', maxWidth: 200, fontSize: 11 }}>{row.pathognomonic?.substring(0, 80)}…</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{row.avg_age_at_dx_yrs}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center', fontWeight: 700 }}>{row.n_patients}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 2 — Clinical Atlas */}
      {tab === 2 && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 16 }}>Per-Gene Clinical Atlas</h2>
          {genes.map(gene => {
            const color = GENE_COLORS[gene];
            const info = GENE_INFO[gene];
            const data = breakdown?.gene_breakdown?.[gene];
            if (!data) return null;
            return (
              <div key={gene} style={{
                border: `2px solid ${color}`, borderRadius: 12, padding: 18, marginBottom: 16,
                background: color + '06',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
                  <span style={{ fontWeight: 800, fontSize: 20, color }}>{gene}</span>
                  <Badge text={info.locus} color={color} />
                  <Badge text={info.inh} color={color} />
                  <Badge text={info.size} color="#555" />
                  <Badge text={`${data.n_patients} pts`} color={color} />
                  <Badge text={`Avg Dx: ${data.avg_age_at_dx_yrs}yr`} color="#555" />
                  {data.collodion_baby_n > 0 && <Badge text={`Collodion: ${data.collodion_baby_n}`} color="#e65100" />}
                </div>

                <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6, marginBottom: 10 }}>{info.disease}</div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color, marginBottom: 6 }}>Complication Distribution</div>
                    {Object.entries(data.complication_distribution || {}).slice(0, 5).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <div style={{ width: `${Math.round(v / 40 * 100)}px`, height: 10, background: color, borderRadius: 3, minWidth: 4 }} />
                        <span style={{ fontSize: 11 }}>{k.replace(/_/g, ' ')} ({v})</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color, marginBottom: 6 }}>Key Features</div>
                    {(data.key_features || []).slice(0, 4).map((f, i) => (
                      <div key={i} style={{ fontSize: 11, marginBottom: 4, lineHeight: 1.5 }}>• {f.substring(0, 120)}{f.length > 120 ? '…' : ''}</div>
                    ))}
                  </div>
                </div>

                <div style={{ background: color + '11', borderRadius: 6, padding: '8px 12px', fontSize: 12 }}>
                  <b>Pathognomonic:</b> {data.pathognomonic}
                </div>

                {data.key_ddx?.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <span style={{ fontWeight: 700, fontSize: 12, color }}>Key DDx: </span>
                    {data.key_ddx.slice(0, 3).map((d, i) => (
                      <span key={i} style={{ fontSize: 11, marginRight: 8 }}>• {d.split(' (')[0]}</span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {/* Emergency flags */}
          <h3 style={{ fontSize: 15, fontWeight: 700, color: '#b71c1c', marginTop: 20, marginBottom: 10 }}>Clinical Emergency Flags</h3>
          {breakdown?.clinical_emergency_flags?.map((flag, i) => (
            <div key={i} style={{
              background: '#fff5f5', border: '2px solid #b71c1c', borderRadius: 8,
              padding: '10px 14px', marginBottom: 8, fontSize: 13, lineHeight: 1.6,
            }}>⚠️ {flag}</div>
          ))}
        </div>
      )}

      {/* TAB 3 — Definitions */}
      {tab === 3 && definitions && (
        <div>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 14 }}>Gene Definitions</h2>
          {genes.map(gene => {
            const entry = definitions.gene_entries?.[gene];
            const color = GENE_COLORS[gene];
            if (!entry) return null;
            return (
              <details key={gene} style={{ marginBottom: 10, border: `1px solid ${color}44`, borderRadius: 8 }}>
                <summary style={{
                  padding: '10px 14px', fontWeight: 700, color, cursor: 'pointer',
                  background: color + '0a', borderRadius: 8,
                }}>
                  {gene} — {GENE_INFO[gene].full} ({GENE_INFO[gene].locus} · {GENE_INFO[gene].inh})
                </summary>
                <div style={{ padding: '12px 16px', fontSize: 13, lineHeight: 1.7 }}>
                  <div style={{ marginBottom: 10 }}>
                    <b>Inheritance:</b> {entry.inheritance_details?.substring(0, 400)}
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <b>Key Features:</b>
                    <ul style={{ margin: '6px 0 0 18px', padding: 0 }}>
                      {(entry.key_features || []).map((f, i) => <li key={i} style={{ marginBottom: 4 }}>{f}</li>)}
                    </ul>
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <b>Treatment:</b> {entry.treatment?.substring(0, 500)}…
                  </div>
                  <div>
                    <b>Monitoring:</b>
                    <ul style={{ margin: '6px 0 0 18px', padding: 0 }}>
                      {(entry.monitoring || []).map((m, i) => <li key={i} style={{ marginBottom: 4 }}>{m}</li>)}
                    </ul>
                  </div>
                </div>
              </details>
            );
          })}

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>Skin Biology Glossary</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.skin_biology_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#f8faff', border: '1px solid #e0e8ff', borderRadius: 8, padding: '10px 14px', fontSize: 13 }}>
                <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 4 }}>{term}</div>
                <div style={{ lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>Ichthyosis Types</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.ichthyosis_type_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#f0f8f0', border: '1px solid #c8e6c9', borderRadius: 8, padding: '10px 14px', fontSize: 13 }}>
                <div style={{ fontWeight: 700, color: '#2e7d32', marginBottom: 4 }}>{term}</div>
                <div style={{ lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>Diagnostic Tests</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.diagnostic_tests || {}).map(([test, desc]) => (
              <div key={test} style={{ background: '#fff8f0', border: '1px solid #ffe0b2', borderRadius: 8, padding: '10px 14px', fontSize: 13 }}>
                <div style={{ fontWeight: 700, color: '#e65100', marginBottom: 4 }}>{test}</div>
                <div style={{ lineHeight: 1.6 }}>{desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
