'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-palmoplantar-keratoderma-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  KRT9:     '#1565c0',  // deep blue   — EPPK, most common hereditary PPK, palm-only
  SLURP1:   '#b71c1c',  // deep red    — Mal de Meleda, transgredient, Adriatic founder
  CTSC:     '#e65100',  // deep orange — Papillon-Lefèvre, periodontitis, dental emergency
  GJB2:     '#4a148c',  // deep purple — Vohwinkel, honeycomb PPK, SNHL, KID syndrome
  DSP:      '#2e7d32',  // deep green  — Carvajal, woolly hair + DCM
  JUP:      '#880e4f',  // deep pink   — Naxos, woolly hair + ARVC
  SERPINB7: '#f57f17',  // amber       — Bothnian, aquagenic wrinkling, Swedish founder
  LORICRIN: '#00695c',  // teal        — Loricrin keratoderma, pseudoainhum, NO SNHL
};

const GENE_INFO = {
  KRT9:     { full: 'KRT9 / 464aa',     locus: '17q21.2',  size: '464 aa / 56 kDa',   inh: 'AD',  disease: 'Epidermolytic PPK (EPPK) / Vörner disease — most common hereditary PPK; PALM-ONLY (feet completely spared) — PATHOGNOMONIC — KRT9 expressed in palms not soles; BIOPSY: suprabasal vacuolation + epidermolysis (EHK) confirms; p.Arg163Trp most common European mutation; urea 40% + keratolytics; acitretin for severe' },
  SLURP1:   { full: 'SLURP1 / 103aa',   locus: '8q24.13',  size: '103 aa / 11 kDa',   inh: 'AR',  disease: 'Mal de Meleda disease — TRANSGREDIENT PPK EXTENDING TO DORSAL HANDS/FEET + ERYTHEMATOUS BORDER PATHOGNOMONIC; hyperhidrosis (severe) + pseudoainhum (constricting bands) + perioral erythema; Mljet Island (Adriatic Croatia) founder; p.W15R most common; retinoids most effective; pseudoainhum surgical release' },
  CTSC:     { full: 'CTSC / 463aa',     locus: '11q14.2',  size: '463 aa / 51 kDa',   inh: 'AR',  disease: 'Papillon-Lefèvre syndrome (PLS) — PPK + SEVERE EARLY-ONSET PERIODONTITIS + PREMATURE TOOTH LOSS PATHOGNOMONIC; ALL permanent teeth lost by age 14 without intervention; PROPHYLACTIC ANTIBIOTICS (amoxicillin+metronidazole) MANDATORY before teeth erupt; Haim-Munk variant (same gene: +arachnodactyly+acro-osteolysis); cathepsin C → neutrophil serine protease activation' },
  GJB2:     { full: 'GJB2 / 226aa',     locus: '13q12.11', size: '226 aa / 26 kDa',   inh: 'AD',  disease: 'Vohwinkel syndrome (mutilating PPK) — HONEYCOMB PPK + STARFISH-SHAPED KNUCKLE PADS + PSEUDOAINHUM PATHOGNOMONIC; SNHL (sensorineural hearing loss) in classic Vohwinkel; KID syndrome (p.Asp50Asn): keratitis+ichthyosis+deafness (same gene, different phenotype); connexin-26 gap junction; cochlear implant for profound SNHL; pseudoainhum surgical release' },
  DSP:      { full: 'DSP / 2871aa',     locus: '6p24.3',   size: '2871 aa / 332 kDa', inh: 'AR',  disease: 'Carvajal syndrome — PPK + WOOLLY HAIR + DILATED CARDIOMYOPATHY (DCM) PATHOGNOMONIC; LEFT ventricular dysfunction; cardiac MRI + ICD if EF<35% MANDATORY; sudden cardiac death in teenagers; desmoplakin largest desmosomal protein; KEY DDx from JUP/Naxos: LEFT DCM not RIGHT ARVC' },
  JUP:      { full: 'JUP / 745aa',      locus: '17q21.2',  size: '745 aa / 81 kDa',   inh: 'AR',  disease: 'Naxos disease — PPK + WOOLLY HAIR + ARVC (RIGHT VENTRICULAR CARDIOMYOPATHY) PATHOGNOMONIC; ICD MANDATORY + SPORTS RESTRICTION ABSOLUTE; ventricular arrhythmias + SCD; Greek Naxos Island founder (c.2157del2); plakoglobin desmosomal protein; KEY DDx from DSP/Carvajal: RIGHT ARVC not LEFT DCM' },
  SERPINB7: { full: 'SERPINB7 / 394aa', locus: '18q21.33', size: '394 aa / 44 kDa',   inh: 'AR',  disease: 'Bothnian type PPK (NEPPK2) — AQUAGENIC WRINKLING OF PALMS (exaggerated wrinkling within minutes of water exposure) PATHOGNOMONIC; palmar pits + punctate hyperkeratosis; MILD disease — no cardiac/dental/systemic involvement; Swedish Bothnian region founder (p.Trp249Ter); emollients + antiperspirant; check CFTR if aquagenic wrinkling without SERPINB7 mutation' },
  LORICRIN: { full: 'LORICRIN / 312aa', locus: '1q21.3',   size: '312 aa / 37 kDa',   inh: 'AD',  disease: 'Loricrin keratoderma (Vohwinkel variant) — PPK + CONSTRICTION BANDS (pseudoainhum) + ICHTHYOTIC VEIL PATHOGNOMONIC; NO HEARING LOSS = KEY DDx from GJB2-Vohwinkel (which has SNHL); audiometry distinguishes; loricrin major CE protein (70-80%); frameshift → nuclear accumulation; retinoids partially effective; surgical release for bands' },
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

export default function HeredPPKAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#1565c0', fontWeight: 700 }}>Loading Hereditary Palmoplantar Keratoderma Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1565c0', marginBottom: 4 }}>
          🧬 Hereditary Palmoplantar Keratoderma Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — KRT9 · SLURP1 · CTSC · GJB2 · DSP · JUP · SERPINB7 · LORICRIN —
          320 patients (8 × 40, seeds 2286-2293)
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#1565c0" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2286-2293" color="#e65100" />
        <StatCard label="Cardiac Emergency" value="2" sub="DSP (DCM) / JUP (ARVC)" color="#880e4f" />
        <StatCard label="Dental Emergency" value="1" sub="CTSC — Papillon-Lefèvre" color="#b71c1c" />
        <StatCard label="Pseudoainhum" value="3" sub="SLURP1 / GJB2 / LORICRIN" color="#4a148c" />
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

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 14 }}>PPK Categories by Mechanism</h2>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            {Object.entries(overview.ppk_categories || {}).map(([cat, genes_]) => (
              <div key={cat} style={{
                border: '2px solid #1565c020', borderRadius: 8, padding: '10px 14px',
                background: '#f8faff', minWidth: 220,
              }}>
                <div style={{ fontWeight: 700, fontSize: 12, color: '#1565c0', marginBottom: 4 }}>{cat}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{genes_}</div>
              </div>
            ))}
          </div>

          {/* Emergency flags */}
          <h2 style={{ fontSize: 15, fontWeight: 700, color: '#b71c1c', marginBottom: 10 }}>Clinical Emergency Genes</h2>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 18 }}>
            <div style={{ background: '#fff5f5', border: '2px solid #b71c1c', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#b71c1c', fontSize: 13 }}>Cardiac Emergency (2)</div>
              {overview.cardiac_emergency_genes?.map(g => <Badge key={g} text={g} color="#b71c1c" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>DSP: DCM (LEFT) | JUP: ARVC (RIGHT) — ICD + sports restriction</div>
            </div>
            <div style={{ background: '#fff8f0', border: '2px solid #e65100', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#e65100', fontSize: 13 }}>Dental Emergency (1)</div>
              {overview.dental_emergency_genes?.map(g => <Badge key={g} text={g} color="#e65100" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>CTSC: prophylactic antibiotics mandatory before teeth erupt</div>
            </div>
            <div style={{ background: '#f5f0ff', border: '2px solid #4a148c', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#4a148c', fontSize: 13 }}>Pseudoainhum Genes (3)</div>
              {overview.pseudoainhum_genes?.map(g => <Badge key={g} text={g} color="#4a148c" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>Surgical release mandatory if vascular compromise</div>
            </div>
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
                  {['Gene', 'Locus', 'Size', 'Inh.', 'PPK Type', 'Pathognomonic', 'Cardiac', 'Dental', 'Avg Dx (yr)', 'N'].map(h => (
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
                    <td style={{ padding: '7px 10px', maxWidth: 160, fontSize: 11 }}>{row.ppk_type?.split(' — ')[0]}</td>
                    <td style={{ padding: '7px 10px', maxWidth: 200, fontSize: 11 }}>{row.pathognomonic?.substring(0, 80)}…</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{row.cardiac_risk ? '⚠️' : '—'}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{row.dental_risk ? '🦷' : '—'}</td>
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
                  {data.cardiac_risk && <Badge text="CARDIAC EMERGENCY" color="#b71c1c" />}
                  {data.dental_risk && <Badge text="DENTAL EMERGENCY" color="#e65100" />}
                  {data.transgredient_pct > 50 && <Badge text={`Transgredient: ${data.transgredient_pct}%`} color="#4a148c" />}
                </div>

                <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6, marginBottom: 10 }}>{info.disease}</div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 12 }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color, marginBottom: 6 }}>Systemic Feature Distribution</div>
                    {Object.entries(data.systemic_distribution || {}).slice(0, 5).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <div style={{ width: `${Math.round(v / 40 * 100)}px`, height: 10, background: color, borderRadius: 3, minWidth: 4 }} />
                        <span style={{ fontSize: 11 }}>{k.replace(/_/g, ' ')} ({v})</span>
                      </div>
                    ))}
                    {Object.keys(data.systemic_distribution || {}).length === 0 && (
                      <div style={{ fontSize: 11, color: '#888' }}>No systemic features (pure skin disease)</div>
                    )}
                  </div>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: 12, color, marginBottom: 6 }}>Key Features</div>
                    {(data.key_features || []).slice(0, 4).map((f, i) => (
                      <div key={i} style={{ fontSize: 11, marginBottom: 4, lineHeight: 1.5 }}>• {f.substring(0, 120)}{f.length > 120 ? '…' : ''}</div>
                    ))}
                  </div>
                </div>

                <div style={{ background: color + '11', borderRadius: 6, padding: '8px 12px', fontSize: 12, marginBottom: 8 }}>
                  <b>Pathognomonic:</b> {data.pathognomonic}
                </div>

                {data.key_ddx?.length > 0 && (
                  <div>
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
                    <b>Disease:</b> {entry.disease_name}
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <b>Pathognomonic:</b> {entry.pathognomonic}
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
                  <div style={{ marginBottom: 10 }}>
                    <b>Key DDx:</b>
                    <ul style={{ margin: '6px 0 0 18px', padding: 0 }}>
                      {(entry.key_ddx || []).map((d, i) => <li key={i} style={{ marginBottom: 4 }}>{d}</li>)}
                    </ul>
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

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>PPK Biology Glossary</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.ppk_biology_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#f8faff', border: '1px solid #e0e8ff', borderRadius: 8, padding: '10px 14px', fontSize: 13 }}>
                <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 4 }}>{term}</div>
                <div style={{ lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>PPK Types</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.ppk_type_glossary || {}).map(([term, def]) => (
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

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>Treatment Glossary</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.treatment_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#f5f0ff', border: '1px solid #e1bee7', borderRadius: 8, padding: '10px 14px', fontSize: 13 }}>
                <div style={{ fontWeight: 700, color: '#4a148c', marginBottom: 4 }}>{term}</div>
                <div style={{ lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
