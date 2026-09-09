'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ectodermal-dysplasia-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  EDA:    '#1565c0',  // deep blue   — XLHED, classic anhidrotic, heat stroke
  EDAR:   '#0d47a1',  // navy        — Autosomal HED, East Asian variant
  WNT10A: '#2e7d32',  // deep green  — OODD, severe oligodontia, WNT pathway
  TP63:   '#6a1b9a',  // deep purple — EEC/AEC, ectrodactyly + cleft + ED
  IKBKG:  '#b71c1c',  // deep red    — Incontinentia Pigmenti, retinal emergency
  GJB6:   '#e65100',  // deep orange — Clouston, hidrotic, normal sweating
  IRF6:   '#880e4f',  // deep pink   — VWS, lip pits + cleft
  PVRL1:  '#004d40',  // dark teal   — CLPED1, Mediterranean, herpetic keratitis
};

const GENE_INFO = {
  EDA:    { full: 'EDA / 391aa',    locus: 'Xq12',    size: '391 aa / 43 kDa',  inh: 'XLR',   disease: 'X-Linked Hypohidrotic Ectodermal Dysplasia (XLHED) — ANHIDROSIS + SPARSE HAIR + HYPODONTIA TRIAD PATHOGNOMONIC; heat stroke LIFE-THREATENING (zero sweat glands); cooling vest mandatory; EDX111 intra-amniotic injection 26-30wks = first prenatal DMT; saddle nose + frontal bossing facies; starch-iodine sweat test confirms' },
  EDAR:   { full: 'EDAR / 448aa',   locus: '2q13',    size: '448 aa / 50 kDa',  inh: 'AD/AR', disease: 'Autosomal Hypohidrotic Ectodermal Dysplasia (HED) — SAME TRIAD as XLHED but autosomal; AD usually milder (partial hypohidrosis); AR biallelic = severe XLHED phenocopy; EDAR-D374A = East Asian thick straight hair variant (NOT disease — positive selection); EDA/EDAR/EDARADD converging NF-kB pathway' },
  WNT10A: { full: 'WNT10A / 417aa', locus: '2q35',    size: '417 aa / 46 kDa',  inh: 'AR',   disease: 'Odonto-Onycho-Dermal Dysplasia (OODD) — SEVERE SELECTIVE OLIGODONTIA (20-28 missing permanent teeth) PATHOGNOMONIC; most common isolated oligodontia gene Europe (20%); SWEATING COMPLETELY NORMAL = KEY DDx from EDA/EDAR; nail dystrophy + PPK + dry hair; SSPS allelic adds eyelid hidrocystomas' },
  TP63:   { full: 'TP63 / 680aa',   locus: '3q28',    size: '680 aa / 72 kDa',  inh: 'AD',   disease: 'EEC syndrome — SPLIT HAND/FOOT (ectrodactyly) + CLEFT LIP/PALATE + EDA TRIAD PATHOGNOMONIC; AEC/Hay-Wells (allelic): ANKYLOBLEPHARON FILIFORM ADNATUM AT BIRTH = neonatal surgical emergency; lacrimal duct atresia 90% EEC; renal USS mandatory (20% anomalies); p63 master regulator stratified epithelium' },
  IKBKG:  { full: 'IKBKG / 419aa',  locus: 'Xq28',   size: '419 aa / 48 kDa',  inh: 'XLD',  disease: 'Incontinentia Pigmenti (IP) — 4-STAGE BLASCHKO SKIN PATHOGNOMONIC: vesicular→verrucous→whorled hyperpigmentation→atrophic; RETINAL TRACTION DETACHMENT = OPHTHO EMERGENCY (RetCam every 3 months); seizures 40%; males usually LETHAL; del(exon4-10) NEMO 80%; EDA-ID in rare surviving males' },
  GJB6:   { full: 'GJB6 / 261aa',   locus: '13q12.11',size: '261 aa / 30 kDa',  inh: 'AD',   disease: 'Clouston syndrome (Hidrotic Ectodermal Dysplasia) — DIFFUSE PROGRESSIVE ALOPECIA + NAIL DYSTROPHY + PPK TRIAD PATHOGNOMONIC; SWEATING COMPLETELY NORMAL = KEY DDx from EDA/EDAR (no heat stroke risk in Clouston); Quebec p.Gly11Arg founder; teeth mostly spared; del(GJB6-D13S1830) causes DFNB1b deafness not Clouston' },
  IRF6:   { full: 'IRF6 / 467aa',   locus: '1q32.3',  size: '467 aa / 53 kDa',  inh: 'AD',   disease: 'Van der Woude syndrome (VWS) — LOWER LIP PITS (paramedian sinuses at vermilion) + CLEFT LIP/PALATE PATHOGNOMONIC; lip pits UNIQUE among cleft syndromes; lip pits alone without cleft = VWS diagnosis (variable expressivity); PPS allelic (popliteal pterygium + syngnathia); 2-5% ALL cleft lip/palate = IRF6' },
  PVRL1:  { full: 'PVRL1 / 517aa',  locus: '11q23.3', size: '517 aa / 58 kDa',  inh: 'AR',   disease: 'CLPED1 / Zlotogora-Ogur syndrome — CLEFT LIP/PALATE + HYPODONTIA + NAIL DYSPLASIA + SPARSE HAIR TETRAD; RECURRENT HERPETIC KERATITIS PATHOGNOMONIC (Nectin-1 = HSV-1 corneal receptor); aciclovir prophylaxis 400mg BD mandatory; Mediterranean AR founder; corneal scarring → blindness without antiviral' },
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

export default function HeredEDAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#1565c0', fontWeight: 700 }}>Loading Hereditary Ectodermal Dysplasia Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;
  if (!overview) return null;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1200 }}>
      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#1565c0', marginBottom: 4 }}>
          🧬 Hereditary Ectodermal Dysplasia Atlas
        </h1>
        <div style={{ fontSize: 13, color: '#555' }}>
          Complete 8-Gene Atlas — EDA · EDAR · WNT10A · TP63 · IKBKG · GJB6 · IRF6 · PVRL1 —
          320 patients (8 × 40, seeds 2294-2301)
        </div>
      </div>

      {/* Stat cards */}
      <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 22 }}>
        <StatCard label="Total Patients" value={overview.n_patients} color="#1565c0" />
        <StatCard label="Genes" value={overview.n_genes} color="#2e7d32" />
        <StatCard label="Seed Range" value="2294-2301" color="#e65100" />
        <StatCard label="Heat Emergency" value="2" sub="EDA (XLHED) / EDAR-AR" color="#1565c0" />
        <StatCard label="Retinal Emergency" value="2" sub="IKBKG (IP) / PVRL1" color="#b71c1c" />
        <StatCard label="Cleft Genes" value="3" sub="TP63 / IRF6 / PVRL1" color="#6a1b9a" />
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

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginBottom: 14 }}>ED Categories by Mechanism</h2>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            {Object.entries(overview.ed_categories || {}).map(([cat, genes_]) => (
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
            <div style={{ background: '#e3f2fd', border: '2px solid #1565c0', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#1565c0', fontSize: 13 }}>Heat Emergency (2)</div>
              {overview.heat_emergency_genes?.map(g => <Badge key={g} text={g} color="#1565c0" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>Anhidrosis → heat stroke LIFE-THREATENING; cooling vest mandatory</div>
            </div>
            <div style={{ background: '#fff5f5', border: '2px solid #b71c1c', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#b71c1c', fontSize: 13 }}>Retinal Emergency (2)</div>
              {overview.retinal_emergency_genes?.map(g => <Badge key={g} text={g} color="#b71c1c" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>IP: traction detachment | PVRL1: herpetic keratitis → blindness</div>
            </div>
            <div style={{ background: '#f5f0ff', border: '2px solid #6a1b9a', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#6a1b9a', fontSize: 13 }}>Cleft Genes (3)</div>
              {overview.cleft_genes?.map(g => <Badge key={g} text={g} color="#6a1b9a" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>TP63: ectrodactyly+CLP | IRF6: lip pits+CLP | PVRL1: CLP+ED</div>
            </div>
            <div style={{ background: '#fff8e1', border: '2px solid #f57f17', borderRadius: 8, padding: '10px 16px' }}>
              <div style={{ fontWeight: 700, color: '#f57f17', fontSize: 13 }}>Immunodeficiency (1)</div>
              {overview.immunodeficiency_genes?.map(g => <Badge key={g} text={g} color="#f57f17" />)}
              <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>IKBKG males: EDA-ID → mycobacterial infections</div>
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
                  {['Gene', 'Locus', 'Size', 'Inh.', 'ED Category', 'Pathognomonic', 'Heat', 'Retinal', 'Cleft', 'Avg Dx (yr)', 'N'].map(h => (
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
                    <td style={{ padding: '7px 10px', maxWidth: 160, fontSize: 11 }}>{row.ed_category?.split(' — ')[0]}</td>
                    <td style={{ padding: '7px 10px', maxWidth: 200, fontSize: 11 }}>{row.pathognomonic?.substring(0, 80)}…</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{row.heat_risk ? '🌡️' : '—'}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{row.retinal_risk ? '👁️' : '—'}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{row.cleft_risk ? '💊' : '—'}</td>
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
                  {data.heat_risk && <Badge text="HEAT EMERGENCY" color="#1565c0" />}
                  {data.retinal_risk && <Badge text="RETINAL EMERGENCY" color="#b71c1c" />}
                  {data.cleft_risk && <Badge text="CLEFT GENE" color="#6a1b9a" />}
                  {data.immunodeficiency_risk && <Badge text="IMMUNODEFICIENCY" color="#f57f17" />}
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

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>ED Biology Glossary</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.ed_biology_glossary || {}).map(([term, def]) => (
              <div key={term} style={{ background: '#f8faff', border: '1px solid #e0e8ff', borderRadius: 8, padding: '10px 14px', fontSize: 13 }}>
                <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 4 }}>{term}</div>
                <div style={{ lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>

          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#1565c0', marginTop: 24, marginBottom: 12 }}>ED Types</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {Object.entries(definitions.ed_type_glossary || {}).map(([term, def]) => (
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
                <div style={{ fontWeight: 700, color: '#6a1b9a', marginBottom: 4 }}>{term}</div>
                <div style={{ lineHeight: 1.6 }}>{def}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
