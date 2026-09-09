'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  POU1F1: '#1565c0',  // deep blue — CPHD1 GH+TSH+PRL spares LH/FSH/ACTH
  PROP1:  '#4a148c',  // deep purple — most common CPHD2 evolving ACTH loss
  HESX1:  '#1b5e20',  // dark green — SOD optic nerve hypoplasia
  OTX2:   '#e65100',  // deep amber — eye anomalies dominant
  SOX3:   '#880e4f',  // dark pink — X-linked ID + GH
  LHX3:   '#006064',  // dark cyan — rigid neck PATHOGNOMONIC
  LHX4:   '#37474f',  // slate — Chiari + EPP + ACTH early
  GLI2:   '#bf360c',  // deep red-orange — HPE spectrum most variable
};

const GENE_DISEASE = {
  POU1F1: 'CPHD1 (AR/AD-DN) — GH+PRL+TSH triple deficiency; LH/FSH/ACTH/ADH SPARED PATHOGNOMONIC; anterior pituitary hypoplasia; normal posterior pituitary',
  PROP1:  'CPHD2 (AR) — MOST COMMON CPHD globally; GH+TSH+PRL+LH/FSH; ACTH evolves LATE adulthood; pituitary mass→involution characteristic MRI',
  HESX1:  'SOD/de Morsier (AR/AD) — optic nerve hypoplasia+absent septum pellucidum+pituitary hypoplasia; pendular nystagmus birth first sign; ACTH deficiency DANGEROUS',
  OTX2:   'Pituitary hypoplasia+eye anomalies (AD) — anophthalmia/microphthalmia/coloboma dominant; GH deficiency; ectopic posterior pituitary; variable expressivity',
  SOX3:   'X-linked hypopituitarism+ID (XLR) — males affected; GH deficiency; intellectual disability common; infundibular hypoplasia on MRI',
  LHX3:   'CPHD3 (AR) — GH+TSH+PRL+LH/FSH; RIGID CERVICAL SPINE PATHOGNOMONIC (cannot rotate neck); SNHL in ~50%; ectopic posterior pituitary',
  LHX4:   'CPHD4 (AD) — GH+TSH+ACTH; ACTH deficient FROM BIRTH (unlike PROP1 late); Arnold-Chiari malformation; ectopic posterior pituitary; variable penetrance',
  GLI2:   'HPE9 (AD) — MOST VARIABLE EXPRESSIVITY; single central incisor MIDLINE MARKER; PSIS; HPE spectrum; GH+ACTH+TSH deficiency with stalk interruption',
};

const INHERITANCE = {
  POU1F1: 'AR/AD-DN', PROP1: 'AR', HESX1: 'AR/AD', OTX2: 'AD',
  SOX3: 'XLR', LHX3: 'AR', LHX4: 'AD', GLI2: 'AD',
};

const PITUITARY_GROUP = {
  POU1F1: 'CPHD1 / PIT-1 TF',
  PROP1:  'CPHD2 / Prophet-of-PIT-1',
  HESX1:  'SOD / de Morsier Syndrome',
  OTX2:   'Eye Anomaly + Pituitary',
  SOX3:   'X-linked / Infundibular',
  LHX3:   'CPHD3 / Rigid Neck',
  LHX4:   'CPHD4 / Chiari + ACTH Early',
  GLI2:   'HPE9 / PSIS + Single Incisor',
};

export default function HereditaryHypopituitarismAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedGene, setSelectedGene] = useState('POU1F1');

  useEffect(() => {
    async function load() {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/hereditary-hypopituitarism-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-hypopituitarism-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-hypopituitarism-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov);
        setBreakdown(bk);
        setDefinitions(df);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#fff', background: '#121212', minHeight: '100vh' }}>Loading Hereditary Hypopituitarism Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#f44', background: '#121212', minHeight: '100vh' }}>Error: {error}</div>;

  const genes = Object.keys(GENE_COLORS);

  return (
    <div style={{ background: '#121212', minHeight: '100vh', color: '#fff', padding: '24px 32px', fontFamily: 'monospace' }}>
      <h1 style={{ color: '#26a69a', fontSize: 22, marginBottom: 4 }}>
        🧬 Hereditary Hypopituitarism Atlas
      </h1>
      <p style={{ color: '#80cbc4', fontSize: 13, marginBottom: 20 }}>
        Complete 8-Gene Reference · POU1F1 · PROP1 · HESX1 · OTX2 · SOX3 · LHX3 · LHX4 · GLI2 ·
        {overview && ` ${overview.total_patients} patients · seeds ${overview.seed_range}`}
      </p>

      {/* Gene legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
        {Object.entries(GENE_COLORS).map(([gene, col]) => (
          <span key={gene} style={{
            background: col, color: '#fff', padding: '3px 10px',
            borderRadius: 4, fontSize: 11, fontWeight: 700,
          }}>{gene} · {INHERITANCE[gene]} · {PITUITARY_GROUP[gene]}</span>
        ))}
      </div>

      {/* Critical clinical alerts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
        {[
          { label: '🔴 LHX4: ACTH deficient FROM BIRTH — start hydrocortisone at diagnosis immediately; adrenal crisis from neonatal period', col: '#37474f' },
          { label: '🚨 HESX1/SOD: ALL SOD patients carry IM HC kit — ACTH deficiency kills if missed; Synacthen test MANDATORY', col: '#1b5e20' },
          { label: '⚡ PROP1: Annual Synacthen from adolescence — evolving ACTH loss in adulthood; anticipate adrenal crisis', col: '#4a148c' },
          { label: '🧠 POU1F1: GH+PRL+TSH deficient but LH/FSH/ACTH ALWAYS intact — NO fludrocortisone EVER; puberty normal', col: '#1565c0' },
          { label: '🦴 LHX3: RIGID NECK — alert anaesthetist; awake fiberoptic intubation for surgery; NO neck hyperextension', col: '#006064' },
          { label: '👁️ GLI2: Single central incisor = midline marker + short stature → MRI + GLI2 test IMMEDIATELY', col: '#bf360c' },
        ].map((a, i) => (
          <div key={i} style={{
            background: a.col + '33', border: `1px solid ${a.col}`,
            borderRadius: 6, padding: '6px 12px', fontSize: 11, color: '#fff', maxWidth: 480,
          }}>{a.label}</div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: tab === t ? '#26a69a' : '#1e1e1e',
            color: tab === t ? '#000' : '#aaa',
            border: '1px solid #333', borderRadius: 4,
            padding: '6px 16px', cursor: 'pointer', fontSize: 12, fontWeight: 700,
          }}>{t}</button>
        ))}
      </div>

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          {/* Summary cards */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, col: '#26a69a' },
              { label: 'Genes Covered', value: overview.genes_covered, col: '#81c784' },
              { label: 'Per Gene Cohort', value: overview.cohort_size_per_gene, col: '#ffb74d' },
              { label: 'Seeds', value: overview.seed_range, col: '#ce93d8' },
            ].map(c => (
              <div key={c.label} style={{
                background: '#1a1a2e', border: `1px solid ${c.col}33`,
                borderRadius: 8, padding: '14px 20px', minWidth: 140,
              }}>
                <div style={{ fontSize: 11, color: '#aaa', marginBottom: 4 }}>{c.label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: c.col }}>{c.value}</div>
              </div>
            ))}
          </div>

          {/* Key diagnostic tests */}
          {overview.key_diagnostic_tests && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: '#26a69a', fontSize: 14, marginBottom: 10 }}>🔬 Key Diagnostic Tests</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {overview.key_diagnostic_tests.map((t, i) => (
                  <div key={i} style={{
                    background: '#1a2a3a', border: '1px solid #26a69a44',
                    borderRadius: 6, padding: '5px 10px', fontSize: 11, color: '#b2dfdb',
                  }}>{t}</div>
                ))}
              </div>
            </div>
          )}

          {/* Emergency rules */}
          {overview.key_emergency_rules && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ color: '#ef5350', fontSize: 14, marginBottom: 10 }}>🚨 Emergency Rules</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {overview.key_emergency_rules.map((r, i) => (
                  <div key={i} style={{
                    background: '#2a1a1a', border: '1px solid #ef535044',
                    borderRadius: 6, padding: '6px 12px', fontSize: 11, color: '#ffcdd2',
                  }}>{r}</div>
                ))}
              </div>
            </div>
          )}

          {/* Cohort breakdown table */}
          {overview.cohort_breakdown && (
            <div>
              <h3 style={{ color: '#81c784', fontSize: 14, marginBottom: 10 }}>📊 Cohort by Gene</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr>
                      {['Gene', 'Locus', 'Protein Size', 'Patients', 'Avg Age Dx', 'Disease Summary'].map(h => (
                        <th key={h} style={{
                          background: '#1a2a1a', color: '#81c784', padding: '8px 10px',
                          textAlign: 'left', borderBottom: '1px solid #333', whiteSpace: 'nowrap',
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {overview.cohort_breakdown.map((row, i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#1a1a1a' : '#1e1e1e' }}>
                        <td style={{ padding: '6px 10px', color: GENE_COLORS[row.gene] || '#fff', fontWeight: 700 }}>{row.gene}</td>
                        <td style={{ padding: '6px 10px', color: '#ccc' }}>{row.locus}</td>
                        <td style={{ padding: '6px 10px', color: '#ccc' }}>{row.protein_size}</td>
                        <td style={{ padding: '6px 10px', color: '#fff', textAlign: 'center' }}>{row.patients}</td>
                        <td style={{ padding: '6px 10px', color: '#ffb74d', textAlign: 'center' }}>{row.avg_age_at_dx}y</td>
                        <td style={{ padding: '6px 10px', color: '#b0bec5', fontSize: 11 }}>{row.disease_summary}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr>
                  {['Gene', 'Inheritance', 'Locus', 'Protein', 'Group', 'Key Disease / Clinical Pearl'].map(h => (
                    <th key={h} style={{
                      background: '#1a1a2e', color: '#26a69a', padding: '8px 10px',
                      textAlign: 'left', borderBottom: '1px solid #333', whiteSpace: 'nowrap',
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {genes.map((gene, i) => (
                  <tr key={gene} style={{ background: i % 2 === 0 ? '#1a1a1a' : '#1e1e1e' }}>
                    <td style={{ padding: '7px 10px', color: GENE_COLORS[gene], fontWeight: 700, fontSize: 13 }}>{gene}</td>
                    <td style={{ padding: '7px 10px', color: '#ce93d8' }}>{INHERITANCE[gene]}</td>
                    <td style={{ padding: '7px 10px', color: '#80cbc4' }}>{breakdown?.breakdown_by_gene?.[gene]?.locus || '—'}</td>
                    <td style={{ padding: '7px 10px', color: '#b0bec5' }}>{breakdown?.breakdown_by_gene?.[gene]?.protein_size || '—'}</td>
                    <td style={{ padding: '7px 10px', color: '#ffcc80' }}>{PITUITARY_GROUP[gene]}</td>
                    <td style={{ padding: '7px 10px', color: '#eceff1', fontSize: 11 }}>{GENE_DISEASE[gene]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {/* Gene selector */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
            {genes.map(g => (
              <button key={g} onClick={() => setSelectedGene(g)} style={{
                background: selectedGene === g ? GENE_COLORS[g] : '#1e1e1e',
                color: '#fff', border: `1px solid ${GENE_COLORS[g]}`,
                borderRadius: 4, padding: '5px 14px', cursor: 'pointer', fontSize: 12, fontWeight: 700,
              }}>{g}</button>
            ))}
          </div>

          {breakdown.breakdown_by_gene?.[selectedGene] && (() => {
            const gd = breakdown.breakdown_by_gene[selectedGene];
            return (
              <div>
                {/* Gene header */}
                <div style={{
                  background: GENE_COLORS[selectedGene] + '22',
                  border: `1px solid ${GENE_COLORS[selectedGene]}`,
                  borderRadius: 8, padding: 16, marginBottom: 16,
                }}>
                  <h2 style={{ color: GENE_COLORS[selectedGene], fontSize: 18, margin: 0 }}>
                    {selectedGene} — {PITUITARY_GROUP[selectedGene]}
                  </h2>
                  <p style={{ color: '#b0bec5', fontSize: 12, margin: '6px 0 0' }}>
                    {gd.locus} · {gd.protein_size} · {INHERITANCE[selectedGene]}
                  </p>
                </div>

                {/* Key features */}
                {gd.key_features && (
                  <div style={{ marginBottom: 16 }}>
                    <h3 style={{ color: '#26a69a', fontSize: 13, marginBottom: 8 }}>⭐ Key Features</h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      {gd.key_features.map((f, i) => (
                        <div key={i} style={{
                          background: '#1a1a2e', border: '1px solid #26a69a33',
                          borderRadius: 5, padding: '5px 10px', fontSize: 11, color: '#e0f2f1',
                        }}>• {f}</div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Clinical sections in 2-col grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
                  {[
                    { label: '🧬 Inheritance / Mechanism', text: gd.inheritance },
                    { label: '🔬 Pathognomonic Pearls', text: gd.pathognomonic },
                    { label: '⚗️ Disease Pathway', text: gd.disease_pathway },
                    { label: '💊 Treatment', text: gd.treatment },
                  ].map(sec => sec.text ? (
                    <div key={sec.label} style={{
                      background: '#1a1a1a', border: '1px solid #333',
                      borderRadius: 8, padding: 14,
                    }}>
                      <h4 style={{ color: '#90caf9', fontSize: 12, marginBottom: 8 }}>{sec.label}</h4>
                      <p style={{ color: '#cfd8dc', fontSize: 11, lineHeight: 1.6, margin: 0, whiteSpace: 'pre-line' }}>
                        {sec.text}
                      </p>
                    </div>
                  ) : null)}
                </div>

                {/* DDx */}
                {gd.key_ddx && (
                  <div style={{
                    background: '#1a1a2e', border: '1px solid #7986cb44',
                    borderRadius: 8, padding: 14, marginBottom: 16,
                  }}>
                    <h4 style={{ color: '#9fa8da', fontSize: 12, marginBottom: 6 }}>⚖️ Key DDx</h4>
                    <p style={{ color: '#c5cae9', fontSize: 11, lineHeight: 1.6, margin: 0 }}>{gd.key_ddx}</p>
                  </div>
                )}

                {/* Emergency protocol */}
                {gd.emergency_protocol && (
                  <div style={{
                    background: '#2a1a1a', border: '1px solid #ef5350',
                    borderRadius: 8, padding: 14, marginBottom: 16,
                  }}>
                    <h4 style={{ color: '#ef5350', fontSize: 12, marginBottom: 6 }}>🚨 Emergency Protocol</h4>
                    <p style={{ color: '#ffcdd2', fontSize: 11, lineHeight: 1.6, margin: 0, whiteSpace: 'pre-line' }}>{gd.emergency_protocol}</p>
                  </div>
                )}

                {/* Systemic involvement */}
                {gd.systemic_involvement && (
                  <div style={{ marginBottom: 16 }}>
                    <h4 style={{ color: '#a5d6a7', fontSize: 12, marginBottom: 8 }}>🫀 Systemic Involvement</h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                      {Object.entries(gd.systemic_involvement).map(([sys, detail]) => (
                        <div key={sys} style={{
                          background: '#1a2a1a', border: '1px solid #2e7d3244',
                          borderRadius: 6, padding: '6px 10px', fontSize: 11,
                        }}>
                          <span style={{ color: '#81c784', fontWeight: 700, textTransform: 'capitalize' }}>{sys.replace(/_/g, ' ')}: </span>
                          <span style={{ color: '#c8e6c9' }}>{detail}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Cascade testing */}
                {gd.cascade_testing && (
                  <div style={{
                    background: '#1a1a2e', border: '1px solid #26a69a44',
                    borderRadius: 8, padding: 14, marginBottom: 16,
                  }}>
                    <h4 style={{ color: '#4db6ac', fontSize: 12, marginBottom: 6 }}>🔗 Cascade Testing</h4>
                    <p style={{ color: '#b2dfdb', fontSize: 11, lineHeight: 1.5, margin: 0 }}>{gd.cascade_testing}</p>
                  </div>
                )}

                {/* Presentation distribution */}
                {gd.presentation_distribution && (
                  <div style={{ marginBottom: 16 }}>
                    <h4 style={{ color: '#ffb74d', fontSize: 12, marginBottom: 8 }}>📈 Presentation Distribution (n=40)</h4>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {Object.entries(gd.presentation_distribution).map(([pres, cnt]) => (
                        <div key={pres} style={{
                          background: '#2a2010', border: '1px solid #f57c0044',
                          borderRadius: 4, padding: '4px 8px', fontSize: 11,
                          color: '#ffe082',
                        }}>{pres}: <strong>{cnt}</strong></div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#26a69a', fontSize: 14, marginBottom: 16 }}>
            📖 Clinical Definitions — Hereditary Hypopituitarism Atlas
          </h3>

          {definitions.key_definitions && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 24 }}>
              {Object.entries(definitions.key_definitions).map(([term, defn]) => (
                <div key={term} style={{
                  background: '#1a1a2e', border: '1px solid #26a69a33',
                  borderRadius: 8, padding: 14,
                }}>
                  <div style={{ color: '#26a69a', fontWeight: 700, fontSize: 13, marginBottom: 6 }}>
                    {term.replace(/_/g, ' ')}
                  </div>
                  <div style={{ color: '#b0bec5', fontSize: 11, lineHeight: 1.6 }}>{defn}</div>
                </div>
              ))}
            </div>
          )}

          {definitions.key_drug_contraindications && (
            <div style={{ marginBottom: 24 }}>
              <h4 style={{ color: '#ef5350', fontSize: 13, marginBottom: 10 }}>⚠️ Key Drug Contraindications</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {definitions.key_drug_contraindications.map((ci, i) => (
                  <div key={i} style={{
                    background: '#2a1a1a', border: '1px solid #ef535044',
                    borderRadius: 6, padding: '6px 12px', fontSize: 11, color: '#ffcdd2',
                  }}>⛔ {ci}</div>
                ))}
              </div>
            </div>
          )}

          {definitions.genes_in_atlas && (
            <div style={{
              background: '#1a1a1a', border: '1px solid #333',
              borderRadius: 8, padding: 14,
            }}>
              <h4 style={{ color: '#81c784', fontSize: 12, marginBottom: 8 }}>🧬 Genes in Atlas</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {definitions.genes_in_atlas.map(g => (
                  <span key={g} style={{
                    background: GENE_COLORS[g] || '#333', color: '#fff',
                    padding: '3px 10px', borderRadius: 4, fontSize: 12, fontWeight: 700,
                  }}>{g}</span>
                ))}
              </div>
              <p style={{ color: '#78909c', fontSize: 11, marginTop: 8 }}>
                Total patients modelled: {definitions.total_patients_modelled} · Seeds: {definitions.seeds?.join(', ')}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
