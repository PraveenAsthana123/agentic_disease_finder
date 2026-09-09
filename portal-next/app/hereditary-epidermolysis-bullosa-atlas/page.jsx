'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-epidermolysis-bullosa-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  KRT5:    '#1565c0',  // deep blue   — EBS-WC/Koebner/DM, dominant-negative keratin
  KRT14:   '#b71c1c',  // deep red    — EBS-DM, p.Arg125His most common worldwide
  COL17A1: '#e65100',  // deep orange — JEB non-Herlitz, tooth loss + cervical cancer
  LAMB3:   '#880e4f',  // deep pink   — JEB-Herlitz, granulation tissue, airway emergency
  COL7A1:  '#2e7d32',  // deep green  — DEB, mitten hand, SCC, B-VEC gene therapy
  ITGB4:   '#4a148c',  // deep purple — JEB-PA, pyloric atresia neonatal emergency
  PLEC:    '#f57f17',  // amber       — EBS-MD, muscular dystrophy, cardiomyopathy
  FERMT1:  '#00695c',  // teal        — Kindler EB, UV-triggered, poikiloderma, colitis
};

const GENE_INFO = {
  KRT5:    { full: 'KRT5 / 590aa',    locus: '12q13.13', size: '590 aa / 62 kDa',   inh: 'AD', disease: 'EBS (Epidermolysis Bullosa Simplex) — TONOFILAMENT CLUMPING ON ELECTRON MICROSCOPY PATHOGNOMONIC (EBS-DM); dominant-negative collapse of keratin 5/14 IF network; intraepidermal basal split; Weber-Cockayne (localized palmoplantar, most common), Koebner (generalized), Dowling-Meara (DM, most severe, herpetiform clustering); heat worsens blistering; p.Glu477Lys most common EBS-DM; cool environment + wound care' },
  KRT14:   { full: 'KRT14 / 472aa',   locus: '17q21.2',  size: '472 aa / 52 kDa',   inh: 'AD', disease: 'EBS — p.Arg125His MOST COMMON KRT14 MUTATION WORLDWIDE (EBS-DM dominant); pairs with KRT5 to form keratin IF; mutations cluster in helix initiation/termination motifs; p.Arg125Cys milder DM; AR biallelic → severe generalized EBS ± muscular features; allele-specific siRNA trials ongoing; cool environment + wound care' },
  COL17A1: { full: 'COL17A1 / 1497aa', locus: '10q25.1', size: '1497 aa / 183 kDa', inh: 'AR', disease: 'JEB non-Herlitz (GABEB) — PREMATURE TOOTH LOSS + HYPOPLASTIC ENAMEL PATHOGNOMONIC; hemidesmosomal transmembrane collagen (BP180/BPAG2); lamina lucida split; atrophic scarring (NOT granulation tissue vs LAMB3); CERVICAL CANCER RISK 5-FOLD — annual Pap + HPV vaccine MANDATORY; GABEB = generalized atrophic benign phenotype; dentin/enamel defects all patients' },
  LAMB3:   { full: 'LAMB3 / 1172aa',  locus: '1q32.2',   size: '1172 aa / 140 kDa', inh: 'AR', disease: 'JEB-Herlitz (most severe JEB) — EXUBERANT PERIORAL/PERINASAL/DIGITAL GRANULATION TISSUE PATHOGNOMONIC; component of laminin-332; anchoring filaments absent → complete loss of attachment; AIRWAY GRANULATION TISSUE = LIFE-THREATENING emergency; p.Arg635X = 70% European JEB-H alleles; most patients die in infancy/early childhood; multidisciplinary specialist centre essential' },
  COL7A1:  { full: 'COL7A1 / 2944aa', locus: '3p21.31',  size: '2944 aa / 290 kDa', inh: 'AD/AR', disease: 'DEB (Dystrophic EB) — MITTEN HAND DEFORMITY / PSEUDOSYNDACTYLY PATHOGNOMONIC (severe RDEB); anchoring fibrils in sublamina densa; CUTANEOUS SCC >70% by age 45yr = LEADING CAUSE OF DEATH in RDEB; BEREMAGENE GEPERPAVEC (B-VEC / Vyjuvek) FDA2023 = FIRST EB GENE THERAPY; esophageal strictures >50%; dilations + nasogastric access; intensive dermatology surveillance mandatory' },
  ITGB4:   { full: 'ITGB4 / 1822aa',  locus: '17q25.1',  size: '1822 aa / 202 kDa', inh: 'AR', disease: 'JEB with Pyloric Atresia (JEB-PA) — PYLORIC ATRESIA AT BIRTH + EB BLISTERING = PATHOGNOMONIC COMBINATION; hemidesmosomal component paired with ITGA6; nonbilious vomiting + gastric bubble on X-ray = neonatal surgical emergency; renal USS baseline (hydronephrosis 10-20%); prognosis variable — severe (lethal) to mild (JEB-nH-like); surgical correction of PA first priority' },
  PLEC:    { full: 'PLEC / 4684aa',   locus: '8q24.13',  size: '4684 aa / 500+ kDa', inh: 'AR', disease: 'EBS with Muscular Dystrophy (EBS-MD) — BLISTERING IN INFANCY + PROGRESSIVE MUSCULAR DYSTROPHY IN ADULTHOOD = PATHOGNOMONIC COMBINATION; largest structural protein in human body; CARDIOMYOPATHY in 30% EBS-MD → echo + Holter monitoring from age 30yr MANDATORY; CK elevated = muscle involvement marker; plectin isoform-specific phenotypes; respiratory surveillance from teenage years' },
  FERMT1:  { full: 'FERMT1 / 677aa',  locus: '20p12.3',  size: '677 aa / 77 kDa',   inh: 'AR', disease: 'Kindler EB (KEB) — PHOTOSENSITIVITY + BLISTERING + PROGRESSIVE POIKILODERMA TRIAD = PATHOGNOMONIC; integrin activator (NOT a structural component — unique among EB genes); UV PROTECTION SPF50+/UPF50+ MANDATORY; COLITIS in >60%; GI surveillance mandatory; progressive skin atrophy and poikiloderma; SCC risk (lower than RDEB); UV-triggered blistering unique to KEB' },
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
          {data.eb_subtype && <span style={{ fontSize: 12, color: '#555' }}>Subtype: <b>{data.eb_subtype.split(' (')[0]}</b></span>}
          {data.n_patients !== undefined && <span style={{ fontSize: 12, color: '#555' }}>Patients: <b>{data.n_patients}</b></span>}
        </div>
      )}
    </div>
  );
}

export default function HeredEpidermolysisBuillosAtlasPage() {
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

  const TITLE = 'Hereditary Epidermolysis Bullosa Atlas';
  const SUBTITLE = 'Complete 8-Gene EB Atlas — KRT5 · KRT14 · COL17A1 · LAMB3 · COL7A1 · ITGB4 · PLEC · FERMT1';

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, color: '#1a237e', marginBottom: 4 }}>🧬 {TITLE}</h1>
        <p style={{ color: '#555', fontSize: 13, margin: 0 }}>{SUBTITLE}</p>
        <p style={{ color: '#888', fontSize: 12, marginTop: 4 }}>320-patient aggregate cohort · 8 × 40 · Seeds 2270–2277 · EBS / JEB / DEB / KEB spectrum</p>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 24, borderBottom: '2px solid #e0e0e0' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', border: 'none', background: tab === i ? '#1a237e' : 'transparent',
            color: tab === i ? '#fff' : '#555', borderRadius: '6px 6px 0 0',
            fontWeight: 700, cursor: 'pointer', fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888', padding: 40, textAlign: 'center' }}>Loading atlas data…</div>}
      {error && <div style={{ color: '#c62828', padding: 16, background: '#ffebee', borderRadius: 8 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {!loading && tab === 0 && overview && (
        <div>
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 24 }}>
            <StatCard label="Total Patients" value={overview.n_patients} color="#1a237e" />
            <StatCard label="Genes" value={overview.n_genes} sub="KRT5·KRT14·COL17A1·LAMB3·COL7A1·ITGB4·PLEC·FERMT1" color="#1565c0" />
            <StatCard label="EB Types" value={Object.keys(overview.eb_types || {}).length} sub="EBS/JEB/DEB/KEB" color="#e65100" />
            <StatCard label="Seeds" value={overview.seed_range} color="#2e7d32" />
          </div>

          {/* EB Type Map */}
          {overview.eb_types && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>EB Subtype–Gene Mapping</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                {Object.entries(overview.eb_types).map(([subtype, genes]) => (
                  <div key={subtype} style={{
                    background: '#e8eaf6', borderRadius: 8, padding: '8px 14px',
                    border: '1px solid #c5cae9',
                  }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: '#1a237e' }}>{subtype}</div>
                    <div style={{ fontSize: 11, color: '#555' }}>{genes}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Skin Split Level */}
          {overview.skin_split_levels && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Skin Split Levels by Gene</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#f5f5f5' }}>
                      <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e0e0e0' }}>Gene</th>
                      <th style={{ padding: '8px 12px', textAlign: 'left', borderBottom: '2px solid #e0e0e0' }}>Split Level</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(overview.skin_split_levels).map(([gene, level], i) => (
                      <tr key={gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa' }}>
                        <td style={{ padding: '7px 12px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                        <td style={{ padding: '7px 12px', color: '#444' }}>{level}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Key Pearls */}
          {overview.key_clinical_pearls && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Key Clinical Pearls — Pathognomonic & Critical</h3>
              <div>
                {overview.key_clinical_pearls.map((pearl, i) => {
                  const gene = pearl.split(':')[0].trim();
                  const color = GENE_COLORS[gene] || '#1565c0';
                  return (
                    <div key={i} style={{
                      padding: '10px 14px', marginBottom: 8,
                      background: color + '10', border: `1px solid ${color}33`,
                      borderLeft: `4px solid ${color}`, borderRadius: 6,
                      fontSize: 12, color: '#333', lineHeight: 1.6,
                    }}>
                      <strong style={{ color }}>{gene}:</strong> {pearl.split(':').slice(1).join(':').trim()}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Gene Summary */}
          {overview.gene_summary && (
            <div>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Gene Reference Summary</h3>
              {overview.gene_summary.map(g => (
                <GeneCard key={g.gene} gene={g.gene} color={GENE_COLORS[g.gene] || '#1565c0'} info={GENE_INFO[g.gene] || { locus: g.locus, inh: g.inheritance, size: g.protein_size, disease: g.pathognomonic }} data={g} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {!loading && tab === 1 && breakdown && (
        <div>
          <h2 style={{ fontWeight: 700, fontSize: 18, marginBottom: 16 }}>Gene-Level Clinical Breakdown</h2>
          {Object.entries(breakdown.gene_breakdown || {}).map(([gene, gd]) => {
            const color = GENE_COLORS[gene] || '#1565c0';
            return (
              <div key={gene} style={{ border: `2px solid ${color}`, borderRadius: 10, marginBottom: 20, overflow: 'hidden' }}>
                <div style={{ background: color, padding: '10px 16px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                  <span style={{ color: '#fff', fontWeight: 800, fontSize: 18 }}>{gene}</span>
                  <span style={{ color: '#fff', fontSize: 12, opacity: 0.9 }}>{gd.locus} · {gd.protein_size} · {gd.inheritance}</span>
                  <span style={{ color: '#fff', fontSize: 12, opacity: 0.9 }}>n={gd.n_patients}</span>
                </div>
                <div style={{ padding: 16 }}>
                  <div style={{ marginBottom: 10, fontSize: 13, color: '#1a237e', fontWeight: 700 }}>
                    EB Subtype: {gd.eb_subtype} | Split: {gd.skin_split_level}
                  </div>
                  {gd.pathognomonic && (
                    <div style={{ background: '#fffde7', border: '1px solid #f9a825', borderRadius: 6, padding: '8px 12px', marginBottom: 10, fontSize: 12 }}>
                      <strong>⚡ Pathognomonic:</strong> {gd.pathognomonic}
                    </div>
                  )}
                  {gd.treatment_highlight && (
                    <div style={{ background: '#e8f5e9', border: '1px solid #43a047', borderRadius: 6, padding: '8px 12px', marginBottom: 10, fontSize: 12 }}>
                      <strong>💊 Treatment Highlight:</strong> {gd.treatment_highlight}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 10 }}>
                    {gd.avg_age_at_dx_yrs !== undefined && (
                      <div style={{ fontSize: 12 }}><b>Avg Age Dx:</b> {gd.avg_age_at_dx_yrs}yr</div>
                    )}
                    {gd.avg_follow_up_yrs !== undefined && (
                      <div style={{ fontSize: 12 }}><b>Avg Follow-up:</b> {gd.avg_follow_up_yrs}yr</div>
                    )}
                  </div>

                  {gd.subtype_distribution && Object.keys(gd.subtype_distribution).length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Subtype Distribution:</div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {Object.entries(gd.subtype_distribution).map(([sub, pct]) => (
                          <Badge key={sub} text={`${sub}: ${pct}%`} color={color} />
                        ))}
                      </div>
                    </div>
                  )}

                  {gd.complication_distribution && Object.keys(gd.complication_distribution).length > 0 && (
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontWeight: 700, fontSize: 12, marginBottom: 4 }}>Complication Distribution:</div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {Object.entries(gd.complication_distribution).map(([comp, pct]) => (
                          <Badge key={comp} text={`${comp}: ${pct}%`} color="#555" />
                        ))}
                      </div>
                    </div>
                  )}

                  {gd.key_ddx && gd.key_ddx.length > 0 && (
                    <div style={{ marginTop: 8, fontSize: 12, color: '#666' }}>
                      <strong>DDx:</strong> {gd.key_ddx.join(' · ')}
                    </div>
                  )}
                </div>
              </div>
            );
          })}

          {breakdown.clinical_emergency_flags && breakdown.clinical_emergency_flags.length > 0 && (
            <div style={{ background: '#ffebee', border: '2px solid #c62828', borderRadius: 10, padding: 16, marginTop: 8 }}>
              <h3 style={{ fontWeight: 800, color: '#c62828', marginBottom: 10 }}>🚨 Clinical Emergency Flags</h3>
              {breakdown.clinical_emergency_flags.map((flag, i) => (
                <div key={i} style={{ fontSize: 12, marginBottom: 6, padding: '6px 10px', background: '#fff', borderRadius: 6, border: '1px solid #ef9a9a' }}>
                  {flag}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {!loading && tab === 2 && breakdown && (
        <div>
          <h2 style={{ fontWeight: 700, fontSize: 18, marginBottom: 16 }}>Clinical Atlas — All 8 Genes</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 }}>
            {Object.keys(GENE_INFO).map(gene => {
              const gd = (breakdown.gene_breakdown || {})[gene];
              const color = GENE_COLORS[gene] || '#1565c0';
              const info = GENE_INFO[gene];
              return (
                <div key={gene} style={{ border: `2px solid ${color}`, borderRadius: 10, padding: 16, background: color + '06' }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontWeight: 800, fontSize: 17, color }}>{gene}</span>
                    <Badge text={info.locus} color={color} />
                    <Badge text={info.inh} color={color} />
                  </div>
                  <div style={{ fontSize: 11, color: '#555', marginBottom: 8 }}>{info.size}</div>
                  <div style={{ fontSize: 12, color: '#333', lineHeight: 1.6, marginBottom: 8 }}>{info.disease.slice(0, 200)}…</div>
                  {gd && (
                    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: '#555' }}>Patients: <b>{gd.n_patients}</b></span>
                      {gd.avg_age_at_dx_yrs !== undefined && <span style={{ fontSize: 11, color: '#555' }}>Age Dx: <b>{gd.avg_age_at_dx_yrs}yr</b></span>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {!loading && tab === 3 && defs && (
        <div>
          <h2 style={{ fontWeight: 700, fontSize: 18, marginBottom: 16 }}>Definitions — EB Glossary</h2>

          {/* Gene Entries */}
          {defs.gene_entries && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Gene Reference Entries</h3>
              {Object.entries(defs.gene_entries).map(([gene, entry]) => {
                const color = GENE_COLORS[gene] || '#1565c0';
                return (
                  <div key={gene} style={{
                    border: `1px solid ${color}55`, borderLeft: `4px solid ${color}`,
                    borderRadius: 8, padding: '10px 14px', marginBottom: 10,
                  }}>
                    <div style={{ fontWeight: 700, color, marginBottom: 4 }}>{gene}</div>
                    {entry.full_name && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>Full name:</b> {entry.full_name}</div>}
                    {entry.locus && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>Locus:</b> {entry.locus}</div>}
                    {entry.protein_size && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>Protein:</b> {entry.protein_size}</div>}
                    {entry.inheritance && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>Inheritance:</b> {entry.inheritance}</div>}
                    {entry.eb_subtype && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>EB Subtype:</b> {entry.eb_subtype}</div>}
                    {entry.skin_split_level && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>Split level:</b> {entry.skin_split_level}</div>}
                    {entry.key_mutations && <div style={{ fontSize: 12, color: '#444', marginBottom: 2 }}><b>Key mutations:</b> {entry.key_mutations}</div>}
                    {entry.pathognomonic && <div style={{ fontSize: 12, color: '#c62828', fontWeight: 600, marginBottom: 2 }}><b>⚡ Pathognomonic:</b> {entry.pathognomonic}</div>}
                    {entry.treatment_note && <div style={{ fontSize: 12, color: '#1b5e20', marginBottom: 2 }}><b>Treatment:</b> {entry.treatment_note}</div>}
                  </div>
                );
              })}
            </div>
          )}

          {/* Skin Anatomy Glossary */}
          {defs.skin_anatomy_glossary && defs.skin_anatomy_glossary.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Skin Anatomy Glossary</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {defs.skin_anatomy_glossary.map((term, i) => (
                  <div key={i} style={{
                    background: '#e8eaf6', borderRadius: 6, padding: '6px 12px',
                    border: '1px solid #c5cae9', fontSize: 12, color: '#333',
                  }}>
                    {typeof term === 'string' ? term : JSON.stringify(term)}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* EB Subtype Glossary */}
          {defs.eb_subtype_glossary && defs.eb_subtype_glossary.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>EB Subtype Glossary</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {defs.eb_subtype_glossary.map((term, i) => (
                  <div key={i} style={{
                    background: '#fce4ec', borderRadius: 6, padding: '6px 12px',
                    border: '1px solid #f48fb1', fontSize: 12, color: '#333',
                  }}>
                    {typeof term === 'string' ? term : JSON.stringify(term)}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Treatment Glossary */}
          {defs.treatment_glossary && defs.treatment_glossary.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Treatment Glossary</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {defs.treatment_glossary.map((term, i) => (
                  <div key={i} style={{
                    background: '#e8f5e9', borderRadius: 6, padding: '6px 12px',
                    border: '1px solid #a5d6a7', fontSize: 12, color: '#333',
                  }}>
                    {typeof term === 'string' ? term : JSON.stringify(term)}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Diagnostic Tests */}
          {defs.diagnostic_tests && defs.diagnostic_tests.length > 0 && (
            <div style={{ marginBottom: 24 }}>
              <h3 style={{ fontWeight: 700, fontSize: 15, marginBottom: 10 }}>Diagnostic Tests</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {defs.diagnostic_tests.map((term, i) => (
                  <div key={i} style={{
                    background: '#fff3e0', borderRadius: 6, padding: '6px 12px',
                    border: '1px solid #ffcc80', fontSize: 12, color: '#333',
                  }}>
                    {typeof term === 'string' ? term : JSON.stringify(term)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
