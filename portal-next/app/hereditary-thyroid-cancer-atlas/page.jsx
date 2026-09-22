'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-thyroid-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'RET':     '#00695c',  // deep teal        — MEN2A/MEN2B/FMTC; MTC 95%; selpercatinib
  'TP53':    '#b71c1c',  // deep red         — LFS; anaplastic TC; AVOID RADIATION ABSOLUTELY
  'PTEN':    '#e65100',  // deep orange      — Cowden/PHTS; follicular TC; macrocephaly PATHOGNOMONIC
  'APC':     '#1565c0',  // deep blue        — FAP/Gardner; cribriform-morular PTC PATHOGNOMONIC
  'PRKAR1A': '#6a1b9a',  // deep purple      — Carney Complex; cardiac myxoma MANDATORY echo
  'DICER1':  '#2e7d32',  // forest green     — DICER1 Syndrome; MNG; PPB PATHOGNOMONIC
  'CDC73':   '#827717',  // dark yellow-olive — HPT-JT; parathyroid carcinoma; jaw ossifying fibroma
  'VHL':     '#283593',  // dark indigo      — VHL Disease; hemangioblastoma PATHOGNOMONIC; belzutifan
};

const GENE_INFO = {
  'RET':     { full: 'MEN2A/MEN2B/FMTC / MTC 95% / Thyroidectomy ≤6mo MEN2B / Selpercatinib',          locus: '10q11.21', size: '1114 aa / 124 kDa',  inh: 'AD GOF' },
  'TP53':    { full: 'LFS / Anaplastic TC / AVOID RADIATION ABSOLUTELY / WBMRI Toronto',                 locus: '17p13.1',  size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'PTEN':    { full: 'Cowden/PHTS / Follicular TC 25-38% / Macrocephaly PATHOGNOMONIC / mTOR',          locus: '10q23.31', size: '403 aa / 47 kDa',   inh: 'AD LOF' },
  'APC':     { full: 'FAP/Gardner / Cribriform-Morular PTC PATHOGNOMONIC / Annual US all FAP',           locus: '5q22.2',   size: '2843 aa / 310 kDa', inh: 'AD LOF' },
  'PRKAR1A': { full: 'Carney Complex / Follicular TC 75% / Cardiac Myxoma — Annual Echo MANDATORY',      locus: '17q24.2',  size: '381 aa / 43 kDa',   inh: 'AD LOF' },
  'DICER1':  { full: 'DICER1 Syndrome / MNG 75% females / PPB Type I PATHOGNOMONIC / SLCT ERMS',        locus: '14q32.13', size: '1922 aa / 218 kDa', inh: 'AD LOF' },
  'CDC73':   { full: 'HPT-JT / Parathyroid Ca PATHOGNOMONIC 10-15% / Jaw Ossifying Fibroma',            locus: '1q31.2',   size: '531 aa / 60 kDa',   inh: 'AD LOF' },
  'VHL':     { full: 'VHL Disease / Hemangioblastoma PATHOGNOMONIC / ccRCC / Belzutifan FDA 2021',       locus: '3p25.3',   size: '213 aa / 24 kDa',   inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color,
      border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px',
      fontSize: 11, fontWeight: 600, marginRight: 4,
    }}>{text}</span>
  );
}

export default function HereditaryThyroidCancerAtlas() {
  const [tab, setTab]                   = useState('Overview');
  const [overview, setOverview]         = useState(null);
  const [breakdown, setBreakdown]       = useState(null);
  const [definitions, setDefinitions]   = useState(null);
  const [loading, setLoading]           = useState(false);
  const [error, setError]               = useState(null);
  const [expandedGene, setExpandedGene] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const geneColor = g => GENE_COLORS[g] || '#607d8b';

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#0a0a0a', minHeight: '100vh', color: '#e8e8e8', padding: 24 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#001a0d 0%,#1a0033 60%,#001a2e 100%)', borderRadius: 12, padding: '28px 32px', marginBottom: 24 }}>
        <div style={{ fontSize: 11, color: '#80cbc4', letterSpacing: 2, textTransform: 'uppercase', marginBottom: 8 }}>
          Hereditary Disease Atlas · Thyroid Cancer Predisposition · 8-Gene Reference
        </div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800, color: '#fff' }}>
          🧬 Hereditary Thyroid Cancer Atlas
        </h1>
        <div style={{ marginTop: 10, color: '#b0bec5', fontSize: 13 }}>
          Complete 8-Gene Predisposition Reference · RET · TP53 · PTEN · APC · PRKAR1A · DICER1 · CDC73 · VHL
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(GENE_COLORS).map(([g, c]) => (
            <Badge key={g} text={g} color={c} />
          ))}
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: '#ef9a9a', fontWeight: 600 }}>
          ⚠ CRITICAL: RET M918T (MEN2B) → thyroidectomy ≤6 months — most urgent hereditary cancer surgery · TP53 LFS → AVOID RADIATION ABSOLUTELY · PRKAR1A → annual echo (cardiac myxoma: life-threatening)
        </div>
        <div style={{ marginTop: 6, fontSize: 11, color: '#80cbc4' }}>
          320-patient aggregate · 8 × 40 seeds · seeds 3134-3141 · APC cribriform-morular PTC PATHOGNOMONIC FAP · PTEN macrocephaly PATHOGNOMONIC · CDC73 parathyroid carcinoma en-bloc — NEVER disrupt capsule · VHL belzutifan HIF-2α FDA 2021
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#00695c' : '#1e1e1e',
            color: tab === t ? '#fff' : '#aaa', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#90caf9', padding: 40, textAlign: 'center' }}>Loading atlas data…</div>}
      {error   && <div style={{ color: '#ef9a9a', padding: 20, background: '#1a0000', borderRadius: 8 }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Genes',    val: overview.total_genes },
              { label: 'Total Patients', val: overview.total_patients },
              { label: 'Seed Range',     val: overview.seed_range },
              { label: 'Patients/Gene',  val: 40 },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: '#1e1e1e', borderRadius: 8, padding: '18px 20px', textAlign: 'center' }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: '#80cbc4' }}>{val}</div>
                <div style={{ fontSize: 12, color: '#888', marginTop: 4 }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Gene cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(340px,1fr))', gap: 16, marginBottom: 24 }}>
            {overview.genes.map(g => {
              const color = geneColor(g);
              const info  = GENE_INFO[g] || {};
              const inh   = (overview.inheritance_modes || {})[g] || '';
              return (
                <div key={g} style={{ background: '#1e1e1e', borderRadius: 8, padding: 18, borderLeft: `4px solid ${color}` }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                      <span style={{ fontSize: 20, fontWeight: 800, color }}>{g}</span>
                      <span style={{ fontSize: 11, color: '#888', marginLeft: 8 }}>{info.locus}</span>
                    </div>
                    <Badge text={info.inh || 'AD'} color={color} />
                  </div>
                  <div style={{ fontSize: 12, color: '#ccc', marginTop: 6 }}>{info.full}</div>
                  <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{info.size}</div>
                  {inh && (
                    <div style={{ fontSize: 11, color: '#b0bec5', marginTop: 8, background: '#111', borderRadius: 4, padding: '6px 8px' }}>
                      {inh.substring(0, 220)}{inh.length > 220 ? '…' : ''}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* RET urgent action box */}
          <div style={{ background: '#001a0d', border: '2px solid #00695c', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
            <div style={{ color: '#80cbc4', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
              ⚠ RET GENOTYPE-RISK STRATIFICATION — THYROIDECTOMY TIMING IS GENE-DEPENDENT
            </div>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              <li style={{ fontSize: 12, color: '#b2dfdb', marginBottom: 5 }}>
                <strong>Category D — M918T (MEN2B):</strong> Thyroidectomy ≤6 months of life — most urgent prophylactic cancer surgery in oncogenetics; delay risks lymph node metastasis
              </li>
              <li style={{ fontSize: 12, color: '#b2dfdb', marginBottom: 5 }}>
                <strong>Category C — C634/A883F (MEN2A):</strong> Thyroidectomy by age 5yr; annual calcitonin + CEA from age 3; annual plasma metanephrines from age 8
              </li>
              <li style={{ fontSize: 12, color: '#b2dfdb' }}>
                <strong>Category B — E768D/V804L/V804M (FMTC):</strong> Individualised 5-10yr if calcitonin normal; selpercatinib (selective RET TKI) for progressive/metastatic MTC
              </li>
            </ul>
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 16 }}>
            <h3 style={{ margin: '0 0 14px', color: '#80cbc4', fontSize: 15 }}>⚠ Key Clinical Rules</h3>
            <ul style={{ margin: 0, padding: '0 0 0 18px' }}>
              {(overview.key_clinical_rules || []).map((rule, i) => (
                <li key={i} style={{ fontSize: 12, color: '#ccc', marginBottom: 7, lineHeight: 1.5 }}>
                  {rule}
                </li>
              ))}
            </ul>
          </div>

          {/* Gene panel note */}
          {overview.gene_panel_note && (
            <div style={{ background: '#0d1b2a', borderRadius: 8, padding: 16, fontSize: 11, color: '#80cbc4', lineHeight: 1.7 }}>
              <strong style={{ color: '#90caf9' }}>Gene Panel &amp; Decision Tree:</strong>{' '}
              {overview.gene_panel_note}
            </div>
          )}
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          {breakdown.genes.map(g => {
            const color   = geneColor(g.gene);
            const isOpen  = expandedGene === g.gene;
            const info    = GENE_INFO[g.gene] || {};
            return (
              <div key={g.gene} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden', borderLeft: `4px solid ${color}` }}>
                <div
                  onClick={() => setExpandedGene(isOpen ? null : g.gene)}
                  style={{ padding: '14px 18px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <div>
                    <span style={{ fontSize: 17, fontWeight: 700, color }}>{g.gene}</span>
                    <span style={{ fontSize: 12, color: '#888', marginLeft: 10 }}>{g.locus} · {info.full}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 12, alignItems: 'center', fontSize: 12 }}>
                    <span style={{ color: '#aaa' }}>n={g.n}</span>
                    <span style={{ color: '#80cbc4' }}>Age {g.mean_age_diagnosis}yr</span>
                    <span style={{ color: isOpen ? '#fff' : '#666', fontSize: 16 }}>{isOpen ? '▲' : '▼'}</span>
                  </div>
                </div>
                {isOpen && (
                  <div style={{ padding: '0 18px 18px', borderTop: '1px solid #333' }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 10, marginTop: 14 }}>
                      {Object.entries(g)
                        .filter(([k]) => k.endsWith('_pct'))
                        .map(([k, v]) => (
                          <div key={k} style={{ background: '#111', borderRadius: 6, padding: '10px 12px' }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color }}>{v}%</div>
                            <div style={{ fontSize: 11, color: '#888', marginTop: 2 }}>
                              {k.replace(/_pct$/, '').replace(/_/g, ' ')}
                            </div>
                          </div>
                        ))}
                    </div>
                    {g.surveillance_key && (
                      <div style={{ marginTop: 12, fontSize: 11, color: '#80cbc4', background: '#0d1b2a', borderRadius: 4, padding: '8px 10px' }}>
                        <strong>Surveillance:</strong> {g.surveillance_key}
                      </div>
                    )}
                    {g.pathognomonic && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#ffcc80', background: '#1a1000', borderRadius: 4, padding: '8px 10px' }}>
                        <strong>Pathognomonic:</strong> {g.pathognomonic}
                      </div>
                    )}
                    {g.inheritance && (
                      <div style={{ marginTop: 8, fontSize: 11, color: '#b0bec5', background: '#111', borderRadius: 4, padding: '8px 10px', lineHeight: 1.6 }}>
                        <strong>Inheritance:</strong> {g.inheritance}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 16px', color: '#80cbc4', fontSize: 15 }}>Syndrome Summary — Hereditary Thyroid Cancer Predisposition</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#111' }}>
                    {['Gene', 'Syndrome', 'Locus', 'Size', 'Inheritance', 'Pathognomonic', 'Surveillance Key', 'n', 'Dx Age'].map(h => (
                      <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#80cbc4', borderBottom: '1px solid #333', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {breakdown.genes.map((g, idx) => {
                    const color = geneColor(g.gene);
                    const info  = GENE_INFO[g.gene] || {};
                    return (
                      <tr key={g.gene} style={{ background: idx % 2 === 0 ? '#181818' : '#1e1e1e' }}>
                        <td style={{ padding: '8px 10px', color, fontWeight: 700 }}>{g.gene}</td>
                        <td style={{ padding: '8px 10px', color: '#ccc', maxWidth: 200 }}>{info.full}</td>
                        <td style={{ padding: '8px 10px', color: '#aaa' }}>{g.locus}</td>
                        <td style={{ padding: '8px 10px', color: '#aaa' }}>{info.size}</td>
                        <td style={{ padding: '8px 10px', color: '#b0bec5' }}>{info.inh}</td>
                        <td style={{ padding: '8px 10px', color: '#ffcc80', fontSize: 11 }}>{g.pathognomonic}</td>
                        <td style={{ padding: '8px 10px', color: '#80cbc4', fontSize: 11 }}>{g.surveillance_key ? g.surveillance_key.split(';')[0] : '—'}</td>
                        <td style={{ padding: '8px 10px', color: '#e0e0e0' }}>{g.n}</td>
                        <td style={{ padding: '8px 10px', color: '#80cbc4' }}>{g.mean_age_diagnosis}yr</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Thyroid Ca pathway classification box */}
          <div style={{ background: '#0d1b2a', border: '1px solid #1565c0', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
            <div style={{ color: '#90caf9', fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
              Hereditary Thyroid Cancer — Pathway Classification
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(220px,1fr))', gap: 10 }}>
              {[
                { label: 'RET Kinase (MEN2/FMTC)', desc: 'MTC dominant · ATA risk D/C/B · Selpercatinib/Vandetanib/Cabozantinib', color: '#00695c' },
                { label: 'WNT / FAP (CMV-PTC)', desc: 'APC · Cribriform-morular PTC PATHOGNOMONIC · β-catenin nuclear IHC+', color: '#1565c0' },
                { label: 'PI3K / Cowden (FTC)', desc: 'PTEN · Follicular TC 25-38% · Macrocephaly/Lhermitte-Duclos PATHOGNOMONIC', color: '#e65100' },
                { label: 'cAMP / Carney (FTC)', desc: 'PRKAR1A · Follicular TC 75% · Cardiac myxoma annual echo MANDATORY', color: '#6a1b9a' },
                { label: 'HIF / VHL (ccTC)', desc: 'VHL · Clear-cell follicular TC · Hemangioblastoma PATHOGNOMONIC · Belzutifan', color: '#283593' },
                { label: 'miRNA / DICER1 (MNG/DTC)', desc: 'DICER1 · MNG 75% females · PPB Type I PATHOGNOMONIC infant', color: '#2e7d32' },
              ].map(({ label, desc, color }) => (
                <div key={label} style={{ background: '#111', borderRadius: 6, padding: '10px 12px', borderTop: `3px solid ${color}` }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color }}>{label}</div>
                  <div style={{ fontSize: 11, color: '#aaa', marginTop: 4 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Protein size reference */}
          <div style={{ background: '#1e1e1e', borderRadius: 8, padding: 20 }}>
            <h3 style={{ margin: '0 0 16px', color: '#80cbc4', fontSize: 15 }}>Protein Size Reference</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 10 }}>
              {Object.entries(GENE_INFO).map(([g, info]) => {
                const color = geneColor(g);
                return (
                  <div key={g} style={{ background: '#111', borderRadius: 6, padding: '12px 14px', borderTop: `3px solid ${color}` }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color }}>{g}</div>
                    <div style={{ fontSize: 11, color: '#888', marginTop: 4 }}>{info.size}</div>
                    <div style={{ fontSize: 11, color: '#aaa', marginTop: 2 }}>{info.locus}</div>
                    <div style={{ fontSize: 10, color: '#666', marginTop: 4 }}>{info.inh}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, fontSize: 12, color: '#888' }}>
            {definitions.definitions ? definitions.definitions.length : 0} clinical definitions · RET ATA risk stratification · APC cribriform-morular PTC FAP protocol · PRKAR1A Carney cardiac myxoma · VHL belzutifan HIF-2α · CDC73 parathyroid carcinoma en-bloc
          </div>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ background: '#1e1e1e', borderRadius: 8, marginBottom: 12, overflow: 'hidden' }}>
              <div style={{ background: '#00695c', padding: '10px 16px', fontSize: 13, fontWeight: 700, color: '#fff' }}>
                {d.term.replace(/-/g, ' ')}
              </div>
              <div style={{ padding: '14px 16px', fontSize: 12, color: '#ccc', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
                {d.definition}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
