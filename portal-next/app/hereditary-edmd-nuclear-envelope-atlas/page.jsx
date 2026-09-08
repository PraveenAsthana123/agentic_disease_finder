'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-edmd-nuclear-envelope-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  EMD:    '#b71c1c',  // deep red     — XLR EDMD1, emerin absent, pacemaker/ICD mandatory
  LMNA:   '#1565c0',  // deep blue    — most allelic gene, DCM-CD, ICD early
  SYNE1:  '#2e7d32',  // deep green   — ARCA1 Quebec, LINC outer NM, 8797aa
  SYNE2:  '#6a1b9a',  // deep purple  — EDMD5 AD, LINC outer NM, very rare
  TMEM43: '#c62828',  // vivid red    — ARVC5, S358L Newfoundland, ICD MANDATORY ALL CARRIERS
  FHL1:   '#e65100',  // deep orange  — XLR EDMD6, reducing body myopathy, female HCM
  SUN1:   '#00695c',  // deep teal    — LINC inner NM, SUN domain, EDMD-like+DCM
  LEMD3:  '#37474f',  // dark grey    — BOS, osteopoikilosis, TGF-β, mostly benign
};

const GENE_INFO = {
  EMD:    { full: 'EMD / 254aa',    locus: 'Xq28',    size: '254 aa / 29 kDa',         inh: 'XLR', disease: 'EDMD1 — ELBOW FLEXION CONTRACTURES EARLIEST PATHOGNOMONIC / emerin IHC ABSENT muscle+skin biopsy / PACEMAKER+ICD MANDATORY SCD RISK / rigid spine / humeroperoneal / XLR males; female carriers cardiac' },
  LMNA:   { full: 'LMNA / 664aa',   locus: '1q22',    size: '664 aa / 74 kDa',          inh: 'AD',  disease: 'EDMD2/LGMD1B/DCM-CD/FPLD2 — MOST ALLELIC GENE 15+ PHENOTYPES / ICD MANDATORY EARLY DCM / SCD RISK even preserved EF / lonafarnib HGPS FDA2020 / laminopathy spectrum' },
  SYNE1:  { full: 'SYNE1 / 8797aa', locus: '6q25.2',  size: '8797 aa / ~1 MDa',         inh: 'AR/AD', disease: 'ARCA1 Quebec founder (AR cerebellar ataxia non-progressive) + EDMD4 (AD) / LINC complex outer NM / largest known human protein / KASH domain / WES challenging' },
  SYNE2:  { full: 'SYNE2 / 6885aa', locus: '14q23.2', size: '6885 aa / ~800 kDa',       inh: 'AD',  disease: 'EDMD5 — LINC complex outer NM Nesprin-2 / KASH domain SUN-binding / AD phenotype identical SYNE1 EDMD4 / very rare <20 families / cardiac surveillance mandatory' },
  TMEM43: { full: 'TMEM43 / 400aa', locus: '3p25.1',  size: '400 aa / 44 kDa',          inh: 'AD',  disease: 'ARVC5/EDMD7 — S358L NEWFOUNDLAND FOUNDER 100% PENETRANCE MALES LETHAL / ICD MANDATORY ALL CARRIERS / SCD mean 41yr untreated / biventricular cardiomyopathy' },
  FHL1:   { full: 'FHL1 / 323aa',   locus: 'Xq26.3',  size: '323 aa / 32 kDa',          inh: 'XLR', disease: 'EDMD6/Scapuloperoneal/Reducing Body Myopathy/HCM — REDUCING BODY MYOPATHY biopsy PATHOGNOMONIC (menadione-nitro BT stain) / XLR males / female carriers HCM risk / scapular winging' },
  SUN1:   { full: 'SUN1 / 916aa',   locus: '7q32.2',  size: '916 aa / 103 kDa',         inh: 'AR/AD', disease: 'LINC complex inner NM SUN domain — EDMD-like + DCM / SUN-KASH bridge with SYNE1/2 / nuclear-cytoskeletal force transduction / very rare / full LINC panel mandatory' },
  LEMD3:  { full: 'LEMD3 / 922aa',  locus: '12q14.3', size: '922 aa / 103 kDa',         inh: 'AD',  disease: 'Buschke-Ollendorff BOS — OSTEOPOIKILOSIS+DERMATOFIBROSIS LENTICULARIS DUAL PATHOGNOMONIC / TGF-β SMAD antagonism / melorheostosis allelic / mostly benign / dermatology+orthopaedics' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22',
      color,
      border: `1px solid ${color}55`,
      borderRadius: 4,
      padding: '2px 7px',
      fontSize: 11,
      fontWeight: 700,
      marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff',
      border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10,
      padding: '14px 18px',
      minWidth: 120,
      textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

export default function HereditaryEDMDNuclearEnvelopeAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Definitions' ? 'definitions'
      : tab === 'Gene Table' || tab === 'Clinical Atlas' ? 'breakdown' : 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.status))
      .then(data => {
        if (ep === 'overview') setOverview(data);
        else if (ep === 'breakdown') setBreakdown(data);
        else setDefinitions(data);
        setLoading(false);
      })
      .catch(e => { setError(String(e)); setLoading(false); });
  }, [tab]);

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#b71c1c 0%,#1565c0 40%,#c62828 70%,#37474f 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, marginBottom: 4 }}>HEREDITARY EDMD / NUCLEAR ENVELOPE MYOPATHY ATLAS</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, lineHeight: 1.2 }}>
          Complete 8-Gene EDMD &amp; Nuclear Envelope Spectrum
        </h1>
        <div style={{ fontSize: 13, opacity: 0.9, marginTop: 6 }}>
          EMD · LMNA · SYNE1 · SYNE2 · TMEM43 · FHL1 · SUN1 · LEMD3 &nbsp;|&nbsp; 320 patients · seeds 2198-2205
        </div>
        {/* Gene colour chips */}
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 12 }}>
          {Object.entries(GENE_COLORS).map(([g, c]) => (
            <span key={g} style={{ background: c, color: '#fff', borderRadius: 5, padding: '3px 10px', fontSize: 11, fontWeight: 700 }}>{g}</span>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '2px solid #e0e0e0', display: 'flex', padding: '0 24px' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 800 : 400,
            color: tab === t ? '#b71c1c' : '#555',
            borderBottom: tab === t ? '3px solid #b71c1c' : '3px solid transparent',
            fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>
        {loading && <div style={{ color: '#888', padding: 32, textAlign: 'center' }}>Loading…</div>}
        {error && <div style={{ color: '#c62828', padding: 16, background: '#ffebee', borderRadius: 8 }}>Error: {error}</div>}

        {/* ── OVERVIEW TAB ── */}
        {tab === 'Overview' && overview && !loading && (
          <div>
            {/* Stats row */}
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
              <StatCard label="Genes Covered" value={overview.n_genes} color="#b71c1c" />
              <StatCard label="Total Patients" value={overview.n_patients} color="#1565c0" />
              <StatCard label="Alive %" value={`${overview.alive_pct}%`} color="#2e7d32" />
              <StatCard label="Ambulant 10yr" value={`${overview.ambulant_10yr_pct}%`} color="#6a1b9a" />
              <StatCard label="Mean Onset" value={`${overview.mean_age_onset_yr}yr`} color="#e65100" />
              <StatCard label="Dx Delay" value={`${overview.mean_dx_delay_yr}yr`} color="#00695c" />
              <StatCard label="Mean CK" value={`${overview.mean_ck_iu_l}`} sub="IU/L" color="#37474f" />
              <StatCard label="Seeds" value={overview.seeds} color="#c62828" />
            </div>

            {/* Key spectrum facts */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 12px', color: '#b71c1c', fontSize: 15 }}>Key Spectrum Facts — Clinical Pearls</h3>
              <ul style={{ margin: 0, paddingLeft: 20 }}>
                {overview.key_spectrum_facts?.map((f, i) => (
                  <li key={i} style={{ marginBottom: 6, fontSize: 13, lineHeight: 1.5 }}>
                    <span style={{ fontWeight: 700, color: GENE_COLORS[f.split(':')[0]] || '#333' }}>{f.split(':')[0]}:</span>
                    {f.slice(f.indexOf(':') + 1)}
                  </li>
                ))}
              </ul>
            </div>

            {/* Critical DDx */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 12px', color: '#1565c0', fontSize: 15 }}>Critical Differential Diagnoses</h3>
              {overview.critical_ddx && Object.entries(overview.critical_ddx).map(([pair, desc]) => (
                <div key={pair} style={{ marginBottom: 8, padding: '8px 12px', background: '#e3f2fd', borderRadius: 6, fontSize: 13 }}>
                  <strong style={{ color: '#1565c0' }}>{pair.replace(/_/g, ' ')}</strong>: {desc}
                </div>
              ))}
            </div>

            {/* Gene summary cards */}
            <h3 style={{ color: '#333', fontSize: 15, margin: '0 0 12px' }}>Gene-Level Summary</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(320px,1fr))', gap: 14 }}>
              {overview.gene_summaries?.map(gs => (
                <div key={gs.gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[gs.gene]}44`, borderRadius: 10, padding: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <span style={{ background: GENE_COLORS[gs.gene], color: '#fff', borderRadius: 5, padding: '3px 10px', fontWeight: 800, fontSize: 13 }}>{gs.gene}</span>
                    <span style={{ fontSize: 12, color: '#666' }}>{gs.locus} · {gs.protein_size}</span>
                  </div>
                  <div style={{ fontSize: 12, color: '#444', marginBottom: 6 }}><strong>Inheritance:</strong> {gs.inheritance}</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 12 }}>
                    <span style={{ background: '#e8f5e9', padding: '2px 7px', borderRadius: 4 }}>Onset: <b>{gs.mean_onset_yr}yr</b></span>
                    <span style={{ background: '#fce4ec', padding: '2px 7px', borderRadius: 4 }}>CK: <b>{gs.mean_ck} IU/L</b></span>
                    <span style={{ background: '#e3f2fd', padding: '2px 7px', borderRadius: 4 }}>Alive: <b>{gs.alive_pct}%</b></span>
                    <span style={{ background: '#f3e5f5', padding: '2px 7px', borderRadius: 4 }}>Ambulant 10yr: <b>{gs.ambulant_10yr_pct}%</b></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {tab === 'Gene Table' && breakdown && !loading && (
          <div>
            <h3 style={{ margin: '0 0 16px', color: '#333' }}>Per-Gene Clinical Data Table</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, background: '#fff', borderRadius: 10, overflow: 'hidden', boxShadow: '0 1px 4px #0001' }}>
                <thead>
                  <tr style={{ background: '#37474f', color: '#fff' }}>
                    {['Gene','Locus','Size','Inh.','n','Onset(yr)','CK(IU/L)','Alive%','Ambul.10yr%','Treated%','Dx Delay'].map(h => (
                      <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(breakdown).map(([gene, d], idx) => (
                    <tr key={gene} style={{ background: idx % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 800, color: GENE_COLORS[gene] }}>{gene}</td>
                      <td style={{ padding: '10px 12px' }}>{d.locus}</td>
                      <td style={{ padding: '10px 12px', whiteSpace: 'nowrap' }}>{d.protein_size}</td>
                      <td style={{ padding: '10px 12px' }}>
                        <Badge text={d.inheritance.split(' ')[0]} color={GENE_COLORS[gene]} />
                      </td>
                      <td style={{ padding: '10px 12px' }}>{d.n_patients}</td>
                      <td style={{ padding: '10px 12px' }}>{d.mean_age_onset}</td>
                      <td style={{ padding: '10px 12px' }}>{d.mean_ck_iu_l.toLocaleString()}</td>
                      <td style={{ padding: '10px 12px' }}>{d.alive_pct}%</td>
                      <td style={{ padding: '10px 12px' }}>{d.ambulant_10yr_pct}%</td>
                      <td style={{ padding: '10px 12px' }}>{d.treated_pct}%</td>
                      <td style={{ padding: '10px 12px' }}>{d.mean_dx_delay_yr}yr</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Key features per gene */}
            <div style={{ marginTop: 24, display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
              {Object.entries(breakdown).map(([gene, d]) => (
                <div key={gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[gene]}44`, borderRadius: 10, padding: 16 }}>
                  <div style={{ fontWeight: 800, color: GENE_COLORS[gene], fontSize: 14, marginBottom: 8 }}>{gene} — Key Features</div>
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {d.key_features?.map((f, i) => <li key={i} style={{ fontSize: 12, marginBottom: 4, lineHeight: 1.4 }}>{f}</li>)}
                  </ul>
                  {d.contraindications?.length > 0 && (
                    <div style={{ marginTop: 8, background: '#ffebee', borderRadius: 6, padding: '6px 10px' }}>
                      <div style={{ fontWeight: 700, color: '#c62828', fontSize: 11, marginBottom: 4 }}>⚠ CONTRAINDICATIONS</div>
                      {d.contraindications.map((c, i) => <div key={i} style={{ fontSize: 11, color: '#c62828' }}>• {c}</div>)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── CLINICAL ATLAS TAB ── */}
        {tab === 'Clinical Atlas' && breakdown && !loading && (
          <div>
            <h3 style={{ margin: '0 0 16px', color: '#333' }}>Clinical Atlas — Treatment, Pearls &amp; Contraindications</h3>
            {Object.entries(breakdown).map(([gene, d]) => (
              <div key={gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[gene]}55`, borderRadius: 10, padding: 20, marginBottom: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  <span style={{ background: GENE_COLORS[gene], color: '#fff', borderRadius: 6, padding: '4px 14px', fontWeight: 800, fontSize: 15 }}>{gene}</span>
                  <span style={{ fontSize: 13, color: '#666' }}>{GENE_INFO[gene]?.locus} · {d.protein_size} · {d.inheritance.split(';')[0]}</span>
                </div>
                <div style={{ fontSize: 13, color: '#444', marginBottom: 10 }}>{GENE_INFO[gene]?.disease}</div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 12 }}>
                  <div style={{ background: '#e8f5e9', borderRadius: 7, padding: 12 }}>
                    <div style={{ fontWeight: 700, color: '#2e7d32', marginBottom: 6 }}>Treatment</div>
                    <div style={{ color: '#333', lineHeight: 1.5 }}>{d.treatment}</div>
                  </div>
                  <div style={{ background: '#fff8e1', borderRadius: 7, padding: 12 }}>
                    <div style={{ fontWeight: 700, color: '#f57f17', marginBottom: 6 }}>Critical Pearls</div>
                    <ul style={{ margin: 0, paddingLeft: 16 }}>
                      {d.critical_pearls?.map((p, i) => <li key={i} style={{ marginBottom: 4, lineHeight: 1.4 }}>{p}</li>)}
                    </ul>
                  </div>
                </div>

                {d.contraindications?.length > 0 && (
                  <div style={{ marginTop: 10, background: '#ffebee', borderRadius: 7, padding: 10 }}>
                    <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 6 }}>⚠ CONTRAINDICATIONS</div>
                    {d.contraindications.map((c, i) => (
                      <div key={i} style={{ fontSize: 12, color: '#b71c1c', marginBottom: 3 }}>• {c}</div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {tab === 'Definitions' && definitions && !loading && (
          <div>
            {/* Onset age spectrum */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 14px', color: '#b71c1c', fontSize: 15 }}>Onset Age Spectrum by Gene</h3>
              {definitions.onset_age_spectrum && Object.entries(definitions.onset_age_spectrum).map(([gene, onset]) => {
                const g = gene.split('_')[0];
                return (
                  <div key={gene} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, fontSize: 13 }}>
                    <span style={{ background: GENE_COLORS[g] || '#888', color: '#fff', borderRadius: 4, padding: '2px 10px', fontWeight: 700, minWidth: 90, display: 'inline-block', fontSize: 12 }}>{gene}</span>
                    <span>{onset}</span>
                  </div>
                );
              })}
            </div>

            {/* Biopsy patterns */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 14px', color: '#1565c0', fontSize: 15 }}>Biopsy &amp; IHC Pattern Table</h3>
              {definitions.biopsy_pattern_table && Object.entries(definitions.biopsy_pattern_table).map(([gene, pat]) => {
                const g = gene.split('_')[0];
                return (
                  <div key={gene} style={{ marginBottom: 8, padding: '8px 12px', background: '#e3f2fd', borderRadius: 6, fontSize: 13 }}>
                    <strong style={{ color: GENE_COLORS[g] || '#1565c0' }}>{gene}</strong>: {pat}
                  </div>
                );
              })}
            </div>

            {/* Cardiac surveillance */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 14px', color: '#c62828', fontSize: 15 }}>Cardiac Surveillance Table</h3>
              {definitions.cardiac_surveillance_table && Object.entries(definitions.cardiac_surveillance_table).map(([gene, rec]) => {
                const mandatory = rec.toLowerCase().includes('mandatory') || rec.toLowerCase().includes('every') || rec.toLowerCase().includes('icd');
                const g = gene.split('_')[0];
                return (
                  <div key={gene} style={{ marginBottom: 6, padding: '8px 12px', background: mandatory ? '#ffebee' : '#f5f5f5', borderRadius: 6, fontSize: 13 }}>
                    <strong style={{ color: mandatory ? '#c62828' : GENE_COLORS[g] || '#555' }}>{gene}</strong>: {rec}
                  </div>
                );
              })}
            </div>

            {/* DDx table */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 14px', color: '#e65100', fontSize: 15 }}>Differential Diagnosis Table</h3>
              {definitions.ddx_table && Object.entries(definitions.ddx_table).map(([pair, desc]) => (
                <div key={pair} style={{ marginBottom: 8, padding: '8px 12px', background: '#fff3e0', borderRadius: 6, fontSize: 13 }}>
                  <strong style={{ color: '#e65100' }}>{pair.replace(/_/g, ' ')}</strong>: {desc}
                </div>
              ))}
            </div>

            {/* Founder mutations */}
            <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
              <h3 style={{ margin: '0 0 14px', color: '#00695c', fontSize: 15 }}>Founder Mutations</h3>
              {definitions.founder_mutations && Object.entries(definitions.founder_mutations).map(([mut, desc]) => {
                const g = mut.split('_')[0];
                return (
                  <div key={mut} style={{ marginBottom: 6, padding: '8px 12px', background: '#e0f2f1', borderRadius: 6, fontSize: 13 }}>
                    <strong style={{ color: GENE_COLORS[g] || '#00695c' }}>{mut.replace(/_/g, ' ')}</strong>: {desc}
                  </div>
                );
              })}
            </div>

            {/* LINC complex panel */}
            {definitions.linc_complex_panel && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
                <h3 style={{ margin: '0 0 14px', color: '#1565c0', fontSize: 15 }}>LINC Complex Panel</h3>
                {Object.entries(definitions.linc_complex_panel).map(([section, content]) => (
                  <div key={section} style={{ marginBottom: 10 }}>
                    <div style={{ fontWeight: 700, color: '#1565c0', fontSize: 13, marginBottom: 4, textTransform: 'capitalize' }}>{section.replace(/_/g, ' ')}</div>
                    {typeof content === 'string' ? (
                      <div style={{ fontSize: 12, color: '#555', padding: '6px 12px', background: '#e3f2fd', borderRadius: 5 }}>{content}</div>
                    ) : (
                      Object.entries(content).map(([gene, desc]) => (
                        <div key={gene} style={{ marginBottom: 4, padding: '6px 12px', background: '#e3f2fd', borderRadius: 5, fontSize: 12 }}>
                          <strong style={{ color: GENE_COLORS[gene] || '#1565c0' }}>{gene}</strong>: {desc}
                        </div>
                      ))
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Emerin IHC decision tree */}
            {definitions.emerin_ihc_decision_tree && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 16, boxShadow: '0 1px 4px #0001' }}>
                <h3 style={{ margin: '0 0 14px', color: '#b71c1c', fontSize: 15 }}>Emerin IHC Decision Tree</h3>
                {Object.entries(definitions.emerin_ihc_decision_tree).map(([cond, action]) => (
                  <div key={cond} style={{ marginBottom: 6, padding: '8px 12px', background: '#fce4ec', borderRadius: 6, fontSize: 13 }}>
                    <strong style={{ color: '#b71c1c' }}>{cond.replace(/_/g, ' ')}</strong>: {action}
                  </div>
                ))}
              </div>
            )}

            {/* Glossary */}
            {definitions.glossary && (
              <div style={{ background: '#fff', borderRadius: 10, padding: 20, boxShadow: '0 1px 4px #0001' }}>
                <h3 style={{ margin: '0 0 14px', color: '#37474f', fontSize: 15 }}>Glossary</h3>
                {Object.entries(definitions.glossary).map(([term, def]) => (
                  <div key={term} style={{ marginBottom: 8, padding: '8px 12px', background: '#eceff1', borderRadius: 6, fontSize: 13 }}>
                    <strong style={{ color: '#37474f' }}>{term.replace(/_/g, ' ')}</strong>: {def}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
