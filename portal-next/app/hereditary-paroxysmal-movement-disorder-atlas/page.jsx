'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-paroxysmal-movement-disorder-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PRRT2:   '#1565c0',  // deep blue    — PKD movement-triggered carbamazepine
  SLC2A1:  '#2e7d32',  // deep green   — GLUT1D ketogenic diet TREATABLE
  PNKD:    '#4a148c',  // deep purple  — PNKD2 caffeine/alcohol trigger
  KCNA1:   '#e65100',  // deep orange  — EA1 myokymia pathognomonic
  CACNA1A: '#880e4f',  // deep pink    — EA2/SCA6/FHM1 acetazolamide
  ATP1A2:  '#b71c1c',  // deep red     — FHM2 triptans absolutely CI
  ADCY5:   '#37474f',  // dark grey    — nocturnal NREM caffeine CI
  SLC6A5:  '#00695c',  // deep teal    — hyperekplexia clonazepam curative
};

const GENE_INFO = {
  PRRT2:   { full: 'PRRT2 / 340aa',    locus: '16p11.2',  size: '340 aa',  inh: 'AD',      disease: 'PKD/BFIS/ICCA — MOVEMENT-TRIGGERED <1 min NO LOC PATHOGNOMONIC / CARBAMAZEPINE >90% EFFECTIVE low dose DRAMATIC / 16p11.2 deletion CNV testing mandatory' },
  SLC2A1:  { full: 'SLC2A1 / 492aa',   locus: '1p34.2',   size: '492 aa',  inh: 'AD',      disease: 'GLUT1D/PED — CSF GLUCOSE <45 mg/dL + CSF:SERUM <0.6 PATHOGNOMONIC / KETOGENIC DIET FIRST-LINE seizures abate dramatically / Prolonged exercise >5 min trigger' },
  PNKD:    { full: 'PNKD / 385aa',     locus: '2q35',     size: '385 aa',  inh: 'AD',      disease: 'PNKD2/BDC — CAFFEINE + ALCOHOL TRIGGER PATHOGNOMONIC / Episodes 10 min–12 hr / CBZ INEFFECTIVE DDx PKD / Clonazepam partially effective' },
  KCNA1:   { full: 'KCNA1 / 495aa',    locus: '12p13.32', size: '495 aa',  inh: 'AD',      disease: 'EA1 — INTERICTAL MYOKYMIA EMG PATHOGNOMONIC / Startle/exercise triggered seconds / NO interictal nystagmus DDx EA2 / Carbamazepine + Acetazolamide' },
  CACNA1A: { full: 'CACNA1A / 2510aa', locus: '19p13.13', size: '2510 aa', inh: 'AD',      disease: 'EA2/SCA6/FHM1 — INTERICTAL NYSTAGMUS PATHOGNOMONIC / Episodes HOURS / ACETAZOLAMIDE >80% reduction / SCA6 CAG repeat requires specific assay' },
  ATP1A2:  { full: 'ATP1A2 / 1020aa',  locus: '1q23.2',   size: '1020 aa', inh: 'AD',      disease: 'FHM2 — HEMIPLEGIC AURA + CONFUSION + PROLONGED WEAKNESS PATHOGNOMONIC / TRIPTANS ABSOLUTELY CONTRAINDICATED / Valproate + Topiramate prevention' },
  ADCY5:   { full: 'ADCY5 / 1261aa',   locus: '3q21.3',   size: '1261 aa', inh: 'AD-GoF',  disease: 'ADCY5-RMD — NOCTURNAL ATTACKS FROM NREM SLEEP PATHOGNOMONIC / EEG NORMAL not epilepsy / CAFFEINE ABSOLUTELY CI / Clonazepam + Acetazolamide first-line' },
  SLC6A5:  { full: 'SLC6A5 / 797aa',   locus: '11p15.1',  size: '797 aa',  inh: 'AR',      disease: 'Hyperekplexia/Startle Disease — EXAGGERATED STARTLE + NEONATAL HYPERTONIA + APNOEA PATHOGNOMONIC / CLONAZEPAM CURATIVE / NOSE-TIPPING MANOEUVRE life-saving' },
};

export default function HereditaryParoxysmalMovementDisorderAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const endpoints = [
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ];
    Promise.all(endpoints)
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#555' }}>Loading Hereditary Paroxysmal Movement Disorder Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;
  if (!overview) return null;

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui,sans-serif', maxWidth: 1200 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, color: '#1a237e', marginBottom: 4 }}>
        🧠 Hereditary Paroxysmal Movement Disorder Atlas
      </h1>
      <p style={{ color: '#555', fontSize: 13, marginBottom: 18 }}>
        Complete 8-Gene Reference — PKD · PED · PNKD · EA1 · EA2/SCA6/FHM1 · FHM2 · ADCY5-RMD · Hyperekplexia
        &nbsp;|&nbsp; {overview.total_patients} patients · Seeds {overview.seeds}
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e3e8f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', border: 'none', borderRadius: '6px 6px 0 0',
            background: tab === t ? '#1a237e' : '#f0f4ff',
            color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400, cursor: 'pointer', fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Alive', value: `${overview.alive_pct}%` },
              { label: 'Treated', value: `${overview.treated_pct}%` },
              { label: 'Avg Age (yr)', value: overview.avg_age },
            ].map(m => (
              <div key={m.label} style={{ background: '#f0f4ff', borderRadius: 8, padding: '14px 18px', textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 700, color: '#1a237e' }}>{m.value}</div>
                <div style={{ fontSize: 11, color: '#555', marginTop: 2 }}>{m.label}</div>
              </div>
            ))}
          </div>

          {/* Clinical Axioms */}
          <div style={{ background: '#fffde7', border: '1px solid #f9a825', borderRadius: 8, padding: '14px 18px', marginBottom: 20 }}>
            <div style={{ fontWeight: 700, color: '#e65100', marginBottom: 8, fontSize: 13 }}>⚡ Clinical Axioms — Paroxysmal Movement Disorders</div>
            {overview.clinical_axioms.map((axiom, i) => (
              <div key={i} style={{ fontSize: 12, color: '#333', marginBottom: 4, paddingLeft: 8, borderLeft: '3px solid #f9a825' }}>
                {axiom}
              </div>
            ))}
          </div>

          {/* Gene Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 14 }}>
            {Object.entries(overview.gene_summaries || {}).map(([gene, gs]) => (
              <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 10, padding: '14px 16px', background: '#fafbff' }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#333', fontSize: 14, marginBottom: 4 }}>
                  {gene} <span style={{ fontSize: 11, color: '#666', fontWeight: 400 }}>— {gs.protein_size} · {gs.locus}</span>
                </div>
                <div style={{ fontSize: 11, color: '#444', marginBottom: 8 }}>{GENE_INFO[gene]?.disease}</div>
                <div style={{ display: 'flex', gap: 12, fontSize: 11, color: '#666' }}>
                  <span>Patients: {gs.n_patients}</span>
                  <span>Alive: {gs.alive_pct}%</span>
                  <span>Treated: {gs.treated_pct}%</span>
                </div>
                {gs.critical_pearls && gs.critical_pearls.length > 0 && (
                  <div style={{ marginTop: 8, background: '#fff3e0', borderRadius: 4, padding: '6px 8px', fontSize: 11, color: '#bf360c' }}>
                    💡 {gs.critical_pearls[0]}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Locus / Size</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Inh.</th>
                <th style={{ padding: '10px 12px', textAlign: 'left' }}>Disease / Phenotype</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Pts</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Alive%</th>
                <th style={{ padding: '10px 12px', textAlign: 'center' }}>Treated%</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, gd], idx) => (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f5f9ff' : '#fff', borderBottom: '1px solid #e0e8f0' }}>
                  <td style={{ padding: '9px 12px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                  <td style={{ padding: '9px 12px', color: '#555', fontSize: 11 }}>{gd.locus}<br/>{gd.protein_size}</td>
                  <td style={{ padding: '9px 12px', color: '#555', fontSize: 11 }}>{GENE_INFO[gene]?.inh}</td>
                  <td style={{ padding: '9px 12px', color: '#333', fontSize: 11 }}>{GENE_INFO[gene]?.disease?.substring(0, 120)}…</td>
                  <td style={{ padding: '9px 12px', textAlign: 'center', fontWeight: 600 }}>{gd.n_patients}</td>
                  <td style={{ padding: '9px 12px', textAlign: 'center' }}>{gd.alive_pct}%</td>
                  <td style={{ padding: '9px 12px', textAlign: 'center' }}>{gd.treated_pct}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {Object.entries(breakdown).map(([gene, gd]) => (
            <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 10, padding: '16px 20px', background: '#fafbff' }}>
              <h3 style={{ color: GENE_COLORS[gene] || '#333', margin: '0 0 4px', fontSize: 15 }}>
                {gene} <span style={{ fontSize: 12, color: '#666', fontWeight: 400 }}>— {gd.protein_size} · {gd.locus}</span>
              </h3>
              <p style={{ fontSize: 12, color: '#555', margin: '0 0 12px' }}>{GENE_INFO[gene]?.disease}</p>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <div style={{ fontWeight: 600, color: '#1a237e', fontSize: 12, marginBottom: 6 }}>🔬 Key Clinical Features</div>
                  {(gd.key_features || []).map((f, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 3, paddingLeft: 8, borderLeft: `3px solid ${GENE_COLORS[gene] || '#ccc'}` }}>{f}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: '#2e7d32', fontSize: 12, marginBottom: 6 }}>💊 Treatment</div>
                  {(gd.treatment || []).map((t, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#333', marginBottom: 3, paddingLeft: 8, borderLeft: '3px solid #2e7d32' }}>{t}</div>
                  ))}
                  {(gd.contraindications || []).length > 0 && (
                    <>
                      <div style={{ fontWeight: 600, color: '#b71c1c', fontSize: 12, marginTop: 10, marginBottom: 6 }}>🚫 Contraindications</div>
                      {gd.contraindications.map((c, i) => (
                        <div key={i} style={{ fontSize: 11, color: '#b71c1c', marginBottom: 3, paddingLeft: 8, borderLeft: '3px solid #b71c1c' }}>{c}</div>
                      ))}
                    </>
                  )}
                </div>
              </div>

              {(gd.critical_pearls || []).length > 0 && (
                <div style={{ marginTop: 12, background: '#fff3e0', borderRadius: 6, padding: '8px 12px' }}>
                  <div style={{ fontWeight: 600, color: '#e65100', fontSize: 12, marginBottom: 4 }}>💡 Critical Pearls</div>
                  {gd.critical_pearls.map((p, i) => (
                    <div key={i} style={{ fontSize: 11, color: '#bf360c', marginBottom: 2 }}>• {p}</div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          {Object.entries(definitions).map(([section, entries]) => (
            <div key={section} style={{ marginBottom: 24 }}>
              <h3 style={{ color: '#1a237e', fontSize: 14, fontWeight: 700, marginBottom: 10, textTransform: 'capitalize' }}>
                {section.replace(/_/g, ' ')}
              </h3>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <tbody>
                  {Object.entries(entries).map(([term, def], i) => (
                    <tr key={term} style={{ background: i % 2 === 0 ? '#f5f9ff' : '#fff' }}>
                      <td style={{ padding: '7px 12px', fontWeight: 600, color: '#1565c0', width: '26%', verticalAlign: 'top' }}>
                        {term.replace(/_/g, ' ')}
                      </td>
                      <td style={{ padding: '7px 12px', color: '#333' }}>{def}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
