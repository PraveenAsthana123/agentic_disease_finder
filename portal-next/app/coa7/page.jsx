'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Patients & Features', 'Treatments & DDx', 'Definitions'];
const COLOR = '#1b5e20';   // deep forest green — ataxia-neuropathy / cerebellar
const LIGHT = '#e8f5e9';

function KPI({ label, value, color }) {
  return (
    <div style={{ background: color || LIGHT, borderRadius: 8, padding: '14px 18px', minWidth: 140 }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: COLOR }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
    </div>
  );
}

function Badge({ text, kind }) {
  const colors = {
    danger:  { bg: '#ffebee', color: '#b71c1c', border: '#ef9a9a' },
    warn:    { bg: '#fff3e0', color: '#e65100', border: '#ffcc80' },
    ok:      { bg: '#e8f5e9', color: '#1b5e20', border: '#a5d6a7' },
    info:    { bg: '#e3f2fd', color: '#0d47a1', border: '#90caf9' },
    neutral: { bg: '#f5f5f5', color: '#333',    border: '#ccc'    },
  };
  const s = colors[kind] || colors.neutral;
  return (
    <span style={{ background: s.bg, color: s.color, border: `1px solid ${s.border}`,
      borderRadius: 12, padding: '2px 10px', fontSize: 11, fontWeight: 600, marginRight: 4 }}>
      {text}
    </span>
  );
}

export default function COA7Page() {
  const [tab, setTab]         = useState(0);
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError]     = useState(null);
  const [loading, setLoading] = useState(false);

  async function fetchData(endpoint, setter) {
    try {
      setLoading(true);
      const r = await fetch(`${API}/api/coa7/${endpoint}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setter(await r.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  useEffect(() => { fetchData('overview', setOverview); }, []);
  useEffect(() => {
    if (tab === 1 && !breakdown)   fetchData('breakdown',   setBreakdown);
    if (tab === 2 && !breakdown)   fetchData('breakdown',   setBreakdown);
    if (tab === 3 && !definitions) fetchData('definitions', setDefinitions);
  }, [tab]);

  const ov = overview;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      {/* Header */}
      <div style={{ background: COLOR, color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <div style={{ fontSize: 11, opacity: 0.8, marginBottom: 4 }}>COXPD16 / AR / Nuclear 6q25.3 / OMIM Gene *615623 / Disease #616838</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
          COA7 — Spinocerebellar Ataxia + Axonal Neuropathy / COXPD16
        </h1>
        <div style={{ fontSize: 13, opacity: 0.9, marginTop: 6 }}>
          Complex IV Assembly Factor 7 (RESA1 / SELRC1) · 231 aa / 26 kDa · ARM/SEL1 repeat scaffold ·
          Late-stage CIV assembly · NO TM helices · Mild CIV deficiency 30–60% residual ·
          Adolescent/adult onset · NO Leigh MRI — cerebellar atrophy instead
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, marginBottom: 20 }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)}
            style={{ padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === i ? COLOR : '#f0f0f0',
              color: tab === i ? '#fff' : '#333', fontWeight: tab === i ? 700 : 400 }}>
            {t}
          </button>
        ))}
      </div>

      {error && <div style={{ color: 'red', padding: 12 }}>Error: {error}</div>}
      {loading && <div style={{ color: '#888', padding: 12 }}>Loading…</div>}

      {/* TAB 0 — OVERVIEW */}
      {tab === 0 && ov && (
        <div>
          {/* Key Insight Banner */}
          <div style={{ background: '#e8f5e9', border: '1px solid #a5d6a7', borderRadius: 8,
            padding: '12px 18px', marginBottom: 20, fontSize: 13, color: '#1b5e20' }}>
            <strong>Key Insight:</strong> {ov.key_insight}
          </div>

          {/* KPIs */}
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            <KPI label="Patients" value={ov.n_patients} />
            <KPI label="Avg Onset (yr)" value={ov.avg_onset_years} />
            <KPI label="Avg CIV Residual" value={`${ov.avg_civ_residual_pct}%`} color="#fff9c4" />
            <KPI label="Avg Lactate (mmol/L)" value={ov.avg_lactate_mmol} />
            <KPI label="Avg SARA Score" value={ov.avg_sara_score} />
            <KPI label="Cerebellar Atrophy MRI" value={`${ov.pct_cerebellar_atrophy}%`} />
            <KPI label="Axonal Neuropathy" value={`${ov.pct_axonal_neuropathy}%`} />
            <KPI label="HCM" value={`${ov.pct_hcm}% ✓ NO HCM`} color="#e8f5e9" />
            <KPI label="Leigh MRI" value={`${ov.pct_leigh_mri}% ✓ NO Leigh`} color="#e8f5e9" />
          </div>

          {/* Gene Card */}
          <div style={{ background: LIGHT, borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 12px', color: COLOR, fontSize: 16 }}>Gene &amp; Protein</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13 }}>
              <tbody>
                {[
                  ['Gene', ov.gene],
                  ['Alias', ov.alias],
                  ['Disease', ov.disease],
                  ['OMIM Gene', `*${ov.omim_gene}`],
                  ['OMIM Disease', `#${ov.omim_disease}`],
                  ['Inheritance', ov.inheritance],
                  ['Locus', ov.locus],
                  ['Protein Size', ov.protein_size],
                  ['Module / Function', ov.protein_module],
                ].map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '6px 12px', fontWeight: 600, color: '#1b5e20', width: 180 }}>{k}</td>
                    <td style={{ padding: '6px 12px', color: '#222' }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Phenotype Distribution */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 12px', color: COLOR, fontSize: 16 }}>Phenotype Distribution</h2>
            {(ov.phenotype_distribution || []).map(ph => (
              <div key={ph.label} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 3 }}>
                  <span style={{ fontWeight: 600 }}>{ph.label.split('(')[0].trim()}</span>
                  <span style={{ color: COLOR, fontWeight: 700 }}>{ph.pct}%</span>
                </div>
                <div style={{ background: '#e8f5e9', borderRadius: 4, height: 10 }}>
                  <div style={{ background: COLOR, borderRadius: 4, height: 10, width: `${ph.pct}%` }} />
                </div>
                <div style={{ fontSize: 11, color: '#777', marginTop: 2 }}>{ph.severity}</div>
              </div>
            ))}
          </div>

          {/* DDx Anchors */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20 }}>
            <h2 style={{ margin: '0 0 12px', color: COLOR, fontSize: 16 }}>Key DDx Anchors for COA7</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 13 }}>
              {[
                ['NO HCM', 'SCO2 = 100% HCM → if HCM present, COA7 excluded; SCO2 not COA7', 'ok'],
                ['NO Leigh MRI', 'Cerebellar atrophy (NOT bilateral BG T2 signal) — if Leigh MRI = SURF1/SCO1/SCO2/COX14', 'ok'],
                ['NO Hepatopathy', 'SCO1 = neonatal hepatopathy. POLG = Alpers hepatopathy. COA7 = no liver disease', 'ok'],
                ['Mild CIV 30–65%', 'Unlike SURF1/SCO2/COA5 (<20%), COA7 CIV deficiency is mild — may be missed by histochem', 'warn'],
                ['Adolescent/Adult Onset', 'Not neonatal/infantile — unlike most COXPD genes. Onset 10–45 yr is COA7 territory', 'info'],
                ['Ataxia > Encephalopathy', 'Dominant cerebellar phenotype: ONLY CIV gene where ataxia-neuropathy > encephalopathy', 'info'],
                ['BTD exclusion MANDATORY', 'Biotinidase deficiency → identical cerebellar ataxia → treatable → must exclude first', 'danger'],
                ['SLC19A3 exclusion MANDATORY', 'BTBGD → biotin-thiamine-responsive basal ganglia disease → treatable mimic → empiric B1+biotin', 'danger'],
                ['WES detects COA7', 'Nuclear gene 6q25.3 — detected by WES (unlike MT-CO1/CO2/CO3 which require dedicated mtDNA seq)', 'info'],
                ['COX20 DDx', 'COX20 = childhood cerebellar ataxia, more severe CIV <25%; COA7 = later onset, milder CIV', 'neutral'],
              ].map(([title, body, kind]) => (
                <div key={title} style={{ background: LIGHT, borderRadius: 6, padding: '10px 14px' }}>
                  <div style={{ fontWeight: 700, marginBottom: 4, color: COLOR }}><Badge text={title} kind={kind} /></div>
                  <div style={{ fontSize: 12, color: '#333' }}>{body}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 1 — PATIENTS & FEATURES */}
      {tab === 1 && breakdown && (
        <div>
          {/* Variants */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Pathogenic Variants</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: COLOR, color: '#fff' }}>
                  {['cDNA', 'Protein', 'Domain', '% Cases', 'Phenotype', 'Ethnic/Notes'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.variants || []).map((v, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: COLOR }}>{v.cdna}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{v.protein}</td>
                    <td style={{ padding: '6px 10px', color: '#444' }}>{v.domain}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: '#1b5e20' }}>{v.pct_cases}%</td>
                    <td style={{ padding: '6px 10px', color: '#333' }}>{v.phenotype}</td>
                    <td style={{ padding: '6px 10px', color: '#666', fontSize: 11 }}>{v.ethnic}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Phenotype Classes */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Phenotype Classes</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: COLOR, color: '#fff' }}>
                  {['Phenotype', '% Cases', 'Severity', 'Onset (yr)', 'DDx Anchor'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.phenotype_classes || []).map((pc, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{pc.label}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: COLOR }}>{pc.pct}%</td>
                    <td style={{ padding: '6px 10px' }}>{pc.severity}</td>
                    <td style={{ padding: '6px 10px' }}>{pc.onset_years}</td>
                    <td style={{ padding: '6px 10px', fontSize: 11, color: '#555' }}>{pc.ddx_anchor}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Patient Sample */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Patient Sample (8 of 40)</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: COLOR, color: '#fff' }}>
                  {['ID', 'Sex', 'Onset yr', 'Age now', 'CIV %', 'Lactate', 'SARA', 'Cer Atrophy', 'Axonal NRP', 'Phenotype'].map(h => (
                    <th key={h} style={{ padding: '7px 9px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.patients_sample || []).map((p, i) => (
                  <tr key={p.id} style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '6px 9px', fontWeight: 700, color: COLOR }}>{p.id}</td>
                    <td style={{ padding: '6px 9px' }}>{p.sex}</td>
                    <td style={{ padding: '6px 9px' }}>{p.age_onset}</td>
                    <td style={{ padding: '6px 9px' }}>{p.age_now}</td>
                    <td style={{ padding: '6px 9px', fontWeight: 700 }}>{p.civ_residual_pct}%</td>
                    <td style={{ padding: '6px 9px' }}>{p.lactate_mmol}</td>
                    <td style={{ padding: '6px 9px' }}>{p.sara_score}</td>
                    <td style={{ padding: '6px 9px' }}>{p.cerebellar_atrophy_mri ? '✓' : '—'}</td>
                    <td style={{ padding: '6px 9px' }}>{p.axonal_neuropathy ? '✓' : '—'}</td>
                    <td style={{ padding: '6px 9px', fontSize: 11, color: '#444' }}>{p.phenotype.split('(')[0].substring(0, 40)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 2 — TREATMENTS & DDX */}
      {tab === 2 && breakdown && (
        <div>
          {/* Absolute Contraindications */}
          <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: '#b71c1c', fontSize: 16 }}>⚠ Contraindications</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  {['Drug / Intervention', 'Class', 'Reason'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.contraindications || []).map((c, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff8f8' : '#ffebee', borderBottom: '1px solid #ef9a9a' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700 }}>{c.item}</td>
                    <td style={{ padding: '6px 10px' }}>
                      <Badge text={c.class} kind={c.class.includes('ABSOLUTE') ? 'danger' : 'warn'} />
                    </td>
                    <td style={{ padding: '6px 10px', color: '#444' }}>{c.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Treatments */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Treatments / Supportive Measures</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: COLOR, color: '#fff' }}>
                  {['Drug / Intervention', 'Evidence', 'Notes'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.treatments || []).map((t, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700 }}>{t.drug}</td>
                    <td style={{ padding: '6px 10px' }}><Badge text={t.evidence} kind="ok" /></td>
                    <td style={{ padding: '6px 10px', color: '#444' }}>{t.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Monitoring */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Surveillance &amp; Monitoring</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <thead>
                <tr style={{ background: COLOR, color: '#fff' }}>
                  {['Item', 'Frequency', 'Notes'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.monitoring || []).map((m, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700 }}>{m.item}</td>
                    <td style={{ padding: '6px 10px', color: '#1b5e20' }}>{m.frequency}</td>
                    <td style={{ padding: '6px 10px', color: '#444' }}>{m.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3 — DEFINITIONS */}
      {tab === 3 && definitions && (
        <div>
          {/* Key Concepts */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 16px', color: COLOR, fontSize: 16 }}>Key Concepts</h2>
            {(definitions.key_concepts || []).map((kc, i) => (
              <div key={i} style={{ background: LIGHT, borderRadius: 6, padding: '12px 16px', marginBottom: 12 }}>
                <div style={{ fontWeight: 700, color: COLOR, marginBottom: 6, fontSize: 13 }}>{kc.title}</div>
                <div style={{ fontSize: 12, color: '#333', lineHeight: 1.6 }}>{kc.body}</div>
              </div>
            ))}
          </div>

          {/* Glossary */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20, marginBottom: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Glossary</h2>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12 }}>
              <tbody>
                {Object.entries(definitions.glossary || {}).map(([term, def], i) => (
                  <tr key={term} style={{ background: i % 2 === 0 ? LIGHT : '#fff', borderBottom: '1px solid #c8e6c9' }}>
                    <td style={{ padding: '7px 12px', fontWeight: 700, color: COLOR, width: 200, whiteSpace: 'nowrap' }}>{term}</td>
                    <td style={{ padding: '7px 12px', color: '#333' }}>{def}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* References */}
          <div style={{ background: '#fff', border: '1px solid #c8e6c9', borderRadius: 8, padding: 20 }}>
            <h2 style={{ margin: '0 0 14px', color: COLOR, fontSize: 16 }}>Key References</h2>
            {(definitions.references || []).map((r, i) => (
              <div key={i} style={{ borderBottom: '1px solid #c8e6c9', padding: '8px 0', fontSize: 12 }}>
                <span style={{ fontWeight: 700, color: COLOR }}>{r.author_year}</span>
                {' · '}
                <em>{r.journal}</em>
                {' — '}
                <span style={{ color: '#444' }}>{r.summary}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
