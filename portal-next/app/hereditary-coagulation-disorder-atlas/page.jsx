'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  F8:    '#1565c0',  // deep blue    — Haemophilia A, FVIII, XLR, emicizumab, inhibitors
  F9:    '#4a148c',  // deep purple  — Haemophilia B, FIX, XLR, Hemgenix gene therapy
  VWF:   '#1b5e20',  // deep green   — VWD most common, Type 2B DDAVP-CI, Type 2N sex trap
  F11:   '#b71c1c',  // deep red     — Haemophilia C, Ashkenazi founder, level≠bleed severity
  F7:    '#e65100',  // deep orange  — FVII deficiency, isolated PT, most common rare factor
  F13A1: '#880e4f',  // deep pink    — FXIII-A, cord stump pathognomonic, normal PT/APTT trap
  F10:   '#0d47a1',  // dark blue    — FX Stuart-Prower, PT+APTT both long, AL amyloid acquired
  LMAN1: '#f57f17',  // amber        — Combined FV+FVIII, FFP+DDAVP, MCFD2 also test
};

const GENE_INFO = {
  F8:    { aa: 2351, locus: 'Xq28',     inh: 'XLR',   disease: 'Haemophilia-A — FVIII-Deficiency — Inhibitor-25-30pct-Severe — Emicizumab-Bispecific-Antibody — Intron-22-Inversion-45pct-Severe — APTT-Long-PT-Normal — Haemarthrosis-Arthropathy' },
  F9:    { aa: 461,  locus: 'Xq27.1',   inh: 'XLR',   disease: 'Haemophilia-B — Christmas-Disease — FIX-Deficiency — Hemgenix-FDA2022 — Leyden-Variant-Promoter-Improves-Puberty — Inhibitor-1-3pct — APTT-Long-PT-Normal' },
  VWF:   { aa: 2813, locus: '12p13.31', inh: 'AD/AR',  disease: 'Von-Willebrand-Disease — Most-Common-Inherited-Bleeding — Type-2B-DDAVP-ABSOLUTELY-CI-Thrombocytopenia — Type-2N-Mimics-Mild-HA-Both-Sexes — Type-3-AR-Severe-No-DDAVP-Response' },
  F11:   { aa: 625,  locus: '4q35.2',   inh: 'AR',     disease: 'Factor-XI-Deficiency — Haemophilia-C — Ashkenazi-Jewish-1-in-450 — Level-Does-NOT-Correlate-Bleeding-Severity — Surgery-Bleeds-More-Than-Spontaneous — E117X-F283L-Founders' },
  F7:    { aa: 444,  locus: '13q34',    inh: 'AR',     disease: 'Factor-VII-Deficiency — Most-Common-Rare-Coagulation-Factor — Isolated-PT-APTT-Normal — ICH-Risk — rFVIIa-NovoSeven-Treatment — F7-F10-Contiguous-13q34-Deletion' },
  F13A1: { aa: 732,  locus: '6p24.3',   inh: 'AR',     disease: 'Factor-XIII-A-Deficiency — Umbilical-Cord-Stump-Bleeding-PATHOGNOMONIC — Normal-PT-APTT-TRAP — 5M-Urea-Clot-Solubility-Screen — ICH-25-30pct-Untreated — Monthly-FXIII-Prophylaxis' },
  F10:   { aa: 488,  locus: '13q34',    inh: 'AR',     disease: 'Factor-X-Stuart-Prower — Both-PT-APTT-Prolonged — AL-Amyloidosis-Acquired-FX-Deficiency-Trap — PCC-Treatment — Severe-Less-1pct-ICH-Haemarthrosis — 4-Factor-PCC-Not-3-Factor' },
  LMAN1: { aa: 525,  locus: '18q21.3',  inh: 'AR',     disease: 'Combined-FV-FVIII-Deficiency — LMAN1-Cargo-Receptor-ER-Golgi — FV-NOT-In-FVIII-Concentrates-FFP-MANDATORY — DDAVP-Raises-FVIII-Not-FV — MCFD2-Also-Test — Middle-Eastern-Founder' },
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}
function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}
function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 160,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}
function Alert({ text }) {
  const lv = /ABSOLUTELY.CI|PATHOGNOMONIC|MANDATORY|ABSOLUTELY|FATAL|NEVER|LETHAL|TRAP/i.test(text) ? 'critical'
           : /\bCI\b|RISK|MONITOR|WARNING|AVOID/i.test(text) ? 'warning' : 'info';
  const colors = { critical: '#fca5a5', warning: '#fcd34d', info: '#93c5fd' };
  const bg     = { critical: '#450a0a', warning: '#451a03', info: '#0c1a3a' };
  return (
    <div style={{
      background: bg[lv], border: `1px solid ${colors[lv]}33`,
      borderLeft: `3px solid ${colors[lv]}`, borderRadius: 6,
      padding: '0.4rem 0.7rem', fontSize: 12, color: colors[lv], marginBottom: 4,
    }}>{text}</div>
  );
}

/* ── OVERVIEW TAB ─────────────────────────────────────────────────────────── */
function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const geneCounts = data.gene_patient_counts || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {data.subtitle}
      </p>

      {/* KPIs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"            value={data.total_patients}               color="#6366f1" />
        <KPI label="Genes Covered"             value={(data.genes || []).length || 8}    color="#10b981" />
        <KPI label="Seeds"                     value={data.seeds}                        color="#f59e0b" />
        <KPI label="XLR Genes (F8, F9)"        value={2}                                 color="#3b82f6" />
        <KPI label="AR Genes (5 genes)"        value={5}                                 color="#8b5cf6" />
        <KPI label="Inhibitor Patients"        value={data.inhibitor_patients}           color="#ef4444" />
        <KPI label="On Prophylaxis"            value={data.on_prophylaxis_patients}      color="#0d9488" />
        <KPI label="Gene Therapy Received"     value={data.gene_therapy_patients}        color="#f472b6" />
      </div>

      {/* Inheritance breakdown bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Inheritance Pattern (8 genes)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'XLR (2 genes: F8, F9)', val: 2, color: '#3b82f6' },
            { label: 'AR (5 genes: F11, F7, F13A1, F10, LMAN1)', val: 5, color: '#10b981' },
            { label: 'AD/AR (1: VWF)', val: 1, color: '#f59e0b' },
          ].map(b => (
            <div key={b.label} title={b.label} style={{ flex: b.val, background: b.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#fff', fontWeight: 700, overflow: 'hidden' }}>
              {b.val}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 8 }}>
          {[
            { label: 'XLR (F8, F9)', color: '#3b82f6' },
            { label: 'AR (F11, F7, F13A1, F10, LMAN1)', color: '#10b981' },
            { label: 'AD/AR (VWF)', color: '#f59e0b' },
          ].map(b => (
            <div key={b.label} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#94a3b8' }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: b.color }} />
              {b.label}
            </div>
          ))}
        </div>
      </div>

      {/* Gene patient counts */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 10 }}>Patients per Gene (40 each)</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {Object.entries(geneCounts).map(([gene, n]) => (
            <div key={gene} style={{
              background: GENE_COLORS[gene] || '#334155', borderRadius: 6,
              padding: '0.4rem 0.8rem', fontSize: 12, color: '#fff', fontWeight: 600,
            }}>{gene}: {n}</div>
          ))}
        </div>
      </div>

      {/* Critical atlas-level alerts */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Atlas-Level Safety Alerts</div>
        {(data.key_clinical_insight || '').split('. ').filter(Boolean).map((s, i) => (
          <Alert key={i} text={s} />
        ))}
      </div>

      {/* Mechanism summary */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Pathway Mechanism Summary</div>
        <p style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
          {data.mechanism_summary}
        </p>
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab() {
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Gene Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene', 'Size (aa)', 'Locus', 'Inheritance', 'Disease / Key Facts'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <tr key={gene} style={{ borderBottom: '1px solid #1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc' }}>{gene}</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{info.aa.toLocaleString()}</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0', whiteSpace: 'nowrap' }}>{info.locus}</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0', whiteSpace: 'nowrap' }}>{info.inh}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 11, lineHeight: 1.4 }}>
                  {info.disease.replace(/-/g, ' — ').replace(/  +/g, ' ')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;

  const genes = Object.values(data);
  const gd = selected ? data[selected] : null;

  return (
    <div style={{ display: 'flex', gap: 16 }}>
      {/* Gene selector */}
      <div style={{ minWidth: 140, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {genes.map(g => (
          <button key={g.gene} onClick={() => setSelected(g.gene)} style={{
            background: selected === g.gene ? GENE_COLORS[g.gene] || '#6366f1' : '#1e293b',
            border: 'none', borderRadius: 6, padding: '0.5rem 0.8rem',
            color: '#fff', fontSize: 12, fontWeight: selected === g.gene ? 700 : 400,
            cursor: 'pointer', textAlign: 'left',
          }}>{g.gene}</button>
        ))}
      </div>

      {/* Gene detail */}
      <div style={{ flex: 1 }}>
        {!gd ? (
          <div style={{ color: '#94a3b8', fontSize: 13, padding: '2rem' }}>← Select a gene to view its clinical profile</div>
        ) : (
          <div>
            <h3 style={{ color: GENE_COLORS[gd.gene] || '#a5b4fc', marginBottom: 4 }}>{gd.gene}</h3>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: '1rem' }}>{gd.alt_name}</div>

            {/* Stats row */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
              {[
                { l: 'Patients', v: gd.n_patients },
                { l: 'Locus', v: gd.locus },
                { l: 'Size', v: gd.protein_size },
                { l: 'Inheritance', v: gd.inheritance },
                { l: 'Inhibitor %', v: `${gd.inhibitor_pct ?? '—'}%` },
                { l: 'Prophylaxis %', v: `${gd.on_prophylaxis_pct ?? '—'}%` },
                { l: 'ICH %', v: `${gd.intracranial_hemorrhage_pct ?? '—'}%` },
                { l: 'Mucocutaneous %', v: `${gd.mucocutaneous_pct ?? '—'}%` },
              ].map(s => (
                <div key={s.l} style={{ background: '#0f172a', borderRadius: 6, padding: '0.4rem 0.8rem' }}>
                  <div style={{ fontSize: 10, color: '#64748b' }}>{s.l}</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>{s.v}</div>
                </div>
              ))}
            </div>

            {/* Clinical sections */}
            {[
              { label: 'Age of Onset / Presentation', text: gd.age_of_onset },
              { label: 'Key Biomarkers', text: gd.key_biomarker },
              { label: 'Pathognomonic Features', text: gd.pathognomonic },
              { label: 'Treatment', text: gd.treatment },
            ].map(sec => (
              <div key={sec.label} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem', marginBottom: 8 }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4, fontWeight: 700 }}>{sec.label}</div>
                <p style={{ fontSize: 12, color: '#cbd5e1', margin: 0, lineHeight: 1.6 }}>{sec.text}</p>
              </div>
            ))}

            {/* Critical flags */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem', marginBottom: 8 }}>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 700 }}>Critical Safety Flags</div>
              {(gd.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
            </div>

            {/* Cohort preview */}
            {gd.cohort_preview && (
              <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem' }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6, fontWeight: 700 }}>Cohort Preview (first 5 patients)</div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: '#0f172a' }}>
                        {['ID', 'Age', 'Sex', 'Severity', 'Factor %', 'Inhibitor', 'Prophylaxis', 'Joint Dis.'].map(h => (
                          <th key={h} style={{ padding: '4px 8px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {gd.cohort_preview.map((p, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #1e293b' }}>
                          <td style={{ padding: '4px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                          <td style={{ padding: '4px 8px', color: '#e2e8f0' }}>{p.age}</td>
                          <td style={{ padding: '4px 8px', color: '#e2e8f0' }}>{p.sex}</td>
                          <td style={{ padding: '4px 8px', color: '#e2e8f0' }}>{p.severity}</td>
                          <td style={{ padding: '4px 8px', color: '#e2e8f0' }}>{p.factor_level_pct ?? p.severity_pct ?? '—'}%</td>
                          <td style={{ padding: '4px 8px', color: p.inhibitor_developed ? '#fca5a5' : '#4ade80' }}>{p.inhibitor_developed ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '4px 8px', color: p.on_prophylaxis ? '#4ade80' : '#94a3b8' }}>{p.on_prophylaxis ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '4px 8px', color: p.joint_disease ? '#fcd34d' : '#94a3b8' }}>{p.joint_disease ? 'Yes' : 'No'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Glossary</h3>
      {Object.entries(data.glossary || {}).map(([term, def]) => (
        <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem 1rem', marginBottom: 8 }}>
          <div style={{ fontSize: 13, color: '#a5b4fc', fontWeight: 700, marginBottom: 4 }}>{term}</div>
          <p style={{ fontSize: 12, color: '#cbd5e1', margin: 0, lineHeight: 1.6 }}>{def}</p>
        </div>
      ))}

      <h3 style={{ color: '#f1f5f9', margin: '1.5rem 0 0.8rem' }}>Clinical Pearls</h3>
      {(data.clinical_pearls || []).map((p, i) => (
        <Alert key={i} text={`${i + 1}. ${p}`} />
      ))}
    </div>
  );
}

/* ── PAGE ROOT ────────────────────────────────────────────────────────────── */
export default function HereditaryCoagulationDisorderAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [err, setErr]           = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-coagulation-disorder-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setErr(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Clinical Atlas' && !breakdown) {
      fetch(`${API}/api/hereditary-coagulation-disorder-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setErr(e.message));
    }
    if (tab === 'Definitions' && !defs) {
      fetch(`${API}/api/hereditary-coagulation-disorder-atlas/definitions`)
        .then(r => r.json()).then(setDefs).catch(e => setErr(e.message));
    }
  }, [tab]);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'Inter, system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: '#1e293b', borderBottom: '1px solid #334155', padding: '1rem 2rem' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 2 }}>Hereditary Genetics Atlas</div>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: '#f1f5f9' }}>
          🧬 Hereditary Coagulation Disorder Atlas
        </h1>
        <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>
          Complete 8-Gene Haemophilia, VWD, and Rare Coagulation Factor Deficiency Atlas —
          F8 · F9 · VWF · F11 · F7 · F13A1 · F10 · LMAN1 · 320 patients · seeds 1982-1989
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#1e293b', borderBottom: '1px solid #334155', padding: '0 2rem', display: 'flex', gap: 4 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            background: 'none', border: 'none', borderBottom: tab === t ? '2px solid #6366f1' : '2px solid transparent',
            color: tab === t ? '#a5b4fc' : '#64748b', padding: '0.6rem 1rem', cursor: 'pointer',
            fontSize: 13, fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* Content */}
      <div style={{ padding: '1.5rem 2rem', maxWidth: 1200 }}>
        {err && <ErrorBox msg={err} />}
        {tab === 'Overview'      && <OverviewTab      data={overview}   />}
        {tab === 'Gene Table'    && <GeneTableTab                        />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown}  />}
        {tab === 'Definitions'   && <DefinitionsTab   data={defs}       />}
      </div>
    </div>
  );
}
