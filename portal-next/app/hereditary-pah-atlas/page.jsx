'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BMPR2:   '#1565c0',  // deep blue    — BMP receptor / most common HPAH
  ACVRL1:  '#0d6e3d',  // deep green   — ALK1 / HHT2-PAH
  ENG:     '#b71c1c',  // deep red     — endoglin / HHT1-PAVM
  SMAD9:   '#880e4f',  // deep magenta — BMP-SMAD downstream
  CAV1:    '#4a148c',  // deep purple  — caveolin / lipodystrophy
  KCNK3:   '#e65100',  // deep orange  — TASK-1 / GOF
  TBX4:    '#1b5e20',  // forest green — paediatric PAH
  EIF2AK4: '#b71c1c',  // crimson      — PVOD / vasodilators CI
};

const GENE_INFO = {
  BMPR2:   { aa: 2037, locus: '2q33.1',  inh: 'AD',    disease: 'Hereditary-PAH-Most-Common-70-80pct — Penetrance-20pct-Only — ERA+PDE5i+Prostacyclin — Pregnancy-30-56pct-Mortality' },
  ACVRL1:  { aa: 503,  locus: '12q13.13',inh: 'AD',    disease: 'HHT2-Associated-PAH — Hepatic-AVM-Overlap — Distinguish-PAH-From-High-Output-Failure — Hepatic-Embolisation-CI' },
  ENG:     { aa: 658,  locus: '9q34.11', inh: 'AD',    disease: 'HHT1-PAVM-Predominant — PAH-Rare-HHT1 — Antibiotic-Prophylaxis-Dental-MANDATORY — Paradoxical-Embolism-Brain-Abscess' },
  SMAD9:   { aa: 530,  locus: '13q13.3', inh: 'AD',    disease: 'Rare-HPAH-BMP-SMAD-Downstream — Phenotype-Identical-BMPR2 — Include-All-PAH-Panels — Sotatercept-Rational' },
  CAV1:    { aa: 178,  locus: '7q31.2',  inh: 'AD/AR', disease: 'Rare-HPAH-Caveolin-1 — AR-Biallelic-CGL3-Lipodystrophy — Low-Penetrance — BMPR2-Caveolae-Interaction' },
  KCNK3:   { aa: 354,  locus: '2p23.3',  inh: 'AD',    disease: 'TASK1-GOF-Unusual-HPAH — Reduced-K-Current-Vasoconstriction — Doxapram-Investigational — Avoid-Bupivacaine' },
  TBX4:    { aa: 520,  locus: '17q23.2', inh: 'AD',    disease: 'Paediatric-PAH-20pct-Childhood-HPAH — Small-Patella-Absent — Many-No-Musculoskeletal — Acinar-Dysplasia' },
  EIF2AK4: { aa: 1649, locus: '15q15.1', inh: 'AR',    disease: 'PVOD-PCH-VASODILATORS-ABSOLUTELY-CI — Lung-Transplant-ONLY-Curative — HRCT-Centrilobular-GGO-PATHOGNOMONIC — List-Transplant-Early' },
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
  const lv = /ABSOLUTELY.CI|PATHOGNOMONIC|MANDATORY|CONTRAINDICATED|FATAL|NEVER|LETHAL|ONLY.CURATIVE/i.test(text) ? 'critical'
           : /\bCI\b|RISK|MONITOR|WARNING|AVOID|CAUTION/i.test(text) ? 'warning' : 'info';
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
        <KPI label="Total Patients"              value={data.total_patients}                color="#6366f1" />
        <KPI label="Genes Covered"               value={(data.genes || []).length || 8}     color="#10b981" />
        <KPI label="Seeds"                       value={data.seeds}                         color="#f59e0b" />
        <KPI label="mPAP Elevated Pts"           value={data.mPAP_elevated_patients}        color="#ef4444" />
        <KPI label="Right Heart Failure Pts"     value={data.right_heart_failure_patients}  color="#dc2626" />
        <KPI label="PVOD Pattern (EIF2AK4)"      value={data.pvod_pattern_patients}         color="#b91c1c" />
        <KPI label="HHT Features Pts"            value={data.hht_features_patients}         color="#0d9488" />
        <KPI label="PAVM Patients (ENG)"         value={data.pavm_patients}                 color="#0ea5e9" />
        <KPI label="Childhood Onset (TBX4+)"     value={data.childhood_onset_patients}      color="#8b5cf6" />
        <KPI label="Small Patella (TBX4)"        value={data.small_patella_patients}        color="#a855f7" />
        <KPI label="Transplant Listed/Done"      value={data.transplant_patients}           color="#f97316" />
        <KPI label="Triple Therapy Pts"          value={data.triple_therapy_patients}       color="#3b82f6" />
      </div>

      {/* Inheritance bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Inheritance Pattern (8 genes)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'AD Group 1 PAH (BMPR2, ACVRL1, ENG, SMAD9, CAV1, KCNK3, TBX4)', val: 7, color: '#3b82f6' },
            { label: 'AR PVOD/PCH (EIF2AK4)', val: 1, color: '#dc2626' },
          ].map(b => (
            <div key={b.label} style={{
              flex: b.val, background: b.color, display: 'flex',
              alignItems: 'center', justifyContent: 'center',
              fontSize: 9, color: '#fff', overflow: 'hidden', whiteSpace: 'nowrap',
            }} title={b.label}>{b.label}</div>
          ))}
        </div>
      </div>

      {/* Critical warning for EIF2AK4 PVOD */}
      <div style={{ background: '#450a0a', border: '2px solid #dc2626', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#fca5a5', marginBottom: 6 }}>
          ⚠ CRITICAL — EIF2AK4 (PVOD/PCH): VASODILATORS ABSOLUTELY CONTRAINDICATED
        </div>
        <div style={{ fontSize: 12, color: '#fca5a5', lineHeight: 1.6 }}>
          EIF2AK4 biallelic LOF causes Pulmonary Veno-Occlusive Disease (PVOD) — NOT arterial PAH.
          Pulmonary vasodilators (ERA, PDE5i, prostacyclin) trigger flash pulmonary oedema and death.
          Lung transplant is the ONLY curative treatment. List for transplant early — rapid deterioration.
          HRCT centrilobular GGO + interlobular septal thickening + mediastinal lymphadenopathy = PVOD pathognomonic.
        </div>
      </div>

      {/* Gene patient breakdown */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 12 }}>Patients per Gene (40 each)</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {Object.entries(geneCounts).map(([gene, n]) => (
            <div key={gene} style={{
              background: '#0f172a', borderRadius: 8, padding: '0.5rem 0.8rem',
              borderLeft: `3px solid ${GENE_COLORS[gene] || '#6366f1'}`,
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc' }}>{gene}</div>
              <div style={{ fontSize: 11, color: '#94a3b8' }}>{n} pts</div>
              <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>
                {GENE_INFO[gene]?.aa}aa · {GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.inh}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key clinical insight */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Key Clinical Insights</div>
        <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{data.key_clinical_insight}</div>
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab() {
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>8-Gene PAH / PVOD Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Protein (aa)', 'Locus', 'Inh.', 'Disease / Syndrome', 'Critical Flag'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600, borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(GENE_INFO).map(([gene, info], idx) => (
              <tr key={gene} style={{ background: idx % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc' }}>{gene}</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{info.aa} aa</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{info.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{info.inh}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 300 }}>{info.disease.replace(/-/g,' ').replace(/(\w)/,c=>c)}</td>
                <td style={{ padding: '8px 10px', color: gene === 'EIF2AK4' ? '#fca5a5' : '#fcd34d', fontWeight: gene === 'EIF2AK4' ? 700 : 400, maxWidth: 220, fontSize: 11 }}>
                  {gene === 'EIF2AK4' ? '⚠ VASODILATORS CONTRAINDICATED — LUNG TRANSPLANT ONLY' :
                   gene === 'BMPR2'   ? 'Penetrance 20% only — cascade test ALL relatives' :
                   gene === 'ACVRL1'  ? 'Hepatic AVM high-output vs PAH — RHC with CO mandatory' :
                   gene === 'ENG'     ? 'Antibiotic prophylaxis dental MANDATORY (PAVM)' :
                   gene === 'TBX4'    ? '~20% childhood HPAH — small patella X-ray' :
                   gene === 'KCNK3'   ? 'GOF mechanism — avoid bupivacaine' :
                   gene === 'CAV1'    ? 'AR biallelic = CGL3 lipodystrophy' :
                                        'Include on all PAH gene panels'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ────────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  const [selGene, setSelGene] = useState('BMPR2');
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  const g = data[selGene];
  if (!g) return <Loading />;

  return (
    <div>
      {/* Gene selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
        {genes.map(gn => (
          <button key={gn} onClick={() => setSelGene(gn)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: selGene === gn ? (GENE_COLORS[gn] || '#6366f1') : '#1e293b',
            color: selGene === gn ? '#fff' : '#94a3b8',
          }}>{gn}</button>
        ))}
      </div>

      {/* Gene panel */}
      <div style={{ background: '#1e293b', borderRadius: 12, padding: '1.2rem', borderLeft: `4px solid ${GENE_COLORS[selGene] || '#6366f1'}` }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[selGene] || '#a5b4fc', marginBottom: 4 }}>{selGene}</div>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 12 }}>
          {g.protein_size} · {g.locus} · {g.inheritance} · n={g.n_patients}
        </div>

        {/* Stats bar */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
          {[
            ['mPAP Elevated', g.mPAP_elevated_pct],
            ['RV Failure', g.right_heart_failure_pct],
            ['Transplant', g.transplant_pct],
            ['Haemoptysis', g.haemoptysis_pct],
            ['PVOD HRCT', g.pvod_pattern_hrct_pct],
            ['HHT Features', g.hht_features_pct],
            ['Childhood Onset', g.childhood_onset_pct],
            ['Small Patella', g.small_patella_pct],
          ].map(([lbl, val]) => val > 0 && (
            <div key={lbl} style={{ background: '#0f172a', borderRadius: 6, padding: '4px 8px', fontSize: 11, color: '#94a3b8' }}>
              {lbl}: <span style={{ color: val > 60 ? '#ef4444' : val > 30 ? '#f59e0b' : '#10b981', fontWeight: 600 }}>{val}%</span>
            </div>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Age of Onset</div>
            <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.age_of_onset}</div>
          </div>
          <div>
            <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Key Biomarker</div>
            <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.key_biomarker}</div>
          </div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Pathognomonic / Distinguishing Features</div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.pathognomonic}</div>
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 4 }}>Treatment</div>
          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.6 }}>{g.treatment}</div>
        </div>

        <div>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8', marginBottom: 6 }}>Critical Flags</div>
          {(g.critical_flags || []).map(f => <Alert key={f} text={f} />)}
        </div>
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '0.5rem' }}>Pathway & Glossary</h3>
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem', fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>
        <strong style={{ color: '#a5b4fc' }}>Pathway:</strong> {data.pathway}<br/>
        <strong style={{ color: '#a5b4fc', marginTop: 8, display: 'block' }}>Shared Mechanism:</strong> {data.shared_mechanism}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {Object.entries(data.glossary || {}).map(([term, def]) => (
          <div key={term} style={{ background: '#1e293b', borderRadius: 8, padding: '0.7rem', fontSize: 11 }}>
            <div style={{ fontWeight: 700, color: '#a5b4fc', marginBottom: 3 }}>{term}</div>
            <div style={{ color: '#94a3b8', lineHeight: 1.5 }}>{def}</div>
          </div>
        ))}
      </div>
      {data.surveillance_protocols && (
        <div style={{ marginTop: '1.5rem' }}>
          <h4 style={{ color: '#f1f5f9', marginBottom: '0.8rem' }}>Surveillance Protocols</h4>
          {Object.entries(data.surveillance_protocols).map(([gene, proto]) => (
            <div key={gene} style={{ background: '#1e293b', borderRadius: 8, padding: '0.6rem 0.8rem', marginBottom: 6, fontSize: 11 }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', marginRight: 8 }}>{gene}</span>
              <span style={{ color: '#94a3b8' }}>{proto}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── MAIN PAGE ────────────────────────────────────────────────────────────── */
export default function HereditaryPAHAtlasPage() {
  const [tab, setTab]           = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]         = useState(null);
  const [err, setErr]           = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-pah-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-pah-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-pah-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefs(df); })
      .catch(e => setErr(e.message));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '2rem 1.5rem' }}>
        <div style={{ marginBottom: '0.3rem', fontSize: 12, color: '#475569' }}>
          Hereditary Disease Atlas · Pulmonary Hypertension · Seeds 1998–2005
        </div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800, color: '#f8fafc', marginBottom: '0.3rem' }}>
          Hereditary PAH Atlas
        </h1>
        <p style={{ margin: '0 0 1.5rem', color: '#94a3b8', fontSize: 13 }}>
          Complete 8-Gene Hereditary PAH / PVOD Atlas — BMPR2 · ACVRL1 · ENG · SMAD9 · CAV1 · KCNK3 · TBX4 · EIF2AK4
        </p>

        {err && <ErrorBox msg={err} />}

        {/* Tab bar */}
        <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', borderBottom: '1px solid #334155', paddingBottom: 4 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '6px 16px', borderRadius: '6px 6px 0 0', border: 'none', cursor: 'pointer',
              background: tab === t ? '#6366f1' : 'transparent',
              color: tab === t ? '#fff' : '#64748b', fontSize: 13, fontWeight: tab === t ? 600 : 400,
            }}>{t}</button>
          ))}
        </div>

        {tab === 'Overview'      && <OverviewTab data={overview} />}
        {tab === 'Gene Table'    && <GeneTableTab />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'   && <DefinitionsTab data={defs} />}
      </div>
    </div>
  );
}
