'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PTPN11: '#1565c0',  // deep blue     — SHP-2 / Noonan NS#1 / JMML risk
  SOS1:   '#0d6e3d',  // deep green    — RAS-GEF / NS type 4 / best cognition
  RAF1:   '#b71c1c',  // deep red      — RAF kinase / NS+HCM highest risk
  RIT1:   '#880e4f',  // deep magenta  — small GTPase / NS type 8 / chylothorax
  BRAF:   '#4a148c',  // deep purple   — B-Raf / CFC #1 / severe ID
  MAP2K1: '#e65100',  // deep orange   — MEK1 / CFC type 3 / ichthyosis
  HRAS:   '#1b5e20',  // forest green  — Harvey RAS / Costello / tumour risk
  SHOC2:  '#f57f17',  // amber         — scaffold / NSLH / loose anagen hair
};

const GENE_INFO = {
  PTPN11: { aa: 593,  locus: '12q24.13', inh: 'AD', disease: 'Noonan-NS1 — SHP2-GOF — JMML-Risk-200x — Pulmonary-Stenosis-60-70pct — Coagulopathy-Pre-Op' },
  SOS1:   { aa: 1333, locus: '2p22.1',   inh: 'AD', disease: 'Noonan-NS4 — RAS-GEF-GOF — Best-Cognition-NS — Sparse-Eyebrows-KP-Ectodermal — Low-JMML-Risk' },
  RAF1:   { aa: 648,  locus: '3p25.2',   inh: 'AD', disease: 'Noonan+HCM — HIGHEST-HCM-Risk-90pct — pS257L-Hotspot — Biventricular-Neonatal — GH-Caution-HCM' },
  RIT1:   { aa: 219,  locus: '1q22',     inh: 'AD', disease: 'Noonan-NS8 — Second-Highest-HCM-72pct — Chylothorax-Lymphatic — MCT-Diet-Octreotide — High-NT-Prenatal' },
  BRAF:   { aa: 766,  locus: '7q34',     inh: 'AD', disease: 'CFC-type1 — Most-Common-CFC-75pct — Severe-ID-DISTINCTIVE — V600E-Cancer-NOT-CFC — Absent-Eyebrows-Ichthyosis' },
  MAP2K1: { aa: 393,  locus: '15q22.31', inh: 'AD', disease: 'CFC-type3 — MEK1-GOF — Ichthyosis-PATHOGNOMONIC — Trametinib-Direct-Target — Test-MAP2K2-Also' },
  HRAS:   { aa: 189,  locus: '11p15.5',  inh: 'AD', disease: 'Costello — Tumour-15-17pct-Age20 — Papillomata-PATHOGNOMONIC — Rhabdomyosarcoma-Bladder-Ca — GH-Controversial' },
  SHOC2:  { aa: 580,  locus: '10q25.2',  inh: 'AD', disease: 'NSLH-Mazzanti — Painless-Hair-Extraction-PATHOGNOMONIC — pS2G-80pct-Hotspot — Anagen-Effluvium — Trichogram' },
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
  const lv = /ABSOLUTELY.CI|PATHOGNOMONIC|MANDATORY|ABSOLUTELY|FATAL|NEVER|LETHAL|HIGHEST/i.test(text) ? 'critical'
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
        <KPI label="Total Patients"             value={data.total_patients}            color="#6366f1" />
        <KPI label="Genes Covered"              value={(data.genes || []).length || 8} color="#10b981" />
        <KPI label="Seeds"                      value={data.seeds}                     color="#f59e0b" />
        <KPI label="HCM Patients"               value={data.hcm_patients}              color="#ef4444" />
        <KPI label="Pulmonary Stenosis Pts"     value={data.pulmonary_stenosis_patients} color="#3b82f6" />
        <KPI label="JMML Patients (PTPN11)"     value={data.jmml_patients}             color="#dc2626" />
        <KPI label="Malignant Tumour Pts (HRAS)"value={data.tumour_patients}           color="#f97316" />
        <KPI label="Papillomata Pts (Costello)" value={data.papillomata_patients}      color="#8b5cf6" />
        <KPI label="Loose Anagen Hair (SHOC2)"  value={data.loose_anagen_hair_patients} color="#f59e0b" />
        <KPI label="Ichthyosis (CFC)"           value={data.ichthyosis_patients}       color="#0d9488" />
        <KPI label="GH Therapy Patients"        value={data.gh_therapy_patients}       color="#6366f1" />
        <KPI label="Chylothorax Pts (RIT1+)"    value={data.chylothorax_patients}      color="#0ea5e9" />
      </div>

      {/* Inheritance breakdown bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Inheritance Pattern (8 genes — all AD)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'AD Noonan spectrum (PTPN11, SOS1, RAF1, RIT1, SHOC2)', val: 5, color: '#3b82f6' },
            { label: 'AD CFC syndrome (BRAF, MAP2K1)', val: 2, color: '#8b5cf6' },
            { label: 'AD Costello syndrome (HRAS)', val: 1, color: '#ef4444' },
          ].map(b => (
            <div key={b.label} title={b.label} style={{ flex: b.val, background: b.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#fff', fontWeight: 700, overflow: 'hidden' }}>
              {b.val}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 6, fontSize: 11, color: '#64748b', flexWrap: 'wrap' }}>
          <span style={{ color: '#3b82f6' }}>■ NS spectrum (5: PTPN11, SOS1, RAF1, RIT1, SHOC2)</span>
          <span style={{ color: '#8b5cf6' }}>■ CFC syndrome (2: BRAF, MAP2K1)</span>
          <span style={{ color: '#ef4444' }}>■ Costello syndrome (1: HRAS)</span>
        </div>
      </div>

      {/* Pathway */}
      {data.pathway && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>RAS-MAPK Molecular Pathway</div>
          <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{data.pathway}</div>
        </div>
      )}

      {/* Key clinical insight */}
      {data.key_clinical_insight && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Key Clinical Insights per Gene</div>
          {data.key_clinical_insight.split('. ').filter(Boolean).map((s, i) => (
            <Alert key={i} text={s.trim()} />
          ))}
        </div>
      )}

      {/* Gene summary table */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Gene Summary (320 patients, 8 × 40)</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ color: '#64748b', borderBottom: '1px solid #334155' }}>
              {['Gene', 'Locus', 'aa', 'Inheritance', 'Disease Class', 'Patients'].map(h => (
                <th key={h} style={{ padding: '6px 8px', textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(geneCounts).map(([gene, cnt]) => {
              const info = GENE_INFO[gene] || {};
              return (
                <tr key={gene} style={{ borderBottom: '1px solid #1e293b55' }}>
                  <td style={{ padding: '5px 8px', color: GENE_COLORS[gene] || '#f1f5f9', fontWeight: 700 }}>{gene}</td>
                  <td style={{ padding: '5px 8px', color: '#cbd5e1', fontFamily: 'monospace' }}>{info.locus || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{info.aa || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#94a3b8' }}>{info.inh || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#64748b', fontSize: 11, maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{info.disease || '—'}</td>
                  <td style={{ padding: '5px 8px', color: '#f1f5f9', fontWeight: 600 }}>{cnt}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>RASopathy Gene Detail Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ background: '#0f172a', color: '#64748b' }}>
              {['Gene', 'Locus', 'aa', 'Inh.', 'n', 'HCM%', 'PS%', 'JMML%', 'Tumour%', 'CogImp%', 'SevID%', 'ShortSt%', 'GH%', 'KP%', 'Papill%', 'LooseHair%', 'Ichthy%', 'Chyloth%', 'MEKinh%'].map(h => (
                <th key={h} style={{ padding: '6px 6px', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '2px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                <td style={{ padding: '5px 6px', color: GENE_COLORS[g.gene] || '#f1f5f9', fontWeight: 700, whiteSpace: 'nowrap' }}>{g.gene}</td>
                <td style={{ padding: '5px 6px', color: '#94a3b8', fontFamily: 'monospace' }}>{g.locus}</td>
                <td style={{ padding: '5px 6px', color: '#94a3b8' }}>{g.protein_size}</td>
                <td style={{ padding: '5px 6px', color: '#94a3b8' }}>{g.inheritance.split(' ')[0]}</td>
                <td style={{ padding: '5px 6px', color: '#f1f5f9', fontWeight: 600 }}>{g.n_patients}</td>
                <td style={{ padding: '5px 6px', color: g.hcm_pct > 60 ? '#ef4444' : '#cbd5e1' }}>{g.hcm_pct}%</td>
                <td style={{ padding: '5px 6px', color: '#cbd5e1' }}>{g.pulmonary_stenosis_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.jmml_pct > 0 ? '#ef4444' : '#475569' }}>{g.jmml_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.tumour_pct > 0 ? '#f97316' : '#475569' }}>{g.tumour_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.cognitive_impairment_pct > 60 ? '#fcd34d' : '#cbd5e1' }}>{g.cognitive_impairment_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.severe_id_pct > 50 ? '#ef4444' : '#cbd5e1' }}>{g.severe_id_pct}%</td>
                <td style={{ padding: '5px 6px', color: '#cbd5e1' }}>{g.short_stature_pct}%</td>
                <td style={{ padding: '5px 6px', color: '#94a3b8' }}>{g.gh_therapy_pct}%</td>
                <td style={{ padding: '5px 6px', color: '#94a3b8' }}>{g.keratosis_pilaris_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.papillomata_pct > 0 ? '#8b5cf6' : '#475569' }}>{g.papillomata_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.loose_anagen_hair_pct > 50 ? '#f59e0b' : '#475569' }}>{g.loose_anagen_hair_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.ichthyosis_pct > 50 ? '#0d9488' : '#475569' }}>{g.ichthyosis_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.chylothorax_pct > 20 ? '#0ea5e9' : '#475569' }}>{g.chylothorax_pct}%</td>
                <td style={{ padding: '5px 6px', color: g.mek_inhibitor_pct > 0 ? '#10b981' : '#475569' }}>{g.mek_inhibitor_pct}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: '1rem', fontSize: 11, color: '#475569' }}>
        KP = keratosis pilaris · Papill = papillomata · LooseHair = loose anagen hair · Ichthy = ichthyosis · Chyloth = chylothorax · MEKinh = on MEK inhibitor (trametinib)
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  const [sel, setSel] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.values(data);
  const cur = sel ? data[sel] : null;
  return (
    <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
      {/* Left: gene selector */}
      <div style={{ minWidth: 180 }}>
        <div style={{ fontSize: 12, color: '#64748b', marginBottom: 8 }}>Select Gene</div>
        {genes.map(g => (
          <div key={g.gene}
            onClick={() => setSel(g.gene)}
            style={{
              padding: '8px 12px', marginBottom: 4, borderRadius: 8, cursor: 'pointer',
              background: sel === g.gene ? '#1e3a5f' : '#1e293b',
              borderLeft: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
              color: sel === g.gene ? '#f1f5f9' : '#94a3b8', fontSize: 13, fontWeight: 600,
            }}>
            {g.gene}
            <div style={{ fontSize: 10, color: '#475569', fontWeight: 400 }}>{g.locus}</div>
          </div>
        ))}
      </div>

      {/* Right: gene details */}
      <div style={{ flex: 1, minWidth: 0 }}>
        {!cur ? (
          <div style={{ color: '#475569', padding: '2rem' }}>Select a gene to view clinical details</div>
        ) : (
          <div>
            <h3 style={{ color: GENE_COLORS[cur.gene] || '#f1f5f9', marginBottom: 4 }}>{cur.gene}</h3>
            <div style={{ fontSize: 12, color: '#64748b', marginBottom: '1rem' }}>{cur.alt_name}</div>

            {/* Critical flags */}
            <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem', marginBottom: '1rem' }}>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Critical Flags</div>
              {(cur.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
            </div>

            {/* Stats bar */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1rem' }}>
              {[
                ['HCM', cur.hcm_pct, '#ef4444'],
                ['PS', cur.pulmonary_stenosis_pct, '#3b82f6'],
                ['Cognit.Imp', cur.cognitive_impairment_pct, '#fcd34d'],
                ['Severe ID', cur.severe_id_pct, '#f97316'],
                ['JMML', cur.jmml_pct, '#dc2626'],
                ['Tumour', cur.tumour_pct, '#f97316'],
                ['ShortSt.', cur.short_stature_pct, '#10b981'],
                ['GH Rx', cur.gh_therapy_pct, '#6366f1'],
                ['KP', cur.keratosis_pilaris_pct, '#0d9488'],
                ['Papillomata', cur.papillomata_pct, '#8b5cf6'],
                ['LooseHair', cur.loose_anagen_hair_pct, '#f59e0b'],
                ['Ichthyosis', cur.ichthyosis_pct, '#0ea5e9'],
                ['Chylothorax', cur.chylothorax_pct, '#0ea5e9'],
              ].map(([label, val, col]) => (
                <div key={label} style={{ background: '#0f172a', borderRadius: 6, padding: '6px 10px', minWidth: 80 }}>
                  <div style={{ fontSize: 10, color: '#64748b' }}>{label}</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: col }}>{val}%</div>
                </div>
              ))}
            </div>

            {/* Clinical details */}
            {[
              ['Age of Onset', cur.age_of_onset],
              ['Key Biomarkers', cur.key_biomarker],
              ['Pathognomonic Features', cur.pathognomonic],
              ['Treatment', cur.treatment],
            ].map(([title, text]) => (
              <div key={title} style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem', marginBottom: '0.8rem' }}>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>{title}</div>
                <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{text}</div>
              </div>
            ))}

            {/* Cohort preview */}
            {cur.cohort_preview && (
              <div style={{ background: '#1e293b', borderRadius: 8, padding: '0.8rem' }}>
                <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>Cohort Preview (first 5)</div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                  <thead>
                    <tr style={{ color: '#475569', borderBottom: '1px solid #334155' }}>
                      {['ID', 'Age', 'Sex', 'HCM', 'PS', 'JMML', 'Tumour', 'Papill', 'LooseHair', 'Ichthy', 'Chyloth'].map(h => (
                        <th key={h} style={{ padding: '4px 6px', textAlign: 'left' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cur.cohort_preview.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #0f172a' }}>
                        <td style={{ padding: '3px 6px', color: '#64748b', fontFamily: 'monospace', fontSize: 10 }}>{p.patient_id}</td>
                        <td style={{ padding: '3px 6px', color: '#94a3b8' }}>{p.age}</td>
                        <td style={{ padding: '3px 6px', color: '#94a3b8' }}>{p.sex}</td>
                        <td style={{ padding: '3px 6px', color: p.hcm ? '#ef4444' : '#475569' }}>{p.hcm ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.pulmonary_stenosis ? '#3b82f6' : '#475569' }}>{p.pulmonary_stenosis ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.jmml ? '#dc2626' : '#475569' }}>{p.jmml ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.tumour_malignant ? '#f97316' : '#475569' }}>{p.tumour_malignant ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.papillomata ? '#8b5cf6' : '#475569' }}>{p.papillomata ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.loose_anagen_hair ? '#f59e0b' : '#475569' }}>{p.loose_anagen_hair ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.ichthyosis ? '#0d9488' : '#475569' }}>{p.ichthyosis ? 'Y' : '—'}</td>
                        <td style={{ padding: '3px 6px', color: p.chylothorax ? '#0ea5e9' : '#475569' }}>{p.chylothorax ? 'Y' : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
      <h3 style={{ color: '#f1f5f9', marginBottom: '0.5rem' }}>RASopathy — Glossary & Pathway</h3>

      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 6 }}>Shared Mechanism</div>
        <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{data.shared_mechanism}</div>
      </div>

      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 6 }}>Tumour Surveillance Protocols</div>
        {Object.entries(data.surveillance_protocols || {}).map(([gene, proto]) => (
          <div key={gene} style={{ marginBottom: 8 }}>
            <span style={{ color: GENE_COLORS[gene] || '#f1f5f9', fontWeight: 700, fontSize: 12 }}>{gene}: </span>
            <span style={{ color: '#94a3b8', fontSize: 12 }}>{proto}</span>
          </div>
        ))}
      </div>

      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 10 }}>Glossary</div>
        {Object.entries(data.glossary || {}).map(([term, def]) => (
          <div key={term} style={{ marginBottom: 10, borderBottom: '1px solid #0f172a', paddingBottom: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: '#a5b4fc', marginBottom: 2 }}>{term}</div>
            <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>{def}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── ROOT PAGE ────────────────────────────────────────────────────────────── */
export default function HereditaryRASopathyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-rasopathy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-rasopathy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-rasopathy-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto' }}>
        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9', marginBottom: 4 }}>
            🧬 Hereditary RASopathy Atlas
          </h1>
          <div style={{ fontSize: 13, color: '#64748b' }}>
            Complete 8-Gene RAS-MAPK Pathway Disorder Reference — PTPN11 · SOS1 · RAF1 · RIT1 · BRAF · MAP2K1 · HRAS · SHOC2 · seeds 1990-1997
          </div>
        </div>

        {error && <ErrorBox msg={error} />}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '6px 16px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t ? '#6366f1' : '#1e293b',
              color: tab === t ? '#fff' : '#94a3b8', fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>

        {/* Tab content */}
        {tab === 'Overview'      && <OverviewTab      data={overview}     />}
        {tab === 'Gene Table'    && <GeneTableTab     data={breakdown}    />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown}    />}
        {tab === 'Definitions'   && <DefinitionsTab   data={definitions}  />}
      </div>
    </div>
  );
}
