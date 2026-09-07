'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PCCA:   '#dc2626',  // red    — propionic acidemia alpha; VPA-CI; DCM risk
  PCCB:   '#c2410c',  // orange — propionic acidemia beta; Korean/Italian founder
  MMUT:   '#b45309',  // amber  — MMA mut type; renal CKD; OHCbl trial mandatory
  IVD:    '#15803d',  // green  — isovaleric acidemia; sweaty feet; glycine therapy
  GCDH:   '#1d4ed8',  // blue   — GA1; macrocephaly; striatal necrosis; NAI DDx
  MCCC1:  '#7c3aed',  // purple — 3-MCC; C5OH; usually benign; NOT biotin-responsive
  ACAT1:  '#0f766e',  // teal   — beta-ketothiolase; episodic ketoacidosis; excellent Px
  HLCS:   '#065f46',  // dark green — HLCS; neonatal MCD; BIOTIN-RESPONSIVE
};

const GENE_DISEASE = {
  PCCA:  'AR Propionic-Acidemia-TypeA — PCCA-728aa — 13q32.3 — PCC-Alpha-Biotin-Carboxylase — C3-NBS — DCM-20-30pct — VPA-ABSOLUTELY-CI — Liver-Tx-NOT-Prevent-Cardiomyopathy',
  PCCB:  'AR Propionic-Acidemia-TypeB — PCCB-539aa — 3q22.3 — PCC-Beta-Carboxyl-Transferase — Korean-Founder-p.Gln272Ter — Identical-Phenotype-PCCA — VPA-ABSOLUTELY-CI',
  MMUT:  'AR MMA-mut-Type — MMUT-750aa — 6p12.3 — Methylmalonyl-CoA-Mutase-AdoCbl — mut0-NOT-OHCbl-Responsive — Renal-CKD-Major-Morbidity — NSAIDs-CI — Liver-Kidney-Transplant',
  IVD:   'AR Isovaleric-Acidemia — IVD-415aa — 15q15.1 — Isovaleryl-CoA-Dehydrogenase-FAD — C5-NBS-PATHOGNOMONIC — Sweaty-Feet-Cheese-Odour — Glycine-250mgkgday — German-Founder-p.Ala282Val',
  GCDH:  'AR Glutaric-Aciduria-Type-1 — GCDH-438aa — 19p13.2 — Glutaryl-CoA-Dehydrogenase-FAD — Macrocephaly-HALLMARK-75-80pct — Frontotemporal-Atrophy-NAI-DDx — Striatal-Necrosis-Window-6m-6y — Amish-Mennonite-Founder',
  MCCC1: 'AR 3-Methylcrotonyl-CoA-Carboxylase-Deficiency — MCCC1-709aa — 3q27.1 — 3-MCC-Alpha-Biotin-Carboxylase — C5OH-NBS-Most-Common-OA-Flag — USUALLY-BENIGN — NOT-Biotin-Responsive',
  ACAT1: 'AR Beta-Ketothiolase-Deficiency-T2 — ACAT1-427aa — 11q22.3 — Mitochondrial-Acetoacetyl-CoA-Thiolase — Episodic-Ketoacidosis-DISPROPORTIONATE — 2-MAA-Urine-MOST-SPECIFIC — Excellent-Prognosis-Between-Episodes — VPA-CI',
  HLCS:  'AR Holocarboxylase-Synthetase-Deficiency-Neonatal-MCD — HLCS-726aa — 21q22.13 — All-4-Biotin-Carboxylases-Inactive — Skin-Rash-Alopecia-Lactic-Acidosis-COMBINATION — BIOTIN-RESPONSIVE-10-40mg-day — Life-Long-Mandatory',
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

function Alert({ text, level }) {
  const colors = { critical: '#fca5a5', warning: '#fcd34d', info: '#93c5fd' };
  const bg     = { critical: '#450a0a', warning: '#451a03', info: '#0c1a3a' };
  const lv = (text || '').includes('ABSOLUTELY-CI') || (text || '').includes('PATHOGNOMONIC') || (text || '').includes('MANDATORY') ? 'critical'
           : (text || '').includes('CI') || (text || '').includes('RISK') || (text || '').includes('RESPONSIVE') ? 'warning' : 'info';
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
  const s = data.aggregate_stats || {};
  const gs = data.gene_summary || [];
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: '1.5rem', lineHeight: 1.5 }}>
        {data.subtitle}
      </p>

      {/* KPIs */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients"    value={data.total_patients}           color="#6366f1" />
        <KPI label="Genes Covered"     value={s.genes_covered || 8}          color="#10b981" />
        <KPI label="Seeds"             value={data.seed_range}               color="#f59e0b" />
        <KPI label="On Diet %"         value={`${s.on_diet_pct ?? '—'}%`}    color="#3b82f6" />
        <KPI label="Crisis History %"  value={`${s.crisis_history_pct ?? '—'}%`} color="#ef4444" />
        <KPI label="Family Cascade %"  value={`${s.family_cascade_pct ?? '—'}%`} color="#8b5cf6" />
        <KPI label="Biotin-Responsive" value={s.biotin_responsive_genes ?? 1} color="#0d9488" />
      </div>

      {/* Severity bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.5rem' }}>
        <div style={{ fontSize: 13, color: '#94a3b8', marginBottom: 8 }}>Severity Distribution (all 320 patients)</div>
        <div style={{ display: 'flex', height: 20, borderRadius: 6, overflow: 'hidden', gap: 2 }}>
          {[
            { label: 'Mild',     val: s.severity_mild_pct,     color: '#10b981' },
            { label: 'Moderate', val: s.severity_moderate_pct, color: '#f59e0b' },
            { label: 'Severe',   val: s.severity_severe_pct,   color: '#ef4444' },
          ].map(b => (
            <div key={b.label} style={{ flex: b.val || 0, background: b.color, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 10, color: '#fff', fontWeight: 700 }}>
              {b.val ? `${b.val}%` : ''}
            </div>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 16, marginTop: 6 }}>
          {[['Mild','#10b981'],['Moderate','#f59e0b'],['Severe','#ef4444']].map(([l,c]) => (
            <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: '#94a3b8' }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: c }} />{l}
            </div>
          ))}
        </div>
      </div>

      {/* Gene summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 12, marginBottom: '1.5rem' }}>
        {gs.map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '0.85rem 1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
              <span style={{ fontSize: 11, color: '#64748b', background: '#0f172a', padding: '2px 7px', borderRadius: 4 }}>{g.inheritance} · {g.chromosome}</span>
            </div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}><b>{g.disease_short}</b></div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>NBS: <span style={{ color: '#fcd34d' }}>{g.nbs_marker}</span></div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>Pearl: <span style={{ color: '#86efac' }}>{g.management_pearl}</span></div>
            {g.key_finding && <div style={{ fontSize: 11, color: '#f87171', marginTop: 3 }}>⚠ {g.key_finding}</div>}
          </div>
        ))}
      </div>

      {/* Critical alerts */}
      {data.top_alerts && data.top_alerts.length > 0 && (
        <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1rem' }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9', marginBottom: 8 }}>Critical Clinical Alerts</div>
          {data.top_alerts.map((a, i) => <Alert key={i} text={a} />)}
        </div>
      )}
    </div>
  );
}

/* ── GENE TABLE TAB ───────────────────────────────────────────────────────── */
function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || {};
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Per-Gene Breakdown — 40 Patients Each</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene','Locus','Protein','NBS Marker','Mild%','Mod%','Sev%','Crisis%','Diet%','Cascade%','Biotin-R'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #1e293b' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.values(genes).map((g, i) => {
              const total = g.n_patients || 40;
              const sv = g.severity || {};
              const mp = v => `${Math.round((v/total)*100)}%`;
              return (
                <tr key={g.gene} style={{ background: i%2===0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '7px 10px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                  <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '7px 10px', color: '#94a3b8' }}>{g.protein_size}</td>
                  <td style={{ padding: '7px 10px', color: '#fcd34d', maxWidth: 220 }}>{g.key_biomarker}</td>
                  <td style={{ padding: '7px 10px', color: '#10b981' }}>{mp(sv.mild||0)}</td>
                  <td style={{ padding: '7px 10px', color: '#f59e0b' }}>{mp(sv.moderate||0)}</td>
                  <td style={{ padding: '7px 10px', color: '#ef4444' }}>{mp(sv.severe||0)}</td>
                  <td style={{ padding: '7px 10px', color: '#f87171' }}>{g.crisis_history_pct}%</td>
                  <td style={{ padding: '7px 10px', color: '#3b82f6' }}>{g.on_diet_pct}%</td>
                  <td style={{ padding: '7px 10px', color: '#8b5cf6' }}>{g.family_cascade_pct}%</td>
                  <td style={{ padding: '7px 10px', color: g.biotin_responsive ? '#0d9488' : '#475569' }}>{g.biotin_responsive ? '✓ YES' : 'No'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Critical flags per gene */}
      <div style={{ marginTop: '1.5rem' }}>
        <h4 style={{ color: '#f1f5f9', marginBottom: '0.75rem' }}>Critical Flags Per Gene</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(360px,1fr))', gap: 12 }}>
          {Object.values(genes).map(g => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 10, padding: '0.85rem 1rem', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}` }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 6 }}>{g.gene}</div>
              {(g.critical_flags || []).map((f, i) => <Alert key={i} text={f} />)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── CLINICAL ATLAS TAB ───────────────────────────────────────────────────── */
function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || {};
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Clinical Atlas — Protein Descriptions</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(480px,1fr))', gap: 14 }}>
        {Object.values(genes).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
              <span style={{ fontWeight: 700, fontSize: 16, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
              <div style={{ display: 'flex', gap: 6 }}>
                <span style={{ fontSize: 10, color: '#64748b', background: '#0f172a', padding: '2px 6px', borderRadius: 4 }}>{g.locus}</span>
                <span style={{ fontSize: 10, color: '#64748b', background: '#0f172a', padding: '2px 6px', borderRadius: 4 }}>{g.protein_size}</span>
              </div>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.5, marginBottom: 8 }}>
              {(g.protein_description || GENE_DISEASE[g.gene] || '').replace(/--/g, '·').replace(/_/g, ' ')}
            </div>
            <div style={{ fontSize: 11, color: '#fcd34d', marginBottom: 4 }}>🧪 <b>Biomarker:</b> {g.key_biomarker}</div>
            <div style={{ fontSize: 11, color: '#f87171', marginBottom: 4 }}>🔴 <b>Pathognomonic:</b> {g.pathognomonic}</div>
            <div style={{ fontSize: 11, color: '#86efac' }}>💊 <b>Treatment:</b> {g.treatment}</div>
            <div style={{ marginTop: 8, fontSize: 11, color: '#64748b' }}>
              Onset: {g.age_of_onset} &nbsp;|&nbsp;
              Crisis: {g.crisis_history_pct}% &nbsp;|&nbsp;
              On-diet: {g.on_diet_pct}% &nbsp;|&nbsp;
              Biotin-R: <span style={{ color: g.biotin_responsive ? '#0d9488' : '#475569' }}>{g.biotin_responsive ? 'YES' : 'No'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── DEFINITIONS TAB ──────────────────────────────────────────────────────── */
function DefinitionsTab({ data }) {
  const [open, setOpen] = useState(null);
  if (!data) return <Loading />;
  const defs = data.definitions || [];
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '1rem' }}>Gene Definitions — 8 Genes</h3>
      {defs.map(d => (
        <div key={d.gene} style={{ background: '#1e293b', borderRadius: 10, marginBottom: 10, overflow: 'hidden' }}>
          <button
            onClick={() => setOpen(open === d.gene ? null : d.gene)}
            style={{
              width: '100%', textAlign: 'left', padding: '0.9rem 1rem', background: 'transparent',
              border: 'none', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}
          >
            <span style={{ fontWeight: 700, color: GENE_COLORS[d.gene] || '#a5b4fc', fontSize: 15 }}>
              {d.gene} — {d.full_name}
            </span>
            <span style={{ color: '#64748b' }}>{open === d.gene ? '▲' : '▼'}</span>
          </button>
          {open === d.gene && (
            <div style={{ padding: '0 1rem 1rem', borderTop: '1px solid #0f172a' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', margin: '0.7rem 0', fontSize: 12 }}>
                {[
                  ['OMIM Gene', d.omim_gene], ['OMIM Disease', d.omim_disease],
                  ['Chromosome', d.chromosome], ['Protein Size', `${d.protein_size_aa} aa`],
                  ['Inheritance', d.inheritance], ['NBS Marker', d.nbs_marker],
                ].map(([lbl, val]) => (
                  <div key={lbl} style={{ background: '#0f172a', borderRadius: 6, padding: '0.4rem 0.6rem' }}>
                    <div style={{ color: '#64748b', fontSize: 10 }}>{lbl}</div>
                    <div style={{ color: '#cbd5e1' }}>{val}</div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>
                <b style={{ color: '#f1f5f9' }}>Function:</b> {d.protein_function}
              </div>
              {d.key_clinical_features?.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>Key Clinical Features</div>
                  {d.key_clinical_features.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: '#cbd5e1', padding: '2px 0' }}>• {f}</div>
                  ))}
                </div>
              )}
              {d.pathognomonic_findings?.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#fcd34d', marginBottom: 4 }}>Pathognomonic Findings</div>
                  {d.pathognomonic_findings.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: '#fcd34d', padding: '2px 0' }}>⚡ {f}</div>
                  ))}
                </div>
              )}
              {d.management_mandatories?.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#86efac', marginBottom: 4 }}>Management Mandatories</div>
                  {d.management_mandatories.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: '#86efac', padding: '2px 0' }}>✓ {f}</div>
                  ))}
                </div>
              )}
              {d.contraindicated?.length > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#f87171', marginBottom: 4 }}>Contraindicated / Avoid</div>
                  {d.contraindicated.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: '#f87171', padding: '2px 0' }}>✗ {f}</div>
                  ))}
                </div>
              )}
              {d.founder_variants?.length > 0 && (
                <div>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#a78bfa', marginBottom: 4 }}>Founder Variants</div>
                  {d.founder_variants.map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: '#a78bfa', padding: '2px 0' }}>◆ {f}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ── MAIN PAGE ────────────────────────────────────────────────────────────── */
export default function HereditaryOrganicAcidemiaAtlasPage() {
  const [tab, setTab]       = useState('Overview');
  const [overview, setOv]   = useState(null);
  const [breakdown, setBk]  = useState(null);
  const [defs, setDefs]     = useState(null);
  const [error, setError]   = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-organic-acidemia-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOv(ov); setBk(bk); setDefs(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'Inter,sans-serif' }}>
      <div style={{ maxWidth: 1280, margin: '0 auto', padding: '1.5rem' }}>

        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
            Hereditary Metabolic Disease Atlas Series · Inborn Errors of Metabolism
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', margin: 0, lineHeight: 1.3 }}>
            🧬 Hereditary-Organic-Acidemia-Atlas
          </h1>
          <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
            Complete 8-Gene Hereditary Organic Acidemia Atlas · 320 Patients (8×40, Seeds 1838–1845) · All AR
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
            {Object.entries(GENE_COLORS).map(([g, c]) => (
              <span key={g} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 12, background: c + '22', color: c, border: `1px solid ${c}44` }}>
                {g}
              </span>
            ))}
          </div>
        </div>

        {error && <ErrorBox msg={error} />}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', borderBottom: '1px solid #1e293b', paddingBottom: '0.5rem' }}>
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: '0.5rem 1rem', background: tab === t ? '#6366f1' : 'transparent',
                color: tab === t ? '#fff' : '#94a3b8', border: 'none', borderRadius: 6,
                cursor: 'pointer', fontSize: 13, fontWeight: tab === t ? 600 : 400,
              }}
            >{t}</button>
          ))}
        </div>

        {/* Tab content */}
        {tab === 'Overview'      && <OverviewTab     data={overview}   />}
        {tab === 'Gene Table'    && <GeneTableTab    data={breakdown}  />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'   && <DefinitionsTab  data={defs}       />}
      </div>
    </div>
  );
}
