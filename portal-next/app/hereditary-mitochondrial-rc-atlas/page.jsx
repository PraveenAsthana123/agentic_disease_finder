'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  POLG:   '#b45309',   // amber — Alpers/SANDO VPA ABSOLUTE CI
  TWNK:   '#7c3aed',   // purple — PEO1 ptosis ophthalmoplegia AD
  SURF1:  '#dc2626',   // red — Leigh CIV most common European
  BCS1L:  '#065f46',   // dark green — GRACILE neonatal lethal Bjornstad
  SCO2:   '#1d4ed8',   // blue — fatal HCM CIV copper
  PDHA1:  '#701a75',   // fuchsia — KD therapeutic X-linked PDH
  ACAD9:  '#0f766e',   // teal — CI assembly riboflavin responsive 50%
  NDUFS1: '#1e1b4b',   // dark indigo — CI subunit Leigh leukoencephalopathy
};

const GENE_DISEASE = {
  POLG:   'AR Alpers-Huttenlocher/SANDO/PEO3 — mtDNA-Pol-Gamma-1240aa — 15q25.1 — VPA-ABSOLUTE-CI-Fatal-Hepatic — Refractory-Seizures-Hepatopathy-Exclude-POLG — p.Ala467Thr-Most-Common-European — Nucleoside-Analogues-CI',
  TWNK:   'AD/AR Twinkle-Helicase-684aa — 10q24.31 — PEO1-Ptosis-Ophthalmoplegia-Multiple-mtDNA-Deletions-AD — IOSCA-Infantile-Spinocerebellar-Ataxia-AR-Finnish — 50pct-Offspring-Risk-AD — RRF-Multiple-Deletions-Muscle',
  SURF1:  'AR Leigh-Syndrome-CIV-Assembly-309aa — 9q34.2 — Most-Common-Nuclear-Leigh-Europe — COX-Deficient-Fibres-COX-SDH-Staining — RRF-ABSENT-Assembly-Factor — Bilateral-BG-Brainstem-T2-Bright — p.845delCT-European',
  BCS1L:  'AR GRACILE-Neonatal-Lethal/Bjornstad-CIII-Assembly-419aa — 2q35 — GRACILE-Growth-Restrict+Fanconi+Cholestasis+Siderosis — p.Ser78Gly-Finno-Ugric-Founder — BN-PAGE-CIII-PreComplex-Accumulates — Bjornstad-Pili-Torti-SNHL',
  SCO2:   'AR Fatal-Infantile-HCM-CIV-Copper-266aa — 22q13.33 — HCM-EARLIEST-Lactic-Acidosis-Hypotonia-TRIAD — Copper-Supplementation-Emerging — p.Glu140Lys-Most-Common — Serum-Cu-Normal-Intramito-Delivery-Defect',
  PDHA1:  'XL PDH-E1-Alpha-390aa — Xp22.12 — Ketogenic-Diet-THERAPEUTIC-Unique-Bypasses-Pyruvate — Thiamine-B1-Trial-MANDATORY — L:P-Ratio-NORMAL-Both-Rise — Glucose-Infusions-WORSEN-Use-Lipids — Leigh-Syndrome-Corpus-Callosum-Agenesis',
  ACAD9:  'AR CI-Assembly-Factor-621aa — 3q21.3 — Riboflavin-B2-RESPONSIVE-50pct-Critical — Exercise-Intolerance+Lactic-Acidosis+Cardiomyopathy-TRIAD — BN-PAGE-CI-Absent-400kDa-Intermediate — Not-FAO-Disorder-Do-Not-Diagnose-VLCAD',
  NDUFS1: 'AR CI-Core-75kDa-Subunit-727aa — 2q33.3 — Most-Common-Nuclear-CI-Subunit-Gene — Leigh-Or-Leukoencephalopathy — BN-PAGE-CI-Absent-No-Intermediate — Riboflavin-NOT-Responsive-Unlike-ACAD9 — CI-Gene-Panel-Mandatory-20plus-Genes',
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

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  return (
    <div style={{ padding: '1rem 0' }}>
      <div style={{ background: '#0f172a', borderRadius: 12, padding: '1.2rem', marginBottom: '1.5rem',
        borderLeft: '5px solid #0ea5e9' }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: '#38bdf8', marginBottom: 6 }}>
          Hereditary-Mitochondrial-RC-Atlas — Complete 8-Gene Nuclear-Encoded OXPHOS Defects
        </div>
        <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6, fontFamily: 'monospace' }}>
          {ov.subtitle}
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients" value={ov.total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={ov.total_genes} color="#0ea5e9" />
        <KPI label="Severe Cases" value={`${ov.severe_n} (${ov.severe_pct}%)`} color="#ef4444" />
        <KPI label="Cardiac Involvement" value={`${ov.cardiac_n} (${ov.cardiac_pct}%)`} color="#f59e0b" />
        <KPI label="Neurological" value={`${ov.neurological_n} (${ov.neurological_pct}%)`} color="#8b5cf6" />
        <KPI label="Leigh MRI" value={ov.leigh_mri_n} color="#dc2626" />
        <KPI label="Riboflavin Resp." value={`${ov.riboflavin_responsive_n} (${ov.riboflavin_responsive_pct}%)`} color="#10b981" />
        <KPI label="KD Therapy (PDHA1)" value={ov.kd_therapy_n} color="#701a75" />
        <KPI label="mtDNA Depletion" value={ov.mtdna_depletion_n} color="#b45309" />
        <KPI label="Seeds" value={ov.seed_range} color="#475569" />
      </div>

      {/* Gene bar */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.2rem' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#cbd5e1', marginBottom: 10 }}>Per-Gene Summary</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {Object.entries(ov.gene_stats || {}).map(([gene, gs]) => (
            <div key={gene} style={{
              background: '#0f172a', borderRadius: 8, padding: '0.6rem 0.9rem',
              borderTop: `3px solid ${GENE_COLORS[gene] || '#6366f1'}`, minWidth: 180,
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc' }}>{gene}</div>
              <div style={{ fontSize: 10, color: '#64748b' }}>{gs.locus} · {gs.protein_size} · {gs.inheritance}</div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>
                Severe: {gs.severe_pct}% · Cardiac: {gs.cardiac_pct}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Contraindications */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.2rem' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#fca5a5', marginBottom: 8 }}>
          Key Drug Contraindications (MITO RC Disease)
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(ov.key_contraindications || []).map((f, i) => (
            <span key={i} style={{
              background: '#450a0a', color: '#fca5a5', borderRadius: 5,
              padding: '3px 8px', fontSize: 10, fontFamily: 'monospace'
            }}>{f}</span>
          ))}
        </div>
      </div>

      {/* Treatment pearls */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem', marginBottom: '1.2rem' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#86efac', marginBottom: 8 }}>
          Critical Treatment Pearls
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(ov.key_treatment_pearls || []).map((f, i) => (
            <span key={i} style={{
              background: '#052e16', color: '#86efac', borderRadius: 5,
              padding: '3px 8px', fontSize: 10, fontFamily: 'monospace'
            }}>{f}</span>
          ))}
        </div>
      </div>

      {/* Diagnostic tools */}
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1rem' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#93c5fd', marginBottom: 8 }}>
          Critical Diagnostic Tools
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(ov.critical_diagnosis_tools || []).map((f, i) => (
            <span key={i} style={{
              background: '#172554', color: '#93c5fd', borderRadius: 5,
              padding: '3px 8px', fontSize: 10, fontFamily: 'monospace'
            }}>{f}</span>
          ))}
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || {};
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
        <thead>
          <tr style={{ background: '#0f172a' }}>
            {['Gene','Locus','Size','Inh.','Onset','Pathognomonic Finding','Treatment','N','Sev%','Card%'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8',
                borderBottom: '1px solid #334155', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(genes).map(([gene, g]) => (
            <tr key={gene} style={{ borderBottom: '1px solid #1e293b' }}>
              <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', whiteSpace:'nowrap' }}>{gene}</td>
              <td style={{ padding: '8px 10px', color: '#cbd5e1', whiteSpace:'nowrap' }}>{g.locus}</td>
              <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace:'nowrap' }}>{g.protein_size}</td>
              <td style={{ padding: '8px 10px', color: '#94a3b8', whiteSpace:'nowrap' }}>{g.inheritance}</td>
              <td style={{ padding: '8px 10px', color: '#94a3b8', fontSize: 10, maxWidth: 120 }}>{g.age_of_onset}</td>
              <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 240, lineHeight: 1.3 }}>{g.pathognomonic}</td>
              <td style={{ padding: '8px 10px', color: '#fbbf24', maxWidth: 240, lineHeight: 1.3, fontSize: 10 }}>{g.treatment}</td>
              <td style={{ padding: '8px 10px', color: '#94a3b8', textAlign:'center' }}>{g.n_patients}</td>
              <td style={{ padding: '8px 10px', color: g.severity?.severe > 20 ? '#ef4444' : '#94a3b8', textAlign:'center' }}>
                {g.severity ? Math.round(g.severity.severe / g.n_patients * 100) : 0}%
              </td>
              <td style={{ padding: '8px 10px', color: '#f59e0b', textAlign:'center' }}>{g.cardiac_pct}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes || {};
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {Object.entries(genes).map(([gene, g]) => (
        <div key={gene} style={{
          background: '#1e293b', borderRadius: 10, padding: '1rem',
          borderLeft: `4px solid ${GENE_COLORS[gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
            <div>
              <span style={{ fontSize: 16, fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc' }}>{gene}</span>
              <span style={{ fontSize: 11, color: '#64748b', marginLeft: 10 }}>
                {g.locus} · {g.protein_size} · {g.inheritance}
              </span>
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              <span style={{ background: '#0f172a', color: '#94a3b8', borderRadius: 4,
                padding: '2px 8px', fontSize: 10 }}>N={g.n_patients}</span>
              <span style={{ background: '#0f172a', color: '#ef4444', borderRadius: 4,
                padding: '2px 8px', fontSize: 10 }}>Severe {g.severity ? Math.round(g.severity.severe/g.n_patients*100) : 0}%</span>
            </div>
          </div>

          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.5 }}>{g.age_of_onset}</div>

          <div style={{ background: '#0f172a', borderRadius: 6, padding: '0.6rem', marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#fbbf24', marginBottom: 4 }}>Pathognomonic</div>
            <div style={{ fontSize: 11, color: '#e2e8f0', lineHeight: 1.4 }}>{g.pathognomonic}</div>
          </div>

          <div style={{ background: '#0f172a', borderRadius: 6, padding: '0.6rem', marginBottom: 8 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#86efac', marginBottom: 4 }}>Treatment</div>
            <div style={{ fontSize: 11, color: '#e2e8f0', lineHeight: 1.4 }}>{g.treatment}</div>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
            {(g.critical_flags || []).map((f, i) => (
              <span key={i} style={{
                background: '#0f172a', color: '#fca5a5', borderRadius: 4,
                padding: '3px 7px', fontSize: 9, fontFamily: 'monospace',
                borderLeft: `2px solid ${GENE_COLORS[gene] || '#6366f1'}`
              }}>{f}</span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || {};
  const [open, setOpen] = useState(null);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {Object.entries(defs).map(([key, val]) => (
        <div key={key} style={{ background: '#1e293b', borderRadius: 8 }}>
          <button
            onClick={() => setOpen(open === key ? null : key)}
            style={{
              width: '100%', textAlign: 'left', padding: '0.75rem 1rem',
              background: 'none', border: 'none', cursor: 'pointer',
              color: '#e2e8f0', fontFamily: 'monospace', fontSize: 12, fontWeight: 600,
            }}
          >
            {open === key ? '▼' : '▶'} {key}
          </button>
          {open === key && (
            <div style={{ padding: '0 1rem 1rem', fontSize: 11, color: '#94a3b8',
              lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {val}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryMitochondrialRCAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-mitochondrial-rc-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-mitochondrial-rc-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-mitochondrial-rc-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'sans-serif' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto', padding: '1.5rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#38bdf8', margin: 0 }}>
            🧬 Hereditary-Mitochondrial-RC-Atlas
          </h1>
          <div style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>
            Complete 8-Gene Nuclear-Encoded Mitochondrial Respiratory Chain Defects Atlas ·
            POLG · TWNK · SURF1 · BCS1L · SCO2 · PDHA1 · ACAD9 · NDUFS1 ·
            320 patients (8×40, seeds 1846–1853)
          </div>
        </div>

        {error && <ErrorBox msg={error} />}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: '1.2rem', borderBottom: '1px solid #1e293b' }}>
          {TABS.map((t, i) => (
            <button
              key={t}
              onClick={() => setTab(i)}
              style={{
                padding: '0.5rem 1rem', background: 'none', border: 'none', cursor: 'pointer',
                color: tab === i ? '#38bdf8' : '#64748b',
                borderBottom: tab === i ? '2px solid #38bdf8' : '2px solid transparent',
                fontWeight: tab === i ? 700 : 400, fontSize: 13,
              }}
            >{t}</button>
          ))}
        </div>

        {/* Gene key */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: '1.2rem' }}>
          {Object.entries(GENE_DISEASE).map(([gene, desc]) => (
            <div key={gene} style={{
              background: '#1e293b', borderRadius: 6, padding: '4px 10px',
              borderLeft: `3px solid ${GENE_COLORS[gene]}`,
              fontSize: 9, maxWidth: 320,
            }}>
              <span style={{ color: GENE_COLORS[gene], fontWeight: 700 }}>{gene} </span>
              <span style={{ color: '#64748b', fontFamily: 'monospace' }}>{desc}</span>
            </div>
          ))}
        </div>

        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <GeneTableTab data={breakdown} />}
        {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
