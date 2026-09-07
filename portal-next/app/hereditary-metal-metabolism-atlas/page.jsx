'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ATP7B:    '#b45309',   // amber — Wilson KF rings ceruloplasmin low D-penicillamine
  ATP7A:    '#7c3aed',   // purple — Menkes kinky hair XLR Cu histidine
  HFE:      '#dc2626',   // red — HH1 C282Y most common phlebotomy
  HAMP:     '#065f46',   // dark green — juvenile HH cardiac LEADING KILLER
  SLC40A1:  '#1d4ed8',   // blue — ferroportin AD type4A-4B distinct
  CP:       '#701a75',   // fuchsia/dark — aceruloplasminemia DM+retina+neuro triad
  SLC30A10: '#0f766e',   // teal — hypermanganesemia polycythemia T1 high BG
  TMPRSS6:  '#1e1b4b',   // dark indigo — IRIDA oral iron COMPLETELY INEFFECTIVE
};

const GENE_DISEASE = {
  ATP7B:    'AR Wilson-Disease — P-type-Cu-ATPase-1465aa — 13q14.3 — KF-Rings-Slit-Lamp-PATHOGNOMONIC — Ceruloplasmin-VERY-LOW-95pct — D-Penicillamine-CI-Pregnancy — Trientine-Zinc — Liver-Tx-Hepatic-Only — p.His1069Gln-40pct-European',
  ATP7A:    'XLR Menkes-Disease — P-type-Cu-ATPase-1500aa — Xq21.1 — Pili-Torti-Kinky-Hair-PATHOGNOMONIC — Cu-Histidine-SQ-Within-4-6-Weeks-Birth — Low-Cu-Low-Ceruloplasmin-BOTH — Occipital-Horn-Syndrome-Milder',
  HFE:      'AR Hereditary-Hemochromatosis-Type-1 — HLA-MHC-Class-I-348aa — 6p22.2 — C282Y-Most-Common-AR-Disorder-Europeans — 2nd-3rd-MCP-Arthropathy-PATHOGNOMONIC — TS->45pct-Screen — Phlebotomy-CURATIVE — Alcohol-VitaminC-CI',
  HAMP:     'AR Juvenile-Hemochromatosis-Type-2B — Hepcidin-84aa — 19q13.12 — Juvenile-Onset-2nd-3rd-Decade — Cardiac-EARLIEST-LEADING-KILLER — Hypogonadism-Primary-Amenorrhoea — Aggressive-Phlebotomy-Urgent',
  SLC40A1:  'AD Ferroportin-Disease-HH-Type-4 — Iron-Exporter-570aa — 2q32.2 — AUTOSOMAL-DOMINANT — Type4A-Macrophage-Iron-LOW-TS — Type4B-Hepcidin-Resistance-HIGH-TS — Distinguish-CRITICAL-Different-Treatment',
  CP:       'AR Aceruloplasminemia — Multicopper-Ferroxidase-1058aa — 3q25.1 — TRIAD-DM+Retinal-Degeneration+Neurodegeneration-PATHOGNOMONIC — Ceruloplasmin-ABSENT-NOT-LOW — Serum-Iron-LOW-Paradox — MRI-T2-Dark-BG',
  SLC30A10: 'AR Hypermanganesemia-with-Dystonia-1 — Mn-Exporter-485aa — 1q41 — Polycythemia-PATHOGNOMONIC-Mn-EPO — MRI-T1-HIGH-Basal-Ganglia-PATHOGNOMONIC — Childhood-Onset-Dystonia — CaEDTA-Chelation',
  TMPRSS6:  'AR IRIDA-Iron-Refractory-IDA — Matriptase-2-811aa — 22q12.3 — Oral-Iron-COMPLETELY-INEFFECTIVE-PATHOGNOMONIC — Hepcidin-Constitutively-HIGH — IV-Iron-Needed-Partial-Response — Childhood-1-3-Years',
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

function Alert({ text, color }) {
  return (
    <div style={{
      background: '#0f172a', border: `1px solid ${color || '#334155'}`,
      borderRadius: 8, padding: '0.65rem 1rem', marginBottom: 8,
      fontSize: 12, color: '#e2e8f0', lineHeight: 1.5,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 6 }}>{data.atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20 }}>{data.subtitle}</p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <KPI label="Total Patients" value={data.total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={s.genes_covered} color="#22c55e" />
        <KPI label="Seeds" value={data.seed_range} color="#f59e0b" />
        <KPI label="AR Genes" value={s.ar_genes} color="#dc2626" />
        <KPI label="AD Genes" value={s.ad_genes} color="#1d4ed8" />
        <KPI label="XLR Genes" value={s.xlr_genes} color="#7c3aed" />
        <KPI label="On Treatment %" value={`${s.on_treatment_pct ?? 0}%`} color="#0f766e" />
        <KPI label="Family Cascade %" value={`${s.family_cascade_pct ?? 0}%`} color="#b45309" />
        <KPI label="Hepatic Disease %" value={`${s.hepatic_disease_pct ?? 0}%`} color="#065f46" />
        <KPI label="Neurological %" value={`${s.neurological_involvement_pct ?? 0}%`} color="#701a75" />
        <KPI label="Mild %" value={`${s.severity_mild_pct ?? 0}%`} color="#22c55e" />
        <KPI label="Moderate %" value={`${s.severity_moderate_pct ?? 0}%`} color="#f59e0b" />
        <KPI label="Severe %" value={`${s.severity_severe_pct ?? 0}%`} color="#dc2626" />
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>🚨 Critical Treatment Alerts</h3>
      {(data.critical_treatment_alerts || []).slice(0, 16).map((a, i) => (
        <Alert key={i} text={a} color={Object.values(GENE_COLORS)[Math.floor(i / 2) % 8]} />
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 10 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 12 }}>
        {(data.genes || []).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
              {g.locus} · {g.protein_size} aa · {g.inheritance}
            </div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 6 }}>
              {GENE_DISEASE[g.gene] || g.disorder}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
              Biomarker: {g.key_biomarker?.substring(0, 90)} · n={g.n_patients}
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>Rx: {g.treatment?.substring(0, 90)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = data.genes ? Object.values(data.genes) : [];
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8-Gene Hereditary Metal & Trace Element Metabolism Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Locus', 'AA', 'Inheritance', 'Key Biomarker', 'Pathognomonic', 'Treatment', 'N'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.protein_size}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 160, wordBreak: 'break-word' }}>{g.key_biomarker}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 200, wordBreak: 'break-word' }}>{g.pathognomonic}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200, wordBreak: 'break-word' }}>{g.treatment}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(0);
  if (!data) return <Loading />;
  const genes = data.genes ? Object.values(data.genes) : [];
  const g = genes[selected];
  if (!g) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '200px 1fr', gap: 20 }}>
      <div>
        {genes.map((gene, i) => (
          <button key={gene.gene} onClick={() => setSelected(i)} style={{
            display: 'block', width: '100%', textAlign: 'left',
            padding: '10px 14px', marginBottom: 6, borderRadius: 8,
            background: selected === i ? GENE_COLORS[gene.gene] : '#1e293b',
            color: selected === i ? '#fff' : '#94a3b8',
            border: 'none', cursor: 'pointer', fontSize: 14, fontWeight: 600,
          }}>
            {gene.gene}
            <div style={{ fontSize: 10, fontWeight: 400, marginTop: 2 }}>{gene.locus}</div>
          </button>
        ))}
      </div>
      <div>
        <h3 style={{ color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 4 }}>{g.gene}</h3>
        <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>
          {g.inheritance} · {g.locus} · {g.protein_size} aa · Age of onset: {g.age_of_onset}
        </div>

        <h4 style={{ color: '#e2e8f0', marginBottom: 6 }}>🚨 Critical Flags</h4>
        {(g.critical_flags || []).map((a, i) => (
          <Alert key={i} text={a} color={GENE_COLORS[g.gene]} />
        ))}

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Key Biomarker</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '0.75rem 1rem',
          fontSize: 13, color: '#fbbf24', fontWeight: 600,
        }}>
          {g.key_biomarker}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Pathognomonic</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '0.75rem 1rem',
          fontSize: 13, color: '#f87171',
        }}>
          {g.pathognomonic}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Treatment</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '0.75rem 1rem',
          fontSize: 13, color: '#6ee7b7',
        }}>
          {g.treatment}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Clinical Description</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '1rem',
          fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          maxHeight: 360, overflowY: 'auto',
        }}>
          {g.protein_description}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Cohort ({g.n_patients} patients)</h4>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#22c55e', fontWeight: 700 }}>{g.severity?.mild ?? 0}</span>
            <span style={{ color: '#64748b' }}> mild</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#f59e0b', fontWeight: 700 }}>{g.severity?.moderate ?? 0}</span>
            <span style={{ color: '#64748b' }}> moderate</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#dc2626', fontWeight: 700 }}>{g.severity?.severe ?? 0}</span>
            <span style={{ color: '#64748b' }}> severe</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#6366f1', fontWeight: 700 }}>{g.on_treatment}</span>
            <span style={{ color: '#64748b' }}> on treatment ({g.on_treatment_pct}%)</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#065f46', fontWeight: 700 }}>{g.hepatic_disease ?? 0}</span>
            <span style={{ color: '#64748b' }}> hepatic disease</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#701a75', fontWeight: 700 }}>{g.neurological_involvement ?? 0}</span>
            <span style={{ color: '#64748b' }}> neurological</span>
          </div>
          <div style={{ background: '#1e293b', borderRadius: 8, padding: '8px 14px', fontSize: 12 }}>
            <span style={{ color: '#7c3aed', fontWeight: 700 }}>{g.family_cascade}</span>
            <span style={{ color: '#64748b' }}> family cascade</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.definitions || {};
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 4 }}>Definitions & Clinical Concepts</h3>
      <p style={{ color: '#64748b', fontSize: 12, marginBottom: 20 }}>
        {data.total_definition_entries} entries · {data.atlas}
      </p>
      {Object.entries(defs).map(([title, body]) => (
        <div key={title} style={{ marginBottom: 20 }}>
          <h4 style={{ color: '#a5b4fc', marginBottom: 6 }}>{title}</h4>
          <div style={{
            background: '#1e293b', borderRadius: 8, padding: '1rem',
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          }}>
            {body}
          </div>
        </div>
      ))}
    </div>
  );
}

export default function HeredMetalMetabolismAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-metal-metabolism-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-metal-metabolism-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-metal-metabolism-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary-Metal-Metabolism-Atlas · 320 patients · Seeds 1838–1845
        </h1>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          ATP7B (Wilson-KF-Rings-PATHOGNOMONIC-D-Penicillamine-Trientine-Zinc-Liver-Tx) ·
          ATP7A (Menkes-Kinky-Hair-XLR-Cu-Histidine-SQ-4-6-Weeks) ·
          HFE (HH1-C282Y-Most-Common-AR-European-Phlebotomy-Curative-MCP-Arthropathy) ·
          HAMP (JHH-Hepcidin-84aa-Juvenile-Cardiac-EARLIEST-KILLER-Hypogonadism) ·
          SLC40A1 (Ferroportin-AD-Type4A-Macrophage-Type4B-Hepcidin-Resistance) ·
          CP (Aceruloplasminemia-DM+Retina+Neuro-TRIAD-Ceruloplasmin-ABSENT) ·
          SLC30A10 (Hypermanganesemia-Polycythemia-PATHOGNOMONIC-T1-High-BG-EDTA) ·
          TMPRSS6 (IRIDA-Oral-Iron-COMPLETELY-INEFFECTIVE-PATHOGNOMONIC-IV-Iron-Partial)
        </p>
      </div>

      {error && <ErrorBox msg={error} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 8,
            background: tab === i ? '#b45309' : '#1e293b',
            color: tab === i ? '#fff' : '#94a3b8',
            border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      <div>
        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <GeneTableTab data={breakdown} />}
        {tab === 2 && <ClinicalAtlasTab data={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
