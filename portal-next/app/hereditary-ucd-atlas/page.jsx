'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  OTC:      '#dc2626',  // red — most common UCD, X-linked, neonatal crisis
  ASS1:     '#1d4ed8',  // blue — CTLN1, citrulline>1000, NBS detectable
  ASL:      '#0f766e',  // teal — trichorrhexis nodosa, ammonia-independent neurotox
  CPS1:     '#b45309',  // amber — absent citrulline, NCG trial mandatory
  ARG1:     '#7c3aed',  // purple — spastic diplegia not hyperammonemia
  NAGS:     '#065f46',  // dark green — ONLY UCD antidote NCG carglumic acid
  SLC25A15: '#c2410c',  // deep orange — HHH, homocitrullinuria pathognomonic
  SLC25A13: '#1e1b4b',  // dark indigo — citrin NICCD/CTLN2, carbohydrate HARMFUL
};

const GENE_DISEASE = {
  OTC:      'XLR OTC Deficiency — Most-Common-UCD-~50pct — Ornithine-Transcarbamylase-354aa — Xp21.1 — Valproate-ABSOLUTE-CI — Orotic-Acid-Elevated-KEY-DDx — Liver-Transplant-Curative',
  ASS1:     'AR Citrullinemia-Type-I CTLN1 — Argininosuccinate-Synthase-412aa — 9q34.11 — Citrulline>1000µmol/L-PATHOGNOMONIC — Arginine-ESSENTIAL-Supplement — NBS-Detectable-DBS',
  ASL:      'AR Argininosuccinic-Aciduria — Argininosuccinate-Lyase-464aa — 7cen-q11.2 — Argininosuccinate-Urine-PATHOGNOMONIC — Trichorrhexis-Nodosa-PATHOGNOMONIC — NH3-Independent-Neurotox',
  CPS1:     'AR CPS1-Deficiency — Carbamoyl-Phosphate-Synthase-1-1500aa — 2q35 — Absent-Citrulline-Normal-Orotic-KEY-DDx — NCG-Trial-MANDATORY-Exclude-NAGS — Citrulline-Supplement-Bypasses-Block',
  ARG1:     'AR Arginemia — Arginase-1-322aa — 6q23.2 — Spastic-Diplegia-NOT-Hyperammonemia — Arginine>400µmol/L-PATHOGNOMONIC — Ammonia-MILDLY-Elevated — RESTRICT-Arginine-NOT-Supplement',
  NAGS:     'AR NAGS-Deficiency — N-Acetylglutamate-Synthase-534aa — 17q21.31 — ONLY-UCD-with-Antidote-NCG — Carglumic-Acid-Carbaglu-DRAMATIC-Response-24-48h — Biochemically-Identical-to-CPS1',
  SLC25A15: 'AR HHH-Syndrome — Ornithine-Carrier-ORC1-301aa — 13q14.11 — Homocitrullinuria-PATHOGNOMONIC — Hyperornithinemia-Hyperammonemia-Homocitrullinuria-Triad — French-Canadian-Founder-pF188del',
  SLC25A13: 'AR Citrin-Deficiency NICCD-CTLN2 — Aspartate-Glutamate-Carrier-675aa — 7q21.3 — Carbohydrate-HARMFUL-UNIQUE — High-Protein-Fat-BENEFICIAL — Sweet-Food-Aversion-PATHOGNOMONIC — East-Asian-Japan-1:17000',
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
        <KPI label="X-Linked Genes" value={s.x_linked_genes} color="#7c3aed" />
        <KPI label="Patients / Gene" value={s.patients_per_gene} color="#0f766e" />
        <KPI label="OTC Orotic↑ %" value={`${s.otc_orotic_acid_elevated_pct}%`} color="#dc2626" />
        <KPI label="NAGS NCG Response %" value={`${s.nags_ncg_dramatic_response_pct}%`} color="#065f46" />
        <KPI label="ARG1 Spastic Diplegia %" value={`${s.arg1_spastic_diplegia_pct}%`} color="#7c3aed" />
        <KPI label="ASL Trichorrhexis %" value={`${s.asl_trichorrhexis_nodosa_pct}%`} color="#0f766e" />
        <KPI label="HHH Homocitrullinuria %" value={`${s.hhh_homocitrullinuria_pct}%`} color="#c2410c" />
        <KPI label="Citrin CHO Aversion %" value={`${s.citrin_carbohydrate_aversion_ctln2_pct}%`} color="#1e1b4b" />
      </div>

      <h3 style={{ color: '#e2e8f0', marginBottom: 10 }}>🚨 Critical Clinical Alerts</h3>
      {(data.top_alerts || []).map((a, i) => (
        <Alert key={i} text={a} color={Object.values(GENE_COLORS)[i % 8]} />
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 24, marginBottom: 10 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
        {(data.genes || []).map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{g.locus} · {g.aa} aa · {g.inheritance}</div>
            <div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 6 }}>
              {GENE_DISEASE[g.gene]?.split(' — ')[0]}
            </div>
            <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>n={g.n_patients} patients</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>8-Gene Hereditary Urea Cycle Disorder Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Protein', 'Locus', 'AA', 'kDa', 'OMIM Gene', 'Inheritance', 'Gene Class', 'Specific Rx'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#cbd5e1', maxWidth: 280, wordBreak: 'break-word' }}>{g.protein}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.kDa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.omim_gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.inheritance}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 200 }}>{g.gene_class}</td>
                <td style={{ padding: '8px 10px', fontWeight: 600,
                  color: g.gene === 'NAGS' ? '#22c55e' : g.gene === 'OTC' ? '#dc2626' : '#94a3b8' }}>
                  {g.gene === 'NAGS' ? 'NCG (carglumic acid)' : g.gene === 'OTC' ? 'No VPA' : 'Protein restriction'}
                </td>
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
  const g = data[selected];
  if (!g) return null;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 20 }}>
      <div>
        {data.map((gene, i) => (
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
        <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>{g.inheritance} · {g.locus} · {g.aa} aa · {g.kDa} kDa</div>

        <h4 style={{ color: '#e2e8f0', marginBottom: 6 }}>🚨 Key Alerts</h4>
        {(g.key_alerts || []).map((a, i) => (
          <Alert key={i} text={a} color={GENE_COLORS[g.gene]} />
        ))}

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Clinical Description</h4>
        <div style={{
          background: '#1e293b', borderRadius: 8, padding: '1rem',
          fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap',
          maxHeight: 400, overflowY: 'auto',
        }}>
          {g.alias}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Etiology / Presentation Distribution</h4>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {Object.entries(g.etiologies || {}).map(([k, v]) => (
            <div key={k} style={{
              background: '#1e293b', borderRadius: 8, padding: '8px 12px',
              fontSize: 12, color: '#94a3b8',
            }}>
              <span style={{ fontWeight: 600, color: GENE_COLORS[g.gene] }}>{v}%</span> {k.replace(/_/g, ' ')}
            </div>
          ))}
        </div>

        <h4 style={{ color: '#e2e8f0', marginTop: 16, marginBottom: 6 }}>Sample Patients (first 10)</h4>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                <th style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>ID</th>
                {Object.keys(g.sample_patients?.[0] || {}).filter(k => k !== 'patient_id').slice(0, 6).map(k => (
                  <th key={k} style={{ padding: '5px 8px', textAlign: 'left', color: '#64748b' }}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(g.sample_patients || []).map((p, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '5px 8px', color: GENE_COLORS[g.gene] }}>{p.patient_id}</td>
                  {Object.entries(p).filter(([k]) => k !== 'patient_id').slice(0, 6).map(([k, v]) => (
                    <td key={k} style={{ padding: '5px 8px', color: '#94a3b8' }}>
                      {typeof v === 'boolean' ? (v ? '✓' : '✗') : String(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Definitions & Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body]) => (
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

export default function HeredUCDAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-ucd-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-ucd-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-ucd-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov); setBreakdown(br); setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', padding: '1.5rem' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, color: '#f1f5f9', marginBottom: 4 }}>
          🧬 Hereditary UCD Atlas — Complete 8-Gene Urea Cycle Disorder Reference
        </h1>
        <p style={{ color: '#64748b', fontSize: 13 }}>
          OTC (XLR-Most-Common-VPA-CI) · ASS1 (CTLN1-Citrulline>1000) · ASL (Trichorrhexis-NH3-Indep-Neurotox) ·
          CPS1 (Absent-Citrulline-NCG-Trial) · ARG1 (Spastic-Diplegia) · NAGS (NCG-Antidote-ONLY-UCD) ·
          SLC25A15 (HHH-Homocitrullinuria) · SLC25A13 (Citrin-CHO-Harmful) —
          320 Patients · Seeds 1798–1805
        </p>
      </div>

      {error && <ErrorBox msg={error} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 8,
            background: tab === i ? '#dc2626' : '#1e293b',
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
