'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  PEX1:    '#0e7490',   // cyan-700 — ZSD1 most common PBD ghost peroxisomes
  PEX6:    '#0369a1',   // blue-700 — ZSD4 attenuated adult onset
  ABCD1:   '#b91c1c',   // red-700 — X-ALD CALD HSCT curative Loes<9
  PHYH:    '#15803d',   // green-700 — Refsum dietary restriction curative
  PEX7:    '#92400e',   // amber-800 — RCDP1 stippled epiphyses
  HSD17B4: '#581c87',   // purple-900 — DBP most severe peroxisomal beta-ox
  ACOX1:   '#c2410c',   // orange-700 — LOF neonatal ALD / GOF inflammatory AID
  AGPS:    '#1e3a8a',   // blue-900 — RCDP3 ether lipid step2 phytanic normal
};

const GENE_DISEASE = {
  PEX1:    'AR ZSD1-Zellweger-Spectrum-Most-Common-PBD-60-70pct-1283aa-7q21.2 — VLCFA-Elevated-Plasmalogens-Reduced-Pipecolic-Elevated — Ghost-Peroxisomes-IF-PATHOGNOMONIC — DHA-Supplementation-MANDATORY — p.Gly843Asp-40pct-European-Attenuated — Adrenal-Insufficiency-Screen-ACTH',
  PEX6:    'AR ZSD4-Second-Most-Common-PBD-980aa-6p21.1 — Biochemistry-Identical-PEX1-VLCFA-Plasmalogens-Pipecolic — Attenuated-ZSD-Adult-Onset-RP-SNHL-Ataxia-Common-Misdiagnosed-Usher — p.Arg860Trp-Most-Common-Attenuated — VLCFA-Borderline-Attenuated',
  ABCD1:   'XLR X-ALD-AMN-745aa-Xq28 — C26:0/C22:0-Ratio-DIAGNOSTIC-Primary-Screen — HSCT-Curative-Loes-Score-<9-Asymptomatic-Only — Lorenzos-Oil-Does-NOT-Halt-Neurological-Progression — Adrenal-Insufficiency-80pct-Hydrocortisone-Mandatory — Lenti-D-Gene-Therapy-FDA2022',
  PHYH:    'AR Refsum-Disease-338aa-10p13 — Phytanic-Acid-Markedly-Elevated-VLCFA-NORMAL — Dietary-Restriction-Curative-No-Green-Veg-No-Dairy-No-Ruminant-Fat — NEVER-Fast-Mobilises-Adipose-Phytanic-Stores — Plasmapheresis-Crisis — Cardiac-Arrhythmia-Sudden-Death-Annual-ECG',
  PEX7:    'AR RCDP1-Most-Common-RCDP-90pct-323aa-6q23.3 — Stippled-Epiphyses-X-ray-PATHOGNOMONIC — VLCFA-Normal-Plasmalogens-Very-Low-Phytanic-Elevated — PTS2-Receptor-Imports-AGPS-PHYH-HACL1-Only — Congenital-Cataracts-Bilateral — Rhizomelic-Proximal-Limb-Shortening',
  HSD17B4: 'AR DBP-Most-Severe-Peroxisomal-Beta-Ox-Disorder-736aa-5q23.1 — Zellweger-Like-Phenotype-Peroxisomes-INTACT-Not-Ghost — VLCFA-Elevated-Pristanic-Acid-Elevated-Bile-Acids — Neonatal-Hypotonia-Seizures-Death-First-Year — No-Disease-Modifying-Therapy-Supportive-Only',
  ACOX1:   'AR-AD VLCFA-Beta-Ox-Step1-700aa-17q25.1 — LOF-Recessive-Pseudo-Neonatal-ALD-Leukodystrophy — GOF-Dominant-ACOX1-Inflammatory-Disease-AID-Steroids — Pristanic-NORMAL-Unlike-DBP — Plasmalogens-NORMAL-Unlike-ZSD — GOF-vs-LOF-Critical-Different-Treatment',
  AGPS:    'AR RCDP3-Ether-Lipid-Step2-728aa-2q31.2 — Plasmalogens-Markedly-Reduced-Phytanic-NORMAL-Unlike-RCDP1 — Stippled-Epiphyses-Cataracts-RCDP-Phenotype — VLCFA-Normal — Rarer-5pct-RCDP — PEMBA-Plasmalogen-Precursor-Trial',
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
          Hereditary-Peroxisomal-Atlas — Complete 8-Gene Hereditary Peroxisomal Disorders Atlas
        </div>
        <div style={{ fontSize: 11, color: '#64748b', lineHeight: 1.6, fontFamily: 'monospace' }}>
          {ov.subtitle}
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '1.5rem' }}>
        <KPI label="Total Patients" value={ov.total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={ov.total_genes} color="#0ea5e9" />
        <KPI label="Severe Cases" value={`${ov.severe_n} (${ov.severe_pct}%)`} color="#ef4444" />
        <KPI label="VLCFA Elevated" value={`${ov.vlcfa_elevated_n} (${ov.vlcfa_elevated_pct}%)`} color="#f59e0b" />
        <KPI label="Plasmalogens ↓" value={`${ov.plasmalogens_reduced_n} (${ov.plasmalogens_reduced_pct}%)`} color="#8b5cf6" />
        <KPI label="Retinal (RP)" value={`${ov.retinal_n} (${ov.retinal_pct}%)`} color="#06b6d4" />
        <KPI label="Adrenal Insuffic." value={`${ov.adrenal_n} (${ov.adrenal_pct}%)`} color="#dc2626" />
        <KPI label="HSCT Eligible" value={`${ov.hsct_eligible_n} (${ov.hsct_eligible_pct}%)`} color="#10b981" />
        <KPI label="Phytanic ↑" value={`${ov.phytanic_elevated_n} (${ov.phytanic_elevated_pct}%)`} color="#84cc16" />
        <KPI label="Cataracts" value={`${ov.cataracts_n} (${ov.cataracts_pct}%)`} color="#a78bfa" />
        <KPI label="Seeds" value={ov.seed_range} color="#475569" />
      </div>

      <div style={{ background: '#1e293b', borderRadius: 12, padding: '1rem', marginBottom: '1.2rem' }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#cbd5e1', marginBottom: 8 }}>Gene Loci & Sizes</div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {ov.genes && ov.genes.map(g => (
            <div key={g} style={{
              background: '#0f172a', borderRadius: 6, padding: '6px 10px',
              borderLeft: `3px solid ${GENE_COLORS[g] || '#6366f1'}`, fontSize: 11,
            }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[g] || '#a5b4fc' }}>{g}</span>
              <span style={{ color: '#94a3b8', marginLeft: 6 }}>{ov.gene_loci?.[g]}</span>
              <span style={{ color: '#64748b', marginLeft: 6 }}>{ov.gene_sizes?.[g]}</span>
            </div>
          ))}
        </div>
      </div>

      <div style={{ background: '#0f172a', borderRadius: 10, padding: '1rem', borderLeft: '4px solid #f59e0b' }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: '#fbbf24', marginBottom: 6 }}>
          ⚡ Critical Clinical Pearls
        </div>
        <ul style={{ fontSize: 11, color: '#94a3b8', margin: 0, paddingLeft: 16, lineHeight: 1.8 }}>
          <li><b style={{ color: '#38bdf8' }}>PEX1/PEX6</b>: DHA supplementation MANDATORY; ghost peroxisomes on fibroblast IF; adrenal screen ACTH</li>
          <li><b style={{ color: '#b91c1c' }}>ABCD1</b>: HSCT curative ONLY if Loes {'<'}9 + asymptomatic; Lorenzo's Oil does NOT halt progression; adrenal 80%</li>
          <li><b style={{ color: '#15803d' }}>PHYH (Refsum)</b>: dietary restriction curative; NEVER fast (mobilises adipose phytanic); plasmapheresis crisis</li>
          <li><b style={{ color: '#92400e' }}>PEX7 (RCDP1)</b>: stippled epiphyses X-ray PATHOGNOMONIC; VLCFA normal (not ZSD); plasmalogens markedly ↓</li>
          <li><b style={{ color: '#581c87' }}>HSD17B4 (DBP)</b>: Zellweger-like but peroxisomes INTACT; pristanic acid elevated (unlike ACOX1)</li>
          <li><b style={{ color: '#c2410c' }}>ACOX1</b>: GOF (dominant) = inflammatory AID → steroids; LOF (recessive) = neonatal ALD → supportive only</li>
          <li><b style={{ color: '#1e3a8a' }}>AGPS (RCDP3)</b>: phytanic acid NORMAL (unlike RCDP1/PEX7); plasmalogens ↓; same stippled epiphyses phenotype</li>
        </ul>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data.breakdown_by_gene || {});
  return (
    <div style={{ padding: '1rem 0' }}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr style={{ background: '#1e293b', color: '#94a3b8' }}>
              {['Gene','Locus','Size','Inheritance','Patients','Severe%','VLCFA↑','Plasmalogens↓','Phytanic↑','Retinal','Adrenal','HSCT','Cataracts','Treatment'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map((g, idx) => (
              <tr key={g.gene} style={{ background: idx % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', whiteSpace: 'nowrap' }}>{g.gene}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                <td style={{ padding: '7px 10px', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                <td style={{ padding: '7px 10px', color: '#64748b', fontSize: 10, maxWidth: 120 }}>{g.inheritance}</td>
                <td style={{ padding: '7px 10px', color: '#e2e8f0', textAlign: 'center' }}>{g.n_patients}</td>
                <td style={{ padding: '7px 10px', color: g.severe_pct > 60 ? '#f87171' : '#fbbf24', textAlign: 'center' }}>{g.severe_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#f59e0b' }}>{g.vlcfa_elevated_n}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#8b5cf6' }}>{g.plasmalogens_reduced_n}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#84cc16' }}>{g.phytanic_elevated_n}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#06b6d4' }}>{g.retinal_n}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#dc2626' }}>{g.adrenal_n}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#10b981' }}>{g.hsct_eligible_n}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: '#a78bfa' }}>{g.cataracts_n}</td>
                <td style={{ padding: '7px 10px', color: '#64748b', fontSize: 10, maxWidth: 180 }}>{g.treatment}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data.breakdown_by_gene || {});
  return (
    <div style={{ padding: '1rem 0' }}>
      {genes.map(g => (
        <div key={g.gene} style={{
          background: '#1e293b', borderRadius: 12, padding: '1rem 1.2rem',
          marginBottom: '1rem', borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
            <span style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
            <span style={{ fontSize: 11, color: '#94a3b8' }}>{g.locus} · {g.protein_size} · {g.inheritance}</span>
            <span style={{ fontSize: 11, color: '#64748b' }}>seed {g.seed}</span>
          </div>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6, fontFamily: 'monospace', lineHeight: 1.5 }}>
            {GENE_DISEASE[g.gene]}
          </div>
          <div style={{ background: '#0f172a', borderRadius: 8, padding: '8px 12px', marginBottom: 6 }}>
            <div style={{ fontSize: 11, color: '#fbbf24', fontWeight: 600 }}>Biomarker: </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>{g.key_biomarker}</div>
          </div>
          <div style={{ background: '#0f172a', borderRadius: 8, padding: '8px 12px', marginBottom: 6 }}>
            <div style={{ fontSize: 11, color: '#f87171', fontWeight: 600 }}>Pathognomonic: </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>{g.pathognomonic}</div>
          </div>
          <div style={{ background: '#0f172a', borderRadius: 8, padding: '8px 12px', marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: '#34d399', fontWeight: 600 }}>Treatment: </div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>{g.treatment}</div>
          </div>
          {g.critical_flags && g.critical_flags.length > 0 && (
            <div style={{ background: '#450a0a', borderRadius: 8, padding: '8px 12px' }}>
              <div style={{ fontSize: 11, color: '#fca5a5', fontWeight: 600, marginBottom: 4 }}>⚠ Critical Flags:</div>
              <ul style={{ margin: 0, paddingLeft: 14 }}>
                {g.critical_flags.map((f, i) => (
                  <li key={i} style={{ fontSize: 10, color: '#fca5a5', lineHeight: 1.7, fontFamily: 'monospace' }}>{f}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const [open, setOpen] = useState({});
  const defs = data.definitions || {};
  return (
    <div style={{ padding: '1rem 0' }}>
      <div style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
        {data.total_definition_entries} definitions · {data.total_genes} genes · seeds {data.seed_range}
      </div>
      {Object.entries(defs).map(([title, body]) => (
        <div key={title} style={{ background: '#1e293b', borderRadius: 10, marginBottom: 8 }}>
          <button
            onClick={() => setOpen(s => ({ ...s, [title]: !s[title] }))}
            style={{
              width: '100%', textAlign: 'left', background: 'none', border: 'none',
              padding: '10px 14px', cursor: 'pointer',
              color: open[title] ? '#38bdf8' : '#cbd5e1', fontWeight: 600, fontSize: 12,
            }}
          >
            {open[title] ? '▼' : '▶'} {title}
          </button>
          {open[title] && (
            <div style={{ padding: '0 14px 12px', fontSize: 11, color: '#94a3b8', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
              {body}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryPeroxisomalAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-peroxisomal-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-peroxisomal-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-peroxisomal-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bd, df]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(df);
    }).catch(e => setErr(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'Inter, system-ui, sans-serif', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#38bdf8', margin: 0, marginBottom: 4 }}>
            🧬 Hereditary-Peroxisomal-Atlas
          </h1>
          <div style={{ fontSize: 12, color: '#64748b' }}>
            Complete 8-Gene Hereditary Peroxisomal Disorders Reference — PEX1 · PEX6 · ABCD1 · PHYH · PEX7 · HSD17B4 · ACOX1 · AGPS — 320 patients (seeds 1854–1861)
          </div>
        </div>

        {err && <ErrorBox msg={err} />}

        <div style={{ display: 'flex', gap: 4, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t ? '#0ea5e9' : '#1e293b',
              color: tab === t ? '#fff' : '#94a3b8',
              fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>

        {tab === 'Overview' && <OverviewTab data={overview} />}
        {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
        {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
