'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ATP7B:    '#6d4c41',  // brown — Wilson's, copper accumulation
  HFE:      '#bf360c',  // deep rust — hemochromatosis, iron overload
  SERPINA1: '#4a148c',  // deep purple — A1ATD, polymer accumulation
  JAG1:     '#1a237e',  // deep navy — Alagille, bile duct paucity
  ATP8B1:   '#1b5e20',  // deep green — PFIC1/Byler, low GGT
  ABCB11:   '#006064',  // teal — PFIC2/BSEP, low GGT + HCC
  ABCB4:    '#e65100',  // deep orange — PFIC3/MDR3, high GGT
  ALDOB:    '#880e4f',  // deep pink — HFI, fructose avoidance curative
};

const GENE_DISEASE = {
  ATP7B:    'Wilson Disease AR — Copper Accumulation KF Rings PATHOGNOMONIC — D-Penicillamine/Trientine/Zinc — NEVER STOP Treatment',
  HFE:      'Hereditary Hemochromatosis AR — C282Y 95% Northern European — Iron Overload — Phlebotomy CURATIVE Pre-Cirrhotic',
  SERPINA1: 'Alpha-1 Antitrypsin Deficiency AR — PiZZ — Z-Polymer ER Retention — Lower-Lobe Emphysema — Augmentation LUNG ONLY',
  JAG1:     'Alagille Syndrome AD — Bile Duct Paucity — Butterfly Vertebrae PATHOGNOMONIC — ICH 15% — PPAS Cardiac',
  ATP8B1:   'PFIC1 / Byler Disease AR — LOW GGT — FIC1 Flippase — Diarrhoea/Pancreatitis Extrahepatic — Worsens Post-Transplant',
  ABCB11:   'PFIC2 / BSEP Deficiency AR — LOW GGT — HCC Without Cirrhosis — Anti-BSEP Antibodies Post-Transplant — ICP Heterozygotes',
  ABCB4:    'PFIC3 / MDR3 Deficiency AR/AD — HIGH GGT KEY DDx — Phosphatidylcholine Transport — UDCA Partial — LPAC Adult',
  ALDOB:    'Hereditary Fructose Intolerance AR — A149P European 67% — Fructose Avoidance CURATIVE — Hypoglycaemia Pi Sequestration',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /URGENT|ABSOLUTE|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|MANDATORY|CONTRAINDICATED|AVOID|CURATIVE|LIFE-SAVING|IMMEDIATELY|PROHIBITED|PROPHYLAXIS|WORSENS/i.test(text);
  const isWarning = /MONITOR|SCREEN|ANNUAL|REQUIRED|PROTOCOL|CONTINUOUS|LIFELONG|CASCADE|DISTINGUISH|START|ASSAY|PARTIAL|BEFORE\s+SURGERY|CONSIDER|TRIAL|RISING|PARADOX|KEY\s+DDx|SURVEILLANCE/i.test(text);
  const bg = isCI ? '#6d4c41' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{
      background: '#fafafa', borderRadius: 8, padding: '12px 16px',
      borderLeft: `4px solid ${color || '#6d4c41'}`,
    }}>
      <div style={{ fontSize: 12, color: '#888' }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#6d4c41' }}>{value}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;
  return (
    <div>
      <h2 style={{ color: '#6d4c41', marginBottom: 8 }}>Hereditary Liver Disease Atlas — 8-Gene Reference</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        320 patients (8 × 40, seeds 1582–1589) · ATP7B / HFE / SERPINA1 / JAG1 / ATP8B1 / ABCB11 / ABCB4 / ALDOB
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Patients" value={s.total_patients} />
        <StatCard label="Mean Dx Age (y)" value={s.mean_dx_age_years} />
        <StatCard label="Mean Dx Delay (m)" value={s.mean_dx_delay_months} />
        <StatCard label="ATP7B KF Rings %" value={s.atp7b_kf_rings_pct + '%'} color="#6d4c41" />
        <StatCard label="HFE Phlebotomy %" value={s.hfe_phlebotomy_pct + '%'} color="#bf360c" />
        <StatCard label="SERPINA1 Smoking %" value={s.serpina1_smoking_history_pct + '%'} color="#4a148c" />
        <StatCard label="JAG1 Cardiac Lesion %" value={s.jag1_cardiac_lesion_pct + '%'} color="#1a237e" />
        <StatCard label="ATP8B1 Low GGT %" value={s.atp8b1_low_ggt_pct + '%'} color="#1b5e20" />
        <StatCard label="ABCB11 Childhood HCC %" value={s.abcb11_hcc_childhood_pct + '%'} color="#006064" />
        <StatCard label="ABCB4 High GGT %" value={s.abcb4_high_ggt_pct + '%'} color="#e65100" />
        <StatCard label="ALDOB Diet Normalisation %" value={s.aldob_liver_normalisation_on_diet_pct + '%'} color="#880e4f" />
        <StatCard label="Cascade Tested %" value={s.cascade_tested_pct + '%'} />
      </div>

      <h3 style={{ marginBottom: 12 }}>Top Clinical Alerts</h3>
      {top_alerts?.slice(0, 12).map((a, i) => <AlertBadge key={i} text={a} />)}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#efebe9' }}>
              {['Gene', 'Protein (short)', 'Locus', 'Inheritance', 'OMIM Disease', 'Mean Dx Age', 'N'].map(h => (
                <th key={h} style={{ padding: '6px 10px', textAlign: 'left', borderBottom: '2px solid #6d4c41' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes?.map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #f0f0f0' }}>
                <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.protein_short}</td>
                <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 12 }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.inheritance}</td>
                <td style={{ padding: '6px 10px', fontSize: 12 }}>{g.omim_disease}</td>
                <td style={{ padding: '6px 10px' }}>{g.mean_dx_age}</td>
                <td style={{ padding: '6px 10px', fontWeight: 600 }}>{g.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h2 style={{ color: '#6d4c41', marginBottom: 16 }}>Gene-Level Clinical Breakdown</h2>
      {genes.map(g => (
        <div key={g.gene} style={{
          marginBottom: 28, border: '1px solid #d7ccc8', borderRadius: 10,
          overflow: 'hidden',
        }}>
          <div style={{ background: GENE_COLORS[g.gene] || '#6d4c41', color: '#fff', padding: '10px 16px' }}>
            <strong>{g.gene}</strong> — {g.locus} — {g.aa} — {g.inheritance.split(';')[0]} — OMIM {g.omim_disease}
          </div>
          <div style={{ padding: 16 }}>
            <p style={{ fontSize: 13, color: '#555', marginBottom: 12, lineHeight: 1.5 }}>
              {g.alias?.slice(0, 500)}…
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <strong style={{ fontSize: 12, color: '#888' }}>Variant Classes</strong>
                <table style={{ width: '100%', fontSize: 12, marginTop: 4, borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(g.etiologies || {}).map(([k, v]) => (
                      <tr key={k} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: '3px 0', color: '#333' }}>{k}</td>
                        <td style={{ padding: '3px 0', fontWeight: 600, textAlign: 'right', color: GENE_COLORS[g.gene] }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <strong style={{ fontSize: 12, color: '#888' }}>Key Stats</strong>
                <table style={{ width: '100%', fontSize: 12, marginTop: 4, borderCollapse: 'collapse' }}>
                  <tbody>
                    {Object.entries(g.stats || {}).map(([k, v]) => (
                      <tr key={k} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: '3px 0', color: '#555' }}>{k.replace(/_/g, ' ')}</td>
                        <td style={{ padding: '3px 0', fontWeight: 600, textAlign: 'right' }}>{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h2 style={{ color: '#6d4c41', marginBottom: 16 }}>Clinical Atlas — Key Alerts by Gene</h2>
      {genes.map(g => (
        <div key={g.gene} style={{ marginBottom: 24 }}>
          <h3 style={{ color: GENE_COLORS[g.gene] || '#6d4c41', marginBottom: 8, fontSize: 15 }}>
            {g.gene} — {GENE_DISEASE[g.gene] || g.alias?.slice(0, 80)}
          </h3>
          {(g.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
          <div style={{
            background: '#fafafa', borderRadius: 8, padding: '10px 14px', marginTop: 8,
            fontSize: 12, color: '#555',
          }}>
            <strong>Dx Delay Distribution:</strong>{' '}
            {Object.entries(g.dx_delay_distribution || {}).map(([k, v]) => `${k}: ${v} pts`).join(' · ')}
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { concepts, pharmacological_distinctions, key_standards } = data;
  return (
    <div>
      <h2 style={{ color: '#6d4c41', marginBottom: 16 }}>Clinical Definitions & Standards</h2>

      <h3 style={{ marginBottom: 12 }}>Core Concepts</h3>
      {Object.entries(concepts || {}).map(([title, text]) => (
        <div key={title} style={{
          marginBottom: 16, background: '#fafafa', borderRadius: 8, padding: '12px 16px',
          borderLeft: '4px solid #6d4c41',
        }}>
          <strong style={{ color: '#6d4c41', fontSize: 14 }}>{title}</strong>
          <p style={{ color: '#555', fontSize: 13, lineHeight: 1.6, marginTop: 6 }}>{text}</p>
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {(pharmacological_distinctions || []).map((d, i) => (
        <div key={i} style={{
          marginBottom: 10, background: '#fff8e1', borderRadius: 6, padding: '10px 14px',
          borderLeft: '3px solid #f57f17', fontSize: 13, color: '#555', lineHeight: 1.5,
        }}>
          {d}
        </div>
      ))}

      <h3 style={{ marginTop: 24, marginBottom: 12 }}>Key Clinical Standards</h3>
      {(key_standards || []).map((s, i) => (
        <div key={i} style={{
          marginBottom: 8, background: '#e8f5e9', borderRadius: 6, padding: '10px 14px',
          borderLeft: '3px solid #2e7d32', fontSize: 13, color: '#333', lineHeight: 1.5,
        }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryLiverDiseaseAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-liver-disease-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' && !breakdown) {
      fetch(`${API}/api/hereditary-liver-disease-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if ((tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-liver-disease-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-liver-disease-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab]);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui, sans-serif' }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#6d4c41', marginBottom: 4 }}>
          🫀 Hereditary Liver Disease Atlas
        </h1>
        <p style={{ color: '#777', fontSize: 14 }}>
          Complete 8-Gene Hereditary Liver Disease Reference ·
          ATP7B (Wilson) · HFE (Hemochromatosis) · SERPINA1 (A1ATD) · JAG1 (Alagille) ·
          ATP8B1 (PFIC1) · ABCB11 (PFIC2) · ABCB4 (PFIC3) · ALDOB (HFI) ·
          320 patients, seeds 1582–1589
        </p>
      </div>

      {error && (
        <div style={{ background: '#ffebee', color: '#c62828', padding: '10px 14px', borderRadius: 6, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 6, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 18px', borderRadius: 20, border: 'none', cursor: 'pointer',
            background: tab === t ? '#6d4c41' : '#efebe9',
            color: tab === t ? '#fff' : '#5d4037',
            fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
