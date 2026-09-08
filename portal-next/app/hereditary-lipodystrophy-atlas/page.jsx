'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BSCL2:   '#1565c0',  // deep blue    — CGL2 most common, all fat absent
  AGPAT2:  '#2e7d32',  // deep green   — CGL1 mechanical fat preserved
  LMNA:    '#b71c1c',  // deep red     — FPLD2 Dunnigan cardiac MANDATORY
  PPARG:   '#e65100',  // deep orange  — FPLD3 TZD target dominant negative
  PLIN1:   '#6a1b9a',  // deep purple  — FPLD4 TG>20 pancreatitis
  AKT2:    '#004d40',  // deep teal    — severe IR / GOF hypoglycaemia
  CAV1:    '#880e4f',  // deep magenta — CGL3/FPLD7 PAH overlap
  ZMPSTE24:'#37474f',  // dark slate   — MADB progeroid acroosteolysis
};

const GENE_INFO = {
  BSCL2:    { aa: 398,  locus: '11q13.1', inh: 'AR',    disease: 'CGL2-Most-Common-ALL-Fat-Absent-Including-Mechanical — Metreleptin-FDA-2014 — Intellectual-Disability-50pct — Pseudoathleticism' },
  AGPAT2:   { aa: 278,  locus: '9q34.3',  inh: 'AR',    disease: 'CGL1-Mechanical-Fat-PRESERVED-Palms-Soles-Periorbit — Bone-Cysts-X-ray-PATHOGNOMONIC — No-Intellectual-Disability — Metreleptin-FDA' },
  LMNA:     { aa: 664,  locus: '1q22',    inh: 'AD',    disease: 'FPLD2-Dunnigan-Fat-REDISTRIBUTION-Not-Absence — CARDIAC-MANDATORY-Annual-ECG-Holter-Echo — ICD-NSVT-HB-EF45 — Exon8-R482W-R482Q' },
  PPARG:    { aa: 505,  locus: '3p25.2',  inh: 'AD',    disease: 'FPLD3-Dominant-Negative-LBD — TZD-Direct-Target-Impaired-Response — Adiponectin-VERY-LOW — P467L-V290M-Hotspots' },
  PLIN1:    { aa: 522,  locus: '15q26.1', inh: 'AD',    disease: 'FPLD4-TG-Above-20mmol-PANCREATITIS-MANDATORY-Fibrates — Subtle-Visible-Fat-Loss-Delayed-Diagnosis — Unregulated-Lipolysis' },
  AKT2:     { aa: 481,  locus: '19q13.2', inh: 'AD',    disease: 'LOF-Severe-IR-Partial-Lipodystrophy — GOF-Neonatal-Hypoglycaemia-Macrosomia — Diazoxide-May-Fail-GOF — mTOR-Sirolimus-Investigational' },
  CAV1:     { aa: 178,  locus: '7q31.2',  inh: 'AR/AD', disease: 'CGL3-AR-PAH-Overlap-Echo-MANDATORY — FPLD7-AD-Partial — Caveolae-Absent-EM — BMPR2-Panel-If-PAH-Prominent' },
  ZMPSTE24: { aa: 475,  locus: '1p34.2',  inh: 'AR',    disease: 'MADB-Progeroid-Mandibular-Hypoplasia-ACROOSTEOLYSIS-PATHOGNOMONIC — Prelamin-A-Accumulation — Lonafarnib-FTI — Distinguish-from-LMNA' },
};

function Loading() {
  return <div style={{ padding: 20, color: '#555' }}>Loading…</div>;
}

function ErrBox({ msg }) {
  return <div style={{ padding: 16, background: '#fce4ec', color: '#b71c1c', borderRadius: 6 }}>{msg}</div>;
}

function StatCard({ label, value, color }) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color || '#1565c0'}`, borderRadius: 8, padding: '12px 16px', minWidth: 160, textAlign: 'center' }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: color || '#1565c0' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  return (
    <div>
      <h2 style={{ color: '#1565c0' }}>Hereditary-Lipodystrophy-Atlas</h2>
      <p style={{ color: '#444', marginBottom: 16 }}>{ov.subtitle}</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Patients" value={ov.total_patients} color="#1565c0" />
        <StatCard label="Genes" value={ov.genes?.length} color="#2e7d32" />
        <StatCard label="Seeds" value={ov.seeds} color="#37474f" />
        <StatCard label="Generalised Lipoatrophy" value={ov.generalised_lipoatrophy_patients} color="#b71c1c" />
        <StatCard label="Severe Hypertrigly" value={ov.severe_hypertrigly_patients} color="#e65100" />
        <StatCard label="T2D" value={ov.t2d_patients} color="#6a1b9a" />
        <StatCard label="Pancreatitis" value={ov.pancreatitis_patients} color="#c62828" />
        <StatCard label="Metreleptin Eligible" value={ov.metreleptin_eligible_patients} color="#004d40" />
        <StatCard label="PAH Overlap" value={ov.pah_patients} color="#880e4f" />
        <StatCard label="Progeroid" value={ov.progeroid_patients} color="#37474f" />
        <StatCard label="Cardiac Conduction" value={ov.cardiac_conduction_patients} color="#b71c1c" />
        <StatCard label="Hepatomegaly" value={ov.hepatomegaly_patients} color="#4e342e" />
      </div>

      <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <strong>Pathway:</strong> <span style={{ fontSize: 13 }}>{ov.pathway}</span>
      </div>

      <div style={{ background: '#fff8e1', borderRadius: 8, padding: 16, marginBottom: 20, border: '1px solid #ffe082' }}>
        <strong style={{ color: '#e65100' }}>Key Clinical Insight:</strong>
        <div style={{ fontSize: 13, marginTop: 8, whiteSpace: 'pre-wrap' }}>{ov.key_clinical_insight}</div>
      </div>

      <div style={{ background: '#f3e5f5', borderRadius: 8, padding: 16 }}>
        <strong>Per-Gene Patient Counts:</strong>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
          {ov.genes?.map(g => (
            <div key={g} style={{ background: GENE_COLORS[g] || '#555', color: '#fff', borderRadius: 6, padding: '6px 14px', fontSize: 13 }}>
              {g}: {ov.gene_patient_counts?.[g] || 40}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  const [filter, setFilter] = useState('');
  const filtered = genes.filter(g =>
    !filter || g.gene?.toLowerCase().includes(filter.toLowerCase()) ||
    g.alt_name?.toLowerCase().includes(filter.toLowerCase()) ||
    g.locus?.toLowerCase().includes(filter.toLowerCase())
  );
  return (
    <div>
      <input
        placeholder="Filter genes…"
        value={filter}
        onChange={e => setFilter(e.target.value)}
        style={{ marginBottom: 16, padding: '8px 12px', width: '100%', maxWidth: 400, borderRadius: 6, border: '1px solid #ccc', fontSize: 14 }}
      />
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
          <thead>
            <tr style={{ background: '#1565c0', color: '#fff' }}>
              {['Gene', 'aa', 'Locus', 'Inh', 'Disease', 'TG%', 'T2D%', 'Panc%', 'Metreleptin%', 'Cardiac%', 'PAH%', 'Progeroid%'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                <td style={{ padding: '7px 10px' }}>{GENE_INFO[g.gene]?.aa}</td>
                <td style={{ padding: '7px 10px' }}>{g.locus}</td>
                <td style={{ padding: '7px 10px' }}>{g.inheritance?.split(' ')[0]}</td>
                <td style={{ padding: '7px 10px', maxWidth: 260, fontSize: 11, color: '#444' }}>{GENE_INFO[g.gene]?.disease}</td>
                <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.severe_hypertrigly_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.t2d_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: g.pancreatitis_pct > 30 ? '#c62828' : 'inherit', fontWeight: g.pancreatitis_pct > 30 ? 700 : 400 }}>{g.pancreatitis_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: g.metreleptin_eligible_pct > 50 ? '#004d40' : 'inherit', fontWeight: g.metreleptin_eligible_pct > 50 ? 700 : 400 }}>{g.metreleptin_eligible_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: g.cardiac_conduction_pct > 50 ? '#b71c1c' : 'inherit', fontWeight: g.cardiac_conduction_pct > 50 ? 700 : 400 }}>{g.cardiac_conduction_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: g.pah_pct > 20 ? '#880e4f' : 'inherit', fontWeight: g.pah_pct > 20 ? 700 : 400 }}>{g.pah_pct}%</td>
                <td style={{ padding: '7px 10px', textAlign: 'center', color: g.progeroid_pct > 50 ? '#37474f' : 'inherit', fontWeight: g.progeroid_pct > 50 ? 700 : 400 }}>{g.progeroid_pct}%</td>
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
  const genes = Object.values(data);
  const [selected, setSelected] = useState(null);
  const gene = selected ? data[selected] : null;

  return (
    <div style={{ display: 'flex', gap: 20 }}>
      <div style={{ minWidth: 160, flexShrink: 0 }}>
        {genes.map(g => (
          <div
            key={g.gene}
            onClick={() => setSelected(g.gene === selected ? null : g.gene)}
            style={{
              padding: '8px 12px', marginBottom: 6, borderRadius: 6, cursor: 'pointer',
              background: selected === g.gene ? (GENE_COLORS[g.gene] || '#1565c0') : '#e3f2fd',
              color: selected === g.gene ? '#fff' : '#1565c0',
              fontWeight: 600, fontSize: 14,
            }}
          >
            {g.gene}
          </div>
        ))}
      </div>

      <div style={{ flex: 1 }}>
        {!selected && (
          <div style={{ color: '#888', padding: 20 }}>← Select a gene to view clinical details</div>
        )}
        {gene && (
          <div>
            <h3 style={{ color: GENE_COLORS[gene.gene] || '#1565c0', marginBottom: 4 }}>{gene.gene}</h3>
            <div style={{ color: '#555', marginBottom: 12, fontSize: 13 }}>{gene.alt_name}</div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 16 }}>
              {[
                ['Locus', gene.locus],
                ['Protein', gene.protein_size],
                ['Inheritance', gene.inheritance?.split(' ')[0]],
                ['Patients', gene.n_patients],
                ['TG%', `${gene.severe_hypertrigly_pct}%`],
                ['T2D%', `${gene.t2d_pct}%`],
                ['Pancreatitis%', `${gene.pancreatitis_pct}%`],
                ['Metreleptin%', `${gene.metreleptin_eligible_pct}%`],
                ['Cardiac%', `${gene.cardiac_conduction_pct}%`],
                ['PAH%', `${gene.pah_pct}%`],
                ['Progeroid%', `${gene.progeroid_pct}%`],
                ['Bone Anomaly%', `${gene.bone_anomaly_pct}%`],
              ].map(([k, v]) => (
                <div key={k} style={{ background: '#f5f5f5', borderRadius: 6, padding: '4px 10px', fontSize: 13 }}>
                  <strong>{k}:</strong> {v}
                </div>
              ))}
            </div>

            {[
              ['Age of Onset', gene.age_of_onset, '#e3f2fd'],
              ['Key Biomarker', gene.key_biomarker, '#f3e5f5'],
              ['Pathognomonic', gene.pathognomonic, '#fff3e0'],
              ['Treatment', gene.treatment, '#e8f5e9'],
            ].map(([label, text, bg]) => (
              <div key={label} style={{ background: bg, borderRadius: 8, padding: 14, marginBottom: 12 }}>
                <strong style={{ fontSize: 13 }}>{label}:</strong>
                <div style={{ fontSize: 13, marginTop: 6, color: '#333', lineHeight: 1.6 }}>{text}</div>
              </div>
            ))}

            <div style={{ marginTop: 12 }}>
              <strong style={{ fontSize: 13 }}>Critical Flags:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                {gene.critical_flags?.map(f => (
                  <span key={f} style={{ background: GENE_COLORS[gene.gene] || '#1565c0', color: '#fff', borderRadius: 4, padding: '3px 8px', fontSize: 11, fontFamily: 'monospace' }}>
                    {f}
                  </span>
                ))}
              </div>
            </div>

            {gene.cohort_preview?.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <strong style={{ fontSize: 13 }}>Cohort Preview (first 5 patients):</strong>
                <div style={{ overflowX: 'auto', marginTop: 8 }}>
                  <table style={{ fontSize: 11, borderCollapse: 'collapse', width: '100%' }}>
                    <thead>
                      <tr style={{ background: GENE_COLORS[gene.gene] || '#1565c0', color: '#fff' }}>
                        {['ID', 'Age', 'Sex', 'Gen.Lipo', 'Mech.Abs', 'TG', 'T2D', 'Panc', 'Metreleptin', 'Cardiac', 'PAH', 'Progeroid'].map(h => (
                          <th key={h} style={{ padding: '4px 8px', textAlign: 'left' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {gene.cohort_preview.map((p, i) => (
                        <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff' }}>
                          <td style={{ padding: '3px 8px' }}>{p.patient_id}</td>
                          <td style={{ padding: '3px 8px' }}>{p.age}</td>
                          <td style={{ padding: '3px 8px' }}>{p.sex}</td>
                          <td style={{ padding: '3px 8px' }}>{p.generalised_lipoatrophy ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.mechanical_fat_absent ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.severe_hypertrigly ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.t2d ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.pancreatitis ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.metreleptin_eligible ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.cardiac_conduction ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.pah ? '✓' : '—'}</td>
                          <td style={{ padding: '3px 8px' }}>{p.progeroid ? '✓' : '—'}</td>
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

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#1565c0' }}>{data.atlas}</h3>
      <div style={{ color: '#555', marginBottom: 16, fontStyle: 'italic' }}>{data.pathway}</div>

      <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <strong>Shared Mechanism:</strong>
        <div style={{ fontSize: 13, marginTop: 8, lineHeight: 1.7 }}>{data.shared_mechanism}</div>
      </div>

      <h4>Gene Summaries</h4>
      {data.genes && Object.entries(data.genes).map(([gene, info]) => (
        <div key={gene} style={{ border: `1px solid ${GENE_COLORS[gene] || '#ccc'}`, borderRadius: 8, padding: 14, marginBottom: 12 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
            <span style={{ background: GENE_COLORS[gene] || '#555', color: '#fff', borderRadius: 4, padding: '3px 10px', fontWeight: 700, fontSize: 14 }}>{gene}</span>
            <span style={{ fontSize: 12, color: '#555' }}>{info.locus} · {info.protein_size} · {info.inheritance}</span>
          </div>
          <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}><strong>Full name:</strong> {info.full_name}</div>
          <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}><strong>Pathognomonic:</strong> {info.pathognomonic}</div>
          <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}><strong>Treatment:</strong> {info.treatment_summary}</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {info.critical_flags?.map(f => (
              <span key={f} style={{ background: '#f5f5f5', borderRadius: 3, padding: '2px 6px', fontSize: 10, fontFamily: 'monospace', color: '#333' }}>{f}</span>
            ))}
          </div>
        </div>
      ))}

      <h4>Glossary</h4>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 10 }}>
        {data.glossary && Object.entries(data.glossary).map(([term, def]) => (
          <div key={term} style={{ background: '#fafafa', border: '1px solid #e0e0e0', borderRadius: 6, padding: 10 }}>
            <div style={{ fontWeight: 600, color: '#1565c0', fontSize: 13, marginBottom: 4 }}>{term}</div>
            <div style={{ fontSize: 12, color: '#444', lineHeight: 1.5 }}>{def}</div>
          </div>
        ))}
      </div>

      <h4 style={{ marginTop: 24 }}>Surveillance Protocols</h4>
      {data.surveillance_protocols && Object.entries(data.surveillance_protocols).map(([gene, protocol]) => (
        <div key={gene} style={{ background: '#f3e5f5', borderRadius: 6, padding: 10, marginBottom: 8 }}>
          <strong style={{ color: GENE_COLORS[gene] || '#880e4f', fontSize: 13 }}>{gene}:</strong>
          <span style={{ fontSize: 13, color: '#333', marginLeft: 8 }}>{protocol}</span>
        </div>
      ))}
    </div>
  );
}

export default function HreditaryLipodystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/hereditary-lipodystrophy-atlas`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, br, def]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(def);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1400, margin: '0 auto', padding: 20 }}>
      <div style={{ background: 'linear-gradient(135deg, #1565c0 0%, #880e4f 100%)', color: '#fff', borderRadius: 10, padding: '18px 24px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary-Lipodystrophy-Atlas</h1>
        <div style={{ fontSize: 13, marginTop: 6, opacity: 0.9 }}>
          Complete 8-Gene CGL/FPLD Atlas · BSCL2 · AGPAT2 · LMNA · PPARG · PLIN1 · AKT2 · CAV1 · ZMPSTE24 · 320 Patients · Seeds 2014–2021
        </div>
      </div>

      {error && <ErrBox msg={error} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 14,
              background: tab === t ? '#1565c0' : '#e3f2fd',
              color: tab === t ? '#fff' : '#1565c0',
              fontWeight: tab === t ? 700 : 400,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Overview'      && <OverviewTab      data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab     data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab   data={definitions} />}
    </div>
  );
}
