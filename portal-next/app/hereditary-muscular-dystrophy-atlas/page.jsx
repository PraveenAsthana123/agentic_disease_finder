'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  HNF1A:  '#1565c0',  // deep blue — MODY3 most common, sulfonylurea star
  GCK:    '#2e7d32',  // dark green — MODY2 benign, no treatment needed
  HNF4A:  '#6a1b9a',  // deep purple — MODY1 neonatal macrosomia
  HNF1B:  '#bf360c',  // deep orange-red — MODY5 RCAD renal cysts insulin required
  ABCC8:  '#00695c',  // dark teal — SUR1 K-ATP glibenclamide NDM
  KCNJ11: '#4a148c',  // dark violet — Kir6.2 DEND glibenclamide NDM
  INS:    '#880e4f',  // dark pink — insulin ER stress dominant-negative
  PDX1:   '#37474f',  // dark slate — master TF pancreatic agenesis biallelic
};

const GENE_DISEASE = {
  HNF1A:  'MODY3 AD — HNF1α Homeodomain TF — Most Common MODY ~50% — Renal Glucosuria SGLT2 Threshold — Sulfonylurea >98% Effective — Insulin→SU Transition >90% — Low HDL ApoA1',
  GCK:    'MODY2 AD — Glucokinase Hexokinase IV Glucose Sensor — Stable Fasting 5.5–8 mmol/L — NO Treatment Needed — NO Complications — Flat OGTT <3.5 mmol/L Rise — Pregnancy: Fetal Genotype Governs',
  HNF4A:  'MODY1 AD — HNF4α Nuclear Receptor TF — Neonatal Macrosomia +800g + Diazoxide-Responsive Hypoglycaemia PATHOGNOMONIC — Sulfonylurea Effective — ApoB/ApoC-III Hepatic',
  HNF1B:  'MODY5/RCAD AD — HNF1β POU Homeodomain — Renal Cysts PATHOGNOMONIC + Diabetes + Genital Tract Anomalies TRIAD — Pancreatic Atrophy — INSULIN Required SU INEFFECTIVE — Gout + Hypomagnesaemia — 50% De Novo',
  ABCC8:  'MODY12/NDM AD GOF / CHI AR LOF — SUR1 K-ATP Regulatory Subunit — Most Variable Phenotype — Neonatal Hypoglycaemia→NDM→Adult MODY — Diazoxide Test — Glibenclamide Replaces Insulin NDM 70%',
  KCNJ11: 'MODY13/NDM AD GOF — Kir6.2 K-ATP Pore Subunit — Permanent NDM → Oral Glibenclamide >90% — DEND Syndrome: Developmental Delay + Epilepsy + NDM — SU Crosses BBB → Neurological Benefit',
  INS:    'MODY10/NDM AD Dominant-Negative — Insulin Preprotein — Misfolding → ER Stress → Beta-Cell Apoptosis — Insulin Required SU Ineffective — Antibody Negative — Serial C-Peptide Decline',
  PDX1:   'MODY4 AD Heterozygous Mild / AR Biallelic = Pancreatic Agenesis — Master Pancreatic TF — Absent Pancreas MRI — Insulin + PERT + Fat-Soluble Vitamins — Parents of Biallelic = MODY4 Carriers',
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
      <span style={{ color: color || '#f59e0b', fontWeight: 700, marginRight: 6 }}>⚠</span>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { atlas, subtitle, total_patients, seed_range, aggregate_stats, genes, top_alerts } = data;
  return (
    <div>
      <h2 style={{ color: '#f1f5f9', marginBottom: 4 }}>{atlas}</h2>
      <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 20 }}>{subtitle}</p>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
        <KPI label="Total Patients" value={total_patients} color="#6366f1" />
        <KPI label="Genes Covered" value={aggregate_stats.genes_covered} color="#0ea5e9" />
        <KPI label="Patients / Gene" value={aggregate_stats.patients_per_gene} color="#10b981" />
        <KPI label="Mean Dx Age (yrs)" value={aggregate_stats.mean_dx_age} color="#f59e0b" />
        <KPI label="Mean Dx Delay (mo)" value={aggregate_stats.mean_dx_delay_months} color="#ef4444" />
        <KPI label="Seed Range" value={seed_range} color="#8b5cf6" />
      </div>

      <h3 style={{ color: '#f59e0b', marginBottom: 12 }}>Critical Clinical Alerts</h3>
      {top_alerts.map((a, i) => <Alert key={i} text={a} color={['#ef4444','#f59e0b','#10b981','#6366f1','#0ea5e9','#ec4899','#8b5cf6','#14b8a6'][i % 8]} />)}

      <h3 style={{ color: '#e2e8f0', marginTop: 28, marginBottom: 12 }}>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(360px,1fr))', gap: 14 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: '#1e293b', borderRadius: 10, padding: '1rem',
            borderTop: `3px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc', fontSize: 16 }}>{g.gene}</span>
              <span style={{ fontSize: 11, color: '#64748b' }}>{g.locus} · {g.aa} aa · {g.kDa} kDa</span>
            </div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8, lineHeight: 1.4 }}>{GENE_DISEASE[g.gene]}</div>
            <div style={{ display: 'flex', gap: 10, fontSize: 11, color: '#64748b' }}>
              <span>Dx Age: <b style={{ color: '#e2e8f0' }}>{g.mean_dx_age}y</b></span>
              <span>Delay: <b style={{ color: '#e2e8f0' }}>{g.mean_dx_delay_months}mo</b></span>
              <span>N: <b style={{ color: '#e2e8f0' }}>{g.n_patients}</b></span>
            </div>
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
      <h3 style={{ color: '#e2e8f0', marginBottom: 12 }}>Gene Reference Table — All 8 MODY Genes</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene','Protein (short)','Locus','aa','kDa','OMIM Gene','Inheritance','Gene Class',
                'Mean Dx Age','Mean Dx Delay','N Patients'].map(h => (
                <th key={h} style={{ padding: '8px 10px', color: '#94a3b8', textAlign: 'left',
                  borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#162032' }}>
                <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0', maxWidth: 220, fontSize: 11 }}>{g.protein.slice(0, 90)}&hellip;</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.locus}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.kDa}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8' }}>{g.omim_gene}</td>
                <td style={{ padding: '8px 10px', color: '#94a3b8', maxWidth: 160, fontSize: 11 }}>{g.inheritance.slice(0, 80)}&hellip;</td>
                <td style={{ padding: '8px 10px', color: '#64748b', maxWidth: 200, fontSize: 11 }}>{g.gene_class.slice(0, 80)}&hellip;</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{g.computed.mean_dx_age}y</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{g.computed.mean_dx_delay_months}mo</td>
                <td style={{ padding: '8px 10px', color: '#e2e8f0' }}>{g.computed.n_patients}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  const [selected, setSelected] = useState(null);
  if (!data) return <Loading />;
  const gene = selected ? data.find(g => g.gene === selected) : null;
  return (
    <div style={{ display: 'flex', gap: 16 }}>
      <div style={{ width: 160, flexShrink: 0 }}>
        {data.map(g => (
          <button key={g.gene}
            onClick={() => setSelected(g.gene)}
            style={{
              display: 'block', width: '100%', textAlign: 'left',
              padding: '8px 12px', marginBottom: 4, borderRadius: 6,
              background: selected === g.gene ? GENE_COLORS[g.gene] || '#6366f1' : '#1e293b',
              color: '#e2e8f0', border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
            }}>{g.gene}</button>
        ))}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        {!gene ? (
          <div style={{ color: '#64748b', padding: '2rem' }}>Select a gene to view full clinical profile.</div>
        ) : (
          <div>
            <h3 style={{ color: GENE_COLORS[gene.gene] || '#a5b4fc', marginBottom: 4 }}>{gene.gene}</h3>
            <p style={{ color: '#94a3b8', fontSize: 12, marginBottom: 12 }}>{gene.protein}</p>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
              <span style={{ background: '#0f172a', padding: '4px 10px', borderRadius: 5, fontSize: 11, color: '#94a3b8' }}>
                {gene.locus} · {gene.aa} aa · {gene.kDa} kDa
              </span>
              <span style={{ background: '#0f172a', padding: '4px 10px', borderRadius: 5, fontSize: 11, color: '#94a3b8' }}>
                OMIM: {gene.omim_gene}
              </span>
              <span style={{ background: '#0f172a', padding: '4px 10px', borderRadius: 5, fontSize: 11, color: '#94a3b8' }}>
                {gene.inheritance.slice(0, 60)}
              </span>
            </div>

            <h4 style={{ color: '#f59e0b', marginBottom: 8 }}>Key Clinical Alerts</h4>
            {gene.key_alerts.map((a, i) => <Alert key={i} text={a} color="#f59e0b" />)}

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Disease Etiologies</h4>
            {gene.etiologies.map((e, i) => (
              <div key={i} style={{ background: '#0f172a', borderRadius: 6, padding: '0.65rem 1rem',
                marginBottom: 8, fontSize: 12, color: '#e2e8f0', lineHeight: 1.5 }}>
                <span style={{ color: '#6366f1', fontWeight: 700, marginRight: 6 }}>{i+1}.</span>{e}
              </div>
            ))}

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Gene Function (Full)</h4>
            <div style={{ background: '#0f172a', borderRadius: 8, padding: '1rem',
              fontSize: 11, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
              {gene.alias}
            </div>

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Cohort Statistics</h4>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {Object.entries(gene.stats).map(([k, v]) => (
                <KPI key={k} label={k.replace(/_/g, ' ')} value={v} color={GENE_COLORS[gene.gene]} />
              ))}
            </div>

            <h4 style={{ color: '#e2e8f0', marginTop: 20, marginBottom: 8 }}>Sample Patients (first 10)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                <thead>
                  <tr style={{ background: '#0f172a' }}>
                    {['ID','Age Dx','Delay(mo)','Treatment','Antibody+','C-Pep Pres',
                      'Renal Gluc','Neonatal Dx','Renal Cysts','Panc Atrophy','SU Effective'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', color: '#64748b', textAlign: 'left',
                        borderBottom: '1px solid #1e293b', whiteSpace: 'nowrap' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {gene.sample_patients.map((p, i) => (
                    <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#1e293b' : '#162032' }}>
                      <td style={{ padding: '6px 8px', color: '#94a3b8' }}>{p.patient_id}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{p.age_at_dx}</td>
                      <td style={{ padding: '6px 8px', color: '#e2e8f0' }}>{p.dx_delay_months}</td>
                      <td style={{ padding: '6px 8px', color: '#94a3b8', fontSize: 10 }}>{p.treatment}</td>
                      <td style={{ padding: '6px 8px', color: p.antibody_positive ? '#ef4444' : '#64748b' }}>
                        {p.antibody_positive ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.c_peptide_preserved ? '#10b981' : '#ef4444' }}>
                        {p.c_peptide_preserved ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.renal_glucosuria ? '#f59e0b' : '#64748b' }}>
                        {p.renal_glucosuria ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.neonatal_dx ? '#ef4444' : '#64748b' }}>
                        {p.neonatal_dx ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.renal_cysts ? '#8b5cf6' : '#64748b' }}>
                        {p.renal_cysts ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.pancreatic_atrophy ? '#f59e0b' : '#64748b' }}>
                        {p.pancreatic_atrophy ? 'Yes' : 'No'}
                      </td>
                      <td style={{ padding: '6px 8px', color: p.sulfonylurea_effective ? '#10b981' : '#ef4444' }}>
                        {p.sulfonylurea_effective ? 'Yes' : 'No'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { concepts, pharmacological_distinctions, key_standards } = data;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: 16 }}>Core Concepts</h3>
      {Object.entries(concepts).map(([title, text]) => (
        <div key={title} style={{ marginBottom: 20 }}>
          <h4 style={{ color: '#6366f1', marginBottom: 8, fontSize: 13 }}>{title}</h4>
          <div style={{ background: '#0f172a', borderRadius: 8, padding: '1rem',
            fontSize: 12, color: '#cbd5e1', lineHeight: 1.7 }}>{text}</div>
        </div>
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 28, marginBottom: 12 }}>Pharmacological Distinctions</h3>
      {pharmacological_distinctions.map((d, i) => (
        <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem',
          marginBottom: 10, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
          <span style={{ color: '#10b981', fontWeight: 700, marginRight: 6 }}>Rx {i+1}:</span>{d}
        </div>
      ))}

      <h3 style={{ color: '#e2e8f0', marginTop: 28, marginBottom: 12 }}>Key Standards &amp; Trials</h3>
      {key_standards.map((s, i) => (
        <div key={i} style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem',
          marginBottom: 10, fontSize: 12, color: '#cbd5e1', lineHeight: 1.6 }}>
          <span style={{ color: '#f59e0b', fontWeight: 700, marginRight: 6 }}>§{i+1}</span>{s}
        </div>
      ))}
    </div>
  );
}

export default function MODYPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-muscular-dystrophy-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-muscular-dystrophy-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-muscular-dystrophy-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#0b1120', color: '#e2e8f0', padding: '2rem' }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>
        <div style={{ marginBottom: 24 }}>
          <h1 style={{ color: '#f1f5f9', fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
            🧬 Hereditary Monogenic Diabetes (MODY) Atlas
          </h1>
          <p style={{ color: '#64748b', fontSize: 13 }}>
            Complete 8-Gene Hereditary Muscular Dystrophy Atlas — HNF1A / GCK / HNF4A / HNF1B / ABCC8 / KCNJ11 / INS / PDX1 — 320 Patients (8×40, Seeds 1694–1701)
          </p>
        </div>

        {error && <ErrorBox msg={error} />}

        <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '1px solid #1e293b', paddingBottom: 8 }}>
          {TABS.map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === t ? '#1565c0' : '#1e293b',
              color: tab === t ? '#fff' : '#94a3b8', fontWeight: tab === t ? 700 : 400,
            }}>{t}</button>
          ))}
        </div>

        {tab === 'Overview'      && <OverviewTab data={overview} />}
        {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
        {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
