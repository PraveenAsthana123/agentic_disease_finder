'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  BRCA2:  '#1565c0',  // deep blue — highest risk, lethal phenotype, PARPi #1
  ATM:    '#4e342e',  // dark brown — intermediate HR, modest PARPi, radiation sensitivity
  CHEK2:  '#e65100',  // deep orange — moderate, NO PARPi, c.1100delC founder
  HOXB13: '#2e7d32',  // dark green — G84E founder, prostate-only, no targeted therapy
  MSH2:   '#6a1b9a',  // deep purple — Lynch, pembrolizumab, multi-organ
  BRCA1:  '#7b1fa2',  // violet — weaker prostate PARPi, female relatives HBOC
  PALB2:  '#00695c',  // dark teal — emerging data, PARPi-sensitive bridge
  NBN:    '#37474f',  // dark slate — 657del5 Slavic founder, MRN complex
};

const GENE_DISEASE = {
  BRCA2:  'HPCA-1 AD — 15–20× Prostate RR — Lethal/Metastatic Enriched — PROfound Olaparib HR 0.22 — PSMA-PET Mandatory',
  ATM:    'HPCA-2 AD-Heterozygous — 2–4× Prostate RR — PROfound Olaparib Modest HR 0.72 — Radiation Sensitivity Intermediate',
  CHEK2:  'HPCA-3 AD — 2–3× Prostate RR — c.1100delC + I157T Founders — NO PARPi Approved — Standard ADT',
  HOXB13: 'HPCA-4 AD — G84E Founder Scandinavian — 4–8× Prostate-Specific — No Targeted Therapy — AR Co-regulator NOT DNA Repair',
  MSH2:   'HPCA-5 AD — Lynch Syndrome 5–10× Prostate RR — dMMR/MSI-H → Pembrolizumab — Multi-Organ Lynch Surveillance',
  BRCA1:  'HPCA-6 AD — 2–3× Prostate RR (Weaker than BRCA2) — PARPi Less Effective HR 0.82 — Female Relatives Full HBOC',
  PALB2:  'HPCA-7 AD — Emerging 2–4× Prostate RR — PARPi-Sensitive BRCA1-BRCA2 Bridge — No Dedicated Prostate PARPi Label',
  NBN:    'HPCA-8 AR-Biallelic-Nijmegen / AD-Heterozygous Prostate — 657del5 Slavic Founder — 3–4× RR — MRN Complex',
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

function AlertCard({ alert, color }) {
  const [code, ...rest] = alert.split(':');
  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: '0.85rem 1rem',
      borderLeft: `4px solid ${color || '#f59e0b'}`, marginBottom: 8,
    }}>
      <span style={{ color: color || '#fbbf24', fontWeight: 700, fontSize: 12 }}>{code}</span>
      {rest.length > 0 && <span style={{ color: '#cbd5e1', fontSize: 12 }}>:{rest.join(':')}</span>}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats;
  return (
    <div>
      <div style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ color: '#e2e8f0', marginBottom: 4 }}>{data.atlas}</h2>
        <p style={{ color: '#94a3b8', fontSize: 13 }}>{data.subtitle}</p>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '2rem' }}>
        <KPI label="Total patients" value={s.total_patients} color="#6366f1" />
        <KPI label="Mean Dx age (y)" value={s.mean_dx_age_years} color="#22c55e" />
        <KPI label="Mean Dx delay (mo)" value={s.mean_dx_delay_months} color="#f59e0b" />
        <KPI label="BRCA2 PSA screen age" value={`${s.brca2_psa_screening_age}y`} color="#1565c0" />
        <KPI label="BRCA2 biopsy PSA" value={`≥${s.brca2_psa_biopsy_threshold_ng_ml} ng/mL`} color="#1565c0" />
        <KPI label="BRCA2 PROfound HR" value={s.brca2_profond_hr_rPFS} color="#1565c0" />
        <KPI label="ATM PROfound HR" value={s.atm_profond_hr_rPFS} color="#4e342e" />
        <KPI label="MSH2 Lynch prostate RR" value={s.msh2_lynch_prostate_rr} color="#6a1b9a" />
        <KPI label="HOXB13 G84E Sweden %" value={`${s.hoxb13_g84e_carrier_pct_sweden}%`} color="#2e7d32" />
        <KPI label="Cascade tested %" value={`${s.cascade_tested_pct}%`} color="#0ea5e9" />
        <KPI label="PSMA-PET performed %" value={`${s.psma_pet_performed_pct}%`} color="#f43f5e" />
      </div>
      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem', fontSize: 14 }}>Gene Summary (8 genes, 40 patients each)</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {data.genes.map(g => (
            <div key={g.gene} style={{
              background: '#1e293b', borderRadius: 8, padding: '0.7rem 1rem',
              borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`, minWidth: 220,
            }}>
              <div style={{ color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700, fontSize: 13 }}>{g.gene}</div>
              <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 2 }}>{g.locus} · {g.inheritance}</div>
              <div style={{ color: '#64748b', fontSize: 10, marginTop: 2 }}>{g.omim_disease.slice(0, 60)}</div>
              <div style={{ color: '#475569', fontSize: 10, marginTop: 2 }}>Mean Dx age: {g.mean_dx_age}y · n={g.n_patients}</div>
            </div>
          ))}
        </div>
      </div>
      <div>
        <h3 style={{ color: '#e2e8f0', marginBottom: '0.8rem', fontSize: 14 }}>Top Clinical Alerts</h3>
        {data.top_alerts.map((a, i) => (
          <AlertCard key={i} alert={a} color={Object.values(GENE_COLORS)[i % 8]} />
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: '1rem' }}>Hereditary Prostate Cancer Gene Reference Table</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#1e293b' }}>
              {['Gene', 'Locus', 'aa', 'kDa', 'OMIM Gene', 'OMIM Disease', 'Inheritance', 'N Patients', 'Gene Class'].map(h => (
                <th key={h} style={{ padding: '0.6rem 0.8rem', textAlign: 'left', color: '#94a3b8', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {genes.map(g => (
              <tr key={g.gene} style={{ borderBottom: '1px solid #1e293b' }}>
                <td style={{ padding: '0.5rem 0.8rem', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#e2e8f0' }}>{g.locus}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#94a3b8' }}>{g.aa}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#94a3b8' }}>{g.kDa}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#64748b' }}>{g.omim_gene}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#64748b', maxWidth: 200 }}>{g.omim_disease.slice(0, 50)}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#94a3b8' }}>{g.inheritance.split(';')[0]}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#e2e8f0' }}>{g.stats.n_patients}</td>
                <td style={{ padding: '0.5rem 0.8rem', color: '#64748b', maxWidth: 180, fontSize: 10 }}>{g.gene_class.slice(0, 60)}</td>
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
  const [selectedGene, setSelectedGene] = useState(Object.keys(data)[0]);
  const g = data[selectedGene];
  if (!g) return <Loading />;
  return (
    <div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: '1.5rem' }}>
        {Object.keys(data).map(gene => (
          <button key={gene} onClick={() => setSelectedGene(gene)} style={{
            padding: '0.4rem 0.9rem', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 700,
            background: selectedGene === gene ? (GENE_COLORS[gene] || '#6366f1') : '#1e293b',
            color: selectedGene === gene ? '#fff' : '#94a3b8',
          }}>{gene}</button>
        ))}
      </div>
      <div style={{ background: '#1e293b', borderRadius: 10, padding: '1.2rem', marginBottom: '1rem' }}>
        <div style={{ color: GENE_COLORS[selectedGene] || '#a5b4fc', fontWeight: 700, fontSize: 15, marginBottom: 4 }}>
          {selectedGene}
        </div>
        <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>
          {g.locus} · {g.aa} aa · {g.kDa} kDa · {g.inheritance.split(';')[0]}
        </div>
        <div style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.6, marginBottom: '1rem' }}>
          {GENE_DISEASE[selectedGene]}
        </div>
        <div style={{ color: '#94a3b8', fontSize: 11, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
          {g.alias}
        </div>
      </div>
      <div style={{ marginBottom: '1rem' }}>
        <h4 style={{ color: '#e2e8f0', marginBottom: 8, fontSize: 13 }}>Clinical Alerts</h4>
        {g.key_alerts.map((a, i) => <AlertCard key={i} alert={a} color={GENE_COLORS[selectedGene]} />)}
      </div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: '1rem' }}>
        <div style={{ flex: 1, minWidth: 240, background: '#1e293b', borderRadius: 8, padding: '1rem' }}>
          <h4 style={{ color: '#e2e8f0', marginBottom: 8, fontSize: 13 }}>Etiologies</h4>
          {Object.entries(g.etiologies).map(([et, pct]) => (
            <div key={et} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>{et}</span>
                <span style={{ color: '#e2e8f0', fontSize: 11, fontWeight: 700 }}>{pct}%</span>
              </div>
              <div style={{ background: '#0f172a', borderRadius: 3, height: 5 }}>
                <div style={{ background: GENE_COLORS[selectedGene] || '#6366f1', width: `${pct}%`, height: 5, borderRadius: 3 }} />
              </div>
            </div>
          ))}
        </div>
        <div style={{ flex: 1, minWidth: 240, background: '#1e293b', borderRadius: 8, padding: '1rem' }}>
          <h4 style={{ color: '#e2e8f0', marginBottom: 8, fontSize: 13 }}>Dx Delay Distribution</h4>
          {Object.entries(g.dx_delay_distribution).map(([bucket, pct]) => (
            <div key={bucket} style={{ marginBottom: 6 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>{bucket}</span>
                <span style={{ color: '#e2e8f0', fontSize: 11, fontWeight: 700 }}>{pct}%</span>
              </div>
              <div style={{ background: '#0f172a', borderRadius: 3, height: 5 }}>
                <div style={{ background: '#f59e0b', width: `${pct}%`, height: 5, borderRadius: 3 }} />
              </div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ background: '#1e293b', borderRadius: 8, padding: '1rem' }}>
        <h4 style={{ color: '#e2e8f0', marginBottom: 8, fontSize: 13 }}>Sample Patients (first 10)</h4>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
          <thead>
            <tr>
              {['Patient ID', 'Age at Dx (y)', 'Dx Delay (mo)', 'Seed'].map(h => (
                <th key={h} style={{ padding: '0.4rem 0.6rem', textAlign: 'left', color: '#64748b', borderBottom: '1px solid #334155' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {g.patients.map(p => (
              <tr key={p.patient_id}>
                <td style={{ padding: '0.35rem 0.6rem', color: '#94a3b8' }}>{p.patient_id}</td>
                <td style={{ padding: '0.35rem 0.6rem', color: '#e2e8f0' }}>{p.age_at_diagnosis}</td>
                <td style={{ padding: '0.35rem 0.6rem', color: '#e2e8f0' }}>{p.diagnosis_delay_months}</td>
                <td style={{ padding: '0.35rem 0.6rem', color: '#64748b' }}>{p.seed}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#e2e8f0', marginBottom: '1rem' }}>Hereditary Prostate Cancer Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body]) => (
        <div key={title} style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', marginBottom: '1rem' }}>
          <h4 style={{ color: '#a5b4fc', marginBottom: 6, fontSize: 13 }}>{title}</h4>
          <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>{body}</p>
        </div>
      ))}
      {data.pharmacological_distinctions && (
        <div style={{ background: '#1e293b', borderRadius: 8, padding: '1rem', marginBottom: '1rem' }}>
          <h4 style={{ color: '#22c55e', marginBottom: 8, fontSize: 13 }}>Pharmacological Distinctions</h4>
          {data.pharmacological_distinctions.map((d, i) => {
            const [title, ...rest] = d.split(':');
            return (
              <div key={i} style={{ marginBottom: 8, paddingLeft: 10, borderLeft: '3px solid #22c55e' }}>
                <span style={{ color: '#86efac', fontWeight: 700, fontSize: 11 }}>{title}</span>
                {rest.length > 0 && <span style={{ color: '#94a3b8', fontSize: 11 }}>:{rest.join(':')}</span>}
              </div>
            );
          })}
        </div>
      )}
      {data.key_standards && (
        <div style={{ background: '#1e293b', borderRadius: 8, padding: '1rem' }}>
          <h4 style={{ color: '#f59e0b', marginBottom: 8, fontSize: 13 }}>Key Standards &amp; Guidelines</h4>
          {data.key_standards.map((s, i) => {
            const [title, ...rest] = s.split(':');
            return (
              <div key={i} style={{ marginBottom: 8, paddingLeft: 10, borderLeft: '3px solid #f59e0b' }}>
                <span style={{ color: '#fcd34d', fontWeight: 700, fontSize: 11 }}>{title}</span>
                {rest.length > 0 && <span style={{ color: '#94a3b8', fontSize: 11 }}>:{rest.join(':')}</span>}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function HeriditaryProstateCancerAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-prostate-cancer-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-prostate-cancer-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-prostate-cancer-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  if (error) return (
    <div style={{ padding: '2rem', background: '#0f172a', minHeight: '100vh' }}>
      <ErrorBox msg={error} />
    </div>
  );

  return (
    <div style={{ padding: '1.5rem 2rem', background: '#0f172a', minHeight: '100vh', fontFamily: 'system-ui,sans-serif' }}>
      <div style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ color: '#e2e8f0', fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
          🧬 Hereditary Prostate Cancer Atlas
        </h1>
        <p style={{ color: '#64748b', fontSize: 12 }}>
          Complete 8-Gene Hereditary Prostate Cancer Reference ·
          BRCA2 · ATM · CHEK2 · HOXB13 · MSH2 · BRCA1 · PALB2 · NBN ·
          320 patients (8×40, seeds 1638–1645)
        </p>
      </div>

      <div style={{ display: 'flex', gap: 6, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '0.45rem 1.1rem', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#6366f1' : '#1e293b',
            color: tab === t ? '#fff' : '#94a3b8',
          }}>{t}</button>
        ))}
      </div>

      <div>
        {tab === 'Overview'       && <OverviewTab      data={overview}    />}
        {tab === 'Gene Table'     && <GeneTableTab     data={breakdown}   />}
        {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown}   />}
        {tab === 'Definitions'    && <DefinitionsTab   data={definitions} />}
      </div>
    </div>
  );
}
