'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ion-channel-disease-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  KCNA1:   '#1565c0',  // deep blue    — EA1 myokymia episodic ataxia
  KCNQ1:   '#2e7d32',  // deep green   — LQT1 swimming CI JLNS
  KCNH2:   '#6a1b9a',  // deep purple  — LQT2 alarm-clock auditory trigger
  SCN5A:   '#b71c1c',  // deep red     — Brugada fever CI coved ST
  RYR1:    '#e65100',  // deep orange  — MH dantrolene emergency
  CACNA1S: '#004d40',  // deep teal    — HypoPP1 paradoxical depolarisation
  CLCN1:   '#880e4f',  // deep magenta — Myotonia congenita warm-up
  KCNJ2:   '#37474f',  // dark slate   — ATS LQT7 triad
};

const GENE_INFO = {
  KCNA1:   { aa: 495,  locus: '12p13.32', inh: 'AD',     disease: 'EA1-Episodic-Ataxia-Type-1 — Myokymia-CMFA-EMG-PATHOGNOMONIC — Seconds-Duration-Distinguish-EA2 — Acetazolamide-First-Line-Carbamazepine-Myokymia' },
  KCNQ1:   { aa: 676,  locus: '11p15.5',  inh: 'AD/AR',  disease: 'LQT1-Broad-T-Wave — SWIMMING-ABSOLUTELY-CI — JLNS-Profound-SNHL-Biallelic — Beta-Blockers-Highly-Effective — Postpartum-Women-Highest-Risk' },
  KCNH2:   { aa: 1159, locus: '7q36.1',   inh: 'AD',     disease: 'LQT2-hERG — ALARM-CLOCK-AUDITORY-TRIGGER-PATHOGNOMONIC — Notched-Bifid-T-Wave — Avoid-ALL-QT-Prolonging-Drugs-CredibleMeds — Hypokalemia-Worsens' },
  SCN5A:   { aa: 2016, locus: '3p22.2',   inh: 'AD',     disease: 'Brugada-Coved-ST-V1-V3-PATHOGNOMONIC — FEVER-ABSOLUTELY-CI — LQT3-GOF — SSS-PCCD — Quinidine-Storm — ICD-Only-Proven-SCD-Prevention' },
  RYR1:    { aa: 5038, locus: '19q13.2',  inh: 'AD/AR',  disease: 'MH-DANTROLENE-2.5mgkg-IV-EMERGENCY — Volatile-Agents-ABSOLUTELY-CI — Succinylcholine-ABSOLUTELY-CI — MH-Alert-Bracelet-MANDATORY — CCD-Central-Cores-AR' },
  CACNA1S: { aa: 1873, locus: '1q32.1',   inh: 'AD',     disease: 'HypoPP1-PARADOXICAL-DEPOLARISATION-Unique — Dichlorphenamide-FDA-Approved — MH-Risk-R1086H — Acetazolamide-May-Worsen — Avoid-IV-Glucose-Saline-Attack' },
  CLCN1:   { aa: 988,  locus: '7q34',     inh: 'AD/AR',  disease: 'Myotonia-Congenita-WARM-UP-PATHOGNOMONIC — Mexiletine-First-Line — Thomsen-AD-Becker-AR-Transient-Weakness — Avoid-Succinylcholine-Intraop' },
  KCNJ2:   { aa: 427,  locus: '17q24.3',  inh: 'AD',     disease: 'ATS-LQT7-TRIAD-PATHOGNOMONIC — Periodic-Paralysis-Cardiac-Dysmorphic — Prominent-U-Waves-ECG — Normokalemic-PP-Unusual-Clue — Flecainide-PVC-Suppression' },
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
      <h2 style={{ color: '#1565c0' }}>Hereditary-Ion-Channel-Disease-Atlas</h2>
      <p style={{ color: '#444', marginBottom: 16 }}>{ov.subtitle}</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Patients" value={ov.total_patients} color="#1565c0" />
        <StatCard label="Genes" value={ov.genes?.length} color="#2e7d32" />
        <StatCard label="Seeds" value={ov.seeds} color="#37474f" />
        <StatCard label="Cardiac Arrhythmia" value={ov.cardiac_arrhythmia_patients} color="#b71c1c" />
        <StatCard label="Malignant Hyperthermia" value={ov.malignant_hyperthermia_patients} color="#e65100" />
        <StatCard label="Periodic Paralysis" value={ov.periodic_paralysis_patients} color="#004d40" />
        <StatCard label="Myotonia" value={ov.myotonia_patients} color="#880e4f" />
        <StatCard label="Episodic Ataxia" value={ov.episodic_ataxia_patients} color="#6a1b9a" />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <StatCard label="Drug Trigger Hazard" value={ov.drug_trigger_hazard_patients} color="#b71c1c" />
        <StatCard label="Fever Hazard" value={ov.fever_hazard_patients} color="#e65100" />
        <StatCard label="Swimming Restriction" value={ov.swimming_restriction_patients} color="#1565c0" />
        <StatCard label="CK Elevated Rest" value={ov.ck_elevated_rest_patients} color="#004d40" />
        <StatCard label="SNHL" value={ov.snhl_patients} color="#2e7d32" />
        <StatCard label="Dysmorphic Features" value={ov.dysmorphic_features_patients} color="#37474f" />
        <StatCard label="Sudden Cardiac Death" value={ov.sudden_cardiac_death_patients} color="#b71c1c" />
        <StatCard label="Warm-up Phenomenon" value={ov.warm_up_phenomenon_patients} color="#880e4f" />
      </div>

      <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 16 }}>
        <strong>Pathway:</strong> {ov.pathway}
      </div>
      <div style={{ background: '#fce4ec', borderRadius: 8, padding: 16 }}>
        <strong>Key Clinical Insight:</strong> {ov.key_clinical_insight}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
            <th>Locus</th>
            <th>Size</th>
            <th>Inh.</th>
            <th>Cardiac Arr.</th>
            <th>SCD</th>
            <th>Paralysis</th>
            <th>Myotonia</th>
            <th>MH</th>
            <th>Weakness</th>
            <th>Ataxia</th>
            <th>Myokymia</th>
            <th>Warm-up</th>
            <th>Drug CI</th>
            <th>Dysmorphic</th>
            <th>SNHL</th>
            <th>Fever Haz.</th>
            <th>Swim CI</th>
            <th>CK↑</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => {
            const d = data[g];
            const color = GENE_COLORS[g] || '#333';
            const bg = i % 2 === 0 ? '#fff' : '#f5f5f5';
            const pct = v => `${v}%`;
            return (
              <tr key={g} style={{ background: bg }}>
                <td style={{ padding: '6px 12px', fontWeight: 700, color }}>{g}</td>
                <td style={{ textAlign: 'center' }}>{d.locus}</td>
                <td style={{ textAlign: 'center' }}>{d.protein_size}</td>
                <td style={{ textAlign: 'center' }}>{d.inheritance?.split(' ')[0]}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.cardiac_arrhythmia_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.sudden_cardiac_death_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.periodic_paralysis_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.myotonia_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.malignant_hyperthermia_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.muscle_weakness_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.episodic_ataxia_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.myokymia_emg_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.warm_up_phenomenon_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.drug_trigger_hazard_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.dysmorphic_features_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.snhl_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.fever_hazard_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.swimming_restriction_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.ck_elevated_rest_pct)}</td>
                <td style={{ textAlign: 'center', fontWeight: 600 }}>{d.n_patients}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [sel, setSel] = useState(Object.keys(data)[0]);
  const d = data[sel];
  const color = GENE_COLORS[sel] || '#333';
  return (
    <div style={{ display: 'flex', gap: 16 }}>
      <div style={{ minWidth: 130 }}>
        {Object.keys(data).map(g => (
          <div
            key={g}
            onClick={() => setSel(g)}
            style={{
              padding: '8px 14px', marginBottom: 4, borderRadius: 6, cursor: 'pointer',
              background: sel === g ? GENE_COLORS[g] : '#f5f5f5',
              color: sel === g ? '#fff' : '#333',
              fontWeight: sel === g ? 700 : 400,
            }}
          >{g}</div>
        ))}
      </div>
      <div style={{ flex: 1 }}>
        <h3 style={{ color }}>{d.gene} — {d.protein_size} — {d.locus} — {d.inheritance?.split(' ')[0]}</h3>
        <div style={{ marginBottom: 12 }}>
          <strong style={{ color }}>Disease: </strong>
          <span>{GENE_INFO[sel]?.disease}</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
          {d.critical_flags?.map(f => (
            <span key={f} style={{ background: color, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>{f}</span>
          ))}
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Age / Onset:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.age_of_onset}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Key Biomarker:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.key_biomarker}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Pathognomonic / DDx:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.pathognomonic}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Treatment:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.treatment}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Patient Preview (first 5 of 40):</strong>
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
              <thead>
                <tr style={{ background: color, color: '#fff' }}>
                  <th style={{ padding: '4px 8px' }}>ID</th>
                  <th>Age</th>
                  <th>Sex</th>
                  <th>Arrhythmia</th>
                  <th>Paralysis</th>
                  <th>Myotonia</th>
                  <th>MH</th>
                  <th>Ataxia</th>
                  <th>Drug CI</th>
                </tr>
              </thead>
              <tbody>
                {d.cohort_preview?.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5' }}>
                    <td style={{ padding: '3px 8px' }}>{p.patient_id}</td>
                    <td style={{ textAlign: 'center' }}>{p.age}</td>
                    <td style={{ textAlign: 'center' }}>{p.sex}</td>
                    <td style={{ textAlign: 'center' }}>{p.cardiac_arrhythmia ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.periodic_paralysis ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.myotonia ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.malignant_hyperthermia ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.episodic_ataxia ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.drug_trigger_hazard ? '✓' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#1565c0' }}>{data.atlas}</h3>
      <p><strong>Pathway:</strong> {data.pathway}</p>
      <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 14, marginBottom: 16 }}>
        <strong>Shared Mechanism:</strong> {data.shared_mechanism}
      </div>

      <h4>Gene Summaries</h4>
      {data.genes && Object.entries(data.genes).map(([g, info]) => (
        <div key={g} style={{ borderLeft: `4px solid ${GENE_COLORS[g] || '#333'}`, paddingLeft: 12, marginBottom: 14 }}>
          <strong style={{ color: GENE_COLORS[g] || '#333' }}>{g}</strong> — {info.locus} — {info.protein_size} — {info.inheritance?.split(' ')[0]}
          <div style={{ fontSize: 12, color: '#444', marginTop: 4 }}>{info.pathognomonic?.substring(0, 250)}…</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
            {info.critical_flags?.map(f => (
              <span key={f} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', borderRadius: 3, padding: '1px 6px', fontSize: 10 }}>{f}</span>
            ))}
          </div>
        </div>
      ))}

      <h4>Glossary</h4>
      {data.glossary && Object.entries(data.glossary).map(([term, def]) => (
        <div key={term} style={{ marginBottom: 10 }}>
          <strong>{term}:</strong> <span style={{ color: '#444' }}>{def}</span>
        </div>
      ))}

      <h4>Surveillance Protocols</h4>
      {data.surveillance_protocols && Object.entries(data.surveillance_protocols).map(([gene, proto]) => (
        <div key={gene} style={{ marginBottom: 8, borderLeft: `3px solid ${GENE_COLORS[gene] || '#ccc'}`, paddingLeft: 10 }}>
          <strong style={{ color: GENE_COLORS[gene] || '#333' }}>{gene}:</strong> <span style={{ color: '#444' }}>{proto}</span>
        </div>
      ))}
    </div>
  );
}

export default function HreditaryIonChannelDiseaseAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const base = `${API}/api/${SLUG}`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '16px 24px', maxWidth: 1400, margin: '0 auto' }}>
      <h1 style={{ color: '#1565c0', marginBottom: 4 }}>🧬 Hereditary Ion Channel Disease Atlas</h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Hereditary Channelopathy Reference — KCNA1 · KCNQ1 · KCNH2 · SCN5A · RYR1 · CACNA1S · CLCN1 · KCNJ2
        (320 patients, seeds 2030-2037)
      </p>
      {err && <ErrBox msg={err} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1565c0' : '#e3f2fd',
              color: tab === t ? '#fff' : '#1565c0',
              fontWeight: tab === t ? 700 : 400,
            }}
          >{t}</button>
        ))}
      </div>

      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
