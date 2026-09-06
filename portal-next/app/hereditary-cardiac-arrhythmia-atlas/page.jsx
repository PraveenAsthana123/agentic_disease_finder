'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  KCNQ1: '#1565c0',  // deep blue — LQT1, swimming trigger, 30–35% LQTS, beta-blockers 97%
  KCNH2: '#7b1fa2',  // deep purple — LQT2, hERG, >200 drug triggers, auditory arousal
  SCN5A: '#b71c1c',  // dark red — Nav1.5 dual: LQT3 (mexiletine) / Brugada (Na-blockers CI)
  CALM1: '#e65100',  // deep orange — calmodulinopathy, lethal neonatal, extreme QTc 600–650 ms
  RYR2:  '#2e7d32',  // dark green — CPVT1, bidirectional VT PATHOGNOMONIC, exercise restriction
  CASQ2: '#00695c',  // dark teal — CPVT2 AR, biallelic, 5% of CPVT, D307H Bedouin founder
  KCNJ2: '#4a148c',  // deep violet — Andersen-Tawil LQT7, triad PATHOGNOMONIC, quinidine
  HCN4:  '#37474f',  // dark slate — SSS2, If pacemaker channel, ivabradine CI, LVNC overlap
};

const GENE_DISEASE = {
  KCNQ1: 'LQT1 Romano-Ward AD / JLNS AR — IKs Kv7.1 Channel — 30–35% LQTS — Swimming CARDINAL Trigger — Nadolol 97% Efficacy — JLNS Biallelic Deafness + Extreme QTc',
  KCNH2: 'LQT2 AD — hERG/Kv11.1 IKr Channel — 25–30% LQTS — Auditory Arousal Triggers — >200 QT-Prolonging Drugs — Hypokalemia Synergistic — Beta-Blockers 38% Efficacy',
  SCN5A: 'LQT3 AD-GOF + Brugada Syndrome1 AD-LOF — Nav1.5 Cardiac Na Channel — Mexiletine LQT3 — Na-Channel-Blockers ABSOLUTE CI Brugada — Fever ABSOLUTE Emergency',
  CALM1: 'Calmodulinopathy AD de novo — LQT14 + CPVT Overlap — Mean QTc 600–650 ms — Lethal Perinatal — Flecainide Dual RYR2+IKr — Rarest Most Severe Arrhythmia Syndrome',
  RYR2:  'CPVT1 AD-GOF — Ryanodine Receptor 2 SR Ca Release — Bidirectional VT PATHOGNOMONIC — Exercise Restriction MANDATORY — Nadolol + Flecainide (RYR2 direct block)',
  CASQ2: 'CPVT2 AR Biallelic — Calsequestrin 2 SR Ca Buffer — 5% of CPVT — D307H Bedouin Founder — Same Treatment as CPVT1 — Carrier Parents Unaffected',
  KCNJ2: 'Andersen-Tawil Syndrome LQT7 AD — Kir2.1 IK1 Channel — Triad: Periodic Paralysis + BiVT + Facial Dysmorphia PATHOGNOMONIC — Quinidine VT — Acetazolamide Paralysis',
  HCN4:  'Sick Sinus Syndrome type 2 AD-LOF — HCN4 If Pacemaker Channel — Bradycardia + Chronotropic Incompetence — LV Non-Compaction Overlap — Ivabradine ABSOLUTE CI',
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

function AlertBadge({ text }) {
  const [key, ...rest] = text.split(': ');
  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: '0.6rem 0.9rem',
      borderLeft: '3px solid #f59e0b', marginBottom: 6, fontSize: 12,
    }}>
      <span style={{ color: '#fcd34d', fontWeight: 700 }}>{key}</span>
      {rest.length > 0 && <span style={{ color: '#cbd5e1' }}>: {rest.join(': ')}</span>}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const s = data.aggregate_stats || {};
  return (
    <div>
      <p style={{ color: '#94a3b8', marginBottom: '1.5rem', fontSize: 13 }}>{data.subtitle}</p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: '2rem' }}>
        <KPI label="Total Patients" value={data.total_patients} color="#6366f1" />
        <KPI label="Mean Dx Age (yr)" value={s.mean_dx_age} color="#06b6d4" />
        <KPI label="Mean Dx Delay (mo)" value={s.mean_dx_delay_months} color="#f59e0b" />
        <KPI label="Genes Covered" value={s.genes_covered} color="#1565c0" />
        <KPI label="LQT1 BB Efficacy" value="97%" color="#1565c0" />
        <KPI label="CALM1 QTc (ms)" value="600–650" color="#e65100" />
        <KPI label="CPVT Mortality" value="1–3%/yr" color="#2e7d32" />
        <KPI label="LQT2 Drug Triggers" value=">200" color="#7b1fa2" />
      </div>
      <h3 style={{ color: '#f1f5f9', marginBottom: '0.8rem', fontSize: 14 }}>Top Clinical Alerts</h3>
      {(data.top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
      <h3 style={{ color: '#f1f5f9', margin: '1.5rem 0 0.8rem', fontSize: 14 }}>Gene Summary</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Gene', 'Locus', 'aa / kDa', 'Inheritance', 'Syndrome', 'Mean Dx Age', 'N Patients'].map(h => (
                <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#94a3b8', fontWeight: 600 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data.genes || []).map((g, i) => (
              <tr key={g.gene} style={{ background: i % 2 === 0 ? '#1e293b' : '#0f172a' }}>
                <td style={{ padding: '6px 10px', color: GENE_COLORS[g.gene] || '#a5b4fc', fontWeight: 700 }}>{g.gene}</td>
                <td style={{ padding: '6px 10px', color: '#cbd5e1' }}>{g.locus}</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.aa} aa / {g.kDa} kDa</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.inheritance?.split(';')[0]}</td>
                <td style={{ padding: '6px 10px', color: '#fcd34d', fontWeight: 600, fontSize: 11 }}>
                  {(GENE_DISEASE[g.gene] || '').split(' — ')[0]}
                </td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.mean_dx_age} yr</td>
                <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.n_patients}</td>
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
  return (
    <div>
      {(data || []).map((g, i) => (
        <div key={g.gene} style={{
          background: '#1e293b', borderRadius: 10, marginBottom: 16, padding: '1.2rem',
          borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>{g.locus} · {g.aa} aa · {g.kDa} kDa · OMIM {g.omim_gene}</div>
              <div style={{ fontSize: 11, color: '#64748b', marginTop: 2 }}>{g.gene_class}</div>
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, color: '#cbd5e1', marginBottom: 4 }}>
                <strong style={{ color: GENE_COLORS[g.gene] || '#a5b4fc' }}>
                  {GENE_DISEASE[g.gene] || g.inheritance}
                </strong>
              </div>
              <div style={{ fontSize: 12, color: '#94a3b8' }}>Mean Dx Age: {g.computed?.mean_dx_age} yr · Seed: {g.computed?.seed}</div>
            </div>
          </div>
          <div style={{ marginTop: '0.8rem' }}>
            <div style={{ fontSize: 12, color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>Key Alerts</div>
            {(g.key_alerts || []).map((a, j) => <AlertBadge key={j} text={a} />)}
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      {(data || []).map((g, i) => (
        <div key={g.gene} style={{
          background: '#0f172a', borderRadius: 10, marginBottom: 20, padding: '1.4rem',
          border: `1px solid ${GENE_COLORS[g.gene] || '#334155'}30`,
        }}>
          <div style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc', marginBottom: 6 }}>
            {g.gene} — {g.omim_disease}
          </div>
          <div style={{ fontSize: 12, color: '#64748b', marginBottom: 10 }}>{g.locus} · {g.aa} aa · {g.gene_class}</div>
          <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.65, marginBottom: 12 }}>
            {(g.alias || '').slice(0, 1200)}{g.alias?.length > 1200 ? '…' : ''}
          </div>
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 12, color: '#f1f5f9', fontWeight: 600, marginBottom: 4 }}>Molecular Etiologies</div>
            {(g.etiologies || []).map((e, j) => (
              <div key={j} style={{ fontSize: 12, color: '#94a3b8', padding: '3px 0 3px 10px', borderLeft: '2px solid #334155' }}>
                {e}
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: '#64748b', fontStyle: 'italic' }}>
            Dx Delay Distribution: {g.dx_delay_distribution}
          </div>
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#f1f5f9', fontSize: 14, marginBottom: '1rem' }}>Clinical Concepts</h3>
      {Object.entries(data.concepts || {}).map(([title, body], i) => (
        <div key={i} style={{
          background: '#1e293b', borderRadius: 8, marginBottom: 14,
          padding: '1rem', borderLeft: '3px solid #6366f1',
        }}>
          <div style={{ color: '#a5b4fc', fontWeight: 700, fontSize: 13, marginBottom: 8 }}>{title}</div>
          <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.65 }}>{body}</div>
        </div>
      ))}
      <h3 style={{ color: '#f1f5f9', fontSize: 14, margin: '1.5rem 0 0.8rem' }}>Pharmacological Distinctions</h3>
      {(data.pharmacological_distinctions || []).map((d, i) => {
        const [first, ...rest] = d.split(':');
        return (
          <div key={i} style={{
            background: '#0f172a', borderRadius: 8, marginBottom: 10,
            padding: '0.8rem', borderLeft: '3px solid #06b6d4',
          }}>
            <div style={{ color: '#67e8f9', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{first}</div>
            <div style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{rest.join(':')}</div>
          </div>
        );
      })}
      <h3 style={{ color: '#f1f5f9', fontSize: 14, margin: '1.5rem 0 0.8rem' }}>Key Standards</h3>
      {(data.key_standards || []).map((s, i) => (
        <div key={i} style={{
          background: '#1e293b', borderRadius: 8, marginBottom: 8,
          padding: '0.7rem', borderLeft: '3px solid #10b981',
          fontSize: 12, color: '#94a3b8', lineHeight: 1.6,
        }}>
          {s}
        </div>
      ))}
    </div>
  );
}

export default function HereditaryCardiacArrhythmiaAtlas() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/hereditary-cardiac-arrhythmia-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-cardiac-arrhythmia-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-cardiac-arrhythmia-atlas/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#f1f5f9', fontFamily: 'monospace', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        <h1 style={{ fontSize: 22, fontWeight: 800, color: '#a5b4fc', marginBottom: 4 }}>
          &#x1f9ec; Hereditary-Cardiac-Arrhythmia-Atlas
        </h1>
        <p style={{ fontSize: 12, color: '#64748b', marginBottom: '1.5rem' }}>
          Complete 8-Gene Hereditary Cardiac Arrhythmia Atlas · KCNQ1 · KCNH2 · SCN5A · CALM1 · RYR2 · CASQ2 · KCNJ2 · HCN4 · 320 patients (8×40, seeds 1662–1669)
        </p>
        {error && <ErrorBox msg={error} />}
        <div style={{ display: 'flex', gap: 8, marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setTab(i)} style={{
              padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13,
              background: tab === i ? '#6366f1' : '#1e293b',
              color: tab === i ? '#fff' : '#94a3b8',
              fontWeight: tab === i ? 700 : 400,
            }}>{t}</button>
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
