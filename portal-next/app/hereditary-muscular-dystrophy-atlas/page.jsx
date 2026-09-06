'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  DMD:   '#0d47a1',  // deep blue — most common, Duchenne
  DYSF:  '#1b5e20',  // deep green — dysferlinopathy
  CAPN3: '#bf360c',  // deep burnt orange — most common AR LGMD
  LMNA:  '#4a148c',  // deep purple — laminopathy, ICD critical
  EMD:   '#880e4f',  // deep magenta — X-linked EDMD1
  SGCA:  '#006064',  // deep teal — sarcoglycanopathy
  DMPK:  '#e65100',  // deep orange — DM1, anaesthesia risk
  CNBP:  '#37474f',  // dark slate — DM2
};

const GENE_DISEASE = {
  DMD:   'Duchenne/Becker MD XLR — Out-of-Frame=DMD In-Frame=BMD — SUCCINYLCHOLINE ABSOLUTE-CI — Exon-Skip Mutation-Specific — Glucocorticoids Mandatory',
  DYSF:  'LGMD2B/Miyoshi AR — Dysferlin Membrane Repair — Western Blot DIAGNOSTIC — AVOID Statins — NOT Polymyositis — CK 5000-100000',
  CAPN3: 'LGMD2A AR — Most Common AR LGMD — Calpainopathy — Scapular Winging — NO Calf Hypertrophy — Arg490Gln Basque Founder',
  LMNA:  'EDMD2/LMNA-DCM AD — AV Block LETHAL — ICD Mandatory When Arrhythmia — Annual Holter From Diagnosis — Elbow Contractures First',
  EMD:   'EDMD1 XLR — Emerin Absent Immunostaining DIAGNOSTIC — ICD/Pacemaker Mandatory — Female Carrier Cardiac Surveillance',
  SGCA:  'LGMD2D AR — Alpha-Sarcoglycanopathy — TEST ALL 4 Sarcoglycans — Secondary Loss All 4 — Calf Hypertrophy — Arg77Cys Founder',
  DMPK:  'DM1 AD — CTG Repeat — Anticipation — ANAESTHESIA ABSOLUTE RISK — Propofol TIVA — AVOID Succinylcholine — Mexiletine Myotonia',
  CNBP:  'DM2 AD — CCTG Repeat — Proximal>Distal — NO Congenital Form — Repeat-Primed PCR Required — Proximal Pain — Mexiletine',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|PATHOGNOMONIC|ALERT|CURATIVE|NO-ANTICOAG|LETHAL|AVOID/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|MANDATORY|STAT|SIMULTANEOUSLY|FIRST|HOTSPOT|FOUNDER/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#0d47a1', marginBottom: 12 }}>Aggregate — 320 Patients (8×40, seeds 1478–1485)</h3>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
          <tbody>
            {[
              ['DMD — Succinylcholine CI (ALL)', s.dmd_succcinyl_ci_pct, '%'],
              ['DMD — Glucocorticoids Prescribed', s.dmd_glucocorticoids_pct, '%'],
              ['DMD — Exon-Skip Eligible', s.dmd_exon_skip_eligible_pct, '%'],
              ['DMD — Cardiac ACE-i Initiated', s.dmd_cardiac_ace_i_pct, '%'],
              ['DMD — NIV Started', s.dmd_niv_pct, '%'],
              ['DMD — Still Ambulatory', s.dmd_ambulatory_pct, '%'],
              ['DYSF — Western Blot Absent (DIAGNOSTIC)', s.dysf_western_blot_absent_pct, '%'],
              ['DYSF — Statin Prescribed Erroneously', s.dysf_statin_prescribed_erroneously_pct, '%'],
              ['DYSF — Polymyositis Misdiagnosis', s.dysf_pm_misdiagnosis_pct, '%'],
              ['CAPN3 — Western Blot Reduced', s.capn3_western_blot_reduced_pct, '%'],
              ['CAPN3 — Scapular Winging', s.capn3_scapular_winging_pct, '%'],
              ['CAPN3 — Cardiac Spared', s.capn3_cardiac_spared_pct, '%'],
              ['LMNA — AV Block Present', s.lmna_av_block_pct, '%'],
              ['LMNA — ICD Implanted', s.lmna_icd_implanted_pct, '%'],
              ['LMNA — LVEF Reduced', s.lmna_lvef_reduced_pct, '%'],
              ['LMNA — Elbow Contractures', s.lmna_elbow_contractures_pct, '%'],
              ['LMNA — Annual Holter Performed', s.lmna_annual_holter_pct, '%'],
              ['EMD — Emerin Absent (Immunostaining)', s.emd_emerin_absent_pct, '%'],
              ['EMD — AV Block Present', s.emd_av_block_pct, '%'],
              ['EMD — Pacemaker/ICD Implanted', s.emd_pacemaker_icd_pct, '%'],
              ['EMD — Elbow Contractures', s.emd_elbow_contractures_pct, '%'],
              ['SGCA — All 4 Sarcoglycans Tested', s.sgca_all_4_tested_pct, '%'],
              ['SGCA — Calf Pseudo-Hypertrophy', s.sgca_calf_hypertrophy_pct, '%'],
              ['SGCA — Arg77Cys Founder', s.sgca_founder_Arg77Cys_pct, '%'],
              ['DMPK — Myotonia (Clinical)', s.dmpk_myotonia_pct, '%'],
              ['DMPK — Anaesthesia Flagged', s.dmpk_anaesthesia_flagged_pct, '%'],
              ['DMPK — Mexiletine Prescribed', s.dmpk_mexiletine_pct, '%'],
              ['DMPK — Annual Holter', s.dmpk_annual_holter_pct, '%'],
              ['DMPK — Cataracts (DM1)', s.dmpk_cataracts_pct, '%'],
              ['CNBP — Proximal Pain (PROMM)', s.cnbp_proximal_pain_pct, '%'],
              ['CNBP — Fibromyalgia Misdiagnosis', s.cnbp_misdiagnosed_fibromyalgia_pct, '%'],
              ['CNBP — Repeat-Primed PCR Used', s.cnbp_repeat_sized_correctly_pct, '%'],
              ['Any Gene — Succcinyl CI / DM1 Anaesthesia Risk', s.any_succcinyl_ci_pct, '%'],
              ['Any Gene — ICD/Pacemaker Implanted', s.any_icd_pacemaker_pct, '%'],
              ['Any Gene — Elbow/Joint Contractures', s.any_contractures_pct, '%'],
            ].map(([label, val, unit]) => (
              <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
                <td style={{ padding: '4px 8px', color: '#333' }}>{label}</td>
                <td style={{ padding: '4px 8px', fontWeight: 600, color: val >= 60 ? '#b71c1c' : val >= 30 ? '#e65100' : '#1b5e20', textAlign: 'right' }}>
                  {val !== undefined ? `${val}${unit}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ flex: '1 1 340px' }}>
        <h3 style={{ color: '#b71c1c', marginBottom: 12 }}>Critical Alerts — 64 Rules</h3>
        {top_alerts && top_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            {['Gene', 'Protein', 'aa / kDa', 'Locus', 'Inheritance', 'Phenotype', 'OMIM', 'n', 'Mean Onset', 'Key Alert #1'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {breakdown.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#000', whiteSpace: 'nowrap' }}>{g.gene}</td>
              <td style={{ padding: '6px 10px', maxWidth: 200 }}>{g.protein}</td>
              <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{g.aa} / {g.kDa}</td>
              <td style={{ padding: '6px 10px', whiteSpace: 'nowrap', fontFamily: 'monospace' }}>{g.locus}</td>
              <td style={{ padding: '6px 10px' }}>{g.inheritance}</td>
              <td style={{ padding: '6px 10px', maxWidth: 220, fontSize: 12 }}>{GENE_DISEASE[g.gene] || '—'}</td>
              <td style={{ padding: '6px 10px', whiteSpace: 'nowrap', fontSize: 12 }}>
                Gene: {g.omim_gene}<br />Dis: {g.omim_disease}
              </td>
              <td style={{ padding: '6px 10px', textAlign: 'center', fontWeight: 600 }}>{g.n_patients}</td>
              <td style={{ padding: '6px 10px', textAlign: 'center' }}>{g.mean_onset_y}y</td>
              <td style={{ padding: '6px 10px', fontSize: 12, color: '#b71c1c', maxWidth: 200 }}>
                {g.key_alerts && g.key_alerts[0]}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;
  const [sel, setSel] = useState(breakdown[0]?.gene || '');
  const gene = breakdown.find(g => g.gene === sel);

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 20 }}>
        {breakdown.map(g => (
          <button key={g.gene} onClick={() => setSel(g.gene)} style={{
            padding: '6px 14px', borderRadius: 6, border: 'none', cursor: 'pointer', fontWeight: 600,
            background: sel === g.gene ? (GENE_COLORS[g.gene] || '#1565c0') : '#e0e0e0',
            color: sel === g.gene ? '#fff' : '#333',
          }}>{g.gene}</button>
        ))}
      </div>
      {gene && (
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
          <div style={{ flex: '1 1 380px' }}>
            <h3 style={{ color: GENE_COLORS[gene.gene] || '#1565c0', marginBottom: 8 }}>
              {gene.gene} — {gene.protein}
            </h3>
            <div style={{ fontSize: 13, color: '#555', marginBottom: 12 }}>
              {gene.aa} · {gene.kDa} · {gene.locus} · {gene.inheritance}
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.7, color: '#333', marginBottom: 16, background: '#f5f5f5', padding: 12, borderRadius: 8 }}>
              {gene.gene_class}
            </div>
            <h4 style={{ marginBottom: 8 }}>Etiology Distribution</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 16 }}>
              <thead>
                <tr style={{ background: '#e3f2fd' }}>
                  <th style={{ padding: '4px 8px', textAlign: 'left' }}>Etiology</th>
                  <th style={{ padding: '4px 8px', textAlign: 'right' }}>n</th>
                </tr>
              </thead>
              <tbody>
                {gene.etiology_counts && Object.entries(gene.etiology_counts).map(([e, n]) => (
                  <tr key={e} style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '4px 8px' }}>{e}</td>
                    <td style={{ padding: '4px 8px', textAlign: 'right', fontWeight: 600 }}>{n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h4 style={{ marginBottom: 8 }}>Clinical Summary</h4>
            <div style={{ fontSize: 12, color: '#444', lineHeight: 1.6, background: '#fff8e1', padding: 10, borderRadius: 8 }}>
              {gene.alias}
            </div>
          </div>
          <div style={{ flex: '1 1 320px' }}>
            <h4 style={{ marginBottom: 8, color: '#b71c1c' }}>Clinical Alerts ({gene.key_alerts?.length})</h4>
            {gene.key_alerts && gene.key_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
            <h4 style={{ marginTop: 16, marginBottom: 8 }}>Patient Demographics (seed {gene.seed})</h4>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <tbody>
                {[
                  ['Patients', gene.n_patients],
                  ['Mean Onset', `${gene.mean_onset_y}y`],
                  ['Mean Dx Delay', `${gene.mean_dx_delay_y}y`],
                  ['Males', gene.sex_distribution?.M],
                  ['Females', gene.sex_distribution?.F],
                  ['OMIM Gene', gene.omim_gene],
                  ['OMIM Disease', gene.omim_disease],
                ].map(([k, v]) => (
                  <tr key={k} style={{ borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '4px 8px', color: '#666' }}>{k}</td>
                    <td style={{ padding: '4px 8px', fontWeight: 600 }}>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <h4 style={{ marginTop: 16, marginBottom: 8 }}>Per-Patient Sample (first 10)</h4>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#f0f0f0' }}>
                    {['PID', 'Sex', 'Onset', 'Current', 'Dx Delay', 'Etiology'].map(h => (
                      <th key={h} style={{ padding: '3px 6px', textAlign: 'left' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {gene.patients && gene.patients.slice(0, 10).map(p => (
                    <tr key={p.pid} style={{ borderBottom: '1px solid #f0f0f0' }}>
                      <td style={{ padding: '3px 6px', fontFamily: 'monospace' }}>{p.pid}</td>
                      <td style={{ padding: '3px 6px' }}>{p.sex}</td>
                      <td style={{ padding: '3px 6px' }}>{p.age_onset}y</td>
                      <td style={{ padding: '3px 6px' }}>{p.age_current}y</td>
                      <td style={{ padding: '3px 6px' }}>{p.dx_delay_y}y</td>
                      <td style={{ padding: '3px 6px', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{p.etiology}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { definitions } = data;
  return (
    <div>
      <h3 style={{ color: '#1565c0', marginBottom: 16 }}>Clinical Definitions — Hereditary Muscular Dystrophy Atlas</h3>
      {definitions && definitions.map((d, i) => (
        <div key={i} style={{
          background: i % 2 === 0 ? '#f9f9f9' : '#fff', borderRadius: 8,
          padding: '14px 18px', marginBottom: 12, border: '1px solid #e0e0e0',
        }}>
          <div style={{ fontWeight: 700, color: '#0d47a1', marginBottom: 6, fontSize: 15 }}>{d.term}</div>
          <div style={{ fontSize: 13, lineHeight: 1.7, color: '#333' }}>{d.definition}</div>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryMuscularDystrophyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-muscular-dystrophy-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' || tab === 'Clinical Atlas') {
      if (!breakdown) {
        fetch(`${API}/api/hereditary-muscular-dystrophy-atlas/breakdown`)
          .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
      }
    }
    if (tab === 'Definitions') {
      if (!definitions) {
        fetch(`${API}/api/hereditary-muscular-dystrophy-atlas/definitions`)
          .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
      }
    }
  }, [tab]);

  return (
    <div style={{ padding: '24px 32px', fontFamily: 'system-ui, sans-serif', maxWidth: 1400 }}>
      <h1 style={{ color: '#0d47a1', marginBottom: 4, fontSize: 22 }}>
        Hereditary-Muscular-Dystrophy-Atlas — Complete 8-Gene Hereditary Muscular Dystrophy Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 20, fontSize: 14 }}>
        <strong>DMD</strong> (Duchenne/Becker XLR — Succinylcholine ABSOLUTE-CI) ·{' '}
        <strong>DYSF</strong> (LGMD2B AR — Dysferlin Membrane Repair — AVOID Statins) ·{' '}
        <strong>CAPN3</strong> (LGMD2A AR — Most Common AR LGMD) ·{' '}
        <strong>LMNA</strong> (EDMD2 AD — AV Block LETHAL — ICD Mandatory) ·{' '}
        <strong>EMD</strong> (EDMD1 XLR — Emerin Absent Immunostaining DIAGNOSTIC) ·{' '}
        <strong>SGCA</strong> (LGMD2D AR — Test ALL 4 Sarcoglycans) ·{' '}
        <strong>DMPK</strong> (DM1 AD — Anaesthesia ABSOLUTE RISK — Anticipation) ·{' '}
        <strong>CNBP</strong> (DM2 AD — CCTG Repeat — Proximal Pain — Repeat-Primed PCR)
        <br />
        <span style={{ color: '#888' }}>320 patients · 8×40 · seeds 1478–1485</span>
      </p>

      {error && <div style={{ background: '#ffebee', color: '#c62828', padding: 12, borderRadius: 6, marginBottom: 16 }}>Error: {error}</div>}

      <div style={{ display: 'flex', gap: 8, marginBottom: 24, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#0d47a1' : '#e0e0e0',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
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
