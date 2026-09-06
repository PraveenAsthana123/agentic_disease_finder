'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  SCN1A: '#880e4f',  // deep magenta — Dravet, SUDEP highest risk
  SCN2A: '#1a237e',  // deep indigo — age-onset pharmacogenomic switch
  KCNQ2: '#1b5e20',  // deep green — M-current neonatal window
  CDKL5: '#4a148c',  // deep purple — CDD, ganaxolone
  PCDH19:'#e65100',  // deep orange — cellular interference female epilepsy
  SCN8A: '#006064',  // deep teal — GOF CBZ-effective opposite SCN1A
  KCNT1: '#37474f',  // dark slate — EIMFS quinidine
  SLC6A1:'#bf360c',  // deep burnt orange — MAE drop attacks helmet
};

const GENE_DISEASE = {
  SCN1A: 'Dravet Syndrome AD — AVOID CBZ/LTG/PHT — Stiripentol/Cannabidiol/Fenfluramine — SUDEP Highest Risk — Fever Protocol',
  SCN2A: 'GOF Early <3m → CBZ EFFECTIVE; LOF Late >3m → CBZ HARMFUL — Age-Onset Pharmacogenomic Switch — ASD Risk LOF',
  KCNQ2: 'BFNE / KCNQ2-DEE — M-Current Controller — CBZ Neonatal Window — Burst-Suppression EEG — SNHL Screen',
  CDKL5: 'CDD XLD — Onset <5m — Ganaxolone FDA-2022 — Ketogenic Diet — NOT MECP2-Rett — Cortical Visual Impairment 60%',
  PCDH19:'EFMR XLD — Females AFFECTED Carrier Males UNAFFECTED — Mosaic Males CAN be Affected — Clustering Febrile — AVOID LEV',
  SCN8A: 'SCN8A-DEE GOF — Phenytoin/CBZ HIGHLY EFFECTIVE — OPPOSITE SCN1A — Asn1768Asp Hotspot — LOF Movement Disorder',
  KCNT1: 'EIMFS / ADNFLE — GOF Excess KNa Current — Quinidine Trial GOF — QTc MANDATORY — Refractory Standard ASM',
  SLC6A1:'MAE / Doose Syndrome — AVOID CBZ/LTG/VGB/Tiagabine — Valproate + KD — Drop Attack Helmet MANDATORY',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|HIGHEST|ALERT|EFFECTIVE|OPPOSITE|WINDOW/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|HOTSPOT|TRIAL|QTc/i.test(text);
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

  const statRows = [
    ['SCN1A — Dravet confirmed', `${s.scn1a_dravet_confirmed_pct}%`],
    ['SCN1A — Na-channel blocker prescribed erroneously', `${s.scn1a_na_blocker_erroneous_pct}%`],
    ['SCN1A — rescue midazolam prescribed', `${s.scn1a_rescue_midazolam_pct}%`],
    ['SCN1A — fever protocol in place', `${s.scn1a_fever_protocol_pct}%`],
    ['SCN1A — SUDEP risk counselled', `${s.scn1a_sudep_counselled_pct}%`],
    ['SCN1A — stiripentol prescribed', `${s.scn1a_stiripentol_pct}%`],
    ['SCN1A — cannabidiol (Epidiolex) prescribed', `${s.scn1a_cannabidiol_pct}%`],
    ['SCN1A — status epilepticus history', `${s.scn1a_status_history_pct}%`],
    ['SCN2A — GOF phenotype (onset <3m)', `${s.scn2a_gof_pct}%`],
    ['SCN2A — LOF phenotype (onset >3m)', `${s.scn2a_lof_pct}%`],
    ['SCN2A — CBZ/PHT good response (GOF)', `${s.scn2a_cbz_good_response_pct}%`],
    ['SCN2A — CBZ/PHT worsened (LOF)', `${s.scn2a_cbz_worsened_pct}%`],
    ['SCN2A — ASD comorbidity', `${s.scn2a_asd_pct}%`],
    ['KCNQ2 — DEE phenotype', `${s.kcnq2_dee_pct}%`],
    ['KCNQ2 — burst-suppression EEG neonatal', `${s.kcnq2_burst_suppression_pct}%`],
    ['KCNQ2 — CBZ/PHT started early', `${s.kcnq2_cbz_early_pct}%`],
    ['KCNQ2 — EEG improved with CBZ', `${s.kcnq2_eeg_improved_cbz_pct}%`],
    ['KCNQ2 — SNHL screened', `${s.kcnq2_snhl_screened_pct}%`],
    ['KCNQ2 — SNHL confirmed', `${s.kcnq2_snhl_confirmed_pct}%`],
    ['CDKL5 — ganaxolone prescribed', `${s.cdkl5_ganaxolone_pct}%`],
    ['CDKL5 — ketogenic diet tried', `${s.cdkl5_kd_tried_pct}%`],
    ['CDKL5 — ketogenic diet response', `${s.cdkl5_kd_response_pct}%`],
    ['CDKL5 — cortical visual impairment', `${s.cdkl5_cvi_pct}%`],
    ['CDKL5 — annual ophthalmology', `${s.cdkl5_ophtho_annual_pct}%`],
    ['CDKL5 — misdiagnosed as MECP2-Rett', `${s.cdkl5_misdiagnosed_rett_pct}%`],
    ['PCDH19 — clustering seizures', `${s.pcdh19_clustering_pct}%`],
    ['PCDH19 — febrile trigger', `${s.pcdh19_febrile_trigger_pct}%`],
    ['PCDH19 — clobazam cluster protocol', `${s.pcdh19_clobazam_protocol_pct}%`],
    ['PCDH19 — LEV worsening behaviour', `${s.pcdh19_lev_worsening_pct}%`],
    ['PCDH19 — psychiatric comorbidity', `${s.pcdh19_psychiatric_pct}%`],
    ['PCDH19 — remission post-puberty', `${s.pcdh19_remission_puberty_pct}%`],
    ['SCN8A — GOF phenotype', `${s.scn8a_gof_pct}%`],
    ['SCN8A — CBZ/PHT response (GOF)', `${s.scn8a_cbz_response_pct}%`],
    ['SCN8A — Asn1768Asp hotspot', `${s.scn8a_asnd1768_pct}%`],
    ['SCN8A — status epilepticus', `${s.scn8a_status_pct}%`],
    ['KCNT1 — EIMFS phenotype', `${s.kcnt1_eimfs_pct}%`],
    ['KCNT1 — quinidine trial', `${s.kcnt1_quinidine_trial_pct}%`],
    ['KCNT1 — quinidine QTc monitored', `${s.kcnt1_quinidine_qtc_pct}%`],
    ['KCNT1 — quinidine response', `${s.kcnt1_quinidine_response_pct}%`],
    ['SLC6A1 — MAE phenotype', `${s.slc6a1_mae_pct}%`],
    ['SLC6A1 — drop attacks', `${s.slc6a1_drop_attacks_pct}%`],
    ['SLC6A1 — helmet prescribed', `${s.slc6a1_helmet_pct}%`],
    ['SLC6A1 — valproate prescribed', `${s.slc6a1_valproate_pct}%`],
    ['SLC6A1 — Na-channel blocker erroneous', `${s.slc6a1_na_blocker_erroneous_pct}%`],
    ['SLC6A1 — ketogenic diet response', `${s.slc6a1_kd_response_pct}%`],
    ['Cross-gene — Na-channel blocker prescribed erroneously', `${s.any_na_blocker_erroneous_pct}%`],
    ['Cross-gene — status epilepticus history', `${s.any_status_epilepticus_pct}%`],
    ['Cross-gene — ketogenic diet tried', `${s.any_kd_tried_pct}%`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>{data.title}</h2>
      <p style={{ color: '#555', marginBottom: 16 }}>{data.subtitle}</p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {genes.map(g => (
          <div key={g.gene} style={{
            background: GENE_COLORS[g.gene], color: '#fff', borderRadius: 8,
            padding: '10px 16px', minWidth: 120,
          }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.locus} · {g.aa}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.inheritance.split('—')[0].trim()}</div>
            <div style={{ fontSize: 11, opacity: 0.9, marginTop: 4 }}>{g.n_patients} pts</div>
          </div>
        ))}
      </div>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Cohort Statistics</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 24 }}>
        <tbody>
          {statRows.map(([label, val]) => (
            <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '5px 8px', color: '#333' }}>{label}</td>
              <td style={{ padding: '5px 8px', fontWeight: 600, color: '#1565c0', textAlign: 'right' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>
        Critical Alerts ({top_alerts.length})
      </h3>
      {top_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Per-Gene Breakdown</h2>
      {breakdown.map(g => (
        <div key={g.gene} style={{
          border: `2px solid ${GENE_COLORS[g.gene]}`,
          borderRadius: 10, marginBottom: 24, overflow: 'hidden',
        }}>
          <div style={{
            background: GENE_COLORS[g.gene], color: '#fff',
            padding: '10px 16px',
          }}>
            <span style={{ fontWeight: 700, fontSize: 16 }}>{g.gene}</span>
            <span style={{ marginLeft: 12, fontSize: 13 }}>{g.protein}</span>
          </div>
          <div style={{ padding: 16 }}>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Locus:</strong> {g.locus} · <strong>Size:</strong> {g.aa} ({g.kDa}) ·
              <strong> OMIM gene:</strong> {g.omim_gene} · <strong>Disease:</strong> {g.omim_disease}
            </p>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Inheritance:</strong> {g.inheritance}
            </p>
            <p style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>
              <strong>Mean onset:</strong> {g.mean_onset_months} mo ·
              <strong> Mean dx delay:</strong> {g.mean_dx_delay_months} mo ·
              <strong> M:</strong> {g.sex_distribution.M} / <strong>F:</strong> {g.sex_distribution.F}
            </p>
            <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>{g.alias.slice(0, 400)}…</p>
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#1565c0', marginBottom: 6 }}>
                Gene-Class Mechanistic Detail
              </summary>
              <p style={{ fontSize: 12, color: '#444', marginTop: 6 }}>{g.gene_class}</p>
            </details>
            <div style={{ marginTop: 10 }}>
              <strong style={{ fontSize: 12 }}>Aetiology distribution:</strong>
              {Object.entries(g.etiology_counts).map(([et, cnt]) => (
                <div key={et} style={{ fontSize: 11, color: '#555', marginTop: 2 }}>
                  {et}: <strong>{cnt}</strong>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10 }}>
              {g.key_alerts.map((a, i) => <AlertBadge key={i} text={a} />)}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  const rows = breakdown.map(g => ({
    gene: g.gene,
    locus: g.locus,
    aa: g.aa,
    inh: g.inheritance.split('—')[0].trim(),
    disease: GENE_DISEASE[g.gene],
    pts: g.n_patients,
    onset: `${g.mean_onset_months}m`,
  }));

  return (
    <div style={{ overflowX: 'auto' }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Clinical Atlas Summary</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Key Rule', 'Pts', 'Onset'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f8f8f8' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] }}>{r.gene}</td>
              <td style={{ padding: '7px 10px' }}>{r.locus}</td>
              <td style={{ padding: '7px 10px' }}>{r.aa}</td>
              <td style={{ padding: '7px 10px' }}>{r.inh}</td>
              <td style={{ padding: '7px 10px', fontSize: 11 }}>{r.disease}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.pts}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.onset}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginTop: 24, marginBottom: 12 }}>
        Precision Pharmacotherapy Matrix
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#880e4f', color: '#fff' }}>
            {['Gene', 'AVOID (worsen)', 'EFFECTIVE / Use', 'Special'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['SCN1A', 'Carbamazepine · Lamotrigine · Phenytoin', 'Valproate · Clobazam · Stiripentol · Cannabidiol · Fenfluramine', 'SUDEP: nighttime supervision, rescue midazolam'],
            ['SCN2A', 'LOF: Carbamazepine · Phenytoin', 'GOF: Carbamazepine · Phenytoin (highly effective)', 'Age <3m → GOF; Age >3m → LOF (pharmacogenomic switch)'],
            ['KCNQ2', 'Ezogabine (withdrawn)', 'CBZ / PHT — neonatal window; treat early', 'SNHL screen; burst-suppression → CBZ STAT'],
            ['CDKL5', 'Standard ASMs mostly ineffective', 'Ganaxolone (FDA 2022) · Ketogenic diet', 'mTOR trials; annual ophthalmology (CVI 60%)'],
            ['PCDH19', 'Levetiracetam (worsens behaviour)', 'Clobazam (clustering) · Stiripentol', 'Cluster protocol; mosaic male: skin biopsy'],
            ['SCN8A', 'LOF: Carbamazepine · Phenytoin', 'GOF: Phenytoin · Carbamazepine (OPPOSITE to SCN1A)', 'Asn1768Asp hotspot; SUDEP counselling'],
            ['KCNT1', 'Standard ASMs refractory', 'Quinidine (GOF trial) · Ketogenic diet', 'QTc monitoring mandatory; functional assay first'],
            ['SLC6A1', 'CBZ · LTG · Vigabatrin · Tiagabine', 'Valproate · Clobazam · Ketogenic diet', 'Drop attack helmet MANDATORY; AVOID GAT-1 inhibitors'],
          ].map(([gene, avoid, use, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#fff3f3' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
              <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 600 }}>{avoid}</td>
              <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: 600 }}>{use}</td>
              <td style={{ padding: '7px 10px', color: '#555' }}>{special}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const { definitions } = data;
  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Clinical Definitions</h2>
      {definitions.map((d, i) => (
        <div key={i} style={{
          border: '1px solid #e0e0e0', borderRadius: 8,
          marginBottom: 16, padding: 16,
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryEpilepsyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-epilepsy-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if (tab === 'Gene Table' && !breakdown) {
      fetch(`${API}/api/hereditary-epilepsy-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-epilepsy-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
    if (tab === 'Clinical Atlas' && !breakdown) {
      fetch(`${API}/api/hereditary-epilepsy-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#880e4f', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Epilepsy Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1486–1493
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Epilepsy-Atlas — Complete 8-Gene Monogenic Epilepsy Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        SCN1A · SCN2A · KCNQ2 · CDKL5 · PCDH19 · SCN8A · KCNT1 · SLC6A1
        — Precision pharmacotherapy, SUDEP risk, and treatment-critical contraindications
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#880e4f' : '#f0f0f0',
              color: tab === t ? '#fff' : '#333',
              fontWeight: tab === t ? 700 : 400,
              fontSize: 13,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
