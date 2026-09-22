'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cervical-cancer-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'STK11':  '#2e7d32',  // dark green    -- PJS; SCTAT PATHOGNOMONIC; adenoma malignum PATHOGNOMONIC
  'BRCA1':  '#880e4f',  // deep pink     -- HBOC; breast 72%; ovarian 44%; BSO 35-40
  'TP53':   '#b71c1c',  // deep red      -- LFS; AVOID RADIATION; surgery not chemoradiation
  'MSH2':   '#1b5e20',  // darkest green -- Lynch2; cervical adenocarcinoma; Muir-Torre PATHOGNOMONIC
  'FANCA':  '#e65100',  // deep orange   -- FA type A; cervical SCC 150-200x; HPV vaccination CRITICAL
  'PTEN':   '#4a148c',  // deep purple   -- Cowden; macrocephaly PATHOGNOMONIC; Lhermitte-Duclos PATHOGNOMONIC
  'ATM':    '#1a237e',  // deep navy     -- A-T; radiosensitivity ABSOLUTE; chemoradiation CI
  'BRCA2':  '#1565c0',  // deep blue     -- HBOC; cervical 2-3x; FA-D1 most severe
};

const GENE_INFO = {
  'STK11':  { full: 'PJS-Peutz-Jeghers / SCTAT-PATHOGNOMONIC / Adenoma-Malignum-PATHOGNOMONIC / Perioral-Pigmentation-PATHOGNOMONIC / Cervical-10-13% / Breast-45% / Pancreatic-130x',  locus: '19p13.3', size: '433 aa / 48 kDa',   inh: 'AD LOF' },
  'BRCA1':  { full: 'HBOC / BSO-Age-35-40-Standard-Care / Breast-72%-DOMINANT / Ovarian-44%-HIGHEST / Cervical-2-3x-Adenocarcinoma / Olaparib-PARP-FDA',                                   locus: '17q21.31', size: '1863 aa / 213 kDa', inh: 'AD LOF' },
  'TP53':   { full: 'LFS / AVOID-RADIATION-ABSOLUTELY / Cervical-Surgery-NOT-Chemoradiation / WBMRI-Toronto-Annually / Sarcoma-50-60%-PRIMARY / R337H-Brazilian-Founder',                  locus: '17p13.1',  size: '393 aa / 43 kDa',   inh: 'AD LOF' },
  'MSH2':   { full: 'Lynch2 / Cervical-Adenocarcinoma-NOT-Squamous-5-10% / Endometrial-40-60%-DOMINANT / Muir-Torre-Sebaceous-PATHOGNOMONIC / EPCAM-3prime / Pembrolizumab',                locus: '2p21',     size: '934 aa / 105 kDa',  inh: 'AD LOF' },
  'FANCA':  { full: 'FA-Type-A-60% / Cervical-SCC-150-200x / HPV-Vaccination-CRITICAL-MANDATORY / AVOID-ALDEHYDE-ABSOLUTELY / DEB-Test-PATHOGNOMONIC / RT-ABSOLUTE-CI',                    locus: '16q24.3',  size: '1455 aa / 163 kDa', inh: 'AR LOF' },
  'PTEN':   { full: 'Cowden-PHTS / Macrocephaly-PATHOGNOMONIC / Lhermitte-Duclos-PATHOGNOMONIC / Trichilemmoma-PATHOGNOMONIC / Endometrial-28-44%-DOMINANT / Everolimus-mTOR',              locus: '10q23.31', size: '403 aa / 47 kDa',   inh: 'AD LOF' },
  'ATM':    { full: 'A-T-Biallelic / RADIOSENSITIVITY-ABSOLUTE / Cervical-Chemoradiation-ABSOLUTE-CI / Surgery-Mandatory-Cervical / Monoallelic-Cervical-2-3x / Ceralasertib-ATRi',        locus: '11q22.3',  size: '3056 aa / 350 kDa', inh: 'AD LOF / AR biallelic' },
  'BRCA2':  { full: 'HBOC / Cervical-2-3x-Adenocarcinoma / FA-D1-Biallelic-MOST-SEVERE / BSO-40-45yr / Olaparib-PARP / Prostate-8x-HIGHEST / 6174delT-Ashkenazi',                          locus: '13q12.3',  size: '3418 aa / 384 kDa', inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#1565c0', color: '#fff', borderRadius: 4,
      padding: '2px 8px', fontSize: 11, fontWeight: 700, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneBar({ gene, pct, color }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color }}>{gene}</span>
        <span style={{ color: '#333' }}>{pct}%</span>
      </div>
      <div style={{ background: '#e0e0e0', borderRadius: 4, height: 14 }}>
        <div style={{ background: color, width: `${pct}%`, height: '100%', borderRadius: 4, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  );
}

export default function HereditoryCervicalCancerAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#2e7d32' }}>Loading Hereditary Cervical Cancer Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f9f9f9', minHeight: '100vh', padding: 0 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#2e7d32 0%,#1b5e20 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, textTransform: 'uppercase' }}>Hereditary Cancer Predisposition Atlas</div>
        <h1 style={{ margin: '6px 0 4px', fontSize: 26, fontWeight: 800 }}>Hereditary Cervical Cancer Predisposition Atlas</h1>
        <div style={{ fontSize: 14, opacity: 0.9 }}>
          Complete 8-Gene Reference · STK11 · BRCA1 · TP53 · MSH2 · FANCA · PTEN · ATM · BRCA2 · Seeds 3222–3229 · 320 patients
        </div>
        <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>
          STK11-PJS-SCTAT-PATHOGNOMONIC · Adenoma-Malignum-PATHOGNOMONIC · FANCA-Cervical-150-200x-HPV-Vaccination-CRITICAL · TP53-Surgery-NOT-Chemoradiation · ATM-RT-ABSOLUTE-CI
        </div>
      </div>

      {/* Tabs */}
      <div style={{ background: '#fff', borderBottom: '2px solid #2e7d32', display: 'flex', paddingLeft: 32 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400, fontSize: 14,
            color: tab === t ? '#2e7d32' : '#555',
            borderBottom: tab === t ? '3px solid #2e7d32' : '3px solid transparent',
            marginBottom: -2,
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '28px 32px' }}>
        {/* OVERVIEW TAB */}
        {tab === 'Overview' && overview && (
          <div>
            {/* KPI cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(180px,1fr))', gap: 16, marginBottom: 28 }}>
              {[
                { label: 'Total Patients',      value: overview.total_patients,    color: '#2e7d32' },
                { label: 'Genes Covered',        value: overview.genes_n,           color: '#1b5e20' },
                { label: 'Severe Events',        value: `${overview.severe_total_pct}%`, color: '#b71c1c' },
                { label: 'Highest-Risk Gene',    value: overview.highest_risk_gene, color: '#e65100' },
                { label: 'Seed Range',           value: overview.seed_range,        color: '#4a148c' },
              ].map(kpi => (
                <div key={kpi.label} style={{ background: '#fff', border: `2px solid ${kpi.color}`, borderRadius: 8, padding: '16px 20px', textAlign: 'center' }}>
                  <div style={{ fontSize: 24, fontWeight: 800, color: kpi.color }}>{kpi.value}</div>
                  <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{kpi.label}</div>
                </div>
              ))}
            </div>

            {/* Severe % bar chart */}
            <div style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 8, padding: 20, marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 16px', color: '#2e7d32', fontSize: 16 }}>Severe Event Rate by Gene</h3>
              {overview.gene_summary.map(r => (
                <GeneBar key={r.gene} gene={r.gene} pct={r.severe_pct} color={GENE_COLORS[r.gene] || '#2e7d32'} />
              ))}
            </div>

            {/* Gene info cards */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 16 }}>
              {overview.gene_summary.map(r => {
                const info = GENE_INFO[r.gene] || {};
                return (
                  <div key={r.gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[r.gene]}`, borderRadius: 8, padding: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 10 }}>
                      <span style={{ background: GENE_COLORS[r.gene], color: '#fff', borderRadius: 4, padding: '3px 10px', fontWeight: 800, fontSize: 15, marginRight: 10 }}>{r.gene}</span>
                      <span style={{ fontSize: 12, color: '#555' }}>{info.locus} · {info.size} · {info.inh}</span>
                    </div>
                    <div style={{ fontSize: 11, color: '#666', marginBottom: 8, lineHeight: 1.5 }}>{info.full}</div>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
                      <span style={{ fontSize: 12, color: '#333' }}>n={r.n}</span>
                      <span style={{ fontSize: 12, color: '#b71c1c', fontWeight: 700 }}>severe={r.severe_pct}%</span>
                      <span style={{ fontSize: 12, color: '#555' }}>mean age={r.mean_age_onset}yr</span>
                    </div>
                    <div style={{ fontSize: 11, color: '#444', borderTop: '1px solid #eee', paddingTop: 8 }}>
                      <b>Pathognomonic:</b> {r.pathognomonic}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && overview && (
          <div style={{ background: '#fff', borderRadius: 8, border: '1px solid #ddd', overflow: 'hidden' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#2e7d32', color: '#fff' }}>
                  {['Gene', 'Locus', 'Size', 'Inheritance', 'n', 'Severe %', 'Mean Age', 'Key Cancer Risk', 'Surveillance Key'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 700, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {overview.gene_summary.map((r, i) => {
                  const info = GENE_INFO[r.gene] || {};
                  return (
                    <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f9fff9' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                      <td style={{ padding: '8px 12px', fontWeight: 800, color: GENE_COLORS[r.gene] }}>{r.gene}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{info.locus || r.locus}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12, whiteSpace: 'nowrap' }}>{info.size || '—'}</td>
                      <td style={{ padding: '8px 12px', fontSize: 12 }}>{info.inh || '—'}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{r.n}</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center', color: '#b71c1c', fontWeight: 700 }}>{r.severe_pct}%</td>
                      <td style={{ padding: '8px 12px', textAlign: 'center' }}>{r.mean_age_onset}yr</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, maxWidth: 240 }}>{r.cancer_risk}</td>
                      <td style={{ padding: '8px 12px', fontSize: 11, maxWidth: 260 }}>{r.surveillance_key}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <h2 style={{ color: '#2e7d32', marginBottom: 20, fontSize: 20 }}>Clinical Atlas — Per-Gene Breakdown</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(360px,1fr))', gap: 20 }}>
              {breakdown.breakdown.map(g => (
                <div key={g.gene} style={{ background: '#fff', border: `2px solid ${GENE_COLORS[g.gene]}`, borderRadius: 8, padding: 18 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                    <span style={{ background: GENE_COLORS[g.gene], color: '#fff', borderRadius: 4, padding: '4px 12px', fontWeight: 800, fontSize: 16 }}>{g.gene}</span>
                    <span style={{ fontSize: 12, color: '#555', textAlign: 'right' }}>{g.locus} · seed {g.seed}</span>
                  </div>
                  <div style={{ display: 'flex', gap: 16, marginBottom: 12 }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 22, fontWeight: 800, color: GENE_COLORS[g.gene] }}>{g.n}</div>
                      <div style={{ fontSize: 11, color: '#555' }}>patients</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 22, fontWeight: 800, color: '#b71c1c' }}>{g.severe_pct}%</div>
                      <div style={{ fontSize: 11, color: '#555' }}>severe</div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 22, fontWeight: 800, color: '#555' }}>{g.mean_age_onset}yr</div>
                      <div style={{ fontSize: 11, color: '#555' }}>mean age</div>
                    </div>
                  </div>
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#333', marginBottom: 4 }}>Key Distinctions:</div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                      {(g.key_distinctions || []).map(d => (
                        <Badge key={d} text={d} color={GENE_COLORS[g.gene]} />
                      ))}
                    </div>
                  </div>
                  <div style={{ fontSize: 11, color: '#444', borderTop: '1px solid #eee', paddingTop: 8 }}>
                    <b>Surveillance:</b> {g.surveillance_key}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            <h2 style={{ color: '#2e7d32', marginBottom: 20, fontSize: 20 }}>Clinical Definitions — Hereditary Cervical Cancer Predisposition Atlas</h2>
            {definitions.definitions.map((d, i) => (
              <div key={i} style={{ background: '#fff', border: '1px solid #ddd', borderRadius: 8, padding: 20, marginBottom: 16 }}>
                <div style={{ fontWeight: 800, fontSize: 14, color: '#2e7d32', marginBottom: 10, borderBottom: '2px solid #2e7d32', paddingBottom: 6 }}>
                  {d.term}
                </div>
                <pre style={{ margin: 0, fontFamily: 'system-ui,sans-serif', fontSize: 13, lineHeight: 1.7, color: '#333', whiteSpace: 'pre-wrap' }}>
                  {d.definition}
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
