'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-haematological-malignancy-predisposition-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  RUNX1:  '#1565c0',  // deep blue   — FPD-AML sibling donor mandatory
  CEBPA:  '#2e7d32',  // deep green  — Familial AML ELN favourable
  DDX41:  '#6a1b9a',  // deep purple — Most prevalent germline MDS/AML adults
  TP53:   '#b71c1c',  // deep red    — Li-Fraumeni venetoclax poor
  ETV6:   '#e65100',  // deep orange — Thrombocytopenia-5 ALL risk ITP mimic
  ANKRD26:'#004d40',  // deep teal   — 5'UTR WES misses targeted mandatory
  SAMD9L: '#880e4f',  // deep magenta— Ataxia-Pancytopenia monosomy 7 reversion
  NF1:    '#37474f',  // dark slate  — JMML trametinib selumetinib
};

const GENE_INFO = {
  RUNX1:   { aa: 453,  locus: '21q22.12', inh: 'AD',     disease: 'FPD-AML-Familial-Platelet-Disorder — 30-44pct-Lifetime-AML-MDS — Dense-Granule-Defect-PATHOGNOMONIC — Sibling-Donor-Testing-MANDATORY-Before-HSCT' },
  CEBPA:   { aa: 358,  locus: '19q13.11', inh: 'AD+biallelic', disease: 'Familial-AML-Germline-N-terminal-Frameshift — ELN-Favourable-Biallelic — CD19-Positive-Blasts-DISTINCTIVE — HSCT-in-CR1-CONTROVERSIAL' },
  DDX41:   { aa: 622,  locus: '5q35.3',   inh: 'AD',     disease: 'Most-Prevalent-Germline-MDS-AML-Adults-3-4pct — Splice-Variants-c.1574+1GA — Somatic-R525H-Second-Hit-PATHOGNOMONIC — Late-Onset-65yr' },
  TP53:    { aa: 393,  locus: '17p13.1',  inh: 'AD',     disease: 'Li-Fraumeni-Syndrome — Therapy-Related-AML-Complex-Karyotype — Venetoclax-POOR-Response — AVOID-Radiation — APR-246-Investigational' },
  ETV6:    { aa: 452,  locus: '12p13.2',  inh: 'AD',     disease: 'Thrombocytopenia-5-THRO5 — 30pct-Lifetime-B-ALL-Risk — ITP-Mimic-Steroid-Unresponsive — ETS-Domain-P214L-R358X-Hotspot' },
  ANKRD26: { aa: 1710, locus: '10p12.1',  inh: 'AD',     disease: 'Thrombocytopenia-2-THRO2 — 5prime-UTR-Standard-WES-MISSES — Targeted-Sequencing-MANDATORY — 5-8pct-MDS-AML-Risk' },
  SAMD9L:  { aa: 1589, locus: '7q21.2',   inh: 'AD GOF', disease: 'Ataxia-Pancytopenia-ATXPC — Monosomy-7-PARADOXICALLY-FAVOURABLE-Reversion — Revertant-Mosaicism-Common — Do-NOT-Rush-HSCT' },
  NF1:     { aa: 2839, locus: '17q11.2',  inh: 'AD',     disease: 'NF1-JMML-30pct-Germline — RAS-GAP-Loss — HSCT-Only-Curative — Trametinib-JMML-Investigational — Selumetinib-FDA-2020-Plexiform' },
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
      <h2 style={{ color: '#1565c0' }}>Hereditary-Haematological-Malignancy-Predisposition-Atlas</h2>
      <p style={{ color: '#444', marginBottom: 16 }}>{ov.subtitle}</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Patients" value={ov.total_patients} color="#1565c0" />
        <StatCard label="Genes" value={ov.genes?.length} color="#2e7d32" />
        <StatCard label="Seeds" value={ov.seeds} color="#37474f" />
        <StatCard label="Thrombocytopenia" value={ov.thrombocytopenia_patients} color="#6a1b9a" />
        <StatCard label="AML / MDS" value={ov.aml_mds_patients} color="#b71c1c" />
        <StatCard label="ALL Risk" value={ov.all_risk_patients} color="#e65100" />
        <StatCard label="JMML" value={ov.jmml_patients} color="#004d40" />
        <StatCard label="Monosomy 7" value={ov.monosomy7_patients} color="#880e4f" />
        <StatCard label="Revertant Mosaicism" value={ov.revertant_mosaicism_patients} color="#880e4f" />
        <StatCard label="Venetoclax Poor" value={ov.venetoclax_poor_patients} color="#b71c1c" />
        <StatCard label="HSCT Required" value={ov.hsct_required_patients} color="#1565c0" />
        <StatCard label="Sibling Donor Risk" value={ov.sibling_donor_risk_patients} color="#c62828" />
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
      <h3>Gene Table (8 Genes)</h3>
      <input
        placeholder="Filter by gene, name, or locus…"
        value={filter} onChange={e => setFilter(e.target.value)}
        style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ccc', marginBottom: 16, width: 320 }}
      />
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            {['Gene','aa','Locus','Inheritance','Disease / Key Discriminator',
              'AML/MDS%','Thrombocytopenia%','HSCT%','Sibling Donor%','Venetoclax Poor%'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filtered.map((g, i) => (
            <tr key={g.gene} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
              <td style={{ padding: '7px 10px' }}>{GENE_INFO[g.gene]?.aa}</td>
              <td style={{ padding: '7px 10px' }}>{g.locus}</td>
              <td style={{ padding: '7px 10px' }}>{GENE_INFO[g.gene]?.inh}</td>
              <td style={{ padding: '7px 10px', fontSize: 12 }}>{GENE_INFO[g.gene]?.disease}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.aml_mds_pct}%</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.thrombocytopenia_pct}%</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.hsct_required_pct}%</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.sibling_donor_risk_pct}%</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{g.venetoclax_poor_pct}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.values(data);
  const [selected, setSelected] = useState(genes[0]?.gene || '');
  const gene = genes.find(g => g.gene === selected);

  return (
    <div style={{ display: 'flex', gap: 20 }}>
      <div style={{ minWidth: 160 }}>
        <div style={{ fontWeight: 700, marginBottom: 10, color: '#1565c0' }}>Gene</div>
        {genes.map(g => (
          <button key={g.gene} onClick={() => setSelected(g.gene)}
            style={{
              display: 'block', width: '100%', textAlign: 'left', padding: '8px 12px',
              marginBottom: 4, borderRadius: 6, border: 'none', cursor: 'pointer',
              background: selected === g.gene ? (GENE_COLORS[g.gene] || '#1565c0') : '#eee',
              color: selected === g.gene ? '#fff' : '#333', fontWeight: selected === g.gene ? 700 : 400,
            }}>
            {g.gene}
          </button>
        ))}
      </div>

      {gene && (
        <div style={{ flex: 1 }}>
          <h3 style={{ color: GENE_COLORS[gene.gene] || '#1565c0' }}>{gene.alt_name}</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginBottom: 16 }}>
            {[
              { l: 'Locus', v: gene.locus },
              { l: 'Protein', v: `${GENE_INFO[gene.gene]?.aa} aa` },
              { l: 'Inheritance', v: gene.inheritance?.split('—')[0] },
              { l: 'N patients', v: gene.n_patients },
              { l: 'AML/MDS', v: `${gene.aml_mds_pct}%` },
              { l: 'Thrombocytopenia', v: `${gene.thrombocytopenia_pct}%` },
              { l: 'HSCT Required', v: `${gene.hsct_required_pct}%` },
              { l: 'Sibling Donor Risk', v: `${gene.sibling_donor_risk_pct}%` },
              { l: 'Venetoclax Poor', v: `${gene.venetoclax_poor_pct}%` },
              { l: 'ALL Risk', v: `${gene.all_risk_pct}%` },
              { l: 'JMML', v: `${gene.jmml_pct}%` },
              { l: 'Monosomy 7', v: `${gene.monosomy7_pct}%` },
              { l: 'Revertant Mosaicism', v: `${gene.revertant_mosaicism_pct}%` },
              { l: 'Ataxia', v: `${gene.ataxia_pct}%` },
              { l: 'Li-Fraumeni', v: `${gene.li_fraumeni_pct}%` },
              { l: 'NF1 Features', v: `${gene.nf1_features_pct}%` },
              { l: '5\'UTR Missed', v: `${gene.five_utr_missed_pct}%` },
              { l: 'Platelet Dysfunction', v: `${gene.platelet_dysfunction_pct}%` },
            ].map(({ l, v }) => (
              <div key={l} style={{ background: '#f5f5f5', borderRadius: 6, padding: '6px 12px' }}>
                <span style={{ fontSize: 11, color: '#888' }}>{l}</span>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene.gene] || '#1565c0' }}>{v}</div>
              </div>
            ))}
          </div>

          <Section title="Age of Onset / Clinical Spectrum" text={gene.age_of_onset} color="#e3f2fd" />
          <Section title="Key Biomarkers" text={gene.key_biomarker} color="#e8f5e9" />
          <Section title="Pathognomonic / Differentiators" text={gene.pathognomonic} color="#fff3e0" highlight />
          <Section title="Treatment & Surveillance" text={gene.treatment} color="#f3e5f5" />

          <div style={{ marginTop: 16 }}>
            <strong style={{ color: '#b71c1c' }}>Critical Flags:</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
              {gene.critical_flags?.map(f => (
                <span key={f} style={{ background: '#b71c1c', color: '#fff', borderRadius: 4, padding: '3px 10px', fontSize: 12 }}>{f}</span>
              ))}
            </div>
          </div>

          {gene.cohort_preview?.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <strong>Patient Preview (first 5):</strong>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 8 }}>
                <thead>
                  <tr style={{ background: '#1565c0', color: '#fff' }}>
                    {['ID','Age','Sex','AML/MDS','Thrombocytopenia','HSCT','Sibling Risk','Venetoclax Poor','ALL Risk','JMML','Ataxia'].map(h => (
                      <th key={h} style={{ padding: '6px 8px' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {gene.cohort_preview.map((p, i) => (
                    <tr key={p.patient_id} style={{ background: i % 2 === 0 ? '#f8f9fa' : '#fff' }}>
                      <td style={{ padding: '5px 8px' }}>{p.patient_id}</td>
                      <td style={{ padding: '5px 8px' }}>{p.age}</td>
                      <td style={{ padding: '5px 8px' }}>{p.sex}</td>
                      <td style={{ padding: '5px 8px', color: p.aml_mds ? '#b71c1c' : '#555' }}>{p.aml_mds ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.thrombocytopenia ? '#1565c0' : '#555' }}>{p.thrombocytopenia ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.hsct_required ? '#6a1b9a' : '#555' }}>{p.hsct_required ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.sibling_donor_risk ? '#c62828' : '#555' }}>{p.sibling_donor_risk ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.venetoclax_poor ? '#880e4f' : '#555' }}>{p.venetoclax_poor ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.all_risk ? '#e65100' : '#555' }}>{p.all_risk ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.jmml ? '#004d40' : '#555' }}>{p.jmml ? '✓' : '–'}</td>
                      <td style={{ padding: '5px 8px', color: p.ataxia ? '#880e4f' : '#555' }}>{p.ataxia ? '✓' : '–'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Section({ title, text, color, highlight }) {
  return (
    <div style={{ background: color || '#f5f5f5', borderRadius: 8, padding: 14, marginBottom: 12,
      border: highlight ? '1px solid #ffe082' : 'none' }}>
      <strong style={{ color: highlight ? '#e65100' : '#333' }}>{title}:</strong>
      <div style={{ fontSize: 13, marginTop: 6, whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>{text}</div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const [section, setSection] = useState('glossary');
  return (
    <div>
      <div style={{ display: 'flex', gap: 10, marginBottom: 20 }}>
        {['glossary','surveillance_protocols','shared_mechanism'].map(s => (
          <button key={s} onClick={() => setSection(s)}
            style={{ padding: '8px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: section === s ? '#1565c0' : '#eee', color: section === s ? '#fff' : '#333' }}>
            {s === 'glossary' ? 'Glossary' : s === 'surveillance_protocols' ? 'Surveillance Protocols' : 'Shared Mechanism'}
          </button>
        ))}
      </div>

      {section === 'shared_mechanism' && (
        <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16 }}>
          <strong>Shared Mechanism:</strong>
          <p style={{ fontSize: 13, marginTop: 8 }}>{data.shared_mechanism}</p>
        </div>
      )}

      {section === 'glossary' && (
        <div>
          {Object.entries(data.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 12, borderBottom: '1px solid #eee', paddingBottom: 10 }}>
              <strong style={{ color: '#1565c0' }}>{term}</strong>
              <p style={{ fontSize: 13, margin: '4px 0 0', color: '#555' }}>{def}</p>
            </div>
          ))}
        </div>
      )}

      {section === 'surveillance_protocols' && (
        <div>
          {Object.entries(data.surveillance_protocols || {}).map(([gene, proto]) => (
            <div key={gene} style={{ marginBottom: 14, background: '#f8f9fa', borderRadius: 8, padding: 12 }}>
              <strong style={{ color: GENE_COLORS[gene] || '#1565c0' }}>{gene}:</strong>
              <p style={{ fontSize: 13, margin: '4px 0 0' }}>{proto}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function HaematologicalMalignancyPredispositionAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview]     = useState(null);
  const [breakdown, setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bd, def]) => {
      setOverview(ov);
      setBreakdown(bd);
      setDefinitions(def);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1400, margin: '0 auto', padding: 24 }}>
      <div style={{ background: 'linear-gradient(135deg, #1565c0 0%, #880e4f 100%)', borderRadius: 12, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 13, opacity: 0.85, marginBottom: 4 }}>🧬 Hereditary Haematological Malignancy Predisposition Atlas</div>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 800 }}>
          Hereditary-Haematological-Malignancy-Predisposition-Atlas
        </h1>
        <div style={{ fontSize: 13, marginTop: 8, opacity: 0.9 }}>
          Complete 8-Gene Inherited Haematological Cancer Predisposition Atlas · 320 Patients · Seeds 2022–2029
        </div>
        <div style={{ fontSize: 12, marginTop: 6, opacity: 0.8 }}>
          RUNX1 · CEBPA · DDX41 · TP53 · ETV6 · ANKRD26 · SAMD9L · NF1
        </div>
      </div>

      {err && <ErrBox msg={err} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{ padding: '9px 20px', borderRadius: 8, border: 'none', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
              background: tab === t ? '#1565c0' : '#eee', color: tab === t ? '#fff' : '#333' }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ background: '#fff', borderRadius: 10, padding: 24, boxShadow: '0 2px 8px rgba(0,0,0,0.07)' }}>
        {tab === 'Overview'       && <OverviewTab data={overview} />}
        {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
        {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
