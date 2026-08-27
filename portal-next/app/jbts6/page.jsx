'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Allele-Tier & COACH Pearl', 'Definitions'];

// JBTS6 colour scheme — TMEM67 / COACH hepatic fibrosis / TZ membrane / North African founder
const ACCENT  = '#1a237e';   // deep indigo — JBTS/MTS neurological
const ACCENT2 = '#b71c1c';   // deep red — COACH hepatic fibrosis (distinctive)
const ACCENT3 = '#006064';   // dark cyan — renal NPHP11 / TZ membrane
const ACCENT4 = '#1b5e20';   // deep green — transplant outcomes (curative)
const ACCENT5 = '#e65100';   // burnt orange — North African p.Cys615Arg founder pearl
const ACCENT6 = '#4a148c';   // deep purple — allele-class tier rule
const ACCENT7 = '#f57f17';   // amber — COACH spectrum / dual organ
const ACCENT8 = '#37474f';   // dark slate — cerebellar ataxia / OMA
const ACCENT9 = '#880e4f';   // deep crimson — MKS3 lethal / biallelic null

const SEED = 419;
const N_COHORT = 40;

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function AlertBox({ color, title, children }) {
  return (
    <div className="alert mb-3" style={{ borderLeft: `5px solid ${color}`, background: '#fafafa' }}>
      <strong style={{ color }}>{title}</strong>
      <div className="mt-1 small">{children}</div>
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color }}>{title}</h6>
      {children}
    </div>
  );
}

function DataTable({ headers, rows }) {
  return (
    <div className="table-responsive mb-3">
      <table className="table table-sm table-bordered table-hover small mb-0">
        <thead className="table-dark">
          <tr>{headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Tab: Overview ────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { kpis = [], hallmark, critical_diagnostic_pearl, north_african_founder_pearl,
          allele_phenotype_rule, prevalence, first_description, gene_summary, phenotype_summary } = data;
  return (
    <div>
      <AlertBox color={ACCENT2} title="⚠ CRITICAL DIAGNOSTIC PEARL — COACH SYNDROME (Hepatic Fibrosis)">
        {critical_diagnostic_pearl}
      </AlertBox>
      <AlertBox color={ACCENT5} title="⚡ NORTH AFRICAN FOUNDER ALLELE — p.Cys615Arg (c.1843T>C)">
        {north_african_founder_pearl}
      </AlertBox>
      <AlertBox color={ACCENT6} title="🧬 ALLELE-CLASS → DISEASE TIER RULE">
        {allele_phenotype_rule}
      </AlertBox>

      <div className="row mb-3">
        {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
      </div>

      <Section title="Molecular Hallmark" color={ACCENT}>
        <p className="small">{hallmark}</p>
      </Section>

      {phenotype_summary && (
        <Section title="Phenotype Frequencies (n=40 cohort)" color={ACCENT8}>
          <DataTable
            headers={['Feature', 'n', '%', 'Note']}
            rows={[
              ['Molar Tooth Sign (MTS)', `${Math.round(phenotype_summary.mts_pct * N_COHORT / 100)}`, `${phenotype_summary.mts_pct}%`, 'Pathognomonic; brain MRI mandatory'],
              ['Cerebellar Ataxia', `${Math.round(phenotype_summary.ataxia_pct * N_COHORT / 100)}`, `${phenotype_summary.ataxia_pct}%`, 'Gait ataxia; truncal instability'],
              ['Neonatal Hypotonia', `${Math.round(phenotype_summary.hypotonia_pct * N_COHORT / 100)}`, `${phenotype_summary.hypotonia_pct}%`, 'Universal early feature'],
              ['Oculomotor Apraxia', `${Math.round(phenotype_summary.oma_pct * N_COHORT / 100)}`, `${phenotype_summary.oma_pct}%`, 'Horizontal gaze initiation failure'],
              ['Intellectual Disability', `${Math.round(phenotype_summary.id_pct * N_COHORT / 100)}`, `${phenotype_summary.id_pct}%`, 'Moderate > severe'],
              ['Breathing Dysregulation', `${Math.round(phenotype_summary.breathing_pct * N_COHORT / 100)}`, `${phenotype_summary.breathing_pct}%`, 'Neonatal episodic apnea; self-resolves'],
              ['Retinal Dystrophy', `${Math.round(phenotype_summary.retinal_pct * N_COHORT / 100)}`, `${phenotype_summary.retinal_pct}%`, 'Rod-cone; annual ERG mandatory'],
              ['Hepatic Fibrosis (COACH)', `${Math.round(phenotype_summary.hepatic_pct * N_COHORT / 100)}`, `${phenotype_summary.hepatic_pct}%`, '⭐ DISTINCTIVE — ductal plate malformation'],
              ['Portal Hypertension', `${Math.round(phenotype_summary.portal_htn_pct * N_COHORT / 100)}`, `${phenotype_summary.portal_htn_pct}%`, 'Varices risk; subset of hepatic fibrosis'],
              ['Renal NPHP11', `${Math.round(phenotype_summary.renal_pct * N_COHORT / 100)}`, `${phenotype_summary.renal_pct}%`, 'TIN; ESRD median ~18yr'],
              ['Polydactyly', `${Math.round(phenotype_summary.polydactyly_pct * N_COHORT / 100)}`, `${phenotype_summary.polydactyly_pct}%`, 'Postaxial; less common than MKS3'],
            ]}
          />
        </Section>
      )}

      {gene_summary && (
        <Section title="Gene & Disease Summary" color={ACCENT3}>
          <DataTable
            headers={['Field', 'Value']}
            rows={Object.entries(gene_summary).map(([k, v]) => [k.replace(/_/g, ' '), v])}
          />
        </Section>
      )}

      <Section title="First Description" color={ACCENT8}>
        <p className="small">{first_description}</p>
      </Section>

      <Section title="Prevalence" color={ACCENT}>
        <p className="small">{prevalence}</p>
      </Section>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { ethnicity = [], allele_class = [], age_at_diagnosis = [],
          coach_by_ethnicity = [], phenotype_matrix = [], transplant_outcomes = {},
          allele_protein_table = [] } = data;
  return (
    <div>
      <Section title="Ethnicity Distribution" color={ACCENT5}>
        <DataTable
          headers={['Ethnicity', 'n', '%']}
          rows={ethnicity.map(e => [e.ethnicity, e.n, `${e.pct}%`])}
        />
      </Section>

      <Section title="Allele Class Distribution" color={ACCENT6}>
        <DataTable
          headers={['Allele Class', 'n', '%']}
          rows={allele_class.map(a => [a.class, a.n, `${a.pct}%`])}
        />
      </Section>

      <Section title="Age at Diagnosis" color={ACCENT8}>
        <DataTable
          headers={['Age Bucket', 'n', '%']}
          rows={age_at_diagnosis.map(a => [a.bucket, a.n, `${a.pct}%`])}
        />
      </Section>

      <Section title="Hepatic Fibrosis (COACH) by Ethnicity" color={ACCENT2}>
        <DataTable
          headers={['Ethnicity', 'n Hepatic', '% of Ethnic Group']}
          rows={coach_by_ethnicity.map(c => [c.ethnicity, c.n_hepatic, `${c.pct_of_group}%`])}
        />
      </Section>

      <Section title="Full Phenotype Matrix" color={ACCENT}>
        <DataTable
          headers={['Feature', 'n', '%', 'Note']}
          rows={phenotype_matrix.map(p => [p.feature, p.n, `${p.pct}%`, p.note])}
        />
      </Section>

      <Section title="Transplant Outcomes" color={ACCENT4}>
        <DataTable
          headers={['Metric', 'Value']}
          rows={[
            ['Renal Tx needed (NPHP11 ESRD)', transplant_outcomes.n_renal_transplant_needed],
            ['Liver Tx needed (portal HTN)', transplant_outcomes.n_liver_transplant_needed],
            ['Combined liver-kidney Tx', transplant_outcomes.n_combined_liver_kidney],
            ['Average ESRD age (yrs)', transplant_outcomes.avg_esrd_age ?? 'N/A'],
            ['Renal Tx outcome', transplant_outcomes.renal_tx_outcome],
            ['Hepatic Tx outcome', transplant_outcomes.hepatic_tx_outcome],
            ['Combined Tx note', transplant_outcomes.combined_tx_note],
          ]}
        />
      </Section>

      <Section title="Allele → Protein Domain → Phenotype" color={ACCENT3}>
        <DataTable
          headers={['Variant', 'cDNA', 'Domain', 'Ethnic Enrichment', 'Disease Tier']}
          rows={allele_protein_table.map(a => [a.variant, a.cdna, a.domain, a.ethnic, a.tier])}
        />
      </Section>
    </div>
  );
}

// ── Tab: Allele-Tier & COACH Pearl ──────────────────────────────────────────
function AlleleTierTab({ overview, breakdown }) {
  return (
    <div>
      <AlertBox color={ACCENT9} title="🔴 BIALLELIC NULL → MKS3 (MECKEL-GRUBER LETHAL)">
        Two truncating TMEM67 alleles → Meckel-Gruber Syndrome 3 (MKS3): encephalocele, post-axial
        polydactyly, polycystic kidneys, oligohydramnios. INCOMPATIBLE with extrauterine life.
        Prenatal diagnosis (chorionic villus sampling) essential in families with prior MKS3.
      </AlertBox>

      <AlertBox color={ACCENT7} title="🟡 ONE NULL + ONE HYPOMORPHIC → JBTS6 (± COACH ± NPHP11)">
        Most common JBTS6 genotype: compound heterozygote (null allele + hypomorphic missense/splice).
        Molar Tooth Sign present in ~88%. COACH hepatic fibrosis in ~26–42% depending on allele.
        Renal NPHP11 in ~38%; ESRD median ~18yr. All patients need brain MRI, annual LFTs, ERG.
      </AlertBox>

      <AlertBox color={ACCENT5} title="🟠 NORTH AFRICAN FOUNDER — p.Cys615Arg HOMOZYGOUS → JBTS6 + COACH">
        Homozygous p.Cys615Arg (c.1843T>C): most common genotype in Moroccan, Algerian, Tunisian
        patients. ~40% hepatic fibrosis rate when homozygous. ABSENT from most European gene panels.
        Targeted TMEM67 sequencing mandatory in North African patients with Molar Tooth Sign.
      </AlertBox>

      <AlertBox color={ACCENT4} title="🟢 BIALLELIC HYPOMORPHIC → JBTS6 MILD / NPHP11">
        Two hypomorphic (missense/missense) alleles → milder JBTS6 or isolated NPHP11 (renal only).
        MTS may be present; hepatic fibrosis less frequent. Transplant outcomes CURATIVE.
      </AlertBox>

      <Section title="COACH Syndrome — 5-Feature Diagnostic Criteria" color={ACCENT2}>
        <DataTable
          headers={['Feature', 'COACH Frequency', 'Clinical Action']}
          rows={[
            ['Cerebellar vermis hypoplasia (MTS)', '~88%', 'Brain MRI at diagnosis; annual neurology'],
            ['Oligophrenia (intellectual disability)', '~68%', 'Neuropsychological assessment; IEP support'],
            ['Ataxia (cerebellar)', '~85%', 'Physiotherapy; gait training; fall prevention'],
            ['Coloboma (iris/retina)', '~20%', 'Ophthalmology at diagnosis; annual ERG'],
            ['Hepatic fibrosis (ductal plate malformation)', '~30%', 'Annual LFTs; liver US; hepatology; varices screen'],
          ]}
        />
        <AlertBox color={ACCENT2} title="COACH Liver Management Algorithm">
          Diagnosis → LFTs + liver US → if fibrosis: hepatology referral → portal pressure estimation
          → varices screening (endoscopy if portal HTN) → beta-blocker prophylaxis if large varices
          → Combined liver-kidney Tx assessment if ESRD + portal HTN coexist.
        </AlertBox>
      </Section>

      <Section title="TMEM67 Protein Domain → Allele-Tier Map" color={ACCENT3}>
        <DataTable
          headers={['Domain (aa)', 'Function', 'Key Alleles', 'Disease Tier']}
          rows={[
            ['Signal peptide (1–25)', 'ER targeting', '—', 'N/A'],
            ['FN-III extracellular (25–750)', 'Wnt/FZD sensing; Ca²⁺ binding', 'Tyr78Cys; Gln376Ter', 'Mild JBTS6 / NULL→MKS3'],
            ['TM-proximal extracellular (600–750)', 'TM anchoring surface', 'Cys615Arg (N. African founder); Trp628Cys', 'JBTS6 + COACH enriched'],
            ['Transmembrane (750–770)', 'TZ membrane anchor', 'Leu736Pro (MENA)', 'JBTS6 / NPHP11'],
            ['Cytoplasmic tail (770–995)', 'NPHP4/CC2D2A binding; TZ scaffold', 'Arg941Gln (European)', 'JBTS6 hypomorphic'],
          ]}
        />
      </Section>

      <Section title="Comparative JBTS Type Features" color={ACCENT}>
        <DataTable
          headers={['JBTS Type', 'Gene', 'Distinctive Feature vs JBTS6', 'COACH?', 'Hepatic?']}
          rows={[
            ['JBTS3', 'AHI1', 'OMA ~75% (highest); Ashkenazi Arg830Trp founder', 'No', 'No'],
            ['JBTS4', 'NPHP1', 'High renal ~45%; MLPA mandatory (610kb del)', 'No', 'No'],
            ['JBTS5', 'CEP290', 'Most common JBTS gene; IVS26 invisible to WES; highest retinal (57%)', 'No', 'No'],
            ['JBTS6', 'TMEM67', 'COACH hepatic fibrosis ~30% DISTINCTIVE; North African founder Cys615Arg', 'Yes', 'Yes ~30%'],
            ['JBTS9', 'CC2D2A', 'SECOND COACH gene; MKS6; JBTS-COACH overlap', 'Yes', 'Yes ~25%'],
          ]}
        />
      </Section>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Section title="Glossary & Definitions" color={ACCENT}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={Object.entries(data).map(([k, v]) => [k.replace(/_/g, ' '), v])}
        />
      </Section>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────────
export default function JBTS6Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts6/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts6/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts6/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      {/* ── sidebar nav ── */}
      <nav style={{ width: '260px', background: '#1a237e', color: '#fff', padding: '1rem', overflowY: 'auto', flexShrink: 0 }}>
        <div className="fw-bold mb-1" style={{ fontSize: '0.85rem' }}>JBTS6 — TMEM67</div>
        <div style={{ fontSize: '0.7rem', color: '#90caf9' }} className="mb-3">
          Meckelin / MKS3 / NPHP11 / COACH<br />8q22.1 · 995 aa · OMIM #610688
        </div>
        <ul className="list-unstyled mb-3" style={{ fontSize: '0.78rem' }}>
          {TABS.map((t, i) => (
            <li key={i} className="mb-1">
              <button
                onClick={() => setTab(i)}
                className="btn btn-sm w-100 text-start"
                style={{ background: tab === i ? '#ffffff22' : 'transparent', color: '#fff', border: 'none' }}
              >
                {t}
              </button>
            </li>
          ))}
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small mb-2 fw-bold" style={{ color: '#90caf9' }}>JBTS Series</div>
        <ul className="list-unstyled" style={{ fontSize: '0.72rem' }}>
          <li><Link href="/jbts3" className="text-white-50">← JBTS3 (AHI1)</Link></li>
          <li><Link href="/jbts4" className="text-white-50">← JBTS4 (NPHP1)</Link></li>
          <li><Link href="/jbts5" className="text-white-50">← JBTS5 (CEP290)</Link></li>
          <li className="fw-bold" style={{ color: '#fff' }}>▶ JBTS6 (TMEM67)</li>
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small mb-2 fw-bold" style={{ color: '#90caf9' }}>Related</div>
        <ul className="list-unstyled" style={{ fontSize: '0.72rem' }}>
          <li><Link href="/nphp11" className="text-white-50">NPHP11 (TMEM67 renal)</Link></li>
          <li><Link href="/joubert" className="text-white-50">Joubert Syndrome overview</Link></li>
          <li><Link href="/bbs14" className="text-white-50">BBS14 (CEP290) →</Link></li>
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small text-white-50">
          Cohort: {N_COHORT} patients · seed {SEED}
          <br />OMIM: #610688 · Gene: *609884
          <br />COACH Pearl: hepatic fibrosis ~30%
        </div>
      </nav>

      {/* ── main content ── */}
      <main style={{ flex: 1, padding: '1.5rem', overflowY: 'auto' }}>
        <div className="d-flex align-items-center mb-1">
          <h4 className="mb-0 me-3" style={{ color: ACCENT }}>
            TMEM67 — Joubert Syndrome Type 6 (JBTS6)
          </h4>
          <span className="badge" style={{ background: ACCENT2, fontSize: '0.75rem' }}>
            COACH Hepatic Fibrosis (~30%) — DISTINCTIVE
          </span>
        </div>
        <div className="text-muted small mb-3">
          Meckelin · MKS3 · NPHP11 · COACH · 8q22.1 · 995 aa · OMIM #610688 · seed {SEED} · {N_COHORT} patients
        </div>

        {error && (
          <div className="alert alert-danger">API error: {error}</div>
        )}

        {/* Tab bar */}
        <ul className="nav nav-tabs mb-4">
          {TABS.map((t, i) => (
            <li key={i} className="nav-item">
              <button
                className={`nav-link ${tab === i ? 'active' : ''}`}
                onClick={() => setTab(i)}
              >
                {t}
              </button>
            </li>
          ))}
        </ul>

        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <BreakdownTab data={breakdown} />}
        {tab === 2 && <AlleleTierTab overview={overview} breakdown={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </main>
    </div>
  );
}
