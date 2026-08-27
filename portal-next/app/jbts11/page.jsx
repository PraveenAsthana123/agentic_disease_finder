'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Tectonic Complex Pearls', 'Definitions'];

// JBTS11 colour scheme — TCTN1 / tectonic complex / lipid gate / TZ scaffold
const ACCENT  = '#1b3a4b';   // deep teal-navy — tectonic complex / TZ gate
const ACCENT2 = '#1a237e';   // deep indigo — MTS / neurological / ciliogenesis
const ACCENT3 = '#004d40';   // deep teal — renal NPHP-like
const ACCENT4 = '#1b5e20';   // deep green — transplant / curative endpoint
const ACCENT5 = '#4a0072';   // deep purple — retinal / rod-cone dystrophy
const ACCENT6 = '#37474f';   // dark slate — TZ Y-link / domain matrix
const ACCENT7 = '#0d47a1';   // deep blue — no MKS tier / TCTN2 distinction
const ACCENT8 = '#e65100';   // burnt orange — polydactyly / JSOFD
const ACCENT9 = '#006064';   // dark cyan — lipid gate / cholesterol mechanism

const SEED = 429;
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { kpis = [], hallmark, tectonic_complex_pearl, no_mks_pearl, renal_retinal_pearl,
          first_description, gene_summary, phenotype_summary,
          allele_class_distribution = [] } = data;
  return (
    <div>
      <AlertBox color={ACCENT9} title="⚠ TECTONIC COMPLEX — LIPID GATE AT THE TRANSITION ZONE (TCTN1+TCTN2+TCTN3)">
        {tectonic_complex_pearl}
      </AlertBox>
      <AlertBox color={ACCENT7} title="⚠ NO MKS TIER — TCTN1 BIALLELIC NULL → JBTS11 (LIVE BIRTH); TCTN2 NULL → MKS8 (LETHAL)">
        {no_mks_pearl}
      </AlertBox>
      <AlertBox color={ACCENT5} title="⚠ RETINAL ~35% (ROD-CONE) + RENAL ~28% (NPHP-LIKE TIN) — ANNUAL SURVEILLANCE">
        {renal_retinal_pearl}
      </AlertBox>

      <Section title="KPI Summary — JBTS11 Cohort (N=40, seed-429)" color={ACCENT2}>
        <div className="row g-2">
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Gene & Disease Summary" color={ACCENT2}>
            <table className="table table-sm table-bordered small">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>TCTN1 (OMIM *609863)</td></tr>
                <tr><td className="fw-bold">Disease JBTS11</td><td>Joubert Syndrome 11 (OMIM #614170) — Autosomal Recessive</td></tr>
                <tr><td className="fw-bold">Chromosome</td><td>12q24.11</td></tr>
                <tr><td className="fw-bold">Protein</td><td>1348 aa — Signal peptide / N-term scaffold (TCTN2/3 dimerisation) / Tectonic domain (ciliary gate core) / C-term membrane anchoring</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td>Autosomal recessive — biallelic LOF; null/null → severe; null/missense → moderate; missense/missense → mild</td></tr>
                <tr><td className="fw-bold">Prevalence</td><td>~1–2% all JBTS; ~1/1,000,000–2,500,000 worldwide</td></tr>
                <tr><td className="fw-bold" style={{ color: ACCENT7 }}>MKS Tier</td><td style={{ color: ACCENT7, fontWeight: 'bold' }}>NONE — TCTN1 biallelic null → JBTS11 (live birth); differs from TCTN2 (→ MKS8, lethal)</td></tr>
                <tr><td className="fw-bold">Tectonic Complex</td><td>TCTN1 + TCTN2 + TCTN3 — TZ lipid gate (cholesterol/sphingolipid enrichment); SMO entry</td></tr>
                <tr><td className="fw-bold">Retinal</td><td>~35% rod-cone dystrophy; annual ERG</td></tr>
                <tr><td className="fw-bold">Renal</td><td>~28% NPHP-like TIN; ESRD median ~22yr; transplant curative</td></tr>
                <tr><td className="fw-bold">Hepatic</td><td>~12% mild CHF; bile duct TZ cilia</td></tr>
                <tr><td className="fw-bold">First Description</td><td>{first_description}</td></tr>
                <tr><td className="fw-bold">Hallmark</td><td>{hallmark}</td></tr>
              </tbody>
            </table>
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Phenotype Frequencies (N=40)" color={ACCENT6}>
            <DataTable
              headers={['Feature', 'Frequency', 'Notes']}
              rows={[
                ['Molar Tooth Sign (MTS)', '100%', 'Pathognomonic — all JBTS11'],
                ['Cerebellar Ataxia', `${phenotype_summary?.ataxia_pct ?? '–'}%`, 'Core feature; SARA tracking'],
                ['Neonatal Hypotonia', `${phenotype_summary?.hypotonia_pct ?? '–'}%`, 'Feeding difficulty in infancy'],
                ['Oculomotor Apraxia', `${phenotype_summary?.oma_pct ?? '–'}%`, 'Head thrust compensation'],
                ['Intellectual Disability', `${phenotype_summary?.id_pct ?? '–'}%`, 'Mild–moderate range'],
                ['Breathing Dysregulation', `${phenotype_summary?.breathing_pct ?? '–'}%`, 'Episodic apnea/hyperpnea'],
                ['Retinal Dystrophy (rod-cone)', `${phenotype_summary?.retinal_pct ?? '–'}%`, 'Annual ERG mandatory'],
                ['Polydactyly (post-axial)', `${phenotype_summary?.polydactyly_pct ?? '–'}%`, 'JSOFD; skeletal survey if present'],
                ['Renal (NPHP-like TIN)', `${phenotype_summary?.renal_pct ?? '–'}%`, 'ESRD median ~22yr; transplant curative'],
                ['Hepatic Fibrosis (mild)', `${phenotype_summary?.hepatic_pct ?? '–'}%`, 'Bile duct cilia; annual LFTs if suspected'],
                ['No MKS Tier', 'JBTS11 only (live birth)', 'TCTN2 null/null → MKS8 (lethal)'],
                ['Tectonic Complex', 'TCTN1+TCTN2+TCTN3', 'Lipid gate; SMO entry; TMEM67/CC2D2A bridge'],
              ]}
            />
          </Section>
        </div>
      </div>

      <Section title="Allele Class Distribution (Cohort)" color={ACCENT6}>
        <DataTable
          headers={['Allele Class', 'Count', '%']}
          rows={allele_class_distribution.map(a => [a.allele_class, a.count, `${a.pct}%`])}
        />
      </Section>

      <Section title="Gene Function Summary" color={ACCENT2}>
        <p className="small">{gene_summary}</p>
      </Section>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ─────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const {
    ethnicity_distribution = [],
    key_variants = [],
    domain_phenotype_matrix = [],
    pathway_steps = [],
    management = [],
    patient_table = [],
  } = data;

  return (
    <div>
      <Section title="Ethnicity Distribution" color={ACCENT2}>
        <DataTable
          headers={['Ethnicity', 'Count', '%']}
          rows={ethnicity_distribution.map(e => [e.ethnicity, e.count, `${e.pct}%`])}
        />
      </Section>

      <Section title="Key Pathogenic Variants (TCTN1)" color={ACCENT5}>
        <DataTable
          headers={['Variant', 'Domain', 'Effect', 'Population', 'Allele Class', 'Severity', 'Retinal Risk', 'Renal Risk']}
          rows={key_variants.map(v => [
            <strong key={v.variant} style={{ color: ACCENT5 }}>{v.variant}</strong>,
            v.domain, v.effect, v.population, v.allele_class, v.severity, v.retinal_risk, v.renal_risk,
          ])}
        />
      </Section>

      <Section title="Domain → Phenotype Severity Matrix" color={ACCENT6}>
        <DataTable
          headers={['Domain', 'Key Variants', 'Function Lost', 'Severity', 'Retinal Risk', 'Renal Risk']}
          rows={domain_phenotype_matrix.map(d => [
            d.domain, d.key_variants, d.function_lost, d.severity, d.retinal_risk, d.renal_risk,
          ])}
        />
      </Section>

      <Section title="TCTN1 → Tectonic Complex → Lipid Gate → MTS Pathway" color={ACCENT2}>
        <DataTable
          headers={['Step', 'Normal Event', 'Effect When TCTN1 Lost']}
          rows={pathway_steps.map(s => [s.step, s.event, s.effect_when_lost])}
        />
      </Section>

      <Section title="Clinical Management" color={ACCENT4}>
        <DataTable
          headers={['Intervention', 'Timing', 'Rationale', 'Level']}
          rows={management.map(m => [m.intervention, m.timing, m.rationale, m.level])}
        />
      </Section>

      <Section title="Per-Patient Table (first 20)" color={ACCENT6}>
        <DataTable
          headers={['ID', 'Sex', 'Ethnicity', 'Allele Class', 'Age Dx', 'MTS', 'Ataxia', 'OMA', 'Retinal', 'Poly', 'Renal', 'Hepatic', 'ID', 'Breathing']}
          rows={patient_table.map(p => [
            p.id, p.sex, p.ethnicity, p.allele, p.age_dx_yr,
            p.mts, p.ataxia, p.oma, p.retinal, p.poly, p.renal, p.hepatic,
            p.id_, p.breathing,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Tab: Tectonic Complex Pearls ──────────────────────────────────────────────
function PearlTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <AlertBox color={ACCENT9} title="⚠ TECTONIC COMPLEX — TZ LIPID GATE: TCTN1+TCTN2+TCTN3 CONTROL CHOLESTEROL ENRICHMENT FOR SMO ENTRY">
        The tectonic complex (TCTN1 + TCTN2 + TCTN3) forms a membrane-associated scaffold at the
        ciliary transition zone that controls lipid composition of the ciliary gate. By enriching
        cholesterol and sphingolipids at the TZ membrane, the tectonic complex creates a 'lipid
        fence' essential for SMOOTHENED (SMO) entry into cilia and Hedgehog pathway activation.
        TCTN1 LOF → tectonic complex collapses → lipid gate opens/fails → SMO excluded → Hedgehog
        failure → cerebellar vermis hypoplasia → Molar Tooth Sign (MTS). TCTN1 also bridges the
        MKS module (TMEM67, CC2D2A, MKS1) and NPHP module (B9D1, TMEM231) at the TZ Y-links,
        making it a key cross-module scaffold protein at the transition zone gate.
      </AlertBox>

      <Section title="JBTS11 in Context — JBTS3-11 Series Comparison" color={ACCENT2}>
        <DataTable
          headers={['JBTS Type', 'Gene', 'Inheritance', 'MKS Tier', 'Tectonic Module?', 'Key Feature']}
          rows={[
            ['JBTS11', 'TCTN1', 'AR biallelic LOF', 'None (JBTS11 only — no MKS)', '⭐ FOUNDING MEMBER — lipid gate', 'Tectonic complex lipid gate; TCTN2 distinction critical'],
            ['JBTS10', 'OFD1', 'X-linked', 'None (X-linked)', 'Centriolar satellite (upstream)', 'Only X-linked JBTS; RP23 allelic; highest retinal ~55%'],
            ['JBTS9', 'CC2D2A', 'AR biallelic LOF', 'MKS6 (null/null → lethal)', 'Yes — tectonic domain partner', 'COACH 2nd most common; JSOFD ~20%'],
            ['JBTS8', 'ARL13B', 'AR biallelic LOF', 'None', 'Lipid axis (PI4P/INPP5E)', 'INPP5E trafficking; no MKS; no CHF'],
            ['JBTS7', 'RPGRIP1L', 'AR biallelic LOF', 'MKS5 (null/null → lethal)', 'Y-link scaffold', 'TZ Y-link; European Ala229Thr founder; NPHP8'],
            ['JBTS6', 'TMEM67', 'AR biallelic LOF', 'MKS3 (null/null → lethal)', 'Yes — tectonic domain partner', 'COACH most common; FN-III Wnt sensing; hepatic ~30%'],
            ['JBTS5', 'CEP290', 'AR biallelic LOF', 'MKS4 (null/null → lethal)', 'Y-link inner plate', 'Most common JBTS (~10-15%); IVS26 deep intronic'],
            ['JBTS4', 'NPHP1', 'AR biallelic LOF', 'None', 'NPHP module (Y-link inner)', 'Pure renal subset; 1q22 deletion'],
            ['JBTS3', 'AHI1', 'AR biallelic LOF', 'None', 'IFT (connecting cilium)', 'Retinal ~40%; Mid-Eastern Arg830Trp founder'],
          ]}
        />
      </Section>

      <AlertBox color={ACCENT7} title="⚠ TCTN1 vs TCTN2 — CRITICAL DISTINCTION: SAME COMPLEX, DIFFERENT DISEASE TIERS">
        TCTN1 (12q24.11, *609863) and TCTN2 (12q24.31, *613885) are both members of the tectonic
        complex on chromosome 12q — but their disease profiles are VERY different.
        TCTN1 biallelic null → JBTS11 (live birth, MTS + ataxia + moderate retinal/renal).
        TCTN2 biallelic null → MKS8 (#613885, perinatal LETHAL — encephalocele, polydactyly,
        cystic kidneys). This distinction is CRITICAL for reproductive counselling: families with
        TCTN2 null variants face lethal recurrence risk with biallelic null offspring, while
        TCTN1 null families have JBTS11 (live birth, significant disability but survivable).
        WES panels covering chromosome 12q must distinguish both genes (different loci: 12q24.11
        vs 12q24.31). TCTN3 (10q24.1) → Oro-Facial-Digital Syndrome Type 4 (OFD4) — separate
        chromosome and disease. Molecular report must specify TCTN1 explicitly.
      </AlertBox>

      <Section title="Tectonic Complex — Module Interactions at the Transition Zone" color={ACCENT9}>
        <DataTable
          headers={['Interaction Partner', 'Domain of TCTN1', 'Function', 'Disease Consequence When Lost']}
          rows={[
            ['TCTN2 (tectonic complex)', 'N-terminal scaffold (aa 29-350)', 'TCTN complex assembly; lipid gate formation', 'TCTN1 N-term variants → partial lipid gate → mild JBTS11 (Ala298Val founder)'],
            ['TCTN3 (tectonic complex)', 'N-terminal scaffold (aa 29-350)', 'Ternary tectonic complex stabilisation; ciliary gate membrane targeting', 'N-term null → complete complex loss → severe JBTS11'],
            ['TMEM67 (MKS3/JBTS6)', 'Tectonic domain (aa 350-900)', 'MKS module bridge; TZ outer plate connection', 'Tectonic domain loss (Gly347Arg/Leu506Pro) → TMEM67 interaction impaired → moderate-severe JBTS11'],
            ['CC2D2A (MKS6/JBTS9)', 'Tectonic domain (aa 350-900)', 'TZ scaffold cross-talk; Y-link outer plate', 'Tectonic domain variants (Leu506Pro) → CC2D2A binding reduced; ataxia + retinal risk elevated'],
            ['MKS1 (B9 module)', 'Tectonic domain (aa 350-900)', 'B9 complex cross-talk; MKS module integrity', 'Tectonic domain null → MKS1 interaction lost → broad ciliary gate failure'],
            ['B9D1 / TMEM231 (NPHP module)', 'C-terminal (aa 900-1348)', 'NPHP module bridge; renal TZ cilia integrity', 'C-terminal null (Arg847*) → B9D1/TMEM231 lost → elevated NPHP-like TIN risk (~45%)'],
            ['SMOOTHENED (SMO) — indirect', 'Lipid gate (whole complex)', 'Cholesterol-rich gate allows SMO ciliary entry', 'TCTN1 LOF → gate fails → SMO excluded → Hedgehog failure → MTS + cerebellar hypoplasia'],
          ]}
        />
        <div className="alert alert-info small mt-2">
          <strong>Key insight:</strong> TCTN1 bridges the three major TZ modules (tectonic, MKS, NPHP) through distinct
          domain interactions. This explains why TCTN1 loss causes a broad ciliopathy phenotype (brain + retina + kidney)
          despite a single gene defect — it is a central cross-module scaffold at the lipid gate of the transition zone.
        </div>
      </Section>

      <Section title="No MKS Tier — TCTN1 vs Other MKS-Tier JBTS Genes" color={ACCENT7}>
        <DataTable
          headers={['Gene', 'JBTS Type', 'Biallelic Null Outcome', 'MKS Disease', 'MKS OMIM', 'Key Difference']}
          rows={[
            ['TCTN1', 'JBTS11', '✅ LIVE BIRTH — JBTS11', 'None (no MKS tier)', '—', 'TCTN1 biallelic null is NOT lethal; JBTS11 survivable'],
            ['TCTN2', '(JBTS variant)', '❌ PERINATAL LETHAL — MKS8', 'Meckel-Gruber Syndrome 8', '#613885', 'TCTN2 biallelic null → MKS8 (encephalocele + PKD + lethal)'],
            ['CEP290', 'JBTS5', '❌ LETHAL — MKS4', 'MKS4', '#611134', 'CEP290 most common — biallelic null → MKS4 lethal'],
            ['TMEM67', 'JBTS6', '❌ LETHAL — MKS3', 'MKS3', '#607361', 'TMEM67 null/null → MKS3 lethal; one null/hypomorphic → JBTS6'],
            ['RPGRIP1L', 'JBTS7', '❌ LETHAL — MKS5', 'MKS5', '#611561', 'RPGRIP1L null/null → MKS5 lethal; Ala229Thr founder → JBTS7 mild'],
            ['CC2D2A', 'JBTS9', '❌ LETHAL — MKS6', 'MKS6', '#612284', 'CC2D2A null/null → MKS6 lethal; null/missense → JBTS9'],
          ]}
        />
        <div className="alert alert-warning small mt-2">
          <strong>Clinical rule:</strong> If a ciliopathy family presents with PERINATAL LETHAL phenotype
          (encephalocele + polydactyly + polycystic kidneys = MKS8 pattern) AND the chromosome 12q region is implicated
          → suspect TCTN2 (12q24.31), NOT TCTN1 (12q24.11). Both genes are on chromosome 12q but at different loci.
          The distinction is critical: TCTN2 null/null → MKS8 (lethal); TCTN1 null/null → JBTS11 (live birth).
        </div>
      </Section>

      <Section title="Renal & Retinal Surveillance — JBTS11 Annual Protocol" color={ACCENT3}>
        <DataTable
          headers={['Surveillance', 'Frequency', 'Tools', 'Action Threshold', 'Curative?']}
          rows={[
            ['ERG + ophthalmology (retinal)', 'Annual from diagnosis', 'ERG, fundoscopy, OCT (when cooperative)', 'ERG amplitude decline → low vision aids; photoreceptor degeneration → no reversible therapy 2026', 'No (symptomatic support only)'],
            ['eGFR + urine protein/creatinine ratio', 'Annual from diagnosis', 'Serum creatinine, cystatin C, spot urine PCR', 'eGFR <60 or PCR >200 mg/g → nephrology; ACE-I for proteinuria', 'Transplant curative for ESRD'],
            ['Renal ultrasound', 'Annual', 'US kidneys + bladder', 'Increased echogenicity / cysts → nephrology referral; NPHP-like TIN pattern', 'Transplant curative'],
            ['LFTs + liver ultrasound', 'Annual (if CHF suspected)', 'ALT, GGT, ALP, bilirubin + liver USS', 'Portal HTN signs → gastroenterology / hepatology; varices surveillance', 'Liver transplant (very rare — only if ESRD + PHT combined)'],
            ['Blood pressure', 'Annual', 'Office BP + 24h ABPM if hypertensive', 'HTN from renal disease → ACE-I/ARB; target <130/80', 'Controlled with medication'],
          ]}
        />
      </Section>

      <Section title="Navigation — Adjacent Joubert Syndrome Dashboards" color={ACCENT2}>
        <ul className="list-unstyled small">
          <li><Link href="/jbts10" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS10 (OFD1) — X-linked / Centriolar Satellite / RP23 Allelic</Link></li>
          <li><Link href="/jbts9" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS9 (CC2D2A) — TZ Scaffold / MKS6 / COACH</Link></li>
          <li><Link href="/jbts8" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS8 (ARL13B) — INPP5E Trafficking / Lipid Axis</Link></li>
          <li><Link href="/jbts7" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS7 (RPGRIP1L) — TZ Y-Link / MKS5</Link></li>
          <li><Link href="/jbts6" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS6 (TMEM67) — MKS3 / COACH / Hepatic</Link></li>
          <li><Link href="/jbts5" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS5 (CEP290) — MKS4 / Most Common JBTS</Link></li>
          <li><Link href="/jbts4" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS4 (NPHP1) — Renal Dominant</Link></li>
          <li><Link href="/jbts3" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS3 (AHI1) — Retinal Dominant</Link></li>
        </ul>
      </Section>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Section title="Glossary — TCTN1 / JBTS11 Key Terms" color={ACCENT2}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={Object.entries(data).map(([k, v]) => [
            <strong key={k} style={{ color: ACCENT2 }}>{k}</strong>, v,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function JBTS11Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts11/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts11/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts11/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefs(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div>
      {/* Header */}
      <div className="px-4 py-3" style={{ background: ACCENT, color: '#fff' }}>
        <div className="d-flex align-items-center gap-3">
          <Link href="/" className="text-white text-decoration-none">← Home</Link>
          <div>
            <h4 className="mb-0 fw-bold">🧬 TCTN1 — Joubert Syndrome Type 11 (JBTS11) / Tectonic Complex / Lipid Gate</h4>
            <div className="small opacity-75">
              AR Tectonic Complex / TZ Lipid Gate / No MKS Tier · 12q24.11 · OMIM Gene *609863 · JBTS11 #614170 · {N_COHORT}-patient cohort (seed-{SEED})
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="px-4 pt-3">
        <ul className="nav nav-tabs mb-3">
          {TABS.map((t, i) => (
            <li key={i} className="nav-item">
              <button
                className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
                onClick={() => setTab(i)}
              >{t}</button>
            </li>
          ))}
        </ul>

        {error && (
          <div className="alert alert-danger small">API error: {error}</div>
        )}

        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <BreakdownTab data={breakdown} />}
        {tab === 2 && <PearlTab data={overview} />}
        {tab === 3 && <DefinitionsTab data={defs} />}
      </div>
    </div>
  );
}
