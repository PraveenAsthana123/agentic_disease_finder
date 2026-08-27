'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'No-MKS Tier & Pathway Pearl', 'Definitions'];

// JBTS8 colour scheme — ARL13B / ciliary GTPase / INPP5E trafficking / no MKS tier
const ACCENT  = '#1a237e';   // deep indigo — JBTS8 GTPase / MTS neurological
const ACCENT2 = '#880e4f';   // deep crimson — INPP5E PI(4,5)P₂ accumulation axis
const ACCENT3 = '#004d40';   // deep teal — renal TIN (mild)
const ACCENT4 = '#1b5e20';   // deep green — transplant / curative renal
const ACCENT5 = '#e65100';   // burnt orange — No-MKS-Tier unique pearl
const ACCENT6 = '#4a148c';   // deep purple — ARL3 GEF / C-terminal CC
const ACCENT7 = '#f57f17';   // amber — Arg79Gln founder allele
const ACCENT8 = '#37474f';   // dark slate — cerebellar ataxia / OMA
const ACCENT9 = '#b71c1c';   // deep red — retinal INPP5E-ARL13B axis

const SEED = 423;
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
  const { kpis = [], hallmark, critical_diagnostic_pearl, inpp5e_axis_pearl,
          no_mks_unique, prevalence, first_description, gene_summary,
          phenotype_summary, allele_class_distribution = [] } = data;
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ UNIQUE FEATURE — ARL13B HAS NO MKS-LETHAL ALLELE TIER">
        {no_mks_unique}
      </AlertBox>
      <AlertBox color={ACCENT7} title="⚠ CRITICAL DIAGNOSTIC PEARL — Arg79Gln NORTH AFRICAN / SOUTHERN EUROPEAN FOUNDER (c.236G>A)">
        {critical_diagnostic_pearl}
      </AlertBox>
      <AlertBox color={ACCENT2} title="⚠ INPP5E-ARL13B PI METABOLISM AXIS — JBTS1 + JBTS8 FUNCTIONAL EQUIVALENCE">
        {inpp5e_axis_pearl}
      </AlertBox>

      <Section title="KPI Summary — JBTS8 Cohort (N=40, seed-423)" color={ACCENT}>
        <div className="row g-2">
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Gene & Disease Summary" color={ACCENT}>
            <table className="table table-sm table-bordered small">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>ARL13B (OMIM *608922)</td></tr>
                <tr><td className="fw-bold">Disease</td><td>Joubert Syndrome 8 (OMIM #612291)</td></tr>
                <tr><td className="fw-bold">Chromosome</td><td>3q11.1</td></tr>
                <tr><td className="fw-bold">Protein</td><td>428 aa — GTPase / ALPS motif / ARL3 GEF coiled-coil</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF</td></tr>
                <tr><td className="fw-bold">Prevalence</td><td>~1% all JBTS; ~1/2,000,000–5,000,000 worldwide</td></tr>
                <tr><td className="fw-bold">MKS Tier</td><td style={{color: ACCENT5, fontWeight:'bold'}}>NONE — all biallelic LOF → JBTS8</td></tr>
                <tr><td className="fw-bold">First Description</td><td>{first_description}</td></tr>
                <tr><td className="fw-bold">Hallmark</td><td>{hallmark}</td></tr>
              </tbody>
            </table>
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Phenotype Frequencies (N=40)" color={ACCENT8}>
            <DataTable
              headers={['Feature', 'Frequency', 'Notes']}
              rows={[
                ['Molar Tooth Sign (MTS)', '100%', 'Pathognomonic — all JBTS8'],
                ['Cerebellar Ataxia', `${phenotype_summary?.ataxia_pct ?? '–'}%`, 'Core feature; SARA tracking'],
                ['Neonatal Hypotonia', `${phenotype_summary?.hypotonia_pct ?? '–'}%`, 'Feeding difficulty in infancy'],
                ['Oculomotor Apraxia', `${phenotype_summary?.oma_pct ?? '–'}%`, 'Head thrust compensation'],
                ['Intellectual Disability', `${phenotype_summary?.id_pct ?? '–'}%`, 'Mild–moderate range'],
                ['Breathing Dysregulation', `${phenotype_summary?.breathing_pct ?? '–'}%`, 'Episodic apnea/hyperpnea'],
                ['Retinal Dystrophy', `${phenotype_summary?.retinal_pct ?? '–'}%`, 'Rod-cone; INPP5E-ARL13B axis; ERG annual'],
                ['Renal (TIN)', `${phenotype_summary?.renal_pct ?? '–'}%`, 'Mild; ESRD risk ~25yr median'],
                ['Polydactyly', `${phenotype_summary?.polydactyly_pct ?? '–'}%`, 'Very rare; not a core feature'],
                ['Hepatic Fibrosis', '0%', 'NEVER — no COACH in ARL13B'],
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

      <Section title="Gene Function Summary" color={ACCENT}>
        <p className="small">{gene_summary}</p>
      </Section>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ────────────────────────────────────────────────
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
      <Section title="Ethnicity Distribution" color={ACCENT}>
        <DataTable
          headers={['Ethnicity', 'Count', '%']}
          rows={ethnicity_distribution.map(e => [e.ethnicity, e.count, `${e.pct}%`])}
        />
      </Section>

      <Section title="Key Pathogenic Variants (ARL13B)" color={ACCENT7}>
        <DataTable
          headers={['Variant', 'Domain', 'Effect', 'Population', 'Phenotype', 'OMIM Class']}
          rows={key_variants.map(v => [
            <strong key={v.variant} style={{ color: ACCENT7 }}>{v.variant}</strong>,
            v.domain, v.effect, v.population, v.phenotype, v.omim_class,
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

      <Section title="ARL13B → INPP5E → PI Metabolism → Hedgehog Pathway" color={ACCENT2}>
        <DataTable
          headers={['Step', 'Normal Event', 'Effect When ARL13B Lost']}
          rows={pathway_steps.map(s => [s.step, s.event, s.effect_when_lost])}
        />
      </Section>

      <Section title="Clinical Management" color={ACCENT4}>
        <DataTable
          headers={['Intervention', 'Timing', 'Rationale', 'Level']}
          rows={management.map(m => [m.intervention, m.timing, m.rationale, m.level])}
        />
      </Section>

      <Section title="Per-Patient Table (first 20)" color={ACCENT8}>
        <DataTable
          headers={['ID', 'Ethnicity', 'Allele', 'Age Dx (yr)', 'MTS', 'Ataxia', 'OMA', 'Retinal', 'Renal', 'ID', 'Breathing', 'Hepatic', 'MKS Tier']}
          rows={patient_table.map(p => [
            p.id, p.ethnicity, p.allele, p.age_dx_yr,
            p.mts, p.ataxia, p.oma, p.retinal, p.renal,
            p.id_, p.breathing, p.hepatic, p.mks_tier,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Tab: No-MKS Tier & Pathway Pearl ────────────────────────────────────────
function PearlTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ NO MKS TIER — ARL13B IS UNIQUE AMONG MAJOR JBTS GENES">
        ARL13B is the ONLY major Joubert gene without an MKS-lethal allele tier.
        Compare: CEP290 (MKS4 — biallelic null lethal), RPGRIP1L (MKS5), TMEM67 (MKS3),
        CC2D2A (MKS6), TCTN2 (MKS8). For ARL13B: ALL biallelic LOF → JBTS8 (live birth).
        No allele-class tier stratification. Genetic counselling: recurrence = JBTS8 (25%), NOT MKS.
      </AlertBox>

      <Section title="MKS Tier Comparison — Major JBTS Genes" color={ACCENT5}>
        <DataTable
          headers={['Gene', 'JBTS Type', 'Biallelic Null Tier', 'NPHP Tier', 'ARL13B Unique?']}
          rows={[
            ['ARL13B', 'JBTS8', 'JBTS8 (live birth — NO MKS)', '—', '✅ YES — no lethal tier'],
            ['CEP290', 'JBTS5', 'MKS4 (lethal)', 'NPHP6', '—'],
            ['TMEM67', 'JBTS6', 'MKS3 (lethal)', 'NPHP11', '—'],
            ['RPGRIP1L', 'JBTS7', 'MKS5 (lethal)', 'NPHP8', '—'],
            ['AHI1', 'JBTS3', 'JBTS3 (no MKS)', '—', 'ARL13B similar'],
            ['NPHP1', 'JBTS4', 'Pure NPHP (no MTS)', 'NPHP1', '—'],
          ]}
        />
      </Section>

      <Section title="INPP5E–ARL13B Functional Axis (JBTS1 + JBTS8)" color={ACCENT2}>
        <DataTable
          headers={['Gene', 'OMIM', 'Function in Pathway', 'If Lost']}
          rows={[
            ['ARL13B', 'JBTS8 #612291', 'Recruits INPP5E to cilia; activates ARL3 (GEF)', 'INPP5E cannot reach cilia → PI(4,5)P₂ accumulates'],
            ['INPP5E', 'JBTS1 #213300', 'Converts PI(4,5)P₂ → PI(4)P at ciliary tip', 'PI(4,5)P₂ accumulates → SMO excluded → Hedgehog fails'],
            ['ARL3', '(substrate)', 'ARL3-GTP releases prenylated cargoes from PDEδ/UNC119', 'Prenylated cargoes (INPP5E/PDE6) stuck in cytosol'],
            ['SMO (GPCR)', '(downstream)', 'Enters cilia when PI(4)P-rich; activates GLI', 'Excluded → no GLI activation → no Hedgehog → MTS'],
          ]}
        />
        <div className="alert alert-info small mt-2">
          <strong>Clinical implication:</strong> WES gene panels for unsolved JBTS must include BOTH ARL13B (JBTS8) and INPP5E (JBTS1).
          Variants in either gene produce functionally identical PI(4,5)P₂ accumulation and identical MTS phenotype.
          When ARL13B sequencing is negative in a JBTS8-compatible presentation, immediately prioritise INPP5E analysis (and vice versa).
        </div>
      </Section>

      <Section title="Arg79Gln Founder Allele — Clinical Decision Guide" color={ACCENT7}>
        <DataTable
          headers={['Genotype', 'Expected Phenotype', 'Retinal Risk', 'Renal Risk', 'Counselling']}
          rows={[
            ['Arg79Gln / Arg79Gln (hom)', 'Mild JBTS8 — MTS + mild ataxia + OMA', 'Low (~20%)', 'Low (~8%)', 'JBTS8 recurrence 25%; no MKS concern'],
            ['Arg79Gln / null (compound het)', 'Moderate JBTS8 — MTS + ataxia + OMA', 'Moderate (~30%)', 'Moderate (~18%)', 'JBTS8 recurrence 25%; no MKS concern'],
            ['Null / null', 'Moderate-severe JBTS8', 'Higher (~35%)', 'Moderate (~25%)', 'JBTS8 recurrence 25%; still NOT MKS'],
          ]}
        />
      </Section>

      <Section title="Allele-Class Tier Rule (Simplified vs Other JBTS Genes)" color={ACCENT6}>
        <div className="alert small" style={{ borderLeft: `5px solid ${ACCENT6}`, background: '#fafafa' }}>
          <strong style={{ color: ACCENT6 }}>ARL13B (JBTS8) — NO tier stratification required:</strong>
          <ul className="mb-0 mt-1">
            <li>Biallelic null/null → <strong>JBTS8</strong> (moderate-severe; NOT MKS)</li>
            <li>Null + hypomorphic missense → <strong>JBTS8</strong> (moderate)</li>
            <li>Missense/missense → <strong>JBTS8</strong> (mild–moderate)</li>
          </ul>
          <hr />
          <strong style={{ color: ACCENT5 }}>Compare — RPGRIP1L (JBTS7) tier rule:</strong>
          <ul className="mb-0 mt-1">
            <li>Biallelic null → <strong>MKS5</strong> (lethal — encephalocele + polydactyly + PKD)</li>
            <li>Null + hypomorphic → <strong>JBTS7</strong></li>
            <li>Biallelic hypomorphic → <strong>JBTS7 mild or NPHP8</strong></li>
          </ul>
        </div>
      </Section>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Section title="Glossary — ARL13B / JBTS8 Key Terms" color={ACCENT}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={Object.entries(data).map(([k, v]) => [
            <strong key={k} style={{ color: ACCENT }}>{k}</strong>, v,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function JBTS8Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts8/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts8/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts8/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold">🧬 ARL13B — Joubert Syndrome Type 8 (JBTS8)</h4>
            <div className="small opacity-75">
              GTPase / INPP5E Trafficking / No-MKS-Tier · 3q11.1 · OMIM Gene *608922 · Disease #612291 · {N_COHORT}-patient cohort (seed-{SEED})
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
