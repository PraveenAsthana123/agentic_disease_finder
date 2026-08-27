'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CEP290 Allele-Tier & IVS26 Pearl', 'Definitions'];

// JBTS5 colour scheme — CEP290 / most-common JBTS gene / retinal-neurological-renal
const ACCENT  = '#1a237e';   // deep indigo — JBTS/MTS neurological
const ACCENT2 = '#b71c1c';   // deep red — HIGH retinal burden (57%) / LCA10 overlap
const ACCENT3 = '#006064';   // dark cyan — renal NPHP6 / transition zone
const ACCENT4 = '#1b5e20';   // deep green — ASO/CRISPR therapy (sepofarsen/EDIT-101)
const ACCENT5 = '#e65100';   // burnt orange — IVS26 diagnostic pearl / panel miss
const ACCENT6 = '#4a148c';   // deep purple — allele-class 5-tier rule
const ACCENT7 = '#f57f17';   // amber — pleiotropic gene spectrum (JBTS/LCA/NPHP/MKS/BBS)
const ACCENT8 = '#37474f';   // dark slate — cerebellar ataxia / movement
const ACCENT9 = '#880e4f';   // deep crimson — MKS4 lethal extreme / biallelic null

const SEED = 417;
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
  const { kpis = [], hallmark, critical_diagnostic_pearl, allele_phenotype_rule, prevalence, first_description } = data;
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ CRITICAL DIAGNOSTIC PEARL — IVS26 ALLELE INVISIBLE TO WES/PANELS">
        {critical_diagnostic_pearl}
      </AlertBox>
      <AlertBox color={ACCENT6} title="⚡ ALLELE-CLASS → 5-TIER DISEASE SPECTRUM RULE">
        {allele_phenotype_rule}
      </AlertBox>
      <AlertBox color={ACCENT4} title="🧬 THERAPY FRONTIER — SEPOFARSEN / EDIT-101 (ASO + CRISPR for IVS26)">
        CEP290 is the ONLY ciliopathy gene with active ASO (sepofarsen/QR-110) AND in-vivo CRISPR (EDIT-101)
        therapy trials for LCA10 (IVS26 carriers). Confirm IVS26 status before referring to trial.
        ASO/CRISPR apply ONLY to retinal endpoint; MTS, renal, and neurological features are NOT corrected.
      </AlertBox>
      <div className="row g-2 mb-4">
        {kpis.map((k, i) => <KPI key={i} {...k} />)}
      </div>
      <Section title="Hallmark & Clinical Signature" color={ACCENT}>
        <p className="small">{hallmark}</p>
        <DataTable
          headers={['Feature', 'Prevalence', 'Notes']}
          rows={[
            ['Molar Tooth Sign (MTS)', '100% in JBTS5 tier', 'Pathognomonic; elongated SCP on axial brain MRI'],
            ['Retinal Dystrophy', '~57%', 'Highest retinal burden of all JBTS types; rod-cone ERG pattern; LCA10 overlap'],
            ['Renal (NPHP-type)', '~35%', 'Tubulointerstitial nephritis; variable age onset; NPHP6 allele tier → 100%'],
            ['Oculomotor Apraxia', '~55%', 'Horizontal gaze initiation failure; intermediate frequency'],
            ['Cerebellar Ataxia', '~90%', 'Truncal → limb; progressive; physio/OT support'],
            ['Neonatal Hypotonia', '~88%', 'Universal near; hypotonia preceding motor milestone delay'],
            ['Breathing Dysregulation', '~60%', 'Episodic apnea/hyperpnea; neonatal NICU monitoring'],
            ['Intellectual Disability', '~73%', 'Moderate–severe; special education referral'],
            ['Hepatic Fibrosis', '~8%', 'Rare; ductal plate malformation; MKS4 overlap alleles'],
            ['Polydactyly', '~8%', 'Rare; post-axial; BBS14 overlap alleles'],
          ]}
        />
      </Section>
      <Section title="Prevalence & Genetics" color={ACCENT3}>
        <p className="small">{prevalence}</p>
        <DataTable
          headers={['Parameter', 'Value']}
          rows={[
            ['Gene', 'CEP290 (OMIM *610142)'],
            ['Chromosome', '12q21.32'],
            ['Protein', 'Centrosomal Protein 290 kDa (2479 aa)'],
            ['Inheritance', 'Autosomal Recessive — biallelic LOF'],
            ['JBTS5 Prevalence', '~10–15% of all Joubert syndrome (most common JBTS gene)'],
            ['JBTS5 Birth Freq.', '~1/500,000–700,000 worldwide'],
            ['CEP290 LCA10', '~1/80,000–100,000 (retinal only, IVS26 biallelic)'],
            ['First description', '2006 (Sayer / Valente / den Hollander — tri-locus discovery)'],
          ]}
        />
      </Section>
      <Section title="Historical / Discovery" color={ACCENT8}>
        <p className="small">{first_description}</p>
      </Section>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ─────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const {
    allele_distribution = [],
    ethnicity_distribution = [],
    age_at_diagnosis = [],
    feature_prevalence = [],
    ivs26_summary = {},
    tier_comparison = [],
    sample_patients = [],
  } = data;

  return (
    <div>
      <AlertBox color={ACCENT5} title={`⚠ IVS26 Panel Miss: ${ivs26_summary.panel_missed_pct ?? '?'}% of cohort`}>
        {ivs26_summary.message}
      </AlertBox>

      <div className="row mb-4">
        <div className="col-md-6">
          <Section title="Allele Class Distribution" color={ACCENT6}>
            <DataTable
              headers={['Allele Class', 'Count', '%']}
              rows={allele_distribution.map(a => [a.allele_class, a.count, `${a.pct}%`])}
            />
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT3}>
            <DataTable
              headers={['Ethnicity', 'Count', '%']}
              rows={ethnicity_distribution.map(e => [e.ethnicity, e.count, `${e.pct}%`])}
            />
          </Section>
        </div>
      </div>

      <div className="row mb-4">
        <div className="col-md-6">
          <Section title="Age at Diagnosis" color={ACCENT8}>
            <DataTable
              headers={['Age Range', 'Count', '%']}
              rows={age_at_diagnosis.map(a => [a.bin, a.count, `${a.pct}%`])}
            />
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Feature Prevalence (cohort)" color={ACCENT2}>
            <DataTable
              headers={['Feature', 'Count', '%']}
              rows={feature_prevalence.map(f => [f.feature, f.count, `${f.pct}%`])}
            />
          </Section>
        </div>
      </div>

      <Section title="CEP290 Allele-Class → Disease Tier Comparison" color={ACCENT6}>
        <DataTable
          headers={['Disease Tier', 'Allele Class', 'Retinal', 'Renal', 'MTS']}
          rows={tier_comparison.map(t => [t.tier, t.allele_class, t.retinal, t.renal, t.mts])}
        />
      </Section>

      <Section title="Sample Patients (first 20)" color={ACCENT}>
        <DataTable
          headers={['ID', 'Ethnicity', 'Allele Class (abbrev.)', 'Age Dx', 'Retinal', 'Renal', 'OMA', 'IVS26', 'IVS26 Missed?']}
          rows={sample_patients.map(p => [
            p.id, p.ethnicity, p.allele_class, p.age_dx,
            p.retinal, p.renal, p.oma, p.ivs26, p.ivs26_missed,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Tab: CEP290 Allele-Tier & IVS26 Pearl ────────────────────────────────────
function AlleleTierTab({ overview, breakdown }) {
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ IVS26 (c.2991+1655A>G) — THE INVISIBLE ALLELE">
        The IVS26 deep intronic allele is THE most common CEP290 disease allele in European LCA10/JBTS5.
        It is a cryptic splice site mutation creating a 128-bp pseudo-exon in intron 26. Standard WES and
        gene panels DO NOT capture deep intronic variants — IVS26 is systematically missed until
        RNA/WGS or targeted assay is performed. ALWAYS add IVS26 targeted testing when CEP290 clinical
        suspicion + negative panel.
      </AlertBox>

      <Section title="CEP290 5-Tier Allele-Phenotype Map" color={ACCENT6}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small">
            <thead className="table-dark">
              <tr>
                <th>Allele 1</th><th>Allele 2</th><th>Disease Tier</th><th>Key Feature</th><th>MTS</th>
              </tr>
            </thead>
            <tbody>
              <tr className="table-danger">
                <td>NULL (truncating)</td><td>NULL (truncating)</td>
                <td><strong>MKS4</strong> — Meckel-Gruber lethal</td>
                <td>Occipital encephalocele, polycystic kidneys, polydactyly</td>
                <td>N/A (lethal)</td>
              </tr>
              <tr className="table-warning">
                <td>NULL (truncating)</td><td>HYPOMORPHIC (IVS26/missense)</td>
                <td><strong>JBTS5</strong> — Joubert syndrome 5</td>
                <td>MTS + retinal (~57%) + renal (~35%) + ataxia</td>
                <td>YES</td>
              </tr>
              <tr className="table-info">
                <td>IVS26 (hypomorphic)</td><td>IVS26 (hypomorphic)</td>
                <td><strong>LCA10</strong> — retinal only</td>
                <td>Severe early-onset retinal dystrophy; NO MTS; ASO/CRISPR candidate</td>
                <td>NO</td>
              </tr>
              <tr className="table-secondary">
                <td>HYPOMORPHIC (C-term)</td><td>TRUNCATING (NPHP5 domain)</td>
                <td><strong>NPHP6</strong> — renal → ESRD</td>
                <td>Tubulointerstitial nephritis; renal transplant curative</td>
                <td>Variable</td>
              </tr>
              <tr className="table-light">
                <td>MILD missense</td><td>MILD missense</td>
                <td><strong>BBS14</strong> — Bardet-Biedl 14</td>
                <td>Obesity + retinal; BBSome partial dysfunction; rare</td>
                <td>Rare</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Section>

      <Section title="CEP290 Protein Domain Map & Pathogenic Variants" color={ACCENT3}>
        <DataTable
          headers={['Domain', 'aa Range', 'Function', 'Key Variants']}
          rows={[
            ['N-terminal CC repeats', 'aa 1–200', 'Self-association; centrosome targeting; PCM1 contact', 'p.Cys89Tyr (South Asian) — PCM1 binding lost'],
            ['SMC-like hinge', 'aa 200–600', 'Ciliary gate scaffolding; structural backbone', 'Multiple frameshift nulls in this region'],
            ['Central CC1', 'aa 600–1100', 'KIF3A kinesin binding; IFT entry platform', 'IVS26 pseudo-exon inserts ~here; reading frame disrupted'],
            ['Central CC2 / NPHP5-binding', 'aa 1100–1700', 'NPHP5 (IQCB1) interaction — retinal photoreceptor critical', 'Truncating here → NPHP6 / retinal-severe'],
            ['C-terminal CC', 'aa 1700–2479', 'Calmodulin binding; TZ Y-link anchoring', 'p.Trp1891Ter · p.Arg1933Ter · p.Glu2089Lys · p.Leu2195Pro'],
          ]}
        />
      </Section>

      <Section title="Diagnostic Workflow — CEP290 Suspicion" color={ACCENT7}>
        <DataTable
          headers={['Step', 'Action', 'Key Point']}
          rows={[
            ['1', 'Brain MRI — MTS confirmation', 'Elongated SCP + vermis hypoplasia = JBTS5 tier (NOT LCA10)'],
            ['2', 'Full ophthalmology + ERG (photopic + scotopic)', 'Rod-cone pattern; high retinal in JBTS5; confirm not LCA10 (retinal only)'],
            ['3', 'Gene panel / WES with CEP290', 'Standard sequencing — detects coding variants; WILL MISS IVS26'],
            ['4', 'If panel negative + CEP290 suspected → IVS26 targeted assay', 'c.2991+1655A>G PCR / Sanger or RNA/cDNA from blood/fibroblasts'],
            ['5', 'If IVS26 negative → genome sequencing (WGS)', 'Detects all deep intronic + SV; mandated if clinical picture strong'],
            ['6', 'Renal function: creatinine, eGFR, urinalysis, renal US', 'NPHP6 tier surveillance; annual'],
            ['7', 'Allele class interpretation → assign disease tier', 'Apply 5-tier rule; confirm MTS vs no-MTS before JBTS5 vs LCA10 label'],
            ['8', 'If IVS26 confirmed → refer to sepofarsen/EDIT-101 trial', 'Retinal endpoint ONLY; MTS/renal/neurological unaffected by ASO/CRISPR'],
          ]}
        />
      </Section>

      <Section title="Therapy Landscape — CEP290 / JBTS5 / LCA10" color={ACCENT4}>
        <DataTable
          headers={['Therapy', 'Mechanism', 'Target', 'Status (2026)', 'Eligibility']}
          rows={[
            ['Sepofarsen (QR-110)', 'ASO — blocks IVS26 pseudo-exon splicing → restores correct CEP290 mRNA', 'Retinal photoreceptors', 'Phase 2/3 (ProQR)', 'IVS26 carriers (homozygous ideal)'],
            ['EDIT-101', 'In-vivo CRISPR — edits IVS26 pseudo-exon donor site in subretinal space', 'Retinal photoreceptors', 'Phase 1/2 (Editas Medicine)', 'IVS26 carriers'],
            ['Renal transplant', 'Organ replacement — curative for NPHP6 renal endpoint', 'Kidney', 'Standard of care', 'NPHP6/JBTS5 with ESRD'],
            ['ACEi/ARB', 'Proteinuria reduction; renal protection', 'Renal', 'Standard of care', 'All JBTS5 with proteinuria'],
            ['Supportive (OT/PT)', 'Ataxia management, mobility, ADL', 'Neurological', 'Standard of care', 'All JBTS5'],
          ]}
        />
      </Section>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { gene, diseases, key_variants, molecular_mechanism, treatment, surveillance, glossary } = data;

  return (
    <div>
      <Section title="Gene Summary" color={ACCENT}>
        <DataTable
          headers={['Property', 'Value']}
          rows={[
            ['Symbol', gene?.symbol],
            ['Full name', gene?.full_name],
            ['Aliases', (gene?.aliases || []).join(', ')],
            ['OMIM Gene', gene?.omim_gene],
            ['Chromosome', gene?.chromosome],
            ['Protein size', gene?.protein_size],
            ['Expression', gene?.expression],
            ['Subcellular location', gene?.subcellular_location],
          ]}
        />
      </Section>

      <Section title="Disease Spectrum (Allele-Dependent)" color={ACCENT6}>
        <DataTable
          headers={['OMIM', 'Disease', 'Allele Class', 'Key Features']}
          rows={(diseases || []).map(d => [d.omim, d.name, d.allele_class, d.key_features])}
        />
      </Section>

      <Section title="Key Pathogenic Variants" color={ACCENT3}>
        <DataTable
          headers={['Variant', 'Protein Effect', 'Location', 'Population', 'Disease Tier', 'Note']}
          rows={(key_variants || []).map(v => [
            v.variant, v.protein, v.location, v.population, v.disease_tier, v.note,
          ])}
        />
      </Section>

      <Section title="Molecular Mechanism" color={ACCENT5}>
        <p className="small">{molecular_mechanism}</p>
      </Section>

      <Section title="Treatment" color={ACCENT4}>
        <DataTable
          headers={['Target', 'Intervention']}
          rows={Object.entries(treatment || {}).map(([k, v]) => [k.replace(/_/g, ' ').toUpperCase(), v])}
        />
      </Section>

      <Section title="Surveillance Protocol" color={ACCENT8}>
        <ul className="small ps-3">
          {(surveillance || []).map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      </Section>

      <Section title="Glossary" color={ACCENT7}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={Object.entries(glossary || {}).map(([k, v]) => [k, v])}
        />
      </Section>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function JBTS5Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts5/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts5/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts5/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, def]) => {
        setOverview(ov);
        setBreakdown(br);
        setDefinitions(def);
      })
      .catch(e => setError(String(e)));
  }, []);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', fontFamily: 'sans-serif' }}>
      {/* ── sidebar nav ── */}
      <nav style={{ width: 260, background: '#1a237e', color: '#fff', padding: '1rem', flexShrink: 0 }}>
        <div className="fw-bold mb-3" style={{ fontSize: '1rem' }}>
          🧬 CEP290 / JBTS5
        </div>
        <div className="small mb-2 text-white-50">Joubert Syndrome Type 5</div>
        <ul className="list-unstyled">
          {TABS.map((t, i) => (
            <li key={i} className="mb-1">
              <button
                className="btn btn-sm w-100 text-start"
                style={{
                  background: tab === i ? '#f57f17' : 'transparent',
                  color: '#fff',
                  border: 'none',
                }}
                onClick={() => setTab(i)}
              >
                {t}
              </button>
            </li>
          ))}
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small text-white-50 mb-2">Related Dashboards</div>
        <ul className="list-unstyled small">
          <li><Link href="/jbts3" className="text-white-50">← JBTS3 (AHI1)</Link></li>
          <li><Link href="/jbts4" className="text-white-50">← JBTS4 (NPHP1)</Link></li>
          <li><Link href="/nphp" className="text-white-50">NPHP series →</Link></li>
          <li><Link href="/bbs14" className="text-white-50">BBS14 (CEP290) →</Link></li>
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small text-white-50">
          Cohort: {N_COHORT} patients · seed {SEED}
          <br />OMIM: #610188 · Gene: *610142
          <br />IVS26 Pearl: c.2991+1655A>G
        </div>
      </nav>

      {/* ── main content ── */}
      <main style={{ flex: 1, padding: '1.5rem', overflowY: 'auto' }}>
        <div className="d-flex align-items-center mb-1">
          <h4 className="mb-0 me-3" style={{ color: ACCENT }}>
            CEP290 — Joubert Syndrome Type 5 (JBTS5)
          </h4>
          <span className="badge" style={{ background: ACCENT5, fontSize: '0.75rem' }}>
            Most Common JBTS Gene (~10–15% of all JBTS)
          </span>
        </div>
        <div className="text-muted small mb-3">
          NPHP6 · MKS4 · BBS14 · LCA10 · 12q21.32 · 2479 aa · OMIM #610188 · seed {SEED} · {N_COHORT} patients
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
