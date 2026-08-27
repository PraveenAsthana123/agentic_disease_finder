'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'X-Linked & RP23 Pearls', 'Definitions'];

// JBTS10 colour scheme — OFD1 / X-linked / centriolar satellite / RP23 / OFD1 syndrome
const ACCENT  = '#4e342e';   // dark brown — X-linked / OFD1 / oral-facial-digital warning
const ACCENT2 = '#1a237e';   // deep indigo — MTS neurological / ciliogenesis
const ACCENT3 = '#004d40';   // deep teal — renal NPHP-like
const ACCENT4 = '#1b5e20';   // deep green — transplant / curative endpoint
const ACCENT5 = '#b71c1c';   // deep red — retinal RP23 high burden warning
const ACCENT6 = '#37474f';   // dark slate — centriolar satellite / basal body
const ACCENT7 = '#4a148c';   // deep purple — X-linked inheritance / carrier female
const ACCENT8 = '#e65100';   // burnt orange — polydactyly / JSOFD / OFD1 syndrome
const ACCENT9 = '#006064';   // dark cyan — OFD1 syndrome / carrier female features

const SEED = 427;
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
  const { kpis = [], hallmark, xlinked_pearl, rp23_pearl, polydactyly_pearl,
          first_description, gene_summary, phenotype_summary,
          allele_class_distribution = [] } = data;
  return (
    <div>
      <AlertBox color={ACCENT7} title="⚠ X-LINKED INHERITANCE — JBTS10 IS THE ONLY X-LINKED JBTS TYPE (JBTS3-10 SERIES)">
        {xlinked_pearl}
      </AlertBox>
      <AlertBox color={ACCENT5} title="⚠ RP23 ALLELIC AXIS — HIGHEST RETINAL DISEASE BURDEN OF JBTS3-10 (~55%)">
        {rp23_pearl}
      </AlertBox>
      <AlertBox color={ACCENT8} title="⚠ POLYDACTYLY ~25% — X-LINKED JSOFD; MATERNAL OFD1 SYNDROME ASSESSMENT MANDATORY">
        {polydactyly_pearl}
      </AlertBox>

      <Section title="KPI Summary — JBTS10 Cohort (N=40, seed-427)" color={ACCENT2}>
        <div className="row g-2">
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Gene & Disease Summary" color={ACCENT2}>
            <table className="table table-sm table-bordered small">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>OFD1 (OMIM *300170)</td></tr>
                <tr><td className="fw-bold">Disease JBTS10</td><td>Joubert Syndrome 10 (OMIM #300804) — X-linked</td></tr>
                <tr><td className="fw-bold">Disease OFD1 Syndrome</td><td style={{color: ACCENT9, fontWeight:'bold'}}>Oral-Facial-Digital Syndrome Type 1 (OMIM #311200) — heterozygous females; X-linked dominant</td></tr>
                <tr><td className="fw-bold">Disease RP23</td><td style={{color: ACCENT5, fontWeight:'bold'}}>Retinitis Pigmentosa 23 (OMIM #300424) — retinal isoform; X-linked; allelic to JBTS10</td></tr>
                <tr><td className="fw-bold">Chromosome</td><td>Xp22.2</td></tr>
                <tr><td className="fw-bold">Protein</td><td>1012 aa — LisH (dimerisation) / CC1 (satellite targeting) / CC2-CC5 (scaffold)</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td style={{color: ACCENT7, fontWeight:'bold'}}>X-linked — hemizygous males → JBTS10; heterozygous females → OFD1 syndrome</td></tr>
                <tr><td className="fw-bold">Prevalence</td><td>~1–2% all JBTS; ~1/1,500,000–3,000,000 worldwide</td></tr>
                <tr><td className="fw-bold">MKS Tier</td><td>None — X-linked; no biallelic null mechanism (unlike JBTS5-7/9)</td></tr>
                <tr><td className="fw-bold">Retinal (RP23 axis)</td><td style={{color: ACCENT5}}>~55% rod-cone — HIGHEST of JBTS3-10; RP23 allelic</td></tr>
                <tr><td className="fw-bold">Polydactyly</td><td>~25% post-axial — X-linked JSOFD; maternal OFD1 assessment mandatory</td></tr>
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
                ['Molar Tooth Sign (MTS)', '100%', 'Pathognomonic — all JBTS10'],
                ['Cerebellar Ataxia', `${phenotype_summary?.ataxia_pct ?? '–'}%`, 'Core feature; SARA tracking'],
                ['Neonatal Hypotonia', `${phenotype_summary?.hypotonia_pct ?? '–'}%`, 'Feeding difficulty in infancy'],
                ['Oculomotor Apraxia', `${phenotype_summary?.oma_pct ?? '–'}%`, 'Head thrust compensation'],
                ['Intellectual Disability', `${phenotype_summary?.id_pct ?? '–'}%`, 'Mild–moderate range'],
                ['Breathing Dysregulation', `${phenotype_summary?.breathing_pct ?? '–'}%`, 'Episodic apnea/hyperpnea'],
                ['Retinal Dystrophy (RP23)', `${phenotype_summary?.retinal_pct ?? '–'}%`, 'HIGHEST of JBTS3-10; ERG mandatory annual'],
                ['Polydactyly (X-JSOFD)', `${phenotype_summary?.polydactyly_pct ?? '–'}%`, 'Post-axial; skeletal survey mandatory'],
                ['Renal (NPHP-like TIN)', `${phenotype_summary?.renal_pct ?? '–'}%`, 'ESRD risk ~25yr median'],
                ['Hepatic Fibrosis (CHF)', `${phenotype_summary?.hepatic_pct ?? '–'}%`, 'Rare (<5%) — OFD1 not a COACH gene'],
                ['No MKS Tier', 'X-linked — no biallelic null', 'Unlike JBTS5-7/9'],
                ['OFD1 Syndrome (carrier ♀)', '~50% daughters of hemizygous', 'Oral hamartomas, tongue nodules'],
              ]}
            />
          </Section>
        </div>
      </div>

      <Section title="Allele Class Distribution (Cohort)" color={ACCENT7}>
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
      <Section title="Ethnicity Distribution" color={ACCENT2}>
        <DataTable
          headers={['Ethnicity', 'Count', '%']}
          rows={ethnicity_distribution.map(e => [e.ethnicity, e.count, `${e.pct}%`])}
        />
      </Section>

      <Section title="Key Pathogenic Variants (OFD1)" color={ACCENT5}>
        <DataTable
          headers={['Variant', 'Domain', 'Effect', 'Population', 'Phenotype', 'OMIM Class']}
          rows={key_variants.map(v => [
            <strong key={v.variant} style={{ color: ACCENT5 }}>{v.variant}</strong>,
            v.domain, v.effect, v.population, v.phenotype, v.omim_class,
          ])}
        />
      </Section>

      <Section title="Domain → Phenotype Severity Matrix" color={ACCENT7}>
        <DataTable
          headers={['Domain', 'Key Variants', 'Function Lost', 'Severity', 'Retinal Risk', 'Renal Risk']}
          rows={domain_phenotype_matrix.map(d => [
            d.domain, d.key_variants, d.function_lost, d.severity, d.retinal_risk, d.renal_risk,
          ])}
        />
      </Section>

      <Section title="OFD1 → Centriolar Satellite → Ciliogenesis → MTS / RP23 Pathway" color={ACCENT2}>
        <DataTable
          headers={['Step', 'Normal Event', 'Effect When OFD1 Lost']}
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
          headers={['ID', 'Sex', 'Ethnicity', 'Allele', 'Age Dx', 'MTS', 'Ataxia', 'OMA', 'Retinal', 'Poly', 'Renal', 'Hepatic', 'ID', 'Breathing', 'X-linked']}
          rows={patient_table.map(p => [
            p.id, p.sex, p.ethnicity, p.allele, p.age_dx_yr,
            p.mts, p.ataxia, p.oma, p.retinal, p.poly, p.renal, p.hepatic,
            p.id_, p.breathing, p.xlinked,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Tab: X-Linked & RP23 Pearls ──────────────────────────────────────────────
function PearlTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <AlertBox color={ACCENT7} title="⚠ X-LINKED JBTS10 — UNIQUE INHERITANCE AMONG JBTS3-10; MATERNAL EXAMINATION MANDATORY">
        JBTS10 is the ONLY X-linked JBTS type. All JBTS3-9 are autosomal recessive.
        OFD1 (Xp22.2) hemizygous LOF in males → JBTS10 (MTS + ataxia + high retinal).
        Heterozygous females with the same variant → OFD1 syndrome (oral hamartomas, tongue
        lobulation, dental anomalies, hypertelorism, digital anomalies) ± mild CNS involvement.
        Severely null alleles → male lethality in utero. Males surviving to birth carry
        hypomorphic or mosaic OFD1 variants. Maternal carrier examination for OFD1 syndrome
        features is MANDATORY for X-linked JBTS pedigree characterisation. Recurrence: 50% of
        male offspring have JBTS10; 50% of female offspring have OFD1 syndrome.
      </AlertBox>

      <Section title="JBTS10 vs JBTS3-9 — Inheritance Comparison" color={ACCENT7}>
        <DataTable
          headers={['Gene', 'JBTS Type', 'Inheritance', 'MKS Tier', 'Unique Feature']}
          rows={[
            ['OFD1', 'JBTS10', 'X-linked — hemizygous males; het females = OFD1 syndrome', 'None (X-linked; no biallelic null)', '⭐ ONLY X-linked JBTS; RP23 allelic; highest retinal (~55%)'],
            ['CC2D2A', 'JBTS9', 'AR biallelic LOF', 'MKS6 (biallelic null → lethal)', 'COACH 2nd most common; JSOFD ~20%'],
            ['ARL13B', 'JBTS8', 'AR biallelic LOF', 'None (all biallelic LOF → JBTS8; no MKS)', 'INPP5E axis; PI4P ciliary identity; no CHF'],
            ['RPGRIP1L', 'JBTS7', 'AR biallelic LOF', 'MKS5 (biallelic null → lethal)', 'TZ Y-link; European Ala229Thr founder'],
            ['TMEM67', 'JBTS6', 'AR biallelic LOF', 'MKS3 (biallelic null → lethal)', 'COACH most common; FN-III Wnt sensing'],
            ['CEP290', 'JBTS5', 'AR biallelic LOF', 'MKS4 (biallelic null → lethal)', 'Most common JBTS gene (~10-15%); IVS26 deep intronic'],
            ['NPHP1', 'JBTS4', 'AR biallelic LOF', 'None', '1q22 deletion; pure renal subset'],
            ['AHI1', 'JBTS3', 'AR biallelic LOF', 'None', 'Retinal ~40%; Arg830Trp Mid-Eastern founder'],
          ]}
        />
      </Section>

      <AlertBox color={ACCENT5} title="⚠ RP23 ALLELIC AXIS — OFD1 RETINAL ISOFORM / CONNECTING CILIUM GATE">
        OFD1 variants in exons 9–11 (CC3 domain) of the shorter retinal isoform → Retinitis
        Pigmentosa 23 (RP23, OMIM #300424) without Molar Tooth Sign. These variants disrupt
        the retinal OFD1 isoform but preserve sufficient global ciliogenesis. JBTS10 patients
        carry full-gene LOF → both MTS AND high retinal disease (~55%). Key diagnostic rule:
        X-linked RP + maternal OFD1 features → sequence full OFD1 (exons 1-23 + retinal
        isoform exons 9-11). WES may MISS deep intronic or regulatory variants affecting only
        the retinal isoform. Annual ERG is mandatory from diagnosis for all JBTS10 patients.
        Ala446Thr (hypomorphic) → RP23-dominant; c.1405+1G>A splice → both MTS + high RP23 risk.
      </AlertBox>

      <Section title="OFD1 Isoform-Phenotype Matrix (JBTS10 vs RP23 vs OFD1 Syndrome)" color={ACCENT5}>
        <DataTable
          headers={['Phenotype', 'Sex', 'OFD1 Variant Class', 'Isoform Affected', 'Brain (MTS)', 'Retinal', 'Oral-Digital']}
          rows={[
            ['JBTS10', 'Hemizygous male', 'Full-gene null / truncating', 'All isoforms (global LOF)', 'Yes — MTS (100%)', 'Yes — rod-cone ~55%', 'Rare (<10%)'],
            ['JBTS10 mild', 'Hemizygous male', 'Hypomorphic missense (e.g. Ala446Thr)', 'Partial — retinal isoform most vulnerable', 'Yes — MTS (100%)', 'Yes — rod-cone ~70%', 'None'],
            ['RP23 (isolated)', 'Hemizygous male', 'Retinal isoform-specific (exons 9-11)', 'Retinal isoform only', 'No — brain normal; no MTS', 'Yes — severe rod-cone', 'None'],
            ['OFD1 syndrome', 'Heterozygous female', 'Full-gene LOF (carrier)', 'Haploinsufficiency (one allele)', 'Variable — mild CNS in some', 'Rare (<10%)', 'Yes — oral hamartomas, tongue, teeth, digital'],
            ['In utero lethal', 'Hemizygous male', 'Severely null (e.g. Arg554*)', 'All isoforms destroyed', 'N/A — miscarriage', 'N/A', 'N/A'],
          ]}
        />
        <div className="alert alert-info small mt-2">
          <strong>Clinical implication:</strong> OFD1 produces a phenotypic spectrum determined by sex and variant class.
          Full-gene null in hemizygous males → JBTS10 (MTS + high retinal). Retinal isoform-specific variants → RP23 only (no brain).
          Heterozygous females → OFD1 syndrome (oral-facial-digital, no MTS). Maternal examination for OFD1 features is
          MANDATORY — it confirms X-linked pedigree and guides counselling without requiring maternal genetic testing first.
        </div>
      </Section>

      <Section title="OFD1 Syndrome — Carrier Female Features (Mandatory Assessment)" color={ACCENT9}>
        <DataTable
          headers={['Feature', 'Frequency in Carrier Females', 'Assessment Tool', 'Clinical Significance']}
          rows={[
            ['Oral hamartomas / tongue nodules / tongue lobulation', '~80%', 'Clinical oral examination', 'Pathognomonic of OFD1 syndrome; confirms carrier status'],
            ['Supernumerary or absent teeth / dental anomalies', '~70%', 'Dental panoramic X-ray', 'Guides dental management; X-linked pedigree marker'],
            ['Gingival hamartomas', '~60%', 'Clinical oral examination', 'Benign; confirms carrier status'],
            ['Hypertelorism / broad nasal bridge', '~50%', 'Clinical dysmorphology', 'Facial feature; variable expression'],
            ['Post-axial polydactyly / syndactyly / brachydactyly', '~40%', 'Skeletal survey', 'Digital anomalies — variable; skeletal survey if suspected'],
            ['Alopecia (sparse/absent hair patches)', '~30%', 'Clinical examination', 'Rare but specific to OFD1 syndrome (Blaschko lines)'],
            ['Mild intellectual disability / learning difficulty', '~25%', 'Neuropsychological assessment', 'Carrier females may have mild cognitive involvement'],
            ['Renal cysts (polycystic kidney)', '~15%', 'Renal ultrasound', 'Rare; renal USS recommended in carrier females'],
          ]}
        />
        <div className="alert alert-warning small mt-2">
          <strong>Critical rule:</strong> In any X-linked JBTS family, the mother of an affected male MUST be examined
          for OFD1 syndrome features (oral examination + dental panoramic X-ray + skeletal assessment).
          Positive maternal OFD1 features confirm X-linked inheritance, distinguish from de novo and
          autosomal variants, and quantify recurrence risk (50% males JBTS10, 50% females OFD1 syndrome).
          PGT-M and prenatal diagnosis (sex determination + OFD1 sequencing) are available.
        </div>
      </Section>

      <Section title="Retinal Surveillance — JBTS10 vs JBTS3-9 Comparison" color={ACCENT5}>
        <DataTable
          headers={['JBTS Type', 'Gene', 'Retinal Frequency', 'RP23 Allelic?', 'Retinal Mechanism', 'ERG Priority']}
          rows={[
            ['JBTS10', 'OFD1', '~55% rod-cone (HIGHEST)', 'Yes — RP23 (#300424)', 'Centriolar satellite → connecting cilium failure', '⭐ HIGHEST PRIORITY — annual from diagnosis'],
            ['JBTS5', 'CEP290', '~57% rod-cone (close second)', '—', 'TZ Y-link + connecting cilium gate', 'Annual from diagnosis'],
            ['JBTS9', 'CC2D2A', '~40% rod-cone + ~15% coloboma', '—', 'TZ scaffold + MKS module', 'Annual (rod-cone + fundoscopy)'],
            ['JBTS3', 'AHI1', '~40% rod-cone', '—', 'Connecting cilium IFT', 'Annual from diagnosis'],
            ['JBTS8', 'ARL13B', '~30% rod-cone', '—', 'INPP5E trafficking / PI4P accumulation', 'Annual (lower priority)'],
            ['JBTS6', 'TMEM67', '~35% rod-cone + coloboma', '—', 'TZ membrane gate / Wnt', 'Annual (rod-cone + fundoscopy)'],
            ['JBTS7', 'RPGRIP1L', '~25% rod-cone', '—', 'TZ Y-link / C2-RPGR axis', 'Annual (moderate priority)'],
            ['JBTS4', 'NPHP1', '~10–15% rod-cone', '—', 'Nephrocystin-1 / connecting cilium', 'Every 2–3yr (low priority)'],
          ]}
        />
      </Section>

      <Section title="X-Linked Genetic Counselling — Recurrence Risk Table" color={ACCENT7}>
        <div className="alert small" style={{ borderLeft: `5px solid ${ACCENT7}`, background: '#fafafa' }}>
          <strong style={{ color: ACCENT7 }}>OFD1 / JBTS10 — X-linked recurrence risk stratification:</strong>
          <ul className="mb-0 mt-1">
            <li>Carrier mother (confirmed) × unaffected father: <strong>50% male offspring → JBTS10; 50% female offspring → OFD1 syndrome</strong></li>
            <li>De novo OFD1 mutation in proband: recurrence risk low (&lt;5%) but maternal examination still mandatory</li>
            <li>PGT-M: available; sex selection alone reduces risk 50% (eliminates JBTS10 in males; still 50% female OFD1 syndrome)</li>
            <li>Prenatal diagnosis: CVS + OFD1 sequencing + sex determination recommended</li>
            <li>Maternal renal USS: recommended for all carrier mothers (renal cysts ~15%)</li>
          </ul>
          <hr />
          <strong style={{ color: ACCENT9 }}>Key genetic counselling points:</strong>
          <ul className="mb-0 mt-1">
            <li>Unlike JBTS3-9 (AR), there is no 25% carrier sibling risk — X-linked pedigree applies</li>
            <li>Daughters of an affected male are all obligate OFD1 syndrome carriers</li>
            <li>Sons of an affected male are unaffected (X-linked; father passes Y to sons)</li>
            <li>No MKS tier risk — OFD1 is X-linked; autosomal biallelic null mechanism does not apply</li>
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
      <Section title="Glossary — OFD1 / JBTS10 Key Terms" color={ACCENT2}>
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

// ── Main page ────────────────────────────────────────────────────────────────
export default function JBTS10Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts10/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts10/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts10/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold">🧬 OFD1 — Joubert Syndrome Type 10 (JBTS10) / X-linked / RP23</h4>
            <div className="small opacity-75">
              X-linked Centriolar Satellite / Ciliogenesis / RP23 Allelic / OFD1 Syndrome · Xp22.2 · OMIM Gene *300170 · JBTS10 #300804 · OFD1 Syn. #311200 · RP23 #300424 · {N_COHORT}-patient cohort (seed-{SEED})
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
