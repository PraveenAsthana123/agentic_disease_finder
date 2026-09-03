'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';   // deep indigo — complete genome atlas
const LIGHT  = '#e8eaf6';
const COLOR2 = '#4527a0';   // purple — protein-coding
const COLOR3 = '#880e4f';   // dark pink — rRNA / L-strand pitfall
const COLOR4 = '#b71c1c';   // dark red — absolute CIs
const COLOR5 = '#1b5e20';   // dark green — OXPHOS pattern
const COLOR6 = '#e65100';   // orange — hallmark phenotypes
const COLOR7 = '#006064';   // teal — tRNA

function KPI({ label, value, color = COLOR }) {
  return (
    <div className="col-6 col-md-3 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function SectionCard({ title, children, borderColor = COLOR }) {
  return (
    <div className="card mb-4 shadow-sm" style={{ borderTop: `3px solid ${borderColor}` }}>
      <div className="card-body">
        {title && <h6 className="fw-bold mb-3" style={{ color: borderColor }}>{title}</h6>}
        {children}
      </div>
    </div>
  );
}

function Badge({ text, color = COLOR }) {
  return (
    <span className="badge me-1 mb-1" style={{ backgroundColor: color, fontSize: '0.72rem' }}>
      {text}
    </span>
  );
}

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const gc   = data.gene_classes || {};
  const pc   = gc.protein_coding || {};
  const trna = gc.tRNA || {};
  const rrna = gc.rRNA || {};
  const cov  = data.oxphos_complex_coverage || {};
  const str  = data.strand_distribution || {};
  const del4977 = data.common_deletion_4977bp || {};
  const uci  = data.universal_absolute_ci || [];
  const uman = data.universal_mandatory || [];
  const hall = data.hallmark_phenotypes || {};
  const coh  = data.cohort || {};

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="MT-Genome-Atlas — Complete 37-Gene Human Mitochondrial Genome">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Genome: </span>mtDNA (rCRS 16,569 bp) — {data.genome_accession}</div>
            <div><span className="fw-semibold">37 genes: </span>
              <span style={{ color: COLOR2 }}>13 protein-coding</span>{' · '}
              <span style={{ color: COLOR7 }}>22 tRNA</span>{' · '}
              <span style={{ color: COLOR3 }}>2 rRNA</span>
            </div>
            <div><span className="fw-semibold">Inheritance: </span>{data.inheritance}</div>
            <div className="mt-1"><span className="fw-semibold text-danger">WES: </span><span className="text-danger">{data.wes_limitation}</span></div>
          </div>
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">BTBGD exclusion: </span>{data.btbgd_slc19a3_exclusion}</div>
            <div className="mt-1"><span className="fw-semibold">Cohort: </span>{coh.total_patients?.toLocaleString()} patients ({coh.genes_included} genes × {coh.patients_per_gene})</div>
            <div><span className="fw-semibold">Seeds: </span>{coh.seed_range}</div>
          </div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total mtDNA Genes" value="37" color={COLOR} />
        <KPI label="Protein-Coding" value={pc.count} color={COLOR2} />
        <KPI label="tRNA Genes" value={trna.count} color={COLOR7} />
        <KPI label="rRNA Genes" value={rrna.count} color={COLOR3} />
        <KPI label="Aggregate Cohort" value={`${coh.total_patients?.toLocaleString()}`} color={COLOR5} />
        <KPI label="Genome (bp)" value="16,569" color={COLOR6} />
        <KPI label="L-strand NGS Pitfall" value={`${str.L_strand?.count} genes`} color={COLOR3} />
        <KPI label="Complex II (CII)" value="0 mtDNA" color={COLOR4} />
      </div>

      {/* Gene class breakdown */}
      <div className="row g-3 mb-4">
        <div className="col-12 col-md-4">
          <SectionCard title="Protein-Coding (13 genes)" borderColor={COLOR2}>
            <div className="small">
              <div className="mb-2">{pc.note}</div>
              <div className="fw-semibold mb-1">CI (7 subunits):</div>
              <div className="mb-2">MT-ND1 · MT-ND2 · MT-ND3 · MT-ND4 · MT-ND4L · MT-ND5 · MT-ND6</div>
              <div className="fw-semibold mb-1">CIII (1 subunit):</div>
              <div className="mb-2">MT-CYB</div>
              <div className="fw-semibold mb-1">CIV (3 subunits):</div>
              <div className="mb-2">MT-CO1 · MT-CO2 · MT-CO3</div>
              <div className="fw-semibold mb-1">CV (2 subunits):</div>
              <div className="mb-2">MT-ATP8 · MT-ATP6</div>
              <div className="p-2 rounded mt-2" style={{ backgroundColor: '#fff3e0', borderLeft: '3px solid #e65100' }}>
                <span className="fw-semibold">CII = 0 mtDNA subunits</span> — SDHA/SDHB/SDHC/SDHD all nuclear. CII ALWAYS normal in primary mtDNA disease — use as biochemical reference.
              </div>
            </div>
          </SectionCard>
        </div>
        <div className="col-12 col-md-4">
          <SectionCard title="tRNA Genes (22 genes)" borderColor={COLOR7}>
            <div className="small">
              <div className="mb-2">{trna.note}</div>
              <div className="fw-semibold mb-1">H-strand ({trna.h_strand?.length} genes, standard NGS):</div>
              <div className="mb-2">{(trna.h_strand || []).join(' · ')}</div>
              <div className="fw-semibold mb-1 text-danger">L-strand NGS pitfall ({trna.l_strand_ngs_pitfall?.length} genes):</div>
              <div className="mb-1">{(trna.l_strand_ngs_pitfall || []).join(' · ')}</div>
              <div className="text-muted">Reverse-complement QC window mandatory for all L-strand genes</div>
              <div className="mt-2 text-primary small">
                <Link href="/mt-trna-atlas">→ MT-tRNA-Atlas: complete 22-gene analysis</Link>
              </div>
            </div>
          </SectionCard>
        </div>
        <div className="col-12 col-md-4">
          <SectionCard title="rRNA Genes (2 genes)" borderColor={COLOR3}>
            <div className="small">
              <div className="mb-3">
                <div className="fw-semibold">MT-RNR1 (12S, mt-SSU 28S)</div>
                <div>rCRS 648–1,601 (954 nt) · H-strand</div>
                <div className="text-danger fw-semibold">AISNHL — m.1555A&gt;G 1:500 population</div>
                <div><strong>UNIQUE:</strong> NO OXPHOS deficiency — aminoglycoside cochlear sensitivity only</div>
                <div className="text-danger">Aminoglycosides ABSOLUTE CI (ANY single dose = permanent deafness)</div>
                <Link href="/mtrnr1" className="text-primary small">→ MT-RNR1 dashboard</Link>
              </div>
              <hr />
              <div>
                <div className="fw-semibold">MT-RNR2 (16S, mt-LSU 39S)</div>
                <div>rCRS 1,671–3,229 (1,559 nt) · H-strand</div>
                <div>Combined CI+CIII+CIV+CV; PTC scaffold</div>
                <div>Humanin 21aa neuroprotective microprotein (rCRS 2706–2768)</div>
                <Link href="/mtrnr2" className="text-primary small">→ MT-RNR2 dashboard</Link>
              </div>
            </div>
          </SectionCard>
        </div>
      </div>

      {/* OXPHOS complex coverage */}
      <SectionCard title="OXPHOS Complex Coverage — mtDNA vs Nuclear Subunits" borderColor={COLOR5}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-dark">
              <tr>
                <th>Complex</th><th>Name</th><th>Total Subunits</th>
                <th>mtDNA Subunits</th><th>Nuclear Subunits</th><th>mtDNA Genes</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(cov).map(([cx, c]) => (
                <tr key={cx} className={cx === 'CII' ? 'table-warning' : ''}>
                  <td className="fw-bold">{cx}</td>
                  <td>{c.name}</td>
                  <td className="text-center">{c.subunits_total}</td>
                  <td className="text-center fw-bold" style={{ color: c.mtDNA_subunits === 0 ? '#b71c1c' : COLOR5 }}>
                    {c.mtDNA_subunits}
                    {c.mtDNA_subunits === 0 && ' ⚠'}
                  </td>
                  <td className="text-center">{c.nuclear_subunits}</td>
                  <td className="small">{(c.genes || []).join(' · ') || (c.note || '—')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="alert alert-warning py-2 mt-2 small mb-0">
          <strong>CII diagnostic key:</strong> Complex II (succinate dehydrogenase) has ZERO mtDNA-encoded subunits — CII activity is ALWAYS normal in primary mtDNA disease. Reduced CII with normal CI/CIV should prompt nuclear gene investigation (SDHA/SDHB/SDHC/SDHD), not mtDNA sequencing.
        </div>
      </SectionCard>

      {/* Common 4977 bp deletion */}
      <SectionCard title="Common 4977 bp Deletion (KSS / PEO Deletion)" borderColor={COLOR6}>
        <div className="small">
          <div className="row g-2">
            <div className="col-12 col-md-6">
              <div><span className="fw-semibold">Coordinates: </span>rCRS {del4977.rCRS_range} ({del4977.size_bp?.toLocaleString()} bp)</div>
              <div><span className="fw-semibold">Phenotypes: </span>{del4977.phenotypes}</div>
              <div className="mt-2 fw-semibold">Protein-coding genes removed ({del4977.genes_removed_protein?.length}):</div>
              <div>{(del4977.genes_removed_protein || []).join(' · ')}</div>
            </div>
            <div className="col-12 col-md-6">
              <div className="fw-semibold">tRNA genes removed ({del4977.genes_removed_tRNA?.length}):</div>
              <div>{(del4977.genes_removed_tRNA || []).join(' · ')}</div>
              <div className="mt-2"><span className="fw-semibold">Total genes affected: </span>{del4977.total_genes_affected}</div>
            </div>
          </div>
        </div>
      </SectionCard>

      {/* Universal drug CIs */}
      <SectionCard title="Universal Absolute Contraindications (ALL 37 mtDNA Genes)" borderColor={COLOR4}>
        <div className="row g-2 small">
          {uci.map((ci, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ backgroundColor: '#ffebee' }}>
                <div className="fw-bold text-danger">{ci.drug}</div>
                <div>{ci.mechanism}</div>
                <div className="text-muted">{ci.applies_to}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Hallmark phenotypes */}
      <SectionCard title="Hallmark Phenotypes by Gene" borderColor={COLOR6}>
        <div className="row g-2 small">
          {Object.entries(hall).map(([name, ph]) => (
            <div key={name} className="col-12 col-md-6 col-lg-4">
              <div className="card h-100 p-2" style={{ borderLeft: `3px solid ${COLOR6}` }}>
                <div className="fw-bold" style={{ color: COLOR6 }}>{name.replace(/_/g, ' ')}</div>
                <div><span className="fw-semibold">Gene: </span>{ph.gene}</div>
                <div><span className="fw-semibold">Variant: </span>{ph.variant}</div>
                <div className="text-muted">{ph.note}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Universal mandatory */}
      <SectionCard title="Universal Mandatory Protocol (ALL 37 mtDNA Genes)" borderColor={COLOR5}>
        <ul className="mb-0 small">
          {uman.map((m, i) => <li key={i}>{m}</li>)}
        </ul>
      </SectionCard>
    </>
  );
}

// ── Tab: Gene Table ───────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  const [filter, setFilter] = useState('all');
  if (!data) return <p className="text-muted">Loading…</p>;
  const rows = data.gene_table || [];
  const filtered = filter === 'all' ? rows : rows.filter(r => r.gene_class === filter);
  const classColors = { protein: COLOR2, tRNA: COLOR7, rRNA: COLOR3 };

  return (
    <>
      <div className="mb-3 d-flex gap-2 flex-wrap">
        {['all','protein','tRNA','rRNA'].map(f => (
          <button key={f} className="btn btn-sm" onClick={() => setFilter(f)}
            style={{ backgroundColor: filter===f ? COLOR : '#eee', color: filter===f ? '#fff' : '#333' }}>
            {f === 'all' ? 'All 37 Genes' : f === 'protein' ? 'Protein-Coding (13)' : f === 'tRNA' ? 'tRNA (22)' : 'rRNA (2)'}
          </button>
        ))}
      </div>
      <div className="table-responsive">
        <table className="table table-sm table-hover table-bordered small">
          <thead className="table-dark">
            <tr>
              <th>Gene</th><th>Class</th><th>Size</th><th>Strand</th>
              <th>rCRS Start</th><th>rCRS End</th><th>Complex</th>
              <th>Primary Disease</th><th>OXPHOS</th><th>Hallmark</th><th>Key CI</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => (
              <tr key={i} className={r.ngs_pitfall ? 'table-warning' : ''}>
                <td className="fw-bold" style={{ color: classColors[r.gene_class] || COLOR }}>{r.gene}</td>
                <td>
                  <span className="badge" style={{ backgroundColor: classColors[r.gene_class] || COLOR, fontSize: '0.68rem' }}>
                    {r.gene_class}
                  </span>
                </td>
                <td className="text-nowrap">{r.size}</td>
                <td>
                  {r.strand === 'L'
                    ? <span className="fw-bold text-danger">L ⚠</span>
                    : <span>H</span>}
                </td>
                <td>{r.rcrs_start?.toLocaleString()}</td>
                <td>{r.rcrs_end?.toLocaleString()}</td>
                <td style={{ maxWidth: 180 }} className="small">{r.complex}</td>
                <td style={{ maxWidth: 200 }} className="small">{r.primary_disease}</td>
                <td style={{ maxWidth: 180 }} className="small">{r.oxphos}</td>
                <td style={{ maxWidth: 200 }} className="small">{r.hallmark}</td>
                <td style={{ maxWidth: 180 }} className="small text-danger">{r.key_ci}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="small text-muted mt-1">
        <span className="badge me-1" style={{ backgroundColor: '#fff3cd', color: '#856404' }}>⚠ highlighted rows</span> = L-strand genes (NGS pitfall — reverse-complement QC mandatory)
      </div>
    </>
  );
}

// ── Tab: Clinical Atlas ───────────────────────────────────────────────────────
function ClinicalAtlasTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const cs  = data.class_summary || {};
  const gst = data.genome_stats || {};
  const kss = data.kss_deletion_genes || [];

  return (
    <>
      <SectionCard title="Gene Class Summary" borderColor={COLOR2}>
        <div className="row g-3 small">
          <div className="col-12 col-md-4">
            <div className="fw-bold mb-2" style={{ color: COLOR2 }}>Protein-Coding ({cs.protein_coding?.count} genes)</div>
            <div><span className="fw-semibold">Complex I: </span>{cs.protein_coding?.complexes?.CI}</div>
            <div><span className="fw-semibold">Complex II: </span><span className="text-danger">{cs.protein_coding?.complexes?.CII} (ALL nuclear)</span></div>
            <div><span className="fw-semibold">Complex III: </span>{cs.protein_coding?.complexes?.CIII}</div>
            <div><span className="fw-semibold">Complex IV: </span>{cs.protein_coding?.complexes?.CIV}</div>
            <div><span className="fw-semibold">Complex V: </span>{cs.protein_coding?.complexes?.CV}</div>
            <div className="text-muted mt-1">{cs.protein_coding?.oxphos_note}</div>
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-bold mb-2" style={{ color: COLOR7 }}>tRNA ({cs.tRNA?.count} genes)</div>
            <div><span className="fw-semibold">H-strand (standard NGS): </span>{cs.tRNA?.h_strand}</div>
            <div><span className="fw-semibold text-danger">L-strand NGS pitfall: </span>{cs.tRNA?.l_strand_ngs_pitfall}</div>
            <div className="mt-1 text-primary small">
              <Link href="/mt-trna-atlas">→ Full 22-gene tRNA Atlas</Link>
            </div>
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-bold mb-2" style={{ color: COLOR3 }}>rRNA ({cs.rRNA?.count} genes)</div>
            <div className="mb-1">{cs.rRNA?.mt_rnr1_note}</div>
            <div>{cs.rRNA?.mt_rnr2_note}</div>
            <div className="mt-1 d-flex gap-2">
              <Link href="/mtrnr1" className="text-primary small">→ MT-RNR1</Link>
              <Link href="/mtrnr2" className="text-primary small">→ MT-RNR2</Link>
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Genomic Statistics" borderColor={COLOR}>
        <div className="small">
          <div><span className="fw-semibold">Total genome: </span>{gst.genome_bp_total?.toLocaleString()} bp (rCRS)</div>
          <div><span className="fw-semibold">First gene: </span>{gst.first_gene} (rCRS {gst.first_gene_rcrs_start})</div>
          <div><span className="fw-semibold">Last gene: </span>{gst.last_gene}</div>
          <div><span className="fw-semibold">Total genes: </span>{gst.total_genes}</div>
          <div className="mt-2 fw-semibold">Non-coding regions:</div>
          {(gst.non_coding_regions || []).map((r, i) => (
            <div key={i}>• <strong>{r.region}</strong> ({r.rCRS}): {r.function}</div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Common 4977 bp KSS Deletion — Genes Removed" borderColor={COLOR6}>
        <div className="small">
          <div className="mb-2">rCRS 8,470–13,447 — removes {kss.length} genes simultaneously:</div>
          <div className="d-flex flex-wrap gap-1">
            {kss.map((g, i) => (
              <Badge key={i} text={g} color={COLOR6} />
            ))}
          </div>
          <div className="mt-2 text-muted">KSS phenotype: CPEO + pigmentary retinopathy + cardiomyopathy. Cardiac pacemaker often required. Onset &lt;20 years. Sporadic (de novo deletion; NOT inherited maternally in most cases).</div>
        </div>
      </SectionCard>

      <SectionCard title="Cross-Gene Drug Safety Summary" borderColor={COLOR4}>
        <div className="table-responsive">
          <table className="table table-sm table-bordered small mb-0">
            <thead className="table-dark">
              <tr><th>Drug/Class</th><th>Protein-Coding CI</th><th>tRNA CI</th><th>rRNA CI</th><th>Mechanism</th></tr>
            </thead>
            <tbody>
              <tr><td className="fw-bold text-danger">Metformin</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Complex I inhibitor — fatal lactic acidosis</td></tr>
              <tr><td className="fw-bold text-danger">VPA</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>β-ox inhibition + CI + POLG inhibition</td></tr>
              <tr><td className="fw-bold text-danger">Propofol</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>PRIS — uncouples OXPHOS; fatal myocardial failure</td></tr>
              <tr><td className="fw-bold text-danger">Linezolid</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Blocks mt-23S rRNA → halts all 13 OXPHOS subunits</td></tr>
              <tr><td className="fw-bold text-danger">Chloramphenicol</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td className="text-center text-danger">ABS CI</td><td>Inhibits mt-ribosome peptidyl transferase</td></tr>
              <tr><td className="fw-bold text-danger">Aminoglycosides</td><td className="text-center text-muted">—</td><td className="text-center text-muted">—</td><td className="text-center text-danger">ABS CI (MT-RNR1)</td><td>12S rRNA Helix44 — permanent cochlear deafness</td></tr>
              <tr><td className="fw-bold text-warning">Ethambutol</td><td className="text-center text-warning">CI (LHON genes)</td><td className="text-center text-muted">—</td><td className="text-center text-warning">CI (MT-RNR2 LHON-like)</td><td>Compounding optic neuropathy</td></tr>
              <tr><td className="fw-bold text-warning">Amiodarone</td><td className="text-center text-muted">—</td><td className="text-center text-danger">ABS CI (MT-TT)</td><td className="text-center text-muted">—</td><td>mt-OXPHOS inhibitor; highest risk in HCM (MT-TT 55–65%)</td></tr>
              <tr><td className="fw-bold text-warning">KD</td><td className="text-center text-warning">CI (CI/CIV/CV)</td><td className="text-center text-warning">CI (all)</td><td className="text-center text-muted">—</td><td>Impairs β-oxidation + anaplerosis in OXPHOS disease</td></tr>
              <tr className="table-success"><td className="fw-bold">LEV (preferred)</td><td className="text-center">Preferred AED</td><td className="text-center">Preferred AED</td><td className="text-center">Preferred AED</td><td>Renal excretion; no CYP450; no mt toxicity</td></tr>
              <tr className="table-success"><td className="fw-bold">Idebenone</td><td className="text-center">LHON treatment</td><td className="text-center text-muted">—</td><td className="text-center text-muted">—</td><td>Raxone — approved LHON; bypasses CI via cyt c</td></tr>
            </tbody>
          </table>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  return (
    <>
      <SectionCard title="Atlas Scope" borderColor={COLOR}>
        <p className="small mb-0">{data.atlas_scope}</p>
      </SectionCard>

      <SectionCard title="Key Concepts" borderColor={COLOR2}>
        <div className="row g-2">
          {(data.key_concepts || []).map((kc, i) => (
            <div key={i} className="col-12 col-md-6">
              <div className="p-2 rounded" style={{ backgroundColor: LIGHT }}>
                <div className="fw-semibold small" style={{ color: COLOR }}>{kc.term}</div>
                <div className="small text-muted">{kc.definition}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      <SectionCard title="Drug Definitions" borderColor={COLOR4}>
        {(data.drug_definitions || []).map((d, i) => (
          <div key={i} className="mb-2 small">
            <span className="fw-bold text-danger">{d.term}: </span>
            <span>{d.definition}</span>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR5}>
        {(data.references || []).map((r, i) => (
          <div key={i} className="mb-1 small">
            <span className="fw-semibold" style={{ color: COLOR5 }}>{r.ref}</span>{' — '}
            <span className="text-muted">{r.citation}</span>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MtGenomeAtlasPage() {
  const [tab,  setTab]  = useState(0);
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defn, setDefn] = useState(null);
  const [err,  setErr]  = useState('');

  useEffect(() => {
    const load = async () => {
      try {
        const [r1, r2, r3] = await Promise.all([
          fetch(`${API}/api/mt-genome-atlas/overview`),
          fetch(`${API}/api/mt-genome-atlas/breakdown`),
          fetch(`${API}/api/mt-genome-atlas/definitions`),
        ]);
        setOv(await r1.json());
        setBd(await r2.json());
        setDefn(await r3.json());
      } catch (e) { setErr(e.message); }
    };
    load();
  }, []);

  const tabData = [ov, bd, bd, defn];

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center gap-3 mb-3 flex-wrap">
        <div>
          <h4 className="fw-bold mb-0" style={{ color: COLOR }}>
            🧬 MT-Genome Atlas — Complete 37-Gene Mitochondrial Genome
          </h4>
          <div className="text-muted small">
            13 Protein-Coding · 22 tRNA · 2 rRNA · 1,480-Patient Aggregate (37×40) · rCRS 16,569 bp
          </div>
        </div>
        <div className="ms-auto d-flex gap-2">
          <Link href="/mt-trna-atlas" className="btn btn-sm btn-outline-secondary">MT-tRNA Atlas</Link>
          <Link href="/mtrnr1" className="btn btn-sm btn-outline-secondary">MT-RNR1</Link>
          <Link href="/mtrnr2" className="btn btn-sm btn-outline-secondary">MT-RNR2</Link>
          <Link href="/expert-dashboards-catalog" className="btn btn-sm btn-outline-primary">All Dashboards</Link>
        </div>
      </div>

      {err && <div className="alert alert-danger small">{err}</div>}

      {/* Tab bar */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}>{t}
            </button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab data={ov} />}
      {tab === 1 && <GeneTableTab data={bd} />}
      {tab === 2 && <ClinicalAtlasTab data={bd} />}
      {tab === 3 && <DefinitionsTab data={defn} />}
    </div>
  );
}
