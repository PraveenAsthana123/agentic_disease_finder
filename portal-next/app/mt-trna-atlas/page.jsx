'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const COLOR  = '#1a237e';   // deep indigo — genome atlas / complete set
const LIGHT  = '#e8eaf6';
const COLOR2 = '#4527a0';   // purple — H-strand
const COLOR3 = '#880e4f';   // dark pink — L-strand / NGS pitfall
const COLOR4 = '#b71c1c';   // dark red — absolute CIs
const COLOR5 = '#1b5e20';   // dark green — OXPHOS pattern / phenotype
const COLOR6 = '#e65100';   // orange — hallmark-unique genes

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
  const hst  = data.strand_distribution?.H_strand || {};
  const lst  = data.strand_distribution?.L_strand || {};
  const ngsg = data.ngs_pitfall_genes || {};
  const span = data.genome_span || {};
  const pats = data.oxphos_patterns || {};
  const hall = data.hallmark_unique_phenotypes || {};
  const kss  = data.kss_4977bp_deletion || {};
  const uci  = data.universal_absolute_ci || [];
  const uman = data.universal_mandatory || [];

  return (
    <>
      {/* Atlas banner */}
      <SectionCard title="MT-tRNA Atlas — 22 Genes | 880-Patient Aggregate Cohort (22×40) | Human Mitochondrial Genome">
        <div className="row g-3 small">
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">Genome: </span>mtDNA (rCRS 16,569 bp) — 22 tRNA genes required for translation of 13 mt-encoded OXPHOS subunits</div>
            <div><span className="fw-semibold">Span: </span>rCRS {span.first_tRNA?.rcrs_start}–{span.last_tRNA?.rcrs_end} (~{span.total_rCRS_span_bp?.toLocaleString()} bp)</div>
            <div><span className="fw-semibold">First tRNA: </span>{span.first_tRNA?.gene} — {span.first_tRNA?.note}</div>
            <div><span className="fw-semibold">Last tRNA: </span>{span.last_tRNA?.gene} — {span.last_tRNA?.note}</div>
          </div>
          <div className="col-12 col-md-6">
            <div><span className="fw-semibold">WES limitation: </span><span className="text-danger fw-semibold">{data.wes_limitation}</span></div>
            <div><span className="fw-semibold">BTBGD exclusion: </span>{data.btbgd_slc19a3_exclusion}</div>
          </div>
        </div>
      </SectionCard>

      {/* KPIs */}
      <div className="row g-2 mb-4">
        <KPI label="Total mt-tRNA Genes" value="22" color={COLOR} />
        <KPI label="H-strand Genes" value={hst.count} color={COLOR2} />
        <KPI label="L-strand (NGS Pitfall)" value={lst.count} color={COLOR3} />
        <KPI label="Hallmark-Unique" value={Object.keys(hall).length} color={COLOR6} />
        <KPI label="Aggregate Patients" value={data.total_cohort_patients} color={COLOR5} />
        <KPI label="KSS 4977bp Affected" value={kss.removes_genes?.length} color={COLOR4} />
      </div>

      {/* Strand distribution */}
      <div className="row mb-4">
        <div className="col-12 col-md-6 mb-3">
          <SectionCard title={`H-strand Genes (${hst.count}/22) — Standard NGS Coverage`} borderColor={COLOR2}>
            <p className="small text-muted mb-2">{hst.note}</p>
            <div className="d-flex flex-wrap">
              {(hst.genes || []).map(g => (
                <Link key={g} href={`/${g.toLowerCase().replace(/-/g,'')}`}>
                  <Badge text={g} color={COLOR2} />
                </Link>
              ))}
            </div>
          </SectionCard>
        </div>
        <div className="col-12 col-md-6 mb-3">
          <SectionCard title={`L-strand Genes (${lst.count}/22) — NGS Pitfall ⚠`} borderColor={COLOR3}>
            <p className="small text-muted mb-2">{lst.note}</p>
            <div className="d-flex flex-wrap">
              {(lst.genes || []).map(g => (
                <Link key={g} href={`/${g.toLowerCase().replace(/-/g,'')}`}>
                  <Badge text={g} color={COLOR3} />
                </Link>
              ))}
            </div>
          </SectionCard>
        </div>
      </div>

      {/* OXPHOS patterns */}
      <SectionCard title="OXPHOS Deficiency Patterns Across All 22 mt-tRNA Genes" borderColor={COLOR5}>
        <div className="row g-3 small">
          {Object.entries(pats).map(([key, val]) => (
            <div key={key} className="col-12 col-md-4">
              <div className="p-2 rounded" style={{ backgroundColor: '#f1f8e9', border: `1px solid ${COLOR5}` }}>
                <div className="fw-semibold text-capitalize mb-1" style={{ color: COLOR5 }}>
                  {key.replace(/_/g, ' ')}
                </div>
                <div className="text-muted">{val.description}</div>
                <div className="mt-1">
                  {(val.genes || []).map(g => <Badge key={g} text={g} color={COLOR5} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Hallmark-unique genes */}
      <SectionCard title="Hallmark-Unique Phenotypes — Genes with DISTINCTIVE Clinical Signatures" borderColor={COLOR6}>
        <div className="row g-2 small">
          {Object.entries(hall).map(([gene, desc]) => (
            <div key={gene} className="col-12 col-md-6">
              <div className="p-2 rounded mb-2" style={{ backgroundColor: '#fff3e0', border: `1px solid ${COLOR6}` }}>
                <Link href={`/${gene.toLowerCase().replace(/-/g,'')}`}>
                  <span className="fw-bold" style={{ color: COLOR6 }}>{gene}</span>
                </Link>
                <span className="text-muted ms-2">{desc}</span>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>

      {/* Universal CIs */}
      <SectionCard title="Universal Drug Contraindications — ALL 22 mt-tRNA Genes" borderColor={COLOR4}>
        <div className="row g-3 small">
          <div className="col-12 col-md-4">
            <div className="fw-semibold text-danger mb-1">ABSOLUTE CI (all 22)</div>
            {uci.map(d => <div key={d}><Badge text={d} color={COLOR4} /></div>)}
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-semibold text-warning mb-1" style={{color:'#e65100'}}>Contraindicated</div>
            {(data.universal_contraindicated || []).map(d => <div key={d}><Badge text={d} color={COLOR6} /></div>)}
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-semibold mb-1" style={{color:COLOR5}}>Mandatory Protocol</div>
            {uman.map(d => <div key={d}><Badge text={d} color={COLOR5} /></div>)}
            <div className="mt-1"><Badge text={`Preferred AED: ${data.preferred_aed}`} color={COLOR} /></div>
          </div>
        </div>
      </SectionCard>

      {/* KSS deletion */}
      <SectionCard title="Common 4977 bp Deletion (KSS) — Compound mt-tRNA Loss" borderColor={COLOR4}>
        <div className="small">
          <div className="fw-semibold text-danger mb-1">Removes tRNA genes: {(kss.removes_genes || []).join(' · ')}</div>
          <div className="text-muted">{kss.remark}</div>
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Gene Table ────────────────────────────────────────────────────────────
function GeneTableTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  const rows  = data.gene_table || [];
  const stats = data.summary_stats || {};

  return (
    <>
      {/* Stats bar */}
      <div className="row g-2 mb-4">
        <KPI label="Total Genes" value={stats.total_genes} color={COLOR} />
        <KPI label="H-strand" value={stats.h_strand_count} color={COLOR2} />
        <KPI label="L-strand" value={stats.l_strand_count} color={COLOR3} />
        <KPI label="NGS Pitfall Genes" value={stats.ngs_pitfall_count} color={COLOR3} />
        <KPI label="Total nt (all tRNAs)" value={stats.total_nucleotides_all_trna} color={COLOR5} />
        <KPI label="Shortest (MT-TS2)" value={`${stats.shortest_length_nt} nt`} color={COLOR} />
      </div>

      {/* Main gene table */}
      <div className="card shadow-sm mb-4">
        <div className="card-body p-0">
          <div className="table-responsive">
            <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
              <thead style={{ backgroundColor: COLOR, color: '#fff', position: 'sticky', top: 0 }}>
                <tr>
                  <th>#</th>
                  <th>Gene</th>
                  <th>aa</th>
                  <th>Anticodon</th>
                  <th>Strand</th>
                  <th>rCRS Start</th>
                  <th>rCRS End</th>
                  <th>nt</th>
                  <th>OMIM</th>
                  <th>NGS Pitfall</th>
                  <th>KSS Del</th>
                  <th>Hallmark</th>
                  <th>Most Common Variant</th>
                  <th>Nuclear DDx</th>
                  <th>Dashboard</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={r.gene}
                      style={{ backgroundColor: r.hallmark_unique ? '#fff8e1' : r.ngs_pitfall ? '#fce4ec' : '' }}>
                    <td className="text-muted">{r.ordinal}</td>
                    <td>
                      <Link href={r.dashboard_route} className="fw-bold" style={{ color: r.ngs_pitfall ? COLOR3 : COLOR }}>
                        {r.gene}
                      </Link>
                    </td>
                    <td>{r.amino_acid}</td>
                    <td><code>{r.anticodon}</code></td>
                    <td>
                      <span className="badge" style={{ backgroundColor: r.strand === 'H' ? COLOR2 : COLOR3 }}>
                        {r.strand}
                      </span>
                    </td>
                    <td>{r.rcrs_start.toLocaleString()}</td>
                    <td>{r.rcrs_end.toLocaleString()}</td>
                    <td>{r.length_nt}</td>
                    <td><a href={`https://omim.org/entry/${r.omim_gene}`} target="_blank" rel="noreferrer" className="text-muted small">*{r.omim_gene}</a></td>
                    <td>{r.ngs_pitfall ? <span className="text-danger fw-bold">⚠ YES</span> : <span className="text-success">—</span>}</td>
                    <td>{r.kss_4977bp_affected ? <span className="text-danger">YES</span> : '—'}</td>
                    <td>{r.hallmark_unique ? <span title={r.hallmark_detail}>⭐</span> : '—'}</td>
                    <td style={{ maxWidth: 200 }} className="text-muted">{r.most_common_variant}</td>
                    <td style={{ maxWidth: 200 }} className="text-muted small">{r.nuclear_ddx}</td>
                    <td>
                      <Link href={r.dashboard_route} className="btn btn-outline-secondary btn-sm py-0" style={{ fontSize: '0.7rem' }}>
                        →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Phenotype breakdown */}
      <div className="row g-3">
        {Object.entries(data.phenotype_breakdown || {}).map(([key, val]) => (
          <div key={key} className="col-12 col-md-6 col-lg-4">
            <div className="card shadow-sm h-100">
              <div className="card-body p-3">
                <div className="fw-semibold small text-uppercase mb-1" style={{ color: COLOR }}>
                  {key.replace(/_/g, ' ')}
                </div>
                <div className="small text-muted mb-2">{val.description}</div>
                <div className="d-flex flex-wrap">
                  {(val.genes || []).map(g => <Badge key={g} text={g} color={COLOR5} />)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

// ── Tab: Clinical Atlas ────────────────────────────────────────────────────────
function ClinicalAtlasTab({ overviewData, breakdownData }) {
  if (!overviewData || !breakdownData) return <p className="text-muted">Loading…</p>;
  const rows = breakdownData.gene_table || [];
  const ddxList = breakdownData.nuclear_ddx_synthetases || [];
  const delPat  = breakdownData.large_deletion_patterns || {};

  return (
    <>
      {/* Nuclear DDx synthetase table */}
      <SectionCard title="Nuclear DDx — mt-aminoacyl-tRNA Synthetases (one per mt-tRNA gene)" borderColor={COLOR}>
        <p className="small text-muted mb-3">
          Each mt-tRNA gene has a corresponding nuclear-encoded mt-aminoacyl-tRNA synthetase.
          Biallelic AR mutations in these synthetases cause different diseases (often neonatal/infantile)
          detectable by WES, while heteroplasmic mt-tRNA mutations cause adult-onset CPEO/Myopathy
          and are MISSED by WES.
        </p>
        <div className="table-responsive">
          <table className="table table-sm table-hover mb-0" style={{ fontSize: '0.78rem' }}>
            <thead style={{ backgroundColor: COLOR, color: '#fff' }}>
              <tr>
                <th>mt-tRNA Gene</th>
                <th>Nuclear DDx Synthetase / Disease</th>
              </tr>
            </thead>
            <tbody>
              {ddxList.map(r => (
                <tr key={r.gene}>
                  <td>
                    <Link href={`/${r.gene.toLowerCase().replace(/-/g,'')}`} className="fw-bold" style={{ color: COLOR }}>
                      {r.gene}
                    </Link>
                  </td>
                  <td className="text-muted small">{r.ddx_nuclear}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      {/* Large deletion compound losses */}
      <SectionCard title="Large Deletion Compound tRNA Losses" borderColor={COLOR4}>
        {Object.entries(delPat).map(([key, val]) => (
          <div key={key} className="mb-3 p-3 rounded" style={{ backgroundColor: '#ffebee', border: `1px solid ${COLOR4}` }}>
            <div className="fw-semibold text-danger mb-1">{val.deletion || val.cluster}</div>
            <div className="small text-muted mb-1">{val.note}</div>
            {val.affected_trna_genes && (
              <div>{val.affected_trna_genes.map(g => <Badge key={g} text={g} color={COLOR4} />)}</div>
            )}
          </div>
        ))}
      </SectionCard>

      {/* Universal clinical protocol */}
      <SectionCard title="Universal Clinical Protocol — Applied to ALL 22 mt-tRNA Genes" borderColor={COLOR5}>
        <div className="row g-3 small">
          <div className="col-12 col-md-4">
            <div className="fw-semibold text-danger mb-2">ABSOLUTE CONTRAINDICATIONS</div>
            {(overviewData.universal_absolute_ci || []).map(d => (
              <div key={d} className="p-1 mb-1 rounded" style={{ backgroundColor: '#ffcdd2' }}>
                <span className="fw-semibold">{d}</span>
              </div>
            ))}
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-semibold mb-2" style={{ color: COLOR6 }}>CONTRAINDICATED</div>
            {(overviewData.universal_contraindicated || []).map(d => (
              <div key={d} className="p-1 mb-1 rounded" style={{ backgroundColor: '#fff3e0' }}>
                <span className="fw-semibold">{d}</span>
              </div>
            ))}
            <div className="fw-semibold mt-2 mb-2" style={{ color: COLOR5 }}>PREFERRED AED</div>
            <div className="p-1 mb-1 rounded" style={{ backgroundColor: '#e8f5e9' }}>
              <span className="fw-semibold">{overviewData.preferred_aed}</span>
            </div>
          </div>
          <div className="col-12 col-md-4">
            <div className="fw-semibold mb-2" style={{ color: COLOR }}>MANDATORY PROTOCOL</div>
            {(overviewData.universal_mandatory || []).map(d => (
              <div key={d} className="p-1 mb-1 rounded" style={{ backgroundColor: LIGHT }}>
                <span className="fw-semibold">{d}</span>
              </div>
            ))}
          </div>
        </div>
      </SectionCard>

      {/* Hallmark-unique per-gene clinical alerts */}
      <SectionCard title="HALLMARK-UNIQUE Clinical Signatures — 7 Genes with Distinctive Phenotypes" borderColor={COLOR6}>
        <div className="row g-3">
          {rows.filter(r => r.hallmark_unique).map(r => (
            <div key={r.gene} className="col-12 col-md-6">
              <div className="p-3 rounded h-100" style={{ backgroundColor: '#fff8e1', border: `2px solid ${COLOR6}` }}>
                <div className="d-flex align-items-center mb-2">
                  <Link href={r.dashboard_route}>
                    <span className="fw-bold fs-6 me-2" style={{ color: COLOR6 }}>{r.gene}</span>
                  </Link>
                  <span className="text-muted small">tRNA-{r.amino_acid} | {r.strand}-strand</span>
                </div>
                <div className="small fw-semibold mb-1" style={{ color: COLOR6 }}>{r.hallmark_detail}</div>
                <div className="small text-muted">{r.primary_phenotype}</div>
                <div className="small mt-1"><span className="fw-semibold">Most common: </span>{r.most_common_variant}</div>
              </div>
            </div>
          ))}
        </div>
      </SectionCard>
    </>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <p className="text-muted">Loading…</p>;
  return (
    <>
      <SectionCard title="Genomic & Biochemical Concepts" borderColor={COLOR}>
        {(data.gene_definitions || []).map(d => (
          <div key={d.term} className="mb-3">
            <div className="fw-semibold" style={{ color: COLOR }}>{d.term}</div>
            <div className="text-muted small">{d.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Drug Reference — Universal Across All 22 mt-tRNA Genes" borderColor={COLOR4}>
        {(data.drug_definitions || []).map(d => (
          <div key={d.term} className="mb-3">
            <div className="fw-semibold" style={{ color: d.definition.startsWith('ABSOLUTE') ? COLOR4 : COLOR }}>
              {d.term}
            </div>
            <div className="text-muted small">{d.definition}</div>
          </div>
        ))}
      </SectionCard>

      <SectionCard title="Key References" borderColor={COLOR}>
        {(data.references || []).map(r => (
          <div key={r.ref} className="mb-2 small">
            <span className="fw-semibold">{r.ref}: </span>
            <span className="text-muted">{r.citation}</span>
          </div>
        ))}
      </SectionCard>
    </>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function MtTrnaAtlasPage() {
  const [tab, setTab]           = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError]       = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/mt-trna-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/mt-trna-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/mt-trna-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div className="container-fluid py-3">
      {/* Page header */}
      <div className="d-flex align-items-start mb-3 flex-wrap gap-2">
        <div className="flex-grow-1">
          <h4 className="mb-0 fw-bold" style={{ color: COLOR }}>
            🧬 MT-tRNA Atlas — Complete 22-Gene Mitochondrial tRNA Genome
          </h4>
          <div className="text-muted small mt-1">
            All 22 human mt-tRNA genes · H-strand (14) + L-strand (8, NGS pitfall) · 880-patient aggregate ·
            Combined CI+CIV fingerprint (20 genes) + MELAS (MT-TL1) + MERRF (MT-TK) hallmarks ·
            Universal drug CIs + nuclear DDx per gene
          </div>
          <div className="mt-1">
            <span className="badge me-1" style={{ backgroundColor: COLOR2 }}>14 H-strand — Standard NGS</span>
            <span className="badge me-1" style={{ backgroundColor: COLOR3 }}>8 L-strand — NGS Pitfall ⚠</span>
            <span className="badge me-1" style={{ backgroundColor: COLOR6 }}>7 Hallmark-Unique Genes</span>
            <span className="badge me-1" style={{ backgroundColor: COLOR4 }}>4 KSS 4977bp Deletion</span>
            <span className="badge" style={{ backgroundColor: COLOR }}>WES Misses ALL 22</span>
          </div>
        </div>
      </div>

      {error && <div className="alert alert-danger">API error: {error}</div>}

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-semibold' : ''}`}
              style={tab === i ? { color: COLOR, borderBottomColor: COLOR } : {}}
              onClick={() => setTab(i)}
            >
              {t}
            </button>
          </li>
        ))}
      </ul>

      {/* Tab content */}
      {tab === 0 && <OverviewTab data={overview} />}
      {tab === 1 && <GeneTableTab data={breakdown} />}
      {tab === 2 && <ClinicalAtlasTab overviewData={overview} breakdownData={breakdown} />}
      {tab === 3 && <DefinitionsTab data={definitions} />}
    </div>
  );
}
