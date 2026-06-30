'use client';
import Link from 'next/link';
import { ROLES, FEATURES, AGENTIC } from '../../lib/nav';
import { OPS } from '../../lib/ops';
export default function SiteNav() {
  const dd = (title, items, base) => (
    <li className="nav-item dropdown">
      <a className="nav-link dropdown-toggle text-white" href="#" data-bs-toggle="dropdown">{title}</a>
      <ul className="dropdown-menu dropdown-menu-dark">
        {items.map(i => <li key={i.id}><Link className="dropdown-item" href={`${base}/${i.id}`}>{i.icon?i.icon+' ':''}{i.name}</Link></li>)}
      </ul>
    </li>);
  return (
    <nav className="navbar navbar-expand-lg navbar-dark" style={{background:'#0b1f3a'}}>
      <div className="container-fluid">
        <Link className="navbar-brand fw-bold" href="/">🧠 NeuroAI</Link>
        <ul className="navbar-nav me-auto">
          <li className="nav-item"><Link className="nav-link text-white" href="/dashboard">Dashboard</Link></li>
          {dd('Departments', ROLES, '/role')}
          {dd('Features', FEATURES, '')}
          {dd('Agentic AI', AGENTIC, '/agentic')}
          {dd('Ops & Governance', OPS, '/ops')}
          <li className="nav-item"><Link className="nav-link text-white" href="/drift">📉 Drift</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognition-link">🔗 Cognition</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-scales">📋 Scales</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-types">AI Types</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epilepsy-nurse">💉 Epilepsy Nurse</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication">&#x1f48a; Medication</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-severity">&#x26a1; Seizure Severity</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mri-review">&#x1f9e0; MRI Review</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/inbox">&#x2709;&#xfe0f; Inbox</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/token-cost">&#x1f4b0; Token/Cost</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vector-db">&#x1f5c4;&#xfe0f; Vector DB</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/knowledge-graph">&#x1f578;&#xfe0f; Knowledge Graph</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-risk">&#x26a0;&#xfe0f; AI Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workflow">Workflow</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mcp-security">MCP Security</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stack">Tech Stack</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/classify">Classify</Link></li>
        </ul>
        <span className="navbar-text text-info small">● System Online · SSR</span>
      </div>
    </nav>);
}
