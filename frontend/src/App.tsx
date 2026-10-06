import { NavLink, Route, Routes } from 'react-router-dom'
import * as P from './pages'

const NAV: [string, string][] = [
  ['/', 'Fleet'], ['/alerts', 'Alerts'], ['/diagnoses', 'Diagnoses'], ['/search', 'Snag & doc search'],
  ['/fta', 'Fault trees'], ['/models', 'Models'], ['/data', 'Data & pipeline'], ['/audit', 'Audit trail'],
]

export default function App() {
  return (
    <div className="layout">
      <nav className="side">
        <h1>ARPMS</h1>
        <small>Predictive maintenance</small>
        {NAV.map(([to, label]) => <NavLink key={to} to={to} end={to === '/'}>{label}</NavLink>)}
      </nav>
      <main>
        <div className="banner">SYNTHETIC DATA - decision support only. Engineer approval is required before any maintenance action.</div>
        <Routes>
          <Route path="/" element={<P.Fleet />} />
          <Route path="/aircraft/:id" element={<P.Aircraft />} />
          <Route path="/flights/:id" element={<P.Flight />} />
          <Route path="/lrus/:id" element={<P.Lru />} />
          <Route path="/alerts" element={<P.Alerts />} />
          <Route path="/diagnoses" element={<P.Diagnoses />} />
          <Route path="/diagnoses/:id" element={<P.Diagnosis />} />
          <Route path="/search" element={<P.Search />} />
          <Route path="/fta" element={<P.Fta />} />
          <Route path="/models" element={<P.Models />} />
          <Route path="/data" element={<P.Data />} />
          <Route path="/audit" element={<P.Audit />} />
        </Routes>
      </main>
    </div>
  )
}
