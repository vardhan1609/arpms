import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev only: same routing as the nginx gateway, services on local ports 8001-8005
const svc = (port: number) => ({ target: `http://localhost:${port}`, changeOrigin: true })
const routes: Record<string, number> = {
  ingest: 8001, files: 8001, aircraft: 8001, lrus: 8001, flights: 8001, parameters: 8001, maintenance: 8001, quality: 8001,
  models: 8002, fleet: 8002, predictions: 8002, anomalies: 8002, changepoints: 8002, alerts: 8002, explanations: 8002,
  documents: 8003, snags: 8003, fta: 8003,
  diagnoses: 8004, recommendations: 8004, audit: 8004, reports: 8004,
  jobs: 8005, pipeline: 8005,
}
export default defineConfig({
  plugins: [react()],
  server: { proxy: Object.fromEntries(Object.entries(routes).map(([k, p]) => [`/api/v1/${k}`, svc(p)])) },
})
