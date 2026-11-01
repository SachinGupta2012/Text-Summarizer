import { ServiceStatus } from '@/components/service-status';
import { fetchApiHealth } from '@/lib/api-health';

export const dynamic = 'force-dynamic';

export default async function Home() {
  const health = await fetchApiHealth();

  return (
    <main>
      <section className="hero" aria-labelledby="project-title">
        <p className="eyebrow">Phase 1 · Architecture foundation</p>
        <h1 id="project-title">Transformer Summarizer</h1>
        <p className="lede">
          A production-style learning project for multi-domain document
          summarization with Qwen3, QLoRA, and a measured base-versus-fine-tuned
          evaluation workflow.
        </p>
      </section>

      <section className="status-grid" aria-label="Foundation status">
        <ServiceStatus
          label="Web application"
          status="healthy"
          detail="Next.js foundation is running."
        />
        <ServiceStatus
          label="API service"
          status={health ? 'healthy' : 'unavailable'}
          detail={
            health
              ? `${health.service} ${health.version} is ready.`
              : 'The API health endpoint is not reachable.'
          }
        />
      </section>

      <section className="boundary" aria-labelledby="boundary-title">
        <h2 id="boundary-title">Current boundary</h2>
        <p>
          This phase establishes architecture, tooling, and runnable service
          shells. Dataset selection begins in Phase 2; no model or dataset is
          downloaded here.
        </p>
      </section>
    </main>
  );
}
