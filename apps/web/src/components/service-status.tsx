type ServiceStatusProps = {
  label: string;
  status: 'healthy' | 'unavailable';
  detail: string;
};

export function ServiceStatus({ label, status, detail }: ServiceStatusProps) {
  return (
    <article className="status-card">
      <div className="status-heading">
        <span
          className="status-indicator"
          data-status={status}
          aria-hidden="true"
        />
        <span>{label}</span>
      </div>
      <p>{detail}</p>
    </article>
  );
}
