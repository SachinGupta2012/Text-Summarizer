export type ApiHealth = {
  status: 'healthy';
  service: 'summarizer-api';
  version: string;
};

function isApiHealth(value: unknown): value is ApiHealth {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  return (
    candidate.status === 'healthy' &&
    candidate.service === 'summarizer-api' &&
    typeof candidate.version === 'string'
  );
}

export async function fetchApiHealth(
  fetcher: typeof fetch = fetch,
): Promise<ApiHealth | null> {
  const baseUrl =
    process.env.API_INTERNAL_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (!baseUrl) {
    return null;
  }

  try {
    const response = await fetcher(`${baseUrl}/health/ready`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(2_000),
    });
    if (!response.ok) {
      return null;
    }

    const payload: unknown = await response.json();
    return isApiHealth(payload) ? payload : null;
  } catch {
    return null;
  }
}
