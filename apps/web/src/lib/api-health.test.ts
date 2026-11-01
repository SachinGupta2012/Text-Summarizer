import { describe, expect, it, vi } from 'vitest';

import { fetchApiHealth } from './api-health';

describe('fetchApiHealth', () => {
  it('returns a validated health response', async () => {
    process.env.API_INTERNAL_URL = 'http://api:8000';
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          status: 'healthy',
          service: 'summarizer-api',
          version: '0.1.0',
        }),
        { status: 200 },
      ),
    );

    await expect(fetchApiHealth(fetcher)).resolves.toEqual({
      status: 'healthy',
      service: 'summarizer-api',
      version: '0.1.0',
    });
  });

  it('returns null when the API payload is invalid', async () => {
    process.env.API_INTERNAL_URL = 'http://api:8000';
    const fetcher = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        new Response(JSON.stringify({ status: 'unknown' }), { status: 200 }),
      );

    await expect(fetchApiHealth(fetcher)).resolves.toBeNull();
  });
});
