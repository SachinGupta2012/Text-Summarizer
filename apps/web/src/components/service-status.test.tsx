import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ServiceStatus } from './service-status';

describe('ServiceStatus', () => {
  it('renders its label and detail', () => {
    render(
      <ServiceStatus
        label="API service"
        status="healthy"
        detail="API is ready."
      />,
    );

    expect(screen.getByText('API service')).toBeInTheDocument();
    expect(screen.getByText('API is ready.')).toBeInTheDocument();
  });
});
