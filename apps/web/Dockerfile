# syntax=docker/dockerfile:1
FROM node:24-alpine AS dependencies

RUN corepack enable
WORKDIR /workspace

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/package.json apps/web/package.json
RUN pnpm install --frozen-lockfile

FROM dependencies AS build
ENV NEXT_TELEMETRY_DISABLED=1
COPY apps/web apps/web
RUN pnpm --filter @summarizer/web build

FROM node:24-alpine AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
WORKDIR /workspace

RUN addgroup --system --gid 1001 nextjs \
    && adduser --system --uid 1001 --ingroup nextjs nextjs

COPY --from=build --chown=nextjs:nextjs /workspace/apps/web/public ./apps/web/public
COPY --from=build --chown=nextjs:nextjs /workspace/apps/web/.next/standalone ./
COPY --from=build --chown=nextjs:nextjs /workspace/apps/web/.next/static ./apps/web/.next/static

USER nextjs
EXPOSE 3000

CMD ["node", "apps/web/server.js"]
