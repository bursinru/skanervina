FROM node:22-slim AS build
WORKDIR /app
COPY gateway/package*.json ./
RUN npm ci
COPY gateway ./
RUN npm run build
FROM node:22-slim
WORKDIR /app
COPY --from=build --chown=node:node /app/.output ./.output
USER node
ENV HOST=0.0.0.0 PORT=3000
EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
