// Same-origin public gateway. The Python service remains internal in Docker.
export default defineEventHandler(async (event) => {
  const upstream = useRuntimeConfig(event).recognitionUrl
  const url = getRequestURL(event)
  // Explicit allowlist: internal docs and arbitrary upstream URLs are not exposed.
  const allowed = /^\/(v1\/(recognize|bottles|profile|search|sommelier|catalog\/[^/]+(?:\/alternatives)?|eval\/predict)|healthz|readyz|config\.js|scanner(?:\/[^/]+)?\/?|app\.js|styles\.css|favicon\.svg|assets\/[^?]+)?$/.test(url.pathname)
  if (!allowed) throw createError({ statusCode: 404, statusMessage: 'Not found' })
  const limit = url.pathname === '/v1/profile' ? 512000 : 16 * 1024 * 1024
  const length = Number(getHeader(event, 'content-length') || 0)
  if (length > limit) throw createError({ statusCode: 413, statusMessage: 'Request too large' })
  return proxyRequest(event, upstream + url.pathname + url.search)
})
