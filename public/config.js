// Fallback for the static preview. scripts/build.mjs rewrites dist/config.js.
// The Docker scanner serves config.js from the recognition service.
window.SCANNER_CONFIG = {
  // No photo is sent when the endpoint is empty.
  recognitionEndpoint: '',
  // The catalog image URL format used by api.vino-svoe.ru.
  imageBaseUrl: 'https://api.vino-svoe.ru/v1/img/str-api/800/800/resize/uploads/'
};
