// Local fallback configuration. For Vercel, set the same values as project
// environment variables; scripts/build.mjs injects them into dist/config.js.
window.SCANNER_CONFIG = {
  // No photo is sent when the endpoint is empty.
  recognitionEndpoint: '',
  // The catalog image URL format used by api.vino-svoe.ru.
  imageBaseUrl: 'https://api.vino-svoe.ru/v1/img/str-api/800/800/resize/uploads/'
};
