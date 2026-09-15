import { mkdir, cp, rm, writeFile } from 'node:fs/promises';
await rm('dist', { recursive: true, force: true });
await mkdir('dist/scanner', { recursive: true });
await cp('public', 'dist', { recursive: true });
const config = {
  recognitionEndpoint: process.env.SCANNER_RECOGNITION_ENDPOINT || '',
  imageBaseUrl: process.env.SCANNER_IMAGE_BASE_URL || 'https://api.vino-svoe.ru/v1/img/str-api/800/800/resize/uploads/'
};
await writeFile('dist/config.js', `window.SCANNER_CONFIG = ${JSON.stringify(config)};\n`);
await cp('public/index.html', 'dist/scanner/index.html');
console.log('Built static scanner → dist/scanner/');
