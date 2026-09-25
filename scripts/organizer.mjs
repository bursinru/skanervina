import { randomBytes } from 'node:crypto';
import { spawn } from 'node:child_process';
import { access, chmod, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dataset = resolve(root, 'Датасет');
const catalog = resolve(dataset, 'strapi_output0709_enriched.csv');
const images = resolve(dataset, 'prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads');
const readyUrl = 'http://127.0.0.1:3000/readyz';
const args = process.argv.slice(2);
const command = args[0] || 'help';

process.chdir(root);

function say(message) {
  console.log(`\n${message}`);
}

function fail(message) {
  console.error(`\n${message}`);
  process.exitCode = 1;
}

function run(program, commandArgs, { stdio = 'inherit' } = {}) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(program, commandArgs, { cwd: root, stdio });
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      if (code === 0 || signal === 'SIGINT') resolvePromise();
      else reject(new Error(`${program} завершился с кодом ${code ?? signal}`));
    });
  });
}

async function checkDocker() {
  try {
    await run('docker', ['compose', 'version'], { stdio: 'ignore' });
    await run('docker', ['info', '--format', '{{.ServerVersion}}'], { stdio: 'ignore' });
  } catch {
    throw new Error('Docker Compose не найден или Docker не запущен. Установите и запустите Docker Desktop, затем повторите команду.');
  }
}

async function checkDataset() {
  const missing = [];
  for (const path of [catalog, images]) {
    try {
      await access(path);
      const info = await stat(path);
      if ((path === catalog && !info.isFile()) || (path === images && !info.isDirectory())) missing.push(path);
    } catch {
      missing.push(path);
    }
  }
  if (missing.length) {
    throw new Error(
      `Не найдены файлы каталога:\n${missing.map((path) => `  • ${path}`).join('\n')}\n` +
      'Скопируйте CSV и папку с фотографиями в Датасет/ по образцу из README.md.',
    );
  }
}

async function ensureEnv() {
  const envPath = resolve(root, '.env');
  let contents = '';
  try {
    contents = await readFile(envPath, 'utf8');
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }

  const additions = [];
  for (const key of ['POSTGRES_PASSWORD', 'SCANNER_ADMIN_TOKEN']) {
    const line = new RegExp(`^[ \\t]*${key}[ \\t]*=[ \\t]*(.*?)[ \\t]*$`, 'm').exec(contents);
    if (!line || !line[1] || line[1] === '""' || line[1] === "''") {
      const value = randomBytes(32).toString('hex');
      if (line) contents = contents.replace(line[0], `${key}=${value}`);
      else additions.push(`${key}=${value}`);
    }
  }
  if (!contents.trim()) {
    contents = '# Создано автоматически. Не публикуйте этот файл.\n';
  }
  if (additions.length) contents = `${contents.trimEnd()}\n${additions.join('\n')}\n`;
  if (contents && !contents.endsWith('\n')) contents += '\n';
  await writeFile(envPath, contents, { mode: 0o600 });
  await chmod(envPath, 0o600);
}

async function waitUntilReady(timeoutMs = 10 * 60 * 1000) {
  const deadline = Date.now() + timeoutMs;
  let lastMessage = '';
  let lastNotice = 0;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(readyUrl, { signal: AbortSignal.timeout(5000) });
      const state = await response.json();
      if (response.ok && state.status === 'ok') {
        say(`Сканер готов: ${readyUrl.replace('/readyz', '/scanner')}`);
        console.log(`В каталоге ${state.catalog_size} вин, проиндексировано изображений: ${state.indexed_images}.`);
        return;
      }
      lastMessage = state.catalog_error || state.status || `HTTP ${response.status}`;
    } catch {
      lastMessage = 'сервисы запускаются';
    }
    if (Date.now() - lastNotice > 30000) {
      console.log(`Ожидание готовности: ${lastMessage}…`);
      lastNotice = Date.now();
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 3000));
  }
  throw new Error(`Сканер не стал готов за отведённое время (${lastMessage}). Посмотрите журналы: npm run organizer:logs`);
}

async function setup() {
  await checkDocker();
  await checkDataset();
  await ensureEnv();
  say('Сборка контейнеров…');
  await run('docker', ['compose', 'build']);
  say('Запуск базы данных…');
  await run('docker', ['compose', 'up', '-d', 'db']);
  say('Импорт каталога и подготовка поиска. При первом запуске загружаются веса модели…');
  await run('docker', [
    'compose', 'run', '--rm', 'recognition', 'python', '-m', 'app.import_catalog',
    '--catalog', '/catalog/strapi_output0709_enriched.csv', '--index',
    '--images', '/catalog/prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads',
  ]);
  say('Запуск сканера…');
  await run('docker', ['compose', 'up', '-d']);
  await waitUntilReady();
  console.log('Настройка завершена. Конфигурация и пароли сохранены в .env.');
}

async function start() {
  await checkDocker();
  await run('docker', ['compose', 'up', '-d']);
  await waitUntilReady();
}

function help() {
  console.log(`Команды для организатора:\n\n  npm run organizer:setup   Первая настройка, сборка и запуск\n  npm run organizer:start   Запустить сервисы\n  npm run organizer:stop    Остановить сервисы\n  npm run organizer:status  Показать состояние сервисов\n  npm run organizer:logs    Смотреть журналы`);
}

try {
  if (args.length > 1 || args[0]?.startsWith('-')) {
    help();
    fail('Команда не распознана.');
  } else if (command === 'setup') {
    await setup();
  } else if (command === 'start') {
    await start();
  } else if (command === 'stop') {
    await checkDocker();
    await run('docker', ['compose', 'stop']);
  } else if (command === 'status') {
    await checkDocker();
    await run('docker', ['compose', 'ps']);
  } else if (command === 'logs') {
    await checkDocker();
    await run('docker', ['compose', 'logs', '--follow', '--tail=100']);
  } else if (command === 'help') {
    help();
  } else {
    help();
    fail(`Неизвестная команда: ${command}`);
  }
} catch (error) {
  fail(error.message);
}
