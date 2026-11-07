/**
 * OncoAI: Multi-process runner for React Frontend + Python Backend
 */
const { spawn } = require('child_process');

console.log('\x1b[36m%s\x1b[0m', '========================================================');
console.log('\x1b[36m%s\x1b[0m', '  🎗️  OncoAI: Cancer Prediction Machine Learning System');
console.log('\x1b[36m%s\x1b[0m', '  React + Node.js (Frontend) & Python Flask API (Backend)');
console.log('\x1b[36m%s\x1b[0m', '========================================================');

// 1. Launch Python Backend
console.log('\x1b[32m%s\x1b[0m', '[1/2] Starting Python Backend on http://127.0.0.1:8000 ...');
const pythonCmd = process.platform === 'win32' ? 'py' : 'python3';
const backend = spawn(pythonCmd, ['backend/main.py'], {
  stdio: 'pipe',
  shell: true
});

backend.stdout.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.log('\x1b[34m[Backend]\x1b[0m', line);
});

backend.stderr.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.log('\x1b[33m[Backend]\x1b[0m', line);
});

// 2. Launch Vite React Frontend
console.log('\x1b[32m%s\x1b[0m', '[2/2] Starting React Frontend on http://localhost:3000 ...');
const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const frontend = spawn(npmCmd, ['--prefix', 'frontend', 'run', 'dev'], {
  stdio: 'pipe',
  shell: true
});

frontend.stdout.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.log('\x1b[35m[Frontend]\x1b[0m', line);
});

frontend.stderr.on('data', (data) => {
  const line = data.toString().trim();
  if (line) console.log('\x1b[35m[Frontend]\x1b[0m', line);
});

function shutdown() {
  console.log('\n\x1b[31m%s\x1b[0m', 'Shutting down OncoAI servers...');
  backend.kill();
  frontend.kill();
  process.exit();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
