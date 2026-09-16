// wa-gateway/server.js — Baileys QR sidecar. ponytail: ~90 lines, stdlib-shaped.
// Run: cd wa-gateway && npm i && node server.js   (listens :3001)
// Pair: GET /qr -> scan with clinic phone (Linked devices) -> /status = connected.
const express = require('express');
const QRCode = require('qrcode');
const { default: makeWASocket, useMultiFileAuthState, DisconnectReason } = require('@whiskeysockets/baileys');

const app = express();
app.use(express.json());
const PY = process.env.PY_BACKEND || 'http://127.0.0.1:8000';
let sock = null, qr = null, status = 'starting';

async function boot() {
  const { state, saveCreds } = await useMultiFileAuthState('./auth');
  sock = makeWASocket({ auth: state });
  sock.ev.on('creds.update', saveCreds);
  sock.ev.on('connection.update', async (u) => {
    if (u.qr) { qr = u.qr; status = 'qr'; }
    if (u.connection === 'open') { qr = null; status = 'connected'; }
    if (u.connection === 'close') {
      const code = u.lastDisconnect?.error?.output?.statusCode;
      status = 'disconnected';
      if (code !== DisconnectReason.loggedOut) boot(); // auto-reconnect, re-QR if needed
    }
  });
  // inbound STOP/HELP -> forward to Python webhook (consent honored there)
  sock.ev.on('messages.upsert', async ({ messages }) => {
    const m = messages?.[0];
    const text = m?.message?.conversation || m?.message?.extendedTextMessage?.text || '';
    const phone = '+' + (m?.key?.remoteJid || '').replace(/[^0-9]/g, '');
    if (m && !m.key.fromMe && text) {
      try { await fetch(PY + '/api/whatsapp/webhook', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone, text }) }); } catch {}
    }
  });
}

app.post('/status', (req, res) => res.json({ status, gateway: true }));
app.post('/qr', async (req, res) => {
  if (!qr) return res.json({ qr: null, status });
  res.json({ qr: await QRCode.toDataURL(qr), status });
});
app.post('/send', async (req, res) => {
  const { to, text } = req.body || {};
  if (!sock || status !== 'connected') return res.status(503).json({ error: 'not-connected', status });
  const jid = String(to).replace(/[^0-9]/g, '') + '@s.whatsapp.net';
  const r = await sock.sendMessage(jid, { text: String(text || '').slice(0, 1500) });
  res.json({ id: r?.key?.id || 'sent' });
});

app.listen(3001, () => console.log('wa-gateway :3001'));
boot();
