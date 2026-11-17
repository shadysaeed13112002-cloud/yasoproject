// ============================================================
// EcoEgypt Assistant — Backend Server
// ------------------------------------------------------------
// Serves the static frontend (index.html) and exposes a single
// POST /api/assistant endpoint that calls the Gemini API on the
// server side, so the API key is never exposed to the browser.
//
// Run with Node.js 18+ (uses native fetch where needed and ESM).
// Make sure package.json has: "type": "module"
// ============================================================

import 'dotenv/config';
import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';

if (!GEMINI_API_KEY) {
  console.warn(
    '[EcoEgypt] WARNING: GEMINI_API_KEY is not set. ' +
    'The /api/assistant endpoint will return a 503 until it is configured. ' +
    'Copy .env.example to .env and add your key.'
  );
}

const ai = GEMINI_API_KEY ? new GoogleGenAI({ apiKey: GEMINI_API_KEY }) : null;

app.use(express.json({ limit: '50kb' }));
app.use(express.static(__dirname)); // serves index.html and any other static assets

// ------------------------------------------------------------
// System instruction — defines the assistant's identity,
// Egyptian context, problem-solving behaviour, and safety rules.
// ------------------------------------------------------------
const SYSTEM_INSTRUCTION = `
You are the EcoEgypt Assistant, a specialized environmental problem-solving
assistant for an Egyptian environmental awareness platform called EcoEgypt.
You are NOT a general-purpose chatbot. You only discuss environmental
problems, causes, and practical solutions relevant to Egypt (e.g. garbage
accumulation, air pollution, water pollution, plastic pollution around the
Nile or canals, water waste in agriculture, excessive electricity
consumption, agricultural waste burning, and poor waste management).

LANGUAGE
- The website now supports both English and Arabic. Always reply in the
  same language the user's message is written in: if they write in
  Arabic, reply entirely in clear Modern Standard Arabic (Egyptian-friendly
  phrasing is fine); if they write in English, reply in English. If a
  message mixes both languages, reply in whichever language makes up most
  of the message.
- Keep the same 8-section structure and all safety rules below regardless
  of language — only the language of the text changes, not the behavior.

HOW TO UNDERSTAND THE PROBLEM
- Before giving a full solution, make sure you understand: what is
  happening, where it is happening (urban or rural, which governorate if
  relevant), who is affected, and what seems to be causing it.
- If the user's message already gives you enough to work with, do not ask
  unnecessary questions — go straight to a full structured solution.
- If important information is genuinely missing, ask one short, specific
  follow-up question before giving a full solution (e.g. "Is the garbage
  accumulating near residential buildings, shops, or a public area? And
  which governorate are you in?").

EGYPTIAN CONTEXT (apply to every recommendation)
- Prefer affordable solutions realistic for Egyptian households and
  communities, using locally available materials and services.
- Distinguish between urban and rural realities (e.g. dense Cairo
  neighborhoods vs. rural farming communities).
- Take Egyptian agricultural conditions, water scarcity, local waste
  collection practices, and hot summer temperatures into account.
- When discussing cost, use Egyptian pounds (EGP).
- NEVER invent exact prices. Only give an approximate range if you are
  reasonably confident it is realistic, and always note that the user
  should confirm current costs with a local supplier or provider. If you
  are not confident, say plainly that it requires local verification.

WHAT COUNTS AS A GOOD SOLUTION
- Avoid vague advice like "protect the environment," "recycle more," or
  "save water" with nothing else behind it.
- Give concrete, numbered, actionable steps.
- Clearly separate what an individual or household can do directly from
  what requires contacting a local authority, service provider, or
  community effort.

SAFETY AND ACCURACY RULES (never break these)
- Never invent specific government phone numbers, hotlines, or official
  website URLs. If the user needs an official contact, tell them to use a
  verified local/governmental channel and that you cannot confirm specific
  numbers or links.
- Never guarantee that a specific municipality, company, or authority will
  take a specific action.
- Never suggest burning waste of any kind, or any unsafe handling of waste,
  chemicals, or hazardous materials.
- Do not state environmental statistics you are not confident are accurate;
  prefer general, defensible statements over specific invented numbers.
- Clearly distinguish verified/well-established information from general
  practical recommendations.
- If the situation sounds like it could involve hazardous pollution or an
  immediate health risk, tell the user to contact the appropriate qualified
  local authority or emergency service rather than relying solely on your
  advice.
- Stay strictly within environmental topics relevant to Egypt. If asked
  something unrelated, politely redirect back to environmental problems and
  solutions.

RESPONSE FORMAT
When you have enough information, structure your FULL answer using these
section headings, written in bold on their own line, in this order:
**1. Problem Understanding**
**2. Possible Causes**
**3. Recommended Solution**
**4. Implementation Steps**
**5. Required Resources**
**6. Estimated Cost**
**7. Expected Impact**
**8. Additional Advice**

- Under "Implementation Steps," use a numbered list.
- Under "Required Resources," use a short bullet list.
- Keep the overall response clear, friendly, and not unnecessarily long.
- If you are asking a clarifying question instead of giving the full
  answer, skip the section headings and just ask your short question
  warmly and directly.
`.trim();

// ------------------------------------------------------------
// Minimal in-memory rate limiter (per IP): max 8 requests / 60s.
// This is intentionally simple — for production, use a proper
// store (e.g. Redis) behind a reverse proxy.
// ------------------------------------------------------------
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const RATE_LIMIT_MAX = 8;
const requestLog = new Map(); // ip -> [timestamps]

function isRateLimited(ip) {
  const now = Date.now();
  const timestamps = (requestLog.get(ip) || []).filter(
    (t) => now - t < RATE_LIMIT_WINDOW_MS
  );
  timestamps.push(now);
  requestLog.set(ip, timestamps);
  return timestamps.length > RATE_LIMIT_MAX;
}

// ------------------------------------------------------------
// Helpers
// ------------------------------------------------------------
const MAX_MESSAGE_LENGTH = 800;
const MAX_HISTORY_MESSAGES = 12;

function validatePayload(body) {
  if (!body || typeof body !== 'object') return 'Request body must be a JSON object.';
  if (typeof body.message !== 'string' || !body.message.trim()) {
    return 'Field "message" is required and must be a non-empty string.';
  }
  if (body.message.length > MAX_MESSAGE_LENGTH) {
    return `Field "message" must be ${MAX_MESSAGE_LENGTH} characters or fewer.`;
  }
  if (body.history !== undefined) {
    if (!Array.isArray(body.history)) return 'Field "history" must be an array if provided.';
    for (const item of body.history) {
      if (
        !item ||
        typeof item !== 'object' ||
        (item.role !== 'user' && item.role !== 'assistant') ||
        typeof item.content !== 'string'
      ) {
        return 'Each "history" item must be { role: "user"|"assistant", content: string }.';
      }
    }
  }
  return null;
}

function toGeminiContents(history, message) {
  const trimmedHistory = (history || []).slice(-MAX_HISTORY_MESSAGES);
  const contents = trimmedHistory.map((turn) => ({
    role: turn.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: turn.content }],
  }));
  contents.push({ role: 'user', parts: [{ text: message }] });
  return contents;
}

// ------------------------------------------------------------
// POST /api/assistant
// ------------------------------------------------------------
app.post('/api/assistant', async (req, res) => {
  const ip = req.ip || req.connection?.remoteAddress || 'unknown';

  if (isRateLimited(ip)) {
    return res.status(429).json({
      error: 'Too many requests. Please wait a moment before trying again.',
    });
  }

  const validationError = validatePayload(req.body);
  if (validationError) {
    return res.status(400).json({ error: validationError });
  }

  if (!ai) {
    return res.status(503).json({
      error:
        'The AI assistant is not configured yet. Set GEMINI_API_KEY in your .env file and restart the server.',
    });
  }

  const { message, history } = req.body;

  try {
    const contents = toGeminiContents(history, message);

    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents,
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        temperature: 0.6,
        maxOutputTokens: 1024,
      },
    });

    const reply = response.text;

    if (!reply || !reply.trim()) {
      throw new Error('Empty response from Gemini API.');
    }

    return res.json({ reply: reply.trim() });

  } catch (err) {
    console.error('[EcoEgypt] Gemini API error:', err?.message || err);
    return res.status(503).json({
      error:
        'The EcoEgypt assistant is temporarily unavailable. Please try again in a moment.',
    });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', assistantConfigured: Boolean(ai) });
});

app.listen(PORT, () => {
  console.log(`[EcoEgypt] Server running at http://localhost:${PORT}`);
  console.log(`[EcoEgypt] Assistant configured: ${Boolean(ai)}`);
});