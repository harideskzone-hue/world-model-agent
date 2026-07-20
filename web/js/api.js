/**
 * api.js — thin fetch wrapper around the Flask API endpoints
 * Falls back to loading sample_data/demo_session.jsonl when running offline
 */

const API_BASE = 'http://localhost:5050/api';
let _useServer = false;

async function _checkServer() {
  try {
    const r = await fetch(`${API_BASE}/sessions`, { signal: AbortSignal.timeout(1000) });
    _useServer = r.ok;
  } catch { _useServer = false; }
  return _useServer;
}

async function _get(path) {
  if (!_useServer) return null;
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) return null;
  return r.json();
}

async function _post(path, body = {}) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
  });
  return r.json();
}

// Load JSONL file (for sample mode)
async function loadSampleSession(filename = 'sample_data/demo_session.jsonl') {
  const r = await fetch(filename);
  const text = await r.text();
  return text.trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
}

const API = {
  init: _checkServer,
  useServer: () => _useServer,

  async getSessions() {
    if (_useServer) return _get('/sessions');
    // Build from sample data
    const lines = await loadSampleSession();
    const hdr = lines.find(l => l.type === 'header') || {};
    const ftr = lines.find(l => l.type === 'footer') || {};
    return [{
      session_id: 'demo_shadfly',
      game: hdr.game || 'TextWorld Demo',
      started_at: hdr.started_at || '',
      mode: 'sample',
      status: 'done',
      turns_so_far: lines.filter(l => l.type === 'turn').length,
      score: ftr.data?.final_score || 1,
      max_score: ftr.data?.max_score || 1,
      won: ftr.data?.won ?? true,
    }];
  },

  async getTurns(sid, since = -1) {
    if (_useServer) return _get(`/sessions/${sid}/turns?since=${since}`) || [];
    const lines = await loadSampleSession();
    return lines.filter(l => l.type === 'turn' && l.turn_id > since);
  },

  async getGraph(sid, atTurn = 999) {
    if (_useServer) return _get(`/sessions/${sid}/graph?at_turn=${atTurn}`);
    // Build graph from facts up to atTurn
    const lines = await loadSampleSession();
    const nodes = {}, edgeMap = {};
    for (const line of lines) {
      if (line.type !== 'turn' || line.turn_id > atTurn) continue;
      const facts = line.data?.extracted_facts || [];
      for (const f of facts) {
        for (const ent of [f.subject, f.object]) {
          if (!ent || ent in nodes) continue;
          const ntype = ['kitchen','attic','restroom','garden'].some(r=>ent.includes(r)) ? 'room'
                       : ent === 'player' ? 'character' : 'object';
          nodes[ent] = { id: ent, label: ent, node_type: ntype, status: 'active', confidence: f.confidence };
        }
        const key = `${f.subject}_${f.relation}_${f.object}`;
        edgeMap[key] = { id: key, subject: f.subject, relation: f.relation, object: f.object,
          confidence: f.confidence, t_valid_from: line.turn_id, t_valid_until: null,
          status: 'active', extraction_method: f.extraction_method };
      }
    }
    return { nodes: Object.values(nodes), edges: Object.values(edgeMap) };
  },

  async getFooter(sid) {
    if (_useServer) return _get(`/sessions/${sid}/footer`);
    const lines = await loadSampleSession();
    return lines.find(l => l.type === 'footer') || null;
  },

  async getTests() { return _get('/tests'); },
  async runTests() { return _post('/tests/run'); },
  async getEvaluation() { return _get('/evaluation'); },
  async startSession(opts) { return _post('/sessions', opts); },
};

window.API = API;
