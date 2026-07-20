/**
 * session-browser.js — Session Browser view
 */
const SessionBrowserView = (() => {
  function render(sessions) {
    const grid = document.getElementById('sessions-grid');
    if (!sessions || sessions.length === 0) {
      grid.innerHTML = `<div class="empty-state"><div class="empty-icon">📂</div>No sessions found.</div>`;
      return;
    }
    grid.innerHTML = sessions.map(s => {
      const isSample = s.mode === 'sample';
      const isWon = s.won;
      const isRunning = s.status === 'running';
      const scoreStr = `${s.score}/${s.max_score}`;
      const badgeClass = isSample ? 'badge-sample' : isWon ? 'badge-won' : isRunning ? 'badge-running' : 'badge-done';
      const badgeText = isSample ? '⚡ Sample Demo' : isWon ? '🏆 Won' : isRunning ? '⟳ Running' : '◉ Done';
      return `
        <div class="session-card ${isSample ? 'sample' : ''}" data-sid="${s.session_id}" onclick="SessionBrowserView.openSession('${s.session_id}')">
          <div class="card-badge ${badgeClass}">${badgeText}</div>
          <div class="card-title">${s.game || 'TextWorld Session'}</div>
          <div class="card-meta">
            <span>🎲 ${s.session_id}</span>
            <span>🔄 ${s.turns_so_far} turns</span>
            ${s.started_at ? `<span>🕐 ${new Date(s.started_at).toLocaleTimeString()}</span>` : ''}
          </div>
          <div class="card-score">${scoreStr} <span>score</span></div>
        </div>`;
    }).join('');
  }

  async function openSession(sid) {
    Store.set({ currentSession: sid, currentTurnId: 0, allTurns: [], graphTurn: 0 });
    const turns = await API.getTurns(sid, -1);
    Store.set({ allTurns: turns || [] });
    showView('trace');
    TurnTraceView.init(sid, turns || []);
  }

  async function load() {
    const sessions = await API.getSessions();
    render(sessions);
  }

  return { load, render, openSession };
})();
window.SessionBrowserView = SessionBrowserView;
