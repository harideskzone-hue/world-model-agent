/**
 * turn-trace.js — main turn-by-turn view (dev + demo)
 */
const TurnTraceView = (() => {
  let _sid = null;
  let _turns = [];
  let _currentIdx = 0;

  // Sub-goals from session
  const SUB_GOALS = [
    'go north to the attic',
    'open safe within the attic',
    'take shadfly',
    'go east to restroom',
    'put shadfly in dresser'
  ];

  function init(sid, turns) {
    _sid = sid;
    _turns = turns;
    _currentIdx = turns.length > 0 ? turns.length - 1 : 0;
    renderSidebar();
    if (turns.length > 0) {
      selectTurn(_currentIdx);
      // Auto-init graph on the graph panel immediately
      const maxTurn = turns[turns.length - 1].turn_id;
      WorldGraphView.init(sid, maxTurn);
      window._graphLoaded = true;
    }
  }

  function renderSidebar() {
    const list = document.getElementById('turn-list');
    list.innerHTML = _turns.map((t, i) => {
      const action = t.data?.slm_decision?.action_text || '—';
      const valid = t.data?.action_valid;
      const tick = valid === true ? '✅' : valid === false ? '⚠️' : '·';
      return `<div class="turn-item ${i === _currentIdx ? 'active' : ''}" onclick="TurnTraceView.selectTurn(${i})">
        <span class="turn-num">${t.turn_id}</span>
        <span class="turn-action">${action}</span>
        <span class="turn-tick">${tick}</span>
      </div>`;
    }).join('');
  }

  function selectTurn(idx) {
    _currentIdx = idx;
    renderSidebar();
    const turn = _turns[idx];
    if (!turn) return;
    renderCenter(turn);
    renderSubTabs(turn);
    // Sync graph slider and re-render
    const slider = document.getElementById('graph-turn-slider');
    const valEl = document.getElementById('graph-turn-val');
    if (slider) {
      slider.value = turn.turn_id;
      if (valEl) valEl.textContent = turn.turn_id;
      if (window._graphLoaded) WorldGraphView.render(turn.turn_id);
    }
  }

  function renderCenter(turn) {
    const d = turn.data || {};
    const obs = d.observation || {};
    const decision = d.slm_decision || {};
    const valid = d.action_valid;
    const done = d.done;

    // Header
    document.getElementById('tc-turn-badge').textContent = `Turn ${turn.turn_id}`;
    document.getElementById('tc-subgoal').textContent = d.sub_goal ? `Sub-goal: ${d.sub_goal}` : '';
    const chip = document.getElementById('tc-action-chip');
    chip.textContent = decision.action_text || '—';
    chip.className = `action-chip ${valid ? 'chip-valid' : 'chip-invalid'}`;

    // Reward bar
    const rewardBar = document.getElementById('tc-reward-bar');
    if (done && d.reward > 0) {
      rewardBar.className = 'reward-bar reward-won';
      rewardBar.innerHTML = `<span class="reward-value">🏆 WON!</span><span class="reward-label">Score: ${obs.score}/${obs.max_score} · Reward: +${d.reward}</span>`;
    } else {
      rewardBar.className = 'reward-bar';
      rewardBar.innerHTML = `<span class="reward-value" style="color:var(--text-primary)">${obs.score || 0}/${obs.max_score || 1}</span><span class="reward-label">score · reward: ${d.reward || 0}</span>`;
    }

    // Observation blocks
    let html = '';
    if (obs.description) {
      html += obsBlock('🌍 Observation', obs.description);
    }
    if (obs.feedback && obs.feedback !== obs.description) {
      html += obsBlock('💬 Feedback', obs.feedback);
    }
    if (obs.inventory && obs.inventory !== 'You are carrying nothing.') {
      html += obsBlock('🎒 Inventory', obs.inventory);
    }
    if (d.admissible_commands?.length) {
      html += obsBlock('✅ Admissible Commands', d.admissible_commands.join(', '));
    }
    if (d.sub_goal_completed) {
      html += `<div class="obs-block" style="border-color:var(--green)">
        <div class="obs-block-header" style="color:var(--green)">✅ Sub-goal Completed</div>
        <div class="obs-block-body">${d.sub_goal_completed}</div>
      </div>`;
    }

    // Sub-goal progress
    const sgDone = d.sub_goal_index ?? -1;
    html += `<div class="subgoal-progress">
      <h4>Quest Progress</h4>
      ${SUB_GOALS.map((sg, i) => {
        const isDone = i < sgDone || (i === sgDone && d.sub_goal_completed);
        const isCurrent = i === sgDone && !d.sub_goal_completed;
        return `<div class="sg-item">
          <span class="sg-check">${isDone ? '✅' : isCurrent ? '▶' : '⬜'}</span>
          <span class="sg-text ${isDone ? 'done' : isCurrent ? 'current' : ''}">${sg}</span>
        </div>`;
      }).join('')}
    </div>`;

    document.getElementById('tc-body').innerHTML = html;
  }

  function obsBlock(label, text) {
    return `<div class="obs-block">
      <div class="obs-block-header">${label}</div>
      <div class="obs-block-body">${escHtml(text)}</div>
    </div>`;
  }

  function renderSubTabs(turn) {
    const d = turn.data || {};
    // Extraction tab
    renderExtractionPanel(d);
    // Decision tab
    renderDecisionPanel(d);
    // Context tab
    renderContextPanel(d);
    // Contradictions tab
    renderContradictionsPanel(d);
    // Graph will lazy-load when tab is clicked
  }

  function renderExtractionPanel(d) {
    const facts = d.extracted_facts || [];
    if (!facts.length) {
      document.getElementById('panel-extraction').innerHTML = '<div class="empty-state"><div class="empty-icon">🔍</div>No facts extracted this turn.</div>';
      return;
    }
    let html = `<div class="section-title">Extracted Facts (${facts.length})</div>
      <table class="fact-table">
        <thead><tr><th>Subject</th><th>Relation</th><th>Object</th><th>Method</th><th>Conf</th></tr></thead>
        <tbody>`;
    for (const f of facts) {
      const conf = Math.round((f.confidence || 0) * 100);
      const barColor = conf >= 75 ? 'var(--green)' : conf >= 50 ? 'var(--accent)' : 'var(--amber)';
      html += `<tr>
        <td style="font-family:var(--font-mono);font-size:11px">${escHtml(f.subject)}</td>
        <td><span style="color:var(--accent);font-family:var(--font-mono);font-size:11px">${escHtml(f.relation)}</span></td>
        <td style="font-family:var(--font-mono);font-size:11px">${escHtml(f.object)}</td>
        <td><span class="method-badge ${f.extraction_method === 'slm' ? 'method-slm' : 'method-rule'}">${f.extraction_method || 'rule'}</span></td>
        <td><div class="conf-bar-wrap"><div class="conf-bar" style="width:${conf}%;background:${barColor}"></div></div></td>
      </tr>`;
    }
    html += '</tbody></table>';
    document.getElementById('panel-extraction').innerHTML = html;
  }

  function renderDecisionPanel(d) {
    const dec = d.slm_decision || {};
    const valid = d.action_valid;
    const html = `
      <div class="decision-action">
        <span>${valid ? '✅' : '⚠️'}</span>
        <span>${escHtml(dec.action_text || '—')}</span>
      </div>
      <div class="decision-meta">
        <strong>Sub-goal:</strong> ${escHtml(d.sub_goal || '—')}<br>
        <strong>Latency:</strong> ${dec.latency_ms || 0} ms &nbsp; 
        <strong>Retries:</strong> ${dec.retry_count || 0} &nbsp;
        <strong>Method:</strong> ${dec.latency_ms === 0 ? 'rule_fallback (SLM offline)' : 'SLM'}
      </div>
      ${dec.restated_sub_goal ? `<div class="section-title">Restated Sub-goal</div>
        <div class="raw-output-box">${escHtml(dec.restated_sub_goal)}</div>` : ''}
      <div class="section-title" style="margin-top:14px">Raw SLM Output</div>
      <div class="raw-output-box">${escHtml(dec.raw_output || dec.action_text || '(rule-based decision)')}</div>
    `;
    document.getElementById('panel-decision').innerHTML = html;
  }

  function renderContextPanel(d) {
    const ctx = d.context_slice || {};
    const tokens = ctx.total_tokens_estimate || 0;
    const budget = 512;
    const pct = Math.min(100, Math.round((tokens / budget) * 100));
    const html = `
      <div class="section-title">Token Budget</div>
      <div class="token-bar-wrap"><div class="token-bar" style="width:${pct}%"></div></div>
      <div class="token-label">${tokens} / ${budget} tokens used (${pct}%)</div>
      <div class="section-title">Context Sent to SLM</div>
      <div class="raw-output-box">${escHtml(ctx.formatted_text || '(no context slice recorded)')}</div>
    `;
    document.getElementById('panel-context').innerHTML = html;
  }

  function renderContradictionsPanel(d) {
    const rpt = d.update_report || {};
    const revisions = rpt.revisions || [];
    let html = `
      <div class="update-stats">
        <div class="stat-box expanded"><div class="stat-num">${rpt.expanded || 0}</div><div class="stat-lbl">Expanded</div></div>
        <div class="stat-box corroborated"><div class="stat-num">${rpt.corroborated || 0}</div><div class="stat-lbl">Corroborated</div></div>
        <div class="stat-box revised"><div class="stat-num">${rpt.revised || 0}</div><div class="stat-lbl">Revised</div></div>
        <div class="stat-box rejected"><div class="stat-num">${rpt.rejected || 0}</div><div class="stat-lbl">Rejected</div></div>
      </div>`;
    if (revisions.length) {
      html += `<div class="section-title">Revision Log (${revisions.length})</div>`;
      for (const rev of revisions) {
        html += `<div class="revision-item">
          <div class="revision-type">${rev.action_type || 'supersede'}</div>
          <div class="revision-reason">${escHtml(rev.revision_reason || '—')}</div>
        </div>`;
      }
    } else {
      html += `<div class="empty-state" style="padding:24px"><div class="empty-icon">✓</div>No contradictions this turn.</div>`;
    }
    document.getElementById('panel-contradictions').innerHTML = html;
  }

  function addTurns(newTurns) {
    _turns = [..._turns, ...newTurns];
    renderSidebar();
    // Auto-scroll to latest
    _currentIdx = _turns.length - 1;
    selectTurn(_currentIdx);
  }

  function escHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  return { init, selectTurn, addTurns };
})();
window.TurnTraceView = TurnTraceView;
