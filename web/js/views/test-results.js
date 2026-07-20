/**
 * test-results.js — Test Results view (V6)
 */
const TestResultsView = (() => {
  async function load() {
    const el = document.getElementById('test-results-body');
    el.innerHTML = '<div class="loading">Loading test results...</div>';
    
    const data = await API.getTests();
    if (!data) {
      el.innerHTML = `<div class="empty-state">
        <div class="empty-icon">🧪</div>
        <p>No test report found. Click <strong>Run Tests</strong> to generate one.</p>
      </div>`;
      return;
    }
    render(data);
  }

  function render(data) {
    const tests = data.tests || [];
    const summary = data.summary || {};
    const passed = summary.passed || tests.filter(t => t.outcome === 'passed').length;
    const failed = summary.failed || tests.filter(t => t.outcome === 'failed').length;
    const total = summary.total || tests.length;
    const allPass = failed === 0;

    // Update header
    document.getElementById('test-pass-count').textContent = `${passed}/${total}`;
    document.getElementById('test-pass-count').className = `pass-count ${allPass ? 'all-pass' : 'some-fail'}`;
    document.getElementById('test-status-text').textContent = allPass ? '✅ All tests passing' : `❌ ${failed} failing`;

    // Group by module
    const groups = {};
    for (const t of tests) {
      const parts = (t.nodeid || t.name || '').split('::');
      const mod = parts[0]?.replace('tests/unit/', '').replace('.py', '') || 'other';
      if (!groups[mod]) groups[mod] = [];
      groups[mod].push(t);
    }

    let html = '';
    for (const [mod, modTests] of Object.entries(groups)) {
      const modPassed = modTests.filter(t => t.outcome === 'passed').length;
      const modFailed = modTests.length - modPassed;
      const statusColor = modFailed === 0 ? 'var(--green)' : 'var(--red)';
      html += `
        <div class="test-module">
          <div class="test-module-header">
            <span style="color:${statusColor};font-size:16px">${modFailed === 0 ? '✅' : '❌'}</span>
            <h3>${mod}</h3>
            <span style="font-size:12px;color:var(--text-secondary)">${modPassed}/${modTests.length} passed</span>
          </div>
          <div class="test-module-body">
            ${modTests.map(t => {
              const name = (t.nodeid || t.name || '').split('::').slice(1).join(' :: ');
              const icon = t.outcome === 'passed' ? '✅' : '❌';
              const dur = t.duration ? `${(t.duration * 1000).toFixed(0)}ms` : '';
              return `<div class="test-row">
                <span style="font-size:13px">${icon}</span>
                <span class="test-name">${name}</span>
                <span class="test-duration">${dur}</span>
              </div>`;
            }).join('')}
          </div>
        </div>`;
    }

    document.getElementById('test-results-body').innerHTML = html || '<div class="empty-state">No test data available.</div>';
  }

  async function runTests() {
    const btn = document.getElementById('run-tests-btn');
    btn.textContent = '⟳ Running...';
    btn.disabled = true;
    try {
      const data = await API.runTests();
      render(data);
    } catch(e) {
      console.error(e);
    }
    btn.textContent = '▶ Run Tests';
    btn.disabled = false;
  }

  return { load, runTests };
})();
window.TestResultsView = TestResultsView;
