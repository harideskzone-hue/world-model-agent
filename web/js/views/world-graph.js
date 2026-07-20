/**
 * world-graph.js — Cytoscape.js belief graph renderer
 */
const WorldGraphView = (() => {
  let _cy = null;
  let _sid = null;
  let _maxTurn = 0;

  const NODE_COLORS = {
    room:      { bg: '#1e3a5f', border: '#5b8dee', text: '#a8d0ff' },
    object:    { bg: '#1e3d2a', border: '#3dca7e', text: '#8ef5b8' },
    character: { bg: '#3d1e3a', border: '#c471d4', text: '#e8a0f0' },
  };

  function init(sid, maxTurn) {
    _sid = sid;
    _maxTurn = maxTurn;
    const slider = document.getElementById('graph-turn-slider');
    if (slider) {
      slider.max = maxTurn;
      slider.value = maxTurn;
      slider.oninput = () => {
        document.getElementById('graph-turn-val').textContent = slider.value;
        render(parseInt(slider.value));
      };
    }
    render(maxTurn);
  }

  async function render(atTurn) {
    const container = document.getElementById('cy');
    if (!container) return;

    const data = await API.getGraph(_sid, atTurn);
    if (!data) return;

    const { nodes, edges } = data;

    const cyNodes = nodes.map(n => ({
      data: {
        id: n.id, label: n.label || n.id,
        node_type: n.node_type || 'object',
        confidence: n.confidence || 0.5
      }
    }));

    const cyEdges = edges.map(e => ({
      data: {
        id: e.id, source: e.subject, target: e.object,
        label: e.relation, confidence: e.confidence || 0.5,
        status: e.status || 'active',
        method: e.extraction_method || 'rule_fallback'
      }
    }));

    if (_cy) { _cy.destroy(); _cy = null; }

    _cy = cytoscape({
      container,
      elements: { nodes: cyNodes, edges: cyEdges },
      style: [
        {
          selector: 'node',
          style: {
            'background-color': ele => {
              const t = ele.data('node_type');
              return (NODE_COLORS[t] || NODE_COLORS.object).bg;
            },
            'border-color': ele => {
              const t = ele.data('node_type');
              return (NODE_COLORS[t] || NODE_COLORS.object).border;
            },
            'border-width': 2,
            'color': ele => {
              const t = ele.data('node_type');
              return (NODE_COLORS[t] || NODE_COLORS.object).text;
            },
            'label': 'data(label)',
            'font-size': '11px',
            'font-family': 'JetBrains Mono, monospace',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': ele => {
              const len = (ele.data('label') || '').length;
              return Math.max(60, len * 7.5) + 'px';
            },
            'height': '36px',
            'shape': ele => {
              const t = ele.data('node_type');
              return t === 'room' ? 'round-rectangle' : t === 'character' ? 'ellipse' : 'rectangle';
            },
            'text-wrap': 'wrap',
            'text-max-width': '100px',
            'padding': '8px',
          }
        },
        {
          selector: 'edge',
          style: {
            'line-color': ele => ele.data('status') === 'superseded' ? '#5a4000' : '#3a5080',
            'target-arrow-color': ele => ele.data('status') === 'superseded' ? '#5a4000' : '#3a5080',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '9px',
            'font-family': 'JetBrains Mono, monospace',
            'color': '#6b7890',
            'text-background-color': '#0d0f14',
            'text-background-opacity': 1,
            'text-background-padding': '2px',
            'width': ele => Math.max(1, (ele.data('confidence') || 0.5) * 3),
            'opacity': ele => ele.data('status') === 'superseded' ? 0.3 : 0.9,
            'line-style': ele => ele.data('status') === 'superseded' ? 'dashed' : 'solid',
          }
        },
        {
          selector: 'node:selected',
          style: { 'border-width': 3, 'border-color': '#ffffff' }
        }
      ],
      layout: {
        name: 'cose',
        animate: false,
        nodeRepulsion: 8000,
        idealEdgeLength: 80,
        gravity: 0.4,
        padding: 20,
      }
    });

    // Show node info on click
    _cy.on('tap', 'node', evt => {
      const n = evt.target.data();
      document.getElementById('graph-info').innerHTML = `
        <strong>${n.id}</strong> · type: <em>${n.node_type}</em> · conf: ${(n.confidence*100).toFixed(0)}%`;
    });
    _cy.on('tap', 'edge', evt => {
      const e = evt.target.data();
      document.getElementById('graph-info').innerHTML = `
        <strong>${e.source}</strong> →<em>${e.label}</em>→ <strong>${e.target}</strong>
        · conf: ${(e.confidence*100).toFixed(0)}% · ${e.status}`;
    });

    // Legend
    const legend = document.getElementById('graph-legend');
    if (legend) {
      legend.innerHTML = `
        <span style="color:#5b8dee">■ room</span>
        <span style="color:#3dca7e">■ object</span>
        <span style="color:#c471d4">■ character</span>
        <span style="color:#3a5080">— active</span>
        <span style="color:#5a4000;opacity:0.5">- - superseded</span>
        <strong style="margin-left:auto">${nodes.length} nodes · ${edges.length} edges</strong>`;
    }
  }

  return { init, render };
})();
window.WorldGraphView = WorldGraphView;
