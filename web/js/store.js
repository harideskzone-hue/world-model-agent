/**
 * store.js — in-memory state for the viewer
 */
const Store = (() => {
  let state = {
    currentSession: null,
    currentTurnId: 0,
    allTurns: [],
    polling: null,
    graphTurn: 0,
  };

  const listeners = [];

  function get() { return state; }
  function set(patch) {
    state = { ...state, ...patch };
    listeners.forEach(fn => fn(state));
  }
  function subscribe(fn) { listeners.push(fn); }

  function startPolling(sid, onNewTurns) {
    if (state.polling) clearInterval(state.polling);
    let since = state.allTurns.length - 1;
    const id = setInterval(async () => {
      const turns = await API.getTurns(sid, since);
      if (turns && turns.length > 0) {
        since = turns[turns.length - 1].turn_id;
        set({ allTurns: [...state.allTurns, ...turns] });
        onNewTurns(turns);
      }
    }, 1500);
    set({ polling: id });
  }

  function stopPolling() {
    if (state.polling) clearInterval(state.polling);
    set({ polling: null });
  }

  return { get, set, subscribe, startPolling, stopPolling };
})();

window.Store = Store;
