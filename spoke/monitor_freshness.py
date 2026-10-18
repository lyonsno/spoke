"""Freshness receipt for the two retained, independently served monitor pages."""

FRESHNESS_SCRIPT = r"""(() => {
 const fleet = location.port === '8770' && location.pathname === '/fleet';
 const greenroom = location.port === '8766' && location.pathname === '/';
 if (!fleet && !greenroom) return;
 const sources = new Map((fleet ? ['Diauloi'] : ['Queue', 'Smokes']).map(name =>
   [name, {observed: null, received: null, error: 'awaiting observation', pending: 0, sequence: 0}]));
 let arrival = Date.now(), visible = true;
 const interval = window.setInterval.bind(window), fetch = window.fetch.bind(window);
 const stamp = value => typeof value === 'number' && Number.isFinite(value) ? value * 1000 :
   typeof value === 'string' ? Date.parse(value) : NaN;
 const age = time => Math.max(0, Math.floor((Date.now() - time) / 1000)) + 's ago';
 function render() {
   if (!document.body) return;
   let node = document.getElementById('spoke-monitor-freshness');
   if (!node) {
     node = document.createElement('div'); node.id = 'spoke-monitor-freshness';
     node.setAttribute('role', 'status');
     node.style.cssText = 'position:sticky;top:0;z-index:20;padding:8px 18px;min-height:34px;' +
       'box-sizing:border-box;background:#28231b;color:#f3cc7b;border-bottom:1px solid #66532e;' +
       'font:12px/18px system-ui;white-space:normal;overflow-wrap:anywhere';
     document.body.prepend(node);
   }
   const parts = [];
   for (const [name, state] of sources) {
     const receipt = state.observed === null ? '' : 'Last observation ' + age(state.observed) + '. ';
     let problem = state.error;
     if (!problem && state.observed < arrival) problem = 'awaiting observation since opening';
     if (!problem && Date.now() - state.received > 5000) problem = 'Stale: refresh overdue';
     if (state.pending) problem = 'refresh pending' + (problem ? '; ' + problem : '');
     parts.push(name + ': ' + (problem ? 'Unverified - ' + receipt + problem :
       'Observed ' + age(state.observed)));
   }
   node.textContent = parts.join(' | ');
   const header = document.querySelector('header');
   if (greenroom && header) header.style.top = node.offsetHeight + 'px';
 }
 function validate(name, value, requested) {
   const observed = Math.floor(stamp(value?.observed_at));
   if (!Number.isFinite(observed) || observed > Date.now()) throw new Error('invalid observation time');
   if (greenroom) {
     const source = JSON.parse(document.querySelector('meta[name="greenroom-smoke-source"]')?.content || 'null');
     if (source?.schema !== 'gpu-greenroom.smoke-source.v1' || !source.queue_dir ||
         value.queue_dir !== source.queue_dir || (name === 'Smokes' && value.reader_sha256 !== source.reader_sha256))
       throw new Error('queue source mismatch');
   }
   if (name === 'Diauloi') {
     if (value.schema !== 'epistaxis.diauloi-monitor.v1' || value.status !== 'complete' ||
         value.stale !== false || !Array.isArray(value.rows) ||
         value.discovery_authority !== 'complete-live-pane-enumeration' ||
         typeof value.repo_root !== 'string' || !value.repo_root.startsWith('/'))
       throw new Error(value.error || 'incomplete Diauloi observation');
   } else if (name === 'Queue') {
     if (value.schema !== 'gpu-greenroom.operator-snapshot.v1' || !Array.isArray(value.jobs) ||
         !Array.isArray(value.running_job_ids) || typeof value.queue_dir !== 'string' ||
         !value.queue_dir.startsWith('/') || value.view !== (requested.searchParams.get('view') || 'active'))
       throw new Error('invalid queue observation');
     if (['active', 'all', 'running'].includes(value.view) && value.running_job_ids.some(id =>
       !value.jobs.some(job => job.job_id === id))) throw new Error('running job missing from queue view');
   } else if (value.schema !== 'gpu-greenroom.smoke-request-list.v1' ||
              !Array.isArray(value.items) || !Array.isArray(value.errors)) {
     throw new Error('incomplete smoke observation');
   }
   return observed;
 }
 function watchRenderer(name) {
   const method = name === 'Smokes' ? 'renderSmoke' : 'render';
   const original = window[method];
   if (original?.__spokeFreshness) return;
   if (typeof original !== 'function') throw new Error('display receipt unavailable');
   const state = sources.get(name);
   const wrapped = function (...args) {
     try {
       const result = original.apply(this, args);
       const value = name === 'Diauloi' ? (typeof last === 'undefined' ? null : last) : args[0];
       const candidate = state.candidate;
       if (!candidate || value !== candidate.value) return result;
       if (name === 'Queue' && typeof draggedJob !== 'undefined' && draggedJob) {
         state.error = 'display update deferred during drag';
         return result;
       }
       if (candidate.sequence !== state.sequence) return result;
       if (name === 'Smokes' && typeof smokeRefreshUnavailable !== 'undefined' && smokeRefreshUnavailable)
         throw new Error('display retained previous observation');
       if (name === 'Queue' && typeof filter !== 'undefined' && value.view !== filter)
         throw new Error('display view no longer selected');
       state.observed = validate(name, value, candidate.url);
       state.received = Date.now(); state.candidate = null;
       state.error = name === 'Smokes' && value.errors.length ? 'incomplete smoke observation' : null;
       return result;
     } catch (error) {
       state.error = 'display update failed: ' + error.message;
       throw error;
     } finally { render(); }
   };
   wrapped.__spokeFreshness = true;
   window[method] = wrapped;
 }
 window.__spokeMonitorFreshness = next => {
   if (next && !visible) arrival = Date.now();
   visible = next; render();
 };
 window.fetch = async (input, options) => {
   const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url, location.href);
   const method = options?.method || input.method || 'GET';
   const name = url.origin !== location.origin || method.toUpperCase() !== 'GET' ? null :
     fleet && url.pathname === '/api/fleet' ? 'Diauloi' :
     greenroom && url.pathname === '/api/state' ? 'Queue' :
     greenroom && url.pathname === '/api/smoke-requests' ? 'Smokes' : null;
   if (!name) return fetch(input, options);
   const state = sources.get(name);
   const sequence = ++state.sequence;
   state.pending++; render();
   try {
     const response = await fetch(input, {...options, cache: 'no-store'});
     if (!response.ok) {
       state.pending--; state.error = 'HTTP ' + response.status; render();
       return response;
     }
     // Observe the same body the page consumes, not a successful navigation or headers alone.
     const json = response.json.bind(response);
     response.json = async () => {
       try {
         const value = await json(), observed = validate(name, value, url);
         if (state.observed !== null && observed < state.observed) throw new Error('older observation arrived late');
         watchRenderer(name);
         state.candidate = {value, url, sequence};
         state.error ||= 'display update pending';
         return value;
       } catch (error) {
         state.error = error.message; throw error;
       } finally { state.pending--; render(); }
     };
     return response;
   } catch (error) {
     state.pending--; state.error = error.message; render(); throw error;
   }
 };
 document.addEventListener('DOMContentLoaded', render);
 document.addEventListener('visibilitychange', () => { if (!document.hidden) render(); });
 interval(() => { if (visible && !document.hidden) render(); }, 1000);
 render();
})();"""
