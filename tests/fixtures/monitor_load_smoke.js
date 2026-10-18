// Observed gpu-greenroom operator_server.py, cedecd7f, 2026-10-18.
// Exact loadSmoke body; generation semantics are the external contract in dispute.
async function loadSmoke(){
  const generation=++smokeLoadGeneration;
  try{
    const r=await request('/api/smoke-requests',{headers:auth()});
    if(!r.ok)throw new Error(await r.text());
    const payload=await r.json();
    if(generation!==smokeLoadGeneration)return;
    renderSmoke(payload);
  }catch(e){
    if(generation!==smokeLoadGeneration)return;
    if(smokeHasLoaded){keepLastSmokeView('Refresh failed; showing last loaded requests');err(e.message);return}
    document.querySelector('#smokeStatus').textContent='State unavailable';
    document.querySelector('#smokeRequestList').innerHTML='<div class="empty">Smoke requests unavailable.</div>';
    err(e.message);
  }
}
