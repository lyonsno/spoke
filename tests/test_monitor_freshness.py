import json
import subprocess
from pathlib import Path

import pytest


HARNESS = r"""
const vm=require('node:vm'),input=JSON.parse(process.argv[1]);
let now=1800000000000;class Clock extends Date{static now(){return now}}
const listeners={},timers=[],nodes={},requests=[];
const body={prepend(n){nodes[n.id]=n}};
const document={body,hidden:false,readyState:'complete',
 createElement(){return {style:{},setAttribute(){}}},getElementById(id){return nodes[id]},querySelector(selector){return selector.startsWith('meta')?{content:JSON.stringify({schema:'gpu-greenroom.smoke-source.v1',queue_dir:'/real/queue',reader_sha256:'reader'})}:null},
 addEventListener(n,f){(listeners[n]??=[]).push(f)}};
const window={setInterval(f){timers.push(f);return timers.length},fetch(url,options){
 return new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}))}};
window.render=()=>{};window.renderSmoke=()=>{};
const location={href:'http://127.0.0.1:'+input.port+(input.port==='8770'?'/fleet':'/'),pathname:input.port==='8770'?'/fleet':'/',port:input.port,origin:'http://127.0.0.1:'+input.port};
const context=vm.createContext({window,document,location,Date:Clock,URL,Response,Request,console});
vm.runInContext(input.source,context);
const status=()=>nodes['spoke-monitor-freshness']?.textContent||'';
const open=()=>window.__spokeMonitorFreshness?.(true);
const close=()=>window.__spokeMonitorFreshness?.(false);
async function fetchData(path,payload,code=200){const p=window.fetch(path);requests.at(-1).resolve(new Response(JSON.stringify(payload),{status:code}));try{const value=await(await p).json();if(path.includes('/api/fleet')){context.last=value;window.render()}else if(path.includes('smoke-requests'))window.renderSmoke(value);else window.render(value);return value}catch(e){return {error:e.message}}}
const queue=(observed=now/1000)=>({schema:'gpu-greenroom.operator-snapshot.v1',observed_at:observed,queue_dir:'/real/queue',view:'active',jobs:[],running_job_ids:[]});
const fleet=(observed=new Clock(now).toISOString())=>({schema:'epistaxis.diauloi-monitor.v1',status:'complete',stale:false,observed_at:observed,rows:[],repo_root:'/real/epistaxis',discovery_authority:'complete-live-pane-enumeration'});
const smoke=()=>({schema:'gpu-greenroom.smoke-request-list.v1',items:[],errors:[],observed_at:now/1000,queue_dir:'/real/queue',reader_sha256:'reader'});
const result={initial:status()};
(async()=>{open();result.open=status();
 if(input.case==='pending'){await fetchData('/api/state?view=active',queue());await fetchData('/api/smoke-requests',smoke());close();now+=60000;open();result.reopen=status();const p=window.fetch('/api/state?view=active');result.pending=status();result.cache=requests.at(-1).options?.cache;requests.at(-1).reject(new Error('offline'));try{await p}catch{}result.failed=status();}
 if(input.case==='old'){await fetchData('/api/fleet',fleet(new Clock(now-30000).toISOString()));result.old=status();now+=1000;await fetchData('/api/fleet',fleet());result.current=status();close();now+=70000;open();result.reopen=status();}
 if(input.case==='invalid'){await fetchData('/api/state',input.payload);await fetchData('/api/smoke-requests',smoke());result.invalid=status();}
 if(input.case==='partial'){await fetchData('/api/state',queue());await fetchData('/api/smoke-requests',{...smoke(),errors:['unreadable record']});result.partial=status();}
 if(input.case==='age'){await fetchData('/api/state',queue());await fetchData('/api/smoke-requests',smoke());result.current=status();now+=10000;for(const f of timers)f();result.aged=status();}
 if(input.case==='order'){const older=window.fetch('/api/state');const first=requests.at(-1);now+=1000;await fetchData('/api/state',queue());await fetchData('/api/smoke-requests',smoke());result.newer=status();first.resolve(new Response(JSON.stringify(queue((now-2000)/1000))));try{await(await older).json()}catch(e){result.rejected=e.message}result.regressed=status();}
 if(input.case==='http'){await fetchData('/api/state',queue(),503);await fetchData('/api/smoke-requests',smoke());result.failed=status();}
 if(input.case==='auth'){const p=window.fetch('/api/state');requests.at(-1).resolve(new Response('{}',{status:401}));try{result.code=(await p).status}catch(e){result.error=e.message}result.failed=status();}
 if(input.case==='wrong-source'){await fetchData('/api/state',{...queue(),queue_dir:'/other/queue'});await fetchData('/api/smoke-requests',smoke());result.failed=status();}
 if(input.case==='render-failure'){window.render=()=>{throw Error('renderer failed')};await fetchData('/api/state',queue());await fetchData('/api/smoke-requests',smoke());result.failed=status();}
 if(input.case==='precision'){await fetchData('/api/state',queue(now/1000+0.0005));await fetchData('/api/smoke-requests',smoke());result.sameMs=status();now+=1000;await fetchData('/api/state',queue(now/1000+1));result.future=status();}
 if(input.case==='drag'||input.case==='drag-failure'){
   context.draggedJob={job_id:'held'};
   window.render=value=>{context.last=value;if(context.draggedJob)return;context.displayed=value};
   await fetchData('/api/state',queue());await fetchData('/api/smoke-requests',smoke());result.deferred=status();
   if(input.case==='drag-failure'){const p=window.fetch('/api/state');requests.at(-1).resolve(new Response('{}',{status:503}));await p;}
   context.draggedJob=null;window.render(context.last);result.finished=status();
 }
 if(input.case==='discarded'){
   vm.runInContext(`let lastSmoke={items:[{id:'retained-old'}]},smokeLoadGeneration=0,smokeHasLoaded=true,smokeRefreshUnavailable=false;
   const request=(...args)=>window.fetch(...args),auth=()=>({});let shown='retained-old',pageMessage='';
   function renderSmoke(payload){return window.renderSmoke(payload)}
   function keepLastSmokeView(message){smokeRefreshUnavailable=true;pageMessage=message}function err(){}
   ${input.loadSmoke}`,context);
   window.renderSmoke=payload=>{context.__accepted=payload;vm.runInContext("lastSmoke=__accepted;shown='accepted';smokeRefreshUnavailable=false",context)};
   const a=vm.runInContext('loadSmoke()',context),ra=requests.at(-1),b=vm.runInContext('loadSmoke()',context),rb=requests.at(-1);
   rb.resolve(new Response('{}',{status:503}));await b;now+=1000;ra.resolve(new Response(JSON.stringify(smoke())));await a;
   result.discarded=status();result.shown=vm.runInContext('shown',context);
   const c=vm.runInContext('loadSmoke()',context);requests.at(-1).resolve(new Response(JSON.stringify(smoke())));await c;result.accepted=status();
 }
 console.log(JSON.stringify(result));})();
"""


def run(case, port="8766", **values):
    from spoke import monitor_window
    result = subprocess.run(
        ["node", "-e", HARNESS, json.dumps({"source": getattr(monitor_window, "_FRESHNESS", ""),
                                          "loadSmoke": Path(__file__).with_name("fixtures").joinpath("monitor_load_smoke.js").read_text(),
                                          "port": port, "case": case, **values})],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_reopen_invalidates_old_empty_view_before_response_and_failure_stays_unverified():
    result = run("pending")
    assert "Unverified" in result["open"]
    assert "Unverified" in result["reopen"]
    assert "Unverified" in result["pending"]
    assert "offline" in result["failed"]
    assert result["cache"] == "no-store"


def test_successful_fetch_of_cached_fleet_observation_is_not_a_new_observation():
    result = run("old", port="8770")
    assert "Unverified" in result["old"]
    assert "Observed" in result["current"] and "Unverified" not in result["current"]
    assert "Unverified" in result["reopen"]


@pytest.mark.parametrize("payload", [
    {}, {"schema": "wrong", "observed_at": 1800000000, "jobs": []},
    {"schema": "gpu-greenroom.operator-snapshot.v1", "jobs": [], "observed_at": None},
    {"schema": "gpu-greenroom.operator-snapshot.v1", "jobs": [], "observed_at": 1800000000,
     "queue_dir": "/real/queue", "view": "active", "running_job_ids": ["missing-running-job"]},
])
def test_missing_or_conflicting_source_cannot_establish_empty_queue(payload):
    assert "Unverified" in run("invalid", payload=payload)["invalid"]


def test_partial_smoke_read_is_not_verified_empty():
    assert "Unverified" in run("partial")["partial"]


def test_source_age_advances_without_network_completion():
    result = run("age")
    assert "Observed" in result["current"]
    assert "Stale" in result["aged"]


def test_late_older_response_marks_display_unverified():
    result = run("order")
    assert "Last observation 0s ago" in result["newer"]
    assert "Unverified" in result["regressed"]
    assert "older observation" in result["rejected"]


def test_http_failure_cannot_be_cleared_by_other_source_success():
    assert "Unverified" in run("http")["failed"]


def test_auth_response_is_preserved_for_page_owned_reconnect():
    result = run("auth")
    assert result["code"] == 401
    assert "HTTP 401" in result["failed"]


def test_wrong_queue_cannot_be_verified_by_current_timestamps():
    assert "queue source mismatch" in run("wrong-source")["failed"]


def test_renderer_failure_cannot_certify_unchanged_view():
    assert "Unverified" in run("render-failure")["failed"]


def test_discarded_old_request_cannot_clear_newer_failure():
    result = run("discarded")
    assert "Unverified" in result["discarded"]
    assert "HTTP 503" in result["discarded"]
    assert result["shown"] == "retained-old"
    assert "Smokes: Observed" in result["accepted"]
    assert "HTTP 503" not in result["accepted"]


def test_same_millisecond_precision_is_valid_but_future_second_is_not():
    result = run("precision")
    assert "Unverified" not in result["sameMs"]
    assert "invalid observation time" in result["future"]


def test_drag_deferral_waits_for_actual_repaint_without_throwing():
    result = run("drag")
    assert "Queue: Unverified" in result["deferred"]
    assert "deferred" in result["deferred"]
    assert "Queue: Observed" in result["finished"]


def test_old_drag_candidate_cannot_clear_newer_refresh_failure():
    assert "HTTP 503" in run("drag-failure")["finished"]
