// Capture an authored local page displaying actual trajectory artifacts. No model calls.
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import {pathToFileURL} from 'node:url';
import assert from 'node:assert/strict';
const args={};for(let i=2;i<process.argv.length;i+=2){if(!process.argv[i].startsWith('--')||!process.argv[i+1])throw Error('Expected --name value');args[process.argv[i].slice(2)]=process.argv[i+1];}
const {chromium}=args['playwright-module']?await import(pathToFileURL(path.resolve(args['playwright-module'])).href):await import('playwright');
const root=path.resolve(args['web-root']??'web'),dataFile=path.resolve(args.data),out=path.resolve(args.output),stepDuration=1/Number(args['steps-per-second']??4);
assert.ok(stepDuration>0&&stepDuration<=5);const data=JSON.parse(await fs.readFile(dataFile,'utf8'));
const allowed=new Map([['/','comparison.html'],['/comparison.html','comparison.html'],['/comparison.js','comparison.js'],['/comparison.css','comparison.css']]);
const server=http.createServer(async(req,res)=>{try{const route=new URL(req.url,'http://127.0.0.1').pathname;let file;if(route==='/comparison_results.json')file=dataFile;else if(allowed.has(route))file=path.join(root,allowed.get(route));else{res.writeHead(404);res.end();return;}const bytes=await fs.readFile(file);res.writeHead(200,{'Content-Type':file.endsWith('.js')?'text/javascript':file.endsWith('.css')?'text/css':file.endsWith('.json')?'application/json':'text/html'});res.end(bytes);}catch{res.writeHead(500);res.end();}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({executablePath:args.chrome,headless:true,args:['--disable-background-networking','--no-first-run','--no-default-browser-check']});
const report={browser_version:browser.version(),page_errors:[],policies:{},network_model_calls:0};
try{const page=await browser.newPage({viewport:{width:1600,height:1080},deviceScaleFactor:1});page.on('pageerror',e=>report.page_errors.push(e.message));await page.goto(`http://127.0.0.1:${server.address().port}`,{waitUntil:'networkidle'});await page.waitForFunction(()=>window.nanojevComparison?.ready||window.nanojevComparison?.error);
 assert.equal(await page.evaluate(()=>window.nanojevComparison.error),undefined);await page.evaluate(()=>document.fonts.ready);
 assert.deepEqual(await page.locator('.system-name').allTextContents(),['NanoJev','Jev','Untuned Qwen']);
 const stage=page.locator('#comparisonStage');assert.deepEqual(await stage.evaluate(e=>({width:e.offsetWidth,height:e.offsetHeight})),{width:1600,height:1000});
 for(const policy of data.policies){const folder=path.join(out,policy);await fs.mkdir(folder,{recursive:true});const frames=[];let sequence=0;
  for(let ci=0;ci<data.cases.length;ci++){const max=await page.evaluate(([p,c])=>window.nanojevComparison.duration(p,c),[policy,ci]);
   for(let si=0;si<=max;si++){const snapshot=await page.evaluate(([p,c,s])=>window.nanojevComparison.setFrame(p,c,s),[policy,ci,si]);
    assert.equal(snapshot.step_index,si);for(const panel of snapshot.panels){const model=data.models.find(m=>m.policy===policy&&m.role===panel.role),ep=model.episodes.find(e=>e.id===data.cases[ci].id),idx=Math.min(si,ep.steps.length),expected=ep.steps[idx]?.state??ep.final_state??ep.steps.at(-1)?.next_state??ep.initial_state;assert.deepEqual(panel.environment,expected);assert.equal(panel.action,ep.steps[idx]?.action??null);assert.equal(panel.finished,idx===ep.steps.length);}
    if(si===max){assert.ok(snapshot.panels.every(p=>p.finished),'all final states must be frozen at final frame');assert.deepEqual(await page.locator('.status').allTextContents(),['Goal reached','Goal reached','Step limit reached']);assert.deepEqual(await page.locator('.final-message').allTextContents(),['✓ Goal reached','✓ Goal reached','× Step limit reached']);}
    const name=`frame_${String(sequence++).padStart(5,'0')}.png`;await stage.screenshot({path:path.join(folder,name)});frames.push({file:name,case_index:ci,case_id:data.cases[ci].id,environment_step:si,duration_seconds:si===0?1.25:si===max?1.8:stepDuration,all_finished:si===max});
   }
  }
  report.policies[policy]={frame_count:frames.length,duration_seconds:frames.reduce((n,f)=>n+f.duration_seconds,0),frames};console.log(JSON.stringify({policy,frames:frames.length,all_case_final_frames_verified:true}));
 }
 assert.deepEqual(report.page_errors,[]);report.passed=true;await fs.writeFile(path.join(out,'capture_check.json'),JSON.stringify(report,null,2)+'\n');
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
