// Budgeted Jev collection with bounded request retries and conservative failure reservations.
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import {createHash} from 'node:crypto';
import {evaluateTeacher} from './teachers.mjs';

const args=Object.fromEntries(process.argv.slice(2).reduce((a,v,i,all)=>{
  if(v.startsWith('--')) a.push([v.slice(2),all[i+1]]);return a;
},[]));
const dir=args['journal-dir'];
const budget=Number(args['budget-usd']??2),concurrency=Number(args.concurrency??8);
if(!dir||!(budget>0&&budget<=24)||!Number.isInteger(concurrency)||concurrency<1||concurrency>8) throw Error('Invalid worker limits');
await fs.mkdir(dir,{recursive:true});
const journalFile=path.join(dir,'calls.jsonl');
const digest=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
let prior=[];
try{prior=(await fs.readFile(journalFile,'utf8')).split('\n').filter(Boolean).map(JSON.parse);}
catch(e){if(e.code!=='ENOENT')throw e;}
const cache=new Map(),unresolved=new Set();
for(const row of prior){
  if(row.status==='started')unresolved.add(row.id);
  else unresolved.delete(row.id);
  if(row.status==='succeeded')cache.set(row.input_sha256,row);
}
let spent=prior.filter(r=>r.status==='succeeded').reduce((s,r)=>s+r.cost_usd,0);
// Unknown request costs remain reserved permanently; they never become free retries.
const starts=new Map(prior.filter(r=>r.status==='started').map(r=>[r.id,r]));
const unknownIds=new Set([...unresolved,...prior.filter(r=>r.status==='failed'&&r.unknown_cost).map(r=>r.id)]);
let unknownReserved=[...unknownIds].reduce((s,id)=>s+(starts.get(id)?.reservation_usd??.002),0);
let sequence=prior.filter(r=>r.status==='started').length,reserved=0;
if(!Number.isFinite(spent))throw Error('Invalid recorded cost');
let writeQueue=Promise.resolve();
const append=row=>{writeQueue=writeQueue.then(()=>fs.appendFile(journalFile,JSON.stringify(row)+'\n'));return writeQueue;};
function converted(row,publicState,cacheHit){
  const answers={};
  for(const [qid,q] of Object.entries(publicState.questions)){
    const native=row.native_probs[qid];
    const expected=q.type==='choice'?Object.keys(q.criteria):q.type==='boolean'?['false','true']:q.criteria.map((_,i)=>String(i));
    if(!native||Object.keys(native).length!==expected.length||!expected.every(k=>Object.hasOwn(native,k)))throw Error('Teacher candidate mismatch');
    if(Object.values(native).some(p=>typeof p!=='number'||!Number.isFinite(p)||p<0||p>1))throw Error('Invalid teacher probabilities');
    const sum=Object.values(native).reduce((a,b)=>a+b,0);
    if(!(sum>0))throw Error('Zero rounded teacher mass: cannot construct a controller');
    const probabilities=Object.fromEntries(expected.map(k=>[k,native[k]/sum]));
    const choice=[...expected].sort().reduce((best,k)=>probabilities[k]>probabilities[best]?k:best,[...expected].sort()[0]);
    const answer={type:q.type,probabilities,native_probabilities:native,native_sum:sum,
      normalization_applied:Math.abs(sum-1)>1e-12,target_kind:'normalized_rounded_controller_proxy',
      source_api_call_id:row.id,source_input_sha256:row.input_sha256,cache_hit:cacheHit};
    if(q.type==='choice')answer.choice=choice;
    if(q.type==='boolean')answer.p_true=probabilities.true;
    if(q.type==='score'){answer.level=Number(choice);answer.score=expected.reduce((s,k)=>s+Number(k)*probabilities[k],0);}
    answers[qid]=answer;
  }
  return {id:publicState.id,answers};
}
async function evaluate(publicState){
  const input={model:'typesafe-ai/jev',state:publicState.state,questions:publicState.questions};
  const key=digest(input);
  if(cache.has(key))return {result:converted(cache.get(key),publicState,true),apiCalls:0};
  const reservation=Math.max(.002,(Buffer.byteLength(JSON.stringify(input))+8192)*.042/1e6);
  if(spent+unknownReserved+reserved+reservation>budget)throw Error('Application API budget reached');
  reserved+=reservation;
  const id=`nanojev-live-${String(++sequence).padStart(6,'0')}`;
  const started_at=new Date().toISOString();
  await append({id,status:'started',input_sha256:key,started_at,reservation_usd:reservation});
  try{
    const teacher=await evaluateTeacher({teacher:'jev',model:input.model,state:input.state,questions:input.questions});
    const cost=Number(teacher.provider_metadata?.gateway?.cost);
    if(!Number.isFinite(cost)||cost<0)throw Error('Missing cost');
    spent+=cost;
    const record={id,status:'succeeded',input_sha256:key,input,model:teacher.model,
      native_probs:teacher.native_probs,rounding:teacher.rounding,cost_usd:cost,started_at,
      finished_at:new Date().toISOString(),response_sha256:digest({native_probs:teacher.native_probs,rounding:teacher.rounding})};
    await append(record);cache.set(key,record);
    return {result:converted(record,publicState,false),apiCalls:1};
  }catch(error){
    unknownReserved+=reservation;
    await append({id,status:'failed',input_sha256:key,error_code:String(error.code??'MEASUREMENT_FAILED'),unknown_cost:true,finished_at:new Date().toISOString()});
    const failure=Error('Jev measurement failed; no fallback');
    failure.retryable=['TEACHER_REQUEST_FAILED','TEACHER_ABORTED'].includes(error.code);
    throw failure;
  }finally{reserved-=reservation;}
}
async function boundedEvaluate(publicState){
  for(let attempt=0;attempt<3;attempt++){
    try{return await evaluate(publicState);}
    catch(error){
      if(!error.retryable||attempt===2)throw error;
      await new Promise(resolve=>setTimeout(resolve,500*(attempt+1)));
    }
  }
}
for await(const line of readline.createInterface({input:process.stdin,crlfDelay:Infinity})){
  if(!line.trim())continue;
  try{
    const request=JSON.parse(line);
    if(!Array.isArray(request.states)||!request.states.length)throw Error('Empty request');
    const results=new Array(request.states.length);let next=0,failed=null;
    await Promise.all(Array.from({length:Math.min(concurrency,results.length)},async()=>{
      while(next<results.length&&!failed){const index=next++;try{results[index]=await boundedEvaluate(request.states[index]);}catch(e){failed=e;}}
    }));
    if(failed)throw failed;
    const calls=results.reduce((s,r)=>s+r.apiCalls,0);
    process.stdout.write(JSON.stringify({states:results.map(r=>r.result),temperature:{value:1},
      execution:{engine:'jev_live_api',forward_passes:null,autoregressive_decode_steps:null,
        network_model_calls:calls,cache_hits:results.length-calls,states:results.length,
        questions:request.states.reduce((s,r)=>s+Object.keys(r.questions).length,0),
        max_concurrent_http_requests:concurrency,cumulative_cost_usd:spent,unknown_cost_reserved_usd:unknownReserved,api_model:'typesafe-ai/jev'}})+'\n');
  }catch{
    process.stdout.write(JSON.stringify({error:'Jev worker stopped after bounded attempts; inspect sanitized local journal, no fallback.'})+'\n');
    process.exitCode=1;break;
  }
}
await writeQueue;
