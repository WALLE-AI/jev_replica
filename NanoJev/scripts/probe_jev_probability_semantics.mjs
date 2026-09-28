import fs from 'node:fs/promises';
import { evaluateTeacher } from './teachers.mjs';

// A fixed, small follow-up to the observed Choice probability discrepancy.
const cases=[];
for (const p of [0.5,0.7]) {
  for (const wording of ['plain','explicit_distribution']) {
    for (const repeat of [0,1]) {
      const state = p===0.5
        ? 'A fair coin is tossed once. Heads and tails each have probability 50%. The outcome is hidden, and there is no information about which outcome occurred.'
        : 'A biased coin is tossed once. Its heads probability is 70% and its tails probability is 30%. The outcome is hidden, and there is no information about which outcome occurred.';
      const instruction = wording==='plain' ? 'What was the outcome of this hidden coin toss?'
        : 'Estimate the conditional probability of each hidden outcome from the given facts. Preserve the stated randomness; the outcome was not observed.';
      const questions = {
        outcome:{type:'choice', instructions:instruction,criteria:{heads:'The coin landed heads.',tails:'The coin landed tails.'}},
        heads:{type:'boolean',instructions:'Did this hidden coin toss land heads? Evaluate its probability from the given facts; the result has not been revealed.'},
        prior:{type:'choice',instructions:'What probability of heads is explicitly specified in the state?',criteria:{zero:'0%',half:'50%',seventy:'70%',certain:'100%'}},
      };
      cases.push({id:`p${p}_${wording}_r${repeat}`,p,wording,repeat,state,questions});
    }
  }
}
if (!process.argv.includes('--live')) {
  console.log(JSON.stringify({requests:cases.length,questions:cases.length*3,live:false}));
} else {
  const file='research/private_probability_semantics.jsonl';
  // Fixed fresh run, never silently repeat a partially completed paid experiment.
  const handle=await fs.open(file,'wx');
  let cost=0;
  try {
    for (const row of cases) {
      if (cost+0.02>0.25) throw new Error('BUDGET_GUARD');
      const teacher=await evaluateTeacher({teacher:'jev',model:'typesafe-ai/jev',state:row.state,questions:row.questions});
      const actual=Number(teacher.provider_metadata?.gateway?.cost);
      if (!Number.isFinite(actual)) throw new Error('MISSING_COST');
      cost+=actual;
      await handle.write(JSON.stringify({...row,teacher,cost_usd:actual})+'\n');
    }
  } finally { await handle.close(); }
  const rows=(await fs.readFile(file,'utf8')).trim().split('\n').map(JSON.parse);
  const summary={purpose:'posthoc_typed_probability_semantics_diagnostic_no_training',requests:rows.length,cost_usd:cost,
    rows:rows.map(r=>({id:r.id,reference_p_heads:r.p,wording:r.wording,repeat:r.repeat,
      choice:r.teacher.native_probs.outcome,boolean_p_heads:r.teacher.native_probs.heads.true,
      specified_prior:r.teacher.answers.prior.choice,prior_distribution:r.teacher.native_probs.prior})),
    limitations:['Small authored diagnostic; no measured real-world calibration.','No temperature selection or student update.','Two identical-request repeats per wording; finite evidence about repeatability.']};
  await fs.writeFile('research/probability_semantics_summary.json',JSON.stringify(summary,null,2)+'\n');
  console.log(JSON.stringify(summary));
}
