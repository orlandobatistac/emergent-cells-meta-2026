// Real HTTP integration of UI handlers with a minimal DOM/canvas stub.
// This checks controls and JS errors, not browser layout or visual rendering.
// Start the Python server first, then: node tests/browser_controls.cjs
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const base=process.env.EMERGENT_TEST_URL||'http://127.0.0.1:8765';
const ids=Object.fromEntries(['world','step','alive','energy','time','mode','play','advance','shock','damage','reset','report','status','challenge-text','challenge-source','challenge-design','challenge-evaluate','challenge-apply','challenge-state','challenge-result','cooperation-run','cooperation-state','cooperation-result','protocol-run','protocol-state','protocol-result'].map(id=>[id,{textContent:'',value:'',disabled:false}]));
ids['challenge-text'].value='Crea un mundo peligroso con recursos escasos y cambios';ids['challenge-source'].value='template';
ids.world.getContext=()=>({clearRect(){},fillRect(){},beginPath(){},arc(){},fill(){}});
let downloaded=false;
const context=vm.createContext({document:{getElementById:id=>ids[id],querySelectorAll:()=>Object.values(ids),createElement:()=>({click(){downloaded=true}})},fetch:(url,args)=>fetch(new URL(url,base),args),setTimeout,clearTimeout,Blob,URL,console});
const source=fs.readFileSync('src/emergent_cells/demo/index.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function wait(test){for(let i=0;i<200;i++){if(test())return;await pause(10)}throw Error('Control timed out')}
(async()=>{vm.runInContext(source,context);await wait(()=>ids.alive.textContent!=='');await ids.reset.onclick();assert.equal(ids.step.textContent,0);await ids.advance.onclick();assert.equal(ids.step.textContent,20);const beforeDamage=Number(ids.alive.textContent.split('/')[0]);await ids.damage.onclick();assert.equal(ids.alive.textContent,`${beforeDamage-Math.floor(beforeDamage*.2)}/256`);await ids.shock.onclick();ids.mode.value='frozen';await ids.mode.onchange();assert.equal(ids.mode.value,'frozen');ids.play.onclick();await wait(()=>ids.step.textContent>20&&!ids.play.disabled);ids.play.onclick();await ids.report.onclick();assert.ok(downloaded);await ids.reset.onclick();assert.equal(ids.alive.textContent,'256/256');assert.equal(ids.step.textContent,0);
await ids['challenge-design'].onclick();assert.match(ids['challenge-state'].textContent,/Plantilla LOCAL/);
await ids['challenge-evaluate'].onclick();assert.match(ids['challenge-result'].textContent,/greedy/);
await ids['challenge-apply'].onclick();assert.equal(ids.step.textContent,0);
assert.match(ids['challenge-state'].textContent,/Mundo aplicado/);
await ids['cooperation-run'].onclick();
assert.match(ids['cooperation-result'].textContent,/local:/);
assert.match(ids['cooperation-result'].textContent,/Mensajes bloqueados/);
assert.match(ids['cooperation-state'].textContent,/Comparación verificada/);
await ids['protocol-run'].onclick();
assert.match(ids['protocol-result'].textContent,/learned_messages/);
assert.match(ids['protocol-result'].textContent,/Bloqueo del mismo modelo/);
assert.match(ids['protocol-state'].textContent,/Prueba exploratoria completada/);
console.log('Real HTTP UI controls + template world design, 5-policy evaluation and apply: passed. Layout not tested.');})().catch(e=>{console.error(e);process.exit(1)});
