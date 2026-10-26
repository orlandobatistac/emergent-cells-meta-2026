// Real HTTP integration of UI handlers with a minimal DOM/canvas stub.
// This checks controls and JS errors, not browser layout or visual rendering.
// Start the Python server first, then: node tests/browser_controls.cjs
const fs=require('node:fs'), vm=require('node:vm'), assert=require('node:assert/strict');
const base=process.env.EMERGENT_TEST_URL||'http://127.0.0.1:8765';
const ids=Object.fromEntries(['world','step','alive','energy','time','mode','play','advance','shock','damage','reset','report','status'].map(id=>[id,{textContent:'',value:'',disabled:false}]));
ids.world.getContext=()=>({clearRect(){},fillRect(){},beginPath(){},arc(){},fill(){}});
let downloaded=false;
const context=vm.createContext({document:{getElementById:id=>ids[id],querySelectorAll:()=>Object.values(ids),createElement:()=>({click(){downloaded=true}})},fetch:(url,args)=>fetch(new URL(url,base),args),setTimeout,clearTimeout,Blob,URL,console});
const source=fs.readFileSync('src/emergent_cells/demo/index.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function wait(test){for(let i=0;i<200;i++){if(test())return;await pause(10)}throw Error('Control timed out')}
(async()=>{vm.runInContext(source,context);await wait(()=>ids.alive.textContent!=='');await ids.reset.onclick();assert.equal(ids.step.textContent,0);await ids.advance.onclick();assert.equal(ids.step.textContent,20);const beforeDamage=Number(ids.alive.textContent.split('/')[0]);await ids.damage.onclick();assert.equal(ids.alive.textContent,`${beforeDamage-Math.floor(beforeDamage*.2)}/256`);await ids.shock.onclick();ids.mode.value='frozen';await ids.mode.onchange();assert.equal(ids.mode.value,'frozen');ids.play.onclick();await wait(()=>ids.step.textContent>20&&!ids.play.disabled);ids.play.onclick();await ids.report.onclick();assert.ok(downloaded);await ids.reset.onclick();assert.equal(ids.alive.textContent,'256/256');assert.equal(ids.step.textContent,0);console.log('Real HTTP UI controls: advance, damage, shock, mode, play/pause, report and reset passed. Layout not tested.');})().catch(e=>{console.error(e);process.exit(1)});
