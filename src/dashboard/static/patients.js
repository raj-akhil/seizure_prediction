// Patient selections are written directly into the same configuration used by jobs.
let patientCatalog=null,patientKey=null,patientRequest=0,patientTimer=null,splitDraft=null;
function patientSettingsKey(){return JSON.stringify([config.dataset,config.sensors,config.alignment,config.preprocessing,config.labels,config.training.inner_folds,config.training.min_positive_events_per_partition,config.training.seed]);}
function includedPatient(id){return config.dataset.subjects===null||config.dataset.subjects.includes(id);}
function patientRoles(id){return ['train','validation','test'].filter(role=>config.training.splits[role].includes(id));}
function updatePatientConfig(){splitDraft=null;$('json').value=JSON.stringify(config,null,2);dirtyJson=false;summary();renderPatients();schedulePatients();}
function assignPatient(id,role){for(const name of ['train','validation','test'])config.training.splits[name]=config.training.splits[name].filter(value=>value!==id);if(role)config.training.splits[role].push(id);updatePatientConfig();}
function choosePatient(id,include){const ids=config.dataset.subjects===null?patientCatalog.patients.map(row=>row.id):config.dataset.subjects;config.dataset.subjects=include?[...new Set([...ids,id])]:ids.filter(value=>value!==id);if(!include)for(const role of ['train','validation','test'])config.training.splits[role]=config.training.splits[role].filter(value=>value!==id);updatePatientConfig();}
function patientNumber(id){const match=String(id).match(/(\d+)(?!.*\d)/);return match?Number(match[1]):null;}
function parseRanges(value,label){
  const numbers=new Set();
  for(const token of value.split(',').map(item=>item.trim()).filter(Boolean)){
    const match=token.match(/^(\d+)(?:\s*-\s*(\d+))?$/);
    if(!match)throw new Error(`${label} must use patient numbers or inclusive ranges, for example 1-50, 60.`);
    const start=Number(match[1]),end=Number(match[2]||match[1]);
    if(start<1||end<start||end-start>10000)throw new Error(`${label} contains an invalid range: ${token}.`);
    for(let number=start;number<=end;number++)numbers.add(number);
  }
  return numbers;
}
function idsForRanges(numbers,byNumber,label){
  const ids=[];const missing=[];
  for(const number of numbers){const matches=byNumber.get(number)||[];if(matches.length===1)ids.push(matches[0]);else missing.push(number);}
  if(missing.length)throw new Error(`${label} contains patient number${missing.length===1?'':'s'} not listed in the current folder: ${missing.slice(0,12).join(', ')}${missing.length>12?'…':''}.`);
  return ids.sort((left,right)=>patientNumber(left)-patientNumber(right)||left.localeCompare(right));
}
function applyRangeAssignments(){
  if(!patientCatalog||patientKey!==patientSettingsKey())throw new Error('Refresh the patient list before applying ranges.');
  const values={train:parseRanges($('range-train').value,'Training'),validation:parseRanges($('range-validation').value,'Validation'),test:parseRanges($('range-test').value,'Test')};
  const owners=new Map();
  for(const [role,numbers] of Object.entries(values))for(const number of numbers){if(owners.has(number))throw new Error(`Patient ${number} appears in both ${owners.get(number)} and ${role}. Ranges must not overlap.`);owners.set(number,role);}
  if(!owners.size)throw new Error('Enter at least one patient range.');
  const byNumber=new Map();
  for(const row of patientCatalog.patients){const number=patientNumber(row.id);if(number!==null)byNumber.set(number,[...(byNumber.get(number)||[]),row.id]);}
  if([...byNumber.values()].some(ids=>ids.length>1))throw new Error('Two listed patients have the same numeric suffix, so ranges would be ambiguous. Use the individual dropdowns for those patients.');
  const splits=Object.fromEntries(Object.entries(values).map(([role,numbers])=>[role,idsForRanges(numbers,byNumber,role)]));
  const selected=Object.values(splits).flat();
  const cohortLocked=configurationScope==='train'&&Boolean(preparedJob);
  if(cohortLocked){const source=new Set(config.dataset.subjects||[]);const outside=selected.filter(id=>!source.has(id));if(outside.length)throw new Error(`These patients are not in the selected prepared snapshot: ${outside.join(', ')}.`);}
  else config.dataset.subjects=selected;
  config.training.splits=splits;
  $('range-status').textContent=`Assigned ${splits.train.length} training, ${splits.validation.length} validation and ${splits.test.length} test patients.`;
  $('range-status').className='patient-ok';
  updatePatientConfig();
}
function renderPatients(){
  const cohortLocked=configurationScope==='train'&&Boolean(preparedJob);
  $('patient-title').textContent=cohortLocked?'Training patient roles':'Patients & split assignments';
  $('patient-raw-dir').value=config.dataset.raw_dir;
  $('patient-raw-dir').disabled=cohortLocked;
  $('all-patients').checked=config.dataset.subjects===null;
  $('all-patients').disabled=cohortLocked;
  $('clear-patients').disabled=cohortLocked;
  const current=patientCatalog&&patientKey===patientSettingsKey();
  $('suggest-split').disabled=!current;
  $('patient-details').disabled=!current||patientCatalog.timing_checked;
  $('patient-table-body').replaceChildren();
  const splits=config.training.splits;
  const selectedIds=config.dataset.subjects===null?(patientCatalog?.patients||[]).map(row=>row.id):config.dataset.subjects;
  $('patient-counts').textContent=`${selectedIds.length} selected · ${splits.train.length} train · ${splits.validation.length} validation · ${splits.test.length} test`;
  $('cohort-summary').textContent=patientCatalog?`${selectedIds.length} selected / ${patientCatalog.patients.length} listed patients`:'Checking patient folders…';
  const query=$('patient-search').value.toLowerCase().trim();
  for(const row of patientCatalog?.patients||[]){
    if(query&&!row.id.toLowerCase().includes(query))continue;
    const tr=element('tr');
    const selectCell=element('td'),box=element('input');box.type='checkbox';box.checked=includedPatient(row.id);box.disabled=cohortLocked;box.setAttribute('aria-label',`Include ${row.id}`);box.onchange=()=>choosePatient(row.id,box.checked);selectCell.append(box);
    const identity=element('td');identity.append(element('strong',row.id));
    const recordings=element('td',`${row.recordings} / ${row.total_recordings}`);
    const events=element('td',!patientCatalog.timing_checked?'Not checked':row.potential_events===null?'Unavailable':String(row.potential_events));
    const status=element('td');
    status.append(element('span',!current?'Refreshing…':!row.recordings?'No matching recordings':row.incomplete_recordings?`${row.incomplete_recordings} incomplete recordings`:!patientCatalog.timing_checked?'Files found · timing not checked':row.potential_events===null?'Review metadata':row.potential_events===0?'No potential forecast events':'Files found',row.issues.length||!row.recordings||(patientCatalog.timing_checked&&row.potential_events===0)?'patient-warning':'patient-ok'));
    if(row.issues.length){const details=element('details');details.append(element('summary','Details'));for(const issue of row.issues)details.append(element('p',issue));if(row.issue_count>row.issues.length)details.append(element('p',`+ ${row.issue_count-row.issues.length} more recordings with issues`));status.append(details);}
    const assignment=element('td'),select=element('select');select.setAttribute('aria-label',`Partition for ${row.id}`);
    for(const [value,label] of [['','Unassigned'],['train','Training'],['validation','Validation'],['test','Test']]){const option=element('option',label);option.value=value;select.append(option);}
    const roles=patientRoles(row.id);select.value=roles.length===1?roles[0]:'';select.disabled=!includedPatient(row.id);select.onchange=()=>assignPatient(row.id,select.value);assignment.append(select);
    if(roles.length>1)assignment.append(element('small','Overlapping assignments: choose one role.','patient-warning'));
    if(!includedPatient(row.id)&&roles.length)assignment.append(element('small','Excluded patient still assigned in JSON.','patient-warning'));
    tr.append(selectCell,identity,recordings,events,status,assignment);$('patient-table-body').append(tr);
  }
  if(patientCatalog&&!patientCatalog.patients.length){const tr=element('tr'),td=element('td','No patient folders or BIDS recordings found in this folder.');td.colSpan=6;tr.append(td);$('patient-table-body').append(tr);}
  $('patient-warnings').replaceChildren();
  for(const warning of patientCatalog?.warnings||[])$('patient-warnings').append(element('p',warning,'patient-warning'));
  $('split-preview').hidden=!splitDraft;
  if(splitDraft){$('split-preview-text').textContent=['train','validation','test'].map(role=>`${role}: ${splitDraft.splits[role].join(', ')}`).join('\n');$('split-preview-note').textContent=splitDraft.note;}
}
function schedulePatients(){clearTimeout(patientTimer);if(patientKey!==patientSettingsKey()){splitDraft=null;$('split-preview').hidden=true;$('suggest-split').disabled=true;patientTimer=setTimeout(()=>refreshPatients().catch(e=>patientError(e)),300);}}
function patientError(error){patientKey=null;$('patient-status').textContent=error.message;$('patient-status').className='patient-warning';$('suggest-split').disabled=true;}
async function refreshPatients(includeTiming=false){
  const request=++patientRequest,key=patientSettingsKey();
  $('patient-status').textContent=includeTiming?'Checking JSON sidecars and annotations…':'Scanning BIDS filenames…';$('patient-status').className='';$('suggest-split').disabled=true;$('patient-details').disabled=true;
  try{const catalog=await api(includeTiming?'/api/patients/details':'/api/patients',{config:structuredClone(config)});if(request!==patientRequest||key!==patientSettingsKey())return;patientCatalog=catalog;patientKey=key;splitDraft=null;renderPatients();$('patient-status').textContent=`${catalog.patients.length} patients listed from ${catalog.raw_dir}. ${catalog.note}`;}
  catch(e){if(request===patientRequest&&key===patientSettingsKey())patientError(e);}
}
function setupPatients(){
  $('patient-raw-dir').onchange=()=>{config.dataset.raw_dir=$('patient-raw-dir').value;patientCatalog=null;patientKey=null;updatePatientConfig();};
  $('patient-refresh').onclick=action(async()=>{ready();clearTimeout(patientTimer);await refreshPatients();});
  $('patient-details').onclick=action(async()=>{ready();clearTimeout(patientTimer);await refreshPatients(true);});
  $('patient-search').oninput=renderPatients;
  $('all-patients').onchange=()=>{config.dataset.subjects=$('all-patients').checked?null:(patientCatalog?.patients||[]).map(row=>row.id);updatePatientConfig();};
  $('clear-patients').onclick=()=>{config.dataset.subjects=[];config.training.splits={train:[],validation:[],test:[]};updatePatientConfig();};
  $('apply-ranges').onclick=action(async()=>{ready();applyRangeAssignments();});
  $('suggest-split').onclick=action(async()=>{
    ready();const key=patientSettingsKey();$('suggest-split').disabled=true;
    try{const response=await api('/api/patients/suggest',{config:structuredClone(config)});if(key!==patientSettingsKey())return;patientCatalog=response;patientKey=key;renderPatients();const suggestion=response.suggestion;
      if(!suggestion.splits){splitDraft=null;$('split-preview').hidden=true;$('patient-warnings').replaceChildren(element('strong','A split cannot be suggested with these settings.'));for(const reason of suggestion.reasons)$('patient-warnings').append(element('p',reason,'patient-warning'));}
      else{splitDraft=suggestion;renderPatients();}
    }finally{$('suggest-split').disabled=patientKey!==patientSettingsKey();}
  });
  $('apply-split').onclick=()=>{if(!splitDraft)return;config.training.splits=structuredClone(splitDraft.splits);updatePatientConfig();notice('Suggested patient split applied to this draft. Review it before starting; pipeline readiness will check eligible classes and events.');};
}
