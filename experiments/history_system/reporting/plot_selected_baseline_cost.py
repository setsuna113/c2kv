"""Render selected-system and published-baseline measurements from source files."""
import csv,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/weekly/figures/history_20260914'
OUT.mkdir(parents=True,exist_ok=True)
legacy=ROOT.parent/'tmp/meeting_20260907/lifecycle_cost/lifecycle_cost_bfcl.csv'
metrics=ROOT/'reports/weekly/2026-09-14-history-system.metrics.json'
source_rows=list(csv.DictReader(legacy.open(encoding='utf-8-sig')))
names={'full_reference':'Full','streamingllm_r250':'StreamingLLM','h2o_r250':'H2O','snapkv_r250':'SnapKV','hiagent':'HiAgent','acon':'ACON','cacheblend':'CacheBlend'}
rows=[]
for r in source_rows:
 if r['method_id'] not in names:continue
 def val(k):return float(r[k]) if r[k] else None
 rows.append({'method':names[r['method_id']],'cohort':'Legacy BFCL F23 / checkpoint-1088','whole_context':val('overall_logical_kv_full_over_actual'),'history':val('history_only_full_over_active_secondary'),'seconds':val('observed_request_time_sum_seconds'),'timer':'outer request (includes in-request maintenance)','requests':int(r['observed_request_time_count'])})
m=next(r['metrics'] for r in json.loads(metrics.read_text(encoding='utf-8'))['cells'] if r['benchmark']=='bfcl')
ours={'method':'Ours','cohort':'BFCL mixed20 / C1000','whole_context':m['aggregate_total_context_kv_compression']['ratio'],'history':m['aggregate_history_kv_compression']['ratio'],'seconds':m['inference_cumulative_seconds']['value'],'timer':'model inference (includes regeneration)','requests':m['inference_cumulative_seconds']['total']}
rows.append(ours)
receipt={'rows':rows,'sample_label':'preliminary, n=1','comparison':'Different fixed cohorts and checkpoints; costs use different timer scopes. No speedup ratio. Logical KV is not device peak HBM. Ours history ratio includes eviction.','sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [legacy,metrics]]}
(OUT/'baseline_compression_cost.data.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,5.4),layout='constrained')
for ax,key,title in zip(axes,['whole_context','history'],['Whole-context logical KV factor (higher)','History KV factor (higher)']):
 for i,r in enumerate(rows):
  value=r[key];color='#D55E00' if r['method']=='Ours' else '#0072B2'
  if value is not None:
   ax.scatter(value,i,color=color,s=65,marker='D' if r['method']=='Ours' else 'o');ax.annotate(f'{value:.3f}x',(value,i),xytext=(8,0),textcoords='offset points',va='center')
  else:ax.text(.04,i,'not measured',transform=ax.get_yaxis_transform(),va='center',color='#666666')
 ax.set_yticks(range(len(rows)),[r['method'] for r in rows]);ax.invert_yaxis();ax.axhline(len(rows)-1.5,color='#999999',linestyle='--');ax.axvline(1,color='#aaaaaa',linewidth=.8);ax.set_xlim(0,max(r[key] or 0 for r in rows)*1.25);ax.set_title(title);ax.set_xlabel('Full-equivalent / actual');ax.grid(axis='x',alpha=.15)
fig.suptitle('BFCL: legacy F23 above divider; Ours mixed20 below | preliminary, n=1')
for ext in ['png','svg']:fig.savefig(OUT/f'compression_vs_baselines.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[3,2]},layout='constrained')
for ax,group,title in [(axes[0],rows[:-1],'Legacy F23: outer-request cumulative time'),(axes[1],[ours],'Ours mixed20: cumulative model inference')]:
 ax.barh([r['method'] for r in group],[r['seconds'] for r in group],color=['#D55E00' if r['method']=='Ours' else '#0072B2' for r in group]);ax.invert_yaxis();ax.set_xlim(0,max(r['seconds'] for r in rows)*1.18);ax.set_xlabel('Seconds (lower)');ax.set_title(title,fontsize=10)
 for i,r in enumerate(group):ax.text(r['seconds']+35,i,f"{r['seconds']:.1f}",va='center')
 ax.grid(axis='x',alpha=.15)
fig.suptitle('Different cohorts and timer scopes; no speedup comparison | preliminary, n=1')
for ext in ['png','svg']:fig.savefig(OUT/f'cost_vs_baselines.{ext}',dpi=170)
plt.close(fig)
print(json.dumps({'data':str(OUT/'baseline_compression_cost.data.json'),'figures':2}))
