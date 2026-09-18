import json, sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PORT = sys.argv[1] if len(sys.argv) > 1 else "36206"
def post(path, body, timeout=300):
    req = urllib.request.Request('http://127.0.0.1:'+PORT+path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(opener.open(req, timeout=timeout))

def probe(tag, msgs, hist_count):
    post('/open_session', {'capacity_of_str_len':0,'session_id':'evictdbg-'+tag,'streaming':True,'timeout':600.0})
    hint = {
      'full_equivalent_history_tokens':0,'active_history_kv_tokens':32,
      'active_full_raw_tokens':0,'active_c2kv_gist_tokens':0,
      'history_kv_method':'h2o','estimated':True,'history_kv_backend':'physical_eviction',
      'history_kv_eviction': {'method':'h2o','history_start_message_count':1,
        'history_message_count':hist_count,'target_tokens':32,
        'history_kv_recent_window':16,'history_kv_kernel_size':5,
        'history_kv_pooling':'avgpool','history_kv_h2o_recent_fraction':0.5,
        'persistent_session':True},
      'persistent_history_session': {'enabled':True},
    }
    r = post('/v1/chat/completions', {'model':'gen-c1000','messages':msgs,'temperature':0,'max_tokens':30,
        'session_params':{'id':'evictdbg-'+tag},'c2kv_kv_memory_hint':hint})
    rep = (r.get('metadata') or {}).get('kv_memory_report') or {}
    ev = rep.get('history_kv_eviction') or {}
    phys = rep.get('history_kv_physical_eviction') or {}
    print('[%s] hist_start=%s hist_end=%s full_hist=%s active=%s src=%s' % (tag, ev.get('history_start'), ev.get('history_end'), rep.get('full_equivalent_history_tokens'), rep.get('active_history_kv_tokens'), rep.get('active_history_kv_tokens_source')))
    print('[%s] phys: success=%s freed=%s kept=%s status=%s' % (tag, phys.get('success'), phys.get('freed_physical_slots'), phys.get('kept_history_tokens'), rep.get('history_kv_runtime_status')))

plain = [{'role':'system','content':'You are a helpful assistant.'}]
for i in range(4):
    plain.append({'role':'user','content':'Question %d: what is %d times %d? Just the number.' % (i,i,i+1)})
    plain.append({'role':'assistant','content':'%d' % (i*(i+1))})
plain.append({'role':'user','content':'What was question 2 about? One line.'})
probe('plain', plain, 8)

tool_msgs = [
    {'role':'system','content':'You have tools. Use them.'},
    {'role':'user','content':'Send hello to user USR001.'},
    {'role':'assistant','content':'', 'tool_calls':[{'id':'c1','type':'function','function':{'name':'send_message','arguments':{'msg':'hello','to':'USR001'}}}]},
    {'role':'tool','tool_call_id':'c1','content':'{"sent": true, "id": 42}'},
    {'role':'assistant','content':'', 'tool_calls':[{'id':'c2','type':'function','function':{'name':'send_message','arguments':{'msg':'world','to':'USR002'}}}]},
    {'role':'tool','tool_call_id':'c2','content':'{"sent": true, "id": 43}'},
    {'role':'user','content':'Did both sends succeed? One line.'},
]
probe('tools', tool_msgs, 5)
