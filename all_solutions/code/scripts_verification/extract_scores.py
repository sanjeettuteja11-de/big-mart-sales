import json, re, os
D=os.path.expanduser('~/Downloads')
d=json.load(open(f'{D}/big-mart-sales-iii-prediction-20260912.json'))
msgs=d['messages']
print('messages:',len(msgs))

pat = re.compile(r'\b11[0-9]\d\.\d{2,}\b')
hits=[]
for i,m in enumerate(msgs):
    c=m.get('content')
    if isinstance(c,list):
        c=' '.join(str(x.get('text','')) if isinstance(x,dict) else str(x) for x in c)
    c=str(c)
    for mt in pat.finditer(c):
        s=max(0,mt.start()-160); e=min(len(c),mt.end()+120)
        hits.append((i,m.get('role'),c[s:e].replace('\n',' ')))
print('score-like hits:',len(hits))
seen=set()
for i,r,ctx in hits:
    key=ctx[:80]
    if key in seen: continue
    seen.add(key)
    print(f'--- msg{i} [{r}] {ctx}')
