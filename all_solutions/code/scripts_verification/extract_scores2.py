import json, re, os
D=os.path.expanduser('~/Downloads')
d=json.load(open(f'{D}/big-mart-sales-iii-prediction-20260912.json'))
msgs=d['messages']

def text(m):
    c=m.get('content')
    if isinstance(c,list):
        c=' '.join(str(x.get('text','')) if isinstance(x,dict) else str(x) for x in c)
    return str(c)

# only USER messages (he reports LB scores), and short ones (not data dumps)
pat = re.compile(r'\b1[01]\d\d\.\d+\b')
for i,m in enumerate(msgs):
    if m.get('role')!='user': continue
    t=text(m)
    if len(t)>3000: continue
    if pat.search(t) or re.search(r'\bscore\b|\brank\b|leaderboard', t, re.I):
        print(f'===== msg{i} len={len(t)}')
        print(t[:900].replace('\n',' | '))
