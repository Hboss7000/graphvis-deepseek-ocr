"""Frozen audit_v3: task/response extraction; only relation-list fallback reads question.

Strict case-sensitive scoring is primary. No gold/metadata enters extract().
Triple scoring policy: actual closed tuple entries with wrong field counts incur
one false-positive penalty each; one-field isolated-node lines and unclosed
trailing tuples are excluded. Prose schemas, examples and parenthetical node
lists are not tuple entries. No missing tuple fields are repaired.
This implementation imports historical audit_v2 syntax helpers and score_stage1
metrics; their hashes are part of the freeze manifest. Audit outputs are never
imported at runtime. Optional path joining and last-integer fallbacks are omitted.
"""
from __future__ import annotations
import copy
import csv
import re
from dataclasses import replace
import score_stage1_audit_v2 as s
import score_stage1 as base

VERSION = 'audit_v3'
TASK_TYPES = base.EXTENDED_TASK_TYPES
WORDS=dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty'.split(),range(21)))
NUM=r'(?:\d+|'+'|'.join(WORDS)+')'
PLACE=r'\s+in\s+(?:the\s+(?:given\s+)?|this\s+)(?:image|diagram|graph|word\s+network)'
def integer(x):return int(x) if x.isdigit() else WORDS[x.lower()]
def name(x):return s.label(x).rstrip(',.')
def labels(payload):
 quoted=list(re.finditer(s.Q,payload));residue=re.sub(s.Q,'',payload)
 if quoted and re.fullmatch(r'[\s,;.]*|[\s,;.]*and[\s,;.]*',residue,re.I):return [name(m[0]) for m in quoted]
 return s.list_payload(payload)
def bullets(text):
 result=[]
 for line in text.splitlines():
  if m:=s.BULLET.match(line):
   p=m[1].strip()
   # Retain partial plain labels; an unmatched opening quote is an incomplete tail.
   if p.count('"')%2 or p.count('`')%2:continue
   if re.search(r'->|→|←|\bis\s+connected\s+to\b|\b(?:total|degree|edges?)\s*[:=]|^There\s+is\s+an?\s+(?:directed\s+)?edge',s.clean(p),re.I):continue
   result.extend(s.list_item(p))
 return result

def relation(text,p):
 if p.decision=='COMMITTED':return p
 payload=rf'(?P<value>{s.Q}|(?:an?\s+)?{s.REL}\b)'
 stems=[rf'relation(?:ship)?(?:\s+on\s+the\s+edge)?\s+between\s+{s.Q}\s+and\s+{s.Q}',rf'relation(?:ship)?\s+(?:that\s+)?(?:connects|links|connecting|linking)\s+(?:the\s+nodes\s+)?{s.Q}\s+(?:and|to)\s+{s.Q}']
 matches=[m for stem in stems for m in re.finditer(rf'\b{stem}(?:{PLACE})?\s+is\s*:?\s*{payload}',text,re.I) if s.safe_assertion(text,m.start())]
 if matches:
  m=max(matches,key=lambda x:x.start());value=s.label(m['value']);value=re.sub(r'^is\s+a\s+(?=part\s+of\b)','',value,flags=re.I)
  if s.RETRACTION.search(text[m.end():]):return p
  return s.committed(s.base.relation_key(value),'candidate_qualified_relation',text)
 if p.decision=='COMMITTED':return p
 # Explicit taxonomy assertions; do not infer a relation from bare endpoint mentions.
 if re.search(rf'{s.Q}\s+is\s+a\s+(?:subclass|specific\s+type)\s+of\s+{s.Q}',text,re.I):
  return s.committed('isa','candidate_explicit_taxonomy',text)
 return p

def numeric(task,text,p):
 plain=s.clean(text)
 if task=='node_degree':
  patterns=[rf'{s.Q}(?:{PLACE})?\s+has\s+(?:a\s+total\s+of\s+|exactly\s+)?(?P<value>{NUM})\s+(?:connections?|edges?)\b',rf'\bdegree\s+of\s+(?:the\s+)?node\s+with\s+the\s+name\s+{s.Q}\s+is\s+(?P<value>{NUM})\b',rf'\btotal\s+number\s+of\s+connections\s+for\s+(?:the\s+)?node\s+{s.Q}\s+is\s+(?P<value>{NUM})\b']
 else:
  noun=r'(?:nodes?|vertices?)' if task=='node_number' else r'(?:edges?|connections?)'
  patterns=[rf'\b(?:we\s+(?:get|have)|there\s+are)\s+(?:a\s+total\s+of\s+)?(?P<value>{NUM})\s+{noun}\s+(?:in\s+total|in\s+the\s+graph)\b']
 matches=[m for pat in patterns for m in re.finditer(pat,plain,re.I) if s.safe_assertion(plain,m.start()) and not s.BULLET.match(plain[plain.rfind('\n',0,m.start())+1:])]
 if not matches:return p
 m=max(matches,key=lambda x:x.start())
 if s.RETRACTION.search(plain[m.end():]):return p
 # Preserve supported current totals; new patterns are fallback, not a new last-number rule.
 return p if p.decision=='COMMITTED' else s.committed(integer(m['value']),'candidate_qualified_count',text)

def highest(text,p):
 plain=s.clean(text)
 plain=re.sub(r'greatest\s+number\s+of\s+connections|most\s+connections','highest degree',plain,flags=re.I)
 pattern=rf'\bnode\s+with\s+the\s+highest\s+degree(?:{PLACE})?\s+is\s+(?P<name>{s.Q})\s*[,.;]?\s*(?:which\s+(?:has|is)|It\s+(?:has|is)|with)\s+(?:a\s+degree\s+of\s+|a\s+total\s+of\s+|connections\s+to\s+|connected\s+to\s+)?(?P<value>{NUM})\b'
 matches=[m for m in re.finditer(pattern,plain,re.I) if s.safe_assertion(plain,m.start())]
 if not matches:return p
 m=matches[-1]
 if s.RETRACTION.search(plain[m.end():]):return p
 return s.committed([name(m['name']),integer(m['value'])],'candidate_selected_maximum_pair',text)

def nodes(task,text,p):
 if task=='node_description':return p
 text=text.replace('**','')
 # Require an explicit neighbor-list assertion, followed by labels or a list.
 patterns=[rf'\bnodes\s+(?:that\s+share\s+an?\s+edge\s+with|(?:directly\s+)?connected\s+to)\s+{s.Q}(?:{PLACE})?\s+are[ \t]*:?[ \t]*(?P<payload>[^\n]*)',rf'\b(?:nodes\s+adjacent\s+to\s+{s.Q}|neighbou?rs?\s+of\s+(?:the\s+node\s+)?{s.Q})(?:{PLACE})?\s+are[ \t]*:?[ \t]*(?P<payload>[^\n]*)']
 matches=[m for pat in patterns for m in re.finditer(pat,text,re.I) if s.safe_assertion(text,m.start())]
 if not matches:return p
 for m in reversed(sorted(matches,key=lambda x:x.start())):
  payload=m['payload'].strip()
  if not payload or re.fullmatch(r'(?:the\s+)?following(?:\s+nodes)?\s*:',payload,re.I):
   bb=bullets(text[m.end():])
   if bb:return s.committed(bb,'candidate_introduced_neighbor_bullets',text)
   continue
  quoted=re.match(rf'{s.Q}(?:(?:\s*[,;]\s*(?:and\s+)?|\s+and\s+|\s+){s.Q})*',payload,re.I)
  if quoted:return s.committed(labels(quoted[0]),'candidate_qualified_neighbor_labels',text)
 return p

def path(text,p):
 indexed_first=p.rule=='path_template' and len(p.value or [])==1 and re.fullmatch(r'\d+[.)]\s+'+s.Q,p.value[0])
 if p.decision=='COMMITTED' and not indexed_first:return p
 header=re.search(r'\bshortest\s+(?:path|route|sequence|chain)(?:\s+of\s+nodes)?\b[^\n]*\bis\s*(?:as\s+follows\s*)?:\s*(?:\n|$)',text,re.I)
 if not header or not s.safe_assertion(text,header.start()):return p
 result=[]
 for line in text[header.end():].splitlines():
  if not line.strip():continue
  m=s.BULLET.match(line)
  if not m:break
  v=m[1].strip()
  if v.count('"')%2:break
  if re.fullmatch(s.Q,v) or not re.search(r'[.!?]|\b(?:start|move|then|from|go)\b',v,re.I):result.append(name(v))
  else:return p
 if result:return s.committed(result,'candidate_introduced_numbered_path',text)
 return p

def triples(text,p):
 # Remove prose-only example/schema tuples before current literal extraction.
 lines=[]
 for line in text.splitlines():
  # Parenthesized e.g./i.e. explanations remain prose even inside a bullet.
  line=re.sub(r'\(\s*(?:e\.g\.|i\.e\.|for\s+example)\s*[,;:]?[^()]*\)', '',line,flags=re.I)
  if not s.BULLET.match(line) and not line.strip().startswith('('):
   line=re.sub(r'\((?:subject|concept1|entity)\s*,\s*(?:predicate|relationship|attribute)\s*,\s*(?:object|concept2|value)\)', '',line,flags=re.I)
   if re.search(r'\b(?:example|first\s+triple)\b',line,re.I):continue
   if not re.search(r'\btriples?[^:]*:\s*\(',line,re.I):line=re.sub(r'\([^()]*,[^()]*\)', '',line)
  lines.append(line)
 cleaned='\n'.join(lines)
 cp=s.triples(cleaned)
 return replace(cp,rule='candidate_triple_prose_context',span=text) if cp.value!=p.value or cp.malformed!=p.malformed else p

EXPERIMENTS=('qualified_relation','qualified_count','selected_maximum','neighbor_syntax','numbered_path','triple_prose','combined')
def extract(task,response,experiment='combined'):
 p=s.extract(task,response);text,_=s.answer_scope(response)
 enabled=set(EXPERIMENTS[:-1]) if experiment=='combined' else {experiment}
 if task=='relation_identification' and 'qualified_relation' in enabled:p=relation(text,p)
 if task in s.base.NUMERIC_TASKS and 'qualified_count' in enabled:p=numeric(task,text,p)
 if task=='highest_node_degree' and 'selected_maximum' in enabled:p=highest(text,p)
 if task in ('node_description','neighbor_listing') and 'neighbor_syntax' in enabled:p=nodes(task,text,p)
 if task=='shortest_path_listing' and 'numbered_path' in enabled:p=path(text,p)
 if task=='triple_listing' and 'triple_prose' in enabled:p=triples(text,p)
 return p

# Additional literal response formats. No statement ids or gold.
legacy_extract=extract
import csv

def explicit_numeric(task,text,p):
 plain=s.clean(text);matches=[]
 if task=='node_degree':
  patterns=[rf'\b(?:total\s+)?degree\s+of\s+(?:the\s+)?{s.Q}\s+node\s+is\s+(?P<value>{NUM})\b',rf'\bdegree\s+of\s+(?:the\s+)?node\s+(?:with\s+the\s+name\s+)?{s.Q}\s+is\s+(?P<value>{NUM})\b',rf'{s.Q}\s+(?:node\s+)?has\s+(?:only\s+|a\s+total\s+of\s+)?(?P<value>{NUM})\s+connections?\b']
  zero=re.search(rf'{s.Q}\s+has\s+no\s+connections\b',plain,re.I)
  if zero and s.safe_assertion(plain,zero.start()):return s.committed(0,'candidate_explicit_zero_degree',text)
 else:
  noun=r'(?:nodes?|vertices?)' if task=='node_number' else r'(?:edges?|connections?)'
  patterns=[rf'\b(?:that(?:\x27s|\s+is)\s+a\s+total\s+of|total\s+(?:number\s+of\s+)?{noun}(?:\s+in\s+the\s+graph)?\s*(?:is|=|:))\s*(?:\d+\s*(?:\+\s*\d+\s*)*=\s*)?(?P<value>\d+)(?:\s+{noun})?\b',rf'\b(?:I|we)\s+(?:can\s+)?count\s+(?:a\s+total\s+of\s+)?(?P<value>\d+)\s+{noun}\b',rf'\btotal\s+number\s+of\s+{noun}(?:\s+in\s+the\s+graph)?\s+is\s+\d+(?:\s*\+\s*\d+)+\s*=\s*(?P<value>\d+)\b']
 for pat in patterns:
  for m in re.finditer(pat,plain,re.I):
   line=plain[plain.rfind('\n',0,m.start())+1:plain.find('\n',m.start()) if '\n' in plain[m.start():] else len(plain)]
   if s.safe_assertion(plain,m.start()) and (not s.BULLET.match(line) or re.match(r'^\s*[-*•]\s+Total:',line,re.I)):matches.append(m)
 if not matches:return p
 m=max(matches,key=lambda x:(x.end(),x.start()))
 if s.RETRACTION.search(plain[m.end():]):return p
 # A supported final degree conclusion takes precedence over earlier subtotals.
 if p.decision=='COMMITTED' and task!='node_degree' and '=' not in m[0]:return p
 return s.committed(integer(m['value']),'candidate_explicit_numeric_conclusion',text)

def explicit_highest(text,p):
 if p.decision=='COMMITTED' and p.value[0]!='It':return p
 plain=s.clean(text)
 plain=re.sub(r'greatest\s+number\s+of\s+connections|most\s+connections(?:\s*\(highest\s+degree\))?','highest degree',plain,flags=re.I)
 selected=re.search(r'\bnode\s+with\s+the\s+highest\s+degree(?:\s+in\s+(?:the|this)\s+graph)?\s+(?:is|appears\s+to\s+be)\s+(?P<name>[^\n.;]+)',plain,re.I)
 if selected:
  n=selected['name'];n=re.split(r'[, ]+\s*(?:with|which)\b',n)[0];n=label_name(n)
  if re.search(r'\band\b',n):return p
  following=plain[selected.end():]
  degree=re.search(r'(?:\b(?:It|Its)\s+(?:has\s+(?:a\s+total\s+of\s+|a\s+)?(?:degree\s+of\s+)?|degree\s+is\s+)|\b(?:the\s+)?degree\s+of\s+(?:the\s+)?[^\n]+?\s+(?:node\s+)?is\s+|\bhas\s+)(?P<value>\d+)\b',following,re.I)
  # Handle same sentence "is gas. It has a degree of 6" and with-degree clauses.
  if not degree:degree=re.search(r'\b(?:with\s+a\s+degree\s+of|a\s+total\s+of)\s+(?P<value>\d+)\b',plain[selected.start():],re.I)
  if degree and s.safe_assertion(plain,selected.start()) and not s.RETRACTION.search(following):return s.committed([n,int(degree['value'])],'candidate_selected_highest_and_degree',text)
 # Separate marked answer fields, with no maximization of a table.
 fields=list(re.finditer(r'(?im)^[ \t]*(?:[-*•][ \t]*)?(?:Node(?:\s+Name|\s+with\s+(?:the\s+)?(?:highest\s+degree|most\s+connections))?|Name)[ \t]*:[ \t]*([^\n]+)',plain))
 if fields:
  last=fields[-1];degree=re.search(r'(?im)^\s*(?:[-*•]\s*)?Degree\s*:\s*(\d+)\b',plain[last.end():]);n=label_name(last[1])
  if degree and ',' not in n and not s.RETRACTION.search(plain[last.end():]):return s.committed([n,int(degree[1])],'candidate_marked_highest_fields',text)
 return p

def label_name(v):return s.label(v.strip().strip('*')).rstrip(',.')

def explicit_relation(text,p):
 if p.decision=='COMMITTED':return p
 text=text.replace('**','');plain=s.clean(text)
 # Quoted predicates, passive endpoint descriptions, and relation-label-first statements.
 patterns=[rf'{s.Q}\s+(?:is\s+)?(?P<value>{s.Q})\s+{s.Q}',rf'\bnodes?\s+{s.Q}\s+and\s+{s.Q}\s+are\s+(?:connected|related)\s+(?:by|through)\s+(?:the\s+)?(?:relation|connection|label|phrase)?\s*(?:labeled\s+)?(?P<value>{s.Q})',rf'{s.Q}\s+and\s+{s.Q}\s+(?:are\s+related\s+by\s+the\s+phrase\s+)(?P<value>{s.Q})',rf'{s.Q}\s+and\s+{s.Q}\s+(?:have\s+an?\s+|are\s+related\s+to\s+each\s+other\s+through\s+the\s+)(?P<value>{s.Q})\s+relationship',rf'{s.Q}\s+has\s+an?\s+(?P<value>{s.Q})\s+relationship\s+with\s+{s.Q}',rf'(?P<value>{s.Q})\s+connects\s+{s.Q}\s+to\s+{s.Q}',rf'\bnodes\s+{s.Q}\s+and\s+{s.Q}\s+are\s+connected\s+by\s+an?\s+(?P<value>{s.Q})\s+relationship']
 matches=[m for pat in patterns for m in re.finditer(pat,text,re.I) if s.safe_assertion(text,m.start())]
 values={s.base.relation_key(s.label(m['value'])) for m in matches}
 # Every quoted predicate must be a vocabulary label; retain unknown asserted labels elsewhere.
 if len(values)==1 and values<=set(s.base.RELATION_VOCABULARY) and not s.RETRACTION.search(plain):return s.committed(values.pop(),'candidate_explicit_relation_syntax',text)
 return p

def explicit_neighbors(text,p):
 # Structured bold annotations may pollute an otherwise committed list.
 if p.decision=='COMMITTED' and not re.search(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+\*\*[^\n]+?\*\*\s*(?:\(|:)|(?m:^\s*[-*•]\s+\*\*[^\n]+?:\*\*)',text):return p
 plain=s.clean(text)
 if not re.search(r'\b(?:nodes?|neighbors?|neighbours?)\b[^\n]*(?:adjacent|connected|share|shares)|\bneighbou?rs\b|\"[^\"]+\"\s+is\s+connected\s+to',plain,re.I):return p
 annotated=[]
 for line in text.splitlines():
  m=re.match(r'^\s*(?:[-*•]|\d+[.)])\s+\*\*(.+?)\*\*\s*\(connected to\s+('+s.Q+r')\)',line,re.I)
  if m:annotated.append((s.base.relation_key(m[1].rstrip(':')),s.base.relation_key(s.label(m[2]))))
 if any(head in s.base.RELATION_VOCABULARY for head,target in annotated) and any(head not in s.base.RELATION_VOCABULARY and target in s.base.RELATION_VOCABULARY for head,target in annotated):
  return s.Extraction(None,'AMBIGUOUS','candidate_mixed_relation_and_neighbor_heads',text)
 vals=[]
 for line in text.splitlines():
  if not (m:=s.BULLET.match(line)):continue
  value=m[1].strip()
  if re.search(r'^There\s+is\s+an?\s+(?:directed\s+)?edge|\bis\s+connected\s+to\b',s.clean(value),re.I) and not re.match(r'\*\*[^*]+?:\*\*',value):continue
  if re.search(r'\(This is the relationship label\)',value,re.I):continue
  # A bold node name before a relation explanation; relation-prefix colon lists.
  b=re.match(r'\*\*(.+?)\*\*(.*)',value)
  if b:
   head,tail=b.groups()
   if head.endswith(':'):head=head[:-1];tail=':'+tail
   if tail.startswith(':') and s.base.relation_key(head) in s.base.RELATION_VOCABULARY:
    payload=tail[1:].strip()
    if re.search(r'pointing\s+towards|direct\s+connection\s+to',payload,re.I) and (qv:=re.search(s.Q,payload)):vals.append(s.label(qv[0]))
    elif re.fullmatch(s.Q,payload):vals.append(s.label(payload))
    elif payload and not payload.startswith('('):vals.append(s.label(payload))
    elif qv:=re.search(r'(?:pointing\s+towards|direct\s+connection\s+to)\s+('+s.Q+')',payload,re.I):vals.append(s.label(qv[1]))
    continue
   if re.search(r'←|→|<-',tail):continue
   if re.search(r'->',tail):
    if qv:=re.search(s.Q,tail):vals.append(s.label(qv[0]))
    continue
   vals.append(s.label(head));continue
  vals.extend(s.list_item(value))
 if vals:return s.committed(vals,'candidate_annotated_neighbor_list',text)
 # One explicitly named adjacent node, including reversed subject form.
 m=re.search(rf'\bnode\s+that\s+shares\s+an?\s+edge\s+with\s+{s.Q}\s+is\s+({s.Q}|[\w ]+)',plain,re.I)
 if m:return s.committed([s.label(m[1]).rstrip('.')],'candidate_single_neighbor',text)
 m=re.search(rf'{s.Q}\s+node\s+is\s+connected\s+to\s+the\s+({s.Q})\s+node',plain,re.I)
 if m:return s.committed([s.label(m[1])],'candidate_single_neighbor',text)
 return p

def explicit_path(text,p):
 if p.decision=='COMMITTED':return p
 plain=text.replace('**','')
 if not re.search(r'\bshortest\b[^\n]*:',plain,re.I):return p
 # Preserve sequence and repeated nodes; stop at prose, never solve or reverse.
 vals=[];started=False
 for line in plain.splitlines():
  if not line.strip():continue
  if m:=s.BULLET.match(line):
   value=m[1].split(' (',1)[0].strip();vals.append(s.label(value));started=True
  elif started:break
 if vals:return s.committed(vals,'candidate_asserted_numbered_path',text)
 return p

def explicit_triples(text,p):
 # Process the declared list before an explicitly headed Notes section.
 parts=re.split(r'(?im)^\s*\*\*(?:Notes?|Important Notes)\s*:',text,maxsplit=1)
 cleantext=parts[0] if len(parts)==1 or re.search(r'(?m)^\s*(?:[-*•]|\d+[.)])\s+',parts[0]) else text
 listed=[]
 for line in cleantext.splitlines():
  if m:=s.BULLET.match(line):
   payload=s.clean(m[1])
   # Comma-delimited fields are tuples even when parentheses are omitted.
   comma_fields=re.fullmatch(r'[^,\n]+,[^,\n]+,[^,\n]+(?:,[^,\n]+)*',payload)
   two_fields=re.fullmatch(r'[^,\n]+,\s*no connections found in KG',payload,re.I)
   if ',' in payload and not payload.startswith('(') and (comma_fields or two_fields):
    listed.append('('+payload+')');continue
   arrow=re.fullmatch(r'(.+?)\s+-([^-]+)->\s+(.+)',payload)
   if arrow:listed.append('('+', '.join(arrow.groups())+')');continue
  listed.append(line)
 return triples('\n'.join(listed),p)

EXPERIMENTS=('qualified_relation','qualified_count','selected_maximum','neighbor_syntax','numbered_path','triple_prose','gemma_numeric','gemma_highest','gemma_relation','gemma_neighbors','gemma_paths','gemma_triples','combined')
def _extract_combined(task,response,experiment='combined'):
 p=legacy_extract(task,response,'combined' if experiment=='combined' else experiment)
 text,_=s.answer_scope(response)
 enabled=set(EXPERIMENTS[:-1]) if experiment=='combined' else {experiment}
 if task in s.base.NUMERIC_TASKS and 'gemma_numeric' in enabled:p=explicit_numeric(task,text,p)
 if task=='highest_node_degree' and 'gemma_highest' in enabled:p=explicit_highest(text,p)
 if task=='relation_identification' and 'gemma_relation' in enabled:p=explicit_relation(text,p)
 if task=='neighbor_listing' and 'gemma_neighbors' in enabled:p=explicit_neighbors(text,p)
 if task=='shortest_path_listing' and 'gemma_paths' in enabled:p=explicit_path(text,p)
 if task=='triple_listing' and 'gemma_triples' in enabled:p=explicit_triples(text,p)
 return p


def candidate_marked_highest(response, parsed):
    scope, explicit = s.answer_scope(response)
    if parsed.value is not None or not explicit or not re.fullmatch(r'(?:the\s+)?node', s.clean(scope), re.I):
        return parsed
    text = re.sub(r'\*\*|(?m:^\s*#+\s*)', '', response).strip()
    final = list(s.FINAL.finditer(text))[-1]
    earlier = s.clean(text[:final.start()])
    matches = list(re.finditer(r'^\s*[-•]\s*(?P<name>[^\n:]+):\s*(?P<degree>\d+)\s*(?:[←→]\s*)?Highest(?:\s+degree)?\s*$', earlier, re.I | re.M))
    values = {(s.label(m['name']), int(m['degree'])) for m in matches}
    if len(values) != 1 or s.RETRACTION.search(earlier[matches[-1].end():]):
        return parsed
    return s.committed(list(values.pop()), 'candidate_prior_explicit_highest', response)

def candidate_question_relation(response, prompt, parsed):
    """Optional QUESTION-AWARE extension; queried endpoints come only from prompt."""
    if parsed.decision == 'COMMITTED':
        return parsed
    endpoints = re.findall(r'"([^"\n]+)"|“([^”\n]+)”', prompt)
    endpoints = [next(x for x in pair if x) for pair in endpoints]
    if len(endpoints) != 2:
        return parsed
    text, _ = s.answer_scope(response)
    header = re.search(rf'\bfollowing\s+relations?\s+for\s+({s.Q})\s*:\s*(.+)', text, re.I)
    if (not header or s.label(header[1]) != endpoints[0]
            or not s.safe_assertion(text, header.start())
            or s.RETRACTION.search(text[header.start():])):
        return parsed
    values = set()
    for field in s.template.list_fields(header[2]):
        match = re.fullmatch(rf'\s*({s.REL})\s+(?:to\s+)?({s.Q})[.\s]*', field, re.I)
        if match and s.label(match[2]) == endpoints[1]:
            values.add(s.base.relation_key(match[1]))
    if len(values) != 1:
        return parsed
    return s.committed(values.pop(), 'candidate_relation_for_question_endpoints', text)

def extract(task, response, question=''):
    """Extract before scoring; question is used only for relation-list endpoints."""
    response = '' if response is None else str(response)
    parsed = _extract_combined(task, response)
    if task == 'highest_node_degree':
        parsed = candidate_marked_highest(response, parsed)
    if task == 'relation_identification':
        parsed = candidate_question_relation(response, question, parsed)
    return parsed

def score_record(record, response, meta, extractor=VERSION):
    if extractor!=VERSION:raise ValueError(f'This scorer only supports {VERSION}')
    response='' if response is None else str(response).strip()
    task=record['task_type'];parsed=extract(task,response,record.get('prompt',''))
    # Empty response initializes the frozen gold validation and metric schema;
    # all prediction fields are then replaced with the gold-free extraction.
    row=base.score_record(record,'',meta,extractor='span_extended')
    value=parsed.value
    if task in base.NUMERIC_TASKS:
        row.update(parsed_integer=value,signed_error=value-row['gold_integer'] if value is not None else None,
                   is_correct=value==row['gold_integer'],lenient_containment=base.integer_containment(response,row['gold_integer']))
    elif task=='highest_node_degree':
        name,degree=value if value is not None else (None,None)
        row.update(parsed_name=name,parsed_degree=degree,degree_correct=degree==row['gold_degree'],
                   signed_error=degree-row['gold_degree'] if degree is not None else None,
                   lenient_containment=base.integer_containment(response,row['gold_degree']))
        for v in base.SET_VARIANTS:
            row[f'name_correct_{v}']=name is not None and base.normalize_component(name,v) in {base.normalize_component(n,v) for n in row['valid_highest_names']}
            row[f'both_correct_{v}']=row['degree_correct'] and row[f'name_correct_{v}']
            row[f'mentioned_valid_names_{v}']=[n for n in row['valid_highest_names'] if base._phrase_present(response,n,v)]
        row['is_correct']=row['both_correct_raw']
    elif task=='relation_identification':
        correct=value==base.relation_key(row['gold_relation'])
        row.update(parsed_relation=value,relation_correct={v:correct for v in base.SET_VARIANTS},is_correct=correct,
                   lenient_containment=bool(base.RELATION_PATTERNS[base.relation_key(row['gold_relation'])].search(response)))
    elif task=='shortest_path_listing':
        gold=base.structured_gold(record);predicted=value or [];exact={};partial={}
        for v in base.SET_VARIANTS:
            pp=[base.normalize_component(n,v) for n in predicted]
            paths=[[base.normalize_component(n,v) for n in p] for p in gold['paths']]
            exact[v]=parsed.decision=='COMMITTED' and pp in paths
            overlap=max(base.set_counts(set(p),set(pp))['f1'] for p in paths)
            partial[v]={'hop_count_correct':bool(predicted) and len(predicted)-1==gold['distance'],'node_overlap_f1':overlap}
        keys=[base.normalize_component(n,'annotation_stripped') for n in predicted]
        byv={v:base.path_containment(response,gold['paths'],v) for v in base.SET_VARIANTS}
        row.update(parsed_path=predicted,path_exact=exact,path_partial=partial,parsed_hop_count=len(predicted)-1 if predicted else None,
                   degenerate_output=len(keys)!=len(set(keys)),is_correct=exact['raw'],lenient_containment=byv['raw'],lenient_containment_by_variant=byv)
    else:
        predicted=value or []
        if task=='triple_listing':
            gold=[tuple(t) for t in row['parsed_gold_triples']]
            predicted=[tuple(t) for t in predicted]
            row.update(parsed_predicted_triples=[list(t) for t in predicted],malformed_triple_line_count=parsed.malformed,
                       incomplete_triple_line_count=parsed.incomplete,
                       gold_node_component_recall={v:base._component_recall(gold,response,v) for v in base.SET_VARIANTS})
        else:
            gold=row['parsed_gold_nodes'];row['parsed_predicted_nodes']=predicted
        metrics=base.set_metrics_by_variant(gold,predicted,parsed.malformed)
        for m in metrics.values():
            if parsed.decision=='COMMITTED' and not (m['true_positive'] or m['false_positive'] or m['false_negative']):
                m.update(precision=1.0,recall=1.0,f1=1.0)
        if parsed.decision!='COMMITTED':
            for m in metrics.values():m['exact_set_equality']=False
        byv={v:base.member_containment(response,gold,v) for v in base.SET_VARIANTS}
        row.update(set_metrics=metrics,is_correct=metrics['raw']['exact_set_equality'],lenient_containment=byv['raw'],lenient_containment_by_variant=byv)
    row.update(extractor=VERSION,scorer_version=VERSION,parser_rule=parsed.rule,extraction_decision=parsed.decision,
               extracted_claim=value,answer_span=parsed.span)
    return row


def aggregate_task(task, rows, extractor=VERSION):
    if extractor!=VERSION:raise ValueError(f'This scorer only supports {VERSION}')
    metrics=base.aggregate_task(task,rows,extractor='span_extended')
    if task=='highest_node_degree':
        # The task asks for a node AND its degree; keep degree-only accuracy as
        # a diagnostic rather than awarding strict credit to a wrong name.
        metrics['strict_degree_accuracy']=metrics['degree_accuracy']
        metrics['strict_accuracy']=metrics['strict_joint_accuracy']
    metrics['extraction_decisions']={d:sum(r['extraction_decision']==d for r in rows) for d in ('COMMITTED','NO_FINAL_CLAIM','AMBIGUOUS')}
    return metrics


def score_files(args):
    if getattr(args,'extractor',VERSION)!=VERSION:raise ValueError(f'This scorer only supports {VERSION}')
    old_args=copy.copy(args);old_args.extractor='span_extended'
    metrics,old_rows=base.score_files(old_args)
    records,metadata=base.validate_inputs(base.read_jsonl(args.input_jsonl),base.read_jsonl(args.graph_metadata),base.TASK_SETS[getattr(args,'task_set','paper')])
    rows={task:[{**r,**score_record(records[(int(r['statement_idx']),task)],r.get('raw_response',''),metadata[int(r['statement_idx'])])} for r in rr] for task,rr in old_rows.items()}
    metrics['tasks']={task:aggregate_task(task,rr) for task,rr in rows.items()}
    metrics['scorer_version']=VERSION
    metrics['scoring_notes'].update(extractor=VERSION,
        answer_span='Last explicit answer or literal task assertion; prior uniquely marked Highest may survive an unfinished final block. Only relation-list endpoint fallback reads the question; never gold.',
        numeric_parse='Asserted totals/degrees, including explicit written arithmetic conclusions; never compute totals from lists or arithmetic operands.',
        highest_node_degree_extraction='Pair a selected node with its asserted degree; candidate degree tables alone make no final selection.',
        strict='Raw normalization; highest-degree strict_accuracy requires BOTH a selected maximum-degree node and its correct degree. strict_degree_accuracy remains diagnostic.',
        literal_list_labels='Quoted/bold/backtick and plain lists retain literal labels, including stopwords and relation-name nodes. Explanation suffixes are syntax.',
        triple_extraction='Closed literal tuple entries with wrong field counts incur FP penalties; one-field isolated-node lines and unclosed trailing tuples are excluded; prose schemas/examples are not entries; no field repairs.',
        relation_identification='Explicit label assertion or one asserted relation triple; unknown asserted labels preserved; absence distinct from candidates and no final claim.',
        shortest_path_listing='One asserted ordered path; no graph solving or gold-endpoint selection. Explicit no-path claim overrides tentative traces.',
        extraction_decision='COMMITTED, NO_FINAL_CLAIM or AMBIGUOUS; missing claims never receive strict exact credit.')
    return metrics,rows
