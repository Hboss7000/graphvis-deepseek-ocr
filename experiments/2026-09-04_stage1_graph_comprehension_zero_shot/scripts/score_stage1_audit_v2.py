"""Gold-independent Stage 1 extraction informed by the sealed answer audit.

Historical scorers are unchanged. Parse assertions before scoring them: never
select a number, name, relation or path because it agrees with the gold graph.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import re

import score_stage1 as base
import score_stage1_template_v1 as template

VERSION = 'audit_v2'
TASK_TYPES = base.EXTENDED_TASK_TYPES
Q = r'(?:"[^"\n]+"|“[^”\n]+”|`[^`\n]+`|\*\*[^*\n]+\*\*)'
RELATIONS = ('not capable of', 'not desires', 'receives action', 'has subevent',
             'has property', 'has context', 'at location', 'capable of',
             'created by', 'related to', 'used for', 'part of', 'made of',
             'is a', 'antonym', 'causes', 'desires')
REL = '(?:' + '|'.join(re.escape(x) for x in RELATIONS) + ')'
FINAL = re.compile(r'(?im)^\s*(?:[✅✔]\s*)?(?:#+\s*)?(?:final\s+answer|final\s+response|answer)\s*:\s*')
BULLET = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+(.+)$')
UNKNOWN = re.compile(r'^(?:unknown|unclear|not sure|cannot determine|unable to determine|i don.t know)[.!\s]*$', re.I)
RETRACTION = re.compile(r"\b(?:my|the)\s+(?:previous|earlier|initial)\s+(?:answer|count|total|path)\s+(?:was|is)\s+(?:wrong|incorrect)|\bi\s+was\s+wrong\b|\bthat(?:'s|\s+is)\s+(?:wrong|incorrect)\b", re.I)


@dataclass(frozen=True)
class Extraction:
    value: object = None
    decision: str = 'NO_FINAL_CLAIM'
    rule: str = 'no_asserted_answer'
    span: str = ''
    malformed: int = 0
    incomplete: int = 0


def clean(text):
    return base.strip_markdown(text).strip()


def label(text):
    value = text.strip()
    for left, right in (('**','**'), ('`','`'), ('"','"'), ('“','”'), ("'","'")):
        if value.startswith(left) and value.endswith(right) and len(value)>len(left)+len(right):
            return label(value[len(left):-len(right)].strip())
    return value


def answer_scope(response):
    # Generic 'therefore' in a trailing explanation must not erase a list.
    # Preserve inline-code quotes for list/triple parsing.
    text = re.sub(r'\*\*|(?m:^\s*#+\s*)', '', response).strip()
    markers = list(FINAL.finditer(text))
    return (text[markers[-1].end():].strip(), True) if markers else (response.strip(), False)


def committed(value, rule, span, **kwargs):
    return Extraction(value, 'COMMITTED', rule, span, **kwargs)


def safe_assertion(text, start):
    prefix = clean(text[max(text.rfind('\n',0,start), text.rfind('.',0,start))+1:start])
    return not re.search(r'\b(?:if|perhaps|maybe|might|could|would|not)\b|\?', prefix, re.I)


def select(matches, text, rule):
    matches = [m for m in matches if safe_assertion(text,m.start()) and not BULLET.match(text[text.rfind('\n',0,m.start())+1:text.find('\n',m.start()) if '\n' in text[m.start():] else len(text)])]
    if not matches:
        return Extraction(span=text)
    match = max(matches, key=lambda m:m.start())
    if RETRACTION.search(text[match.end():]) or re.search(r"(?:let.s|let\s+me)\s+recount|I\s+(?:think\s+I\s+)?(?:missed|double.counted)",text[match.end():],re.I):
        return Extraction(None, 'AMBIGUOUS', 'retracted_without_replacement', text)
    return committed(int(match['value']), rule, text)


def numeric(task, text, explicit=False):
    plain = clean(text)
    if explicit and re.fullmatch(r'[-+]?\d+[.]?',plain.splitlines()[0]):
        return committed(int(plain.splitlines()[0].rstrip('.')), 'final_integer_line',text)
    words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10}
    plain=re.sub(r'\b(?:'+ '|'.join(words) +r')\b',lambda m:str(words[m[0].lower()]),plain,flags=re.I)
    if re.fullmatch(r'[-+]?\d+[.]?',plain):
        return committed(int(plain.rstrip('.')), 'bare_integer', text)
    if task in ('node_number','edge_number'):
        noun = r'(?:nodes?|vertices?)' if task=='node_number' else r'(?:edges?|connections?)'
        patterns = [
            rf'\b(?:there\s+are|(?:the\s+)?graph\s+(?:has|contains))\s+(?:a\s+total\s+of\s+)?(?P<value>[-+]?\d+)\s+{noun}\b',
            rf'\b(?:total\s+)?(?:number|count)\s+of\s+{noun}(?:\s+in\s+(?:the\s+)?graph)?\s*(?:is|=|:)\s*(?P<value>[-+]?\d+)\b',
            rf'\btotal\s+{noun}\s*(?:is|=|:)\s*(?:\d+\s*(?:\+\s*\d+\s*)*=\s*)?(?P<value>[-+]?\d+)\b',
            rf"\b(?:that(?:'s|\s+is)|we\s+(?:have|count)|counting\s+all[^\n:]*:)\s+(?P<value>[-+]?\d+)\s+{noun}(?!\s+from)\b",
            rf'^\s*(?P<value>[-+]?\d+)\s+{noun}(?:\s+(?:in\s+total|in\s+the\s+graph))?[.\s]*$',
        ]
    else:
        patterns = [
            rf'(?<![-\w])(?:total\s+)?degree\s+(?:of|for)\s+(?:the\s+)?(?:node\s+)?(?:labeled\s+)?{Q}\s*(?:is|=|:)\s*(?P<value>[-+]?\d+)\b',
            r'(?<![-\w])(?:total\s+)?degree\s*(?:is|=|:|of)\s*(?P<value>[-+]?\d+)\b',
            rf'{Q}\s+has\s+(?P<value>[-+]?\d+)\s+(?:connections?|incident\s+edges?|edges?|neighbou?rs?)\b',
            r'\btotal\s+(?:number\s+of\s+)?connections\s*(?:is|=|:)\s*(?P<value>[-+]?\d+)\b',
            rf'\btotal\s+number\s+of\s+connections\s+that\s+(?:the\s+)?node\s+{Q}\s+has\s+is\s+(?P<value>\d+)\b',
            rf'{Q}\s+has\s+(?:exactly\s+|a\s+total\s+of\s+)?(?P<value>\d+)\s+(?:connections?|edges?)\b',
            rf'\b(?:total\s+)?number\s+of\s+connections\s+(?:the\s+)?node\s+{Q}\s+has\s+is\s+(?P<value>\d+)\b',
            r'\bonly\s+connections\s+are\s+the\s+(?P<value>\d+)\s+listed\b',
        ]
    matches=[m for p in patterns for m in re.finditer(p,plain,re.I|re.M)]
    if task=='edge_number':
        # Per-node/cluster subtotals in a counting draft are not graph totals.
        headings=list(re.finditer(r'^\s*(?:\d+[.)]\s*)?From\s+[^\n:]+:',plain,re.I|re.M))
        matches=[m for m in matches if not (re.match(r"that(?:'s|\s+is)",m[0],re.I) and any(h.end()<m.start() for h in headings))]
    return select(matches,plain,'asserted_'+task)


def highest(text, explicit):
    plain = clean(text)
    plain=re.sub(r'greatest\s+number\s+of\s+connections|most\s+connections','highest degree',plain,flags=re.I)
    name=rf'(?P<name>{Q}|[^\n,;:.]+?)'
    number=r'(?P<value>[-+]?\d+)'
    patterns=[
        rf'^\s*(?:the\s+node\s+)?{name}\s*,?\s+(?:with\s+)?(?:a\s+)?(?:degree\s*(?:of|is|:|=)?\s*|(?=\d))?{number}(?:\s+connections?)?[.\s]*$',
        rf'\b(?:nodes?\s+with\s+(?:the\s+)?highest\s+degree|highest[- ]degree\s+node)(?:\s+in\s+the\s+graph)?\s+(?:is|:)\s*{name}\s*,?\s*(?:with|having|and\s+(?:its\s+degree|its\s+total|it)\s+(?:is|has))\s+(?:a\s+)?(?:degree\s+of\s+|total\s+of\s+)?{number}\b',
        rf'(?:^|\n|\bthe\s+node\s+){name}\s+has\s+(?:the\s+)?highest\s+degree\s*,?\s*(?:with\s+(?:a\s+)?(?:degree\s+of\s+|total\s+of\s+)?|of\s+)?{number}\b',
        rf'(?:^|[,\n]\s*){name}\s+is\s+(?:the\s+)?node\s+with\s+(?:the\s+)?highest\s+degree\s*[, ]*\s*(?:\(\s*|and\s+its\s+degree\s+is\s+){number}\b',
        rf'\b(?:node\s+)?name\s*(?:is|:)\s*{name}\s*(?:,?\s+and\s+(?:the\s+)?degree\s+is|\n\s*degree\s*:)\s*{number}\b',
        rf'^\s*{name}\s+has\s+(?:a\s+)?degree\s+of\s+{number}[.\s]*$',
    ]
    matches=[m for p in patterns[1:] for m in re.finditer(p,plain,re.I|re.M) if safe_assertion(plain,m.start())]
    if not matches and (explicit or len(plain.splitlines())==1):
        compact=re.fullmatch(patterns[0],plain,re.I)
        if compact and not re.search(r'\b(?:has|is|that|it|incoming|outgoing)\b|->|→',compact['name'],re.I):matches=[compact]
    if matches:
        m=max(matches,key=lambda m:m.start())
        selected=label(re.sub(r'^(?:the\s+node\s+|therefore,?\s+|thus,?\s+)','',m['name'],flags=re.I))
        selected=label(re.split(r'\s+\(or\s+',selected,maxsplit=1,flags=re.I)[0])
        if RETRACTION.search(plain[m.end():]):return Extraction(None,'AMBIGUOUS','retracted_highest',text)
        return committed([selected,int(m['value'])],'highest_assertion',text)
    # A selected maximum can be followed by its degree in the next sentence.
    chosen=list(re.finditer(rf'\bnode\s+with\s+(?:the\s+)?highest\s+degree(?:\s+in\s+the\s+graph)?\s+is\s+({Q})[.]',plain,re.I))
    if chosen:
        m=chosen[-1]
        degree=re.match(r'\s*(?:It\s+has|Its\s+degree\s+is)\s+(?:a\s+total\s+of\s+)?(\d+)\s*(?:connections?|edges?|[.])',plain[m.end():],re.I)
        if degree:return committed([label(m[1]),int(degree[1])],'highest_following_degree',text)
    tie=re.search(r'\bnodes?\s+with\s+(?:the\s+)?highest\s+degree\s+(?:are|is)\s+(.+?)\s*,?\s+(?:(?:both|each)\s+)?(?:with|having)\s+(?:a\s+degree\s+of\s+)?(\d+)\b',plain,re.I)
    if tie:
        names=re.split(r'\s+and\s+|,\s*',tie[1].rstrip(','))
        return committed([label(names[0]),int(tie[2])],'highest_asserted_tie',text)
    return Extraction(span=text)


def list_payload(payload):
    # Keep syntax inside quotes/renderer annotations. Do not use a dictionary
    # of graph members or discard labels such as 'being', 'but', 'related to'.
    payload=payload.strip().rstrip('.')
    quoted=list(re.finditer(Q,payload))
    residue=re.sub(Q,'',payload)
    if quoted and re.fullmatch(r'[\s,;]*(?:(?:and|or)[\s,;]*)*',residue,re.I):
        return [label(m.group()) for m in quoted]
    return [label(x.strip()) for x in template.list_fields(payload) if x.strip()]


def list_item(payload):
    quoted=re.match(Q,payload)
    if quoted:
        tail=payload[quoted.end():].strip()
        if not tail or re.match(r'^(?:\(|:|[-–—]|(?:is\s+)?connected\b)',tail,re.I):
            return [label(quoted.group())]
    payload=re.sub(r'\s+\((?:connected|via|appears|edge|outgoing|incoming)[^)]*\).*$', '',payload,flags=re.I)
    payload=re.split(r':\s*(?:connected|edges?|There\s+is)\b',payload,maxsplit=1,flags=re.I)[0]
    payload=re.split(r'\s+-\s+with\s+(?:the\s+)?relationship\b',payload,maxsplit=1,flags=re.I)[0]
    payload=re.split(r'\s+[-–—]\s+(?:via|connected|edge|related\b)',payload,maxsplit=1,flags=re.I)[0]
    return list_payload(payload)


def nodes(task, text):
    text=text.replace('**','')
    plain=clean(text)
    if task=='neighbor_listing' and re.fullmatch(r'none|\[\]|no\s+neighbou?rs?[.!]?',plain,re.I):
        return committed([],'explicit_empty_list',text)
    bullets=[]
    for line in text.splitlines():
        if m:=BULLET.match(line):
            # Arrow traces and counts in a reasoning section are not labels.
            if re.search(r'->|→|←|\bis\s+connected\s+to\b|\b(?:total|degree|edges?)\s*[:=]|^There\s+is\s+an?\s+(?:directed\s+)?edge',clean(m[1]),re.I):
                continue
            bullets.extend(list_item(m[1].strip()))
    noun=r'(?:neighbou?rs?|adjacent\s+nodes?|nodes\s+adjacent\s+to[^\n:]+)' if task=='neighbor_listing' else r'(?:nodes|vertices)'
    patterns=[rf'\b{noun}\s*(?:are|include|consist\s+of)?\s*:\s*(?P<payload>[^\n]+)',
              rf'\b{noun}\s+(?:is|are|include)\s+(?P<payload>[^\n]+)']
    if task=='node_description':
        patterns.append(r'\bthe\s+image\s+depicts\s+the\s+following\s+nodes\s*:\s*(?P<payload>[^\n]+)')
    else:
        patterns.extend([rf'\b(?:is|are)\s+(?:directly\s+)?connected\s+to\s*:?\s*(?:the\s+node\s+)?(?P<payload>{Q}(?:[^\n]*))',
                         r'\b(?:is|are)\s+(?:directly\s+)?connected\s+to\s*:\s*(?P<payload>[^\n]+)',
                         rf'\b(?:one\s+)?node\s+that\s+shares?\s+an?\s+edge\s+with\s+{Q}\s+is\s*:?\s*(?P<payload>[^\n]+)',
                         rf'\b(?:only\s+)?node\s+connected\s+to\s+{Q}\s+is\s*:?\s*(?P<payload>[^\n]+)',
                         rf'\b(?:only\s+)?neighbou?r\s+of\s+(?:the\s+node\s+)?{Q}\s+is\s*:?\s*(?P<payload>[^\n]+)',
                         rf'\badjacent\s+(?:node\s+)?to\s+{Q}\s+is\s*:?\s*(?P<payload>[^\n]+)'])
    matches=[m for p in patterns for m in re.finditer(p,text,re.I) if safe_assertion(text,m.start())]
    if task=='node_description' and bullets:
        return committed(bullets,'literal_bullet_list',text)
    for m in sorted(matches,key=lambda m:m.start(),reverse=True):
        if bullets and m.start()>text.rfind('\n',0,len(text)) and 'connected to' in m.group().lower():continue
        if bullets and re.search(r'\b(?:these|all|no\s+other)\b',text[text.rfind('\n',0,m.start())+1:m.start()],re.I):continue
        payload=m['payload'].strip()
        # An unquoted sentence ends at its punctuation; periods inside a
        # quoted name stay literal (e.g. "St. Louis").
        quoted=re.match(rf'(?:{Q})(?:\s*(?:[,;]\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)(?:{Q}))*',payload,re.I)
        if quoted:
            payload=quoted.group()
        else:
            payload=re.split(r'[.!?](?:\s|$)',payload,maxsplit=1)[0]
        if not payload or BULLET.match(payload) or re.match(r'^(?:the\s+)?(?:following|listed|shown|represented|connected|\d+\s+nodes?)\b',clean(payload),re.I):
            continue
        if re.fullmatch(r'(?:none|no\s+(?:adjacent\s+)?nodes?|no\s+neighbou?rs?)',clean(payload),re.I):
            return committed([],'explicit_empty_list',text)
        return committed(list_payload(payload),'list_assertion',text)
    if bullets:
        return committed(bullets,'literal_bullet_list',text)
    if task=='neighbor_listing' and re.search(r'\b(?:has\s+no\s+neighbou?rs?|no\s+(?:direct\s+)?connections?\s+(?:to|with)\s+(?:any\s+)?other\s+nodes)\b',plain,re.I):
        return committed([],'explicit_empty_list',text)
    if UNKNOWN.fullmatch(plain) or re.search(r'\bdegree\b|\bconnections?\b|\bgraph\b',plain,re.I):
        return Extraction(span=text)
    if plain and len(plain.splitlines())==1 and not re.search(r'[.!?]\s|\b(?:is|are|because|therefore)\b',plain,re.I):
        return committed(list_payload(text),'bare_label_list',text)
    return Extraction(span=text)


def triples(text):
    predicted=[];malformed=0;incomplete=0
    for raw in text.splitlines():
        line=re.sub(r'^\s*(?:[-*•]|\d+[.)])\s+','',raw).strip()
        if not line or re.match(r'^(?:note|example|e\.g\.|these|therefore)\b',clean(line),re.I):
            continue
        if re.search(r'\b(?:node\s+is\s+isolated|no\s+triples\s+can\s+be\s+formed)\b',line,re.I):continue
        if re.match(r'^[^:]+:\s*(?:The\s+relationship|This\s+is)\b',clean(line),re.I):continue
        quoted=re.match(rf'({Q})\s+(?:-\s*)?({Q})\s+(?:-\s*)?({Q})',line)
        if quoted:
            predicted.append(tuple(label(c) for c in quoted.groups()));continue
        groups=list(base.PARENTHESIZED_RE.finditer(line))
        tuple_groups=[m for m in groups if ',' in m[1]]
        if tuple_groups:
            for m in tuple_groups:
                fields=[label(c.strip()) for c in template.list_fields(m[1])]
                if len(fields)==3 and all(fields):predicted.append(tuple(fields))
                else:malformed+=1
            continue
        # Only list-like triple lines, or a whole single-line triple, may use
        # unquoted delimiters. Introductory/explanatory prose is not a triple.
        listed=BULLET.match(raw) is not None or len(text.splitlines())==1
        if line.startswith('(') and ')' not in line:
            incomplete+=1
            continue
        if not listed:continue
        if line.count('`')%2:
            incomplete+=1
            continue
        if m:=re.fullmatch(r'(.+?)\s+-\s+(.+?)\s+-\s+(.+)',clean(line)):
            predicted.append(tuple(label(c) for c in m.groups()));continue
        if m:=re.fullmatch(rf'(.+?)\s+({REL})\s+(.+)',clean(line),re.I):
            predicted.append(tuple(label(c) for c in m.groups()));continue
        if re.search(r'\bno\s+connections?\s+found\b',clean(line),re.I):continue
        if line.startswith('(') or line.startswith(('`','**')):
            incomplete+=1
    if predicted or malformed:
        return committed(predicted,'literal_triples',text,malformed=malformed,incomplete=incomplete)
    return Extraction(span=text,incomplete=incomplete)


def relation(text):
    text=text.replace('**','');plain=clean(text)
    if re.search(r'\bno\s+(?:direct\s+)?(?:edge|connection|relation(?:ship)?)\b|\bedge\b[^.\n]*\b(?:does\s+not\s+exist|not\s+present)\b',plain,re.I):
        return committed('NO_EDGE','explicit_no_edge',text)
    payload=rf'(?P<value>{Q}|(?:an?\s+)?{REL}\b)'
    patterns=[
        rf'\brelation(?:ship)?(?:\s+on\s+the\s+edge)?\s+between\s+{Q}\s+and\s+{Q}(?:\s+in\s+the\s+graph)?\s+is\s*:?\s*{payload}',
        rf'\brelation(?:ship)?\s+that\s+(?:connects|links)\s+(?:the\s+nodes\s+)?{Q}\s+(?:and|to)\s+{Q}\s+is\s+{payload}',
        rf'\brelation(?:ship)?\s+linking\s+{Q}\s+to\s+{Q}\s+is\s+{payload}',
        rf'\brelation(?:ship)?\s+{payload}\s+(?:links|connecting|connects|between)\b',
        rf'\bfollowing\s+relations?\s*:\s*{payload}[.\s]*$',
        rf'\b(?:the\s+)?(?:relation(?:ship)?|edge\s+label)\s*(?:is|:)\s*{payload}[.\s]*$',
    ]
    matches=[m for p in patterns for m in re.finditer(p,text,re.I) if safe_assertion(text,m.start())]
    # A quoted endpoint followed by a predicate is a triple, not a label.
    matches=[m for m in matches if not re.match(rf'\s+(?:{REL}\b|is\s+(?:{REL}\b|{Q}))',text[m.end():],re.I)]
    direct=re.search(rf'\b(?:specific[, ]+)?direct\s+relationship[^.\n]*?\bis\s+({Q})',text,re.I)
    if direct:return committed(base.relation_key(label(direct[1])),'direct_relation_conclusion',text)
    if matches:
        value=label(max(matches,key=lambda m:m.start())['value'])
        value=re.sub(r'^an?\s+(?='+REL+r')','',value,flags=re.I)
        return committed(base.relation_key(value),'relation_assertion',text)
    if re.fullmatch(REL,plain,re.I) or base.relation_key(plain) in base.RELATION_VOCABULARY:
        return committed(base.relation_key(plain),'bare_relation',text)
    # Explicit relationship paraphrases and subject-predicate-object claims.
    assertion=re.search(r'\b(?:following\s+relationship|relationship\s+between[^\n]+?is)\s*:\s*(.+?)[.]?$',plain,re.I)
    if assertion:
        value=assertion[1]
        if m:=re.search(rf'\s+(?:is\s+(?:a\s+)?)?({REL})\s+',value,re.I):
            return committed(base.relation_key(m[1]),'relation_triple_payload',text)
    if re.search(r'\bis\s+a\s+type\s+of\b',plain,re.I) and re.search(r'\brelationship\s+between\b',plain,re.I):
        return committed('isa','relation_type_of',text)
    concluding=re.search(rf'\b(?:relationship|relation)[^.\n]*(?:is\s+that\s+they\s+are\s+(antonyms)|is\s*:\s*{Q}\s+({REL})\s+{Q}|is\s*:\s*{Q}\s+is\s+({Q}))',text,re.I)
    if concluding:
        value=next(v for v in concluding.groups() if v)
        return committed('antonym' if value.lower()=='antonyms' else base.relation_key(label(value)),'relation_conclusion',text)
    # Without an explicit answer, several distinct labels are ambiguous.
    values=set()
    for m in re.finditer(rf'{Q}\s+(?:is\s+)?({REL})(?:\s+to)?\s+{Q}',text,re.I):
        if safe_assertion(text,m.start()):values.add(base.relation_key(m[1]))
    for m in re.finditer(rf'\blabel(?:ed|led)?(?:\s+(?:on\s+this\s+arrow\s+is|with\s+the\s+text|with\s+the\s+label|with))?\s+({Q})',text,re.I):
        key=base.relation_key(label(m[1]))
        if key in base.RELATION_VOCABULARY and safe_assertion(text,m.start()):values.add(key)
    for m in re.finditer(rf'\bone\s+is\s+an?\s+({REL})\s+relationship|\bother\s+is\s+an?\s+({REL})\s+relationship',plain,re.I):
        values.add(base.relation_key(next(v for v in m.groups() if v)))
    if len(values)==1:return committed(values.pop(),'one_asserted_relation',text)
    return Extraction(None,'AMBIGUOUS' if len(values)>1 else 'NO_FINAL_CLAIM','no_unique_relation',text)


def path(text, explicit):
    text=text.replace('**','')
    plain=clean(text)
    # Explicit absence takes precedence over failed attempts in the reasoning.
    if re.search(r'\b(?:no\s+(?:valid\s+|shortest\s+)?path|(?:path|route)\s+does\s+not\s+exist|cannot\s+(?:find|reach))\b',plain,re.I):
        return committed([],'explicit_no_path',text)
    canonical=re.search(rf'\bshortest\s+(?:path|sequence)(?:\s+from\s+{Q}\s+to\s+{Q})?\s+is\s*:\s*([^\n]+)',text,re.I)
    if canonical:
        payload=canonical[1].strip().rstrip('.')
        parts=re.split(r'\s*(?:->|→|<-|←)\s*',payload)
        return committed([label(p) for p in parts] if len(parts)>1 else list_payload(payload),'path_template',text)
    candidates=[]
    for line in text.splitlines():
        if not re.search(r'->|→|<-|←',line):continue
        value=clean(line)
        if re.search(r'\?|\b(?:what\s+if|perhaps|maybe|let.s\s+check|is\s+there|that\s+doesn.t|not\s+(?:a\s+)?path)\b',value,re.I):continue
        if not (explicit or re.search(r'\b(?:path|route|sequence|chain)\b',value,re.I) or re.fullmatch(r'\s*(?:\d+[.)]\s*)?(?:'+Q+r'|[\w [\],.\'-]+)(?:\s*(?:->|→)\s*(?:'+Q+r'|[\w [\],.\'-]+))+\s*',line)):
            continue
        if ':' in value:value=value.split(':',1)[1].strip()
        value=re.sub(r'^.*?\b(?:path|chain|sequence|route)\s+is\s+','',value,flags=re.I)
        value=re.sub(r'^\s*\d+[.)]\s*','',value)
        parts=[label(p.strip().rstrip('.')) for p in re.split(r'\s*(?:->|→|<-|←)\s*',value)]
        if all(parts) and not RETRACTION.search(plain[plain.find(clean(line))+len(clean(line)):]):
            candidates.append(parts)
    if candidates:return committed(candidates[-1],'asserted_path',text)
    if explicit and plain and not UNKNOWN.fullmatch(plain):
        bullet_values=[list_item(m[1]) for line in text.splitlines() if (m:=BULLET.match(line))]
        if bullet_values and all(len(x)==1 for x in bullet_values):
            return committed([x[0] for x in bullet_values],'numbered_path',text)
        for line in text.splitlines():
            if ',' in line and not re.search(r'\b(?:is|are|check|nodes|path)\b|[?!]',line,re.I):return committed(list_payload(line),'comma_path',text)
    return Extraction(span=text)


def extract(task, response):
    """Return one task assertion using only task and response, without gold."""
    base.require_task(task,TASK_TYPES)
    text, explicit=answer_scope('' if response is None else str(response))
    if not text or UNKNOWN.fullmatch(clean(text)):return Extraction(span=text)
    if task in base.NUMERIC_TASKS:return numeric(task,text,explicit)
    if task=='highest_node_degree':return highest(text,explicit)
    if task in ('node_description','neighbor_listing'):return nodes(task,text)
    if task=='triple_listing':return triples(text)
    if task=='relation_identification':return relation(text)
    return path(text,explicit)


def score_record(record, response, meta, extractor=VERSION):
    if extractor!=VERSION:raise ValueError(f'This scorer only supports {VERSION}')
    response='' if response is None else str(response).strip()
    task=record['task_type'];parsed=extract(task,response)
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
        answer_span='Last explicit Answer/Final Answer block. Otherwise task-specific assertions and literal lists; no last-number or gold-endpoint fallback.',
        numeric_parse='Only asserted task totals/degrees or a bare integer. Candidate list indices and unfinished reasoning are not answers.',
        highest_node_degree_extraction='Pair a selected node with its asserted degree; candidate degree tables alone make no final selection.',
        strict='Raw normalization; highest-degree strict_accuracy requires BOTH a selected maximum-degree node and its correct degree. strict_degree_accuracy remains diagnostic.',
        literal_list_labels='Quoted/bold/backtick and plain lists retain literal labels, including stopwords and relation-name nodes. Explanation suffixes are syntax.',
        triple_extraction='Complete literal parenthesized, formatted, hyphen-separated and predicate-separated triples; malformed closed tuples incur FP penalties; incomplete tails are separately diagnosed.',
        relation_identification='Explicit label assertion or one asserted relation triple; unknown asserted labels preserved; absence distinct from candidates and no final claim.',
        shortest_path_listing='One asserted ordered path; no graph solving or gold-endpoint selection. Explicit no-path claim overrides tentative traces.',
        extraction_decision='COMMITTED, NO_FINAL_CLAIM or AMBIGUOUS; missing claims never receive strict exact credit.')
    return metrics,rows
