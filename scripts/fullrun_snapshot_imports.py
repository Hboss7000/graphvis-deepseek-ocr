"""Audit reachable imports in actual offline pinned remote code, without executing it."""
import ast
import json
from pathlib import Path
import sys
from fullrun_common import sha


def snapshot_imports(snapshot):
    snapshot = Path(snapshot)
    config = json.loads((snapshot / 'config.json').read_text())
    mapping = config.get('auto_map')
    if not mapping or not list(snapshot.glob('modeling*.py')):
        raise ValueError('Pinned snapshot has no remote-code entry points')
    required, optional, sources = set(), set(), {}
    pending = []
    def local_file(parts):
        path = snapshot.joinpath(*parts)
        for candidate in (path.with_suffix('.py'), path / '__init__.py'):
            if candidate.is_file(): return candidate
        return None
    for values in mapping.values():
        for value in values if isinstance(values,(list,tuple)) else [values]:
            if not value: continue
            module = value.split('--')[-1].rsplit('.',1)[0]
            path = local_file(module.split('.'))
            if path is None: raise ValueError('Missing remote-code entry point: '+module)
            pending.append((path,False))
    def visit(node, path, conditional=False):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            for name in names:
                parts=name.split('.') if name else []
                if isinstance(node, ast.ImportFrom) and node.level:
                    parent=list(path.relative_to(snapshot).parts[:-1])
                    parent=parent[:len(parent)-node.level+1] if node.level>1 else parent
                    local=local_file(parent+parts)
                    if local: pending.append((local,conditional))
                    if not name:
                        for alias in node.names:
                            local=local_file(parent+[alias.name])
                            if local: pending.append((local,conditional))
                    continue
                local=local_file(parts)
                if local:
                    pending.append((local,conditional));continue
                top=name.split('.')[0]
                if top and top not in sys.stdlib_module_names:
                    (optional if conditional else required).add(top)
            return
        deferred = conditional or isinstance(node, (ast.Try, ast.If, ast.FunctionDef, ast.AsyncFunctionDef))
        for child in ast.iter_child_nodes(node): visit(child,path,deferred)
    visited=set()
    while pending:
        path,conditional=pending.pop();key=str(path.relative_to(snapshot))
        if (key,conditional) in visited:continue
        visited.add((key,conditional));sources[key]=sha(path)
        visit(ast.parse(path.read_text(),filename=str(path)),path,conditional)
    return {'snapshot':str(snapshot),'source_sha256':dict(sorted(sources.items())),
            'required':sorted(required),'conditional_or_deferred':sorted(optional-required),
            'note':'Traverse auto_map entry points and their local imports. Unconditional module-scope external imports are required; conditional/deferred imports are audited, not forced (eager attention preserved).'}
