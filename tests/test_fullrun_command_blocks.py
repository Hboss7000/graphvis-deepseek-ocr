"""Execute every documented block with local mock transports; no network or pod actions."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def test_all_first_session_blocks_run_independently_with_mock_transports(tmp_path):
    scripts=tmp_path/'bin';scripts.mkdir();remote=tmp_path/'remote_workspace'
    repo=remote/'bachelorArbeit';(repo/'scripts').mkdir(parents=True)
    (repo/'scripts/fullrun_manual.sh').write_text('exit 0\n')
    py=remote/'venvs/venv_qwen/bin/python';py.parent.mkdir(parents=True)
    py.write_text('#!/bin/bash\nexit 0\n');py.chmod(0o755)
    log=tmp_path/'calls.jsonl'
    def tool(name,body):
        p=scripts/name;p.write_text('#!'+sys.executable+'\n'+body);p.chmod(0o755)
    common='import os,sys,json,subprocess\nfrom pathlib import Path\nargs=sys.argv[1:]\nwith open(os.environ["MOCK_CALLS"],"a") as f:f.write(json.dumps([Path(sys.argv[0]).name,args])+"\\n")\n'
    tool('git',common+'\nif args[0]=="rev-parse":print("abc123")\nelif args[0]=="ls-remote":print("abc123\\trefs/heads/inference50-core-keep")\n')
    tool('ssh',common+'\ncommand=args[-1].replace("/workspace",os.environ["MOCK_WORKSPACE"])\nsubprocess.run(["bash","-n","-c",command],check=True)\nsubprocess.run(["bash","-c",command],check=True)\n')
    tool('tmux',common+'\nsubprocess.run(["bash","-n","-c",args[-1]],check=True)\n# Do not start a detached job, but validate the final command after all SSH/tmux quoting.\nassert "MAX_HOURS=3.3 bash scripts/fullrun_manual.sh llava" in args[-1]\nassert "read -r -p \\\"Session ended:" in args[-1]\n')
    tool('watch',common+'\nassert "fullrun_monitor.py --gpu" in args[-1]\n')
    tool('rsync',common+'\n')
    tool('runpodctl',common+'\nassert args==["pod","delete","--help"] or args==["pod","delete","test-pod"]\n')
    (tmp_path/'outputs').mkdir()
    env={**os.environ,'PATH':str(scripts)+os.pathsep+os.environ['PATH'],'MOCK_CALLS':str(log),'MOCK_WORKSPACE':str(remote)}
    blocks=re.findall(r'```bash\n(.*?)\n```',(ROOT/'scripts/FULLRUN.md').read_text(),re.S)
    assert len(blocks)==7
    for index,block in enumerate(blocks):
        assert all(name+'=' in block for name in ('SSH','RSYNC_SSH'))
        assert 'FETCH=' in block or re.search(r'read -rp .* FETCH',block)
        assert 'read -rp' in block and ' IP' in block and ' PORT' in block
        assert 'tail -F' not in block
        stdin='127.0.0.1\n22\n'
        if index==6:stdin+=str(tmp_path/'outputs')+'\ntest-pod\nTERMINATE test-pod\n'
        result=subprocess.run(['bash','-c',block],input=stdin,text=True,capture_output=True,cwd=tmp_path,env=env,timeout=10)
        assert result.returncode==0,(index+1,result.stdout,result.stderr)
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert any(name=='tmux' for name,_ in calls)
    assert any(name=='runpodctl' and args[-1]=='test-pod' for name,args in calls)
