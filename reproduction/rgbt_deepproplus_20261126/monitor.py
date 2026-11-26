"""Keep an on-disk status while detached jobs continue beyond the terminal."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

RUN = Path(__file__).resolve().parent


def main():
    while True:
        snapshot = {'updated_at': datetime.now(timezone.utc).isoformat(), 'jobs': {}}
        active = False
        for phase in ['train', 'released']:
            name = 'deeppro-rgbt-' + phase + '-20261126'
            process = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name],
                                     capture_output=True, text=True, timeout=20)
            state = json.loads(process.stdout) if process.returncode == 0 else {'Status': 'missing'}
            path = RUN / (phase + '_status.json')
            pipeline = json.loads(path.read_text()) if path.exists() else {'state': 'not_started'}
            snapshot['jobs'][phase] = {'container': state, 'pipeline_state': pipeline['state']}
            if state.get('Running'):
                active = True
            elif not (RUN / (phase + '_console.log')).exists():
                with (RUN / (phase + '_console.log')).open('w') as file:
                    subprocess.run(['docker', 'logs', name], stdout=file, stderr=subprocess.STDOUT,
                                   timeout=30, check=False)
        temp = RUN / 'STATUS.json.tmp'
        temp.write_text(json.dumps(snapshot, indent=2) + '\n')
        temp.replace(RUN / 'STATUS.json')
        subprocess.run([sys.executable, str(RUN / 'report.py')], check=True, timeout=20)
        if not active:
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
