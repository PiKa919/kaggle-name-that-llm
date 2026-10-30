"""Authoritative score monitor for the two scheduled submissions.

The v13 and v12-fast jobs each poll ``submissions[0]``, which is the *newest*
submission and can therefore be the other job's entry.  This monitor identifies
each submission by its description and prints an unambiguous pair of scores.
"""
import time
from kaggle.api.kaggle_api_extended import KaggleApi

TARGETS = {
    'v13': 'v13 curated-pool',
    'v12fast': 'v12 fast',
}
DEADLINE = time.time() + 30 * 3600


def main():
    api = KaggleApi()
    api.authenticate()
    print('MONITOR_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
    last = {}
    while time.time() < DEADLINE:
        try:
            subs = api.competition_submissions('name-that-llm') or []
        except Exception as exc:
            print('MONITOR_ERROR', type(exc).__name__, str(exc)[:300], flush=True)
            time.sleep(300)
            continue

        done = 0
        for name, needle in TARGETS.items():
            matches = [s for s in subs if needle in str(getattr(s, 'description', ''))]
            if not matches:
                print(f'MONITOR {time.strftime("%H:%M:%SZ", time.gmtime())} {name} MISSING', flush=True)
                continue
            s = matches[0]  # list is newest first
            status = str(s.status)
            line = (f'MONITOR {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {name} '
                    f'ref={s.ref} status={status} public={s.public_score!r} '
                    f'private={s.private_score!r} error={s.error_description!r}')
            if line != last.get(name):
                print(line, flush=True)
                last[name] = line
            if 'COMPLETE' in status or 'ERROR' in status:
                done += 1
            elif 'PENDING' in status or 'RUNNING' in status:
                pass

        if done == len(TARGETS):
            print('MONITOR_ALL_DONE', flush=True)
            return
        time.sleep(300)

    print('MONITOR_TIMEOUT', flush=True)


if __name__ == '__main__':
    main()
