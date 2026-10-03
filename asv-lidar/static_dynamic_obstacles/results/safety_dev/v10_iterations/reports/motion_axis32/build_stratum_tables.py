"""Derive tables from this completed validated report; no simulator imports."""
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
sys.path.insert(0, str(ROOT / 'tools/diagnostics/safety'))
from suite_status import read_shared_bytes
from report_safety_iteration import paired_counts

inputs = {name: read_shared_bytes(OUT/name) for name in ('paired.csv', 'summary.json', 'provenance.json')}
summary = json.loads(inputs['summary.json'])
rows = list(csv.DictReader(io.StringIO(inputs['paired.csv'].decode('utf-8'))))
assert summary['completed_iteration_records'] == len(rows) == 32
assert len({(r['case'], r['mode']) for r in rows}) == len(rows)
assert all(r['off_evidence'] == 'fresh_prior_pilot' for r in rows)
tables = []
for stratum in ['all'] + sorted({r['stratum'] for r in rows}):
    selected = rows if stratum == 'all' else [r for r in rows if r['stratum'] == stratum]
    counts = paired_counts(selected, 'off')
    tables.append(dict(stratum=stratum, evidence='fresh_prior_pilot', matched=counts['matched_cases'],
        sac_goals=counts['reference_goals'], preserved_sac_goals=counts['preserved_reference_goals'],
        broken_sac_goals=counts['lost_goals'], sac_failures=counts['reference_failures'],
        rescued_sac_failures=counts['gained_goals'], unresolved_sac_failures=counts['unresolved_reference_failures']))
broken = [{k: r[k] for k in ('case', 'mode', 'stratum', 'outcome', 'off_evidence', 'off_source')}
          for r in rows if r['off_outcome'] == 'goal' and r['outcome'] != 'goal']
for filename, values, columns in [('stratum_preservation.csv', tables, list(tables[0])),
    ('broken_sac_successes.csv', broken, ['case', 'mode', 'stratum', 'outcome', 'off_evidence', 'off_source'])]:
    with (OUT/filename).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(values)
lines = ['# V16 preservation and rescue by selection stratum', '',
    'Derived from the completed, fully validated report in this directory. SAC references are the prior fresh matched pilot episodes.', '',
    '| Stratum | Matched | Preserved / SAC goals | Rescued / SAC failures | Broken | Unresolved |',
    '|---|---:|---:|---:|---:|---:|']
for row in tables:
    lines.append(f"| {row['stratum']} | {row['matched']} | {row['preserved_sac_goals']}/{row['sac_goals']} | {row['rescued_sac_failures']}/{row['sac_failures']} | {row['broken_sac_goals']} | {row['unresolved_sac_failures']} |")
lines += ['', 'Broken SAC successes:', '', *[f"- {r['case']}: {r['outcome']} ({r['stratum']})." for r in broken], '',
    'The cohort is outcome-enriched development data. These counts do not estimate population performance or establish a safety guarantee.', '']
with (OUT/'stratum_preservation.md').open('x', encoding='utf-8', newline='\n') as handle:
    handle.write('\n'.join(lines))
provenance = dict(derivation='Completed validated summary/paired records only; original report files preserved.',
    input_sha256={k: hashlib.sha256(v).hexdigest() for k,v in inputs.items()},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    count_helper_sha256=hashlib.sha256((ROOT/'tools/diagnostics/safety/report_safety_iteration.py').read_bytes()).hexdigest(),
    new_episodes=0)
with (OUT/'stratum_provenance.json').open('x', encoding='utf-8', newline='\n') as handle:
    json.dump(provenance, handle, indent=2); handle.write('\n')
print(json.dumps({'overall':tables[0], 'broken':broken}))
