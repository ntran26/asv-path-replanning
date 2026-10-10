# ts4_r2_c21: interrupted, not reused

This run stopped after 24 of 25 completed episodes when the evaluation container was paused and restarted on 2026-10-10 (machine uptime reset at about 18:12 UTC; the runner process was lost mid-episode). There is no `completion.json`, so the paired report refuses this directory. Its files are kept unchanged as a record. Following the no-retry rule, the tag is not resumed or overwritten. The same 25 scenarios run again from scratch under the new tag `ts4_r2_c21b`, with the same code, options and selection.
