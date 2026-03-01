## Decode Log File

### log_parser.py


### Render: log_viewer.py


## Simulate UDP Server

### fake_vessel_replay.py

```
python fake_vessel_replay.py --log data/test_1.log --bind-ip 0.0.0.0 --port 5050
```

This script simulates the vessel telemetry server by replaying a recorded log file over UDP. It waits for a handshake message to start communicating with the client, then streams log lines.

### Imports

- **socket**: UDP networking
- **time**: sleep between lines to match timing
- **re**: parse data from log
- **argparse**: command line flags

```python
import socket
import time
import re
import argparse
```

### 



