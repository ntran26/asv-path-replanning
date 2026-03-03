## Decode Log File

### File: log_parser.py
This file containts helper functions that parse and decode lines from the log file.

### Imports
```python
from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple, Dict, Any
import time
```

- **re**: regex parsing log lines
- **numpy**: for lidar arrays and math
- **dataclass**: define frame object
- **typing**: hints for data types (for readability and debugging)

### Bluefin frame data container
```python
@dataclass
class BluefinFrame:
    t_sec: float
    ts_str: str

    x_m: float
    y_m: float
    yaw_deg: float

    vx_mps: float
    vy_mps: float
    speed_mps: float

    lidar_m: np.ndarray

    hdg_ref_deg: Optional[float] = None
    s1: Optional[int] = None
    s2: Optional[int] = None
    seq: Optional[int] = None
```
A Bluefin frame is the decoded state snapshot produced each time a LiDAR scan line arrives.

It includes:
- Time:
    - **ts_str**: original timestamp string from the log (13:32:07.817313)
    - **t_sec**: time since start of run (relative time, seconds)
- Pose:
    - **x_m, y_m, yaw_deg**: pose in meters and heading/yaw angle in degrees
- Derived velocity:
    - **vx_mps, vy_mps**: computed from pose vs time
    - **speed_mps**: speed magnitude (hypot)
- Sensor:
    - **lidar_m**: LiDAR ranges in meters (numpy array)
- Latched metadata (optional):
    - **hdg_ref_deg**: reference heading (HDG)
    - **s1, s2**: rudder, thruster values
    - **seq**: sequence number if present


### Render: log_viewer.py


## Simulate UDP Server

### fake_vessel_replay.py

```
python fake_vessel_replay.py --log data/test_1.log --bind-ip 0.0.0.0 --port 5050
```

This script simulates the vessel telemetry server by replaying a recorded log file over UDP. It waits for a handshake message to start communicating with the client, then streams log lines.

### Imports

```python
import socket
import time
import re
import argparse
```

- **socket**: UDP networking
- **time**: sleep between lines to match timing
- **re**: parse data from log
- **argparse**: command line flags

### 



