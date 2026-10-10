"""Frozen forecast-error allowance tables for the experimental V20 filter.

Values are split-conformal 95 % per-decision quantiles of a hull-point
displacement score, calibrated by gate G1 on saved V14/V15/V16 decision
snapshots of the 32- and 40-case development cohorts against later onboard
estimates (never truth). Rows are (horizon_s, allowance_m), monotone in the
horizon; lookups use the next tabulated horizon and extrapolate linearly from
the last two rows. Source artifact and procedure:
results/safety_dev/v20_development/allowance_calibration/g1_v1/calibration.json
(SHA-256 below), tools/diagnostics/safety/v20_calibrate_allowances.py.

OWN_TABLE: own-ship pose forecast with the issued commands, score
|dp| + R |dpsi| with R the inflated hull half-diagonal. The filter subtracts
only the excess over the existing 0.15 m hull margin.
TARGET_TABLE: constant-velocity forecast of V16 track views of targets in
constant-velocity scenarios. Held-out truth coverage of this table is poor at
short horizons (onboard track estimates are themselves biased), so it is a
declared contract, not a validated bound.
"""

CALIBRATION_ARTIFACT = "results/safety_dev/v20_development/allowance_calibration/g1_v1/calibration.json"
CALIBRATION_SHA256 = "bfd5ccae96867b94685eaaf329403cea0372aabab0f9cfddcff004adb7bfa640"
LEVEL = 0.95

OWN_TABLE = (
    (0.5, 0.1269),
    (1.0, 0.1767),
    (1.5, 0.2608),
    (2.0, 0.3661),
    (2.5, 0.4841),
    (3.0, 0.5982),
    (3.5, 0.7352),
    (4.0, 0.8758),
    (4.5, 1.0224),
    (5.0, 1.1748),
    (5.5, 1.3272),
    (6.0, 1.4864),
    (6.5, 1.6664),
    (7.0, 1.8166),
    (7.5, 1.9755),
    (8.0, 2.1445),
    (8.5, 2.3121),
    (9.0, 2.4753),
    (9.5, 2.6318),
    (10.0, 2.7936),
    (10.5, 2.9193),
    (11.0, 3.0585),
    (11.5, 3.2033),
    (12.0, 3.3527),
)

TARGET_TABLE = (
    (0.5, 0.5658),
    (1.0, 0.7601),
    (1.5, 0.8882),
    (2.0, 1.0216),
    (2.5, 1.169),
    (3.0, 1.2725),
    (3.5, 1.3717),
    (4.0, 1.5255),
    (4.5, 1.6157),
    (5.0, 1.7668),
    (5.5, 1.943),
    (6.0, 2.1029),
    (6.5, 2.257),
    (7.0, 2.443),
    (7.5, 3.0004),
    (8.0, 3.392),
    (8.5, 3.7758),
    (9.0, 4.2017),
    (9.5, 4.219),
    (10.0, 4.219),
    (10.5, 4.5454),
    (11.0, 5.0186),
    (11.5, 5.2588),
    (12.0, 5.2588),
    (12.5, 5.2588),
    (13.0, 5.3362),
    (13.5, 5.3362),
    (14.0, 5.4576),
    (14.5, 5.5643),
    (15.0, 5.7859),
    (15.5, 5.9687),
    (16.0, 6.0933),
    (16.5, 6.0933),
    (17.0, 6.3568),
    (17.5, 6.396),
    (18.0, 6.396),
    (18.5, 6.396),
    (19.0, 6.396),
    (19.5, 6.396),
    (20.0, 6.396),
)
