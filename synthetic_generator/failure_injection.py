"""Synthetic fault / degradation mechanisms.

Faults change physical behaviour (applied while signals are generated, so they
propagate to dependent parameters) or sensor behaviour (applied to the measured
value only). Labels are never written into telemetry.
"""
from dataclasses import dataclass, field

import numpy as np

from .flight_profiles import DT, lag


@dataclass
class Mechanism:
    code: str  # FTA basic-event code
    name: str
    lru: str
    description: str
    effects: list  # (param, mode, magnitude, ctx) ; mode in add|noise|lag|rate
    action: str  # corrective action
    symptoms: list  # natural-language symptom phrases (paraphrased later)
    observed: str  # parameter most often quoted in snags
    intermittent: bool = False
    sensor: bool = False
    extra: dict = field(default_factory=dict)


M = Mechanism
MECHANISMS = [
    M("BE-ENG-EMU-01", "BEARING_DEGRADATION", "EMU", "Engine bearing wear / spalling",
      [("ENGINE_BEARING_VIBRATION_1", "add", 9.0, "rpm2"), ("ENGINE_BEARING_VIBRATION_2", "add", 2.5, "rpm2"),
       ("ENGINE_BEARING_TEMPERATURE_1", "add", 22.0, "rpm"), ("ENGINE_OIL_TEMPERATURE", "add", 6.0, "rpm")],
      "Inspect engine bearing No.1, perform oil debris analysis, replace bearing assembly if spalling confirmed",
      ["high engine vibration", "bearing vibration exceedance", "rumble felt at high power", "vibration indication rising with RPM",
       "No.1 bearing temperature high"], "ENGINE_BEARING_VIBRATION_1"),
    M("BE-ENG-EMU-02", "LUBE_PRESSURE_LOSS", "EMU", "Engine oil pump wear / lubrication pressure loss",
      [("ENGINE_OIL_PRESSURE", "add", -140.0, "rpm"), ("ENGINE_LUBE_FLOW", "add", -14.0, "rpm"),
       ("ENGINE_OIL_TEMPERATURE", "add", 12.0, "eng_warm")],
      "Check oil level and filter, inspect oil pump, replace oil pump if output pressure below test limit",
      ["oil pressure low", "low oil press caution", "oil pressure fluctuating at idle", "oil temperature rising with low pressure"],
      "ENGINE_OIL_PRESSURE"),
    M("BE-FUL-FPA-01", "FUEL_PUMP_DEGRADATION", "FPA", "Fuel boost pump wear",
      [("FUEL_PUMP_PRESSURE", "add", -110.0, "rpm"), ("FUEL_PUMP_CURRENT", "add", 6.0, "eng_on"),
       ("FUEL_PUMP_TEMPERATURE", "add", 16.0, "eng_warm"), ("ENGINE_FUEL_PRESSURE", "add", -55.0, "rpm")],
      "Perform fuel boost pump output test, replace fuel pump assembly", ["fuel pump pressure low", "boost pump press dropping",
       "fuel press caution at high power", "fuel pump running hot"], "FUEL_PUMP_PRESSURE"),
    M("BE-FUL-FPA-02", "FUEL_FILTER_RESTRICTION", "FPA", "Fuel filter clogging",
      [("FUEL_FILTER_PRESSURE_DROP", "add", 38.0, "rpm"), ("FUEL_PUMP_PRESSURE", "add", -25.0, "rpm")],
      "Replace fuel filter element, sample fuel for contamination", ["fuel filter bypass indication", "fuel filter delta P high",
       "filter impending bypass"], "FUEL_FILTER_PRESSURE_DROP"),
    M("BE-HYD-HPA-01", "HYD_PUMP_A_DEGRADATION", "HPA", "Hydraulic pump A internal leakage / wear",
      [("HYD_A_PRESSURE", "add", -60.0, "hyd"), ("HYD_PUMP_CURRENT", "add", 9.0, "eng_on"),
       ("HYD_PUMP_TEMPERATURE", "add", 20.0, "eng_warm"), ("HYD_FLOW_RATE", "add", -9.0, "hyd"),
       ("ACTUATOR_POSITION", "lag", 0.25, None)],
      "Check case drain flow of hydraulic pump A, replace pump A if case drain exceeds limit",
      ["HYD A pressure fluctuating", "hydraulic system A pressure low", "pump A press drops during manoeuvre",
       "HYD-A pump hot", "sluggish control response with HYD A caution"], "HYD_A_PRESSURE"),
    M("BE-HYD-HPB-01", "HYD_PUMP_B_DEGRADATION", "HPB", "Hydraulic pump B internal leakage / wear",
      [("HYD_B_PRESSURE", "add", -60.0, "hyd"), ("HYD_B_PUMP_CURRENT", "add", 9.0, "eng_on"),
       ("HYD_B_PUMP_TEMPERATURE", "add", 20.0, "eng_warm")],
      "Check case drain flow of hydraulic pump B, replace pump B if case drain exceeds limit",
      ["HYD B pressure low", "hydraulic system B pressure fluctuating", "pump B running hot"], "HYD_B_PRESSURE"),
    M("BE-HYD-HMU-01", "HYD_FILTER_RESTRICTION", "HMU", "Hydraulic return filter clogging",
      [("HYD_FILTER_PRESSURE_DROP", "add", 12.0, "hyd"), ("HYD_RESERVOIR_TEMPERATURE", "add", 6.0, "eng_warm")],
      "Replace hydraulic return filter element, check fluid contamination", ["hyd filter clog indicator popped",
       "hydraulic filter delta P high"], "HYD_FILTER_PRESSURE_DROP"),
    M("BE-HYD-HMU-02", "HYD_VALVE_DEGRADATION", "HMU", "Hydraulic priority valve sticking",
      [("HYD_VALVE_POSITION", "noise", 3.0, None), ("HYD_VALVE_POSITION", "add", -8.0, "eng_on"),
       ("HYD_ACTUATOR_PRESSURE", "noise", 4.0, None)],
      "Clean / replace hydraulic priority valve", ["hyd valve position erratic", "priority valve chatter"], "HYD_VALVE_POSITION"),
    M("BE-FCS-SAA-01", "ACTUATOR_DEGRADATION", "SAA", "Control surface actuator seal wear",
      [("ACTUATOR_POSITION", "lag", 0.45, None), ("ELEVATOR_POSITION", "lag", 0.35, None),
       ("ACTUATOR_CURRENT", "add", 3.0, "eng_on"), ("ACTUATOR_TEMPERATURE", "add", 14.0, "eng_warm")],
      "Perform actuator frequency-response test, replace actuator seals or actuator",
      ["actuator position error high", "elevator response sluggish", "FCS actuator monitor trip", "surface lag noted"],
      "POSITION_ERROR"),
    M("BE-ELE-PDU-01", "ELECTRICAL_INSTABILITY", "PDU", "Power distribution contactor / feeder degradation",
      [("DC_BUS_VOLTAGE", "noise", 0.7, None), ("AC_BUS_VOLTAGE", "noise", 2.2, None), ("BUS_FREQUENCY", "noise", 2.5, None)],
      "Inspect PDU contactors and feeder terminations, retorque / replace contactor",
      ["DC bus voltage fluctuation", "electrical flicker", "bus voltage unstable", "AC bus transient"], "DC_BUS_VOLTAGE",
      intermittent=True),
    M("BE-ELE-BAT-01", "BATTERY_DEGRADATION", "BAT", "Battery cell capacity loss",
      [("BATTERY_VOLTAGE", "add", -1.8, None), ("BATTERY_TEMPERATURE", "add", 9.0, None), ("BATTERY_SOC", "add", -14.0, None)],
      "Perform battery capacity check, replace battery unit", ["battery voltage low on start", "BATT caution", "battery hot"],
      "BATTERY_VOLTAGE"),
    M("BE-ELE-GCU-01", "GENERATOR_DEGRADATION", "GCU", "Generator winding / regulator degradation",
      [("GENERATOR_VOLTAGE", "add", -6.0, "eng_on"), ("GENERATOR_VOLTAGE", "noise", 1.4, None),
       ("GENERATOR_FREQUENCY", "noise", 3.5, None), ("GENERATOR_TEMPERATURE", "add", 28.0, "eng_warm")],
      "Check generator output on ground run, inspect GCU regulator, replace generator", ["generator voltage low",
       "GEN caution flickering", "generator overheat", "AC voltage fluctuating"], "GENERATOR_VOLTAGE"),
    M("BE-ENV-ACU-01", "COOLING_DEGRADATION", "ACU", "Avionics cooling fan bearing wear",
      [("COOLING_FAN_RPM", "add", -2600.0, "eng_on"), ("COOLING_FAN_CURRENT", "add", 2.2, "eng_on")],
      "Inspect avionics cooling fan, replace fan assembly", ["avionics cooling fan noisy", "equipment bay temperature high",
       "cooling flow low caution", "avionics bay overheat"], "COOLING_FLOW_RATE"),
    M("BE-COM-CTR-01", "COMMUNICATION_DEGRADATION", "CTR", "Transceiver RF front-end / antenna connector degradation",
      [("SIGNAL_TO_NOISE_RATIO", "add", -20.0, None), ("RX_POWER", "add", -9.0, None)],
      "Inspect antenna connector and coax, perform transceiver BITE and RF power test, replace transceiver",
      ["comm link intermittent", "radio dropouts", "data link loss reported", "poor reception", "link quality degraded"],
      "SIGNAL_TO_NOISE_RATIO", intermittent=True),
    M("BE-AVI-MCP-01", "PROCESSOR_OVERHEATING", "MCP", "Mission computer thermal interface degradation",
      [("CPU_TEMPERATURE", "add", 24.0, "eng_warm"), ("BOARD_TEMPERATURE", "add", 11.0, "eng_warm"),
       ("PROCESS_LATENCY", "add", 6.0, None), ("WATCHDOG_COUNT", "rate", 0.004, None)],
      "Inspect mission computer heat sink and thermal pads, replace processor module", ["MC overtemp", "mission computer hot",
       "display freeze with watchdog reset"], "CPU_TEMPERATURE"),
    M("BE-AVI-MCP-02", "MEMORY_DEGRADATION", "MCP", "Mission computer memory module degradation",
      [("MEMORY_USAGE", "add", 28.0, None), ("BIT_ERROR_COUNT", "rate", 0.05, None), ("RESTART_COUNT", "rate", 0.0008, None),
       ("MESSAGE_ERROR_COUNT", "rate", 0.02, None)],
      "Run extended memory BITE, replace memory module", ["MC restart in flight", "memory BIT fail", "mission computer reboot"],
      "BIT_ERROR_COUNT"),
    M("BE-NAV-INU-01", "NAV_DEGRADATION", "INU", "Inertial / GNSS receiver degradation",
      [("NAV_POSITION_ERROR", "add", 55.0, None), ("NAV_VELOCITY_ERROR", "add", 2.5, None),
       ("GPS_SIGNAL_QUALITY", "add", -25.0, None), ("GPS_SATELLITE_COUNT", "add", -4.0, None)],
      "Perform INU alignment test, check GNSS antenna, replace INU", ["nav position error large", "GPS degraded",
       "INU drift", "nav accuracy poor"], "NAV_POSITION_ERROR"),
]

# Sensor fault mechanisms: applied to the measured value of `target`
SENSOR_KINDS = {
    "SENSOR_DRIFT": ("drift", "sensor drift", "Recalibrate / replace {p} sensor", ["{p} reading drifting", "{p} disagrees with cross-check", "{p} indication slowly off"]),
    "SENSOR_BIAS": ("bias", "sensor bias", "Recalibrate {p} sensor, check connector", ["{p} reading offset", "{p} indication biased", "{p} disagree"]),
    "SENSOR_STUCK": ("stuck", "sensor stuck-at", "Replace {p} sensor", ["{p} indication frozen", "{p} not changing", "{p} stuck"]),
    "SENSOR_INTERMITTENT": ("intermittent", "sensor intermittent", "Inspect wiring / connector of {p} sensor", ["{p} indication intermittent", "{p} dropouts", "{p} flickering"]),
}
SENSOR_TARGETS = ["ENGINE_OIL_PRESSURE", "BAROMETRIC_ALTITUDE", "HYD_A_PRESSURE", "ENGINE_EXHAUST_TEMPERATURE",
                  "COOLING_TEMPERATURE", "FUEL_TANK_TEMPERATURE", "CABIN_PRESSURE", "BATTERY_TEMPERATURE"]


def sensor_mechanism(kind, target, catalog):
    p = catalog[target]
    mode, desc, action, symptoms = SENSOR_KINDS[kind]
    code = f"BE-{p.lru}-S{list(SENSOR_KINDS).index(kind) + 1}"
    label = target.replace("_", " ").lower()
    return M(code, kind, p.lru, f"{p.lru} {desc}", [(target, mode, 0.05 * (p.hi - p.lo), None)],
             action.format(p=label), [s.format(p=label) for s in symptoms], target, sensor=True,
             intermittent=(mode == "intermittent"))


def all_mechanisms(catalog):
    """Every mechanism incl. one generic sensor-fault event per (kind, LRU) used in the FTA."""
    out = {m.code: m for m in MECHANISMS}
    for lru in sorted({p.lru for p in catalog.values()}):
        target = next(n for n, p in catalog.items() if p.lru == lru and p.kind == "analog")
        for kind in SENSOR_KINDS:
            m = sensor_mechanism(kind, target, catalog)
            m.description = f"{lru} sensor: {SENSOR_KINDS[kind][1]}"
            out[m.code] = m
    return out


TRAJECTORIES = ["linear", "exponential", "piecewise", "accelerating", "intermittent", "step"]


def severity(traj, x):
    """Severity in [0, inf) as a function of normalised life x (x=1 -> failure)."""
    x = np.maximum(np.asarray(x, float), 0)
    if traj == "exponential":
        return (np.exp(3 * x) - 1) / (np.exp(3) - 1)
    if traj == "piecewise":
        return np.where(x < 0.6, 0.5 * x, 0.3 + 0.7 * (x - 0.6) / 0.4)
    if traj == "accelerating":
        return x ** 2.5
    if traj == "step":
        return np.where(x < 0.3, 0.0, 0.45 + 0.55 * (x - 0.3) / 0.7)
    return x  # linear, intermittent


STATES = [(0.0, "HEALTHY"), (0.05, "EARLY"), (0.25, "MILD"), (0.45, "MODERATE"), (0.7, "SEVERE"), (1.0, "FAILURE")]


def degradation_state(s):
    state = "HEALTHY"
    for thr, name in STATES:
        if s >= thr and (thr > 0 or s > 0):
            state = name
    return state


def ctx_factor(d, ctx):
    if ctx is None:
        return 1.0
    if ctx == "rpm":
        return d.rpm / 0.8
    if ctx == "rpm2":
        return (d.rpm / 0.8) ** 2
    if ctx == "hyd":
        return d.eng_on * (0.4 + d.hyd_dem)
    return d[ctx]


def episode_mask(rng, n, s_mean):
    """Random on/off episodes, more frequent with severity (intermittent faults)."""
    mask = np.zeros(n)
    k = rng.poisson(1 + 8 * s_mean)
    for _ in range(k):
        c, w = rng.integers(0, n), int(rng.uniform(5, 40) / DT)
        mask[c:c + w] = 1
    return mask


def physical_effect(rng, d, active, pname, value):
    """Add physical fault effects for parameter `pname`. active: list of (mechanism, s(t) array)."""
    extra_tau = 0.0
    for mech, s in active:
        if mech.sensor:
            continue
        gate = episode_mask(rng, len(s), s.mean()) if mech.intermittent else 1.0
        for p, mode, mag, ctx in mech.effects:
            if p != pname:
                continue
            if mode == "add":
                value = value + mag * np.minimum(s, 1.2) * ctx_factor(d, ctx) * gate
            elif mode == "noise":
                value = value + rng.normal(0, 1, len(s)) * mag * np.minimum(s, 1.2) * gate
            elif mode == "lag":
                extra_tau += mag * float(np.minimum(s, 1.2).mean())
            elif mode == "rate":
                value = value + mag * np.minimum(s, 1.2)
    return value, extra_tau


def sensor_effect(rng, active, pname, measured):
    for mech, s in active:
        if not mech.sensor:
            continue
        for p, mode, mag, _ in mech.effects:
            if p != pname:
                continue
            sc = np.minimum(s, 1.2)
            if mode == "drift":
                measured = measured + mag * sc
            elif mode == "bias":
                measured = measured + mag * (sc > 0.05) * (0.3 + 0.7 * sc)
            elif mode == "stuck" and sc.max() > 0.3:
                mask = episode_mask(rng, len(s), sc.mean()) > 0 if sc.max() < 1 else np.ones(len(s), bool)
                idx = np.where(mask)[0]
                if len(idx):
                    starts = idx[np.r_[True, np.diff(idx) > 1]]
                    for st in starts:
                        run = idx[(idx >= st)]
                        run = run[: np.argmax(np.diff(np.r_[run, -10]) != 1) + 1]
                        measured[run] = measured[st]
            elif mode == "intermittent" and sc.max() > 0.2:
                mask = episode_mask(rng, len(s), sc.mean()) > 0
                measured = np.where(mask & (rng.random(len(s)) < 0.5), measured + mag * rng.choice([-1, 1], len(s)), measured)
    return measured


__all__ = ["MECHANISMS", "SENSOR_KINDS", "SENSOR_TARGETS", "TRAJECTORIES", "severity", "degradation_state",
           "physical_effect", "sensor_effect", "sensor_mechanism", "all_mechanisms", "lag"]
