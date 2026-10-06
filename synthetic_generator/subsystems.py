"""SYNTHETIC LRU and parameter catalog.

Every name, range, coefficient and address here is invented for software
development. None of it describes a real aircraft, ICD or engineering limit.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# code: (name, subsystem, 1553 remote-terminal address)
LRU_TYPES = {
    "EMU": ("Engine Monitoring Unit", "ENGINE", 1),
    "FMU": ("Fuel Management Unit", "FUEL", 2),
    "FPA": ("Fuel Pump Assembly", "FUEL", 3),
    "HPA": ("Hydraulic Pump A", "HYDRAULIC", 4),
    "HPB": ("Hydraulic Pump B", "HYDRAULIC", 5),
    "HMU": ("Hydraulic Management Unit", "HYDRAULIC", 17),
    "GCU": ("Generator Control Unit", "ELECTRICAL", 6),
    "BAT": ("Battery Unit", "ELECTRICAL", 7),
    "PDU": ("Power Distribution Unit", "ELECTRICAL", 8),
    "FCC": ("Flight Control Computer", "FLIGHT_CONTROL", 9),
    "SAA": ("Surface Actuator Assembly", "FLIGHT_CONTROL", 10),
    "ADC": ("Air Data Computer", "NAVIGATION", 11),
    "INU": ("Inertial Navigation Unit", "NAVIGATION", 12),
    "ECU": ("Environmental Control Unit", "ENVIRONMENTAL", 13),
    "ACU": ("Avionics Cooling Unit", "ENVIRONMENTAL", 14),
    "MCP": ("Mission Computer", "AVIONICS", 15),
    "CTR": ("Communication Transceiver", "COMMUNICATION", 16),
}

# Sampling classes (Hz). EVENT parameters are sent on change plus a heartbeat.
RATES = {"FAST": 5.0, "MEDIUM": 2.0, "SLOW": 0.5, "EVENT": 0.1}

# Every LRU log carries this bus-received context parameter; the product uses it to
# align LRU clocks with bus time.
LOG_SYNC_PARAMETER = "BAROMETRIC_ALTITUDE"


@dataclass
class Param:
    name: str
    lru: str
    unit: str
    rate: str
    lo: float
    hi: float
    f: Callable  # nominal value from context/other params
    tau: float = 0.0  # first-order lag (s)
    noise: float = 0.0  # white measurement noise sd
    src: str = "1553"  # "1553" or "LOG" (LRU internal log file)
    kind: str = "analog"  # analog | counter | discrete
    decimals: int = 2
    extra: dict = field(default_factory=dict)


def _p(*a, **k):
    return Param(*a, **k)


# Context series available in `d`: t, alt, spd, thr, rpm, load, man, amb_t, amb_p, hum,
# wind, gnd, stick, ped, elec, hyd_dem, eng_on, mach, aoa, roll, pitch, hdg, lat, lon,
# vs, cpu_dem, link_dist. Earlier parameters are also available by name.
CATALOG = [
    # ENGINE ---------------------------------------------------------------
    _p("ENGINE_THROTTLE_POSITION", "EMU", "%", "FAST", 0, 100, lambda d: 100 * d.thr, noise=0.2),
    _p("ENGINE_RPM", "EMU", "%", "FAST", 0, 110, lambda d: 100 * d.rpm, tau=1.0, noise=0.3),
    _p("ENGINE_CORE_RPM", "EMU", "%", "MEDIUM", 0, 110, lambda d: 0.97 * d.ENGINE_RPM + 2 * d.eng_on, tau=0.5, noise=0.3),
    _p("ENGINE_FAN_RPM", "EMU", "%", "MEDIUM", 0, 110, lambda d: 0.92 * d.ENGINE_RPM, tau=0.8, noise=0.3),
    _p("ENGINE_LOAD", "EMU", "%", "MEDIUM", 0, 120, lambda d: 100 * d.rpm ** 1.5 * (1 + 0.04 * (d.load - 1)), tau=1, noise=0.5),
    _p("ENGINE_TORQUE", "EMU", "kNm", "MEDIUM", 0, 30, lambda d: 0.22 * d.ENGINE_LOAD, noise=0.1),
    _p("ENGINE_FUEL_FLOW", "EMU", "kg/h", "MEDIUM", 0, 6000, lambda d: d.eng_on * (250 + 4800 * d.rpm ** 2.2) * (d.amb_p / 1013) ** 0.3, tau=1.5, noise=15),
    _p("ENGINE_FUEL_PRESSURE", "EMU", "kPa", "MEDIUM", 0, 1200, lambda d: d.eng_on * (300 + 600 * d.rpm), tau=1, noise=4),
    _p("ENGINE_OIL_PRESSURE", "EMU", "kPa", "MEDIUM", 0, 700, lambda d: d.eng_on * (150 + 380 * d.rpm), tau=2, noise=3),
    _p("ENGINE_OIL_TEMPERATURE", "EMU", "degC", "SLOW", -40, 200, lambda d: d.amb_g + d.eng_warm * (45 + 55 * d.rpm + 3 * (d.load - 1)), tau=90, noise=0.4),
    _p("ENGINE_EXHAUST_TEMPERATURE", "EMU", "degC", "MEDIUM", -40, 1100, lambda d: d.amb_t + d.eng_on * (330 + 520 * d.rpm ** 1.6 + 6 * (d.load - 1)), tau=3, noise=3),
    _p("ENGINE_TURBINE_TEMPERATURE", "EMU", "degC", "MEDIUM", -40, 1400, lambda d: 1.18 * d.ENGINE_EXHAUST_TEMPERATURE + 40 * d.eng_on, tau=1, noise=3),
    _p("ENGINE_COMPRESSOR_PRESSURE", "EMU", "kPa", "MEDIUM", 0, 3000, lambda d: d.amb_p / 10 * (1 + 14 * d.rpm ** 1.8), tau=1, noise=6),
    _p("ENGINE_AIR_INLET_PRESSURE", "EMU", "kPa", "SLOW", 0, 200, lambda d: d.amb_p / 10 * (1 + 0.2 * d.mach ** 2), noise=0.2),
    _p("ENGINE_AIR_INLET_TEMPERATURE", "EMU", "degC", "SLOW", -70, 120, lambda d: d.amb_t + 0.2 * d.mach ** 2 * (d.amb_t + 273) , tau=5, noise=0.3),
    _p("ENGINE_BEARING_VIBRATION_1", "EMU", "mm/s", "FAST", 0, 50, lambda d: d.eng_on * (1.2 + 4.0 * d.rpm ** 2 + 0.35 * d.man), noise=0.25),
    _p("ENGINE_BEARING_VIBRATION_2", "EMU", "mm/s", "FAST", 0, 50, lambda d: d.eng_on * (1.0 + 3.4 * d.rpm ** 2 + 0.3 * d.man), noise=0.25),
    _p("ENGINE_VIBRATION", "EMU", "mm/s", "FAST", 0, 50, lambda d: 0.6 * d.ENGINE_BEARING_VIBRATION_1 + 0.5 * d.ENGINE_BEARING_VIBRATION_2, noise=0.2),
    _p("ENGINE_BEARING_TEMPERATURE_1", "EMU", "degC", "SLOW", -40, 250, lambda d: d.ENGINE_OIL_TEMPERATURE + d.eng_warm * (12 + 18 * d.rpm), tau=40, noise=0.4, src="LOG"),
    _p("ENGINE_BEARING_TEMPERATURE_2", "EMU", "degC", "SLOW", -40, 250, lambda d: d.ENGINE_OIL_TEMPERATURE + d.eng_warm * (10 + 15 * d.rpm), tau=40, noise=0.4, src="LOG"),
    _p("ENGINE_LUBE_FLOW", "EMU", "l/min", "MEDIUM", 0, 120, lambda d: d.eng_on * (12 + 48 * d.rpm), tau=2, noise=0.4, src="LOG"),
    _p("ENGINE_LUBE_PRESSURE", "EMU", "kPa", "MEDIUM", 0, 700, lambda d: 0.92 * d.ENGINE_OIL_PRESSURE, noise=3, src="LOG"),
    # FUEL -----------------------------------------------------------------
    _p("FUEL_FLOW_RATE", "FMU", "kg/h", "MEDIUM", 0, 6000, lambda d: 1.02 * d.ENGINE_FUEL_FLOW, tau=1, noise=12),
    _p("FUEL_CONSUMPTION_RATE", "FMU", "kg/h", "SLOW", 0, 6000, lambda d: d.FUEL_FLOW_RATE, tau=20, noise=8),
    _p("FUEL_TANK_LEVEL", "FMU", "%", "SLOW", 0, 100, lambda d: d.fuel0 - np.cumsum(d.ENGINE_FUEL_FLOW) * d.dt / 3600 / 30, noise=0.1),
    _p("FUEL_TANK_PRESSURE", "FMU", "kPa", "SLOW", 0, 200, lambda d: 25 + d.amb_p / 10 * 0.3, tau=10, noise=0.3),
    _p("FUEL_TANK_TEMPERATURE", "FMU", "degC", "SLOW", -60, 90, lambda d: d.amb_g + 0.15 * (d.amb_t - d.amb_g) + 4 * d.eng_warm, tau=300, noise=0.2),
    _p("FUEL_VALVE_POSITION", "FMU", "%", "MEDIUM", 0, 100, lambda d: d.eng_on * (8 + 85 * d.rpm), tau=0.5, noise=0.3),
    _p("FUEL_RETURN_FLOW", "FMU", "kg/h", "MEDIUM", 0, 2000, lambda d: d.eng_on * (400 - 280 * d.rpm), tau=2, noise=6),
    _p("FUEL_PUMP_PRESSURE", "FPA", "kPa", "MEDIUM", 0, 1000, lambda d: d.eng_on * (260 + 280 * d.rpm), tau=1, noise=3),
    _p("FUEL_FILTER_PRESSURE_DROP", "FPA", "kPa", "SLOW", 0, 150, lambda d: 6 + 18 * (d.FUEL_FLOW_RATE / 5000) ** 2, tau=2, noise=0.3),
    _p("FUEL_PUMP_CURRENT", "FPA", "A", "MEDIUM", 0, 60, lambda d: d.eng_on * (6 + 12 * d.rpm), tau=0.5, noise=0.2, src="LOG"),
    _p("FUEL_PUMP_TEMPERATURE", "FPA", "degC", "SLOW", -50, 150, lambda d: d.FUEL_TANK_TEMPERATURE + d.eng_warm * (14 + 10 * d.rpm), tau=120, noise=0.3, src="LOG"),
    # HYDRAULIC ------------------------------------------------------------
    _p("HYD_PUMP_RPM", "HPA", "rpm", "MEDIUM", 0, 8000, lambda d: 6200 * d.ENGINE_CORE_RPM / 100, tau=0.5, noise=10),
    _p("HYD_A_PRESSURE", "HPA", "bar", "MEDIUM", 0, 350, lambda d: d.eng_on * (205 - 9 * d.hyd_dem), tau=0.6, noise=0.8),
    _p("HYD_FLOW_RATE", "HPA", "l/min", "MEDIUM", 0, 150, lambda d: d.eng_on * (8 + 55 * d.hyd_dem), tau=0.5, noise=0.5),
    _p("HYD_PUMP_CURRENT", "HPA", "A", "MEDIUM", 0, 80, lambda d: d.eng_on * (9 + 0.1 * d.HYD_A_PRESSURE * (0.5 + d.hyd_dem)), tau=0.5, noise=0.3, src="LOG"),
    _p("HYD_PUMP_TEMPERATURE", "HPA", "degC", "SLOW", -40, 160, lambda d: d.amb_g + d.eng_warm * (30 + 22 * d.hyd_dem_s), tau=120, noise=0.3, src="LOG"),
    _p("HYD_B_PRESSURE", "HPB", "bar", "MEDIUM", 0, 350, lambda d: d.eng_on * (203 - 9 * d.hyd_dem), tau=0.6, noise=0.8),
    _p("HYD_B_PUMP_CURRENT", "HPB", "A", "MEDIUM", 0, 80, lambda d: d.eng_on * (9 + 0.1 * d.HYD_B_PRESSURE * (0.5 + d.hyd_dem)), tau=0.5, noise=0.3, src="LOG"),
    _p("HYD_B_PUMP_TEMPERATURE", "HPB", "degC", "SLOW", -40, 160, lambda d: d.amb_g + d.eng_warm * (29 + 22 * d.hyd_dem_s), tau=120, noise=0.3, src="LOG"),
    _p("HYD_MAIN_PRESSURE", "HMU", "bar", "MEDIUM", 0, 350, lambda d: 0.5 * (d.HYD_A_PRESSURE + d.HYD_B_PRESSURE), noise=0.6),
    _p("HYD_RESERVOIR_LEVEL", "HMU", "%", "SLOW", 0, 100, lambda d: 78 - 6 * d.hyd_dem_s + 2 * (d.HYD_PUMP_TEMPERATURE - 40) / 40, tau=10, noise=0.2),
    _p("HYD_RESERVOIR_TEMPERATURE", "HMU", "degC", "SLOW", -40, 150, lambda d: 0.5 * (d.HYD_PUMP_TEMPERATURE + d.HYD_B_PUMP_TEMPERATURE) - 4, tau=60, noise=0.3),
    _p("HYD_FILTER_PRESSURE_DROP", "HMU", "bar", "SLOW", 0, 40, lambda d: 1.5 + 4 * (d.HYD_FLOW_RATE / 60) ** 2, tau=2, noise=0.08),
    _p("HYD_VALVE_POSITION", "HMU", "%", "MEDIUM", 0, 100, lambda d: 20 + 60 * d.hyd_dem, tau=0.3, noise=0.3),
    # ELECTRICAL -----------------------------------------------------------
    _p("GENERATOR_LOAD", "GCU", "%", "MEDIUM", 0, 120, lambda d: d.eng_on * 100 * d.elec, tau=1, noise=0.4),
    _p("GENERATOR_VOLTAGE", "GCU", "V", "MEDIUM", 0, 140, lambda d: d.eng_on * (115.5 - 1.2 * d.elec), tau=0.3, noise=0.15),
    _p("GENERATOR_CURRENT", "GCU", "A", "MEDIUM", 0, 300, lambda d: d.eng_on * 2.2 * d.GENERATOR_LOAD, noise=0.6),
    _p("GENERATOR_POWER", "GCU", "kW", "MEDIUM", 0, 60, lambda d: d.GENERATOR_VOLTAGE * d.GENERATOR_CURRENT / 1000 * 0.95, noise=0.1),
    _p("GENERATOR_FREQUENCY", "GCU", "Hz", "MEDIUM", 0, 450, lambda d: d.eng_on * 400, tau=0.2, noise=0.3),
    _p("GENERATOR_TEMPERATURE", "GCU", "degC", "SLOW", -40, 200, lambda d: d.amb_g + d.eng_warm * (35 + 0.35 * d.GENERATOR_LOAD), tau=150, noise=0.3, src="LOG"),
    _p("BATTERY_SOC", "BAT", "%", "SLOW", 0, 100, lambda d: d.soc0 - 5 * (1 - d.eng_warm), tau=5, noise=0.1),
    _p("BATTERY_CURRENT", "BAT", "A", "MEDIUM", -100, 100, lambda d: d.eng_on * (4 + 0.08 * (100 - d.BATTERY_SOC)) - (1 - d.eng_on) * 18, tau=2, noise=0.2),
    _p("BATTERY_VOLTAGE", "BAT", "V", "MEDIUM", 0, 32, lambda d: 24.2 + 0.035 * d.BATTERY_SOC + 0.02 * d.BATTERY_CURRENT, tau=1, noise=0.03),
    _p("BATTERY_TEMPERATURE", "BAT", "degC", "SLOW", -40, 100, lambda d: d.amb_g + 0.15 * (d.amb_t - d.amb_g) + 0.08 * abs(d.BATTERY_CURRENT), tau=200, noise=0.2, src="LOG"),
    _p("POWER_LOAD", "PDU", "kW", "MEDIUM", 0, 60, lambda d: 4 + 30 * d.elec, tau=0.5, noise=0.15),
    _p("DC_BUS_VOLTAGE", "PDU", "V", "FAST", 0, 32, lambda d: 27.8 * d.eng_on + d.BATTERY_VOLTAGE * (1 - d.eng_on) - 0.012 * d.POWER_LOAD, tau=0.2, noise=0.04),
    _p("DC_BUS_CURRENT", "PDU", "A", "MEDIUM", 0, 400, lambda d: 1000 * 0.4 * d.POWER_LOAD / 28, noise=0.8),
    _p("DC_BUS_POWER", "PDU", "kW", "MEDIUM", 0, 20, lambda d: d.DC_BUS_VOLTAGE * d.DC_BUS_CURRENT / 1000, noise=0.03),
    _p("AC_BUS_VOLTAGE", "PDU", "V", "FAST", 0, 140, lambda d: d.GENERATOR_VOLTAGE - 0.4, tau=0.1, noise=0.12),
    _p("AC_BUS_CURRENT", "PDU", "A", "MEDIUM", 0, 300, lambda d: 0.95 * d.GENERATOR_CURRENT, noise=0.6),
    _p("AC_BUS_POWER", "PDU", "kW", "MEDIUM", 0, 60, lambda d: d.AC_BUS_VOLTAGE * d.AC_BUS_CURRENT / 1000 * 0.95, noise=0.1),
    _p("BUS_FREQUENCY", "PDU", "Hz", "FAST", 0, 450, lambda d: d.GENERATOR_FREQUENCY, noise=0.25),
    # FLIGHT CONTROL -------------------------------------------------------
    _p("CONTROL_STICK_POSITION", "FCC", "%", "FAST", -100, 100, lambda d: 100 * d.stick, noise=0.3),
    _p("ELEVATOR_COMMAND", "FCC", "deg", "FAST", -30, 30, lambda d: 0.22 * d.CONTROL_STICK_POSITION + 2 * d.load_dev, noise=0.05),
    _p("AILERON_COMMAND", "FCC", "deg", "FAST", -30, 30, lambda d: 18 * d.roll_cmd, noise=0.05),
    _p("RUDDER_COMMAND", "FCC", "deg", "FAST", -30, 30, lambda d: 12 * d.ped, noise=0.05),
    _p("ACTUATOR_COMMAND", "FCC", "deg", "FAST", -30, 30, lambda d: d.ELEVATOR_COMMAND, noise=0.03),
    _p("FLAP_COMMAND", "FCC", "deg", "MEDIUM", 0, 40, lambda d: d.flap, noise=0.0, decimals=1),
    _p("SPOILER_COMMAND", "FCC", "deg", "MEDIUM", 0, 60, lambda d: d.spoiler, noise=0.0, decimals=1),
    _p("ELEVATOR_POSITION", "SAA", "deg", "FAST", -30, 30, lambda d: d.ELEVATOR_COMMAND, tau=0.15, noise=0.05, extra={"follows": "ELEVATOR_COMMAND"}),
    _p("AILERON_POSITION", "SAA", "deg", "FAST", -30, 30, lambda d: d.AILERON_COMMAND, tau=0.12, noise=0.05, extra={"follows": "AILERON_COMMAND"}),
    _p("RUDDER_POSITION", "SAA", "deg", "FAST", -30, 30, lambda d: d.RUDDER_COMMAND, tau=0.15, noise=0.05, extra={"follows": "RUDDER_COMMAND"}),
    _p("ACTUATOR_POSITION", "SAA", "deg", "FAST", -30, 30, lambda d: d.ACTUATOR_COMMAND, tau=0.15, noise=0.04, extra={"follows": "ACTUATOR_COMMAND"}),
    _p("FLAP_POSITION", "SAA", "deg", "MEDIUM", 0, 40, lambda d: d.FLAP_COMMAND, tau=3, noise=0.05, extra={"follows": "FLAP_COMMAND"}),
    _p("SPOILER_POSITION", "SAA", "deg", "MEDIUM", 0, 60, lambda d: d.SPOILER_COMMAND, tau=0.8, noise=0.05, extra={"follows": "SPOILER_COMMAND"}),
    _p("POSITION_ERROR", "SAA", "deg", "MEDIUM", 0, 10, lambda d: abs(d.ACTUATOR_COMMAND - d.ACTUATOR_POSITION), noise=0.01),
    _p("COMMAND_ERROR", "FCC", "deg", "MEDIUM", 0, 10, lambda d: abs(d.ELEVATOR_COMMAND - d.ELEVATOR_POSITION) + abs(d.AILERON_COMMAND - d.AILERON_POSITION), noise=0.01),
    _p("CONTROL_SURFACE_RATE", "SAA", "deg/s", "FAST", -200, 200, lambda d: d.surface_rate, noise=0.3),
    _p("HYD_ACTUATOR_PRESSURE", "SAA", "bar", "MEDIUM", 0, 350, lambda d: 0.97 * d.HYD_MAIN_PRESSURE - 6 * d.hyd_dem, noise=0.7),
    _p("ACTUATOR_PRESSURE", "SAA", "bar", "MEDIUM", 0, 350, lambda d: d.HYD_ACTUATOR_PRESSURE - 3, noise=0.7),
    _p("ACTUATOR_CURRENT", "SAA", "A", "MEDIUM", 0, 40, lambda d: 1.2 + 0.05 * abs(d.surface_rate) + 2.5 * d.hyd_dem, tau=0.3, noise=0.05),
    _p("ACTUATOR_TEMPERATURE", "SAA", "degC", "SLOW", -50, 150, lambda d: d.amb_g + 0.3 * (d.amb_t - d.amb_g) + d.eng_warm * (12 + 15 * d.hyd_dem_s), tau=150, noise=0.3, src="LOG"),
    # NAVIGATION / AIR DATA -------------------------------------------------
    _p("ALTITUDE", "ADC", "ft", "MEDIUM", -2000, 60000, lambda d: d.alt, noise=4, decimals=0),
    _p("BAROMETRIC_ALTITUDE", "ADC", "ft", "MEDIUM", -2000, 60000, lambda d: d.alt + 0.003 * d.alt, noise=6, decimals=0),
    _p("VERTICAL_SPEED", "ADC", "ft/min", "MEDIUM", -30000, 30000, lambda d: d.vs, tau=1, noise=20, decimals=0),
    _p("STATIC_PRESSURE", "ADC", "hPa", "MEDIUM", 0, 1100, lambda d: d.amb_p, noise=0.2),
    _p("TOTAL_PRESSURE", "ADC", "hPa", "MEDIUM", 0, 3000, lambda d: d.amb_p * (1 + 0.2 * d.mach ** 2) ** 3.5, noise=0.3),
    _p("STATIC_TEMPERATURE", "ADC", "degC", "SLOW", -80, 60, lambda d: d.amb_t, tau=3, noise=0.2),
    _p("TOTAL_TEMPERATURE", "ADC", "degC", "SLOW", -80, 150, lambda d: (d.amb_t + 273.15) * (1 + 0.2 * d.mach ** 2) - 273.15, tau=3, noise=0.2),
    _p("MACH_NUMBER", "ADC", "", "MEDIUM", 0, 2.5, lambda d: d.mach, noise=0.002, decimals=3),
    _p("TRUE_AIRSPEED", "ADC", "kt", "MEDIUM", 0, 1200, lambda d: d.spd, noise=1),
    _p("INDICATED_AIRSPEED", "ADC", "kt", "MEDIUM", 0, 1000, lambda d: d.spd * (d.amb_p / 1013.25) ** 0.5, noise=1),
    _p("ANGLE_OF_ATTACK", "ADC", "deg", "FAST", -20, 40, lambda d: d.aoa, noise=0.1),
    _p("SIDESLIP_ANGLE", "ADC", "deg", "FAST", -20, 20, lambda d: 0.4 * d.ped + 0.02 * d.wind_x, noise=0.08),
    _p("LATITUDE", "INU", "deg", "SLOW", -90, 90, lambda d: d.lat, noise=0.00001, decimals=6),
    _p("LONGITUDE", "INU", "deg", "SLOW", -180, 180, lambda d: d.lon, noise=0.00001, decimals=6),
    _p("GPS_ALTITUDE", "INU", "ft", "SLOW", -2000, 60000, lambda d: d.alt + 35, noise=10, decimals=0),
    _p("GROUND_SPEED", "INU", "kt", "MEDIUM", 0, 1200, lambda d: d.spd + d.wind_x, tau=1, noise=0.8),
    _p("HEADING", "INU", "deg", "MEDIUM", 0, 360, lambda d: d.hdg, noise=0.05),
    _p("ROLL", "INU", "deg", "FAST", -180, 180, lambda d: d.roll, noise=0.05),
    _p("PITCH", "INU", "deg", "FAST", -90, 90, lambda d: d.pitch, noise=0.05),
    _p("YAW", "INU", "deg", "FAST", -180, 180, lambda d: d.yaw, noise=0.05),
    _p("ROLL_RATE", "INU", "deg/s", "FAST", -300, 300, lambda d: d.roll_rate, noise=0.2),
    _p("PITCH_RATE", "INU", "deg/s", "FAST", -100, 100, lambda d: d.pitch_rate, noise=0.1),
    _p("YAW_RATE", "INU", "deg/s", "FAST", -100, 100, lambda d: d.yaw_rate, noise=0.1),
    _p("ACCEL_X", "INU", "g", "FAST", -5, 5, lambda d: d.accel_x, noise=0.01, decimals=3),
    _p("ACCEL_Y", "INU", "g", "FAST", -5, 5, lambda d: 0.02 * d.ped, noise=0.01, decimals=3),
    _p("ACCEL_Z", "INU", "g", "FAST", -4, 10, lambda d: d.load, noise=0.02, decimals=3),
    _p("GPS_SATELLITE_COUNT", "INU", "count", "SLOW", 0, 20, lambda d: 11 - 2 * d.man, noise=0.3, kind="discrete"),
    _p("GPS_SIGNAL_QUALITY", "INU", "%", "SLOW", 0, 100, lambda d: 94 - 6 * d.man - 2 * d.gnd, tau=2, noise=0.6),
    _p("NAV_POSITION_ERROR", "INU", "m", "SLOW", 0, 500, lambda d: 4 + 3 * d.man, tau=10, noise=0.2),
    _p("NAV_VELOCITY_ERROR", "INU", "m/s", "SLOW", 0, 50, lambda d: 0.15 + 0.1 * d.man, tau=10, noise=0.01, decimals=3),
    # ENVIRONMENTAL ---------------------------------------------------------
    _p("CABIN_PRESSURE", "ECU", "hPa", "SLOW", 0, 1100, lambda d: d.amb_g_p - (d.amb_g_p - 750) * (d.alt / 40000).clip(0, 1) * 0.9, tau=20, noise=0.3),
    _p("CABIN_TEMPERATURE", "ECU", "degC", "SLOW", -20, 60, lambda d: 21 + 0.12 * (d.amb_g - 21) * (1 - d.eng_warm) + 0.5 * d.man, tau=120, noise=0.1),
    _p("CABIN_HUMIDITY", "ECU", "%", "SLOW", 0, 100, lambda d: d.hum * (1 - 0.6 * d.eng_warm) + 10, tau=200, noise=0.4),
    _p("EQUIPMENT_BAY_HUMIDITY", "ECU", "%", "SLOW", 0, 100, lambda d: d.hum * (1 - 0.5 * d.eng_warm) + 5, tau=250, noise=0.4),
    _p("COOLING_FAN_RPM", "ACU", "rpm", "MEDIUM", 0, 12000, lambda d: 4000 + 5500 * d.elec * d.eng_on + 1500 * d.gnd * d.eng_on, tau=2, noise=15),
    _p("COOLING_FAN_CURRENT", "ACU", "A", "MEDIUM", 0, 30, lambda d: 0.9 + 0.0009 * d.COOLING_FAN_RPM, tau=0.5, noise=0.03, src="LOG"),
    _p("COOLING_FLOW_RATE", "ACU", "kg/min", "MEDIUM", 0, 60, lambda d: 4 + 0.0028 * d.COOLING_FAN_RPM, tau=1, noise=0.1),
    _p("COOLING_PRESSURE", "ACU", "kPa", "MEDIUM", 0, 400, lambda d: 90 + 9 * d.COOLING_FLOW_RATE, tau=1, noise=0.5),
    _p("COOLING_TEMPERATURE", "ACU", "degC", "SLOW", -40, 100, lambda d: d.amb_g * 0.5 + 0.3 * d.amb_t + 20 * d.elec - 0.25 * d.COOLING_FLOW_RATE + 6, tau=60, noise=0.2),
    _p("EQUIPMENT_BAY_TEMPERATURE", "ECU", "degC", "SLOW", -40, 100, lambda d: d.COOLING_TEMPERATURE + 9 + 10 * d.elec, tau=120, noise=0.2),
    # AVIONICS / COMPUTING --------------------------------------------------
    _p("CPU_LOAD", "MCP", "%", "MEDIUM", 0, 100, lambda d: 100 * d.cpu_dem, tau=1, noise=0.8),
    _p("MEMORY_USAGE", "MCP", "%", "SLOW", 0, 100, lambda d: 38 + 18 * d.cpu_dem_s, tau=30, noise=0.2),
    _p("PROCESSOR_VOLTAGE", "MCP", "V", "MEDIUM", 0, 6, lambda d: 3.30 - 0.04 * d.cpu_dem, noise=0.004, decimals=3),
    _p("PROCESSOR_CURRENT", "MCP", "A", "MEDIUM", 0, 20, lambda d: 2.2 + 5.5 * d.cpu_dem, tau=0.5, noise=0.05, src="LOG"),
    _p("CPU_TEMPERATURE", "MCP", "degC", "SLOW", -40, 125, lambda d: d.EQUIPMENT_BAY_TEMPERATURE + 14 + 22 * d.cpu_dem_s, tau=40, noise=0.25),
    _p("BOARD_TEMPERATURE", "MCP", "degC", "SLOW", -40, 125, lambda d: d.EQUIPMENT_BAY_TEMPERATURE + 6 + 10 * d.cpu_dem_s, tau=80, noise=0.2, src="LOG"),
    _p("DATA_BUS_LOAD", "MCP", "%", "MEDIUM", 0, 100, lambda d: 28 + 22 * d.cpu_dem, tau=1, noise=0.4),
    _p("MESSAGE_RATE", "MCP", "msg/s", "MEDIUM", 0, 2000, lambda d: 420 + 400 * d.cpu_dem, tau=1, noise=4, decimals=0),
    _p("PROCESS_LATENCY", "MCP", "ms", "MEDIUM", 0, 500, lambda d: 4 + 9 * d.cpu_dem ** 2, tau=0.5, noise=0.15),
    _p("MESSAGE_ERROR_COUNT", "MCP", "count", "SLOW", 0, 65535, lambda d: 0.002 + 0 * d.t, kind="counter"),
    _p("BIT_ERROR_COUNT", "MCP", "count", "SLOW", 0, 65535, lambda d: 0.001 + 0 * d.t, kind="counter"),
    _p("WATCHDOG_COUNT", "MCP", "count", "EVENT", 0, 255, lambda d: 0.00002 + 0 * d.t, kind="counter"),
    _p("RESTART_COUNT", "MCP", "count", "EVENT", 0, 255, lambda d: 0.000005 + 0 * d.t, kind="counter"),
    # COMMUNICATION ---------------------------------------------------------
    _p("TX_POWER", "CTR", "dBm", "MEDIUM", 0, 60, lambda d: 40 + 0 * d.t, noise=0.1),
    _p("RX_POWER", "CTR", "dBm", "MEDIUM", -140, 0, lambda d: -62 - 20 * np.log10(1 + d.link_dist / 10), tau=1, noise=0.6),
    _p("SIGNAL_STRENGTH", "CTR", "%", "MEDIUM", 0, 100, lambda d: (d.RX_POWER + 120) * 1.4, noise=0.4),
    _p("SIGNAL_TO_NOISE_RATIO", "CTR", "dB", "MEDIUM", -10, 60, lambda d: d.RX_POWER + 110 - 2 * d.man, tau=0.5, noise=0.4),
    _p("BIT_ERROR_RATE", "CTR", "1e-6", "SLOW", 0, 10000, lambda d: 10 ** (2.5 - 0.06 * d.SIGNAL_TO_NOISE_RATIO), tau=2, noise=0.05),
    _p("PACKET_ERROR_RATE", "CTR", "%", "SLOW", 0, 100, lambda d: 0.02 * d.BIT_ERROR_RATE, tau=2, noise=0.005, decimals=3),
    _p("LINK_LATENCY", "CTR", "ms", "MEDIUM", 0, 2000, lambda d: 18 + 0.05 * d.link_dist + 2 * d.PACKET_ERROR_RATE, tau=1, noise=0.4),
    _p("DATA_RATE", "CTR", "kbps", "MEDIUM", 0, 10000, lambda d: 2048 * (1 - d.PACKET_ERROR_RATE / 100), noise=6, decimals=0),
    _p("CHANNEL_STATUS", "CTR", "enum", "EVENT", 0, 3, lambda d: 1 + (d.SIGNAL_TO_NOISE_RATIO < 12), kind="discrete"),
    _p("ANTENNA_STATUS", "CTR", "enum", "EVENT", 0, 3, lambda d: 1 + 0 * d.t, kind="discrete"),
]

BY_NAME = {p.name: p for p in CATALOG}
assert len(BY_NAME) == len(CATALOG), "duplicate parameter names"
