"""Natural-language snag text with paraphrase / terminology variation."""

SYN = {
    "pressure": ["pressure", "press", "pr."], "temperature": ["temperature", "temp"], "hydraulic": ["hydraulic", "hyd"],
    "low": ["low", "dropping", "below normal", "reduced"], "high": ["high", "elevated", "above normal", "excessive"],
    "engine": ["engine", "eng"], "generator": ["generator", "gen"], "indication": ["indication", "ind", "reading"],
    "vibration": ["vibration", "vib", "vibes"], "intermittent": ["intermittent", "sporadic", "on/off"],
    "fluctuating": ["fluctuating", "unsteady", "oscillating", "hunting"], "observed": ["observed", "noted", "seen", "reported"],
}
TEMPLATES = [
    "During {phase} pilot reported {symptom}.",
    "{Symptom} {observed} in {phase}; {param} indicated approx {value} {unit}.",
    "Crew report: {symptom} - {param} ~{value} {unit} during {phase}.",
    "{Symptom} after landing, BITE message logged.",
    "On {phase}, {symptom}. Caution reset in flight.",
    "{Symptom} {observed} by pilot, recurring at {phase}.",
]
ATA = {"ENGINE": 72, "FUEL": 28, "HYDRAULIC": 29, "ELECTRICAL": 24, "FLIGHT_CONTROL": 27, "NAVIGATION": 34,
       "ENVIRONMENTAL": 21, "AVIONICS": 46, "COMMUNICATION": 23, "GENERAL": 5}
NUISANCE = [
    ("Cockpit map light inoperative", "Replaced lamp, ops check OK"),
    ("Canopy seal worn at aft edge", "Seal replaced"),
    ("Minor oil seep at gearbox drain plug", "Drain plug re-torqued, leak check OK"),
    ("Main wheel tyre worn to limit", "Tyre changed"),
    ("Paint damage on access panel", "Touched up"),
    ("Seat harness buckle stiff", "Buckle lubricated"),
]


def paraphrase(rng, text):
    words = []
    for w in text.split():
        key = w.lower().strip(".,")
        words.append(str(rng.choice(SYN[key])) if key in SYN and rng.random() < 0.6 else w)
    return " ".join(words)


def snag_text(rng, mech, phase, param, value, unit):
    symptom = paraphrase(rng, str(rng.choice(mech.symptoms)))
    tpl = str(rng.choice(TEMPLATES))
    desc = tpl.format(phase=phase.lower().replace("_", " "), symptom=symptom, Symptom=symptom[:1].upper() + symptom[1:],
                      observed=str(rng.choice(SYN["observed"])), param=param.replace("_", " ").lower(), value=value, unit=unit)
    return symptom[:1].upper() + symptom[1:], desc, symptom
