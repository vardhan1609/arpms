"""Maintenance-record text generation."""


def fim_ref(code):
    return "FIM-" + code[3:]


def mm_ref(lru):
    return f"MM-{lru}-01"


def disposition(rng, mech, outcome):
    fim = fim_ref(mech.code)
    if outcome == "NFF":
        return (f"Troubleshooting IAW {fim} carried out. Ground run satisfactory, fault could not be reproduced. NFF, aircraft released with monitoring.",
                "Ground run / BITE - no fault found", "NO FAULT FOUND", "PASS")
    finding = "found failed" if outcome == "FAILURE" else "found degraded on inspection"
    return (f"Troubleshooting IAW {fim}. {mech.description} {finding}. {mech.action}. Ops check satisfactory.",
            mech.action, "CONFIRMED FAILURE" if outcome == "FAILURE" else "CONFIRMED DEGRADED", "PASS")
