"""Register junk → NONE (plan-09 Task 4): what clerks put in a country or jurisdiction field that
is not a place — company names ("Summit Ridge Holdings LLC"), legal and administrative terms
("PERPETUAL", "S-Corporation"), "ATTN:" lines, street lines ("1675 Broadway"), written dates,
single letters. Made from word lists (no real company or person); the strict G1 checker drops any
string holding a place name (containment), so a "Jordan Capital LLC" never becomes NONE.
"""

from __future__ import annotations

import numpy as np

from countrycode import labels
from countrycode.synth.framework import BaseAlias

WORDS = ("Summit", "Ridge", "Pine", "Oak", "Silver", "Golden", "Eagle", "Liberty", "Pioneer",
         "Harbor", "Cedar", "Maple", "Granite", "Crown", "Apex", "Prime", "Sterling", "Evergreen",
         "Horizon", "Bright", "Clear", "Stone", "Falcon", "Willow", "Meridian", "Beacon",
         "Anchor", "Northstar", "Bluebird", "Redwood", "Ironwood", "Lighthouse", "Keystone",
         "Vista")  # fmt: skip
NOUNS = ("Capital", "Holdings", "Partners", "Ventures", "Properties", "Realty", "Investments",
         "Services", "Solutions", "Group", "Trust", "Management", "Consulting", "Development",
         "Enterprises", "Funding", "Technologies", "Logistics", "Energy", "Media", "Builders",
         "Equipment", "Leasing", "Advisors")  # fmt: skip
LEGAL = ("LLC", "L.L.C.", "Inc.", "INC", "Corp.", "Corporation", "LP", "L.P.", "LLP", "Ltd.",
         "Co.", "PLLC", "PC")  # fmt: skip
TERMS = ("Common Law", "Perpetual", "Intrastate", "Interstate", "S-Corporation", "C Corporation",
         "Nonprofit", "Non-profit corporation", "Limited Liability Company", "Sole Proprietorship",
         "General Partnership", "Statutory Trust", "Business Trust", "Cooperative",
         "Benefit Corporation", "Professional Corporation", "Memorandum of Understanding",
         "Articles of Incorporation", "Certificate of Good Standing", "Registered Agent",
         "President", "Secretary", "Treasurer", "Director", "Manager", "Member", "Organizer",
         "Incorporator", "See attached", "As filed", "Pending", "Withdrawn", "Dissolved", "Revoked",
         "Merged", "Converted", "Reinstated", "Active", "Inactive", "Good Standing", "Delinquent",
         "Expired", "Servicing contracts", "All legal entities", "Above")  # fmt: skip
DEPARTMENTS = ("Corporate Trust Dept.", "Legal Department", "Accounts Payable", "Compliance",
               "Tax Department", "General Counsel", "Registered Agent")  # fmt: skip
STREETS = ("Main", "Oak", "Maple", "Elm", "Pine", "Cedar", "Park", "Lake", "Hill", "Broadway",
           "Market", "Church", "Mill", "River", "Spring", "Center", "High", "Union", "Water",
           "Bridge", "Highland", "Sunset", "Forest", "Meadow", "Grove", "Chestnut")  # fmt: skip
SUFFIXES = ("St", "St.", "Street", "Ave", "Avenue", "Rd", "Road", "Blvd", "Dr", "Drive", "Ln",
            "Way", "Ct", "Pl", "Pkwy")  # fmt: skip
UNITS = ("", "", "", ", Suite 100", ", Ste 210", ", Floor 3", ", Unit 4B", " #12")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")  # fmt: skip
ROMAN = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")


def _pick(rng: np.random.Generator, seq: tuple[str, ...]) -> str:
    return str(seq[int(rng.integers(0, len(seq)))])


class RegisterJunk:
    name = "regjunk"
    version = "1"
    near_check = False  # NONE by construction; exact and containment checks still apply

    def __init__(self, n: int = 4000) -> None:
        self.n = n
        self.config = {"n": n, "kinds": ["company", "term", "attn", "street", "date", "letter"],
                       "words": len(WORDS), "terms": len(TERMS)}  # fmt: skip

    def bases(self) -> list[BaseAlias]:
        return [
            BaseAlias(f"__regjunk_{i}", (labels.NONE,), "regjunk", "zxx") for i in range(self.n)
        ]

    def _company(self, rng: np.random.Generator) -> str:
        parts = [_pick(rng, WORDS)]
        if rng.random() < 0.6:
            parts.append(_pick(rng, NOUNS))
        if rng.random() < 0.15:
            parts.append(_pick(rng, ROMAN) if rng.random() < 0.5 else str(rng.integers(2001, 2026)))
        sep = ", " if rng.random() < 0.5 else " "
        return " ".join(parts) + sep + _pick(rng, LEGAL)

    def generate(self, row: BaseAlias, rng: np.random.Generator) -> list[str]:
        kind = int(rng.choice(6, p=[0.35, 0.2, 0.05, 0.2, 0.1, 0.1]))
        if kind == 0:
            s = self._company(rng)
        elif kind == 1:
            s = _pick(rng, TERMS)
        elif kind == 2:
            s = f"ATTN: {_pick(rng, DEPARTMENTS)}"
        elif kind == 3:
            s = (f"{rng.integers(1, 9999)} {_pick(rng, STREETS)} {_pick(rng, SUFFIXES)}"
                 f"{_pick(rng, UNITS)}")  # fmt: skip
        elif kind == 4:
            m, d, y = _pick(rng, MONTHS), int(rng.integers(1, 29)), int(rng.integers(1950, 2026))
            s = f"{m} {d}, {y}" if rng.random() < 0.5 else f"{d} {m[:3]} {y}"
        else:
            s = chr(int(rng.integers(65, 91))) + ("." if rng.random() < 0.5 else "")
        return [s.upper() if rng.random() < 0.4 else s]
