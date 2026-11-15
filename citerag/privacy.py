"""Finding and masking private data in answers and passages. (Phase 15)

Three kinds, chosen by the user: phone numbers, ID numbers (citizenship,
national ID, PAN, passport, driving licence, voter ID), and bank account and
payment card numbers. Detection is local pattern matching with checks that cut
false alarms, so nothing is sent anywhere:

- phone numbers: Nepali mobiles (96/97/98 + 8 digits) and landlines with an
  area code, numbers written with a "+" country code, or digits right after a
  word like "phone" / "mobile" / "फोन";
- card numbers: 13-19 digits that pass the Luhn checksum and start like a real
  card (Visa, Mastercard, Amex, UnionPay, ...) or follow the word "card";
- bank accounts: digits after "account" / "A/C" / "खाता"; IBANs by checksum;
- ID numbers: digits after the name of the document (citizenship, PAN, ...).

Devanagari digits (०-९) count as digits. Masking replaces each digit (and each
letter of an ID) with "•" and keeps separators, so the text keeps its length:
offsets into it (highlighted quotes and statements) stay valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DOT = "•"
D = "[0-9०-९]"  # a digit, ASCII or Devanagari
SEP = r"[ \-.]"
LABELS = {"phone": "phone number", "card": "card number", "bank_account": "bank account number",
          "id_number": "ID number"}


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int

    def label(self) -> str:
        return LABELS[self.kind]


def _digits(s: str) -> str:
    return "".join(str(int(c)) for c in s if c.isdigit())  # int() reads Devanagari digits too


# -- patterns ------------------------------------------------------------------------

_EDGE_L, _EDGE_R = r"(?<![\w+])", r"(?![\w])"
_CC = rf"(?:(?:\+|00)977{SEP}?)"  # Nepal's country code
PHONE_RES = [
    # Nepali mobile: 96/97/98 and 8 more digits, optionally +977, with single separators
    re.compile(rf"{_EDGE_L}{_CC}?[9९][678६७८](?:{SEP}?{D}){{8}}{_EDGE_R}"),
    # Nepali landline: 0 + area code, separated from the 6-7 digit number ("01-4123456", "(061) 523456")
    re.compile(rf"{_EDGE_L}(?:\(\s*[0०]{D}{{1,2}}\s*\)\s?|[0०]{D}{{1,2}}[ \-]){D}{{6,7}}{_EDGE_R}"),
    re.compile(rf"{_EDGE_L}\+977{SEP}?(?:\(?{D}{{1,2}}\)?){SEP}?{D}{{6,7}}{_EDGE_R}"),
    # any number written with a "+" country code: 8-15 digits in groups
    re.compile(rf"(?<![\w+])\+{D}{{1,3}}(?:{SEP}?\(?{D}{{1,4}}\)?)(?:{SEP}?{D}{{2,5}}){{1,5}}{_EDGE_R}"),
]
_PHONE_WORDS = r"(?:tel(?:ephone)?|phone|ph|mobile|mob|cell(?:phone)?|fax|whats\s?app|viber|contact(?:\s+(?:no|number))?|फोन|मोबाइल|सम्पर्क)"
PHONE_CONTEXT_RE = re.compile(
    rf"\b{_PHONE_WORDS}\b\.?\s*(?:no\.?|number|num|#)?\s*[:\-–]?\s*(?P<value>\+?\(?{D}(?:[ \-.()]?{D}){{6,14}})(?![\w])",
    re.IGNORECASE)

CARD_RE = re.compile(rf"(?<![\w]){D}(?:[ \-]?{D}){{12,18}}(?![\w])")
_CARD_PREFIXES = ("4", "51", "52", "53", "54", "55", "2221", "2720", "34", "37", "6011", "65", "62", "35", "36", "300",
                  "301", "302", "303", "304", "305", "60", "81")
CARD_WORD_RE = re.compile(r"\b(?:card|debit|credit|visa|mastercard|amex)\b", re.IGNORECASE)

_ACCOUNT_WORDS = r"(?:(?:bank\s+)?(?:account|acct|a/c|ac)(?:\s+(?:no|number|num))?|खाता(?:\s*(?:नं|नम्बर)\.?)?)"
ACCOUNT_RE = re.compile(
    rf"(?<![\w/]){_ACCOUNT_WORDS}\.?\s*[:\-–#]?\s*(?:is\s+)?(?P<value>[A-Z]{{0,4}}{D}(?:[ \-]?{D}){{6,23}})(?![\w])",
    re.IGNORECASE)
IBAN_RE = re.compile(r"(?<![\w])[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?(?![\w])")

_ID_WORDS = (r"(?:citizenship(?:\s+(?:certificate|card))?|nagarikta|नागरिकता(?:को)?(?:\s*प्रमाणपत्र)?|ना\.?\s?प्र\.?"
             r"|national\s+id(?:entity)?(?:\s+card)?|(?-i:NID|NIN)|राष्ट्रिय\s+परिचय\s*(?:पत्र)?"
             r"|(?-i:PAN)|permanent\s+account|स्थायी\s+लेखा|passport|राहदानी"
             r"|driving\s+licen[cs]e|(?:driver'?s?\s+)?licen[cs]e|सवारी\s+चालक\s+अनुमतिपत्र"
             r"|voter(?:'?s)?\s+id|मतदाता\s+परिचयपत्र|social\s+security|(?-i:SSN)|id\s+card|identity\s+card"
             r"|employer\s+identification|taxpayer\s+identification|tax\s+id|(?-i:EIN|TIN))")
ID_RE = re.compile(
    rf"\b{_ID_WORDS}\s*(?:no|number|num|नं|नम्बर|#)?\.?\s*[:\-–]?\s*(?P<value>[A-Z]{{0,3}}{D}[{D[1:-1]}\-/ ]{{2,22}}{D}|[A-Z]{{1,3}}{D}{{5,9}})(?![\w])",
    re.IGNORECASE)


# On a form the value sits a line or two below its label ("2c Employer's telephone
# number / In care of name   011536259"). A label then, within 160 characters and
# with no other number in between, one run of 7-15 digits that isn't a year range.
_NEARBY = rf"[^0-9०-९]{{0,160}}?(?<![\w/.\-])(?P<value>\+?\(?{D}(?:[ \-()]?{D}){{6,14}})(?![\w/.\-])"
NEARBY_RES = {
    "phone": re.compile(rf"\b{_PHONE_WORDS}\b{_NEARBY}", re.IGNORECASE),
    "id_number": re.compile(rf"\b{_ID_WORDS}{_NEARBY}", re.IGNORECASE),
}
_YEAR_RANGE_RE = re.compile(r"^(?:1[89]|20)\d\d\s*[-–/]\s*(?:(?:1[89]|20)\d\d|\d\d)$")


def _luhn(digits: str) -> bool:
    total, double = 0, False
    for c in reversed(digits):
        n = int(c) * (2 if double else 1)
        total += n - 9 if n > 9 else n
        double = not double
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    s = value.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


# -- finding -------------------------------------------------------------------------


def find_private(text: str) -> list[Finding]:
    """Private values in `text`, in order, not overlapping (the more specific kind wins)."""
    found: list[Finding] = []

    def add(kind: str, start: int, end: int) -> None:
        while end > start and text[end - 1] in " -./(":
            end -= 1
        found.append(Finding(kind, start, end))

    for m in ID_RE.finditer(text):
        if len(_digits(m["value"])) >= 5:
            add("id_number", m.start("value"), m.end("value"))
    for m in ACCOUNT_RE.finditer(text):
        if len(_digits(m["value"])) >= 8:
            add("bank_account", m.start("value"), m.end("value"))
    for m in IBAN_RE.finditer(text):
        if _iban_ok(m.group()):
            add("bank_account", m.start(), m.end())
    for m in CARD_RE.finditer(text):
        digits = _digits(m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits) and (
                digits.startswith(_CARD_PREFIXES) or CARD_WORD_RE.search(text, max(0, m.start() - 30), m.start())):
            add("card", m.start(), m.end())
    for pattern in PHONE_RES:
        for m in pattern.finditer(text):
            if 7 <= len(_digits(m.group())) <= 15:
                add("phone", m.start(), m.end())
    for m in PHONE_CONTEXT_RE.finditer(text):
        add("phone", m.start("value"), m.end("value"))
    for kind, pattern in NEARBY_RES.items():
        for m in pattern.finditer(text):
            if not _YEAR_RANGE_RE.match(m["value"].strip()):
                add(kind, m.start("value"), m.end("value"))

    priority = {"id_number": 0, "bank_account": 1, "card": 2, "phone": 3}
    found.sort(key=lambda f: (f.start, priority[f.kind], -(f.end - f.start)))
    kept: list[Finding] = []
    for f in found:
        if kept and f.start < kept[-1].end:
            continue
        kept.append(f)
    return kept


# -- masking -------------------------------------------------------------------------


def _keep(kind: str, value: str) -> tuple[int, int]:
    """How many digits/letters to leave visible at the start and at the end."""
    if kind == "phone":
        digits = _digits(value)
        lead = 2
        if value.lstrip("(").startswith(("+977", "00977", "977")):
            lead = 3 + 2  # the country code and the first two digits of the number
        elif value.startswith("+"):
            lead = 3
        return (lead if len(digits) > lead + 4 else 0), 2
    if kind == "card":
        return 0, 4
    if kind == "bank_account":
        return 0, 3
    return 0, 2


def mask_value(kind: str, value: str) -> str:
    """`value` with its digits (and an ID's letters) replaced by "•", except a few at either end."""
    maskable = [i for i, c in enumerate(value) if c.isdigit() or (kind in ("id_number", "bank_account") and c.isalpha())]
    lead, tail = _keep(kind, value)
    hide = set(maskable[lead:len(maskable) - tail] if tail else maskable[lead:])
    return "".join(DOT if i in hide else c for i, c in enumerate(value))


def mask(text: str, findings: list[Finding] | None = None) -> str:
    """`text` with every private value masked; same length, so offsets into it stay valid."""
    findings = find_private(text) if findings is None else findings
    out, i = [], 0
    for f in findings:
        out.append(text[i:f.start])
        out.append(mask_value(f.kind, text[f.start:f.end]))
        i = f.end
    out.append(text[i:])
    return "".join(out)


_TRAILING_NUMBER_RE = re.compile(rf"(?:\+|\(|{D})[{D[1:-1]} \-.()+/]*$")


def streaming_safe(text: str) -> str:
    """A partial answer cut before a number that may still be growing.

    While an answer streams, "My number is 98412" may become a phone number
    with the next piece; until the number is complete it can't be recognised
    and masked, so it isn't shown yet.
    """
    return text[:m.start()] if (m := _TRAILING_NUMBER_RE.search(text)) else text


def summary(findings: list[Finding]) -> str:
    """'2 phone numbers and 1 ID number'."""
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.label()] = counts.get(f.label(), 0) + 1
    parts = [f"{n} {label}{'s' if n > 1 else ''}" for label, n in counts.items()]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
