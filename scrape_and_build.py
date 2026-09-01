"""
scrape_and_build.py

Reads Richmond Hill's official "Adults 55+ Drop-In Programs" page, pulls out the
table for Oak Ridges Community Centre for the CURRENT month, translates the
activity names, and rewrites data.json (which index.html reads at runtime).

Run manually:   python scrape_and_build.py
Run automatically: see .github/workflows/update.yml (runs weekly)

Design notes for future-you (or whoever maintains this):
- This script was written without the ability to test it against the live
  page (the environment that authored it had no internet access). The page
  structure was inspected via a fetched snapshot, so the parsing logic below
  is a best effort, not a guarantee. It is written defensively: if anything
  about the page has changed and parsing fails, the script prints a clear
  error and LEAVES data.json UNTOUCHED, so the website never breaks or shows
  wrong data — it just goes a bit stale until someone fixes the parser.
- If it breaks, paste the printed error (and ideally the page URL) to
  Claude / Claude Code and ask it to update the parsing section only.
"""

import json
import re
import sys
from datetime import date

import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

URL = "https://www.richmondhill.ca/en/things-to-do/55-drop-in-programs.aspx"
CENTRE_HEADING_TEXT = "Oak Ridges Community Centre"
OUTPUT_FILE = "data.json"

# English activity name -> (Chinese label, category)
# category is one of: sport, game, hobby  (used for the colour dot in the UI)
# Add to this dictionary if Oak Ridges (or another centre you add later)
# ever offers something not listed here — unmapped names fall back to
# showing the English name untranslated rather than failing.
TRANSLATIONS = {
    "Badminton": ("羽毛球", "", "sport"),
    "Table Tennis": ("乒乓球", "", "sport"),
    "Tai Chi": ("太极", "", "sport"),
    "Pickleball": ("匹克球", "", "sport"),
    "Chinese Mahjong": ("麻将", "", "game"),
    "Vietnamese Mahjong": ("越南麻将", "", "game"),
    "Euchre": ("尤克纸牌", "", "game"),
    "Bid Euchre": ("叫牌尤克", "", "game"),
    "Bridge": ("桥牌", "", "game"),
    "Social Bridge": ("社交桥牌", "", "game"),
    "Billiards": ("台球", "", "game"),
    "Darts": ("飞镖", "", "game"),
    "Dominoes": ("骨牌", "", "game"),
    "Knitting": ("编织", "", "hobby"),
    "Crafts": ("手工艺", "", "hobby"),
    "Karaoke": ("卡拉OK", "", "hobby"),
    "Wellness Room": ("保健室", "", "hobby"),
    "Chess": ("象棋", "", "game"),
}

# Per-visit drop-in fee once you hold a 55+ membership. Pickleball is priced
# differently from everything else; everything else shares the flat rate.
DEFAULT_DROP_IN_FEE = "$1.30"
FEE_OVERRIDES = {
    "Pickleball": "$3.15",
}

DAY_HEADER_MAP = {
    "sunday": "Sun", "monday": "Mon", "tuesday": "Tue", "wednesday": "Wed",
    "thursday": "Thu", "friday": "Fri", "saturday": "Sat",
}


def translate(activity_raw: str):
    """Match the scraped activity label against TRANSLATIONS, stripping any
    parenthetical note like '(pre-registration required)' first, and pull
    out a note (e.g. booking requirement, skill level) separately."""
    note = ""
    base = activity_raw
    m = re.search(r"\(([^)]+)\)", activity_raw)
    if m:
        note_raw = m.group(1)
        base = activity_raw[: m.start()].strip()
        if "pre-registration" in note_raw.lower() or "preregist" in note_raw.lower():
            note = "需提前预约"
        elif "beginner" in note_raw.lower():
            note = (note + "，初学者" if note else "初学者")
        else:
            note = note_raw

    if "*ALL PREREGISTERED" in activity_raw.upper():
        note = "需提前预约"
        base = re.sub(r"\*ALL PREREGISTERED", "", base, flags=re.I).strip()

    zh, en_display, category = TRANSLATIONS.get(base.strip(), (base.strip(), "", "hobby"))
    return zh, en_display, category, note


def normalize_time(t: str) -> str:
    """Convert scraped 12-hour times like '4:15 – 6:15 p.m.' (or bare-hour
    forms like '7 – 9 p.m.') into 24-hour 'HH:MM – HH:MM', matching the
    format used everywhere else on the site."""
    # First pad any bare hour (no colon) that precedes a dash or am/pm marker.
    t = re.sub(r"(?<!:)\b(\d{1,2})\b(?=\s*(?:[–-]|a\.m\.|p\.m\.))", r"\1:00", t)

    tokens = list(re.finditer(r"(\d{1,2}):(\d{2})\s*(a\.m\.|p\.m\.)?", t))
    if len(tokens) != 2:
        return t  # unexpected shape — leave untouched rather than guess

    (h1, m1, p1), (h2, m2, p2) = (
        (int(g[0]), g[1], g[2]) for g in (m.groups() for m in tokens)
    )
    if p1 is None and p2 is not None:
        p1 = p2
    if p2 is None and p1 is not None:
        p2 = p1

    def to24(h, period):
        if period == "a.m.":
            return 0 if h == 12 else h
        return 12 if h == 12 else h + 12

    return f"{to24(h1, p1):02d}:{m1} – {to24(h2, p2):02d}:{m2}"


def parse_time_sort_key(time_str: str):
    """Best-effort sort key so activities within a day list roughly by start
    time. Falls back to 0 (keeps original order) if it can't parse."""
    m = re.match(r"\s*(\d{1,2})(?::(\d{2}))?\s*([ap])", time_str.lower())
    if not m:
        return 0
    hour = int(m.group(1)) % 12
    minute = int(m.group(2) or 0)
    if m.group(3) == "p":
        hour += 12
    return hour * 60 + minute


def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


def slice_section(html: str, heading_text: str) -> str:
    """Return the chunk of HTML between the heading matching heading_text and
    the next heading of the same level, so we don't accidentally parse a
    different community centre's table."""
    soup = BeautifulSoup(html, "lxml")
    heading = None
    for tag in soup.find_all(["h2", "h3"]):
        if heading_text.lower() in tag.get_text(strip=True).lower():
            heading = tag
            break
    if heading is None:
        raise ValueError(f"Could not find heading '{heading_text}' on the page")

    chunk_parts = []
    for sib in heading.find_all_next():
        if sib.name in ("h2", "h3") and sib is not heading:
            break
        chunk_parts.append(str(sib))
    return "".join(chunk_parts)


def find_current_month_table(section_html: str) -> pd.DataFrame:
    """The page lists one table per month (e.g. captioned 'July: ...',
    'August: ...'). Pick the table whose nearby caption text contains the
    current month name; fall back to the last table in the section if that
    fails (usually the most recent one)."""
    month_name = date.today().strftime("%B")  # e.g. "August"
    soup = BeautifulSoup(section_html, "lxml")
    tables = soup.find_all("table")
    if not tables:
        raise ValueError("No tables found in this section")

    chosen = None
    for t in tables:
        # look at text just before this table (caption / preceding siblings)
        preceding_text = ""
        prev = t.find_previous(string=True)
        hop = 0
        node = t
        while node is not None and hop < 6:
            node = node.find_previous(string=True)
            if node:
                preceding_text += " " + str(node)
            hop += 1
        if month_name.lower() in preceding_text.lower():
            chosen = t
            break

    if chosen is None:
        chosen = tables[-1]  # fall back to last table = most recently listed month

    dfs = pd.read_html(StringIO(str(chosen)), header=0)
    if not dfs:
        raise ValueError("pandas could not parse the chosen table")
    return dfs[0]


def table_to_days(df: pd.DataFrame):
    """Convert the raw dataframe (rows = activities, columns = days) into the
    {day_code: [ {en, zh, category, time, note}, ... ]} structure."""
    df.columns = [str(c).strip() for c in df.columns]
    col_to_day = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in DAY_HEADER_MAP:
            col_to_day[col] = DAY_HEADER_MAP[key]

    if not col_to_day:
        raise ValueError(f"No day-of-week columns recognised: {list(df.columns)}")

    activity_col = df.columns[0]
    days = {code: [] for code in DAY_HEADER_MAP.values()}

    for _, row in df.iterrows():
        activity_raw = str(row[activity_col]).strip()
        if not activity_raw or activity_raw.lower() == "nan":
            continue
        zh, en_display, category, note = translate(activity_raw)
        base_name = re.sub(r"\s*\([^)]*\)", "", activity_raw).strip()
        fee = FEE_OVERRIDES.get(base_name, DEFAULT_DROP_IN_FEE)
        need_booking = "预约" in note or base_name == "Pickleball"
        for col, day_code in col_to_day.items():
            cell = str(row[col]).strip()
            if not cell or cell.lower() == "nan":
                continue
            days[day_code].append({
                "en": activity_raw,
                "zh": zh,
                "en_display": en_display,
                "category": category,
                "time": normalize_time(cell),
                "fee": fee,
                "need_booking": need_booking,
                "note": note,
            })

    for day_code in days:
        days[day_code].sort(key=lambda item: parse_time_sort_key(item["time"]))

    return days


def main():
    try:
        html = fetch_html(URL)
        section = slice_section(html, CENTRE_HEADING_TEXT)
        df = find_current_month_table(section)
        days = table_to_days(df)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: see module docstring
        print(f"[scrape_and_build] FAILED to update schedule: {exc}", file=sys.stderr)
        print("[scrape_and_build] Leaving existing data.json untouched.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = {"categories": {}}

    existing.setdefault("categories", {})
    old_55plus = existing["categories"].get("55plus", {})

    # NOTE: only the "55plus" category is auto-scraped (the official 55+ page
    # has one clean table we can parse reliably). The "adult" category (the
    # all-ages fitness classes) comes from a much messier multi-season,
    # multi-centre page that isn't safe to auto-parse yet — it's seeded by
    # hand in data.json and this script leaves it untouched. If you want to
    # automate that too later, ask Claude / Claude Code to extend this
    # script rather than editing data.json's "adult" block directly, so the
    # two don't drift out of sync.
    existing["categories"]["55plus"] = {
        "label_zh": old_55plus.get("label_zh", "55+ 专属活动"),
        "subtitle_zh": old_55plus.get("subtitle_zh", "需要 Adults 55+ 会员"),
        "updated": date.today().isoformat(),
        "period_label": f"{date.today().strftime('%Y年%m月')}排班",
        "membership_note": old_55plus.get("membership_note", ""),
        "closures": old_55plus.get("closures", []),  # closures aren't auto-parsed yet; edit by hand if needed
        "days": days,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print("[scrape_and_build] data.json ('55plus' category) updated successfully.")


if __name__ == "__main__":
    main()
