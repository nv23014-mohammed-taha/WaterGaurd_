"""
Plant Care Tracker — single-file application (interactive + scripted fallback)

This file fixes an OSError: [Errno 29] I/O error that appears when running in
sandboxed environments where interactive `input()` is not supported.

Approach taken:
- Provide a safe `safe_input()` wrapper that tries `input()` but falls back to a
  scripted input source when interactive input raises OSError.
- Allow running the program in a fully non-interactive/scripted mode by
  supplying `--script <file>` on the command line. The script file contains one
  command per line and simulates user responses.
- Provide `--test` mode which runs a small set of automated tests (no user
  input) to validate core flows and CSV read/write behavior.
- Keep the original CLI menu behavior when interactive input is available.

Usage:
- Interactive (normal terminal): python plant_care_tracker.py
- Scripted (non-interactive): python plant_care_tracker.py --script commands.txt
  where commands.txt contains the answers the program would normally read from
  the user (one per prompt).
- Run tests (no interactive input): python plant_care_tracker.py --test

Notes for the user:
- If you'd like the program to behave differently when run non-interactively
  (e.g. use JSON instead of line-based scripts), tell me what you expect and I
  will adapt the format.
"""

import csv
import os
import sys
from datetime import datetime, timedelta
import uuid
from typing import List, Optional, Iterator

PLANTS_FILE = "plants.csv"
ACTIVITIES_FILE = "activities.csv"
GROWTH_FILE = "growth.csv"

DATE_FMT = "%Y-%m-%d"

# --- Safe input handling (fixes OSError in sandbox) ----------------------------

class ScriptedInput:
    """Provides scripted inputs (lines) to simulate user responses."""
    def __init__(self, lines: Optional[List[str]] = None):
        self.lines = (lines or [])
        self.index = 0

    def next(self, prompt: str = "") -> str:
        if self.index < len(self.lines):
            val = self.lines[self.index]
            self.index += 1
            # echo the prompt and the scripted response so logs show what's happening
            print(f"{prompt}{val}")
            return val
        # If script exhausted, return empty string rather than crash
        print(f"{prompt}")
        return ""

# Global runtime holder for scripted input (set in main)
_SCRIPT: Optional[ScriptedInput] = None


def safe_input(prompt: str = "") -> str:
    """Try to read from interactive input; if that fails use scripted input.

    Catches OSError (common in sandbox/non-interactive environments) and
    falls back to reading from the global _SCRIPT object. If neither is
    available, returns an empty string.
    """
    global _SCRIPT
    try:
        return input(prompt)
    except (OSError, RuntimeError) as e:
        # input() not supported in environment — use scripted input if provided
        if _SCRIPT is not None:
            return _SCRIPT.next(prompt)
        # last resort: print prompt and return empty string
        print(prompt)
        return ""

# --- Helpers for CSV persistence -------------------------------------------------

def ensure_csv(file_path, fieldnames):
    """Create file with header if missing."""
    if not os.path.exists(file_path):
        # make sure directory exists
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def load_csv(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_csv(file_path, row, fieldnames):
    ensure_csv(file_path, fieldnames)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

# --- Data model helpers ---------------------------------------------------------

PLANT_FIELDS = [
    "id",
    "name",
    "location",
    "date_acquired",
    "watering_freq",
    "sunlight",
    "photo_path",
    "notes",
]

ACTIVITY_FIELDS = ["id", "plant_id", "activity_type", "date", "notes"]
GROWTH_FIELDS = ["id", "plant_id", "date", "height_cm", "notes"]


def generate_id():
    return str(uuid.uuid4())

# --- Core features --------------------------------------------------------------

def add_new_plant():
    print("\nAdd a new plant")
    name = safe_input("Plant name/species: ").strip()
    location = safe_input("Location in home: ").strip()
    date_acquired = safe_input(f"Date acquired ({DATE_FMT}) — leave blank for today: ").strip()
    if not date_acquired:
        date_acquired = datetime.today().strftime(DATE_FMT)
    watering_freq = safe_input("Watering frequency (in days): ").strip()
    try:
        watering_freq = int(watering_freq)
    except Exception:
        print("Invalid number — defaulting watering frequency to 7 days.")
        watering_freq = 7
    sunlight = safe_input("Sunlight needs (Low, Medium, High): ").strip().capitalize()
    photo_path = safe_input("(Optional) Photo file path: ").strip()
    notes = safe_input("Any notes: ").strip()

    plant = {
        "id": generate_id(),
        "name": name,
        "location": location,
        "date_acquired": date_acquired,
        "watering_freq": str(watering_freq),
        "sunlight": sunlight,
        "photo_path": photo_path,
        "notes": notes,
    }
    append_csv(PLANTS_FILE, plant, PLANT_FIELDS)
    print(f"Added plant '{name}'.\n")


def record_care_activity():
    plants = load_csv(PLANTS_FILE)
    if not plants:
        print("No plants found — add a plant first.\n")
        return
    print("\nRecord care activity")
    for i, p in enumerate(plants, 1):
        print(f"{i}. {p['name']} (Location: {p['location']})")
    choice = safe_input("Choose plant by number: ").strip()
    try:
        idx = int(choice) - 1
        plant = plants[idx]
    except Exception:
        print("Invalid choice.\n")
        return
    print("Activity types: 1) Watering 2) Fertilizing 3) Repotting 4) Pruning 5) Other")
    atype_map = {"1": "Watering", "2": "Fertilizing", "3": "Repotting", "4": "Pruning", "5": "Other"}
    achoice = safe_input("Choose activity: ").strip()
    activity_type = atype_map.get(achoice, "Other")
    notes = safe_input("Notes (optional): ").strip()
    date = datetime.today().strftime(DATE_FMT)
    activity = {"id": generate_id(), "plant_id": plant["id"], "activity_type": activity_type, "date": date, "notes": notes}
    append_csv(ACTIVITIES_FILE, activity, ACTIVITY_FIELDS)
    print(f"Recorded {activity_type} for {plant['name']} on {date}.\n")


def view_plants_due_for_care():
    plants = load_csv(PLANTS_FILE)
    activities = load_csv(ACTIVITIES_FILE)
    today = datetime.today().date()
    due_plants = []

    # Index last watering per plant
    last_water = {}
    for a in activities:
        if a.get("activity_type") == "Watering":
            try:
                d = datetime.strptime(a.get("date", ""), DATE_FMT).date()
                pid = a.get("plant_id")
                if pid not in last_water or d > last_water[pid]:
                    last_water[pid] = d
            except Exception:
                continue

    for p in plants:
        pid = p.get("id")
        try:
            freq = int(p.get("watering_freq") or 7)
        except Exception:
            freq = 7
        last = last_water.get(pid)
        if last is None:
            try:
                last = datetime.strptime(p.get("date_acquired", ""), DATE_FMT).date()
            except Exception:
                last = today
        next_due = last + timedelta(days=freq)
        days_until = (next_due - today).days
        if days_until <= 0:
            due_plants.append((p, days_until, next_due))

    if not due_plants:
        print("\nNo plants are due for watering today.\n")
        return

    print("\nPlants due for care:\n")
    for p, days_until, next_due in due_plants:
        print(f"- {p.get('name')} (Location: {p.get('location')}) — next watering was due on {next_due.strftime(DATE_FMT)}")
    print()


def search_plants():
    term = safe_input("\nSearch term (name or location): ").strip().lower()
    plants = load_csv(PLANTS_FILE)
    matches = [p for p in plants if term in p.get('name','').lower() or term in p.get('location','').lower()]
    if not matches:
        print("No matching plants found.\n")
        return
    print(f"\nFound {len(matches)} match(es):\n")
    for p in matches:
        print_plant_summary(p)


def view_all_plants():
    plants = load_csv(PLANTS_FILE)
    if not plants:
        print("No plants added yet.\n")
        return
    print("\nAll plants:\n")
    for p in plants:
        print_plant_summary(p)


def print_plant_summary(p):
    print(f"Name: {p.get('name')}")
    print(f"  ID: {p.get('id')}")
    print(f"  Location: {p.get('location')}")
    print(f"  Acquired: {p.get('date_acquired')}")
    print(f"  Water every: {p.get('watering_freq')} days")
    print(f"  Sunlight: {p.get('sunlight')}")
    if p.get('photo_path'):
        print(f"  Photo: {p.get('photo_path')}")
    if p.get('notes'):
        print(f"  Notes: {p.get('notes')}")
    print()

# --- Stretch features (basic implementations) ----------------------------------

def add_growth_measurement():
    plants = load_csv(PLANTS_FILE)
    if not plants:
        print("No plants found — add a plant first.\n")
        return
    print("\nAdd growth measurement")
    for i, p in enumerate(plants, 1):
        print(f"{i}. {p.get('name')}")
    idx = safe_input("Choose plant by number: ").strip()
    try:
        plant = plants[int(idx)-1]
    except Exception:
        print("Invalid choice.\n")
        return
    date = safe_input(f"Date ({DATE_FMT}) — leave blank for today: ").strip() or datetime.today().strftime(DATE_FMT)
    h = safe_input("Height in cm: ").strip()
    try:
        float(h)
    except Exception:
        print("Invalid height.\n")
        return
    notes = safe_input("Notes: ").strip()
    row = {"id": generate_id(), "plant_id": plant.get('id'), "date": date, "height_cm": h, "notes": notes}
    append_csv(GROWTH_FILE, row, GROWTH_FIELDS)
    print("Measurement saved.\n")


def seasonal_care_reminders():
    plants = load_csv(PLANTS_FILE)
    if not plants:
        print("No plants found.\n")
        return
    month = datetime.today().month
    season = ("Winter" if month in (12,1,2) else "Spring" if month in (3,4,5) else "Summer" if month in (6,7,8) else "Autumn")
    print(f"\nSeasonal reminders for {season}:")
    for p in plants:
        s = p.get('sunlight','').lower()
        if s == 'low':
            print(f"- {p.get('name')}: May need less frequent watering and move closer to light if indoors.")
        elif s == 'medium':
            print(f"- {p.get('name')}: Check soil moisture more often in hot months.")
        elif s == 'high':
            print(f"- {p.get('name')}: Watch for sunburn in very hot months; increase watering during summer.")
        else:
            print(f"- {p.get('name')}: General check — adjust care by observing soil moisture and leaf health.")
    print()


def adjust_care_schedule():
    plants = load_csv(PLANTS_FILE)
    if not plants:
        print("No plants found.\n")
        return
    print("\nAdjust care schedule (change watering frequency)")
    for i, p in enumerate(plants, 1):
        print(f"{i}. {p.get('name')} — every {p.get('watering_freq')} days")
    idx = safe_input("Choose plant by number: ").strip()
    try:
        plant = plants[int(idx)-1]
    except Exception:
        print("Invalid choice.\n")
        return
    new_freq = safe_input("New watering frequency (days): ").strip()
    try:
        new_freq = int(new_freq)
    except Exception:
        print("Invalid number.\n")
        return
    # update CSV (overwrite with modified rows)
    all_plants = load_csv(PLANTS_FILE)
    for row in all_plants:
        if row.get('id') == plant.get('id'):
            row['watering_freq'] = str(new_freq)
    with open(PLANTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PLANT_FIELDS)
        writer.writeheader()
        writer.writerows(all_plants)
    print(f"Updated watering frequency for {plant.get('name')} to {new_freq} days.\n")


def diagnose_plant():
    print("\nPlant diagnosis (simple heuristic)")
    print("Enter symptoms (comma separated): yellow leaves, brown spots, wilting, drooping, dry soil, pests")
    symptoms = safe_input("Symptoms: ").strip().lower().split(",")
    symptoms = [s.strip() for s in symptoms if s.strip()]
    advice = []
    if 'yellow leaves' in symptoms:
        advice.append("Possible overwatering or nutrient deficiency — check soil moisture and consider reducing watering.")
    if 'dry soil' in symptoms or 'wilting' in symptoms:
        advice.append("Likely underwatering — water thoroughly and check humidity.")
    if 'brown spots' in symptoms:
        advice.append("Could be fungal or sunburn — reduce direct sun and inspect for disease.")
    if 'pests' in symptoms:
        advice.append("Inspect underside of leaves, treat with insecticidal soap or remove pests manually.")
    if not advice:
        advice.append("Symptoms unclear — inspect plant, consider light, water, and pests.")
    print('\nDiagnosis suggestions:')
    for a in advice:
        print(f"- {a}")
    print()

# --- Small utility menu --------------------------------------------------------

def main_menu_loop():
    ensure_csv(PLANTS_FILE, PLANT_FIELDS)
    ensure_csv(ACTIVITIES_FILE, ACTIVITY_FIELDS)
    ensure_csv(GROWTH_FILE, GROWTH_FIELDS)

    while True:
        print("Plant Care Tracker — Menu")
        print("1) Add a new plant")
        print("2) Record a plant care activity")
        print("3) View plants due for care")
        print("4) Search plants by name or location")
        print("5) View all plants")
        print("6) Add growth measurement")
        print("7) Seasonal care reminders")
        print("8) Adjust care schedule")
        print("9) Diagnose plant (by symptoms)")
        print("0) Exit")
        choice = safe_input("Choose an option: ").strip()
        if choice == '1':
            add_new_plant()
        elif choice == '2':
            record_care_activity()
        elif choice == '3':
            view_plants_due_for_care()
        elif choice == '4':
            search_plants()
        elif choice == '5':
            view_all_plants()
        elif choice == '6':
            add_growth_measurement()
        elif choice == '7':
            seasonal_care_reminders()
        elif choice == '8':
            adjust_care_schedule()
        elif choice == '9':
            diagnose_plant()
        elif choice == '0':
            print("Goodbye!")
            break
        else:
            print("Unknown option — try again.\n")

# --- Script runner -------------------------------------------------------------

def run_script_lines(lines: List[str]):
    """Run the main menu with a provided list of scripted responses.

    The lines are consumed in order whenever `safe_input()` is called.
    """
    global _SCRIPT
    _SCRIPT = ScriptedInput(lines)
    try:
        main_menu_loop()
    finally:
        _SCRIPT = None

# --- Basic automated tests -----------------------------------------------------

def run_tests():
    """Run some simple tests to make sure program can write/read CSVs and run flows
    in a non-interactive mode. These are intentionally lightweight and filesystem
    isolated (use files in a temp folder).
    """
    print("Running tests...")

    # Use a temporary directory so tests don't clobber user's files if present.
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp(prefix="plant_test_")
    old_files = (PLANTS_FILE, ACTIVITIES_FILE, GROWTH_FILE)
    try:
        # point files to tmpdir
        global PLANTS_FILE, ACTIVITIES_FILE, GROWTH_FILE
        PLANTS_FILE = os.path.join(tmpdir, "plants.csv")
        ACTIVITIES_FILE = os.path.join(tmpdir, "activities.csv")
        GROWTH_FILE = os.path.join(tmpdir, "growth.csv")

        # Test 1: Add a plant programmatically
        p = {
            "id": "test-1",
            "name": "Test Plant",
            "location": "Living Room",
            "date_acquired": datetime.today().strftime(DATE_FMT),
            "watering_freq": "3",
            "sunlight": "Medium",
            "photo_path": "",
            "notes": "",
        }
        append_csv(PLANTS_FILE, p, PLANT_FIELDS)
        plants = load_csv(PLANTS_FILE)
        assert len(plants) == 1 and plants[0]['name'] == "Test Plant"
        print("Test 1 passed: add/load plant")

        # Test 2: Record watering activity and check due calculation
        activity = {"id": "a1", "plant_id": "test-1", "activity_type": "Watering", "date": datetime.today().strftime(DATE_FMT), "notes": ""}
        append_csv(ACTIVITIES_FILE, activity, ACTIVITY_FIELDS)
        activities = load_csv(ACTIVITIES_FILE)
        assert len(activities) == 1 and activities[0]['activity_type'] == "Watering"
        print("Test 2 passed: record/load activity")

        # Test 3: Run view_plants_due_for_care() — should not show plant due (just watered)
        print("-- view_plants_due_for_care() output (expected: no plants due) --")
        view_plants_due_for_care()
        print("Test 3 (manual inspection): view_plants_due_for_care ran")

        # Test 4: Scripted run to add a plant and then list all plants
        script_lines = [
            "1",  # Add new plant
            "Scripted Plant",  # name
            "Office",  # location
            "",  # date acquired -> today
            "5",  # watering freq
            "Low",  # sunlight
            "",  # photo path
            "note from script",  # notes
            "5",  # View all plants
            "0",  # Exit
        ]
        run_script_lines(script_lines)
        # load and confirm the scripted plant exists
        plants = load_csv(PLANTS_FILE)
        names = [r['name'] for r in plants]
        assert "Scripted Plant" in names
        print("Test 4 passed: scripted add & view")

        print("All tests passed.")
    finally:
        # cleanup
        shutil.rmtree(tmpdir)

# --- Command-line parsing and entry point -------------------------------------

def print_usage():
    print("Usage: python plant_care_tracker.py [--script <file>] [--test]")


def load_script_file(path: str) -> List[str]:
    if not os.path.exists(path):
        print(f"Script file not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        # strip trailing newlines but keep blank responses as empty strings
        return [line.rstrip('\n') for line in f]


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        # try interactive main menu — safe_input will fallback if not supported
        main_menu_loop()
    else:
        if args[0] in ("-h", "--help"):
            print_usage()
        elif args[0] == "--script":
            if len(args) < 2:
                print("--script requires a file path")
                print_usage()
            else:
                lines = load_script_file(args[1])
                run_script_lines(lines)
        elif args[0] == "--test":
            run_tests()
        else:
            print_usage()

