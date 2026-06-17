import sqlite3
import os
import random
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "calibration.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))
READING_DIR = os.path.join(UPLOAD_DIR, "readings")
TRAINING_DIR = os.path.join(UPLOAD_DIR, "training")

# ── Name pools for anonymised author tags ─────────────────────────────────────
_TURKISH_FIRST = [
    "Şener","Ayşe","Kemal","Fatma","Murat","Zeynep","Ahmet","Selin","Burak","Deniz",
    "Emre","Gül","Hasan","İpek","Koray","Leyla","Mehmet","Neslihan","Ozan","Pınar",
    "Rana","Sercan","Tuba","Ufuk","Vildan","Yasemin","Zafer","Alper","Beren","Ceren",
]
_TURKISH_LAST = [
    "Yıldız","Kaya","Demir","Çelik","Şahin","Doğan","Arslan","Koç","Öztürk","Aydın",
    "Yılmaz","Polat","Güneş","Tekin","Çetin","Aksoy","Şen","Tan","Çoban","Acar",
    "Güler","Turan","Kaplan","Aslan","Yavuz","Demirci","Özcan","Bulut","Karahan","Erdoğan",
]
_ENGLISH_FIRST = [
    "Charlie","Sarah","Alice","Henry","Emma","James","Lucy","Oliver","Claire","George",
    "Sophie","William","Kate","Thomas","Emily","Arthur","Rose","Edward","Grace","Samuel",
    "Laura","Peter","Anna","Jack","Mia","Richard","Julia","Daniel","Helen","Mark",
]
_ENGLISH_LAST = [
    "Chaplin","Mitchell","Turner","Clarke","Hughes","Fletcher","Shaw","Reynolds","Walker","Hall",
    "Robinson","Davies","Evans","Wilson","Thomas","White","Moore","Taylor","Martin","Anderson",
    "Thompson","Jackson","Harris","Brown","Jones","Smith","Williams","Lee","Khan","Chen",
]


def generate_author_tag():
    """Return a random mixed Turkish-English pseudonym."""
    if random.random() < 0.5:
        return f"{random.choice(_TURKISH_FIRST)} {random.choice(_ENGLISH_LAST)}"
    return f"{random.choice(_ENGLISH_FIRST)} {random.choice(_TURKISH_LAST)}"


def generate_system_id():
    return "SMP-" + uuid.uuid4().hex[:8].upper()


# ── Core DB helpers ───────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _safe_add_column(cursor, table, column, col_type):
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:
        pass


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'instructor',
            bilkent_id TEXT,
            avg_leniency REAL DEFAULT 0,
            avg_distance REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS essays (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            essay_number INTEGER NOT NULL,
            title TEXT NOT NULL,
            notes TEXT,
            task_question TEXT,
            author_tag TEXT,
            source_instructor_bilkent_id TEXT,
            system_id TEXT,
            reading_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS essay_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            essay_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            page_number INTEGER NOT NULL DEFAULT 1,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (essay_id) REFERENCES essays(id)
        );
        CREATE TABLE IF NOT EXISTS gold_standards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            essay_id INTEGER NOT NULL UNIQUE,
            task_req INTEGER, argument INTEGER, support INTEGER,
            language INTEGER, readability INTEGER, formatting INTEGER,
            score_variance REAL,
            score_std_dev REAL,
            set_by INTEGER, set_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (essay_id) REFERENCES essays(id),
            FOREIGN KEY (set_by) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            essay_id INTEGER NOT NULL,
            task_req INTEGER, argument INTEGER, support INTEGER,
            language INTEGER, readability INTEGER, formatting INTEGER,
            certainty INTEGER,
            comments TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, essay_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (essay_id) REFERENCES essays(id)
        );
        CREATE TABLE IF NOT EXISTS rater_assignments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            essay_id INTEGER NOT NULL,
            position REAL NOT NULL,
            UNIQUE(user_id, essay_id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (essay_id) REFERENCES essays(id)
        );
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
    """)

    # Safe migrations for existing databases
    for col, ctype in [
        ("bilkent_id",   "TEXT"),
        ("avg_leniency", "REAL DEFAULT 0"),
        ("avg_distance", "REAL DEFAULT 0"),
    ]:
        _safe_add_column(c, "users", col, ctype)

    for col, ctype in [
        ("task_question",               "TEXT"),
        ("author_tag",                  "TEXT"),
        ("source_instructor_bilkent_id","TEXT"),
        ("system_id",                   "TEXT"),
        ("reading_filename",            "TEXT"),
    ]:
        _safe_add_column(c, "essays", col, ctype)

    _safe_add_column(c, "scores", "certainty", "INTEGER")

    for col, ctype in [
        ("score_variance", "REAL"),
        ("score_std_dev",  "REAL"),
    ]:
        _safe_add_column(c, "gold_standards", col, ctype)

    # Standardizations table
    c.execute("""
        CREATE TABLE IF NOT EXISTS standardizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            year INTEGER NOT NULL,
            term TEXT,
            reading_task TEXT,
            reading_filename TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            archived_at TIMESTAMP
        )
    """)

    # Migrate existing standardizations tables that are missing new columns
    _safe_add_column(c, "standardizations", "term", "TEXT")
    _safe_add_column(c, "standardizations", "reading_task", "TEXT")
    _safe_add_column(c, "standardizations", "reading_filename", "TEXT")
    _safe_add_column(c, "standardizations", "archived_at", "TIMESTAMP")

    # Link essays to standardizations
    _safe_add_column(c, "essays", "standardization_id", "INTEGER REFERENCES standardizations(id)")

    # Migrate orphan essays into a "Legacy" standardization
    import datetime as _dt
    orphans = c.execute("SELECT id FROM essays WHERE standardization_id IS NULL").fetchall()
    if orphans:
        existing_legacy = c.execute(
            "SELECT id FROM standardizations WHERE name = 'Legacy Standardization'"
        ).fetchone()
        if not existing_legacy:
            c.execute(
                "INSERT INTO standardizations (name, year, term, status) VALUES (?, ?, ?, ?)",
                ("Legacy Standardization", _dt.datetime.now().year, "Legacy", "archived"),
            )
            legacy_id = c.lastrowid
        else:
            legacy_id = existing_legacy["id"]
        c.execute(
            "UPDATE essays SET standardization_id = ? WHERE standardization_id IS NULL",
            (legacy_id,),
        )

    # Backfill system_id for existing essays
    essays_no_sid = c.execute("SELECT id FROM essays WHERE system_id IS NULL").fetchall()
    for e in essays_no_sid:
        c.execute("UPDATE essays SET system_id = ? WHERE id = ?", (generate_system_id(), e["id"]))

    # Default phase
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('phase', 'data_collection')")

    # Default admin account
    if not c.execute("SELECT id FROM users WHERE role='admin'").fetchone():
        c.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
            ("Admin", "admin@eng101.com", generate_password_hash("admin2025"), "admin"),
        )
    # Training tables
    c.executescript("""
        CREATE TABLE IF NOT EXISTS training_materials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_type TEXT NOT NULL,
            description TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS training_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            color TEXT DEFAULT 'yellow',
            pos_x INTEGER DEFAULT 50,
            pos_y INTEGER DEFAULT 50,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(READING_DIR, exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_user_by_email(email):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    conn.close()
    return user


def verify_password(user, password):
    return check_password_hash(user["password_hash"], password)


# ── App settings / phase ──────────────────────────────────────────────────────

def get_phase():
    conn = get_db()
    row = conn.execute("SELECT value FROM app_settings WHERE key = 'phase'").fetchone()
    conn.close()
    return row["value"] if row else "data_collection"


def set_phase(phase):
    conn = get_db()
    conn.execute(
        "INSERT INTO app_settings (key, value) VALUES ('phase', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (phase,),
    )
    conn.commit()
    conn.close()


# ── Essays ────────────────────────────────────────────────────────────────────

def get_all_essays():
    conn = get_db()
    essays = conn.execute("SELECT * FROM essays ORDER BY essay_number").fetchall()
    conn.close()
    return essays


def get_essay(essay_id):
    conn = get_db()
    essay = conn.execute("SELECT * FROM essays WHERE id = ?", (essay_id,)).fetchone()
    conn.close()
    return essay


def get_essay_images(essay_id):
    conn = get_db()
    images = conn.execute(
        "SELECT * FROM essay_images WHERE essay_id = ? ORDER BY page_number",
        (essay_id,),
    ).fetchall()
    conn.close()
    return images


# ── Scores ────────────────────────────────────────────────────────────────────

def get_scores_for_user(user_id):
    conn = get_db()
    scores = conn.execute(
        "SELECT s.*, e.title, e.essay_number FROM scores s "
        "JOIN essays e ON s.essay_id = e.id WHERE s.user_id = ?",
        (user_id,),
    ).fetchall()
    conn.close()
    return scores


def get_all_scores():
    conn = get_db()
    scores = conn.execute("""
        SELECT s.*, u.name AS instructor_name, u.bilkent_id AS instructor_bilkent_id,
               e.title AS essay_title, e.essay_number, e.author_tag,
               e.source_instructor_bilkent_id
        FROM scores s
        JOIN users u ON s.user_id = u.id
        JOIN essays e ON s.essay_id = e.id
        ORDER BY e.essay_number, u.name
    """).fetchall()
    conn.close()
    return scores


def get_gold_standard(essay_id):
    conn = get_db()
    gs = conn.execute("SELECT * FROM gold_standards WHERE essay_id = ?", (essay_id,)).fetchone()
    conn.close()
    return gs


def get_score_for_user_essay(user_id, essay_id):
    conn = get_db()
    score = conn.execute(
        "SELECT * FROM scores WHERE user_id = ? AND essay_id = ?", (user_id, essay_id)
    ).fetchone()
    conn.close()
    return score


# ── Instructors ───────────────────────────────────────────────────────────────

def get_all_instructors():
    conn = get_db()
    users = conn.execute(
        "SELECT * FROM users WHERE role = 'instructor' ORDER BY name"
    ).fetchall()
    conn.close()
    return users


def get_submission_count(user_id):
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM scores WHERE user_id = ?", (user_id,)
    ).fetchone()["cnt"]
    conn.close()
    return count


# ── Rater assignments ─────────────────────────────────────────────────────────

def get_rater_assignments(user_id):
    """Return all assignment rows for a rater, ordered by position."""
    conn = get_db()
    rows = conn.execute(
        "SELECT ra.essay_id, ra.position FROM rater_assignments ra "
        "WHERE ra.user_id = ? ORDER BY ra.position",
        (user_id,),
    ).fetchall()
    conn.close()
    return rows


def get_next_essay_for_rater(user_id):
    """Return the next unscored essay_id in this rater's assigned order, or None."""
    conn = get_db()
    row = conn.execute("""
        SELECT ra.essay_id FROM rater_assignments ra
        LEFT JOIN scores s ON s.user_id = ra.user_id AND s.essay_id = ra.essay_id
        WHERE ra.user_id = ? AND s.id IS NULL
        ORDER BY ra.position ASC
        LIMIT 1
    """, (user_id,)).fetchone()
    conn.close()
    return row["essay_id"] if row else None


def get_rater_position_info(user_id, essay_id):
    """Return (position_label, total_assigned) for a given rater and essay."""
    conn = get_db()
    all_assigned = conn.execute(
        "SELECT essay_id FROM rater_assignments WHERE user_id = ? ORDER BY position",
        (user_id,),
    ).fetchall()
    total = len(all_assigned)
    pos = next((i + 1 for i, r in enumerate(all_assigned) if r["essay_id"] == essay_id), None)
    conn.close()
    return pos, total


def generate_assignments_for_rater(user_id):
    """
    Create or refresh randomised essay assignments for one rater.
    Scored essays keep their historical position.
    Unscored essays (including any newly added) are re-shuffled after the last scored position.
    Essays written by this rater's own students are excluded.
    Only essays belonging to active standardizations are included.
    """
    conn = get_db()

    user_row = conn.execute("SELECT bilkent_id FROM users WHERE id = ?", (user_id,)).fetchone()
    bilkent_id = user_row["bilkent_id"] if user_row else None

    # Only assign essays from active standardizations (or unassigned essays)
    all_essay_ids = [r["id"] for r in conn.execute("""
        SELECT e.id FROM essays e
        LEFT JOIN standardizations s ON e.standardization_id = s.id
        WHERE s.status = 'active' OR e.standardization_id IS NULL
    """).fetchall()]

    if bilkent_id:
        excluded = {r["id"] for r in conn.execute(
            "SELECT id FROM essays WHERE source_instructor_bilkent_id = ?", (bilkent_id,)
        ).fetchall()}
    else:
        excluded = set()

    eligible = [eid for eid in all_essay_ids if eid not in excluded]

    scored_ids = {r["essay_id"] for r in conn.execute(
        "SELECT essay_id FROM scores WHERE user_id = ?", (user_id,)
    ).fetchall()}

    existing = {r["essay_id"]: r["position"] for r in conn.execute(
        "SELECT essay_id, position FROM rater_assignments WHERE user_id = ?", (user_id,)
    ).fetchall()}

    scored_positions = {eid: existing[eid] for eid in eligible if eid in existing and eid in scored_ids}
    max_scored_pos = max(scored_positions.values(), default=0)

    unscored_eligible = [eid for eid in eligible if eid not in scored_ids]
    random.shuffle(unscored_eligible)

    for eid in unscored_eligible:
        conn.execute(
            "DELETE FROM rater_assignments WHERE user_id = ? AND essay_id = ?", (user_id, eid)
        )

    for i, eid in enumerate(unscored_eligible):
        conn.execute(
            "INSERT OR IGNORE INTO rater_assignments (user_id, essay_id, position) VALUES (?, ?, ?)",
            (user_id, eid, max_scored_pos + i + 1),
        )

    conn.commit()
    conn.close()


def assign_new_essay_to_all_raters(essay_id):
    """
    When a new essay is added, slot it into each rater's unscored queue at a random position.
    Skips raters for whom this essay is excluded (self-grading rule).
    Only assigns if the essay belongs to an active standardization (or has none).
    """
    conn = get_db()
    essay_row = conn.execute(
        "SELECT source_instructor_bilkent_id, standardization_id FROM essays WHERE id = ?", (essay_id,)
    ).fetchone()
    source_bilkent = essay_row["source_instructor_bilkent_id"] if essay_row else None

    # Skip assignment if essay belongs to an archived standardization
    if essay_row and essay_row["standardization_id"]:
        std_row = conn.execute(
            "SELECT status FROM standardizations WHERE id = ?", (essay_row["standardization_id"],)
        ).fetchone()
        if std_row and std_row["status"] != "active":
            conn.close()
            return

    instructors = conn.execute(
        "SELECT id, bilkent_id FROM users WHERE role = 'instructor'"
    ).fetchall()

    for instr in instructors:
        if source_bilkent and instr["bilkent_id"] == source_bilkent:
            continue
        if conn.execute(
            "SELECT id FROM rater_assignments WHERE user_id = ? AND essay_id = ?",
            (instr["id"], essay_id),
        ).fetchone():
            continue

        scored_ids = {r["essay_id"] for r in conn.execute(
            "SELECT essay_id FROM scores WHERE user_id = ?", (instr["id"],)
        ).fetchall()}

        unscored = conn.execute(
            "SELECT essay_id, position FROM rater_assignments WHERE user_id = ? ORDER BY position",
            (instr["id"],),
        ).fetchall()
        unscored_rows = [r for r in unscored if r["essay_id"] not in scored_ids]
        max_pos = conn.execute(
            "SELECT MAX(position) as mp FROM rater_assignments WHERE user_id = ?",
            (instr["id"],),
        ).fetchone()["mp"] or 0

        if unscored_rows:
            positions = [r["position"] for r in unscored_rows]
            insert_pos = random.uniform(min(positions) - 0.5, max(positions) + 0.5)
        else:
            insert_pos = max_pos + 1

        conn.execute(
            "INSERT OR IGNORE INTO rater_assignments (user_id, essay_id, position) VALUES (?, ?, ?)",
            (instr["id"], essay_id, insert_pos),
        )

    conn.commit()
    conn.close()


# ── Metrics ───────────────────────────────────────────────────────────────────

def recompute_instructor_metrics():
    """
    Recompute avg_leniency and avg_distance for every instructor and store
    the results back in the users table.
    """
    import statistics as st
    conn = get_db()

    essays = conn.execute("SELECT id FROM essays").fetchall()
    essay_ids = [e["id"] for e in essays]
    instructors = conn.execute("SELECT id FROM users WHERE role = 'instructor'").fetchall()

    all_scores = conn.execute("SELECT * FROM scores").fetchall()
    matrix = {}
    for s in all_scores:
        matrix.setdefault(s["essay_id"], {})[s["user_id"]] = s

    gold = {}
    for eid in essay_ids:
        gs = conn.execute(
            "SELECT * FROM gold_standards WHERE essay_id = ?", (eid,)
        ).fetchone()
        if gs:
            gold[eid] = gs

    cats = ["task_req", "argument", "support", "language", "readability", "formatting"]

    def total(row):
        return sum(row[c] or 0 for c in cats)

    all_totals = [total(s) for s in all_scores]
    group_mean = st.mean(all_totals) if all_totals else 0

    for instr in instructors:
        uid = instr["id"]
        my_scores = [s for s in all_scores if s["user_id"] == uid]
        if not my_scores:
            conn.execute(
                "UPDATE users SET avg_leniency=0, avg_distance=0 WHERE id=?", (uid,)
            )
            continue

        my_totals = [total(s) for s in my_scores]
        my_mean = st.mean(my_totals)
        leniency = round(my_mean - group_mean, 3)

        dists = []
        for s in my_scores:
            eid = s["essay_id"]
            if eid in gold:
                gs_total = total(gold[eid])
                dists.append(abs(total(s) - gs_total))
        avg_dist = round(st.mean(dists), 3) if dists else None

        conn.execute(
            "UPDATE users SET avg_leniency=?, avg_distance=? WHERE id=?",
            (leniency, avg_dist, uid),
        )

    conn.commit()
    conn.close()


# ── Standardizations ──────────────────────────────────────────────────────────

def create_standardization(name, year, term, reading_task=None, reading_filename=None):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO standardizations (name, year, term, reading_task, reading_filename) VALUES (?, ?, ?, ?, ?)",
        (name, int(year), term or None, reading_task or None, reading_filename or None),
    )
    std_id = cur.lastrowid
    conn.commit()
    conn.close()
    return std_id


def get_standardization(std_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM standardizations WHERE id = ?", (std_id,)).fetchone()
    conn.close()
    return row


def get_all_standardizations():
    conn = get_db()
    rows = conn.execute("""
        SELECT s.*, COUNT(e.id) AS essay_count
        FROM standardizations s
        LEFT JOIN essays e ON e.standardization_id = s.id
        GROUP BY s.id
        ORDER BY s.year DESC, s.term DESC, s.created_at DESC
    """).fetchall()
    conn.close()
    return rows


def get_active_standardizations():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM standardizations WHERE status = 'active' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows


def archive_standardization(std_id):
    conn = get_db()
    conn.execute(
        "UPDATE standardizations SET status='archived', archived_at=CURRENT_TIMESTAMP WHERE id=?",
        (std_id,),
    )
    conn.commit()
    conn.close()


def reopen_standardization(std_id):
    conn = get_db()
    conn.execute(
        "UPDATE standardizations SET status='active', archived_at=NULL WHERE id=?",
        (std_id,),
    )
    conn.commit()
    conn.close()


def get_essays_by_standardization(std_id):
    conn = get_db()
    essays = conn.execute(
        "SELECT * FROM essays WHERE standardization_id = ? ORDER BY essay_number",
        (std_id,),
    ).fetchall()
    conn.close()
    return essays


def get_cross_standardization_stats():
    """
    Return per-standardization instructor performance for cross-standardization comparison.
    Each entry: {standardization: row, essay_count: int, group_mean: float,
                 instructor_stats: [{id, name, bilkent_id, essays_scored,
                                     mean_total, leniency, tendency, avg_distance}]}
    """
    import statistics as st
    conn = get_db()

    standardizations = conn.execute(
        "SELECT * FROM standardizations ORDER BY year ASC, term ASC"
    ).fetchall()

    cats = ["task_req", "argument", "support", "language", "readability", "formatting"]

    def total(row):
        return sum(row[c] or 0 for c in cats)

    result = []
    for std in standardizations:
        std_id = std["id"]
        essay_rows = conn.execute(
            "SELECT id FROM essays WHERE standardization_id = ?", (std_id,)
        ).fetchall()
        essay_ids = [r["id"] for r in essay_rows]
        if not essay_ids:
            continue

        placeholders = ",".join("?" * len(essay_ids))
        scores = conn.execute(
            f"SELECT s.*, u.name AS instructor_name, u.bilkent_id AS instructor_bilkent_id "
            f"FROM scores s JOIN users u ON s.user_id = u.id "
            f"WHERE s.essay_id IN ({placeholders})",
            essay_ids,
        ).fetchall()

        if not scores:
            continue

        all_totals = [total(s) for s in scores]
        group_mean = st.mean(all_totals) if all_totals else 0

        by_instructor = {}
        for s in scores:
            uid = s["user_id"]
            if uid not in by_instructor:
                by_instructor[uid] = {
                    "id": uid,
                    "name": s["instructor_name"],
                    "bilkent_id": s["instructor_bilkent_id"],
                    "scores": [],
                }
            by_instructor[uid]["scores"].append(s)

        gold = {}
        for eid in essay_ids:
            gs = conn.execute(
                "SELECT * FROM gold_standards WHERE essay_id = ?", (eid,)
            ).fetchone()
            if gs:
                gold[eid] = gs

        instr_stats = []
        for uid, data in by_instructor.items():
            totals = [total(s) for s in data["scores"]]
            mean_t = st.mean(totals)
            leniency = round(mean_t - group_mean, 2)

            dists = []
            for s in data["scores"]:
                eid = s["essay_id"]
                if eid in gold:
                    gs_total = total(gold[eid])
                    dists.append(abs(total(s) - gs_total))

            instr_stats.append({
                "id": uid,
                "name": data["name"],
                "bilkent_id": data["bilkent_id"] or "",
                "essays_scored": len(data["scores"]),
                "mean_total": round(mean_t, 2),
                "leniency": leniency,
                "tendency": (
                    "Lenient" if leniency > 1.0 else
                    "Harsh" if leniency < -1.0 else
                    "Calibrated"
                ),
                "avg_distance": round(st.mean(dists), 2) if dists else None,
            })

        result.append({
            "standardization": dict(std),
            "essay_count": len(essay_ids),
            "group_mean": round(group_mean, 2),
            "instructor_stats": sorted(instr_stats, key=lambda x: x["name"]),
        })

    conn.close()
    return result


# ── Training ──────────────────────────────────────────────────────────────────

def get_training_materials():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM training_materials ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return rows


def add_training_material(filename, original_name, file_type, description=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO training_materials (filename, original_name, file_type, description) VALUES (?, ?, ?, ?)",
        (filename, original_name, file_type, description),
    )
    conn.commit()
    conn.close()


def delete_training_material(material_id):
    conn = get_db()
    row = conn.execute("SELECT filename FROM training_materials WHERE id=?", (material_id,)).fetchone()
    conn.execute("DELETE FROM training_materials WHERE id=?", (material_id,))
    conn.commit()
    conn.close()
    return row["filename"] if row else None


def get_training_notes():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM training_notes ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return rows


def add_training_note(content, color, pos_x=50, pos_y=50):
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO training_notes (content, color, pos_x, pos_y) VALUES (?, ?, ?, ?)",
        (content, color, pos_x, pos_y),
    )
    note_id = cur.lastrowid
    conn.commit()
    conn.close()
    return note_id


def update_training_note_position(note_id, pos_x, pos_y):
    conn = get_db()
    conn.execute(
        "UPDATE training_notes SET pos_x=?, pos_y=? WHERE id=?",
        (pos_x, pos_y, note_id),
    )
    conn.commit()
    conn.close()


def delete_training_note(note_id):
    conn = get_db()
    conn.execute("DELETE FROM training_notes WHERE id=?", (note_id,))
    conn.commit()
    conn.close()
