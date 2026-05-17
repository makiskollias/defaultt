import streamlit as st
import json
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Φόρτωση ρυθμίσεων
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Ρύθμιση Σελίδας
st.set_page_config(
    page_title="Vouli-AI: Βοηθός Νομοθεσίας",
    page_icon="🏛️",
    layout="centered",
)

# 2. Φόρτωση Δεδομένων
@st.cache_data
def load_knowledge_base():
    chunks = []
    if os.path.exists("chunks.jsonl"):
        with open("chunks.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks

chunks = load_knowledge_base()

st.markdown(
    """
    <style>
        /* Focused centred column (~800px) — not full-bleed desktop */
        .main .block-container {
            max-width: 800px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-left: clamp(1rem, 4vw, 1.35rem);
            padding-right: clamp(1rem, 4vw, 1.35rem);
            padding-top: 10px !important;
            padding-bottom: 1.55rem !important;
        }

        /* Page title — larger, tighter measure */
        .main .block-container h1 {
            font-size: clamp(1.9rem, 4.8vw, 2.15rem);
            font-weight: 650;
            line-height: 1.22 !important;
            letter-spacing: -0.017em;
        }

        /* Body / subtitles (Markdown in main pane) */
        .main .block-container .stMarkdown p {
            font-size: 1.09rem !important;
            line-height: 1.6 !important;
            letter-spacing: 0.01em;
        }

        /* Buttons — taller text, clearer line-height */
        .main .block-container .stButton > button {
            font-size: 0.98rem !important;
            line-height: 1.5 !important;
            padding-top: 0.52rem !important;
            padding-bottom: 0.52rem !important;
        }

        /* Chat transcripts (narrower specificity so they win vs body markdown) */
        .main .block-container section[data-testid="stChatMessage"] .stMarkdown p,
        .main .block-container section[data-testid="stChatMessage"] .stMarkdown li {
            font-size: 1.06rem !important;
            line-height: 1.6 !important;
        }

        [data-testid="stChatInput"] textarea {
            font-size: 1.035rem !important;
            line-height: 1.52 !important;
        }

        /* Sidebar typography */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 1.06rem !important;
            line-height: 1.45 !important;
        }
        [data-testid="stSidebar"] .stMarkdown p {
            font-size: 0.98rem !important;
            line-height: 1.55 !important;
        }
        [data-testid="stSidebar"] label p {
            font-size: 0.96rem !important;
            line-height: 1.45 !important;
        }
        [data-testid="stSidebar"] button {
            font-size: 0.97rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Πληροφορίες")
    st.write(
        "Ο βοηθός αντλεί από τα ενσωματωμένα επίσημα έγγραφα (PDF/κείμενα που έχουν φορτωθεί). "
        "Για επίκαιρη πληροφόρηση απαιτείται να υπάρχουν στο σύστημα τα αντίστοιχα κείμενα (π.χ. νέο νομοσχέδιο, συνημμένα, εκθέσεις)."
    )
    st.divider()

    top_k = st.slider("Αποσπάσματα (top-k)", 3, 12, 6, 1)
    min_sim = st.slider("Ελάχιστη ομοιότητα (threshold)", 0.10, 0.40, 0.18, 0.01)

    if chunks:
        sources = sorted(list(set([c.get('source', 'Άγνωστη Πηγή') for c in chunks])))
        st.subheader("Διαθέσιμες πηγές στη βάση")
        for s in sources:
            st.caption(f"• {s}")

# --- Κεντρικό μέτωπο ---
st.title("🏛️ Vouli-AI: Βοηθός Νομοθεσίας")

st.markdown(
    '<div style="height:0.45rem;" aria-hidden="true"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    "Το **Vouli-AI** βοηθά πολίτες και χρήστες να **κατανοήσουν** νόμους, **τρέχοντα νομοσχέδια**, "
    "συνεδριάσεις και νομοθετική διαδικασία. **Βασίζεται αποκλειστικά σε επίσημα κείμενα** που έχουν "
    "ενσωματωθεί στη βάση (πράξεις, σχέδια νόμου, εισηγήσεις, πρακτικά ή άλλα στοιχεία που έχουν εισαχθεί)· "
    "οι παραπομπές **Πηγή** και **Σελίδα** προέρχονται από τα αντίστοιχα αποσπάσματα."
)

st.markdown(
    '<p style="font-size:0.865rem;font-weight:500;color:rgba(105,113,129,0.96);margin:0.52rem 0 0;line-height:1.45;">'
    "Πώς μπορεί να βοηθήσει"
    '</p>'
    '<div style="font-size:0.93rem;color:rgba(115,120,133,0.94);line-height:1.5;margin:0.22rem 0 0;font-weight:400;">'
    "• Σύνοψη νομοσχεδίων<br>"
    "• Ανάλυση νόμων<br>"
    "• Κατανόηση συνεδριάσεων<br>"
    "• Επίσημες παραπομπές"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="height:0.95rem;" aria-hidden="true"></div>',
    unsafe_allow_html=True,
)

# --- Γρήγορη διάδραση κοντά στο κείμενο εισαγωγής ---
if "messages" not in st.session_state:
    st.session_state.messages = []

PRIMARY_DEMO = "Τι ψηφίστηκε στο τελευταίο νομοσχέδιο;"
DEMO_LATEST_BILL_REWRITE_QUESTION = (
    "Τι προβλέπει το νομοσχέδιο για το Ταμείο Καινοτομίας και την πρόσβαση ασθενών σε νέα φάρμακα και θεραπείες;"
)

DEMO_PROMPTS_SECONDARY = [
    "Τι αλλάζει με την επιστολική ψήφο;",
    "Ποιους αφορά ο νόμος;",
    "Δώσε μου σύντομη περίληψη σε απλά ελληνικά.",
]

clicked_prompt_demo = None
st.markdown(
    '<p style="margin:0 0 0.35rem;font-size:1.05rem;font-weight:600;color:#31333F;letter-spacing:0.015em;line-height:1.45;">Δοκιμάστε</p>'
    '<p style="margin:0 0 0.5rem;font-size:0.895rem;line-height:1.5;color:#69717d;font-weight:400;">'
    "Γρήγορη υπόδειξη ερωτημάτων· επίσης μπορείτε να συντάξετε δική σας ερώτηση παρακάτω."
    '</p>',
    unsafe_allow_html=True,
)

_pb_l, _pb_mid, _pb_r = st.columns([2.2, 4.85, 2.2])
with _pb_mid:
    if st.button(
        PRIMARY_DEMO,
        key="demo_last_bill_vote",
        type="primary",
        use_container_width=True,
    ):
        clicked_prompt_demo = PRIMARY_DEMO

st.markdown('<div style="height:0.42rem;"></div>', unsafe_allow_html=True)
_sc1, _sc2, _sc3 = st.columns(3)
with _sc1:
    if st.button(DEMO_PROMPTS_SECONDARY[0], key="demo_vote_by_mail"):
        clicked_prompt_demo = DEMO_PROMPTS_SECONDARY[0]
with _sc2:
    if st.button(DEMO_PROMPTS_SECONDARY[1], key="demo_who_law"):
        clicked_prompt_demo = DEMO_PROMPTS_SECONDARY[1]
with _sc3:
    if st.button(DEMO_PROMPTS_SECONDARY[2], key="demo_summary"):
        clicked_prompt_demo = DEMO_PROMPTS_SECONDARY[2]

st.markdown(
    '<div style="height:0.55rem;"></div>'
    '<hr style="margin:0.4rem 0 0;border:none;border-top:1px solid rgba(229,231,239,0.88);" />'
    '<div style="height:1rem;"></div>',
    unsafe_allow_html=True,
)

# --- ΒΟΗΘΗΤΙΚΑ ---
def cosine_sim(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

@st.cache_data(show_spinner=False)
def embed_text(text: str):
    # Προτεινόμενο πιο σύγχρονο embedding model:
    # text-embedding-3-small (value) ή text-embedding-3-large (quality)
    return client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    ).data[0].embedding

def build_context(scored_chunks):
    # Συμμαζεμένη μορφοποίηση context με σαφείς αναφορές
    parts = []
    for sim, ch in scored_chunks:
        source = ch.get("source", ch.get("source_id", "Άγνωστη Πηγή"))
        page = ch.get("page", "?")
        text = ch.get("text", "")
        parts.append(f"[Πηγή: {source} | Σελίδα: {page}]\n{text}")
    return "\n\n---\n\n".join(parts)

def used_sources_footer(scored_chunks) -> str:
    """Λίστα πηγών/σελίδων από τα αποσπάσματα που χρησιμοποιήθηκαν στην αναζήτηση (χωρίς διπλότυπα)."""
    seen = set()
    lines = []
    for _sim, ch in scored_chunks:
        source = ch.get("source", ch.get("source_id", "Άγνωστη Πηγή"))
        page = ch.get("page", "?")
        key = (str(source), str(page))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **Πηγή:** {source} · **Σελίδα:** {page}")
    if not lines:
        return ""
    body = "\n".join(lines)
    return (
        "\n\n---\n\n"
        "#### Χρησιμοποιήθηκαν πηγές\n\n"
        f"{body}"
    )


def get_knowledge_base_summary():
    """Αριθμός διακριτών πηγών, συνολικά chunks και λίστα πηγών από τα φορτωμένα chunks."""
    unique_sources = sorted(
        set(
            c.get("source", c.get("source_id", "Άγνωστη πηγή"))
            for c in chunks
        )
    )
    return len(unique_sources), len(chunks), unique_sources


def is_metadata_question(question: str) -> bool:
    """Ερωτήσεις inventory για τι έχει φορτωθεί στη βάση — όχι περιεχόμενο νόμου."""
    q = " ".join((question or "").strip().lower().split())
    if not q:
        return False

    needles = (
        "πόσους νόμους έχεις",
        "πόσους νόμους έχουν",
        "πόσους νόμους",
        "πόσοι νόμοι έχουν",
        "πόσοι νόμοι",
        "πόσα έγγραφα έχεις",
        "πόσα έγγραφα έχουν",
        "πόσα έγγραφα",
        "τι πηγές έχεις",
        "τι πηγές έχουν",
        "τι πηγές",
        "ποιες πηγές",
        "τι έχεις μέσα",
        "τι έχεις στη βάση",
        "τι έχεις στην βάση",
    )
    if any(s in q for s in needles):
        return True
    if "τι έχεις" in q and "βάση" in q:
        return True
    if "ποιοι νόμοι" in q and (
        "καταχωρημέν" in q or "βάση" in q or "έχεις" in q
    ):
        return True
    return False


def _normalize_q(question: str) -> str:
    return " ".join((question or "").strip().lower().split())


def is_latest_bill_question(question: str) -> bool:
    """Εντοπισμός αναφορών «τελευταίο / πιο πρόσφατο νομοσχέδιο» ή «ψηφίστηκε πρόσφατα» (χωρίς χρονική ερμηνεία στο embeddings)."""
    n = _normalize_q(question)
    needles = (
        "τελευταίο νομοσχέδιο",
        "τελευταιο νομοσχεδιο",
        "πιο πρόσφατο νομοσχέδιο",
        "πιο προσφατο νομοσχεδιο",
        "τι ψηφίστηκε πρόσφατα",
        "τι ψηφιστηκε προσφατα",
    )
    return any(marker in n for marker in needles)


def is_greeting(question: str) -> bool:
    """Απλός χαιρετισμός / έναρξη συνομιλίας χωρίς νομοθετικό αντικείμενο."""
    n = _normalize_q(question)
    if not n or len(n) > 72:
        return False
    if n.endswith("?") or n.endswith(";"):
        return False
    tokens = frozenset(
        """
        γεια γειά χαίρετε χαιρετε καλημέρα καλημερα καλησπέρα καλησπερα hey hi hello καληνυχτα
        καληνύχτα good morning gd
        """.split()
    )
    parts = frozenset(n.replace("!", "").replace(".", "").split())
    return bool(parts and parts <= tokens)


def is_capabilities_question(question: str) -> bool:
    n = _normalize_q(question)
    needles = (
        "τι μπορείς να κάνεις",
        "τι μπορεις να κανεις",
        "τι μπορείτε να κάνετε",
        "τι δυνατότητες έχεις",
        "με τι μπορείς να βοηθήσεις",
        "με τι μπορεις να βοηθησεις",
        "τι κάνεις ως εργαλείο",
        "τι κάνεις ως εφαρμογή",
        "πώς σε χρησιμοποιώ",
        "πως σε χρησιμοποιω",
    )
    return any(k in n for k in needles)


def is_general_assistant_question(question: str) -> bool:
    """Γενικές ερωτήσεις χωρίς ανάγνωση περιεχομένου νόμου (γειαρισμός, δυνατότητες, μικρά FAQ)."""
    return is_greeting(question) or is_capabilities_question(question) or (
        _looks_like_general_assistant_ping(question)
    )


def is_general_question(question: str) -> bool:
    """Ίδια λογική με `is_general_assistant_question` (για σαφέστερη ονοματολογία ροής)."""
    return is_general_assistant_question(question)


def _looks_like_general_assistant_ping(question: str) -> bool:
    n = _normalize_q(question)
    if not n:
        return False
    if ("τι κάνεις" in n or "τι κανείς" in n) and (
        "νόμο" not in n
        and "νομοσχέδιο" not in n
        and "κείμενο" not in n
        and "άρθρο" not in n
        and "αρθρο" not in n
    ):
        return True
    if n in {"τι είσαι", "τι είσαι εσύ"}:
        return True
    if any(n.startswith(p) for p in ("ευχαριστώ", "ευχαριστω", "thanks", "thank you")):
        return True
    return False


def try_general_conversational_response(question: str) -> str | None:
    """Φυσικές σύντομες απαντήσεις χωρίς embeddings/LLM (εκτός αν δεν ταιριάζει κανένα template)."""
    if is_metadata_question(question):
        return None
    n = _normalize_q(question)

    caps_text = (
        "Μπορώ να βοηθήσω να κατανοήσεις νόμους, νομοσχέδια και κοινοβουλευτικές διαδικασίες που έχουν "
        "ενσωματωθεί στη βάση μου. Μπορώ να δώσω περιλήψεις, να εξηγήσω τι αλλάζει πρακτικά και να "
        "δείξω τις σχετικές πηγές/σελίδες."
    )
    greeting_text = (
        "Καλησπέρα! Μπορείτε να με ρωτήσετε για νόμους, νομοσχέδια ή κοινοβουλευτικές συνεδριάσεις που "
        "έχουν ενσωματωθεί στη βάση."
    )
    tic_kaneis = (
        "Είμαι βοηθός νομοθεσίας· αναζητώ στα επίσημα κείμενα που είναι φορτωμένα εδώ και σας "
        "απαντάω με κατανοητικό τρόπο. Αν ρωτήσετε για συγκεκριμένο άρθρο, διάταξη ή διαδικασία που "
        "συμπεριλαμβάνεται στα διαθέσιμα απόσπασματα, μπορώ να επεξηγήσω και να συνοψίσω."
    )

    if is_greeting(question):
        return greeting_text
    if is_capabilities_question(question):
        return caps_text
    if "τι κάνεις" in n or "τι κανείς" in n:
        return tic_kaneis
    if any(n.startswith(p) for p in ("ευχαριστώ", "ευχαριστω", "thanks", "thank you")):
        return "Παρακαλώ! Αν προκύψει άλλη ερώτηση επί των διαθέσιμων εγγράφων, είμαι στη διάθεσή σας."
    if n in {"τι είσαι", "τι είσαι εσύ"}:
        return caps_text

    return None


def needs_structured_legal_sections(question: str) -> bool:
    """
    True όταν η ερώτηση αφορά περιεχόμενο νόμου, νομοσχεδίου, εκθέσεων ή συστατικών κοινοβούλου όπως πρακτικά/ψηφοφορίες.
    Για χαιρετισμούς/καταμέτρηση βάσης/γενικά FAQ επιστρέφει False.
    """
    if is_general_assistant_question(question) or is_metadata_question(question):
        return False

    n = _normalize_q(question)
    if not n:
        return False

    markers = (
        "τι προβλέπει",
        "τι προβλεπει",
        "τι προνοεί",
        "τι προνοει",
        "τι διατάζει",
        "τι διαταζει",
        "πότε ισχύει",
        "ποτε ισχυει",
        "ως προς τις διατάξεις",
        "ως προς τισ διατάξεις",
        "τι λέει ο νόμος",
        "τι λεει ο νομος",
        " τι λέει ο ",
        "άρθρο ",
        "άρθρου ",
        "άρθρα ",
        "αρθρο ",
        "αρθρα ",
        "διάταξη",
        "διαταξη",
        "διατάξεις",
        "καταργείται",
        "καταργειται",
        "αντικαθίσταται",
        "νομοσχέδιο",
        "νομοσχεδιο",
        "σχέδιο νόμου",
        "σχεδιο νομου",
        " νόμο ",
        " νόμους",
        "νόμους",
        " τι αλλάζει ",
        "τι αλλαγές",
        " τι αλλαγές ",
        "τι ψηφίστηκε",
        "τι ψηφιστηκε",
        "ποιους αφορά",
        "ποιους αφορα",
        " τι αφορά ",
        "περίληψη νομοσχεδ",
        "σύνοψη νομοσχεδ",
        " συνοψη ",
        "περιληψη",
        "αιτιολογική έκθεση",
        "αιτιολογικη εκθεση",
        "συζητήθηκε στην ολομέλεια",
        " τι συζητήθηκε ",
        "συνεδρίαση",
        "συνεδριάσεις",
        "ολομέλεια",
        "ολομελεια",
        "επιτροπή νομικών",
        "ψηφοφορία",
        " ψηφοφορία",
        "τροπολογία",
        "τροπολογι",
        " επίσημο κείμενο ",
    )
    return any(m in n for m in markers)


def format_knowledge_base_metadata_answer() -> str:
    """Στατική ενημέρωση φορτωμένης βάσης (χωρίς LLM/embeddings)."""
    n_sources, n_chunks, unique_sources = get_knowledge_base_summary()
    if not unique_sources:
        line_list = "_Δεν διακρίνονται επωνυμίες πηγών στο τρέχον αρχείο._"
    else:
        line_list = "\n".join(f"- {s}" for s in unique_sources)

    return (
        f"Αυτή τη στιγμή έχουν ενσωματωθεί **{n_sources}** πηγές/έγγραφα στη βάση γνώσης, "
        f"χωρισμένα σε **{n_chunks}** αποσπάσματα για αναζήτηση.\n\n"
        "Οι διαθέσιμες πηγές είναι:\n"
        f"{line_list}\n\n"
        "Σημείωση: ο αριθμός αυτός αφορά τα έγγραφα που έχουν εισαχθεί στο demo, όχι το σύνολο της ελληνικής νομοθεσίας."
    )


# --- ΛΟΓΙΚΗ ΑΠΑΝΤΗΣΕΩΝ ---
def get_answer(question: str) -> str:
    if not chunks:
        return "Δεν υπάρχουν ακόμη δεδομένα στη βάση. Χρειάζεται εισαγωγή νόμων/εγγράφων."

    if is_metadata_question(question):
        return format_knowledge_base_metadata_answer()

    gen_reply = try_general_conversational_response(question)
    if gen_reply is not None:
        return gen_reply

    try:
        # Demo mapping for latest active bill
        retrieval_question = (
            DEMO_LATEST_BILL_REWRITE_QUESTION
            if is_latest_bill_question(question)
            else question
        )

        q_emb = embed_text(retrieval_question)

        scores = []
        for ch in chunks:
            ch_emb = ch.get("embedding")
            if not ch_emb:
                continue
            sim = cosine_sim(q_emb, ch_emb)
            scores.append((sim, ch))

        if not scores:
            return "Δεν βρέθηκαν embeddings στη βάση (λείπει το πεδίο 'embedding' στα chunks)."

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]

        # Guardrail: αν ακόμα και το καλύτερο αποτέλεσμα είναι χαμηλό, καλύτερα “δεν ξέρω”
        if top[0][0] < min_sim:
            return (
                "Δεν βρήκα επαρκές σχετικό απόσπασμα στα διαθέσιμα κείμενα για να απαντήσω με ασφάλεια.\n\n"
                "Δοκίμασε να:\n"
                "- αναφέρεις αριθμό νόμου/νομοσχεδίου/άρθρο/παράγραφο, ή\n"
                "- κάνεις την ερώτηση πιο συγκεκριμένη."
            )

        context = build_context(top)
        use_structured = needs_structured_legal_sections(question)

        base_scope = (
            "Είσαι ακριβής, ευγενικός βοηθός κοινοβουλευτικής και νομοθετικής πληροφόρησης.\n"
            "Βασίζεσαι στο «Κείμενο» (επίσημα αποσπάσματα με Πηγή/Σελίδα): νόμοι σε ισχύ, νομοσχέδια, εκθέσεις "
            "(π.χ. αιτιολογική ή επεξηγηματική), τροπολογίες, πρακτικά/συνεδριάσεις όταν εμφανίζονται στα αποσπάσματα.\n\n"
            "ΚΑΝΟΝΑΣ ΤΕΚΜΗΡΙΩΣΗΣ: Μην παρουσιάζεις ως γεγονός ό,τι δεν προκύπτει από το Κείμενο. Μην ισχυριστείς "
            "«πιο πρόσφατη» εξέλιξη αν δεν εμφανίζεται στο Κείμενο· για ερωτήσεις «τελευταίο νομοσχέδιο / τελευταία ψηφοφορία» "
            "διευκρινίζεις αν τα έγγραφα συγκεντρωμένα στην αναζήτηση επαρκούν ή όχι για τέτοια χρονική κρίση.\n\n"
        )

        if use_structured:
            system_prompt = (
                base_scope
                + "Η ερώτηση αφορά περιεχόμενο νόμου, νομοσχεδίου, έκθεσης, τροπολογίας ή συνεδρίασης που αξιολογείται "
                "από τα κείμενα.\n"
                "ΚΡΙΣΙΜΟ — Οργάνωσε την απάντηση με ακριβώς τέσσερις ενότητες markdown (## και τίτλος στην ίδια γραμμή), "
                "με την παρακάτω σειρά· μην αλλάξεις ούτε παραλείψεις τίτλους:\n"
                "## Σύντομη απάντηση\n"
                "(1–3 προτάσεις· αν ζητείται «τι ψηφίστηκε στο τελευταίο νομοσχέδιο» χωρίς διακριτή «τελευταία» διάσταση "
                "από το Κείμενο, το διευκρινίζεις ρητά)\n\n"
                "## Τι αλλάζει πρακτικά\n"
                "(σε απλά ελληνικά πρακτικές συνέπειες/ρυθμίσεις όπως στο Κείμενο)\n\n"
                "## Ποιους αφορά\n"
                "(κατηγορίες φορέων/πολιτών αν προκύπτει· αλλιώς ρητά πως δεν καθορίζεται)\n\n"
                "## Πηγές\n"
                "(βασικές παραπομπές που στηρίζουν την απάντηση, με Πηγή και Σελίδα όπως στο Κείμενο). Μην προσθέτεις "
                "ψευδο-γεγονότα· αν υπάρχει επέκταση ή υπόθεση χωρίς στήριξη σε απόσπασμα, το επισημαίνεις ρητά.\n"
                "Όπου επαρκεί το Κείμενο για νομοσχέδιο/ψήφιση/συζήτηση, κάλυψέ τα ανάλογα τα εμφανιζόμενα αποσπάσματα.\n"
                "Αν δεν βρίσκεις επαρκές σχετικό απόσπασμα, μη μένεις μόνο σε «δεν προκύπτει»: πρότεινε λέξεις-κλειδιά "
                "από τις διαθέσιμες Πηγές.\n"
            )
        else:
            system_prompt = (
                base_scope
                + "Χρησιμοποίησε τις επικεφαλίδες ## Σύντομη απάντηση, ## Τι αλλάζει πρακτικά, ## Ποιους αφορά, ## Πηγές "
                "μόνο όταν η ερώτηση αφορά συγκεκριμένο περιεχόμενο νόμου, νομοσχεδίου, έκθεσης, τροπολογίας ή συνεδρίασης που "
                "απαιτεί ανάγνωση επί των κειμένων· εδώ η ερώτηση δεν πληροί την τυποποιημένη τετράδα.\n"
                "Απάντα φυσικά και συνοπτικά — συνήθως ένα ή δύο μικρά εδάφια ή σύντομη λίστα, χωρίς να επιβάλλεις αυτές τις ενότητες.\n"
                "Παραπομπές Πηγή/Σελίδα όταν τις χρειάζεται ρητά η απάντηση ή ο χρήστης· μην επιβάλλεις υποχρεωτικά τμήμα "
                "«Πηγές» ή κατάλογο πηγών σε εισαγωγικά θέματα.\n"
                "Αν το Κείμενο είναι άτοπο ή ανεπαρκές, εξήγησέ το και πρότεινε πώς να γίνει πιο συγκεκριμένη η ερώτηση ή "
                "λέξεις-κλειδιά από τις διαθέσιμες Πηγές.\n"
            )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"},
            ],
            temperature=0.2,
        )
        body = response.choices[0].message.content
        if use_structured:
            return body + used_sources_footer(top)
        return body

    except Exception:
        return "⚠️ Προσωρινό σφάλμα. Δοκίμασε ξανά."

# --- CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if clicked_prompt_demo is not None:
    st.session_state.messages.append({"role": "user", "content": clicked_prompt_demo})
    with st.chat_message("user"):
        st.markdown(clicked_prompt_demo)
    with st.chat_message("assistant"):
        with st.spinner("Αναζήτηση στα επίσημα κείμενα..."):
            answer = get_answer(clicked_prompt_demo)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if prompt := st.chat_input("Ερώτηση προς τον βοηθό νομοθεσίας…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Αναζήτηση στα επίσημα κείμενα..."):
            answer = get_answer(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<p style="font-size:0.74rem;line-height:1.45;color:rgba(107,114,128,0.78);text-align:center;max-width:36rem;margin:1.65rem auto 0.4rem;padding:0 0.5rem;">Το εργαλείο παρέχει ενημερωτική υποστήριξη βασισμένη σε επίσημα κείμενα και δεν υποκαθιστά νομική ή θεσμική κρίση.</p>',
    unsafe_allow_html=True,
)
