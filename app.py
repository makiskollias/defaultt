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
    layout="centered"
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

# --- UI ΕΦΑΡΜΟΓΗΣ ---
st.title("🏛️ Vouli-AI: Βοηθός Νομοθεσίας")
st.markdown("""
Το **Vouli-AI** βοηθά πολίτες και χρήστες να **κατανοήσουν** νόμους, **τρέχοντα νομοσχέδια**, συνεδριάσεις και νομοθετική διαδικασία,
**βασιζόμενο μόνο σε επίσημα κείμενα** που έχουν ενσωματωθεί στη βάση (πράξεις, σχέδια νόμου, εισηγήσεις, πρακτικά/στοιχεία όπου έχουν εισαχθεί)· οι παραπομπές **Πηγή** και **Σελίδα** προέρχονται από αυτά τα αποσπάσματα.
""")

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

st.divider()

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

# --- ΛΟΓΙΚΗ ΑΠΑΝΤΗΣΕΩΝ ---
def get_answer(question: str) -> str:
    if not chunks:
        return "Δεν υπάρχουν ακόμη δεδομένα στη βάση. Χρειάζεται εισαγωγή νόμων/εγγράφων."

    try:
        q_emb = embed_text(question)

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

        system_prompt = (
            "Είσαι ένας ακριβής, ευγενικός βοηθός κοινοβουλευτικής και νομοθετικής πληροφόρησης.\n"
            "Μπορείς να βοηθάς με νόμους σε ισχύ, αλλά ΚΑΙ με υλικό που αφορά **τρέχουσα νομοθετική δραστηριότητα**, όσο υπάρχει σχετικό περιεχόμενο "
            "στο Κείμενο: π.χ. **νομοσχέδια / σχέδια νόμου**, **συνεδριάσεις ή πρακτικά αν εμφανίζονται στα αποσπάσματα**, "
            "**συζητήσεις επιτροπών** αν αναφέρονται σε αυτά, **επεξηγηματικές εκθέσεις ή εισηγητικά κείμενα**, **τροπολογίες** "
            "ή άλλα συνημμένα όπως διακρίνονται στις Πηγές.\n"
            "\n"
            "ΚΑΝΟΝΑΣ ΤΕΚΜΗΡΙΩΣΗΣ: Απαντάς ΜΟΝΟ με ό,τι υποστηρίζεται από το Κείμενο (με παραπομπές Πηγή/Σελίδα). "
            "Μην ισχυριστείς ότι υπάρχει «πιο πρόσφατη» εξέλιξη αν δεν εμφανίζεται στο Κείμενο· "
            "αν λείπει η σχετική πληροφορία ή το κείμενο είναι παλαιό σε σχέση με την ερώτηση για «τελευταίο νομοσχέδιο / τελευταία ψηφοφορία», "
            "διευκρίνισε ρητά ότι προκύπτει ΜΟΝΟ από τις διαθέσιμες σελίδες και τι δεν διαπιστώνεται από αυτές.\n"
            "\n"
            "Όταν η ερώτηση αφορά νομοσχέδιο / ψήφιση / συνεδρίαση, και το Κείμενο επαρκεί, κάλυψε όσο είναι δυνατό τα εξής (πάντα με βάση το Κείμενο):\n"
            "- **Τι συζητήθηκε / τι περιλαμβάνει η πρόταση ή η συζήτηση** (με βάση σχετικά αποσπάσματα)\n"
            "- **Τι ψηφίστηκε / τι καταλήγει το κείμενο ή η ψηφοφορία** αν περιέχεται στα διαθέσιμα αποσπάσματα\n"
            "- **Ποιους αφορά** και **τι αλλάζει πρακτικά**\n"
            "\n"
            "1) Αν βρίσκεις πληροφορία στο Κείμενο, εξήγησέ την με σαφήνεια και παραπομπές (Πηγή, Σελίδα).\n"
            "2) Αν η ερώτηση είναι γενική (π.χ. 'τι μπορείς να κάνεις;' ή 'τι υλικό έχεις;'), περιέγραψε τι καλύπτεται από τις διαθέσιμες Πηγές όπως εμφανίζονται στο Κείμενο.\n"
            "3) Αν δεν βρίσκεις τίποτα σχετικό, μη λες μόνο 'Δεν προκύπτει'. Πρότεινε στον χρήστη λέξεις-κλειδιά από το Κείμενο για επαναλαμβανόμενη αναζήτηση.\n"
            "\n"
            "ΚΡΙΣΙΜΟ — Κάθε απάντηση πρέπει να ακολουθεί πάντα την ίδια δομή με τέσσερις ενότητες, με επικεφαλίδες markdown "
            "(με ## και τον τίτλο στην ίδια γραμμή), με τη σειρά που ακολουθεί· μην αλλάξεις ούτε παραλείψεις τίτλους:\n"
            "## Σύντομη απάντηση\n"
            "(1–3 προτάσεις· αν η ερώτηση ζητά «τι ψηφίστηκε στο τελευταίο νομοσχέδιο», αν δεν διακρίνεται «τελευταίο» από το Κείμενο, πες ρητά ποια πράξη ή σχετικό μέρος βρέθηκε ή ότι από τα έγγραφα δεν διαπιστώνεται επάρκεια για «τελευταίο»)\n\n"
            "## Τι αλλάζει πρακτικά\n"
            "(σε απλά ελληνικά τι ισχύει, ποιες ρυθμίσεις/συνέπειες απορρέουν ή τι συμπέρασμα υποδεικνύει το Κείμενο)\n\n"
            "## Ποιους αφορά\n"
            "(κατηγορίες ανθρώπων ή φορέων· αν δεν διευκρινίζεται στο Κείμενο, διευκρίνισέ το ρητά)\n\n"
            "## Πηγές\n"
            "(κύριες παραπομπές που στηρίζουν την απάντηση — Πηγή και Σελίδα όπως στο Κείμενο). "
            "Μην προσθέτεις πληροφορία που δεν επαληθεύεται στο Κείμενο· αν κάνεις υποθέσεις ή συμπληρώνεις χωρίς στήριξη από τα αποσπάσματα, "
            "διευκρίνισέ το ρητά και μην την παρουσιάζεις ως περιεχόμενο των εγγράφων.\n"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"}
            ],
            temperature=0.2
        )
        body = response.choices[0].message.content
        return body + used_sources_footer(top)

    except Exception:
        return "⚠️ Προσωρινό σφάλμα. Δοκίμασε ξανά."

# --- CHAT INTERFACE ---

if "messages" not in st.session_state:
    st.session_state.messages = []

DEMO_PROMPTS = [
    "Τι αλλάζει με την επιστολική ψήφο;",
    "Ποιους αφορά ο νόμος;",
    "Δώσε μου σύντομη περίληψη σε απλά ελληνικά.",
    "Τι ψηφίστηκε στο τελευταίο νομοσχέδιο;",
]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.caption("Παράδειγμα ερωτήσεων για την επίδειξη")
dc1, dc2, dc3, dc4 = st.columns(4)
clicked_prompt = None
with dc1:
    if st.button(DEMO_PROMPTS[0], key="demo_vote_by_mail"):
        clicked_prompt = DEMO_PROMPTS[0]
with dc2:
    if st.button(DEMO_PROMPTS[1], key="demo_who_law"):
        clicked_prompt = DEMO_PROMPTS[1]
with dc3:
    if st.button(DEMO_PROMPTS[2], key="demo_summary"):
        clicked_prompt = DEMO_PROMPTS[2]
with dc4:
    if st.button(DEMO_PROMPTS[3], key="demo_last_bill_vote"):
        clicked_prompt = DEMO_PROMPTS[3]

st.caption(
    "Το εργαλείο παρέχει ενημερωτική υποστήριξη βασισμένη σε επίσημα κείμενα "
    "και δεν υποκαθιστά νομική ή θεσμική κρίση."
)

if clicked_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": clicked_prompt})
    with st.chat_message("user"):
        st.markdown(clicked_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Σκέφτεται..."):
            answer = get_answer(clicked_prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if prompt := st.chat_input("Ερώτηση προς τον βοηθό νομοθεσίας…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Σκέφτεται..."):
            answer = get_answer(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
