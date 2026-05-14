import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ringkasan Berita · mBART",
    page_icon="📰",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #FAFAF7;
    color: #1a1a1a;
}

/* Header */
.header-wrap {
    background: #1a1a1a;
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.header-wrap h1 {
    font-family: 'Playfair Display', serif;
    color: #F5E6C8;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.header-wrap p {
    color: #9a9a8a;
    font-size: 0.95rem;
    margin: 0;
    font-style: italic;
}
.badge {
    display: inline-block;
    background: #C8A96E;
    color: #1a1a1a;
    font-size: 0.7rem;
    font-family: monospace;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
}

/* Input area */
.stTextArea textarea {
    font-family: 'Source Serif 4', serif !important;
    font-size: 0.95rem !important;
    background: #FFFFFF !important;
    border: 1.5px solid #ddd !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    line-height: 1.7 !important;
}
.stTextArea textarea:focus {
    border-color: #C8A96E !important;
    box-shadow: 0 0 0 2px rgba(200,169,110,0.15) !important;
}

/* Button */
.stButton > button {
    background: #1a1a1a !important;
    color: #F5E6C8 !important;
    font-family: 'Playfair Display', serif !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #C8A96E !important;
    color: #1a1a1a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* Result box */
.result-box {
    background: #FFFFFF;
    border-left: 4px solid #C8A96E;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem 1.8rem;
    margin-top: 1.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
}
.result-label {
    font-family: monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C8A96E;
    margin-bottom: 0.6rem;
    font-weight: 700;
}
.result-text {
    font-family: 'Source Serif 4', serif;
    font-size: 1.05rem;
    line-height: 1.8;
    color: #1a1a1a;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}
.stat-card {
    flex: 1;
    background: #F5F5F0;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #1a1a1a;
    line-height: 1;
}
.stat-label {
    font-size: 0.72rem;
    color: #888;
    margin-top: 0.3rem;
    font-style: italic;
}

/* Warning / info */
.stAlert {
    border-radius: 10px !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #bbb;
    margin-top: 3rem;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <div class="badge">mBART</div>
    <h1>📰 Ringkasan Berita Otomatis</h1>
</div>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "gevira/mbart-ringkasan-berita-id"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_id)
    model = MBartForConditionalGeneration.from_pretrained(model_id)
    model.eval()
    return tokenizer, model

with st.spinner("⏳ Memuat model mBART, harap tunggu..."):
    try:
        tokenizer, model = load_model()
        st.success("✅ Model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("#### Masukkan Teks Berita")
input_text = st.text_area(
    label="",
    placeholder="Tempelkan teks berita berbahasa Indonesia di sini...",
    height=250,
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run = st.button("🔍 Ringkas Sekarang")

# ── Generate ──────────────────────────────────────────────────────────────────
if run:
    if not input_text.strip():
        st.warning("⚠️ Teks berita tidak boleh kosong.")
    else:
        with st.spinner("✍️ Sedang membuat ringkasan..."):
            try:
                tokenizer.src_lang = "id_ID"
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                )
                forced_bos = tokenizer.lang_code_to_id["id_ID"]
                with torch.no_grad():
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        forced_bos_token_id=forced_bos,
                        max_length=200,
                        min_length=30,
                        num_beams=4,
                        length_penalty=2.0,
                        early_stopping=True,
                    )
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                # Stats
                words_in  = len(input_text.split())
                words_out = len(summary.split())
                compress  = round((1 - words_out / words_in) * 100) if words_in > 0 else 0

                # Result box
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Hasil Ringkasan</div>
                    <div class="result-text">{summary}</div>
                </div>
                <div class="stats-row">
                    <div class="stat-card">
                        <div class="stat-value">{words_in}</div>
                        <div class="stat-label">Kata asli</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{words_out}</div>
                        <div class="stat-label">Kata ringkasan</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{compress}%</div>
                        <div class="stat-label">Kompresi teks</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat generate: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Penerapan Model mBART untuk Ringkasan Abstraktif Teks Berita Bahasa Indonesia · 2025
</div>
""", unsafe_allow_html=True)
