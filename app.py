import streamlit as st
import os
import torch
import torchaudio
import pandas as pd
import difflib
import re
import tempfile
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# الإعدادات
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_dir = "audio"

# تحميل النموذج
processor = AutoProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
model = AutoModelForSpeechSeq2Seq.from_pretrained("tarteel-ai/whisper-base-ar-quran")
model.to(device).eval()

st.title("📖 مراجعة تلاوة سورة الفاتحة")

# الآيات بالترتيب مع أرقامها
verse_texts = [
    ("001001", "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"),
    ("001002", "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"),
    ("001003", "الرَّحْمَٰنِ الرَّحِيمِ"),
    ("001004", "مَالِكِ يَوْمِ الدِّينِ"),
    ("001005", "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ"),
    ("001006", "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ"),
    ("001007", "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ")
]

# حالة التطبيق
if "verse_index" not in st.session_state:
    st.session_state.verse_index = 0

# أزرار التنقل
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ السابق") and st.session_state.verse_index > 0:
        st.session_state.verse_index -= 1
with col3:
    if st.button("التالي ➡️") and st.session_state.verse_index < len(verse_texts) - 1:
        st.session_state.verse_index += 1

# معلومات الآية الحالية
verse_id, verse_text = verse_texts[st.session_state.verse_index]
st.markdown(f"### الآية {st.session_state.verse_index + 1}:\n📖 {verse_text}")

# تشغيل الصوت المرتبط
selected_audio = os.path.join(audio_dir, f"{verse_id}.mp3")
if os.path.exists(selected_audio):
    st.audio(selected_audio, format="audio/mp3")
else:
    st.warning("⚠️ لا يوجد ملف صوتي لهذه الآية.")

# نص صحيح للاختبار
true_text = verse_text

# رفع تلاوة المستخدم
st.markdown("### 🎙️ ارفع تلاوتك لهذه الآية:")
user_audio = st.file_uploader("ارفع ملف صوتك (WAV/MP3)", type=["wav", "mp3"])

if user_audio is not None:
    # ✅ حفظ الصوت مؤقتًا
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(user_audio.read())
        tmpfile_path = tmpfile.name

    # ✅ تحميل الملف المؤقت
    waveform, sr = torchaudio.load(tmpfile_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # تحويل الصوت إلى نص
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"], max_new_tokens=200)
    user_transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    st.subheader("📜 تلاوتك المكتشفة:")
    st.write(user_transcription)

    st.subheader("📜 الآية الصحيحة:")
    st.write(true_text)

    # المقارنة
    diff = list(difflib.ndiff(user_transcription.split(), true_text.split()))
    results = []
    for word in diff:
        token = word[2:].strip()
        if not token or re.fullmatch(r'[\W_]+', token):
            continue
        if word.startswith("+ "):
            results.append({"الكلمة المنطوقة": token, "الحالة": "🆕 زائدة/خطأ", "التفصيل": "زادت أو بها خطأ"})
        elif not word.startswith("- "):
            results.append({"الكلمة المنطوقة": token, "الحالة": "✅ صحيح", "التفصيل": ""})

    df = pd.DataFrame(results)
    df.index += 1
    df.index.name = "#"

    st.subheader("📊 تحليل النطق:")
    st.dataframe(df)

