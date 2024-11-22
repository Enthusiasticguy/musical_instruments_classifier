import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

# Yo'llarni moslashtirish
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Sarlavha va uslub
st.set_page_config(page_title="Musiqiy asbob Klassifikatsiya", page_icon="🎸")
st.title('🚀 Musiqiy asbobni Klassifikatsiya qiluvchi Model')
st.markdown("""
    **Gitara? Pianinomi? Skripkami? yo Baraban?**   
    Ushbu ilova yuklangan rasmni klassifikatsiya qiladi va natijani ehtimollik bilan ko'rsatadi.   
    Yuklang va sinab ko'ring! 😎
""")
st.info('https://t.me/ismoilov_husan')


# Rasm yuklash
file = st.file_uploader(
    'Rasm yuklang (faqat *gitara*, *skripka*, *baraban*, yo *piano*)',
    type=['png', 'jpeg', 'jpg', 'gif']
)

# Modelni yuklash
model = load_learner('musical_instruments_model.pkl')

# Rasmni tahlil qilish
if file:
    st.image(file, caption="Yuklangan rasm", use_container_width=True)

    with st.spinner("🔍 Bashorat qilinmoqda..."):
        img = PILImage.create(file)
        pred, pred_id, probs = model.predict(img)

    st.success(f'**Bashorat**: {pred}')
    st.info(f'**Ehtimollik**: {probs[pred_id] * 100:.1f}%')

    
    # Bar grafigi
    fig = px.bar(
        x=model.dls.vocab,
        y=probs * 100,
        color=model.dls.vocab,
        labels={"x": "Asbob turi", "y": "Ehtimollik (%)"},
        title="Ehtimolliklar taqsimoti",
        template="plotly_dark"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
else:
    st.info("❗️ Rasm yuklang va natijani ko'ring.")
