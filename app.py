import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
import io
import matplotlib.pyplot as plt

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def predict_fraud(model, data):
    df = data.copy()
    if 'Class' in df.columns:
        df = df.drop(columns=['Class'])
    if 'Prediction' in df.columns:
        df = df.drop(columns=['Prediction'])
    if df.shape[1] != 30:
        raise ValueError(f"Le modèle attend 30 features, mais le fichier en contient {df.shape[1]}.")
    preds = model.predict(df)
    df['Prediction'] = preds
    return df

def generate_pdf(data: pd.DataFrame, nb_fraudes: int, nb_normales: int) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Rapport de détection de fraude", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"✅ Transactions normales : {nb_normales}", styles['Normal']))
    elements.append(Paragraph(f"⚠️ Transactions frauduleuses : {nb_fraudes}", styles['Normal']))
    elements.append(Spacer(1, 12))
    sample_data = data.head(20)
    table_data = [list(sample_data.columns)] + sample_data.values.tolist()
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def to_excel(df: pd.DataFrame) -> BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Résultats')
        writer.save()
    output.seek(0)
    return output

st.title("🕵️ Détection de Fraude Bancaire")

model_path = "model_fraud.pkl"
model = load_model(model_path)

uploaded_file = st.file_uploader("📂 Charger un fichier CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Fichier chargé avec succès ✅")

        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        with st.spinner("🔍 Analyse en cours..."):
            results = predict_fraud(model, df)

        nb_fraudes = int((results['Prediction'] == 1).sum())
        nb_normales = int((results['Prediction'] == 0).sum())

        col1, col2 = st.columns(2)
        col1.metric("✅ Transactions normales", nb_normales)
        col2.metric("⚠️ Fraudes détectées", nb_fraudes)

        fig, ax = plt.subplots()
        counts = results['Prediction'].value_counts()
        ax.pie(counts, labels=['Normale', 'Fraude'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        ax.set_title("Répartition des transactions")
        st.pyplot(fig)

        if st.checkbox("Afficher uniquement les fraudes"):
            st.dataframe(results[results['Prediction'] == 1])
        else:
            st.dataframe(results.head(20))

        pdf_buffer = generate_pdf(results, nb_fraudes, nb_normales)
        st.download_button("📄 Télécharger PDF", data=pdf_buffer, file_name="rapport_fraude.pdf", mime="application/pdf")

        excel_buffer = to_excel(results)
        st.download_button("📊 Télécharger Excel complet", data=excel_buffer, file_name="resultats_fraude.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        excel_fraudes = to_excel(results[results['Prediction'] == 1])
        st.download_button("🚨 Télécharger uniquement fraudes", data=excel_fraudes, file_name="fraudes_detectees.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"❌ Erreur : {e}")
