"""Streamlit app: credit risk scoring with a probability, a cost-based decision and a local explanation."""
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import credit_utils  # noqa: F401  (needed so joblib can unpickle the pipeline's custom step)

st.set_page_config(page_title='Credit Risk Prediction', page_icon='💳', layout='centered')


@st.cache_resource
def load_artifacts():
    art = joblib.load('credit_risk_model.joblib')
    return art['pipeline'], art['threshold'], art['model_name']


pipeline, threshold, model_name = load_artifacts()

st.title('Credit Risk Prediction')
st.caption(f'Model: {model_name} | Trained on the German Credit (Statlog) data | Decision threshold: {threshold:.2f} | by Lehlohonolo Saohatse')
st.write('Enter applicant details to estimate the probability that the credit is a **bad** risk.')

JOBS = {0: '0 - Unskilled, non-resident', 1: '1 - Unskilled, resident', 2: '2 - Skilled', 3: '3 - Highly skilled'}

col1, col2 = st.columns(2)
with col1:
    age = st.number_input('Age', min_value=18, max_value=80, value=30)
    sex = st.selectbox('Sex', ['male', 'female'])
    job = st.selectbox('Job', list(JOBS), format_func=JOBS.get, index=2)
    housing = st.selectbox('Housing', ['own', 'rent', 'free'])
    purpose = st.selectbox('Purpose', ['car', 'radio/TV', 'furniture/equipment', 'business', 'education',
                                       'repairs', 'domestic appliances', 'vacation/others'])
with col2:
    saving = st.selectbox('Saving accounts', ['no account', 'little', 'moderate', 'quite rich', 'rich'])
    checking = st.selectbox('Checking account', ['no account', 'little', 'moderate', 'rich'])
    amount = st.number_input('Credit amount (DM)', min_value=100, max_value=20000, value=2000, step=100)
    duration = st.number_input('Duration (months)', min_value=1, max_value=72, value=12)

applicant = pd.DataFrame([{
    'Age': age, 'Sex': sex, 'Job': job, 'Housing': housing, 'Saving accounts': saving,
    'Checking account': checking, 'Credit amount': amount, 'Duration': duration, 'Purpose': purpose,
}])

if st.button('Predict risk', type='primary'):
    p_bad = float(pipeline.predict_proba(applicant)[0, 1])
    flagged = p_bad >= threshold

    st.metric('Estimated probability of BAD risk', f'{p_bad:.1%}')
    st.progress(min(max(p_bad, 0.0), 1.0))
    if flagged:
        st.error(f'Decision: **HIGH RISK** (probability is at or above the {threshold:.0%} cut-off)')
    else:
        st.success(f'Decision: **LOW RISK** (probability is below the {threshold:.0%} cut-off)')

    # Local explanation (SHAP) for tree models
    try:
        import shap
        prep = pipeline[:-1]
        clf = pipeline.named_steps['model']
        row = prep.transform(applicant)
        sv = shap.TreeExplainer(clf).shap_values(row)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
        contrib = pd.Series(sv[0], index=[c.split('__', 1)[-1] for c in row.columns])
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.barh(top.index, top.values, color=['#d62728' if v > 0 else '#1f77b4' for v in top.values])
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel('Impact on risk (red raises, blue lowers)')
        ax.set_title('Top drivers for this applicant')
        plt.tight_layout()
        st.pyplot(fig)
    except Exception:
        st.caption('Local explanation unavailable for this model type.')

st.divider()
st.caption('Educational project on a public 1994 dataset. Not for real lending decisions. '
           'A "no account" balance is a known quirk of this dataset and tends to lower modelled risk.')
