import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cloud Misconfiguration Risk Detector", layout="wide")
st.title("AI-Driven Cloud Misconfiguration Risk Detector")
st.write("Upload your cloud configuration CSV or use Demo Mode to see everything in action.")

@st.cache_data
def generate_demo_data():
    resource_types = ['S3', 'EC2', 'IAM', 'RDS', 'Lambda']
    data = []
    np.random.seed(42)
    for i in range(150):
        res = np.random.choice(resource_types)
        public_access = np.random.choice([0,1]) if res in ['S3','EC2','Lambda'] else 0
        encryption_enabled = np.random.choice([0,1])
        logging_enabled = np.random.choice([0,1])
        excessive_priv = np.random.choice([0,1]) if res == 'IAM' else 0
        outdated = np.random.choice([0,1]) if res in ['EC2','RDS'] else 0
        # Label logic
        if res == 'S3' and public_access and not encryption_enabled:
            misconf_type, risk = 'Insecure Storage', 'High'
        elif res == 'IAM' and excessive_priv:
            misconf_type, risk = 'Excessive Privileges', 'High'
        elif res in ['EC2','Lambda'] and public_access:
            misconf_type, risk = 'Unsecured Service', 'High'
        elif not logging_enabled:
            misconf_type, risk = 'No Monitoring', 'Medium'
        elif not encryption_enabled:
            misconf_type, risk = 'Lack of Encryption', 'Medium'
        elif outdated:
            misconf_type, risk = 'Outdated Component', 'Low'
        else:
            misconf_type, risk = 'Compliant', 'Low'
        data.append([res, public_access, encryption_enabled, logging_enabled, excessive_priv, outdated, misconf_type, risk])
    df = pd.DataFrame(data, columns=['resource_type','public_access','encryption_enabled','logging_enabled','excessive_priv','outdated','misconf_type','risk_level'])
    return df

def explain_risk(row):
    if row['risk_level'] == 'High':
        if row['resource_type'] == 'S3' and row['public_access'] and not row['encryption_enabled']:
            return 'Insecure Storage: Public and Unencrypted S3 Bucket'
        if row['resource_type'] == 'IAM' and row['excessive_priv']:
            return 'Excessive Privileges: Overly Broad IAM Permissions'
        if row['resource_type'] in ['EC2', 'Lambda'] and row['public_access']:
            return f'Unsecured Service: Public {row["resource_type"]} Endpoint'
        return 'High risk misconfiguration detected.'
    elif row['risk_level'] == 'Medium':
        if not row['logging_enabled']:
            return 'No Monitoring: Logging Not Enabled'
        if not row['encryption_enabled']:
            return 'Lack of Encryption: Data Not Encrypted'
        return 'Medium risk misconfiguration detected.'
    elif row['risk_level'] == 'Low':
        if row['outdated']:
            return 'Outdated Component: Software Not Updated'
        return 'Compliant or Low-risk configuration'
    else:
        return 'Unknown risk level'

def suggest_remediation(row):
    if 'Insecure Storage' in row['explanation']:
        return 'Set S3 bucket to private and enable encryption.'
    if 'Excessive Privileges' in row['explanation']:
        return 'Review IAM permissions and apply least privilege. Enable MFA.'
    if 'Unsecured Service' in row['explanation']:
        return f'Limit {row["resource_type"]} access to internal IPs only.'
    if 'No Monitoring' in row['explanation']:
        return 'Enable logging/monitoring for this resource.'
    if 'Lack of Encryption' in row['explanation']:
        return 'Enable encryption at rest and in transit.'
    if 'Outdated Component' in row['explanation']:
        return 'Update software and enable automatic updates.'
    return 'No action needed.'

uploaded_file = st.file_uploader("Upload your cloud asset CSV (with correct columns)")

demo = st.button("Demo Mode (Generate Example Data)")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
if demo and df is None:
    df = generate_demo_data()
    st.info("Demo data loaded!")

if df is not None:
    st.subheader("Input Data Preview")
    st.dataframe(df.head(15))

    # Label Encoding
    label_cols = ['resource_type', 'misconf_type', 'risk_level']
    encoders = {}
    df_encoded = df.copy()
    for col in label_cols:
        enc = LabelEncoder()
        df_encoded[col+'_enc'] = enc.fit_transform(df_encoded[col])
        encoders[col] = enc

    feature_cols = ['resource_type_enc', 'public_access', 'encryption_enabled', 'logging_enabled', 'excessive_priv', 'outdated']
    target_col = 'risk_level_enc'
    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    # Train/test split and training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics display
    st.subheader("Model Performance")
    st.write(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))
    st.text("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

    # Feature importance
    importances = clf.feature_importances_
    feat_names = feature_cols
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feat_names, importances)
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance (Random Forest)')
    st.pyplot(fig)

    # Predict, Explain, Remediate
    df_encoded['pred_risk_level'] = encoders['risk_level'].inverse_transform(clf.predict(X))
    df_encoded['explanation'] = df_encoded.apply(explain_risk, axis=1)
    df_encoded['remediation'] = df_encoded.apply(suggest_remediation, axis=1)

    # Results table
    st.subheader("Risk Assessment Results")
    outcols = ['resource_type', 'public_access', 'encryption_enabled', 'logging_enabled', 'excessive_priv', 'outdated',
               'pred_risk_level', 'explanation', 'remediation']
    st.dataframe(df_encoded[outcols].head(20))

    # Download results
    csv = df_encoded[outcols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="cloud_risk_assessment_results.csv"
    )
else:
    st.info("Upload a CSV file or use Demo Mode to continue.")
