
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules

# Safe SHAP import
try:
    import shap
    import matplotlib.pyplot as plt
    shap_available = True
except:
    shap_available = False

st.set_page_config(page_title="Credit Intelligence System", layout="wide")

st.title("🚀 Credit Intelligence System (Final Stable Version)")

df = pd.read_csv("data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

df["LoanStatus"] = df["LoanStatus"].map({"Approved":1,"Rejected":0})
df = df.drop("Applicant_ID", axis=1)

df["MarketingScore"] = (
    (df["Income"]/100000)*0.4 +
    (1 - df["DTI_Ratio"])*0.3 +
    (1 - df["Late_Payments"]/10)*0.3
)

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("LoanStatus", axis=1)
y = df_encoded["LoanStatus"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

models = {
    "Logistic": LogisticRegression(max_iter=1000, solver="liblinear"),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

st.header("Model Performance")

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    pre = precision_score(y_test,y_pred)
    rec = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)

    st.subheader(name)
    st.write({"Accuracy":acc,"Precision":pre,"Recall":rec,"F1":f1})

    if acc > best_acc:
        best_acc = acc
        best_model = model

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test,y_prob)
        fig = px.line(x=fpr, y=tpr, title=f"ROC - {name}")
        st.plotly_chart(fig)

st.header("Feature Importance")

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig = px.bar(importance.head(10), x="Importance", y="Feature", orientation='h')
st.plotly_chart(fig)

st.header("SHAP Explainability")

if shap_available:
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)
else:
    st.warning("SHAP not available")

st.header("Clustering")

kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

fig = px.scatter(df, x="Income", y="DTI_Ratio", color=df["Cluster"].astype(str))
st.plotly_chart(fig)

st.header("Association Rules")

df_rules = pd.get_dummies(df[["Risk","LoanStatus"]])
freq = apriori(df_rules, min_support=0.1, use_colnames=True)
rules = association_rules(freq, metric="confidence", min_threshold=0.6)

st.dataframe(rules[["antecedents","consequents","confidence","lift"]])

st.header("Prediction")

income = st.number_input("Income",20000,100000,40000)
debt = st.number_input("Debt",0,100000,20000)
emi = st.slider("EMI Ratio",0,100,30)
late = st.slider("Late Payments",0,10,1)
emp = st.selectbox("Employment",["Salaried","SelfEmployed"])

dti = debt/income

input_df = pd.DataFrame([{
    "Income":income,
    "Debt":debt,
    "EMI_Ratio":emi,
    "Late_Payments":late,
    "Employment":emp,
    "CreditScore":700,
    "DTI_Ratio":dti,
    "Risk":"Medium",
    "MarketingScore":0.5
}])

input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

best_model.fit(X,y)

if st.button("Predict"):
    pred = best_model.predict(input_encoded)[0]
    st.success("Approved" if pred==1 else "Rejected")

st.header("Upload CSV")

file = st.file_uploader("Upload CSV")

if file:
    new_df = pd.read_csv(file)
    new_df = new_df.drop("Applicant_ID", axis=1, errors="ignore")

    new_encoded = pd.get_dummies(new_df)
    new_encoded = new_encoded.reindex(columns=X.columns, fill_value=0)

    preds = best_model.predict(new_encoded)
    new_df["Prediction"] = np.where(preds==1,"Approved","Rejected")

    st.dataframe(new_df.head())
