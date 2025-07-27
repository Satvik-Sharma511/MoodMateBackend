import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# column questions
column_questions = {
    'gender': "What is your gender?",
    'country': "Which country are you from?",
    'occupation': "What is your occupation?",
    'self_employed': "Are you self-employed?",
    'family_history': "Do you have a family history of mental illness?",
    'days_indoors': "How many days have you stayed indoors recently?",
    'growing_stress': "Have you been experiencing growing stress?",
    'changes_habits': "Have you noticed changes in your habits?",
    'mental_health_history': "Do you have a history of mental health issues?",
    'mood_swings': "Do you experience mood swings?",
    'coping_struggles': "Are you struggling to cope?",
    'work_interest': "Have you lost interest in work/studies?",
    'social_weakness': "Are you feeling socially disconnected?",
    'mental_health_interview': "Would you feel comfortable discussing mental health in an interview?",
    'care_options': "Are you aware of mental health care options available to you?",
}

#Load and preprocess data
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'timestamp' in df.columns:
        df.drop('timestamp', axis=1, inplace=True)

    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# Train automatically
def train_model(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, scaler, X.columns.tolist()

#prediction
def predict(model, scaler, label_encoders, user_input, feature_names):
    input_df = pd.DataFrame([user_input])
    input_df.columns = input_df.columns.str.strip().str.lower().str.replace(' ', '_')

    for col in label_encoders:
        if col in input_df.columns:
            le = label_encoders[col]
            val = input_df[col][0]
            input_df[col] = le.transform([val])[0] if val in le.classes_ else -1

    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    input_df = scaler.transform(input_df)
    prediction = model.predict(input_df)
    return prediction[0]

st.markdown(
    """
    <style>
    body {
        color: #000; /* Black text by default */
        background-color: #4682B4; /* Darker light blue (Steel Blue) */
    }
    .stApp {
        background-color: #4682B4; /* Darker light blue for the app */
    }
    .st-header {
        background-color: #4169E1; /* Even darker blue for header (Royal Blue) */
        color: #fff; /* White header text for contrast */
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1e88e5; /* Darker blue for headings */
    }
    .stButton>button {
        color: #fff;
        background-color: #1e88e5;
        border-color: #1e88e5;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        border-color: #1565c0;
    }
    .stSelectbox label div div {
        color: #000 !important; /* Black text for selectbox label */
    }
    .stSelectbox>div>div>div>div {
        color: #fff; /* White text for selectbox input */
        background-color: #2e3740; /* Dark background for selectbox input */
    }
    .stNumberInput label div div {
        color: #000 !important; /* Black text for number input label */
    }
    .stNumberInput>div>div>input {
        color: #fff !important; /* White text for number input */
        background-color: #2e3740 !important; /* Dark background for number input */
    }
    .stSuccess {
        color: #2e7d32;
        background-color: #e8f5e9;
    }
    .stError {
        color: #d32f2f;
        background-color: #ffebee;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load dataset
dataset_path = "new mental health.csv"

try:
    df, label_encoders = load_dataset(dataset_path)
    st.session_state['df'] = df
    st.session_state['label_encoders'] = label_encoders
except Exception as e:
    st.error(f"Error loading dataset: {e}")

if 'df' in st.session_state:
    df = st.session_state['df']
    label_encoders = st.session_state['label_encoders']

    # Set the target column manually
    target_column = 'treatment'

    # Automatically train the model
    model, X_test, y_test, scaler, feature_names = train_model(df, target_column)
    st.session_state['model'] = model
    st.session_state['scaler'] = scaler
    st.session_state['feature_names'] = feature_names

    st.write("### 🧾 Answer the Questions Below")
    user_input = {}

    for feature in feature_names:
        question = column_questions.get(feature, f"Enter {feature}:")
        if feature in label_encoders:
            options = list(label_encoders[feature].classes_)
            if feature == 'gender' and 'Others' not in options:
                options.append('Others')
            user_input[feature] = st.selectbox(question, options)
        else:
            user_input[feature] = st.number_input(question)

    if st.button("Predict Treatment Need"):
        result = predict(model, scaler, label_encoders, user_input, feature_names)
        readable = label_encoders[target_column].inverse_transform([result])[0] \
                    if target_column in label_encoders else result
        st.success(f"Prediction: **{readable}**")