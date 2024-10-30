import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import altair as alt

# Load pre-trained models
decision_tree_model = joblib.load('dt_model.joblib')
naive_bayes_model = joblib.load('nb_model.joblib')
knn_model = joblib.load('knn_model.joblib')
log_model = joblib.load('log_model.joblib')

receiving_currency_dict = {
    'US Dollar': 0.7810775145601503, 
    'Euro' : 0.009370519343746431, 
    'Rupee':0.011527578637499841, 
    'Mexican Peso':0.10220020047962848, 
    'Ruble':0.007866922130159496, 
    'Shekel':0.01641902780068772,
    'Bitcoin':0.0029754729669716155, 
    'Yuan':0.02876502011140577, 
    'Yen':0.0070611970410221925, 
    'Australian Dollar':0.015086726472192969, 
    'Canadian Dollar':0.0037114108436639556,
    'Brazil Real':0.002835898542081689, 
    'UK Pound':0.003444950577965005, 
    'Saudi Riyal':0.0053926482343835255, 
    'Swiss Franc':0.0022649122584410806}

payment_currency_dict = {
    'Euro' : 0.10290441689611857, 
    'US Dollar': 0.7803415766834579, 
    'Australian Dollar' : 0.009351486467625078, 
    'Saudi Riyal' : 0.002258567966400629, 
    'UK Pound': 0.016406339216606818, 
    'Yuan': 0.028872873076093437, 
    'Rupee': 0.007860577838119043, 
    'Swiss Franc': 0.003705066551623504, 
    'Yen':0.011527578637499841 , 
    'Shekel': 0.0070611970410221925 , 
    'Ruble': 0.015080382180152516, 
    'Canadian Dollar': 0.005379959650302623, 
    'Brazil Real': 0.0029754729669716155, 
    'Bitcoin':0.002835898542081689, 
    'Mexican Peso':0.003438606285924554}


def get_currency_name(value, currency_dict):
    for name, val in currency_dict.items():
        if abs(val - value) < 1e-6: 
            return name
    return "Unknown" 
def user_input_features():

    
    col1, col2 = st.columns(2)
    
    with col1:
        from_bank = st.number_input('From Bank', min_value=0, max_value=1000000, value=0)
        to_bank = st.number_input('To Bank', min_value=0, max_value=1000000, value=0)
        amount_received = st.number_input('Amount Received', min_value=0.0, max_value=1000000.0, value=0.0)
        receiving_currency = st.selectbox('Receiving Currency', ('US Dollar', 'Euro', 'Rupee', 'Mexican Peso', 'Ruble', 'Shekel', 'Bitcoin', 'Yuan', 'Yen', 'Australian Dollar', 'Canadian Dollar', 'Brazil Real', 'UK Pound', 'Saudi Riyal', 'Swiss Franc'))
        amount_paid = st.number_input('Amount Paid', min_value=0.0, max_value=1000000.0, value=0.0)
        payment_currency = st.selectbox('Payment Currency', ('US Dollar', 'Euro', 'Rupee', 'Mexican Peso', 'Ruble', 'Shekel', 'Bitcoin', 'Yuan', 'Yen', 'Australian Dollar', 'Canadian Dollar', 'Brazil Real', 'UK Pound', 'Saudi Riyal', 'Swiss Franc'))
        
    
    with col2:
        hour = st.number_input('Hour', min_value=0, max_value=23, value=0)
        minutes = st.number_input('Minutes', min_value=0, max_value=59, value=0)
        different_account = st.selectbox('Different Account', [True, False])
        
        payment_format = st.radio(
            'Payment Format',
            ['ACH', 'Bitcoin', 'Cash', 'Cheque', 'Credit Card', 'Reinvestment', 'Wire']
        )
    
    data = {
        'From Bank': from_bank,
        'To Bank': to_bank,
        'Amount Received': amount_received,
        'Receiving Currency': receiving_currency_dict[receiving_currency],
        'Amount Paid': amount_paid,
        'Payment Currency': payment_currency_dict[payment_currency],
        'Hour': hour,
        'Minutes': minutes,
        'Different Account': 1 if different_account else 0,
        'Payment Format_ACH': payment_format == 'ACH',
        'Payment Format_Bitcoin': payment_format == 'Bitcoin',
        'Payment Format_Cash': payment_format == 'Cash',
        'Payment Format_Cheque': payment_format == 'Cheque',
        'Payment Format_Credit Card': payment_format == 'Credit Card',
        'Payment Format_Reinvestment': payment_format == 'Reinvestment',
        'Payment Format_Wire': payment_format == 'Wire'
    }
    features = pd.DataFrame(data, index=[0])
    return features

def home():
    st.markdown("<h1 style='text-align: center;'>Money Laundering Prediction</h1>", unsafe_allow_html=True)
    
    # Display image
    st.image("img.jpg", use_column_width=True)
    
    
    st.write("""
    The money laundering detection project using machine learning aims to identify suspicious financial transactions that may be related to money laundering activities. Money laundering is the process of disguising the origin of illegally obtained money so that it appears legitimate. This project uses several machine learning models, namely Decision Tree, Naive Bayes, K-Nearest Neighbors (kNN), and Logistic Regression. Decision Tree, was used because of its ability to map complex decisions into a set of easy-to-understand rules. Naive Bayes, assuming independence between features, is very effective in quick and simple classification. Meanwhile, kNN is used to identify transactions that are similar to patterns already known to be suspicious, based on proximity in feature space.

    """)
    
    st.write("""
    The main benefit of this project is the increased efficiency and accuracy in detecting suspicious transactions. Using machine learning models, the system can quickly analyze large amounts of transaction data and provide early warnings to financial investigators. For example, the kNN model developed using a dataset from Kaggle.com achieved 98.4% accuracy in detecting suspicious transactions. In addition, the Naive Bayes model has also shown promising results in classifying transactions as illegal or not. The implementation of these models is expected to reduce the number of false positives and increase the detection of true positives, thereby minimizing financial and reputational risks for financial institutions.


    """)
    
    # Button to navigate to prediction page
    if st.button('Test Run'):
        st.session_state['page'] = 'Prediction'
        st.experimental_rerun()



def history():
    st.title('Transaction History')
    if 'transactions' in st.session_state and st.session_state['transactions']:
        data = []
        for i, transaction in enumerate(st.session_state['transactions'], 1):
            input_features = transaction['input']
            
            transaction_data = {
                'Transaction': i,
                'From Bank': input_features['From Bank'],
                'To Bank': input_features['To Bank'],
                'Amount Received': input_features['Amount Received'],
                'Receiving Currency': transaction['receiving_currency_name'],
                'Amount Paid': input_features['Amount Paid'],
                'Payment Currency': transaction['payment_currency_name'],
                'Hour': input_features['Hour'],
                'Minutes': input_features['Minutes'],
                'Different Account': 'Yes' if input_features['Different Account'] == 1 else 'No',
                'Payment Format': next(key.replace('Payment Format_', '') for key, value in input_features.items() if key.startswith('Payment Format_') and value),
                'Model Used': transaction['model'],
                'Prediction': 'Money Laundering' if transaction['prediction'] == 1 else 'Not Money Laundering',
                'Prediction Probability': f"{transaction['probability']:.2f}%"
            }
            
            data.append(transaction_data)
        
        df = pd.DataFrame(data)
        
        def highlight_prediction(val):
            color = 'green' if val == 'Money Laundering' else 'red'
            return f'color: {color};'
        
        def highlight_probability(val):
            return 'font-weight: bold; color: blue;'
        
        styled_df = df.style.applymap(highlight_prediction, subset=['Prediction'])
        styled_df = styled_df.applymap(highlight_probability, subset=['Prediction Probability'])
        
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.write("No transactions recorded yet.") 

# Function to display the prediction page
def prediction():
    st.title('Money Laundering Detection')
    st.write('Input the features to predict if the transaction is money laundering or not.')

    model_name = st.selectbox('Select Model', ('Decision Tree', 'Naive Bayes', 'KNN', 'Logistic Regression'))
    input_df = user_input_features()

    st.subheader('User Input Summary')
    st.write(input_df)

    if st.button('Predict'):
        if model_name == 'Decision Tree':
            model = decision_tree_model
        elif model_name == 'Naive Bayes':
            model = naive_bayes_model
        elif model_name == 'Logistic Regression':
            model = log_model
        else:
            model = knn_model

        prediction = model.predict(input_df)
        prediction_proba = np.max(model.predict_proba(input_df))

        st.subheader('Prediction')
        st.success('Money Laundering' if prediction[0] == 1 else 'Not Money Laundering')

        st.subheader('Prediction Probability')
        st.success(f'Accuracy: {prediction_proba * 100:.2f}%')

        # Save transaction details
        if 'transactions' not in st.session_state:
            st.session_state['transactions'] = []
        
        st.session_state['transactions'].append({
    'input': input_df.to_dict(orient='records')[0],
    'model': model_name,
    'prediction': prediction[0],
    'probability': prediction_proba * 100,
    'receiving_currency_name': get_currency_name(input_df['Receiving Currency'].values[0], receiving_currency_dict),
    'payment_currency_name': get_currency_name(input_df['Payment Currency'].values[0], payment_currency_dict)
})
        

def bar_graph(df, title, column_name):
    chart = alt.Chart(df, title=title).mark_bar(
        opacity=1,
    ).encode(
        column=alt.Column(f'{column_name}:N', spacing=10, 
        header=alt.Header(labelOrient="bottom", labelAnchor="start", labelAngle=90)),
        x=alt.X('Is Laundering:N', axis=None),
        y=alt.Y('count()', title='Count', axis=alt.Axis(grid=True)),
        color=alt.Color('Is Laundering:N', scale=alt.Scale(range=['blue', 'orange']), legend=alt.Legend(title="Is Laundering"))
    ).configure_view(
        stroke='transparent',
    ).properties(
        width=450/df[column_name].nunique()
    )

    st.altair_chart(chart)

def line_graph(df, title, column_name):
    dataset = df[df['Is Laundering'] == 1]
    chart = alt.Chart(dataset, title=title).mark_line(point=True).encode(
        x=f'{column_name}:Q',
        y=alt.Y(f'count()', title="Amount of Money Laundering")
    ).properties(
        width=650, 
        height=500
    )

    st.altair_chart(chart)
    

def graph():
    st.title("Data Graph")
    df = pd.read_csv('money_laundry_dataset.csv')
    tab_names = ["Receiving Currency", "Payment Currency", "Hour", "Minutes", "Different Account", "Payment Format"]
    all_tabs = st.tabs(tab_names)

    for index, tab in enumerate(all_tabs):
        with tab:
            if df[tab_names[index]].dtypes != 'O' and tab_names[index] != "Different Account":
                line_graph(df, tab_names[index], tab_names[index])
            else:
                bar_graph(df, tab_names[index], tab_names[index])

# Buat tampilin Main menu
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Prediction", "History", "Graph"],  # Added "History"
        icons=["house", "graph-up", "clock-history", "bar-chart-line"],  # Added icon for History
        menu_icon="cast",
        styles={
            "nav-link-selected": {"background-color": "red"},
        },
        default_index=0,
    )

if selected == "Home":
    home()
elif selected == "Prediction":
    prediction()
elif selected == "History":
    history()
elif selected == "Graph":
    graph()