import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from io import StringIO

data = '''
100M	Jump	Gender	Age	100m_bin
16.06	3.47	w	40	16
14.29	3.79	m	70	14
57.85	0.63	w	90	57
15.92	3.39	w	35	15
12.52	4.57	w	35	12
14.25	4.25	w	55	14
10.92	6.76	m	30	10
14.91	4.26	w	45	14
13.86	3.67	w	45	13
14.6	3.82	w	30	14
14.08	4.32	m	65	14
14.65	3.64	w	45	14
17.07	2.28	w	40	17
18.71	2.92	w	70	18
18.64	2.61	m	85	18
24.05	1.95	m	85	24
14.57	4.42	w	30	14
12.75	5.41	m	60	12
12.48	4.73	m	50	12
14.93	4.1	    w	35	14
15.67	3.6	    m	75	15
12.08	5.44	m	50	12
15.43	3.87	w	45	15
11.18	6.82	m	35	11
26.04	1.89	m	90	26
14.79	3.57	w	35	14
24.31	1.79	w	85	24
19.42	2.63	w	70	19
16.01	2.48	w	65	16
15.94	3.65	w	60	15
20.24	2.71	w	70	20
10.35	7.55	m	30	10
12.59	4.57	m	70	12
14.58	3.78	w	60	14
102.53	0.29	m	90	102
13.7	4.3	    m	70	13
14.5	3.72	m	70	14
13.48	3.53	w	45	13
35.34	1.65	m	90	35
25.53	1.64	m	90	25
19.09	2.5	    m	90	19
26.92	1.21	w	90	26
16.04	3.59	m	80	16
35.46	0.78	w	85	35
13.16	4.76	w	35	13
24.68	2	    w	85	24
11.81	5.73	m	50	11
19.73	1.96	w	80	19
17.87	3.13	w	70	17
14.07	4.82	w	50	14
12.24	5.95	m	40	12
15.68	3.47	m	75	15
12.19	5.33	m	35	12
14.8	3.92	w	35	14
15.22	3.99	w	60	15
19.66	2.79	w	75	19
11.85	5.89	m	40	11
26.01	1.52	m	90	26
16.26	3.18	w	60	16
20.4	1.9	    w	40	20
14.39	3.93	w	50	14
15.34	3.76	w	60	15
12.59	5.97	m	55	12
12.73	5.12	m	50	12
12.61	5.32	m	55	12
24.53	1.78	m	85	24
13.48	4.27	m	65	13
14.3	3.92	w	60	14
19.4	3.46	w	75	19
15.17	3.61	w	55	15
15.94	3.18	w	55	15
11.78	5.51	m	50	11
31.55	1.08	w	80	31
16.37	4.06	w	70	16
15.36	3.9	    w	45	15
14.53	4.6	    m	70	14
11.92	5.3	    m	35	11
15.83	3.87	w	35	15
14.2	4.66	w	50	14
12.23	5	    m	45	12
11.48	5.62	m	35	11
12.87	5.2	    w	30	12
14.78	4.19	w	45	14
13.76	4.08	w	40	13
13.55	4.82	w	30	13
15.65	3.88	w	40	15
17.46	3.55	w	70	17
13.67	3.91	w	45	13
16.11	3.4	    w	60	16
23.19	2.18	w	85	23
12.12	4.82	m	40	12
26.02	1.77	w	80	26
12.45	6.55	w	30	12
16.8	3.11	w	40	16
15.87	3.6	    w	50	15
12.89	4.83	m	60	12
21.31	2.5	    m	90	21
16.01	3.41	w	55	16
14.29	4.24	w	45	14
16.78	3.11	w	40	16
18.42	3.3	    w	40	18
12.7	5.34	m	50	12
26.18	1.6	    w	80	26
20.36	2.32	w	75	20
14.44	4.61	w	55	14
17.33	2.7	    w	65	17
14.64	3.56	w	35	14
11.63	6.32	m	35	11
17.16	2.94	w	75	17
21.01	2.41	w	70	21
53.14	0.62	w	90	53
15.78	2.94	w	70	15
15.81	2.78	m	80	15
21.82	3.05	m	85	21
25.25	2.17	m	85	25
12.15	5.77	m	40	12
13.9	5	    m	65	13
18.04	3.12	w	40	18
24.6	1.44	m	90	24
37.34	1.11	m	95	37
12.33	5.81	m	40	12
13.7	4.72	m	65	13
14.04	4.47	m	70	14
17.78	2.83	m	85	17
11.26	6.38	m	35	11
12.8	4.6	    m	65	12
11.57	5.16	m	45	11
12.87	5.01	m	60	12
12.94	4.75	m	60	12
13.94	4.32	m	70	13
17.05	3.41	w	65	17
13.67	5.2	    w	40	13
24.43	1.8	    w	75	24
13.8	4.08	m	65	13
12.7	5.06	m	50	12
11.88	4.88	m	50	11
24.45	1.8	    m	95	24
14.34	4.12	w	60	14
20.64	2.19	w	70	20
16.97	3.16	m	80	16
13.41	4.51	m	70	13
14.53	3.86	m	70	14
13.97	3.87	w	35	13
11.74	5.54	m	40	11
18.08	2.76	w	65	18
19.94	2.46	m	85	19
15.85	3.59	w	65	15
14.97	3.35	w	40	14
17.63	2.59	m	80	17
15.27	3.21	m	75	15
22.04	2.31	m	95	22
25.4	1.66	m	85	25
16.27	3.43	m	80	16
10.58	7.62	m	30	10
15.08	3.98	m	75	15
'''

# Convert the string data to a pandas DataFrame
df = pd.read_csv(StringIO(data), sep='\t')

# Data cleaning steps
df['Gender'] = df['Gender'].str.upper()
df['100M'] = pd.to_numeric(df['100M'], errors='coerce')
df['Jump'] = pd.to_numeric(df['Jump'], errors='coerce')
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df.dropna(subset=['100M', 'Jump', 'Age', 'Gender'], inplace=True)
df = df[df['100M'] < 50]

# Define regression function
def perform_regression(x, y, regression_type):
    # Remove NaN values
    df_reg = pd.DataFrame({'x': x, 'y': y}).dropna()
    
    if regression_type == 'Linear':
        X = sm.add_constant(df_reg['x'])  # Adds a constant term to the predictor
        model = sm.OLS(df_reg['y'], X)
        results = model.fit()
        # Get equation parameters
        a = results.params['const']
        b = results.params['x']
        # Equation string
        equation = f'y = {a:.3f} + {b:.3f} * x'
        # R-squared
        r_squared = results.rsquared
        # Predicted values
        y_pred = results.predict(X)
        
    elif regression_type == 'Exponential':
        # Filter out y <= 0
        df_reg = df_reg[df_reg['y'] > 0]
        if df_reg.empty:
            raise ValueError("No data available after filtering out non-positive y values.")
        x = df_reg['x']
        y = df_reg['y']
        y_ln = np.log(y)
        X = sm.add_constant(x)
        model = sm.OLS(y_ln, X)
        results = model.fit()
        a_ln = results.params['const']
        b = results.params['x']
        a = np.exp(a_ln)
        equation = f'y = {a:.3f} * exp({b:.3f} * x)'
        y_pred = a * np.exp(b * x)
        # Calculate R-squared manually
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot
        
    elif regression_type == 'Logarithmic':
        # Filter out x <= 0
        df_reg = df_reg[df_reg['x'] > 0]
        if df_reg.empty:
            raise ValueError("No data available after filtering out non-positive x values.")
        x = df_reg['x']
        y = df_reg['y']
        x_ln = np.log(x)
        X = sm.add_constant(x_ln)
        model = sm.OLS(y, X)
        results = model.fit()
        a = results.params['const']
        b = results.params['x']
        equation = f'y = {a:.3f} + {b:.3f} * ln(x)'
        y_pred = results.predict(X)
        r_squared = results.rsquared
        
    else:
        raise ValueError("Invalid regression type")
    
    return equation, r_squared, x, y, y_pred


# Streamlit app code
st.title('Interactive Regression Dashboard')

# Sidebar for user inputs
st.sidebar.header('Filter Options')

gender_options = ['M', 'W', 'Both']
gender = st.sidebar.selectbox('Gender:', gender_options, index=2)

min_age = int(df['Age'].min())
max_age = int(df['Age'].max())
age_range = st.sidebar.slider('Age Range:', min_value=min_age, max_value=max_age, value=(min_age, max_age))

regression_type = st.sidebar.selectbox('Regression Type:', ['Linear', 'Exponential', 'Logarithmic'])

# Filter data based on selections
filtered_df = df.copy()
if gender != 'Both':
    filtered_df = filtered_df[filtered_df['Gender'] == gender]
filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    x = filtered_df['100M']
    y = filtered_df['Jump']

    # Perform regression
    try:
        equation, r_squared, x_vals, y_vals, y_pred = perform_regression(x, y, regression_type)
    except Exception as e:
        st.error(f"Error in regression: {e}")
        st.stop()

    # Sort x values for plotting the regression line
    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals.iloc[sorted_indices]
    y_pred_sorted = y_pred.iloc[sorted_indices]

    # Create the plot
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='markers',
        name='Data',
        marker=dict(color='blue')
    ))

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_sorted, y=y_pred_sorted,
        mode='lines',
        name=f'{regression_type} Regression',
        line=dict(color='red')
    ))

    # Update layout
    gender_label = 'Both Genders' if gender == 'Both' else ('Male' if gender == 'M' else 'Female')
    fig.update_layout(
        title=f'100m Time vs. Long Jump Distance ({gender_label}, Ages {age_range[0]}-{age_range[1]})',
        xaxis_title='100m Time (s)',
        yaxis_title='Long Jump Distance (m)',
        height=500,
        width=700
    )

    # Add equation and R-squared to the plot
    fig.add_annotation(
        x=0.5, y=1.1, xref='paper', yref='paper',
        text=f'{equation}<br>R² = {r_squared:.4f}',
        showarrow=False,
        font=dict(size=12),
        align='left',
        bordercolor='black',
        borderwidth=1
    )

    # Display the plot
    st.plotly_chart(fig)
