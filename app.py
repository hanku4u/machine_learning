import streamlit as st
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np
import pandas as pd

# Load your model (Update path and method according to your model type)
@st.cache_data()
def load_model():
    # return load('voting_classifier.pkl')
    return load('voting_regressor.pkl')

def create_game_helper(
        teamID_1: int,
        teamID_2: int,
        team1_data: pd.DataFrame,
        team2_data: pd.DataFrame) -> pd.DataFrame:
    """helper function that will look up agg data for two selected teams and concat them into single dataframe
    that can be passed to the model to make predictions

    Args:
        teamID_1 (int): id of the first team
        teamID_2 (int): id of the second team
        team1_data (pd.DataFrame): appropriate data source for the first team. i.e. home, away or neutral
        team2_data (pd.DataFrame): appropriate data source for the second team. i.e. home, away or neutral

    Returns:
        pd.DataFrame: dataframe with a single row of data with stat aggregations from each team
    """

    # select two teams and concat them into on df. then add the 'Loc_A', 'Loc_H', 'Loc_N' columns
    predict_game = team1_data[team1_data['TeamID'] == teamID_1].reset_index(drop=True)
    team2 = team2_data[team2_data['TeamID'] == teamID_2].reset_index(drop=True)

    # for team2 drop the teamID and add prefix for columns
    team2 = team2.add_prefix('opp_')

    # predict_game = pd.concat([team1, team2], axis=1)
    for col in team2.columns:
        predict_game[col] = team2[col]

    # add location columns
    predict_game['Loc_A'] = False
    predict_game['Loc_H'] = False
    predict_game['Loc_N'] = True

    # reorder columns to match training data
    predict_game = predict_game[[
        'TeamID','Score','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl',
        'Blk','PF','Loc_A','Loc_H','Loc_N','opp_TeamID','opp_Score','opp_FGM','opp_FGA','opp_FGM3',
        'opp_FGA3','opp_FTM','opp_FTA','opp_OR','opp_DR','opp_Ast','opp_TO','opp_Stl','opp_Blk','opp_PF'
    ]]

    return predict_game

# initialize a standard scaler and scale it on the training data
scaler = StandardScaler()

# Fit the scaler on the training data
training_data = pd.read_csv('./training_data.csv')
scaler.fit(training_data)

model = load_model()

# Teams for Men's and Women's basketball
m_teams = pd.read_csv('./data/MTeams.csv')
w_teams = pd.read_csv('./data/WTeams.csv')

# Load Home, Away and Neutral location game aggregations
home_data = pd.read_csv('./home_game_aggregations.csv')
away_data = pd.read_csv('./away_game_aggregations.csv')
neutral_data = pd.read_csv('./neutral_game_aggregations.csv')

# Setting up the title of the app
st.markdown("""
    <div style='text-align: center'>
        <h1>NCAA Basketball Game Predictor</h1>
        <h3>Nicholas Haight<br>
        Austin Community College<br>
        COSC-3380-002: Machine Learning I<br>
        Dr. Sajjad Mohsin</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown('---')

# Selecting the type of game
game_type = st.radio("Select the type of game", ('Men\'s Basketball', 'Women\'s Basketball'))

# Updating teams based on game type selected
if game_type == 'Men\'s Basketball':
    teams = m_teams
    teams_list = m_teams['TeamName'].unique().tolist()
    st.subheader("Means Teams Selected")
else:
    teams = w_teams
    teams_list = w_teams['TeamName'].unique().tolist()
    st.subheader("Womens Teams Selected")

# Creating a form for user input
with st.form(key='my_form'):
    home_team = st.selectbox('Select Home Team', teams_list)
    away_team = st.selectbox('Select Away Team', teams_list)
    neutral_location = st.checkbox('Neutral location')
    submit_button = st.form_submit_button(label='Predict')

# Making predictions
if submit_button:
    # Get teamID for each team
    team1_ID = teams[teams['TeamName'] == home_team]['TeamID'].iloc[0]
    team1_ID = team1_ID

    team2_ID = teams[teams['TeamName'] == away_team]['TeamID'].iloc[0]
    team2_ID = team2_ID

    # select data sources for each team
    if neutral_location:
        team1_data = neutral_data
        team2_data = neutral_data
    else:
        team1_data = home_data
        team2_data = away_data

    prediction_data = create_game_helper(
        teamID_1=team1_ID,
        teamID_2=team2_ID,
        team1_data=team1_data,
        team2_data=team2_data
    )

    pred_scaled = scaler.transform(prediction_data)
    prediction = model.predict(pred_scaled)
    threshold = 0.5
    binary_predictions = np.where(prediction > threshold, 1, 0)
    binary_predictions = binary_predictions.astype(bool)
    
    st.markdown('-----')
    if binary_predictions:
        st.write('Prediciton is true')
        st.header(f'{home_team} is the predicted winner')
    
    elif not binary_predictions:
        st.write('Prediction is false')
        st.header(f'{away_team} is the predicted winner')

    st.markdown('---')
    st.header('Data used to make prediction')

    col1, col2, col3 = st.columns([5, 1, 5])
    with col1:
        if neutral_location:
            st.header("Neutral Location")
            st.subheader(home_team)
        else:
            st.header("Home Team")
            st.subheader(home_team)
        st.write(f"Score: {prediction_data['Score'].iloc[0].round(2)}")
        st.write(f"Field Goals Attempted: {prediction_data['FGA'].iloc[0].round(2)}")
        st.write(f"Field Goals Made: {prediction_data['FGM'].iloc[0].round(2)}")
        st.write(f"3-pointers Attempted: {prediction_data['FGA3'].iloc[0].round(2)}")
        st.write(f"3-pointers Made: {prediction_data['FGM3'].iloc[0].round(2)}")
        st.write(f"Free Throws Attempted: {prediction_data['FTA'].iloc[0].round(2)}")
        st.write(f"Free Throws Made: {prediction_data['FTM'].iloc[0].round(2)}")
        st.write(f"Offensive Rebounds: {prediction_data['OR'].iloc[0].round(2)}")
        st.write(f"Defensive Rebounds: {prediction_data['DR'].iloc[0].round(2)}")
        st.write(f"Assists: {prediction_data['Ast'].iloc[0].round(2)}")
        st.write(f"Turnovers: {prediction_data['TO'].iloc[0].round(2)}")
        st.write(f"Blocks: {prediction_data['Blk'].iloc[0].round(2)}")
        st.write(f"Steals: {prediction_data['Stl'].iloc[0].round(2)}")
        st.write(f"Total Personal Fouls: {prediction_data['PF'].iloc[0].round(2)}")

    with col3:
        if neutral_location:
            st.header("Neutral Location")
            st.subheader(away_team)
        else:
            st.header("Away Team")
            st.subheader(away_team)
        st.write(f"Score: {prediction_data['opp_Score'].iloc[0].round(2)}")
        st.write(f"Field Goals Attempted: {prediction_data['opp_FGA'].iloc[0].round(2)}")
        st.write(f"Field Goals Made: {prediction_data['opp_FGM'].iloc[0].round(2)}")
        st.write(f"3-pointers Attempted: {prediction_data['opp_FGA3'].iloc[0].round(2)}")
        st.write(f"3-pointers Made: {prediction_data['opp_FGM3'].iloc[0].round(2)}")
        st.write(f"Free Throws Attempted: {prediction_data['opp_FTA'].iloc[0].round(2)}")
        st.write(f"Free Throws Made: {prediction_data['opp_FTM'].iloc[0].round(2)}")
        st.write(f"Offensive Rebounds: {prediction_data['opp_OR'].iloc[0].round(2)}")
        st.write(f"Defensive Rebounds: {prediction_data['opp_DR'].iloc[0].round(2)}")
        st.write(f"Assists: {prediction_data['opp_Ast'].iloc[0].round(2)}")
        st.write(f"Turnovers: {prediction_data['opp_TO'].iloc[0].round(2)}")
        st.write(f"Blocks: {prediction_data['opp_Blk'].iloc[0].round(2)}")
        st.write(f"Steals: {prediction_data['opp_Stl'].iloc[0].round(2)}")
        st.write(f"Total Personal Fouls: {prediction_data['opp_PF'].iloc[0].round(2)}")
