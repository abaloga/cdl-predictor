import pandas as pd
from rapidfuzz import process
from rapidfuzz.fuzz import token_ratio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Part 1: Getting urls

wb_id = "1XghAKHG41UdaJJ2-XizsmcDv-Pf8t5lLLA_eKtUEkBM"

# HP data
hp_sheet = "Hardpoint Stats"
hp_url = f"https://docs.google.com/spreadsheets/d/{wb_id}/gviz/tq?tqx=out:csv&sheet={hp_sheet}"
hp_url = hp_url.replace(" ", "%20")  # Replace spaces with %20 for hp_url encoding

# Control data
ctrl_sheet = "Control Stats"
ctrl_url = f"https://docs.google.com/spreadsheets/d/{wb_id}/gviz/tq?tqx=out:csv&sheet={ctrl_sheet}"
ctrl_url = ctrl_url.replace(" ", "%20")  # Replace spaces with %20 for hp_url encoding

# S&D data
snd_sheet = "SnD Stats"
snd_url = f"https://docs.google.com/spreadsheets/d/{wb_id}/gviz/tq?tqx=out:csv&sheet={snd_sheet}"
snd_url = snd_url.replace(" ", "%20")  # Replace spaces with %20 for hp_url encoding

###



### Part 2: Creating df'S & cleaning data
    # HP df
hp_df = pd.read_csv(hp_url) # Create hp_df
hp_df = hp_df.drop(hp_df.columns[[32, 33, 34, 35, 36, 37, 38, 39, 40]], axis=1)  # Drop unnecessary columns
hp_df = hp_df.rename(columns={"Overall Team": "Team"})  # Rename 1st column to "Team"
hp_df = hp_df[hp_df['PA'].notnull()] # drop unnecessary rows

hp_map_list = (
    ['Overall'] * 12 +
    ['Protocol'] * 12 +
    ['Red Card'] * 12 +
    ['Rewind'] * 12 +
    ['Skyline'] * 12 +
    ['Vault'] * 12 +
    ['Hacienda'] * 12
)
hp_df.insert(1, "Map", hp_map_list)  # Insert "Map" column for HPs

hp_df.to_csv('hp.csv', index=True) # save hp df to csv


    # SnD df
snd_df = pd.read_csv(snd_url) # Create snd_df
snd_df = snd_df.drop(snd_df.columns[[49]], axis = 1)  # Drop unnecessary columns
snd_df = snd_df.rename(columns={"OVERALL Team": "Team"})  # Rename 1st column to "Team"
snd_df = snd_df[snd_df['ROUNDS Total'].notnull()] # drop unnecessary rows

snd_map_list = (
    ['Overall'] * 12 +
    ['Protocol'] * 12 +
    ['Red Card'] * 12 +
    ['Rewind'] * 12 +
    ['Skyline'] * 12 +
    ['Vault'] * 12 +
    ['Hacienda'] * 12 +
    ['Dealership'] * 12
)
snd_df.insert(1, "Map", snd_map_list)  # Insert "Map" column for SnD

snd_df.to_csv("snd.csv", index=True) # save snd df to csv


    #Control df
ctrl_df = pd.read_csv(ctrl_url, skip_blank_lines=True) # Create ctrl_df
ctrl_df = ctrl_df.drop(ctrl_df.columns[[55]], axis = 1) # Drop unnecessary columns
ctrl_df = ctrl_df.rename(columns={"Overall Team": "Team"}) # Rename 1st column to Team
ctrl_df = ctrl_df[ctrl_df['Rounds Total'].notnull()] # drop unnecessary rows


ctrl_map_list = (
    ['Overall'] * 12 +
    ['Protocol'] * 12 +
    ['Vault'] * 12 +
    ['Hacienda'] * 12
)
ctrl_df.insert(1, 'Map', ctrl_map_list)

ctrl_df.to_csv("ctrl.csv", index=True) # save ctrl df to csv

###



### Part 3: Getting Team Inputs

team_list = {
    'FaZe',
    'Thieves',
    'Ultra',
    'OpTic',
    'Breach',
    'Falcons',
    'Cloud9',
    'Rokkr',
    'Guerrillas',
    'Heretics',
    'Royal Ravens',
    'Surge'
}

#team_one  = input("Enter Team 1:")
team_one = "Heretics"
team_one, team_one_score, _ = process.extractOne(team_one, team_list, scorer=token_ratio) # normalizing user inputted team

#team_two = input("Enter Team 2:")
team_two = "guerrillas"
team_two, team_two_score, _ = process.extractOne(team_two, team_list,scorer=token_ratio) # normalizing user inputted team

if team_one_score >= 50 and team_two_score >= 50:   # check if team is found
    print(team_one)
    print(team_two)
else:
    print("One or both teams not found")

###



### Part 3.5: Getting Map Inputs

while True:   # Map 1
    map_one = "protocol"
#    map_one = input("Enter Map 1:")
    map_one, map_one_score, _ = process.extractOne(map_one, hp_map_list, scorer=token_ratio) # normalizing user inputted map
    if map_one_score >= 60:   # check if map is found
        break
    else:
        excluded_hp = {"Overall"}
        print("Please enter a valid map from the list:" , set(m for m in hp_map_list if m not in excluded_hp))

while True:   # Map 2
    map_two = "hacienda"
#    map_two = input("Enter Map 2:")
    map_two, map_two_score, _ = process.extractOne(map_two, snd_map_list, scorer=token_ratio) # normalizing user inputted map
    if map_two_score >= 60:   # check if map is found
        break
    else:
        excluded_snd = {"Overall"}
        print("Please enter a valid map from the list:" , set(m for m in snd_map_list if m not in excluded_snd))

while True:   # Map 3
    map_three = "vault"
#    map_three = input("Enter Map 3:")
    map_three, map_three_score, _ = process.extractOne(map_three, ctrl_map_list, scorer=token_ratio) # normalizing user inputted map
    if map_three_score >= 60:   # check if map is found
        break
    else:
        excluded_ctrl = {"Overall"}
        print("Please enter a valid map from the list:" , set(m for m in ctrl_map_list if m not in excluded_ctrl))

while True:   # Map 4
    map_four = "hacienda"
#    map_four = input("Enter Map 4:")
    map_four, map_four_score, _ = process.extractOne(map_four, hp_map_list, scorer=token_ratio) # normalizing user inputted map
    if map_four_score >= 60 and map_four != map_one:   # check if map is found
        break
    else:
        excluded_hp = {"Overall", map_one}
        print("Please enter a valid map from the list:" , set(m for m in hp_map_list if m not in excluded_hp))

while True:   # Map 5
    map_five = "red card"
#    map_five = input("Enter Map 5:")
    map_five, map_five_score, _ = process.extractOne(map_five, snd_map_list, scorer=token_ratio) # normalizing user inputted map
    if map_five_score >= 60 and map_five != map_two:   # check if map is found
        break
    else:
        excluded_snd = {"Overall", map_two}
        print("Please enter a valid map from the list:" , set(m for m in snd_map_list if m not in excluded_snd))

###



### Part 4: Model Data

# HP Data
t1_m1_ppg = hp_df.loc[((hp_df['Team'] == team_one) & (hp_df['Map'] == map_one)), 'PPG'].iloc[0]

t2_m1_ppg = hp_df.loc[((hp_df['Team'] == team_two) & (hp_df['Map'] == map_one)), 'PPG'].iloc[0]

t1_m4_ppg = hp_df.loc[((hp_df['Team'] == team_one) & (hp_df['Map'] == map_four)), 'PPG'].iloc[0]

t2_m4_ppg = hp_df.loc[((hp_df['Team'] == team_two) & (hp_df['Map'] == map_four)), 'PPG'].iloc[0]

# SnD Data
t1_m2_ppg = snd_df.loc[((snd_df['Team'] == team_one) & (snd_df['Map'] == map_two)), 'PPG'].iloc[0]

t2_m2_ppg = snd_df.loc[((snd_df['Team'] == team_two) & (snd_df['Map'] == map_two)), 'PPG'].iloc[0]

t1_m5_ppg = snd_df.loc[((snd_df['Team'] == team_one) & (snd_df['Map'] == map_five)), 'PPG'].iloc[0]

t2_m5_ppg = snd_df.loc[((snd_df['Team'] == team_two) & (snd_df['Map'] == map_five)), 'PPG'].iloc[0]

# Control Data
t1_m3_ppg = ctrl_df.loc[((ctrl_df['Team'] == team_one) & (ctrl_df['Map'] == map_three)), 'PPG'].iloc[0]

t2_m3_ppg = ctrl_df.loc[((ctrl_df['Team'] == team_two) & (ctrl_df['Map'] == map_three)), 'PPG'].iloc[0]

###



### Part 6: Training and testing control model

# Load and manipulate control training data
ctrl_train_df = pd.read_csv("model 101/ctrl_training.csv")

x_ctrl = ctrl_train_df.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
y_ctrl = ctrl_train_df["Winner"]

# Split into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x_ctrl, y_ctrl, test_size=0.2, random_state=42)

# Create and train the model
ctrl_model = RandomForestClassifier(n_estimators=100, random_state=42)
ctrl_model.fit(x_train, y_train)
y_pred = ctrl_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Control Model Accuracy: {accuracy:.2%}")

# Load and manipulate HP training data
hp_train_df = pd.read_csv("model 101/hp_training.csv")


x_hp = hp_train_df.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
y_hp = hp_train_df["Winner"]

# Split into train/test sets
x_hp_train, x_hp_test, y_hp_train, y_hp_test = train_test_split(x_hp, y_hp, test_size=0.2, random_state=42)

# Train the model
hp_model = RandomForestClassifier(n_estimators=100, random_state=42)
hp_model.fit(x_hp_train, y_hp_train)

y_hp_pred = hp_model.predict(x_hp_test)
hp_accuracy = accuracy_score(y_hp_test, y_hp_pred)

print(f"Hardpoint Model Accuracy: {hp_accuracy:.2%}")

# Load and manipulate SND training set
snd_train_df = pd.read_csv("model 101/snd_training.csv")

x_snd = snd_train_df.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
y_snd = snd_train_df["Winner"]

#split into training and test sets
x_snd_train, x_snd_test, y_snd_train, y_snd_test = train_test_split(x_snd, y_snd, test_size=0.2, random_state=42)

# Training the model
snd_model = RandomForestClassifier(n_estimators=100, random_state=42)
snd_model.fit(x_snd_train, y_snd_train)

y_snd_pred = snd_model.predict(x_snd_test)
snd_accuracy = accuracy_score(y_snd_test, y_snd_pred)

print(f"Search and Destroy Model Accuracy: {snd_accuracy: .2%}")

###

### Part 7: Predicting Matches

# Map 1 winner
t1_stats = hp_train_df[(hp_train_df['team_one'] == team_one) & (hp_train_df['Map'] == map_one)].copy()
t2_stats = hp_train_df[(hp_train_df['team_two'] == team_two) & (hp_train_df['Map'] == map_one)].copy()

t1_stats = t1_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
t2_stats = t2_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])

m1_features = pd.concat([t1_stats.reset_index(drop=True), t2_stats.reset_index(drop=True)], axis=1)

m1_features.to_csv("m1_features.csv", index=True)

m1_pred = hp_model.predict(m1_features)[0]

print(f"Map 1 Winner Prediction: {m1_pred}")

# Map 2 winner
t1_stats = snd_train_df[(snd_train_df['team_one'] == team_one) & (snd_train_df['Map'] == map_two)].copy()
t2_stats = snd_train_df[(snd_train_df['team_two'] == team_two) & (snd_train_df['Map'] == map_two)].copy()

t1_stats = t1_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
t2_stats = t2_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])

m2_features = pd.concat([t1_stats.reset_index(drop=True), t2_stats.reset_index(drop=True)], axis=1)

m2_pred = snd_model.predict(m2_features)[0]

print(f"Map 2 Winner Prediction: {m2_pred}")

# Map 3 winner

t1_stats = ctrl_train_df[(ctrl_train_df['team_one'] == team_one) & (ctrl_train_df['Map'] == map_three)].copy()
t2_stats = ctrl_train_df[(ctrl_train_df['team_two'] == team_two) & (ctrl_train_df['Map'] == map_three)].copy()

t1_stats = t1_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
t2_stats = t2_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])

m3_features = pd.concat([t1_stats.reset_index(drop=True), t2_stats.reset_index(drop=True)], axis=1)

m3_pred = ctrl_model.predict(m3_features)[0]

print(f"Map 3 Winner Prediction: {m3_pred}")

# Map 4 winner
t1_stats = hp_train_df[(hp_train_df['team_one'] == team_one) & (hp_train_df['Map'] == map_four)].copy()
t2_stats = hp_train_df[(hp_train_df['team_two'] == team_two) & (hp_train_df['Map'] == map_four)].copy()

t1_stats = t1_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
t2_stats = t2_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])

m4_features = pd.concat([t1_stats.reset_index(drop=True), t2_stats.reset_index(drop=True)], axis=1)

m4_pred = hp_model.predict(m4_features)[0]

print(f"Map 4 Winner Prediction: {m4_pred}")

# Map 5 winner
t1_stats = snd_train_df[(snd_train_df['team_one'] == team_one) & (snd_train_df['Map'] == map_five)].copy()
t2_stats = snd_train_df[(snd_train_df['team_two'] == team_two) & (snd_train_df['Map'] == map_five)].copy()

t1_stats = t1_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])
t2_stats = t2_stats.drop(columns=["Match #", "Game #", "Mode", "Map", "team_one", "team_two", "Winner", "team_score", "Score Against"])

m5_features = pd.concat([t1_stats.reset_index(drop=True), t2_stats.reset_index(drop=True)], axis=1)

m5_pred = snd_model.predict(m5_features)[0]


match_data = [
    {'Winner': m1_pred,'Mode': 'Hardpoint', 'Map': map_one, 'Team 1': team_one, 'Team 2': team_two},
    {'Winner': m2_pred,'Mode': 'Search and Destroy', 'Map': map_two, 'Team 1': team_one, 'Team 2': team_two},
    {'Winner': m3_pred,'Mode': 'Control', 'Map': map_three, 'Team 1': team_one, 'Team 2': team_two},
    {'Winner': m4_pred,'Mode': 'Hardpoint', 'Map': map_four, 'Team 1': team_one, 'Team 2': team_two},
    {'Winner': m5_pred,'Mode': 'Search and Destroy', 'Map': map_five, 'Team 1': team_one, 'Team 2': team_two},
]   # create columns, rows, data for df

match_df = pd.DataFrame(match_data) # Create match dataframe

match_df.index = range(1, len(match_df) + 1) # Make index start at 1
#match_df.to_csv("match.csv", index=True) # Save match to csv

print()
print(match_df)
print()