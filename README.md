# CDL Predictor
Match Predictor for the Call of Duty League

## What it does
Predictive model that forecasts the outcome of a best-of-5 CDL match using user inputted maps and the standard best-of-5 gamemode order. Built in Python and trained on historical team specific map and mode data.

## How a CDL match works
A typical CDL match consists of 5 maps: Hardpoint (HP), Search and Destroy (SND), Control (CTRL), HP, SND. The 2 teams facing off go to through a pick and ban system where each team gets to ban and pick a map for each gamemode.
> For HP: Team 1 bans Map A, Team 2 bans Map B, Team 1 picks Map 1, Team 2 picks Map 4. Same process for SND and CTRL.

## How the model works
1. Data collection
   - Pulls team and map data from [Google Sheets](https://docs.google.com/spreadsheets/d/1XghAKHG41UdaJJ2-XizsmcDv-Pf8t5lLLA_eKtUEkBM/edit?gid=0#gid=0)
   - Uses fuzzy matching to match user inputs with team and map variables
     
2. User Inputs
   - User selects two teams and maps for each of the 5 games
     
3. Model training
   - Trains three Random Forest classifiers, one per game mode
   - Features include team performance per map/mode, points per game, etc.

2. Output
   - Displays predicted scoreline and winning team
   - Includes model accuracy for transparency

## Sample Output

| Winner  | Mode                | Map      | Team 1  | Team 2 |
|---------|---------------------|----------|---------|--------|
| Breach  | Hardpoint           | Vault    | Thieves | Breach |
| Breach  | Search and Destroy  | Hacienda | Thieves | Breach |
| Thieves | Control             | Vault    | Thieves | Breach |
| Thieves | Hardpoint           | Rewind   | Thieves | Breach |
| Breach  | Search and Destroy  | Red Card | Thieves | Breach |

## Accuracy
Hardpoint Model Accuracy: 66.00%

Search and Destroy Model Accuracy:  62.50%

Control Model Accuracy: 83.87%
