# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:00:54 2025

@author: mjmde
"""



############################################################
# TREATMENT SUMMARY FILE
############################################################

import pandas as pd
import ast

# Load event and meta data
event_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04.csv")
meta_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04-2.csv")

# Extract unique session-round-group combinations from event data
event_combos = event_df[['subsession', 'round', 'group']].drop_duplicates()
event_combos.rename(columns={
    'subsession': 'session.code',
    'round': 'subsession.round_number',
    'group': 'group.id_in_subsession'
}, inplace=True)

# Merge with meta data to pull treatment parameters
treatment_data = pd.merge(
    event_combos,
    meta_df[[
        'session.code',
        'subsession.round_number',
        'group.id_in_subsession',
        'subsession.players_per_group',
        'subsession.initial_stock',
        'subsession.cost_per_second',
        'subsession.show_chain'
    ]].drop_duplicates(),
    on=['session.code', 'subsession.round_number', 'group.id_in_subsession'],
    how='left'
)

# Clean inventory distribution format
def clean_inventory(stock):
    try:
        parsed = ast.literal_eval(stock)
        return ', '.join(map(str, parsed))
    except:
        return stock

treatment_data['Inventory Distribution'] = treatment_data['subsession.initial_stock'].apply(clean_inventory)
treatment_data['Agents'] = treatment_data['subsession.players_per_group']
treatment_data['Cost per Second'] = treatment_data['subsession.cost_per_second']
treatment_data['Transparency'] = treatment_data['subsession.show_chain'].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})

# Select and rename final columns
treatment_summary = treatment_data[[
    'session.code',
    'subsession.round_number',
    'group.id_in_subsession',
    'Agents',
    'Inventory Distribution',
    'Cost per Second',
    'Transparency'
]]
treatment_summary.rename(columns={
    'session.code': 'Session Code',
    'subsession.round_number': 'Round',
    'group.id_in_subsession': 'Group'
}, inplace=True)

# Save to CSV
treatment_summary.to_csv("treatments_summary.csv", index=False)
print("treatments_summary.csv saved successfully.")




############################################################
# CLEANING DATA
############################################################


import pandas as pd
import os
import ast

# Load data
event_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04.csv")
meta_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04-2.csv")
# Load treatments summary
treatments_summary = pd.read_csv("treatments_summary.csv")

# Create output directory
output_dir = "treatment_outputs"
os.makedirs(output_dir, exist_ok=True)

# Prepare treatment info including countdown
treatment_info = meta_df[[
    'session.code',
    'subsession.round_number',
    'group.id_in_subsession',
    'subsession.players_per_group',
    'subsession.initial_stock',
    'subsession.initial_cash',
    'subsession.cost_per_second',
    'subsession.price_per_unit',
    'subsession.show_chain',
    'subsession.countdown_seconds'
]].drop_duplicates()

# Loop per session, round, group in event data
for (session, round_num, group_num), group_df in event_df.groupby(['subsession', 'round', 'group']):
    
    # Match treatment on full triple key
    treatment = treatment_info[
        (treatment_info['session.code'] == session) &
        (treatment_info['subsession.round_number'] == round_num) &
        (treatment_info['group.id_in_subsession'] == group_num)
    ]
    if treatment.empty:
        continue

    row = treatment.iloc[0]
    players = row['subsession.players_per_group']
    stock_list = ast.literal_eval(str(row['subsession.initial_stock']))
    cash_list = ast.literal_eval(str(row['subsession.initial_cash']))
    earnings_per_item = float(row['subsession.price_per_unit'])
    cost_per_sec = float(row['subsession.cost_per_second'])
    chain = row['subsession.show_chain']
    countdown = float(row['subsession.countdown_seconds'])
    stock_str = '-'.join(map(str, stock_list))
    filename = f"session{session}_round{round_num}_group{group_num}_Agents{players}_IniInv{stock_str}_Cost{cost_per_sec}_Transp{chain}.csv"

    # Normalize timestamps using max init_time + countdown_seconds
    meta_subset = meta_df[
        (meta_df['session.code'] == session) &
        (meta_df['subsession.round_number'] == round_num) &
        (meta_df['group.id_in_subsession'] == group_num)
    ]
    if not meta_subset.empty and 'player.init_time' in meta_subset.columns:
        max_init_time = meta_subset['player.init_time'].max()
        group_df['time'] = group_df['time'] - (max_init_time + countdown)

    group_df['time'] = group_df['time'].clip(lower=0)

    # Duplicate rows and assign Agent
    new_rows = []
    for _, row in group_df.iterrows():
        original = row.copy()
        original['Agent'] = row['requested_by']
        new_rows.append(original)
        if row['requested_by'] != row['requested_from']:
            duplicate = row.copy()
            duplicate['Agent'] = row['requested_from']
            new_rows.append(duplicate)

    group_df = pd.DataFrame(new_rows)

    # Add action flags
    group_df['Requested'] = (group_df['Agent'] == group_df['requested_by']).astype(int)
    group_df['Collected'] = ((group_df['Agent'] == group_df['requested_by']) & (group_df['transferred'] == 1)).astype(int)
    group_df['Completed'] = ((group_df['Agent'] == group_df['requested_from']) & (group_df['transferred'] == 1)).astype(int)

    # Initialize columns
    group_df['Inventory'] = 0
    group_df['Balance'] = 0

    # Sort
    group_df = group_df.sort_values(by=['time', 'Agent']).reset_index(drop=True)

    # Add idle rows per second
    for t in range(1, 181):
        for a in range(1, players + 1):
            group_df.loc[len(group_df)] = {
                'time': float(t),
                'Agent': a,
                'Requested': 0,
                'Collected': 0,
                'Completed': 0,
                'Inventory': None,
                'Balance': 0
            }

    group_df = group_df.sort_values(by=['time', 'Agent']).reset_index(drop=True)

    # Initial values
    inventory_map = {i + 1: int(stock_list[i]) if i < len(stock_list) else 0 for i in range(players)}
    balance_map = {i + 1: float(cash_list[i]) if i < len(cash_list) else 0 for i in range(players)}

    # Update inventory and balance over time
    for idx, row in group_df.iterrows():
        agent = int(row['Agent'])
        time = float(row['time'])

        inv = int(inventory_map.get(agent, 0))
        bal = float(balance_map.get(agent, 0))

        if row['Requested'] == 1 and row['Collected'] == 1:
            inv += 1
        if row['Completed'] == 1:
            inv -= 1
            bal += earnings_per_item

        if time.is_integer() and time > 0:
            bal -= inv * cost_per_sec

        group_df.at[idx, 'Inventory'] = inv
        group_df.at[idx, 'Balance'] = bal

        inventory_map[agent] = inv
        balance_map[agent] = bal

    # Rename and reorder columns
    group_df.rename(columns={'time': 'Time Step'}, inplace=True)
    cols = list(group_df.columns)
    for col in ['Completed', 'Collected', 'Requested', 'Inventory', 'Balance', 'Agent']:
        cols.remove(col)
    insert_at = cols.index('Time Step') + 1
    reordered = cols[:insert_at] + ['Agent', 'Requested', 'Collected', 'Completed', 'Inventory', 'Balance'] + cols[insert_at:]
    group_df = group_df[reordered]

    # Drop unnecessary columns
    drop_cols = [
        'from_inventory', 'from_balance', 'to_inventory', 'to_balance',
        'subsession', 'round', 'group', 'kind',
        'requested_from', 'requested_by', 'units', 'transferred'
    ]
    group_df.drop(columns=[col for col in drop_cols if col in group_df.columns], inplace=True)
    
    # --- Add t=0 rows for all agents ---

    # Find the treatment row for this session/round/group
    treatment_row = treatments_summary[
        (treatments_summary['Session Code'] == session) &
        (treatments_summary['Round'] == round_num) &
        (treatments_summary['Group'] == group_num)
    ].iloc[0]
    
    # Parse inventory distribution
    initial_inventories = [int(x.strip()) for x in treatment_row['Inventory Distribution'].split(',')]
    
    # Build t=0 rows
    t0_rows = []
    for agent_id in range(1, players + 1):
        t0_rows.append({
            'Time Step': 0.0,
            'Agent': agent_id,
            'Requested': 0,
            'Collected': 0,
            'Completed': 0,
            'Inventory': initial_inventories[agent_id - 1] if agent_id - 1 < len(initial_inventories) else 0,
            'Balance': 300.0
        })
    
    # Convert t=0 rows to DataFrame
    t0_df = pd.DataFrame(t0_rows)
    
    # Append t=0 rows to group_df
    group_df = pd.concat([t0_df, group_df], ignore_index=True)
    
    # Resort by Time Step and Agent to keep nice order
    group_df = group_df.sort_values(by=['Time Step', 'Agent']).reset_index(drop=True)
    
    
       
    # Drop rows where Time Step >= 181
    group_df = group_df[group_df['Time Step'] < 181]

    # Save output
    filepath = os.path.join(output_dir, filename)
    group_df.to_csv(filepath, index=False)

print("All files saved.")



# delete files of technical problem groups

import os

output_dir = "treatment_outputs"

# Build the pattern prefix for the files to delete
prefix_to_delete = "sessionrjeu75l3_round"
group_str = "_group1_"

# List all files
filenames = os.listdir(output_dir)

# Filter files that match the session and group
files_to_delete = [
    f for f in filenames
    if f.startswith(prefix_to_delete) and group_str in f
]

# Print which files will be deleted
print("Files to delete:")
for f in files_to_delete:
    print(f)

# Actually delete them
for f in files_to_delete:
    os.remove(os.path.join(output_dir, f))

print("Selected files deleted.")




############################################################
# REGRESSION DATA
############################################################


# Clean Data even more to match with controls

import pandas as pd

# Load the metadata CSV
meta_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04-2.csv")

# Keep only the relevant columns
relevant_cols = [
    "session.code",
    "participant.code",
    "participant.id_in_session",
    "subsession.round_number",
    "group.id_in_subsession",
    "player.id_in_group"
]

meta_df = meta_df[relevant_cols].dropna()

# Assign group number based on participant.id_in_session
def assign_group(df):
    df = df.sort_values("participant.id_in_session")
    df["group"] = ((df["participant.id_in_session"] - 1) // 5) + 1
    return df

# Apply group assignment per session
meta_df = meta_df.groupby("session.code").apply(assign_group).reset_index(drop=True)

# Rename columns for clarity
meta_df = meta_df.rename(columns={
    "session.code": "session",
    "participant.code": "participant_code",
    "participant.id_in_session": "participant_id_in_session",
    "subsession.round_number": "round",
    "player.id_in_group": "agent"
})

# Select final columns in desired order
final_df = meta_df[[
    "participant_code", "session", "participant_id_in_session", "group", "round", "agent"
]]

# Sort for consistency
final_df = final_df.sort_values(by=["participant_code", "round"])

# Save to CSV
final_df.to_csv("agent_round_mapping.csv", index=False)
print("File saved as agent_round_mapping.csv")




import ast

# Load event and meta data
event_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04.csv")
meta_df = pd.read_csv("Sessions 1-6/ringsupplychain_4_2025-06-04-2.csv")

# Extract unique session-round-group combinations
event_combos = event_df[['subsession', 'round', 'group']].drop_duplicates()
event_combos.rename(columns={
    'subsession': 'session.code',
    'round': 'subsession.round_number',
    'group': 'group.id_in_subsession'
}, inplace=True)

# Merge with meta data to pull treatment parameters
treatment_data = pd.merge(
    event_combos,
    meta_df[[
        'session.code',
        'subsession.round_number',
        'group.id_in_subsession',
        'subsession.players_per_group',
        'subsession.initial_stock',
        'subsession.cost_per_second',
        'subsession.show_chain'
    ]].drop_duplicates(),
    on=['session.code', 'subsession.round_number', 'group.id_in_subsession'],
    how='left'
)

# Clean inventory distribution
def clean_inventory(stock):
    try:
        parsed = ast.literal_eval(stock)
        return ', '.join(map(str, parsed))
    except:
        return stock

treatment_data['Inventory Distribution'] = treatment_data['subsession.initial_stock'].apply(clean_inventory)
treatment_data['Cost per Second'] = treatment_data['subsession.cost_per_second']
treatment_data['Transparency'] = treatment_data['subsession.show_chain'].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No'})

# Rename for consistency with agent mapping
treatment_summary = treatment_data.rename(columns={
    'session.code': 'session',
    'subsession.round_number': 'round',
    'group.id_in_subsession': 'group'
})[[
    'session', 'round', 'group',
    'Inventory Distribution', 'Cost per Second', 'Transparency'
]]

# Expand to 5 rows per agent
treatment_expanded = treatment_summary.loc[treatment_summary.index.repeat(5)].copy()
treatment_expanded["agent"] = treatment_expanded.groupby(['session', 'round', 'group']).cumcount() + 1

# Reorder columns for final format
final_df = treatment_expanded[[
    "session", "group", "round", "agent",
    "Inventory Distribution", "Cost per Second", "Transparency"
]]

# Save
final_df.to_csv("treatments_per_agent_clean.csv", index=False)
print("treatments_per_agent_clean.csv created with correct headers.")




# Load the treatment-expanded file
treatment_df = pd.read_csv("treatments_per_agent_clean.csv")

# Load the agent mapping file (which includes participant_code)
agent_map_df = pd.read_csv("agent_round_mapping.csv")

# Merge on session, group, round, agent
merged_df = pd.merge(
    treatment_df,
    agent_map_df[['participant_code', 'session', 'group', 'round', 'agent']],
    on=['session', 'group', 'round', 'agent'],
    how='left'
)

# Reorder columns (optional)
merged_df = merged_df[[
    'participant_code', 'session', 'group', 'round', 'agent',
    'Inventory Distribution', 'Cost per Second', 'Transparency'
]]

# Save the final merged file
merged_df.to_csv("treatment_data_with_participant_code.csv", index=False)
print("treatment_data_with_participant_code.csv created successfully.")




# Load the treatment data file that already includes participant_code
treatment_df = pd.read_csv("treatment_data_with_participant_code.csv")

# Load the questionnaire data
questionnaire_df = pd.read_csv("Sessions 1-6/questionnaires_2025-06-04.csv")

# Select and rename relevant control columns
questionnaire_subset = questionnaire_df[[
    'participant.code',
    'player.gender',
    'player.birth_year',
    'player.education_level',
    'player.risk_general',
    'player.instructions_understood',
    'player.specific_strategy'
]].rename(columns={'participant.code': 'participant_code'})

# Merge on participant_code
merged_df = pd.merge(
    treatment_df,
    questionnaire_subset,
    on='participant_code',
    how='left'
)

# Save the enriched file
merged_df.to_csv("treatment_data_with_controls.csv", index=False)
print("File saved: treatment_data_with_controls.csv")


# Load full dataset
df = pd.read_csv("treatment_data_with_controls.csv")

# Filter out the unwanted rows
filtered_df = df[~((df['session'] == 'rjeu75l3') & (df['group'] == 1))]

# Save the cleaned file
filtered_df.to_csv("treatment_data_with_controls_filtered.csv", index=False)
print("Filtered file saved as treatment_data_with_controls_filtered.csv")


import pandas as pd
import os

# Load the filtered data (one row per agent per round)
master_df = pd.read_csv("treatment_data_with_controls_filtered.csv")

# Folder where the time-series files per session-round-group are saved
event_folder = "treatment_outputs"

# Add empty columns for the new data
master_df["balance_60s"] = None
master_df["balance_120s"] = None
master_df["balance_180s"] = None

master_df["click_freq_180s"] = None
master_df["click_freq_60s"] = None
master_df["click_freq_120s"] = None
master_df["click_freq_180s_last"] = None

# Loop through each row in master dataset
for idx, row in master_df.iterrows():
    session = row['session']
    group = row['group']
    round_ = row['round']
    agent = row['agent']

    # Identify correct file based on naming structure
    matching_file = None
    for file in os.listdir(event_folder):
        if file.startswith(f"session{session}_round{round_}_group{group}_") and file.endswith(".csv"):
            matching_file = os.path.join(event_folder, file)
            break

    if matching_file is None:
        print(f"File not found for {session}, round {round_}, group {group}")
        continue

    # Load matching time-series file
    try:
        ts_df = pd.read_csv(matching_file)
    except:
        print(f"Failed to read {matching_file}")
        continue

    agent_data = ts_df[ts_df["Agent"] == agent]

    # Balance at exact time points (take last row at or before the second)
    for t in [60, 120, 180]:
        bal_row = agent_data[agent_data["Time Step"] <= t].tail(1)
        if not bal_row.empty:
            master_df.at[idx, f"balance_{t}s"] = bal_row["Balance"].values[0]

    # Clicking frequency = sum(Requested) / time window
    # Window 1: 0–60
    window_1 = agent_data[(agent_data["Time Step"] >= 0) & (agent_data["Time Step"] <= 60)]
    window_2 = agent_data[(agent_data["Time Step"] > 60) & (agent_data["Time Step"] <= 120)]
    window_3 = agent_data[(agent_data["Time Step"] > 120) & (agent_data["Time Step"] <= 180)]
    
    master_df.at[idx, "click_freq_60s"] = window_1["Requested"].sum() / 60
    master_df.at[idx, "click_freq_120s"] = window_2["Requested"].sum() / 60
    master_df.at[idx, "click_freq_180s_last"] = window_3["Requested"].sum() / 60
    
    # Full 180s average
    master_df.at[idx, "click_freq_180s"] = agent_data["Requested"].sum() / 180

# Save the final DataFrame
master_df.to_csv("final_regression_ready_data.csv", index=False)
print("Saved as final_regression_ready_data.csv")




# Load your final dataset
df = pd.read_csv("final_regression_ready_data.csv")

# --- 1. Transparency Dummy ---
df["NT"] = (df["Transparency"] == "No").astype(int)

# --- 2. Inventory Total ---
def sum_inventory(inv_str):
    try:
        return sum(int(x.strip()) for x in inv_str.split(","))
    except:
        return None

df["inventory_total"] = df["Inventory Distribution"].apply(sum_inventory)

# --- 3. Symmetric Dummy ---
def is_symmetric(inv_str):
    try:
        values = [int(x.strip()) for x in inv_str.split(",")]
        return int(all(v == values[0] for v in values))
    except:
        return None

df["symmetric"] = df["Inventory Distribution"].apply(is_symmetric)

# Save result
df.to_csv("final_data_with_treatment_dummies.csv", index=False)
print("Dummies added and saved to final_data_with_treatment_dummies.csv")



# Finalized data to use for regression!
import pandas as pd

# Load the base dataset
df = pd.read_csv("final_data_with_treatment_dummies.csv")

# --- A. Create fixed effects group ID based on unique combination of session and group
df["group_fe"] = df.groupby(["session", "group"]).ngroup() + 1  # IDs start at 1

# --- 1. Create asymmetry dummy: 1 if asymmetric, 0 if symmetric
df["asymmetry"] = 1 - df["symmetric"]

# --- 2. Create transparency dummy: 1 = transparent, 0 = NT
df["transparent"] = 1 - df["NT"]

# --- 3. Inventory dummies (drop inventory=1 as baseline)
df["inv_3"] = (df["inventory_total"] == 3).astype(int)
df["inv_5"] = (df["inventory_total"] == 5).astype(int)
df["inv_10"] = (df["inventory_total"] == 10).astype(int)

# --- 4. Round dummies (drop round 1 as reference)
df["round_2"] = (df["round"] == 2).astype(int)
df["round_3"] = (df["round"] == 3).astype(int)

# --- 5. Gender dummy (1 = Female)
df["female"] = df["player.gender"].str.lower().eq("female").astype(int)

# --- 6. Age
df["age"] = 2025 - df["player.birth_year"]

# --- 7. Rename core variables for regression clarity
df.rename(columns={
    "click_freq_180s": "clickfreq",
    "balance_180s": "balance",
    "player.risk_general": "risk",
    "player.instructions_understood": "understood",
    "player.specific_strategy": "strategy",
    "player.education_level": "education"
}, inplace=True)

# --- 8. Save the enriched dataframe
df.to_csv("regression_ready_extended.csv", index=False)
print("Done. File: regression_ready_extended.csv")





###########################################################################
# REGRESSSIONS
###########################################################################

##########
# Balance and Clicking Frequency
##########


import pandas as pd

# Load data
df = pd.read_csv("regression_ready_extended.csv")

# Ensure education_high exists
# (optional if already created earlier)
df["education"] = df["education"].str.lower().str.strip()
df["education_high"] = df["education"].isin(["master", "phd"]).astype(int)

# Drop missing values
sub_df = df[["age", "education_high"]].dropna()

# Correlation
correlation = sub_df.corr().loc["age", "education_high"]
print(f"Pearson correlation between age and education_high: {correlation:.3f}")


# Run VIF for correlation

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices

# Load your data
df = pd.read_csv("regression_ready_extended.csv")

# Create formula for only the controls you're concerned about (not fixed effects)
controls = [
    "asymmetry", "transparent", "inv_3", "inv_5", "inv_10",
    "round_2", "round_3", "risk", "understood", "strategy",
    "female", "education", "age"
]

# Patsy formula to prepare the design matrix
formula = ' + '.join(controls)
y, X = dmatrices(f'balance ~ {formula}', data=df, return_type='dataframe')

# Calculate VIFs
vif_df = pd.DataFrame()
vif_df["Variable"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_df)



import pandas as pd
import statsmodels.formula.api as smf

# --- Load data
df = pd.read_csv("regression_ready_extended.csv")

# --- Setup fixed effect and cluster identifiers
df["group_fe"] = df["group_fe"].astype("category")
df["session"] = df["session"].astype("category")

# --- Create new education_high dummy
high_levels = ["master", "phd"]
df["education"] = df["education"].str.lower().str.strip()
df["education_high"] = df["education"].isin(high_levels).astype(int)

# --- Ensure 'female' is numeric
df["female"] = df["female"].astype(int)

# --- Define model formulas
formulas = [
    "balance ~ clickfreq + C(group_fe)",
    "balance ~ clickfreq + asymmetry + transparent + inv_3 + inv_5 + inv_10 + C(group_fe)",
    "balance ~ clickfreq + asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)",
    "balance_60s ~ click_freq_60s + asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)",
    "balance_120s ~ click_freq_120s + asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)",
    "balance ~ click_freq_180s_last + asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)"
]

model_names = [
    "1_Click_Only", "2_Treatments", "3_Controls",
    "4_60s", "5_120s", "6_180s"
]

# --- Save unformatted regression output to Excel
with pd.ExcelWriter("unformatted_balance_regressions.xlsx") as writer:
    # Group clustered models
    for name, formula in zip(model_names, formulas):
        model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["group_fe"]})
        model.summary2().tables[1].to_excel(writer, sheet_name=f"group_{name}")

    # Session clustered models
    for name, formula in zip(model_names, formulas):
        model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["session"]})
        model.summary2().tables[1].to_excel(writer, sheet_name=f"session_{name}")





##########
# Clicking Frequency and Treatment
##########



import pandas as pd
import statsmodels.formula.api as smf

# --- Load data
df = pd.read_csv("regression_ready_extended.csv")

# --- Setup fixed effect and cluster identifiers
df["group_fe"] = df["group_fe"].astype("category")
df["session"] = df["session"].astype("category")

# --- Create new education_high dummy
high_levels = ["master", "phd"]
df["education"] = df["education"].str.lower().str.strip()
df["education_high"] = df["education"].isin(high_levels).astype(int)

# --- Ensure 'female' is numeric
df["female"] = df["female"].astype(int)

# --- Define formulas
formulas = [
    "clickfreq ~ asymmetry + transparent + inv_3 + inv_5 + inv_10 + C(group_fe)",
    "clickfreq ~ asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)",
    "click_freq_60s ~ asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)",
    "click_freq_120s ~ asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)",
    "click_freq_180s_last ~ asymmetry + transparent + inv_3 + inv_5 + inv_10 + round + "
    "risk + female + education_high + age + understood + C(group_fe)"
]

model_names = [
    "1_Treatments_Only", "2_Controls",
    "3_60s", "4_120s", "5_180s"
]

# --- Save unformatted outputs to Excel
with pd.ExcelWriter("unformatted_regression_outputs_treatment and cliking frequency.xlsx") as writer:
    # Group clustered models
    for name, formula in zip(model_names, formulas):
        model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["group_fe"]})
        summary_df = model.summary2().tables[1]
        summary_df.to_excel(writer, sheet_name=f"group_{name}")

    # Session clustered models
    for name, formula in zip(model_names, formulas):
        model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["session"]})
        summary_df = model.summary2().tables[1]
        summary_df.to_excel(writer, sheet_name=f"session_{name}")





######################################################################
# ANALYSIS
######################################################################

######################################################################
# Hypothesis 1: End Balance vs. Market activity 
######################################################################


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro, spearmanr
import statsmodels.api as sm

# === Create output directory if needed ===
graph_output_dir = "treatment_graphs"
os.makedirs(graph_output_dir, exist_ok=True)



# Helper to parse Group ID and Round from filename
def parse_filename(filename):
    parts = filename.split('_')
    session = parts[0].replace('session', '')
    round_num = int(parts[1].replace('round', ''))
    group_id = int(parts[2].replace('group', ''))
    return session, group_id, round_num

# Prepare output list
summary_rows = []

# Loop over treatment output files
for filepath in glob.glob(os.path.join("treatment_outputs", "*.csv")):
    filename = os.path.basename(filepath)
    
    # Parse identifiers
    session, group_id, round_num = parse_filename(filename)
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Compute click frequency per agent
    click_freq_df = df[df["Requested"] == 1].groupby("Agent")["Requested"].count().reset_index()
    click_freq_df["Click Frequency"] = click_freq_df["Requested"] / 180.0
    click_freq_df.drop(columns=["Requested"], inplace=True)
    
    # Get final balance per agent at last time step
    final_bal_df = df[df["Time Step"] == df["Time Step"].max()].groupby("Agent")["Balance"].first().reset_index()
    final_bal_df.rename(columns={"Balance": "Final Balance"}, inplace=True)
    
    # Merge agent-level summaries
    agent_summary_df = pd.merge(click_freq_df, final_bal_df, on="Agent")
    
    # Compute group-round median for click frequency and final balance
    median_click_freq = agent_summary_df["Click Frequency"].median()
    median_final_bal = agent_summary_df["Final Balance"].median()
    
    # Append to summary
    summary_rows.append({
        "Session": session,
        "Group ID": group_id,
        "Round": round_num,
        "Click Frequency": median_click_freq,
        "Final Balance": median_final_bal
    })

# Create h6_df in memory
h6_df = pd.DataFrame(summary_rows)

print(f"Generated h6_df in memory with {len(h6_df)} rows (group-round).")



group_median_df = h6_df.groupby(["Session", "Group ID"]).agg({
    "Click Frequency": "median",
    "Final Balance": "median"
}).reset_index()

print(f"Aggregated to {len(group_median_df)} rows (one per experimental group).")



# === Prepare data for regression ===
X = group_median_df["Click Frequency"].values.reshape(-1, 1)
y = group_median_df["Final Balance"].values

# === Linear Regression using sklearn ===
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred

# === Linear Regression using statsmodels (for full statistics) ===
X_sm = sm.add_constant(group_median_df["Click Frequency"])  # add intercept
ols_model = sm.OLS(y, X_sm).fit()

# === Plot: Scatter with regression line ===
plt.figure(figsize=(8, 6))
plt.scatter(
    group_median_df["Click Frequency"],
    group_median_df["Final Balance"],
    color="#444444",
    marker='o',
    alpha=0.7,
    edgecolors='black',
    linewidths=0.6
)

# Plot regression line
x_range = np.linspace(X.min(), X.max(), 100)
x_range_sm = sm.add_constant(x_range)
y_range_pred = ols_model.predict(x_range_sm)
plt.plot(x_range, y_range_pred, color="#444444", linestyle='-', 
         label=f"$y = {ols_model.params[0]:.2f} + {ols_model.params[1]:.2f} \\cdot x$\n$R^2 = {ols_model.rsquared:.3f}$")

plt.title("Final Balance vs. Click Frequency")
plt.xlabel("Median Click Frequency per Group [clicks/s]")
plt.ylabel("Median Final Balance per Group [ECU]")
plt.legend()

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "Balance vs Market Activity Plot.png"), dpi=600)
plt.close()

# === Residuals vs Fitted ===
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, color="#444444", alpha=0.7, edgecolors='black', linewidths=0.6)
plt.axhline(0, color='black', linestyle='dashed')
plt.title("Residuals vs. Fitted (Group Medians, Linear model)")
plt.xlabel("Fitted Final Balance")
plt.ylabel("Residuals")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "h6_residuals_vs_fitted_group_medians.png"), dpi=600)
plt.close()

# === QQ Plot of residuals ===
plt.figure(figsize=(6, 4))
sm.qqplot(residuals, line='s', color="#444444", alpha=0.6, markerfacecolor='white', markeredgecolor='#444444')
plt.title("QQ Plot of Residuals (Group Medians, Linear model)")
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "h6_qqplot_residuals_group_medians.png"), dpi=600)
plt.close()

# === Shapiro-Wilk test ===
shapiro_p = shapiro(residuals).pvalue
print("\n=== Residual normality test (Shapiro-Wilk) on Group Medians ===")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("Residuals are likely NOT normally distributed (reject H0).")
else:
    print("Residuals may be considered normally distributed (fail to reject H0).")

# === Linear Regression Summary ===
print("\n=== Linear Regression Summary ===")
print(f"Intercept (alpha)      = {ols_model.params[0]:.4f}")
print(f"Slope (beta)           = {ols_model.params[1]:.4f}")
print(f"R-squared (R^2)        = {ols_model.rsquared:.4f}")
print(f"p-value of slope (beta)= {ols_model.pvalues[1]:.4f}")

# === Spearman correlation on Group Medians ===
spearman_group_median = spearmanr(group_median_df["Click Frequency"], group_median_df["Final Balance"])
print("\n=== Spearman Correlation (Group Medians) ===")
print(f"Spearman rho = {spearman_group_median.correlation:.4f}")
print(f"p-value      = {spearman_group_median.pvalue:.4f}")








######################################################################
# Hypothesis 2: Distribution and Market Activity
######################################################################

#----------------------------------------------------------------------------
# Clicking Frequency Assymetric vs Symmetric Distribution Plots
#----------------------------------------------------------------------------
import random 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import wilcoxon

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Create output directory for plots ===
graph_output_dir = "Organized Graphs"
os.makedirs(graph_output_dir, exist_ok=True)

# === Load treatment summary ===
treatments = pd.read_csv("treatments_summary.csv")

# === Classify treatments ===
def classify_row(row):
    if row["Transparency"] == "No":
        dist = row["Inventory Distribution"].strip()
        if dist == "2, 2, 2, 2, 2":
            return "Symmetric"
        elif dist == "10, 0, 0, 0, 0":
            return "Asymmetric"
    return None

treatments["Treatment Type"] = treatments.apply(classify_row, axis=1)
valid_treatments = treatments.dropna(subset=["Treatment Type"])

# === Collect clicking frequencies ===
records = []
for _, row in valid_treatments.iterrows():
    session = row["Session Code"]
    round_num = row["Round"]
    group = row["Group"]
    treatment_type = row["Treatment Type"]

    inv_dist = row['Inventory Distribution'].replace(', ', '-')
    cost = row['Cost per Second']
    transp = 0 if row['Transparency'] == "No" else 1

    filename = f"treatment_outputs/session{session}_round{round_num}_group{group}_Agents5_IniInv{inv_dist}_Cost{cost}_Transp{transp}.csv"
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(filename)
    for agent, agent_df in df.groupby("Agent"):
        total_clicks = agent_df["Requested"].sum()
        max_time = df["Time Step"].max()
        freq = total_clicks / max_time if max_time > 0 else 0
        records.append({
            "Agent": agent,
            "Group": group,
            "Session": session,
            "Round": round_num,
            "Treatment Type": treatment_type,
            "Click Frequency": freq,
            "Group ID": f"{session}_{group}"
        })

click_data = pd.DataFrame(records)

# === Identify groups that experienced BOTH treatments ===
pivot = click_data.pivot_table(index="Group ID", columns="Treatment Type", values="Click Frequency", aggfunc='count')
paired_groups = pivot.dropna(subset=["Symmetric", "Asymmetric"]).index.tolist()

# === Subset data ===
click_data_paired = click_data[click_data["Group ID"].isin(paired_groups)]
click_data_all = click_data  # All groups
all_groups = click_data["Group ID"].unique().tolist()
unpaired_groups = [g for g in all_groups if g not in paired_groups]
click_data_unpaired = click_data[click_data["Group ID"].isin(unpaired_groups)]

# === CHECK: print group numbers ===
n_both = len(paired_groups)
n_sym = click_data[click_data["Treatment Type"] == "Symmetric"]["Group ID"].nunique()
n_asym = click_data[click_data["Treatment Type"] == "Asymmetric"]["Group ID"].nunique()

print(f"Number of groups with BOTH treatments: {n_both}")
print(f"Total number of symmetric groups: {n_sym}")
print(f"Total number of asymmetric groups: {n_asym}")

# === Prepare consistent Group mapping and color palette ===
# Assign Group 1, Group 2, ... arbitrarily (does not matter which Group ID)
unique_all_groups_sorted = sorted(click_data['Group ID'].unique())
group_id_map_all = {group: f"Group {i+1}" for i, group in enumerate(unique_all_groups_sorted)}

# Apply this mapping to all dataframes
click_data['Group Simple'] = click_data['Group ID'].map(group_id_map_all)
click_data_paired['Group Simple'] = click_data_paired['Group ID'].map(group_id_map_all)
click_data_all['Group Simple'] = click_data_all['Group ID'].map(group_id_map_all)
click_data_unpaired['Group Simple'] = click_data_unpaired['Group ID'].map(group_id_map_all)

# === Color palette (12-15 distinct colors) ===
color_palette_distinct = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#98df8a",
    "#ffbb78", "#f7b6d2", "#c5b0d5"
]

group_palette = {}
for i, group in enumerate(unique_all_groups_sorted):
    color = color_palette_distinct[i % len(color_palette_distinct)]
    group_palette[group_id_map_all[group]] = color

# === PLOT 1: Paired groups only, colored ===

# Prepare group names for paired groups → Group 1..N
unique_paired_groups_sorted = sorted(click_data_paired['Group ID'].unique())
group_id_map_paired = {group: f"Group {i+1}" for i, group in enumerate(unique_paired_groups_sorted)}
click_data_paired['Group Simple Paired'] = click_data_paired['Group ID'].map(group_id_map_paired)

# Prepare color palette → consistent with global group_palette → but order for paired groups
paired_palette = {group_id_map_paired[group]: group_palette[group_id_map_all[group]] for group in unique_paired_groups_sorted}

plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_paired, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_paired,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple Paired",
    palette=paired_palette,
    dodge=False,
    jitter=True,
    alpha=0.7
)
plt.title("Click Frequency by Treatment (Paired Groups Only)")
plt.ylabel("Clicks per Second")
plt.xlabel("Treatment Type")

# Force legend order Group 1..N
legend_order = [f"Group {i+1}" for i in range(len(unique_paired_groups_sorted))]
handles, labels = plt.gca().get_legend_handles_labels()
handles_ordered = [handles[labels.index(lab)] for lab in legend_order]
plt.legend(handles_ordered, legend_order, title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_paired_groups.png"))
plt.close()


# === PLOT 2: All groups, colored ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_all, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_all,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette,
    dodge=False,
    jitter=True,
    alpha=0.7
)
plt.title("Click Frequency by Treatment (All Groups)")
plt.ylabel("Clicks per Second")
plt.xlabel("Treatment Type")
plt.legend([],[], frameon=False)  # No legend
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_all_groups.png"))
plt.close()

# === PLOT 3: UNPAIRED groups, colored ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_unpaired, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_unpaired,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette,
    dodge=False,
    jitter=True,
    alpha=0.7
)
plt.title("Click Frequency by Treatment (Unpaired Groups Only)")
plt.ylabel("Clicks per Second")
plt.xlabel("Treatment Type")
plt.legend([],[], frameon=False)  # No legend
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_unpaired_groups.png"))
plt.close()

print("All box plots saved.")





#----------------------------------------------------------------------------
# Clicking Frequency Asymmetric vs Symmetric Distribution Statistics — using group MEDIANS
#----------------------------------------------------------------------------

import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

# === Step 1: compute group medians and variances ===

# Group median click frequency
group_median = click_data.groupby(['Group ID', 'Treatment Type'])['Click Frequency'].median().reset_index()

# Group variance of click frequency
group_var = click_data.groupby(['Group ID', 'Treatment Type'])['Click Frequency'].var().reset_index()

# === Step 2: build pivot tables ===

# Pivot for medians
pivot_median = group_median.pivot(index='Group ID', columns='Treatment Type', values='Click Frequency')

# Pivot for variance
pivot_var = group_var.pivot(index='Group ID', columns='Treatment Type', values='Click Frequency')

# === Step 3: identify paired, unpaired, all groups ===

# Paired groups
paired_groups = pivot_median.dropna(subset=["Symmetric", "Asymmetric"])

# Prepare paired_var table (variance for paired groups)
paired_var = pivot_var.loc[paired_groups.index]

# All groups
all_groups_median = pivot_median
all_groups_var = pivot_var

# Unpaired groups
paired_group_ids = paired_groups.index.tolist()
all_group_ids = pivot_median.index.tolist()
unpaired_group_ids = [g for g in all_group_ids if g not in paired_group_ids]

# Subset unpaired
unpaired_median = pivot_median.loc[unpaired_group_ids]
unpaired_var = pivot_var.loc[unpaired_group_ids]

# === Step 4: compute statistics ===

results = []

# Helper function to compute mean / median string
def describe_values(series):
    mean_val = series.mean()
    median_val = series.median()
    return f"{mean_val:.3f} ({median_val:.3f})"

# --- Paired groups ---

# Median
wilcoxon_median = wilcoxon(paired_groups["Symmetric"], paired_groups["Asymmetric"])
results.append({
    "Section": "Paired groups",
    "Metric": "Median",
    "Symmetric (Mean/Median)": describe_values(paired_groups["Symmetric"]),
    "Asymmetric (Mean/Median)": describe_values(paired_groups["Asymmetric"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_median.pvalue:.4f}"
})

# Variance
wilcoxon_var = wilcoxon(paired_var["Symmetric"], paired_var["Asymmetric"])
results.append({
    "Section": "Paired groups",
    "Metric": "Variance",
    "Symmetric (Mean/Median)": describe_values(paired_var["Symmetric"]),
    "Asymmetric (Mean/Median)": describe_values(paired_var["Asymmetric"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_var.pvalue:.4f}"
})

# --- All groups ---

# Median
all_median_sym = all_groups_median["Symmetric"].dropna()
all_median_asym = all_groups_median["Asymmetric"].dropna()
mann_median_all = mannwhitneyu(all_median_sym, all_median_asym, alternative='two-sided')
results.append({
    "Section": "All groups",
    "Metric": "Median",
    "Symmetric (Mean/Median)": describe_values(all_median_sym),
    "Asymmetric (Mean/Median)": describe_values(all_median_asym),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_median_all.pvalue:.4f}"
})

# Variance
all_var_sym = all_groups_var["Symmetric"].dropna()
all_var_asym = all_groups_var["Asymmetric"].dropna()
mann_var_all = mannwhitneyu(all_var_sym, all_var_asym, alternative='two-sided')
results.append({
    "Section": "All groups",
    "Metric": "Variance",
    "Symmetric (Mean/Median)": describe_values(all_var_sym),
    "Asymmetric (Mean/Median)": describe_values(all_var_asym),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_var_all.pvalue:.4f}"
})

# --- Unpaired groups ---

# Median
unpaired_median_sym = unpaired_median["Symmetric"].dropna()
unpaired_median_asym = unpaired_median["Asymmetric"].dropna()
mann_median_unpaired = mannwhitneyu(unpaired_median_sym, unpaired_median_asym, alternative='two-sided')
results.append({
    "Section": "Unpaired groups",
    "Metric": "Median",
    "Symmetric (Mean/Median)": describe_values(unpaired_median_sym),
    "Asymmetric (Mean/Median)": describe_values(unpaired_median_asym),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_median_unpaired.pvalue:.4f}"
})

# Variance
unpaired_var_sym = unpaired_var["Symmetric"].dropna()
unpaired_var_asym = unpaired_var["Asymmetric"].dropna()
mann_var_unpaired = mannwhitneyu(unpaired_var_sym, unpaired_var_asym, alternative='two-sided')
results.append({
    "Section": "Unpaired groups",
    "Metric": "Variance",
    "Symmetric (Mean/Median)": describe_values(unpaired_var_sym),
    "Asymmetric (Mean/Median)": describe_values(unpaired_var_asym),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_var_unpaired.pvalue:.4f}"
})

# === Step 5: assemble results table ===

results_df = pd.DataFrame(results)

# Show in console
print(results_df)

# Save to CSV
results_df.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table.csv"), index=False)

print("Statistics table saved to click_frequency_statistics_table.csv.")



#-----------------------------------------------------------
# PLOTS WITH GROUP-LEVEL MEDIANS AS SQUARES
#-----------------------------------------------------------

# === Compute correct group-level medians for each plot ===

# First: define Paired groups DataFrame correctly
paired_groups_df = pivot_median.dropna(subset=["Symmetric", "Asymmetric"])

# Paired group IDs → for legend
paired_group_ids = paired_groups_df.index.tolist()

# Now compute group-level medians:

# Paired groups → keep full DataFrame
paired_groups_df_plot = paired_groups_df.copy()

# All groups → keep full DataFrame
all_groups_median_df_plot = pivot_median.copy()

# Unpaired groups
unpaired_group_ids = [g for g in pivot_median.index.tolist() if g not in paired_group_ids]
unpaired_groups_df_plot = pivot_median.loc[unpaired_group_ids]

# === 1) PLOT Paired Groups ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_paired, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_paired,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple Paired",
    palette=paired_palette,
    dodge=False,
    jitter=True,
    alpha=0.7
)

# Get actual category order
xtick_labels = [t.get_text() for t in plt.gca().get_xticklabels()]

# Add squares per group median (correct value!)
for i, treatment in enumerate(xtick_labels):
    for group_id in paired_groups_df_plot.index:
        group_simple = group_id_map_all[group_id]
        color = group_palette[group_simple]
        median_val = paired_groups_df_plot.loc[group_id, treatment]
        plt.scatter(i, median_val, marker='s', s=100, color=color, edgecolor='black', zorder=5)

plt.title("Click Frequency by Treatment (Paired Groups Only)\nSquares = Group medians")
plt.ylabel("Clicks per Second")
plt.xlabel("Treatment Type")

# Force legend order Group 1..N
legend_order = [f"Group {i+1}" for i in range(len(unique_paired_groups_sorted))]
handles, labels = plt.gca().get_legend_handles_labels()
handles_ordered = [handles[labels.index(lab)] for lab in legend_order]
plt.legend(handles_ordered, legend_order, title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_paired_groups_with_correct_groupmean.png"))
plt.close()

# === 2) PLOT All Groups ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_all, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_all,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette,
    dodge=False,
    jitter=True,
    alpha=0.7
)

# Get actual category order
xtick_labels = [t.get_text() for t in plt.gca().get_xticklabels()]

# Add squares per group median (correct value!)
for i, treatment in enumerate(xtick_labels):
    for group_id in all_groups_median_df_plot.index:
        group_simple = group_id_map_all[group_id]
        color = group_palette[group_simple]
        median_val = all_groups_median_df_plot.loc[group_id, treatment] if treatment in all_groups_median_df_plot.columns else None
        if pd.notna(median_val):
            plt.scatter(i, median_val, marker='s', s=100, color=color, edgecolor='black', zorder=5)

plt.title("Click Frequency by Treatment (All Groups)\nSquares = Group medians")
plt.ylabel("Clicks per Second")
plt.xlabel("Treatment Type")
plt.legend([],[], frameon=False)  # No legend

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_all_groups_with_correct_groupmean.png"))
plt.close()

# === 3) PLOT Unpaired Groups ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_unpaired, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_unpaired,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette,
    dodge=False,
    jitter=True,
    alpha=0.7
)

# Get actual category order
xtick_labels = [t.get_text() for t in plt.gca().get_xticklabels()]

# Add squares per group median (correct value!)
for i, treatment in enumerate(xtick_labels):
    for group_id in unpaired_groups_df_plot.index:
        group_simple = group_id_map_all[group_id]
        color = group_palette[group_simple]
        median_val = unpaired_groups_df_plot.loc[group_id, treatment] if treatment in unpaired_groups_df_plot.columns else None
        if pd.notna(median_val):
            plt.scatter(i, median_val, marker='s', s=100, color=color, edgecolor='black', zorder=5)

plt.title("Click Frequency by Treatment (Unpaired Groups)\nSquares = Group medians")
plt.ylabel("Clicks per Second")
plt.xlabel("Treatment Type")
plt.legend([],[], frameon=False)  # No legend

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_unpaired_groups_with_correct_groupmean.png"))
plt.close()

print("All box plots saved.")

# === Combined PANEL PLOTS: with median squares and full titles ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Plot 1: Paired groups
sns.boxplot(ax=axes[0], data=click_data_paired, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    ax=axes[0],
    data=click_data_paired,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple Paired",
    palette=paired_palette,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)
xtick_labels = [t.get_text() for t in axes[0].get_xticklabels()]
for i, treatment in enumerate(xtick_labels):
    for group_id in paired_groups_df_plot.index:
        group_simple = group_id_map_all[group_id]
        color = group_palette[group_simple]
        median_val = paired_groups_df_plot.loc[group_id, treatment]
        dx = random.uniform(-0.08, 0.08)  
        axes[0].scatter(i + dx, median_val, marker='s', s=80, color=color, edgecolor='black', zorder=5)

axes[0].set_title("a) Click Frequency by Treatment \n(Paired Groups)\nSquares = Group medians")
axes[0].set_xlabel("Treatment Type")
axes[0].set_ylabel("Clicks per Second")
axes[0].grid(True, linestyle="--", alpha=0.4)

# For panel a) → force legend with Group 1..N, inside plot area (upper left)
legend_order = [f"Group {i+1}" for i in range(len(unique_paired_groups_sorted))]
handles, labels = axes[0].get_legend_handles_labels()
handles_ordered = [handles[labels.index(lab)] for lab in legend_order if lab in labels]

axes[0].legend(
    handles_ordered, legend_order,
    title="Group",
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.,
    frameon=True  # or False if you want it without box
)

# Plot 2: All groups
sns.boxplot(ax=axes[1], data=click_data_all, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    ax=axes[1],
    data=click_data_all,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)
xtick_labels = [t.get_text() for t in axes[1].get_xticklabels()]
for i, treatment in enumerate(xtick_labels):
    for group_id in all_groups_median_df_plot.index:
        group_simple = group_id_map_all[group_id]
        color = group_palette[group_simple]
        median_val = all_groups_median_df_plot.loc[group_id, treatment] if treatment in all_groups_median_df_plot.columns else None
        if pd.notna(median_val):
            dx = random.uniform(-0.08, 0.08)  
            axes[1].scatter(i + dx, median_val, marker='s', s=80, color=color, edgecolor='black', zorder=5)

axes[1].set_title("b) Click Frequency by Treatment \n(All Groups)\nSquares = Group medians")
axes[1].set_xlabel("Treatment Type")
axes[1].set_ylabel("")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend([],[], frameon=False)

# Plot 3: Unpaired groups
sns.boxplot(ax=axes[2], data=click_data_unpaired, x="Treatment Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    ax=axes[2],
    data=click_data_unpaired,
    x="Treatment Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)
xtick_labels = [t.get_text() for t in axes[2].get_xticklabels()]
for i, treatment in enumerate(xtick_labels):
    for group_id in unpaired_groups_df_plot.index:
        group_simple = group_id_map_all[group_id]
        color = group_palette[group_simple]
        median_val = unpaired_groups_df_plot.loc[group_id, treatment] if treatment in unpaired_groups_df_plot.columns else None
        if pd.notna(median_val):
            dx = random.uniform(-0.08, 0.08)  
            axes[2].scatter(i + dx, median_val, marker='s', s=80, color=color, edgecolor='black', zorder=5)

axes[2].set_title("c) Click Frequency by Treatment \n(Unpaired Groups)\nSquares = Group medians")
axes[2].set_xlabel("Treatment Type")
axes[2].set_ylabel("")
axes[2].grid(True, linestyle="--", alpha=0.4)
axes[2].legend([],[], frameon=False)

# Final layout
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "panel_boxplots_click_frequency_with_medians_FULL.png"), dpi=600)
plt.close()

print("Panel box plot with group medians and full titles saved.")




############################################################
# Hypotheis 3:  Market Activity vs. Initial Inventory Distribution
############################################################

#---------------------------------------------------------------------------- 
# Clicking Frequency by Initial Inventory Distribution — All Groups
#----------------------------------------------------------------------------

# === Classify treatments for this analysis ===
def classify_inventory(row):
    dist = row["Inventory Distribution"].strip()
    if dist == "10, 0, 0, 0, 0":
        return 10
    elif dist == "5, 0, 0, 0, 0":
        return 5
    elif dist == "3, 0, 0, 0, 0":
        return 3
    elif dist == "1, 0, 0, 0, 0":
        return 1
    else:
        return None

treatments["Inventory Type"] = treatments.apply(classify_inventory, axis=1)
valid_treatments_inventory = treatments.dropna(subset=["Inventory Type"])

# === Collect clicking frequencies ===
records_inventory = []
for _, row in valid_treatments_inventory.iterrows():
    session = row["Session Code"]
    round_num = row["Round"]
    group = row["Group"]
    inventory_type = row["Inventory Type"]

    inv_dist = row['Inventory Distribution'].replace(', ', '-')
    cost = row['Cost per Second']
    transp = 0 if row['Transparency'] == "No" else 1

    filename = f"treatment_outputs/session{session}_round{round_num}_group{group}_Agents5_IniInv{inv_dist}_Cost{cost}_Transp{transp}.csv"
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(filename)
    for agent, agent_df in df.groupby("Agent"):
        total_clicks = agent_df["Requested"].sum()
        max_time = df["Time Step"].max()
        freq = total_clicks / max_time if max_time > 0 else 0
        records_inventory.append({
            "Agent": agent,
            "Group": group,
            "Session": session,
            "Round": round_num,
            "Inventory Type": inventory_type,
            "Click Frequency": freq,
            "Group ID": f"{session}_{group}"
        })

click_data_inventory = pd.DataFrame(records_inventory)

# === CHECK: print group numbers ===
group_counts_inventory = click_data_inventory.groupby('Inventory Type')['Group ID'].nunique()

print("\nNumber of groups per Inventory Type:")
for inv_type in sorted(group_counts_inventory.index, reverse=True):
    print(f"Inventory {inv_type}: {group_counts_inventory[inv_type]} groups")

# === Prepare consistent Group mapping (optional — here we will not use colors, but still nice for clean code)
unique_all_groups_sorted_inventory = sorted(click_data_inventory['Group ID'].unique())
group_id_map_all_inventory = {group: f"Group {i+1}" for i, group in enumerate(unique_all_groups_sorted_inventory)}

click_data_inventory['Group Simple'] = click_data_inventory['Group ID'].map(group_id_map_all_inventory)

# === PLOT: All groups — monochrome ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_inventory, x="Inventory Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_inventory,
    x="Inventory Type",
    y="Click Frequency",
    color="black",
    dodge=False,
    jitter=True,
    alpha=0.5
)
plt.title("Click Frequency by Initial Inventory (All Groups)")
plt.ylabel("Clicks per Second")
plt.xlabel("Initial Inventory (total units in ring)")
plt.legend([],[], frameon=False)  # No legend
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_all_groups_inventory.png"))
plt.close()

print("Inventory box plot saved.")



#---------------------------------------------------------------------------- 
# Clicking Frequency by Initial Inventory — Statistics Table (with MEDIANS)
#----------------------------------------------------------------------------

import pandas as pd
from scipy.stats import kruskal, spearmanr

# === Step 1: compute group medians and variances ===

# Group median click frequency
group_median_inventory = click_data_inventory.groupby(['Group ID', 'Inventory Type'])['Click Frequency'].median().reset_index()

# Group variance of click frequency
group_var_inventory = click_data_inventory.groupby(['Group ID', 'Inventory Type'])['Click Frequency'].var().reset_index()

# === Step 2: build pivot tables ===

# Pivot for medians
pivot_median_inventory = group_median_inventory.pivot(index='Group ID', columns='Inventory Type', values='Click Frequency')

# Pivot for variances
pivot_var_inventory = group_var_inventory.pivot(index='Group ID', columns='Inventory Type', values='Click Frequency')

# === Step 3: prepare data for tests ===

# Prepare lists of click frequency medians per Inventory Type
median_lists = []
inventory_types_sorted = sorted(click_data_inventory['Inventory Type'].unique(), reverse=True)
for inv_type in inventory_types_sorted:
    median_lists.append(group_median_inventory[group_median_inventory['Inventory Type'] == inv_type]['Click Frequency'])

# === Step 4: perform tests ===

# Kruskal-Wallis test for medians
kruskal_median_result = kruskal(*median_lists)

# Spearman correlation for median click frequency vs inventory type
median_click_per_inventory = group_median_inventory.groupby('Inventory Type')['Click Frequency'].median()
spearman_median_result = spearmanr(median_click_per_inventory.index, median_click_per_inventory.values)

# Spearman correlation for variance of click frequency vs inventory type (keep this)
var_click_per_inventory = group_var_inventory.groupby('Inventory Type')['Click Frequency'].mean()
spearman_var_result = spearmanr(var_click_per_inventory.index, var_click_per_inventory.values)



# === Step 5: assemble results table ===

# Helper function to format value nicely
def format_value(x):
    return f"{x:.3f}"

# Build table rows
row_median = {
    "Metric": "Median",
    "Inventory 10": format_value(median_click_per_inventory[10]),
    "Inventory 5": format_value(median_click_per_inventory[5]),
    "Inventory 3": format_value(median_click_per_inventory[3]),
    "Inventory 1": format_value(median_click_per_inventory[1]),
    "Kruskal-Wallis p": f"{kruskal_median_result.pvalue:.4f}",
    "Spearman rho": f"{spearman_median_result.correlation:.4f}",
    "Spearman p": f"{spearman_median_result.pvalue:.4f}"
}

row_var = {
    "Metric": "Variance",
    "Inventory 10": format_value(var_click_per_inventory[10]),
    "Inventory 5": format_value(var_click_per_inventory[5]),
    "Inventory 3": format_value(var_click_per_inventory[3]),
    "Inventory 1": format_value(var_click_per_inventory[1]),
    "Kruskal-Wallis p": "",  # No Kruskal for variance here
    "Spearman rho": f"{spearman_var_result.correlation:.4f}",
    "Spearman p": f"{spearman_var_result.pvalue:.4f}"
}

# Assemble DataFrame
results_inventory = pd.DataFrame([row_median, row_var])

# Show in console
print("\n=== Clicking Frequency Statistics Table — Inventory Type ===")
print(results_inventory)

# Save to CSV
results_inventory.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_inventory.csv"), index=False)

print("Statistics table saved to click_frequency_statistics_table_inventory.csv.")


#---------------------------------------------------------------------------- 
# Clicking Frequency vs Initial Inventory — Median + Variance Plot
#----------------------------------------------------------------------------

import matplotlib.pyplot as plt

# === Plot 1: Median Click Frequency vs Initial Inventory ===
plt.figure(figsize=(6, 4))
plt.plot(median_click_per_inventory.index, median_click_per_inventory.values, marker='o', linestyle='-', color='black')
plt.title("Median Click Frequency vs Initial Inventory")
plt.xlabel("Initial Inventory (total units in ring)")
plt.ylabel("Median Click Frequency")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "click_frequency_medians_vs_inventory.png"))
plt.close()

# === Plot 2: Variance of Click Frequency vs Initial Inventory ===
plt.figure(figsize=(6, 4))
plt.plot(var_click_per_inventory.index, var_click_per_inventory.values, marker='o', linestyle='-', color='black')
plt.title("Variance of Click Frequency vs Initial Inventory")
plt.xlabel("Initial Inventory (total units in ring)")
plt.ylabel("Variance of Click Frequency")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "click_frequency_variances_vs_inventory.png"))
plt.close()

print("Median + Variance vs Initial Inventory plots saved.")


# === Plot 1b: Median Click Frequency vs Initial Inventory (LOG-LOG) ===
plt.figure(figsize=(6, 4))
plt.plot(median_click_per_inventory.index, median_click_per_inventory.values, marker='o', linestyle='-', color='black')
plt.xscale("log")
plt.yscale("log")
plt.title("Median Click Frequency vs Initial Inventory (Log-Log)")
plt.xlabel("Initial Inventory (total units in ring) — log scale")
plt.ylabel("Median Click Frequency — log scale")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "click_frequency_medians_vs_inventory_loglog.png"))
plt.close()

# === Plot 2b: Variance of Click Frequency vs Initial Inventory (LOG-LOG) ===
plt.figure(figsize=(6, 4))
plt.plot(var_click_per_inventory.index, var_click_per_inventory.values, marker='o', linestyle='-', color='black')
plt.xscale("log")
plt.yscale("log")
plt.title("Variance of Click Frequency vs Initial Inventory (Log-Log)")
plt.xlabel("Initial Inventory (total units in ring) — log scale")
plt.ylabel("Variance of Click Frequency — log scale")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "click_frequency_variances_vs_inventory_loglog.png"))
plt.close()

print("Log-Log Median + Variance vs Initial Inventory plots saved.")




#---------------------------------------------------------------------------- 
# Clicking Frequency — Paired Sessions by Initial Inventory (MEDIANS version)
#----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, spearmanr

# === Define target inventory distributions ===
inventory_target_dists = [
    "10, 0, 0, 0, 0",
    "5, 0, 0, 0, 0",
    "3, 0, 0, 0, 0",
    "1, 0, 0, 0, 0"
]

# === Classify inventory type ===
def classify_inventory(row):
    dist = row["Inventory Distribution"].strip()
    if dist == "10, 0, 0, 0, 0":
        return 10
    elif dist == "5, 0, 0, 0, 0":
        return 5
    elif dist == "3, 0, 0, 0, 0":
        return 3
    elif dist == "1, 0, 0, 0, 0":
        return 1
    else:
        return None

treatments["Inventory Type"] = treatments.apply(classify_inventory, axis=1)

# === Define sessions ===
session_1_code = "n30xby42"
session_2_code = "5lnljat1"

# === Filter treatments ===
def filter_session(session_code):
    return treatments[(treatments["Session Code"] == session_code) & (treatments["Inventory Type"].notna())]

valid_treatments_session1 = filter_session(session_1_code)
valid_treatments_session2 = filter_session(session_2_code)

# === Function to collect click data ===
def collect_click_data(valid_treatments_session):
    records = []
    for _, row in valid_treatments_session.iterrows():
        session = row["Session Code"]
        round_num = row["Round"]
        group = row["Group"]
        inventory_type = row["Inventory Type"]

        inv_dist = row['Inventory Distribution'].replace(', ', '-')
        cost = row['Cost per Second']
        transp = 0 if row['Transparency'] == "No" else 1

        filename = f"treatment_outputs/session{session}_round{round_num}_group{group}_Agents5_IniInv{inv_dist}_Cost{cost}_Transp{transp}.csv"
        if not os.path.exists(filename):
            print(f"Missing file: {filename}")
            continue

        df = pd.read_csv(filename)
        for agent, agent_df in df.groupby("Agent"):
            total_clicks = agent_df["Requested"].sum()
            max_time = df["Time Step"].max()
            freq = total_clicks / max_time if max_time > 0 else 0
            records.append({
                "Agent": agent,
                "Group": group,
                "Session": session,
                "Round": round_num,
                "Inventory Type": inventory_type,
                "Click Frequency": freq,
                "Group ID": f"{session}_{group}"
            })

    click_data_df = pd.DataFrame(records)
    return click_data_df

# === Collect click data ===
click_data_paired_session_1 = collect_click_data(valid_treatments_session1)
click_data_paired_session_2 = collect_click_data(valid_treatments_session2)

# === Prepare group color mapping ===
def prepare_group_palette(click_data_df):
    unique_groups_sorted = sorted(click_data_df['Group ID'].unique())
    group_id_map = {group: f"Group {i+1}" for i, group in enumerate(unique_groups_sorted)}
    click_data_df['Group Simple'] = click_data_df['Group ID'].map(group_id_map)

    # Color palette
    color_palette_distinct = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    group_palette = {}
    for i, group in enumerate(unique_groups_sorted):
        color = color_palette_distinct[i % len(color_palette_distinct)]
        group_palette[group_id_map[group]] = color

    return group_id_map, group_palette

# === Prepare group palettes ===
group_id_map_s1, group_palette_s1 = prepare_group_palette(click_data_paired_session_1)
group_id_map_s2, group_palette_s2 = prepare_group_palette(click_data_paired_session_2)

# === Function to plot paired session (with median squares!) ===
def plot_paired_session(click_data_df, group_palette, group_id_map, session_name, filename_suffix):
    import numpy as np

    # --- Compute group-level median for squares ---
    group_median = click_data_df.groupby(['Group ID', 'Inventory Type'])['Click Frequency'].median().reset_index()
    pivot_median = group_median.pivot(index='Group ID', columns='Inventory Type', values='Click Frequency')
    group_level_medians = pivot_median.median()

    # --- Define order and mapping ---
    order = sorted(click_data_df["Inventory Type"].unique(), reverse=True)
    x_mapping = {inv: i for i, inv in enumerate(order)}

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=click_data_df,
        x="Inventory Type",
        y="Click Frequency",
        order=order,
        color="white",
        fliersize=0
    )
    sns.stripplot(
        data=click_data_df,
        x="Inventory Type",
        y="Click Frequency",
        order=order,
        hue="Group Simple",
        palette=group_palette,
        dodge=False,
        jitter=True,
        alpha=0.7
    )

    # --- Add median squares ---
    for _, row in group_median.iterrows():
        inv_type = row["Inventory Type"]
        x = x_mapping[inv_type]
        y = row["Click Frequency"]
        group_id = row["Group ID"]
        group_simple = group_id_map[group_id]
        color = group_palette[group_simple]

        plt.scatter(
            x, y,
            marker='s',
            s=100,
            color=color,
            edgecolor='black',
            linewidth=0.8,
            zorder=5
        )

    # --- Title, labels ---
    plt.title(f"Click Frequency by Initial Inventory — {session_name}\nSquares = Group medians")
    plt.ylabel("Clicks per Second")
    plt.xlabel("Initial Inventory (total units in ring)")

    # --- Force legend order ---
    legend_order = [f"Group {i+1}" for i in range(len(group_id_map))]
    handles, labels = plt.gca().get_legend_handles_labels()
    handles_ordered = [handles[labels.index(lab)] for lab in legend_order]
    plt.legend(handles_ordered, legend_order, title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')

    # --- Final ---
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(graph_output_dir, f"boxplot_click_frequency_paired_session{filename_suffix}_inventory.png"))
    plt.close()



    print(f"Paired Session {filename_suffix} boxplot saved.")

    return pivot_median  # Return MEDIAN pivot!

# === Plot paired session 1 ===
pivot_median_s1 = plot_paired_session(click_data_paired_session_1, group_palette_s1, group_id_map_s1, "Paired Session 1", "1")

# === Plot paired session 2 ===
pivot_median_s2 = plot_paired_session(click_data_paired_session_2, group_palette_s2, group_id_map_s2, "Paired Session 2", "2")

#---------------------------------------------------------------------------- 
# Statistics table for Paired Sessions (MEDIANS version)
#----------------------------------------------------------------------------

# Helper function to compute statistics for a pivot table (MEDIANS)
def compute_stats_table(pivot_median_df, session_label):
    results = []

    # Prepare lists for Kruskal-Wallis
    median_lists = []
    inventory_types_sorted = sorted(pivot_median_df.columns, reverse=True)
    for inv_type in inventory_types_sorted:
        median_lists.append(pivot_median_df[inv_type].dropna())

    # Kruskal-Wallis
    kruskal_median_result = kruskal(*median_lists)

    # Spearman on medians
    median_click_per_inventory = pivot_median_df.median()
    spearman_median_result = spearmanr(median_click_per_inventory.index, median_click_per_inventory.values)

    # Variance of group medians
    var_click_per_inventory = pivot_median_df.var()
    spearman_var_result = spearmanr(var_click_per_inventory.index, var_click_per_inventory.values)

    # Format function
    def format_value(x):
        return f"{x:.3f}"

    # Row Median
    row_median = {
        "Section": session_label,
        "Metric": "Median",
        "Inventory 10": format_value(median_click_per_inventory.get(10, float('nan'))),
        "Inventory 5": format_value(median_click_per_inventory.get(5, float('nan'))),
        "Inventory 3": format_value(median_click_per_inventory.get(3, float('nan'))),
        "Inventory 1": format_value(median_click_per_inventory.get(1, float('nan'))),
        "Kruskal-Wallis p": f"{kruskal_median_result.pvalue:.4f}",
        "Spearman rho": f"{spearman_median_result.correlation:.4f}",
        "Spearman p": f"{spearman_median_result.pvalue:.4f}"
    }

    # Row Variance
    row_var = {
        "Section": session_label,
        "Metric": "Variance",
        "Inventory 10": format_value(var_click_per_inventory.get(10, float('nan'))),
        "Inventory 5": format_value(var_click_per_inventory.get(5, float('nan'))),
        "Inventory 3": format_value(var_click_per_inventory.get(3, float('nan'))),
        "Inventory 1": format_value(var_click_per_inventory.get(1, float('nan'))),
        "Kruskal-Wallis p": "",
        "Spearman rho": f"{spearman_var_result.correlation:.4f}",
        "Spearman p": f"{spearman_var_result.pvalue:.4f}"
    }

    results.append(row_median)
    results.append(row_var)

    return results

# === Compute stats table ===
results_paired_sessions = []

results_paired_sessions += compute_stats_table(pivot_median_s1, "Paired Session 1")
results_paired_sessions += compute_stats_table(pivot_median_s2, "Paired Session 2")

# Assemble DataFrame
results_df_paired_sessions = pd.DataFrame(results_paired_sessions)

# Show in console
print("\n=== Clicking Frequency Statistics Table — Paired Sessions ===")
print(results_df_paired_sessions)

# Save to CSV
results_df_paired_sessions.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_paired_inventory_sessions.csv"), index=False)

print("Statistics table saved to click_frequency_statistics_table_paired_inventory_sessions.csv.")


#---------------------------------------------------------------------------- 
# === 3-PANEL PLOT: All Groups + Paired Session 1 + Paired Session 2 (with median squares and updated style) ===
#---------------------------------------------------------------------------- 

# Panel b) — define new group names and colors
group_id_map_s1_custom = {}
group_palette_s1_custom = {}

custom_group_names_s1 = [
    "Group 1.1", "Group 1.2", "Group 1.3"
]

color_palette_distinct_b = [
    "#e41a1c", "#377eb8", "#4daf4a"
]

unique_groups_sorted_s1 = sorted(click_data_paired_session_1['Group ID'].unique())
for i, group in enumerate(unique_groups_sorted_s1):
    group_id_map_s1_custom[group] = custom_group_names_s1[i]
    group_palette_s1_custom[custom_group_names_s1[i]] = color_palette_distinct_b[i]

click_data_paired_session_1['Group Simple Custom'] = click_data_paired_session_1['Group ID'].map(group_id_map_s1_custom)

# Panel c) — define correct group names and colors
group_id_map_s2_custom = {}
group_palette_s2_custom = {}

custom_group_names_s2 = [
    "Group 2.1", "Group 2.2", "Group 2.3"
]

color_palette_distinct_c = [
    "#984ea3", "#ff7f00", "#ffff33"
]

unique_groups_sorted_s2 = sorted(click_data_paired_session_2['Group ID'].unique())
for i, group in enumerate(unique_groups_sorted_s2):
    group_id_map_s2_custom[group] = custom_group_names_s2[i]
    group_palette_s2_custom[custom_group_names_s2[i]] = color_palette_distinct_c[i]

click_data_paired_session_2['Group Simple Custom'] = click_data_paired_session_2['Group ID'].map(group_id_map_s2_custom)

# Now create the panel plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

order_inventory = sorted(click_data_inventory["Inventory Type"].unique(), reverse=True)

# Jitter strength for the square positions
jitter_strength = 0.08  # Tune this between 0.05 - 0.1

# Now create the panel plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

order_inventory = sorted(click_data_inventory["Inventory Type"].unique(), reverse=True)

# Panel a) All Groups
sns.boxplot(ax=axes[0], data=click_data_inventory, x="Inventory Type", y="Click Frequency", order=order_inventory, color="white", fliersize=0)
sns.stripplot(
    ax=axes[0],
    data=click_data_inventory,
    x="Inventory Type",
    y="Click Frequency",
    order=order_inventory,
    color="black",
    dodge=False,
    jitter=0.1,
    alpha=0.5
)
axes[0].set_title("a) Click Frequency by Total Inventory\n(All Groups)\nSquares = Group medians")
axes[0].set_xlabel("Total Inventory")
axes[0].set_ylabel("Clicks per Second")
axes[0].grid(True, linestyle="--", alpha=0.4)

for _, row in group_median_inventory.iterrows():
    inv_type = row["Inventory Type"]
    x = order_inventory.index(inv_type)
    x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength)
    y = row["Click Frequency"]
    axes[0].scatter(
        x_jittered, y,
        marker='s',
        s=80,
        facecolors='none',
        edgecolors='black',
        linewidth=1.2,
        zorder=5
    )

# Panel b) Paired Session 1
sns.boxplot(ax=axes[1], data=click_data_paired_session_1, x="Inventory Type", y="Click Frequency", order=order_inventory, color="white", fliersize=0)
sns.stripplot(
    ax=axes[1],
    data=click_data_paired_session_1,
    x="Inventory Type",
    y="Click Frequency",
    order=order_inventory,
    hue="Group Simple Custom",
    palette=group_palette_s1_custom,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)
axes[1].set_title("b) Click Frequency by Total Inventory\n(Paired Session 1)\nSquares = Group medians")
axes[1].set_xlabel("Total Inventory")
axes[1].set_ylabel("")
axes[1].grid(True, linestyle="--", alpha=0.4)

for _, row in pivot_median_s1.reset_index().melt(id_vars="Group ID", var_name="Inventory Type", value_name="Click Frequency").dropna().iterrows():
    inv_type = row["Inventory Type"]
    x = order_inventory.index(inv_type)
    x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength)
    y = row["Click Frequency"]
    group_id = row["Group ID"]
    group_simple_custom = group_id_map_s1_custom[group_id]
    color = group_palette_s1_custom[group_simple_custom]
    axes[1].scatter(
        x_jittered, y,
        marker='s',
        s=80,
        color=color,
        edgecolor='black',
        zorder=5
    )

# Legend inside panel b)
legend_order_b = custom_group_names_s1
handles_b, labels_b = axes[1].get_legend_handles_labels()
handles_ordered_b = [handles_b[labels_b.index(lab)] for lab in legend_order_b if lab in labels_b]

axes[1].legend(
    handles_ordered_b, legend_order_b,
    title="Group",
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.,
    frameon=True
)

# Panel c) Paired Session 2
sns.boxplot(ax=axes[2], data=click_data_paired_session_2, x="Inventory Type", y="Click Frequency", order=order_inventory, color="white", fliersize=0)
sns.stripplot(
    ax=axes[2],
    data=click_data_paired_session_2,
    x="Inventory Type",
    y="Click Frequency",
    order=order_inventory,
    hue="Group Simple Custom",
    palette=group_palette_s2_custom,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)
axes[2].set_title("c) Click Frequency by Total Inventory\n(Paired Session 2)\nSquares = Group medians")
axes[2].set_xlabel("Total Inventory")
axes[2].set_ylabel("")
axes[2].grid(True, linestyle="--", alpha=0.4)

for _, row in pivot_median_s2.reset_index().melt(id_vars="Group ID", var_name="Inventory Type", value_name="Click Frequency").dropna().iterrows():
    inv_type = row["Inventory Type"]
    x = order_inventory.index(inv_type)
    x_jittered = x + np.random.uniform(-jitter_strength, jitter_strength)
    y = row["Click Frequency"]
    group_id = row["Group ID"]
    group_simple_custom = group_id_map_s2_custom[group_id]
    color = group_palette_s2_custom[group_simple_custom]
    axes[2].scatter(
        x_jittered, y,
        marker='s',
        s=80,
        color=color,
        edgecolor='black',
        zorder=5
    )

# Legend inside panel c)
legend_order_c = custom_group_names_s2
handles_c, labels_c = axes[2].get_legend_handles_labels()
handles_ordered_c = [handles_c[labels_c.index(lab)] for lab in legend_order_c if lab in labels_c]

axes[2].legend(
    handles_ordered_c, legend_order_c,
    title="Group",
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.,
    frameon=True
)

# Final layout and save
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "panel_boxplots_click_frequency_inventory_paired_sessions_FINAL.png"), dpi = 600)
plt.close()

print("FINAL 3-panel Inventory boxplot (All Groups + Paired S1 + Paired S2) saved.")




#---------------------------------------------------------------------------- 
# FINAL 4-PANEL PLOT: Median + Standard Deviation vs Initial Inventory (linear + log-log)
#----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Compute Standard Deviation from Variance ===
std_click_per_inventory = np.sqrt(var_click_per_inventory)

# === Plotting setup ===
grey_color = "#444444"
marker_style = 'o'

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel a) Median - linear
axes[0, 0].plot(median_click_per_inventory.index, median_click_per_inventory.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[0, 0].set_title("a) Median Click Frequency\nvs. Initial Inventory")
axes[0, 0].set_xlabel("Total Inventory")
axes[0, 0].set_ylabel("Median Click Frequency")
axes[0, 0].grid(True)

# Panel b) StdDev - linear
axes[0, 1].plot(std_click_per_inventory.index, std_click_per_inventory.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[0, 1].set_title("b) StdDev of Click Frequency\nvs. Initial Inventory")
axes[0, 1].set_xlabel("Total Inventory")
axes[0, 1].set_ylabel("Standard Deviation of Click Frequency")
axes[0, 1].grid(True)

# Panel c) Median - log-log
axes[1, 0].plot(median_click_per_inventory.index, median_click_per_inventory.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[1, 0].set_xscale("log")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("c) Median Click Frequency\nvs. Initial Inventory (log-log)")
axes[1, 0].set_xlabel("Total Inventory (log)")
axes[1, 0].set_ylabel("Median Click Frequency (log)")
axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Panel d) StdDev - log-log
axes[1, 1].plot(std_click_per_inventory.index, std_click_per_inventory.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("d) StdDev of Click Frequency\nvs. Initial Inventory (log-log)")
axes[1, 1].set_xlabel("Total Inventory (log)")
axes[1, 1].set_ylabel("Standard Deviation of Click Frequency (log)")
axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

# === Final layout and save ===
plt.tight_layout()

output_path = os.path.join(graph_output_dir, "FluxClickFrequency_MedianStdDev_4Panel_ClickSpace.png")
plt.savefig(output_path, dpi=600)
plt.close()

print(f"Saved plot to: {output_path}")


#-----------------------------------------------------------------------
# === Levene test for All Groups ===

import pandas as pd
from scipy.stats import levene

# Load your existing CSV
results_inventory = pd.read_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_inventory.csv"))

# Prepare lists for Levene (agent-level data)
inventory_types_sorted = sorted(click_data_inventory['Inventory Type'].unique(), reverse=True)
var_lists = []
for inv_type in inventory_types_sorted:
    values = click_data_inventory[click_data_inventory['Inventory Type'] == inv_type]['Click Frequency']
    var_lists.append(values)

# Perform Levene test
levene_result = levene(*var_lists)

# Build row
row_levene = {
    "Metric": "Levene Test (Variance equality)",
    "Inventory 10": "",
    "Inventory 5": "",
    "Inventory 3": "",
    "Inventory 1": "",
    "Kruskal-Wallis p": "",
    "Spearman rho": "",
    "Spearman p": "",
    "Levene W": f"{levene_result.statistic:.4f}",
    "Levene p": f"{levene_result.pvalue:.4f}"
}

# If Levene columns don't exist yet, add them
for col in ["Levene W", "Levene p"]:
    if col not in results_inventory.columns:
        results_inventory[col] = ""

# Append row
results_inventory = pd.concat([results_inventory, pd.DataFrame([row_levene])], ignore_index=True)

# Save back to CSV
results_inventory.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_inventory.csv"), index=False)

print("Levene test appended to All Groups CSV.")


# === Levene test for Paired Session 1 ===

results_paired_s1 = pd.read_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_paired_inventory_sessions.csv"))


# Prepare lists
inventory_types_sorted = sorted(pivot_median_s1.columns, reverse=True)
var_lists = []
for inv_type in inventory_types_sorted:
    values = pivot_median_s1[inv_type].dropna()
    var_lists.append(values)

# Perform Levene
levene_result = levene(*var_lists)

# Build row
row_levene = {
    "Section": "Paired Session 1",
    "Metric": "Levene Test (Variance equality)",
    "Inventory 10": "",
    "Inventory 5": "",
    "Inventory 3": "",
    "Inventory 1": "",
    "Kruskal-Wallis p": "",
    "Spearman rho": "",
    "Spearman p": "",
    "Levene W": f"{levene_result.statistic:.4f}",
    "Levene p": f"{levene_result.pvalue:.4f}"
}

# If Levene columns don't exist yet, add them
for col in ["Levene W", "Levene p"]:
    if col not in results_paired_s1.columns:
        results_paired_s1[col] = ""

# Append row
results_paired_s1 = pd.concat([results_paired_s1, pd.DataFrame([row_levene])], ignore_index=True)

# Save back
results_paired_s1.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_paired_inventory_sessions.csv"), index=False)

print("Levene test appended to Paired Sessions CSV.")

# === Levene test for Paired Session 2 ===

results_paired_s2 = pd.read_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_paired_inventory_sessions.csv"))

# Prepare lists from pivot_median_s2
inventory_types_sorted = sorted(pivot_median_s2.columns, reverse=True)
var_lists = []
for inv_type in inventory_types_sorted:
    values = pivot_median_s2[inv_type].dropna()
    var_lists.append(values)

# Perform Levene
levene_result = levene(*var_lists)

# Build row
row_levene = {
    "Section": "Paired Session 2",
    "Metric": "Levene Test (Variance equality)",
    "Inventory 10": "",
    "Inventory 5": "",
    "Inventory 3": "",
    "Inventory 1": "",
    "Kruskal-Wallis p": "",
    "Spearman rho": "",
    "Spearman p": "",
    "Levene W": f"{levene_result.statistic:.4f}",
    "Levene p": f"{levene_result.pvalue:.4f}"
}

# If Levene columns don't exist yet, add them
for col in ["Levene W", "Levene p"]:
    if col not in results_paired_s2.columns:
        results_paired_s2[col] = ""

# Append row
results_paired_s2 = pd.concat([results_paired_s2, pd.DataFrame([row_levene])], ignore_index=True)

# Save back
results_paired_s2.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_paired_inventory_sessions.csv"), index=False)

print("Levene test appended to Paired Session 2 CSV.")







############################################################
# Hypothesis 4: TRANSPARENCY vs NON-TRANSPARENCY in market activity
############################################################

#---------------------------------------------------------------------------- 
# Clicking Frequency Transparent vs Non-Transparent Distribution Plots
#----------------------------------------------------------------------------

# === Classify treatments for this analysis ===
def classify_transparency(row):
    dist = row["Inventory Distribution"].strip()
    if dist == "2, 2, 2, 2, 2":
        if row["Transparency"] == "No":
            return "Non-Transparent"
        elif row["Transparency"] == "Yes":
            return "Transparent"
    return None  # All other cases are ignored


treatments["Transparency Type"] = treatments.apply(classify_transparency, axis=1)
valid_treatments_transp = treatments.dropna(subset=["Transparency Type"])

# === Collect clicking frequencies ===
records_transp = []
for _, row in valid_treatments_transp.iterrows():
    session = row["Session Code"]
    round_num = row["Round"]
    group = row["Group"]
    transparency_type = row["Transparency Type"]

    inv_dist = row['Inventory Distribution'].replace(', ', '-')
    cost = row['Cost per Second']
    transp = 0 if row['Transparency'] == "No" else 1

    filename = f"treatment_outputs/session{session}_round{round_num}_group{group}_Agents5_IniInv{inv_dist}_Cost{cost}_Transp{transp}.csv"
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(filename)
    for agent, agent_df in df.groupby("Agent"):
        total_clicks = agent_df["Requested"].sum()
        max_time = df["Time Step"].max()
        freq = total_clicks / max_time if max_time > 0 else 0
        records_transp.append({
            "Agent": agent,
            "Group": group,
            "Session": session,
            "Round": round_num,
            "Transparency Type": transparency_type,
            "Click Frequency": freq,
            "Group ID": f"{session}_{group}"
        })

click_data_transp = pd.DataFrame(records_transp)

# === Identify groups that experienced BOTH transparency levels ===
pivot_transp = click_data_transp.pivot_table(index="Group ID", columns="Transparency Type", values="Click Frequency", aggfunc='count')
paired_groups_transp = pivot_transp.dropna(subset=["Transparent", "Non-Transparent"]).index.tolist()

# === Subset data ===
click_data_paired_transp = click_data_transp[click_data_transp["Group ID"].isin(paired_groups_transp)]
click_data_all_transp = click_data_transp  # All groups
all_groups_transp = click_data_transp["Group ID"].unique().tolist()
unpaired_groups_transp = [g for g in all_groups_transp if g not in paired_groups_transp]
click_data_unpaired_transp = click_data_transp[click_data_transp["Group ID"].isin(unpaired_groups_transp)]

# === CHECK: print group numbers ===
n_both_transp = len(paired_groups_transp)
n_transparent = click_data_transp[click_data_transp["Transparency Type"] == "Transparent"]["Group ID"].nunique()
n_nontransparent = click_data_transp[click_data_transp["Transparency Type"] == "Non-Transparent"]["Group ID"].nunique()

print(f"Number of groups with BOTH transparency levels: {n_both_transp}")
print(f"Total number of Transparent groups: {n_transparent}")
print(f"Total number of Non-Transparent groups: {n_nontransparent}")

# === Prepare consistent Group mapping and color palette ===
# Assign Group 1, Group 2, ... arbitrarily (does not matter which Group ID)
unique_all_groups_sorted_transp = sorted(click_data_transp['Group ID'].unique())
group_id_map_all_transp = {group: f"Group {i+1}" for i, group in enumerate(unique_all_groups_sorted_transp)}

# Apply this mapping to all dataframes
click_data_transp['Group Simple'] = click_data_transp['Group ID'].map(group_id_map_all_transp)
click_data_paired_transp['Group Simple'] = click_data_paired_transp['Group ID'].map(group_id_map_all_transp)
click_data_all_transp['Group Simple'] = click_data_all_transp['Group ID'].map(group_id_map_all_transp)
click_data_unpaired_transp['Group Simple'] = click_data_unpaired_transp['Group ID'].map(group_id_map_all_transp)

# === Color palette (reuse same as before) ===
group_palette_transp = {}
for i, group in enumerate(unique_all_groups_sorted_transp):
    color = color_palette_distinct[i % len(color_palette_distinct)]
    group_palette_transp[group_id_map_all_transp[group]] = color

# === PLOT 1: Paired groups only, colored ===

# Prepare group names for paired groups → Group 1..N
unique_paired_groups_sorted_transp = sorted(click_data_paired_transp['Group ID'].unique())
group_id_map_paired_transp = {group: f"Group {i+1}" for i, group in enumerate(unique_paired_groups_sorted_transp)}
click_data_paired_transp['Group Simple Paired'] = click_data_paired_transp['Group ID'].map(group_id_map_paired_transp)

# Prepare color palette → consistent with global group_palette → but order for paired groups
paired_palette_transp = {group_id_map_paired_transp[group]: group_palette_transp[group_id_map_all_transp[group]] for group in unique_paired_groups_sorted_transp}

plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_paired_transp, x="Transparency Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_paired_transp,
    x="Transparency Type",
    y="Click Frequency",
    hue="Group Simple Paired",
    palette=paired_palette_transp,
    dodge=False,
    jitter=True,
    alpha=0.7
)
plt.title("Click Frequency by Transparency (Paired Groups Only)")
plt.ylabel("Clicks per Second")
plt.xlabel("Transparency Type")

# Force legend order Group 1..N
legend_order_transp = [f"Group {i+1}" for i in range(len(unique_paired_groups_sorted_transp))]
handles, labels = plt.gca().get_legend_handles_labels()
handles_ordered = [handles[labels.index(lab)] for lab in legend_order_transp]
plt.legend(handles_ordered, legend_order_transp, title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_paired_groups_transparency.png"))
plt.close()

# === PLOT 2: All groups, colored ===
plt.figure(figsize=(8, 6))
sns.boxplot(data=click_data_all_transp, x="Transparency Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    data=click_data_all_transp,
    x="Transparency Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette_transp,
    dodge=False,
    jitter=True,
    alpha=0.7
)
plt.title("Click Frequency by Transparency (All Groups)")
plt.ylabel("Clicks per Second")
plt.xlabel("Transparency Type")
plt.legend([],[], frameon=False)  # No legend
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_all_groups_transparency.png"))
plt.close()



print("All Transparency box plots saved.")




#---------------------------------------------------------------------------- 
# Clicking Frequency Transparent vs Non-Transparent Distribution Statistics
#----------------------------------------------------------------------------

import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

# === Step 1: compute group **medians** and variances ===

# Group median click frequency
group_median_transp = click_data_transp.groupby(['Group ID', 'Transparency Type'])['Click Frequency'].median().reset_index()

# Group variance of click frequency
group_var_transp = click_data_transp.groupby(['Group ID', 'Transparency Type'])['Click Frequency'].var().reset_index()

# === Step 2: build pivot tables ===

# Pivot for medians
pivot_median_transp = group_median_transp.pivot(index='Group ID', columns='Transparency Type', values='Click Frequency')

# Pivot for variance
pivot_var_transp = group_var_transp.pivot(index='Group ID', columns='Transparency Type', values='Click Frequency')

# === Step 3: identify paired, all groups ===

# Paired groups
paired_groups_transp_df = pivot_median_transp.dropna(subset=["Transparent", "Non-Transparent"])

# Prepare paired_var table (variance for paired groups)
paired_var_transp = pivot_var_transp.loc[paired_groups_transp_df.index]

# All groups
all_groups_median_transp = pivot_median_transp
all_groups_var_transp = pivot_var_transp

# === Step 4: compute statistics ===

results_transp = []

# Helper function to compute mean / median string
def describe_values(series):
    mean_val = series.mean()
    median_val = series.median()
    return f"{mean_val:.3f} ({median_val:.3f})"

# --- Paired groups ---

# Median of group medians
wilcoxon_median_transp = wilcoxon(paired_groups_transp_df["Transparent"], paired_groups_transp_df["Non-Transparent"])
results_transp.append({
    "Section": "Paired groups",
    "Metric": "Median",
    "Transparent (Mean/Median)": describe_values(paired_groups_transp_df["Transparent"]),
    "Non-Transparent (Mean/Median)": describe_values(paired_groups_transp_df["Non-Transparent"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_median_transp.pvalue:.4f}"
})

# Variance
wilcoxon_var_transp = wilcoxon(paired_var_transp["Transparent"], paired_var_transp["Non-Transparent"])
results_transp.append({
    "Section": "Paired groups",
    "Metric": "Variance",
    "Transparent (Mean/Median)": describe_values(paired_var_transp["Transparent"]),
    "Non-Transparent (Mean/Median)": describe_values(paired_var_transp["Non-Transparent"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_var_transp.pvalue:.4f}"
})

# --- All groups ---

# Median of group medians
all_median_transparent = all_groups_median_transp["Transparent"].dropna()
all_median_nontransparent = all_groups_median_transp["Non-Transparent"].dropna()
mann_median_all_transp = mannwhitneyu(all_median_transparent, all_median_nontransparent, alternative='two-sided')
results_transp.append({
    "Section": "All groups",
    "Metric": "Median",
    "Transparent (Mean/Median)": describe_values(all_median_transparent),
    "Non-Transparent (Mean/Median)": describe_values(all_median_nontransparent),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_median_all_transp.pvalue:.4f}"
})

# Variance
all_var_transparent = all_groups_var_transp["Transparent"].dropna()
all_var_nontransparent = all_groups_var_transp["Non-Transparent"].dropna()
mann_var_all_transp = mannwhitneyu(all_var_transparent, all_var_nontransparent, alternative='two-sided')
results_transp.append({
    "Section": "All groups",
    "Metric": "Variance",
    "Transparent (Mean/Median)": describe_values(all_var_transparent),
    "Non-Transparent (Mean/Median)": describe_values(all_var_nontransparent),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_var_all_transp.pvalue:.4f}"
})

# === Step 5: assemble results table ===

results_df_transp = pd.DataFrame(results_transp)

# Show in console
print(results_df_transp)

# Save to CSV
results_df_transp.to_csv(os.path.join(graph_output_dir, "click_frequency_statistics_table_transparency.csv"), index=False)

print("Statistics table saved to click_frequency_statistics_table_transparency.csv.")




#-----------------------------------------------------------
# PLOTS WITH GROUP-LEVEL MEDIANS AS SQUARES (Transparency)
#-----------------------------------------------------------


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1) compute per‐group median click frequency for each Transparency Type ---
group_median_transp = (
    click_data_transp
    .groupby(['Group ID','Transparency Type'])['Click Frequency']
    .median()
    .reset_index()
)

# map back to your simple Group names
group_median_transp['Group Simple All'] = group_median_transp['Group ID'].map(group_id_map_all_transp)
group_median_transp['Group Simple Paired'] = group_median_transp['Group ID'].map(group_id_map_paired_transp)

# --- 2) PLOT 1: Paired groups only, with coloured squares ---
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=click_data_paired_transp,
    x="Transparency Type", 
    y="Click Frequency",
    color="white", 
    fliersize=0
)
sns.stripplot(
    data=click_data_paired_transp,
    x="Transparency Type",
    y="Click Frequency",
    hue="Group Simple Paired",
    palette=paired_palette_transp,
    dodge=False,
    jitter=True,
    alpha=0.7
)

# overlay the medians as squares WITH JITTER
x_mapping = {"Transparent": 0, "Non-Transparent": 1}

for _, row in group_median_transp[group_median_transp['Group ID'].isin(unique_paired_groups_sorted_transp)].iterrows():
    x_val_nominal = x_mapping[row['Transparency Type']]
    jitter = np.random.uniform(-0.05, 0.05)  # small jitter
    x_val = x_val_nominal + jitter
    col   = paired_palette_transp[row['Group Simple Paired']]
    plt.scatter(
        x_val,
        row['Click Frequency'],
        marker='s',
        s=100,
        edgecolor='black',
        color=col,
        zorder=4,
        label='_nolegend_'
    )

plt.title("Click Frequency by Transparency (Paired Groups Only)\nSquares = Group medians")
plt.ylabel("Clicks per Second")
plt.xlabel("Transparency Type")

# re‐order legend to Group 1…N
legend_order = [f"Group {i+1}" for i in range(len(unique_paired_groups_sorted_transp))]
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[labels.index(l)] for l in legend_order]
plt.legend(handles, legend_order, title="Group", bbox_to_anchor=(1.05,1), loc='upper left')

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_paired_groups_transparency.png"))
plt.close()

# --- 3) PLOT 2: All groups, with coloured squares ---
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=click_data_all_transp,
    x="Transparency Type",
    y="Click Frequency",
    color="white",
    fliersize=0
)
sns.stripplot(
    data=click_data_all_transp,
    x="Transparency Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette_transp,
    dodge=False,
    jitter=True,
    alpha=0.7
)

# overlay medians WITH JITTER
for _, row in group_median_transp.iterrows():
    x_val_nominal = x_mapping[row['Transparency Type']]
    jitter = np.random.uniform(-0.05, 0.05)
    x_val = x_val_nominal + jitter
    col   = group_palette_transp[row['Group Simple All']]
    plt.scatter(
        x_val,
        row['Click Frequency'],
        marker='s',
        s=100,
        edgecolor='black',
        color=col,
        zorder=4,
        label='_nolegend_'
    )

plt.title("Click Frequency by Transparency (All Groups)\nSquares = Group medians")
plt.ylabel("Clicks per Second")
plt.xlabel("Transparency Type")

# remove legend
plt.legend([],[], frameon=False)

plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "boxplot_click_frequency_all_groups_transparency.png"))
plt.close()

# === 2-PANEL PLOTS (Transparency): with median squares and full titles ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# --- 1) Panel a) Paired groups ---
sns.boxplot(ax=axes[0], data=click_data_paired_transp, x="Transparency Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    ax=axes[0],
    data=click_data_paired_transp,
    x="Transparency Type",
    y="Click Frequency",
    hue="Group Simple Paired",
    palette=paired_palette_transp,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)

xtick_labels = [t.get_text() for t in axes[0].get_xticklabels()]
for i, treatment in enumerate(xtick_labels):
    for group_id in unique_paired_groups_sorted_transp:
        group_simple = group_id_map_all_transp[group_id]
        color = group_palette_transp[group_simple]
        median_val = pivot_median_transp.loc[group_id, treatment]
        jitter = np.random.uniform(-0.05, 0.05)
        x_val = i + jitter
        axes[0].scatter(x_val, median_val, marker='s', s=80, color=color, edgecolor='black', zorder=5)

axes[0].set_title("a) Click Frequency by Treatment (Paired Groups)\nSquares = Group medians")
axes[0].set_xlabel("Treatment Type")
axes[0].set_ylabel("Clicks per Second")
axes[0].grid(True, linestyle="--", alpha=0.4)

# Legend inside panel a)
legend_order_transp = [f"Group {i+1}" for i in range(len(unique_paired_groups_sorted_transp))]
handles, labels = axes[0].get_legend_handles_labels()
handles_ordered = [handles[labels.index(lab)] for lab in legend_order_transp if lab in labels]

axes[0].legend(
    handles_ordered, legend_order_transp,
    title="Group",
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    borderaxespad=0.,
    frameon=True
)

# --- 2) Panel b) All groups ---
sns.boxplot(ax=axes[1], data=click_data_all_transp, x="Transparency Type", y="Click Frequency", color="white", fliersize=0)
sns.stripplot(
    ax=axes[1],
    data=click_data_all_transp,
    x="Transparency Type",
    y="Click Frequency",
    hue="Group Simple",
    palette=group_palette_transp,
    dodge=False,
    jitter=0.1,
    alpha=0.6
)

xtick_labels = [t.get_text() for t in axes[1].get_xticklabels()]
for i, treatment in enumerate(xtick_labels):
    for group_id in pivot_median_transp.index:
        group_simple = group_id_map_all_transp[group_id]
        color = group_palette_transp[group_simple]
        median_val = pivot_median_transp.loc[group_id, treatment] if treatment in pivot_median_transp.columns else None
        if pd.notna(median_val):
            jitter = np.random.uniform(-0.05, 0.05)
            x_val = i + jitter
            axes[1].scatter(x_val, median_val, marker='s', s=80, color=color, edgecolor='black', zorder=5)

axes[1].set_title("b) Click Frequency by Treatment (All Groups)\nSquares = Group medians")
axes[1].set_xlabel("Treatment Type")
axes[1].set_ylabel("")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend([],[], frameon=False)

# Final layout
plt.tight_layout()
plt.savefig(os.path.join(graph_output_dir, "panel_boxplots_click_frequency_transparency_with_medians_2panel.png"), dpi=600)
plt.close()

print("2-panel box plot (Transparency) with group medians and full titles saved (with jitter).")


########################################################################
# H5a) b) 6 7: Market Fragility
########################################################################

#####################################################################
# Compute Inverse Flow Times in Click Space — ALL Treatments, ALL Agents
#####################################################################

import os
import csv
import pandas as pd

# === Load treatments ===
treatments = pd.read_csv("treatments_summary.csv")

# === Function to compute flux times (click space) from agent dataframe ===
def extract_flux_times_click_space_from_df(agent_df, block=6):
    click_id = 0
    collected_click_ids = []

    for _, row in agent_df.iterrows():
        if row["Requested"] == 1:
            click_id += 1
        if row["Collected"] == 1:
            collected_click_ids.append(click_id)

    # Rolling window computation
    flux_clicks = []
    if len(collected_click_ids) >= block:
        for k in range(len(collected_click_ids) - block + 1):
            start = collected_click_ids[k]
            end = collected_click_ids[k + block - 1]
            flux_clicks.append(end - start)

    return flux_clicks

# === Prepare output container ===
all_flux_rows = []

# === Main loop over ALL treatments in treatments_summary.csv ===
block_size = 6  # same as your simulation

for _, row in treatments.iterrows():
    session = row["Session Code"]
    round_num = row["Round"]
    group = row["Group"]

    inv_dist = row['Inventory Distribution'].replace(', ', '-')
    cost = row['Cost per Second']
    transp = 0 if row['Transparency'] == "No" else 1

    filename = f"treatment_outputs/session{session}_round{round_num}_group{group}_Agents5_IniInv{inv_dist}_Cost{cost}_Transp{transp}.csv"
    if not os.path.exists(filename):
        print(f"Missing file: {filename}")
        continue

    df = pd.read_csv(filename)

    # Make sure columns are numeric
    df["Requested"] = pd.to_numeric(df["Requested"], errors='coerce').fillna(0).astype(int)
    df["Collected"] = pd.to_numeric(df["Collected"], errors='coerce').fillna(0).astype(int)

    for agent in df["Agent"].unique():
        agent_df = df[df["Agent"] == agent]

        flux_clicks = extract_flux_times_click_space_from_df(agent_df, block=block_size)

        for flux_id, flux_value in enumerate(flux_clicks, start=1):
            all_flux_rows.append({
                "Session": session,
                "Round": round_num,
                "Group": group,
                "Agent": agent,
                "Inventory Distribution": row['Inventory Distribution'],
                "Cost per Second": cost,
                "Transparency": row['Transparency'],
                "Group ID": f"{session}_{group}",
                "Flux ID": flux_id,
                "Flux Click Span": flux_value
            })

    print(f" Processed Session={session} Round={round_num} Group={group}")

# === Convert to DataFrame and save ===
flux_df = pd.DataFrame(all_flux_rows)

output_csv = "inverse_flow_times_click_space_all_agents.csv"
flux_df.to_csv(output_csv, index=False)

print(f"\n ALL flux times in CLICK SPACE saved to: {output_csv}")
print(f"Total number of flux rows: {len(flux_df)}")





############################################################
# H5a) b) INEVNTROY TREATMENT and market fragility
############################################################

#------------------------------------------------------------
# Preprocess Inverse Flow Times — Per Agent Summary
#------------------------------------------------------------

import pandas as pd
import numpy as np

# Load inverse flow data
inverse_flow_df = pd.read_csv("inverse_flow_times_click_space_all_agents.csv")

# === Classify Inventory Type ===
def classify_inventory(dist):
    dist = dist.strip()
    if dist == "10, 0, 0, 0, 0":
        return 10
    elif dist == "5, 0, 0, 0, 0":
        return 5
    elif dist == "3, 0, 0, 0, 0":
        return 3
    elif dist == "1, 0, 0, 0, 0":
        return 1
    else:
        return None

inverse_flow_df["Inventory Type"] = inverse_flow_df["Inventory Distribution"].apply(classify_inventory)
inverse_flow_df = inverse_flow_df.dropna(subset=["Inventory Type"])

# === Compute Mean + StdDev per Agent ===
agent_summary_df = inverse_flow_df.groupby(
    ["Session", "Round", "Group", "Agent", "Inventory Type"]
)["Flux Click Span"].agg(
    MeanInverseFlow = "mean",
    StdDevInverseFlow = "std"
).reset_index()

# === Add Group ID for plotting ===
agent_summary_df["Group ID"] = agent_summary_df.apply(
    lambda row: f"{row['Session']}_{row['Group']}", axis=1
)

# Save preprocessed agent summary (nice for re-use!)
agent_summary_df.to_csv("inverse_flow_per_agent_summary.csv", index=False)
print("Saved inverse_flow_per_agent_summary.csv")




############################################################
# PLOT: Inverse Flow by Initial Inventory — Mean + StdDev 
############################################################

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Output directory
graph_output_dir = "Organized Graphs"
os.makedirs(graph_output_dir, exist_ok=True)

# === Prepare consistent Group mapping ===
unique_groups_sorted = sorted(agent_summary_df['Group ID'].unique())
group_id_map = {group: f"Group {i+1}" for i, group in enumerate(unique_groups_sorted)}
agent_summary_df['Group Simple'] = agent_summary_df['Group ID'].map(group_id_map)

# === Prepare Inventory Order to match reference ===
inventory_order = [1.0, 3.0, 5.0, 10.0]  # left to right = 1 → 10

# === Compute Group Medians ===
group_median_mean_flow = agent_summary_df.groupby(['Group ID', 'Inventory Type'])['MeanInverseFlow'].median().reset_index()
group_median_std_flow  = agent_summary_df.groupby(['Group ID', 'Inventory Type'])['StdDevInverseFlow'].median().reset_index()

# === 2-panel plot ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# --- Common boxplot style ---
boxplot_style = dict(
    boxprops=dict(edgecolor='black', linewidth=1),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    medianprops=dict(color='black', linewidth=1),
    flierprops=dict(marker='o', markersize=3, linestyle='none', color='black', alpha=0.5)
)

# Panel a) Mean Inverse Flow
sns.boxplot(
    ax=axes[0],
    data=agent_summary_df,
    x="Inventory Type",
    y="MeanInverseFlow",
    order=inventory_order,
    color="white",
    showfliers=False,
    **boxplot_style
)
sns.stripplot(
    ax=axes[0],
    data=agent_summary_df,
    x="Inventory Type",
    y="MeanInverseFlow",
    order=inventory_order,
    color="black",
    dodge=False,
    jitter=True,
    alpha=0.5
)
# Add squares = group medians
for _, row in group_median_mean_flow.iterrows():
    x_pos = inventory_order.index(row['Inventory Type'])
    y_val = row['MeanInverseFlow']
    axes[0].scatter(
        x_pos, y_val,
        marker='s',
        s=80,
        facecolors='white',
        edgecolors='black',
        linewidth=1.2,
        zorder=5
    )

axes[0].set_title("a) Mean Inverse Flow by Total Inventory\n(All Groups)\nSquares = Group medians")
axes[0].set_xlabel("Total Inventory")
axes[0].set_ylabel("Mean Inverse Flow [Click Space]")
axes[0].grid(True, linestyle="--", alpha=0.4)
axes[0].legend([],[], frameon=False)

# Panel b) StdDev Inverse Flow
sns.boxplot(
    ax=axes[1],
    data=agent_summary_df,
    x="Inventory Type",
    y="StdDevInverseFlow",
    order=inventory_order,
    color="white",
    showfliers=False,
    **boxplot_style
)
sns.stripplot(
    ax=axes[1],
    data=agent_summary_df,
    x="Inventory Type",
    y="StdDevInverseFlow",
    order=inventory_order,
    color="black",
    dodge=False,
    jitter=True,
    alpha=0.5
)
# Add squares = group medians
for _, row in group_median_std_flow.iterrows():
    x_pos = inventory_order.index(row['Inventory Type'])
    y_val = row['StdDevInverseFlow']
    axes[1].scatter(
        x_pos, y_val,
        marker='s',
        s=80,
        facecolors='white',
        edgecolors='black',
        linewidth=1.2,
        zorder=5
    )

axes[1].set_title("b) StdDev of Inverse Flow by Total Inventory\n(All Groups)\nSquares = Group medians")
axes[1].set_xlabel("Total Inventory")
axes[1].set_ylabel("StdDev of Inverse Flow [Click Space]")
axes[1].grid(True, linestyle="--", alpha=0.4)
axes[1].legend([],[], frameon=False)

# Final layout and save
plt.tight_layout()
output_path = os.path.join(graph_output_dir, "boxplot_inverse_flow_mean_stddev_all_groups_inventory_FINAL_SQUARES.png")
plt.savefig(output_path, dpi=600)
plt.close()

print(f"Saved FINAL figure with group median squares to:\n{output_path}")

# === Print Group Medians ===
print("\n=== Group-level medians (shown as squares) ===")

print("\nMean Inverse Flow (group medians):")
for inv in inventory_order:
    vals = group_median_mean_flow[group_median_mean_flow['Inventory Type'] == inv]['MeanInverseFlow'].values
    print(f"Inventory {int(inv)}: Median Mean Inverse Flow = {vals.mean():.2f}")

print("\nStdDev Inverse Flow (group medians):")
for inv in inventory_order:
    vals = group_median_std_flow[group_median_std_flow['Inventory Type'] == inv]['StdDevInverseFlow'].values
    print(f"Inventory {int(inv)}: Median StdDev Inverse Flow = {vals.mean():.2f}")

############################################################
# Seaborn Distributions 
############################################################

############################################################
# KDE Plot of Inverse Flow Times [Click Space] per Inventory Type
############################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import kruskal, spearmanr

# === Settings ===
graph_output_dir = "Organized Graphs"
os.makedirs(graph_output_dir, exist_ok=True)

# === Load inverse flow data ===
inverse_flow_df = pd.read_csv("inverse_flow_times_click_space_all_agents.csv")

# === Classify Inventory Type ===
def classify_inventory(dist):
    dist = dist.strip()
    if dist == "10, 0, 0, 0, 0":
        return 10
    elif dist == "5, 0, 0, 0, 0":
        return 5
    elif dist == "3, 0, 0, 0, 0":
        return 3
    elif dist == "1, 0, 0, 0, 0":
        return 1
    else:
        return None

inverse_flow_df["Inventory Type"] = inverse_flow_df["Inventory Distribution"].apply(classify_inventory)
inverse_flow_df = inverse_flow_df.dropna(subset=["Inventory Type"])

# === Prepare data ===
# One big list of Flux Click Span per Inventory Type
inventory_order = [1.0, 3.0, 5.0, 10.0]  # for ordering in plot

# === Plot KDE ===
plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set2", len(inventory_order))

for i, inv_type in enumerate(inventory_order):
    data_subset = inverse_flow_df[inverse_flow_df["Inventory Type"] == inv_type]["Flux Click Span"]
    sns.kdeplot(
        data_subset,
        label=f"Inventory {int(inv_type)}",
        linewidth=2,
        fill=True,
        alpha=0.4,
        color=palette[i],
    )

plt.title("Density Plot of Inverse Flow Times [Click Space]\nby Initial Inventory")
plt.xlabel("Inverse Flow Time [Click Space]")
plt.ylabel("Density")
plt.legend(title="Inventory Type")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

output_path = os.path.join(graph_output_dir, "kde_inverse_flow_times_click_space_inventory.png")
plt.savefig(output_path, dpi=600)
plt.close()

print(f"KDE plot saved to: {output_path}")


import numpy as np  # Add this import at the top with your other imports

# === Plot KDE with log(Flux Click Span + 1) on x-axis ===
plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set2", len(inventory_order))

for i, inv_type in enumerate(inventory_order):
    data_subset = inverse_flow_df[inverse_flow_df["Inventory Type"] == inv_type]["Flux Click Span"]
    data_subset_log = np.log(data_subset + 1)  # Apply log transform
    
    sns.kdeplot(
        data_subset_log,
        label=f"Inventory {int(inv_type)}",
        linewidth=2,
        fill=True,
        alpha=0.4,
        color=palette[i],
    )

plt.title("Density Plot of log(1 + Inverse Flow Times) [Click Space]\nby Initial Inventory")
plt.xlabel("log(1 + Inverse Flow Time) [Click Space]")
plt.ylabel("Density")
plt.legend(title="Inventory Type")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()

output_path = os.path.join(graph_output_dir, "kde_log_inverse_flow_times_click_space_inventory.png")
plt.savefig(output_path, dpi=600)
plt.close()

print(f"KDE plot saved to: {output_path}")


############################################################
# FINAL 4-PANEL PLOT + STATISTICS: Mean + StdDev of Inverse Flow Times (linear + log-log)
# Based on full KDE distributions (all agent flux times pooled)
############################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kruskal
import os

# === Load data ===
inverse_flow_df = pd.read_csv("inverse_flow_times_click_space_all_agents.csv")

# === Classify Inventory Type ===
def classify_inventory(dist):
    dist = dist.strip()
    if dist == "10, 0, 0, 0, 0":
        return 10
    elif dist == "5, 0, 0, 0, 0":
        return 5
    elif dist == "3, 0, 0, 0, 0":
        return 3
    elif dist == "1, 0, 0, 0, 0":
        return 1
    else:
        return None

inverse_flow_df["Inventory Type"] = inverse_flow_df["Inventory Distribution"].apply(classify_inventory)
inverse_flow_df = inverse_flow_df.dropna(subset=["Inventory Type"])
inverse_flow_df["Inventory Type"] = inverse_flow_df["Inventory Type"].astype(int)

# === Prepare inventory order ===
inventory_order = [1, 3, 5, 10]

# === Group by Inventory Type — compute mean and stddev of Flux Click Span (pooled across agents/groups)
mean_inverse_flow = inverse_flow_df.groupby("Inventory Type")["Flux Click Span"].mean()
stddev_inverse_flow = inverse_flow_df.groupby("Inventory Type")["Flux Click Span"].std()

# === Run Spearman correlations ===
spearman_mean_result = spearmanr(mean_inverse_flow.index, mean_inverse_flow.values)
spearman_stddev_result = spearmanr(stddev_inverse_flow.index, stddev_inverse_flow.values)

# === Run Kruskal-Wallis test on full distributions ===
flow_lists = []
for inv_type in inventory_order:
    flow_values = inverse_flow_df[inverse_flow_df["Inventory Type"] == inv_type]["Flux Click Span"].values
    flow_lists.append(flow_values)

kruskal_result = kruskal(*flow_lists)

# === Save table to CSV ===
results_df = pd.DataFrame({
    "Inventory Type": mean_inverse_flow.index,
    "Mean Inverse Flow (Flux Click Span)": mean_inverse_flow.values,
    "StdDev Inverse Flow (Flux Click Span)": stddev_inverse_flow.values
})

# Add global statistics to the table (same value for all rows for documentation)
results_df["Spearman Mean rho"] = spearman_mean_result.correlation
results_df["Spearman Mean p"] = spearman_mean_result.pvalue
results_df["Spearman StdDev rho"] = spearman_stddev_result.correlation
results_df["Spearman StdDev p"] = spearman_stddev_result.pvalue
results_df["Kruskal-Wallis H"] = kruskal_result.statistic
results_df["Kruskal-Wallis p"] = kruskal_result.pvalue

# Save CSV
results_csv_path = "inverse_flow_mean_stddev_stats_table_FINAL.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Saved statistics table to: {results_csv_path}")

# === Plotting ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

grey_color = "#444444"
marker_style = 'o'

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Panel a) Mean - linear
axes[0, 0].plot(mean_inverse_flow.index, mean_inverse_flow.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[0, 0].set_title("a) Mean Inverse Flow\nvs. Initial Inventory")
axes[0, 0].set_xlabel("Total Inventory")
axes[0, 0].set_ylabel("Mean Inverse Flow [Click Space]")
axes[0, 0].grid(True)

# Panel b) StdDev - linear
axes[0, 1].plot(stddev_inverse_flow.index, stddev_inverse_flow.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[0, 1].set_title("b) StdDev of Inverse Flow\nvs. Initial Inventory")
axes[0, 1].set_xlabel("Total Inventory")
axes[0, 1].set_ylabel("StdDev of Inverse Flow [Click Space]")
axes[0, 1].grid(True)

# Panel c) Mean - log-log
axes[1, 0].plot(mean_inverse_flow.index, mean_inverse_flow.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[1, 0].set_xscale("log")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("c) Mean Inverse Flow\nvs. Initial Inventory (log-log)")
axes[1, 0].set_xlabel("Total Inventory (log)")
axes[1, 0].set_ylabel("Mean Inverse Flow (log)")
axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.5)

# Panel d) StdDev - log-log
axes[1, 1].plot(stddev_inverse_flow.index, stddev_inverse_flow.values,
                color=grey_color,
                marker=marker_style,
                markersize=5,
                markerfacecolor='white',
                markeredgewidth=1.0,
                markeredgecolor=grey_color,
                linestyle='-')
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("d) StdDev of Inverse Flow\nvs. Initial Inventory (log-log)")
axes[1, 1].set_xlabel("Total Inventory (log)")
axes[1, 1].set_ylabel("StdDev of Inverse Flow (log)")
axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.5)

# === Final layout and save ===
plt.tight_layout()
output_path = os.path.join("Organized Graphs", "InverseFlow_MeanStdDev_4Panel_ClickSpace_FINAL.png")
plt.savefig(output_path, dpi=600)
plt.close()

# === Print results ===
print("\n=== Spearman Correlation Results ===")
print(f"Mean Inverse Flow → Spearman rho = {spearman_mean_result.correlation:.4f}, p = {spearman_mean_result.pvalue:.4f}")
print(f"StdDev Inverse Flow → Spearman rho = {spearman_stddev_result.correlation:.4f}, p = {spearman_stddev_result.pvalue:.4f}")

print("\n=== Kruskal-Wallis Test ===")
print(f"Kruskal-Wallis H = {kruskal_result.statistic:.4f}, p = {kruskal_result.pvalue:.4f}")

print(f"\nSaved 4-panel plot to: {output_path}")


# === Print means and stddevs per Inventory Type ===
print("\n=== Mean and StdDev of Inverse Flow (raw distributions) ===")
for inv_type in inventory_order:
    mean_val = mean_inverse_flow.loc[inv_type]
    std_val = stddev_inverse_flow.loc[inv_type]
    print(f"Inventory {inv_type}: Mean = {mean_val:.2f}, StdDev = {std_val:.2f}")







##############################################################################
# TRANSPARENCY
##############################################################################

############################################################
# FINAL Analysis — Inverse Flow by TRANSPARENCY
# All Groups and Paired Groups
############################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import kruskal, spearmanr

# === Settings ===
graph_output_dir = "Organized Graphs"
os.makedirs(graph_output_dir, exist_ok=True)

# === Load inverse flow data ===
inverse_flow_df = pd.read_csv("inverse_flow_times_click_space_all_agents.csv")
treatments = pd.read_csv("treatments_summary.csv")

# === Classify Transparency Type ===
def classify_transparency_existing(row):
    if row["Inventory Distribution"].strip() == "2, 2, 2, 2, 2":
        if row["Transparency"] == "No":
            return "Non-Transparent"
        elif row["Transparency"] == "Yes":
            return "Transparent"
    return None

inverse_flow_df["Transparency Type"] = inverse_flow_df.apply(classify_transparency_existing, axis=1)
inverse_flow_df = inverse_flow_df.dropna(subset=["Transparency Type"])

# === Compute Mean + StdDev per Agent ===
agent_summary_df = inverse_flow_df.groupby(
    ["Session", "Round", "Group", "Agent", "Transparency Type"]
)["Flux Click Span"].agg(
    MeanInverseFlow="mean",
    StdDevInverseFlow="std"
).reset_index()

# === Add Group ID ===
agent_summary_df["Group ID"] = agent_summary_df.apply(
    lambda row: f"{row['Session']}_{row['Group']}", axis=1
)

# === Identify Paired Sessions ===
pivot_transp_session = agent_summary_df.pivot_table(
    index="Session", columns="Transparency Type", values="MeanInverseFlow", aggfunc="count"
)
paired_sessions = pivot_transp_session.dropna(subset=["Transparent", "Non-Transparent"]).index.tolist()

# Split data
agent_summary_all = agent_summary_df.copy()
agent_summary_paired = agent_summary_df[agent_summary_df["Session"].isin(paired_sessions)]

# Prepare raw data subset for KDE paired groups
inverse_flow_df_paired = inverse_flow_df[inverse_flow_df["Session"].isin(paired_sessions)]

# === Prepare Group mappings — CONSISTENT across All and Paired Groups — FIX LEGEND ORDER ===
# Sort Group ID -> map to Group 1, Group 2, Group 3...
unique_groups_sorted_all = sorted(agent_summary_all["Group ID"].unique())
group_id_map_consistent = {group: f"Group {i+1}" for i, group in enumerate(unique_groups_sorted_all)}

# Apply to both All and Paired
agent_summary_all["Group Simple"] = agent_summary_all["Group ID"].map(group_id_map_consistent)
agent_summary_paired["Group Simple"] = agent_summary_paired["Group ID"].map(group_id_map_consistent)

# Define consistent group color map — match correct order of Group Simple
all_group_simples_sorted = sorted(agent_summary_all["Group Simple"].unique(), key=lambda x: int(x.split(" ")[1]))
group_palette = sns.color_palette("Set2", n_colors=len(all_group_simples_sorted))
group_color_map = {group_simple: group_palette[i % len(group_palette)]
                   for i, group_simple in enumerate(all_group_simples_sorted)}

# === Function to compute and plot boxplots ===
def plot_boxplot_panels(agent_df, group_color_map, label_for_title, output_prefix):
    # Compute group medians
    group_median_mean = agent_df.groupby(["Group ID", "Transparency Type"])["MeanInverseFlow"].median().reset_index()
    group_median_std = agent_df.groupby(["Group ID", "Transparency Type"])["StdDevInverseFlow"].median().reset_index()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    boxplot_style = dict(
        boxprops=dict(edgecolor='black', linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1),
        medianprops=dict(color='black', linewidth=1),
        flierprops=dict(marker='o', markersize=3, linestyle='none', color='black', alpha=0.5)
    )

    # Panel a) Mean Inverse Flow
    sns.boxplot(
        ax=axes[0],
        data=agent_df,
        x="Transparency Type",
        y="MeanInverseFlow",
        color="white",
        showfliers=False,
        **boxplot_style
    )
    strip_ax0 = sns.stripplot(
        ax=axes[0],
        data=agent_df,
        x="Transparency Type",
        y="MeanInverseFlow",
        hue="Group Simple",
        palette=group_color_map,
        dodge=False,
        jitter=True,  
        alpha=0.8,
        size=6
    )
    for _, row in group_median_mean.iterrows():
        x_pos = 0 if row["Transparency Type"] == "Non-Transparent" else 1
        group_simple = agent_df[agent_df["Group ID"] == row["Group ID"]]["Group Simple"].iloc[0]
        color = group_color_map[group_simple]

        # Add small x jitter to median square for visibility
        x_jitter = 0.01 * (hash(group_simple) % 4 - 2)  # VERY small jitter
        axes[0].scatter(
            x_pos + x_jitter, row["MeanInverseFlow"],
            marker='s',
            s=120,
            facecolors=color,
            edgecolors='black',
            linewidth=1.5,
            zorder=6
        )
    axes[0].set_title(f"a) Mean Inverse Flow by Transparency\n({label_for_title})\nSquares = Group medians")
    axes[0].set_xlabel("Transparency Type")
    axes[0].set_ylabel("Mean Inverse Flow [Click Space]")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Add legend only for paired groups — clean order of Group 1, 2, 3
    if "PAIREDGROUPS" in output_prefix:
        handles, labels_ = axes[0].get_legend_handles_labels()
        sorted_pairs = sorted(zip(labels_, handles), key=lambda x: int(x[0].split(" ")[1]))
        labels_sorted, handles_sorted = zip(*sorted_pairs)
        axes[0].legend(handles=handles_sorted, labels=labels_sorted, title="Group", loc="best")
    else:
        axes[0].legend([], [], frameon=False)

    # Panel b) StdDev Inverse Flow
    sns.boxplot(
        ax=axes[1],
        data=agent_df,
        x="Transparency Type",
        y="StdDevInverseFlow",
        color="white",
        showfliers=False,
        **boxplot_style
    )
    strip_ax1 = sns.stripplot(
        ax=axes[1],
        data=agent_df,
        x="Transparency Type",
        y="StdDevInverseFlow",
        hue="Group Simple",
        palette=group_color_map,
        dodge=False,
        jitter=True,  
        alpha=0.8,
        size=6
    )
    for _, row in group_median_std.iterrows():
        x_pos = 0 if row["Transparency Type"] == "Non-Transparent" else 1
        group_simple = agent_df[agent_df["Group ID"] == row["Group ID"]]["Group Simple"].iloc[0]
        color = group_color_map[group_simple]

        # Add small x jitter to median square for visibility
        x_jitter = 0.01 * (hash(group_simple) % 4 - 2)  # VERY small jitter
        axes[1].scatter(
            x_pos + x_jitter, row["StdDevInverseFlow"],
            marker='s',
            s=120,
            facecolors=color,
            edgecolors='black',
            linewidth=1.5,
            zorder=6
        )
    axes[1].set_title(f"b) StdDev of Inverse Flow by Transparency\n({label_for_title})\nSquares = Group medians")
    axes[1].set_xlabel("Transparency Type")
    axes[1].set_ylabel("StdDev of Inverse Flow [Click Space]")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Add legend only for paired groups — clean order of Group 1, 2, 3
    if "PAIREDGROUPS" in output_prefix:
        handles, labels_ = axes[1].get_legend_handles_labels()
        sorted_pairs = sorted(zip(labels_, handles), key=lambda x: int(x[0].split(" ")[1]))
        labels_sorted, handles_sorted = zip(*sorted_pairs)
        axes[1].legend(handles=handles_sorted, labels=labels_sorted, title="Group", loc="best")
    else:
        axes[1].legend([], [], frameon=False)

    plt.tight_layout()
    output_path = os.path.join(graph_output_dir, f"boxplot_inverse_flow_mean_stddev_transparency_{output_prefix}_FINAL_SQUARES.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {label_for_title} boxplot to: {output_path}")

# === KDE Plot function — log(Flux Click Span + 1) ===
def plot_kde(inverse_flow_subset, label_for_title, output_prefix):
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("Set2", 2)

    for i, ttype in enumerate(["Non-Transparent", "Transparent"]):
        data_subset = inverse_flow_subset[inverse_flow_subset["Transparency Type"] == ttype]["Flux Click Span"]
        
        # Transform to log(flow + 1)
        data_subset_log = np.log(data_subset + 1)
        
        sns.kdeplot(
            data_subset_log,
            label=f"{ttype}",
            linewidth=2,
            fill=True,
            alpha=0.4,
            color=palette[i],
        )
    plt.title(f"Density Plot of log(Flux Click Span + 1)\nby Transparency ({label_for_title})")
    plt.xlabel("log(Flux Click Span + 1)")
    plt.ylabel("Density")
    plt.legend(title="Transparency Type")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = os.path.join(graph_output_dir, f"kde_inverse_flow_times_click_space_transparency_{output_prefix}_LOG.png")
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"KDE plot saved to: {output_path}")


# === PLOT for ALL GROUPS ===
plot_boxplot_panels(agent_summary_all, group_color_map, label_for_title="All Groups", output_prefix="ALLGROUPS")

# === PLOT for PAIRED GROUPS ===
plot_boxplot_panels(agent_summary_paired, group_color_map, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS")

# === KDE for ALL GROUPS ===
plot_kde(inverse_flow_df, label_for_title="All Groups", output_prefix="ALLGROUPS")

# === KDE for PAIRED GROUPS ===
plot_kde(inverse_flow_df_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS")



#####################
# 6 PANEL
#####################

# === Combined 6-panel figure ===

fig, axes = plt.subplots(3, 2, figsize=(12, 18))

# === Panels a) and b) — All Groups Boxplots ===
group_median_mean_all = agent_summary_all.groupby(["Group ID", "Transparency Type"])["MeanInverseFlow"].median().reset_index()
group_median_std_all = agent_summary_all.groupby(["Group ID", "Transparency Type"])["StdDevInverseFlow"].median().reset_index()

boxplot_style = dict(
    boxprops=dict(edgecolor='black', linewidth=1),
    whiskerprops=dict(color='black', linewidth=1),
    capprops=dict(color='black', linewidth=1),
    medianprops=dict(color='black', linewidth=1),
    flierprops=dict(marker='o', markersize=3, linestyle='none', color='black', alpha=0.5)
)

# Panel a
sns.boxplot(
    ax=axes[0, 0],
    data=agent_summary_all,
    x="Transparency Type",
    y="MeanInverseFlow",
    color="white",
    showfliers=False,
    **boxplot_style
)
sns.stripplot(
    ax=axes[0, 0],
    data=agent_summary_all,
    x="Transparency Type",
    y="MeanInverseFlow",
    hue="Group Simple",
    palette=group_color_map,
    dodge=False,
    jitter=True,
    alpha=0.8,
    size=6
)
for _, row in group_median_mean_all.iterrows():
    x_pos = 0 if row["Transparency Type"] == "Non-Transparent" else 1
    group_simple = agent_summary_all[agent_summary_all["Group ID"] == row["Group ID"]]["Group Simple"].iloc[0]
    color = group_color_map[group_simple]
    x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
    axes[0, 0].scatter(
        x_pos + x_jitter, row["MeanInverseFlow"],
        marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
    )
axes[0, 0].set_title(f"a) Mean Inverse Flow by Transparency\n(All Groups)\nSquares = Group medians")
axes[0, 0].set_xlabel("Transparency Type")
axes[0, 0].set_ylabel("Mean Inverse Flow [Click Space]")
axes[0, 0].grid(True, linestyle="--", alpha=0.4)
axes[0, 0].legend([], [], frameon=False)

# Panel b
sns.boxplot(
    ax=axes[0, 1],
    data=agent_summary_all,
    x="Transparency Type",
    y="StdDevInverseFlow",
    color="white",
    showfliers=False,
    **boxplot_style
)
sns.stripplot(
    ax=axes[0, 1],
    data=agent_summary_all,
    x="Transparency Type",
    y="StdDevInverseFlow",
    hue="Group Simple",
    palette=group_color_map,
    dodge=False,
    jitter=True,
    alpha=0.8,
    size=6
)
for _, row in group_median_std_all.iterrows():
    x_pos = 0 if row["Transparency Type"] == "Non-Transparent" else 1
    group_simple = agent_summary_all[agent_summary_all["Group ID"] == row["Group ID"]]["Group Simple"].iloc[0]
    color = group_color_map[group_simple]
    x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
    axes[0, 1].scatter(
        x_pos + x_jitter, row["StdDevInverseFlow"],
        marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
    )
axes[0, 1].set_title(f"b) StdDev of Inverse Flow by Transparency\n(All Groups)\nSquares = Group medians")
axes[0, 1].set_xlabel("Transparency Type")
axes[0, 1].set_ylabel("StdDev of Inverse Flow [Click Space]")
axes[0, 1].grid(True, linestyle="--", alpha=0.4)
axes[0, 1].legend([], [], frameon=False)

# === Panels c) and d) — Paired Groups Boxplots ===
group_median_mean_paired = agent_summary_paired.groupby(["Group ID", "Transparency Type"])["MeanInverseFlow"].median().reset_index()
group_median_std_paired = agent_summary_paired.groupby(["Group ID", "Transparency Type"])["StdDevInverseFlow"].median().reset_index()

# Panel c
sns.boxplot(
    ax=axes[1, 0],
    data=agent_summary_paired,
    x="Transparency Type",
    y="MeanInverseFlow",
    color="white",
    showfliers=False,
    **boxplot_style
)
sns.stripplot(
    ax=axes[1, 0],
    data=agent_summary_paired,
    x="Transparency Type",
    y="MeanInverseFlow",
    hue="Group Simple",
    palette=group_color_map,
    dodge=False,
    jitter=True,
    alpha=0.8,
    size=6
)
for _, row in group_median_mean_paired.iterrows():
    x_pos = 0 if row["Transparency Type"] == "Non-Transparent" else 1
    group_simple = agent_summary_paired[agent_summary_paired["Group ID"] == row["Group ID"]]["Group Simple"].iloc[0]
    color = group_color_map[group_simple]
    x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
    axes[1, 0].scatter(
        x_pos + x_jitter, row["MeanInverseFlow"],
        marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
    )
axes[1, 0].set_title(f"c) Mean Inverse Flow by Transparency\n(Paired Groups)\nSquares = Group medians")
axes[1, 0].set_xlabel("Transparency Type")
axes[1, 0].set_ylabel("Mean Inverse Flow [Click Space]")
axes[1, 0].grid(True, linestyle="--", alpha=0.4)
# Add legend
handles, labels_ = axes[1, 0].get_legend_handles_labels()
sorted_pairs = sorted(zip(labels_, handles), key=lambda x: int(x[0].split(" ")[1]))
labels_sorted, handles_sorted = zip(*sorted_pairs)
axes[1, 0].legend(handles=handles_sorted, labels=labels_sorted, title="Group", loc="best")

# Panel d
sns.boxplot(
    ax=axes[1, 1],
    data=agent_summary_paired,
    x="Transparency Type",
    y="StdDevInverseFlow",
    color="white",
    showfliers=False,
    **boxplot_style
)
sns.stripplot(
    ax=axes[1, 1],
    data=agent_summary_paired,
    x="Transparency Type",
    y="StdDevInverseFlow",
    hue="Group Simple",
    palette=group_color_map,
    dodge=False,
    jitter=True,
    alpha=0.8,
    size=6
)
for _, row in group_median_std_paired.iterrows():
    x_pos = 0 if row["Transparency Type"] == "Non-Transparent" else 1
    group_simple = agent_summary_paired[agent_summary_paired["Group ID"] == row["Group ID"]]["Group Simple"].iloc[0]
    color = group_color_map[group_simple]
    x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
    axes[1, 1].scatter(
        x_pos + x_jitter, row["StdDevInverseFlow"],
        marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
    )
axes[1, 1].set_title(f"d) StdDev of Inverse Flow by Transparency\n(Paired Groups)\nSquares = Group medians")
axes[1, 1].set_xlabel("Transparency Type")
axes[1, 1].set_ylabel("StdDev of Inverse Flow [Click Space]")
axes[1, 1].grid(True, linestyle="--", alpha=0.4)
# Add legend
handles, labels_ = axes[1, 1].get_legend_handles_labels()
sorted_pairs = sorted(zip(labels_, handles), key=lambda x: int(x[0].split(" ")[1]))
labels_sorted, handles_sorted = zip(*sorted_pairs)
axes[1, 1].legend(handles=handles_sorted, labels=labels_sorted, title="Group", loc="best")

# === Panels e) and f) — KDE ===
palette_kde = sns.color_palette("Set2", 2)

# Panel e
for i, ttype in enumerate(["Non-Transparent", "Transparent"]):
    data_subset = inverse_flow_df[inverse_flow_df["Transparency Type"] == ttype]["Flux Click Span"]
    data_subset_log = np.log(data_subset + 1)
    sns.kdeplot(
        ax=axes[2, 0],
        data=data_subset_log,
        label=f"{ttype}",
        linewidth=2,
        fill=True,
        alpha=0.4,
        color=palette_kde[i]
    )
axes[2, 0].set_title(f"e) Density Plot of log(Flux Click Span + 1)\nby Transparency (All Groups)")
axes[2, 0].set_xlabel("log(Flux Click Span + 1)")
axes[2, 0].set_ylabel("Density")
axes[2, 0].grid(True, linestyle="--", alpha=0.4)
axes[2, 0].legend(title="Transparency Type")

# Panel f
for i, ttype in enumerate(["Non-Transparent", "Transparent"]):
    data_subset = inverse_flow_df_paired[inverse_flow_df_paired["Transparency Type"] == ttype]["Flux Click Span"]
    data_subset_log = np.log(data_subset + 1)
    sns.kdeplot(
        ax=axes[2, 1],
        data=data_subset_log,
        label=f"{ttype}",
        linewidth=2,
        fill=True,
        alpha=0.4,
        color=palette_kde[i]
    )
axes[2, 1].set_title(f"f) Density Plot of log(Flux Click Span + 1)\nby Transparency (Paired Groups)")
axes[2, 1].set_xlabel("log(Flux Click Span + 1)")
axes[2, 1].set_ylabel("Density")
axes[2, 1].grid(True, linestyle="--", alpha=0.4)
axes[2, 1].legend(title="Transparency Type")

# Final save
plt.tight_layout()
output_path_combined = os.path.join(graph_output_dir, "combined_6_panel_inverse_flow_FINAL_SQUARES.png")
plt.savefig(output_path_combined, dpi=600)
plt.close()
print(f"Combined 6-panel figure saved to: {output_path_combined}")



#------------------------------------------------------------
# Inverse Flow Transparency Statistics — using GROUP MEDIANS and STDDEVS
#------------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, kruskal

# === Step 1: compute group medians and stddevs ===

# Group median of agent mean inverse flow
group_median = agent_summary_df.groupby(['Group ID', 'Transparency Type'])['MeanInverseFlow'].median().reset_index()

# Group stddev of agent mean inverse flow
group_stddev = agent_summary_df.groupby(['Group ID', 'Transparency Type'])['StdDevInverseFlow'].median().reset_index()

# === Step 2: build pivot tables ===

# Pivot for group medians
pivot_median = group_median.pivot(index='Group ID', columns='Transparency Type', values='MeanInverseFlow')

# Pivot for group stddevs
pivot_stddev = group_stddev.pivot(index='Group ID', columns='Transparency Type', values='StdDevInverseFlow')

# === Step 3: identify paired and all groups ===

# Paired groups
paired_groups = pivot_median.dropna(subset=["Transparent", "Non-Transparent"])
paired_stddev = pivot_stddev.loc[paired_groups.index]

# All groups
all_groups_median = pivot_median
all_groups_stddev = pivot_stddev

# === Step 4: Kruskal-Wallis on raw distributions ===

# All groups
k_all = kruskal(
    inverse_flow_df[inverse_flow_df["Transparency Type"] == "Transparent"]["Flux Click Span"],
    inverse_flow_df[inverse_flow_df["Transparency Type"] == "Non-Transparent"]["Flux Click Span"]
)

# Paired groups
k_paired = kruskal(
    inverse_flow_df_paired[inverse_flow_df_paired["Transparency Type"] == "Transparent"]["Flux Click Span"],
    inverse_flow_df_paired[inverse_flow_df_paired["Transparency Type"] == "Non-Transparent"]["Flux Click Span"]
)

# === Step 5: compute tests ===

results = []

# Helper function to compute mean / median string
def describe_values(series):
    mean_val = series.mean()
    median_val = series.median()
    return f"{mean_val:.3f} ({median_val:.3f})"

# --- Kruskal-Wallis on raw data ---
results.append({
    "Section": "All groups",
    "Metric": "Raw Flux Click Span",
    "Transparent (Mean/Median)": describe_values(inverse_flow_df[inverse_flow_df["Transparency Type"] == "Transparent"]["Flux Click Span"]),
    "Non-Transparent (Mean/Median)": describe_values(inverse_flow_df[inverse_flow_df["Transparency Type"] == "Non-Transparent"]["Flux Click Span"]),
    "Test": "Kruskal-Wallis",
    "p-value": f"{k_all.pvalue:.4f}"
})

results.append({
    "Section": "Paired groups",
    "Metric": "Raw Flux Click Span",
    "Transparent (Mean/Median)": describe_values(inverse_flow_df_paired[inverse_flow_df_paired["Transparency Type"] == "Transparent"]["Flux Click Span"]),
    "Non-Transparent (Mean/Median)": describe_values(inverse_flow_df_paired[inverse_flow_df_paired["Transparency Type"] == "Non-Transparent"]["Flux Click Span"]),
    "Test": "Kruskal-Wallis",
    "p-value": f"{k_paired.pvalue:.4f}"
})

# --- Paired groups ---

# Median
wilcoxon_median = wilcoxon(paired_groups["Transparent"], paired_groups["Non-Transparent"])
results.append({
    "Section": "Paired groups",
    "Metric": "Group Median of MeanInverseFlow",
    "Transparent (Mean/Median)": describe_values(paired_groups["Transparent"]),
    "Non-Transparent (Mean/Median)": describe_values(paired_groups["Non-Transparent"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_median.pvalue:.4f}"
})

# StdDev
wilcoxon_stddev = wilcoxon(paired_stddev["Transparent"], paired_stddev["Non-Transparent"])
results.append({
    "Section": "Paired groups",
    "Metric": "Group Median of StdDevInverseFlow",
    "Transparent (Mean/Median)": describe_values(paired_stddev["Transparent"]),
    "Non-Transparent (Mean/Median)": describe_values(paired_stddev["Non-Transparent"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_stddev.pvalue:.4f}"
})

# --- All groups ---

# Median
all_median_transp = all_groups_median["Transparent"].dropna()
all_median_nontransp = all_groups_median["Non-Transparent"].dropna()
mann_median_all = mannwhitneyu(all_median_transp, all_median_nontransp, alternative='two-sided')
results.append({
    "Section": "All groups",
    "Metric": "Group Median of MeanInverseFlow",
    "Transparent (Mean/Median)": describe_values(all_median_transp),
    "Non-Transparent (Mean/Median)": describe_values(all_median_nontransp),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_median_all.pvalue:.4f}"
})

# StdDev
all_stddev_transp = all_groups_stddev["Transparent"].dropna()
all_stddev_nontransp = all_groups_stddev["Non-Transparent"].dropna()
mann_stddev_all = mannwhitneyu(all_stddev_transp, all_stddev_nontransp, alternative='two-sided')
results.append({
    "Section": "All groups",
    "Metric": "Group Median of StdDevInverseFlow",
    "Transparent (Mean/Median)": describe_values(all_stddev_transp),
    "Non-Transparent (Mean/Median)": describe_values(all_stddev_nontransp),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_stddev_all.pvalue:.4f}"
})

# === Step 6: assemble results table ===

results_df = pd.DataFrame(results)

# Show in console
print(results_df)

# Save to CSV
results_df.to_csv(os.path.join(graph_output_dir, "inverse_flow_transparency_statistics_table.csv"), index=False)

print("Statistics table saved to inverse_flow_transparency_statistics_table.csv.")




#############################################
# H7: SYM vs ASYM — Inverse Flow Version
#############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Settings ===
graph_output_dir = "Organized Graphs"
os.makedirs(graph_output_dir, exist_ok=True)

# === Load data ===
inverse_flow_df = pd.read_csv("inverse_flow_times_click_space_all_agents.csv")
treatments = pd.read_csv("treatments_summary.csv")

# === Classify treatments: SYM / ASYM ===
def classify_row(row):
    if row["Transparency"] == "No":
        dist = row["Inventory Distribution"].strip()
        if dist == "2, 2, 2, 2, 2":
            return "Symmetric"
        elif dist == "10, 0, 0, 0, 0":
            return "Asymmetric"
    return None

treatments["Treatment Type"] = treatments.apply(classify_row, axis=1)
valid_treatments = treatments.dropna(subset=["Treatment Type"])

# === Merge Treatment Type into inverse_flow_df ===
inverse_flow_df = inverse_flow_df.merge(
    treatments[["Session Code", "Round", "Group", "Treatment Type"]],
    how="left",
    left_on=["Session", "Round", "Group"],
    right_on=["Session Code", "Round", "Group"]
)

# Drop rows with no Treatment Type assigned
inverse_flow_df = inverse_flow_df.dropna(subset=["Treatment Type"])

# === Compute Mean + StdDev of Flux Click Span per Agent ===
records = []
for (session, round_num, group, agent), agent_df in inverse_flow_df.groupby(["Session", "Round", "Group", "Agent"]):
    treatment_type = agent_df["Treatment Type"].iloc[0]

    mean_inverse_flow = agent_df["Flux Click Span"].mean()
    stddev_inverse_flow = agent_df["Flux Click Span"].std()

    records.append({
        "Agent": agent,
        "Group": group,
        "Session": session,
        "Round": round_num,
        "Treatment Type": treatment_type,
        "Mean Inverse Flow": mean_inverse_flow,
        "StdDev Inverse Flow": stddev_inverse_flow,
        "Group ID": f"{session}_{group}"
    })

# === Create dataframe ===
inverse_flow_data = pd.DataFrame(records)

# === Identify Paired Groups ===
pivot = inverse_flow_data.pivot_table(index="Group ID", columns="Treatment Type", values="Mean Inverse Flow", aggfunc='count')
paired_groups = pivot.dropna(subset=["Symmetric", "Asymmetric"]).index.tolist()

# === Split data ===
inverse_flow_data_paired = inverse_flow_data[inverse_flow_data["Group ID"].isin(paired_groups)]
inverse_flow_data_all = inverse_flow_data.copy()
inverse_flow_data_unpaired = inverse_flow_data[~inverse_flow_data["Group ID"].isin(paired_groups)]

# === Global Group mappings — CONSISTENT ===
all_unique_group_ids = sorted(inverse_flow_data["Group ID"].unique())
global_group_id_map = {group: f"Group {i+1}" for i, group in enumerate(all_unique_group_ids)}

# Apply global mapping
inverse_flow_data_all["Group Simple Global"] = inverse_flow_data_all["Group ID"].map(global_group_id_map)
inverse_flow_data_paired["Group Simple Global"] = inverse_flow_data_paired["Group ID"].map(global_group_id_map)
inverse_flow_data_unpaired["Group Simple Global"] = inverse_flow_data_unpaired["Group ID"].map(global_group_id_map)

# === Global color map ===
global_palette_colors = sns.color_palette("tab20", n_colors=len(all_unique_group_ids))
global_color_map = {global_group_id_map[group]: global_palette_colors[i % len(global_palette_colors)]
                    for i, group in enumerate(all_unique_group_ids)}

# === Function to plot boxplots ===
def plot_boxplot_panels_sym_asym(inverse_df, label_for_title, output_prefix, force_paired_legend=False):
    group_median_mean = inverse_df.groupby(["Group ID", "Treatment Type"])["Mean Inverse Flow"].median().reset_index()
    group_median_std = inverse_df.groupby(["Group ID", "Treatment Type"])["StdDev Inverse Flow"].median().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    boxplot_style = dict(
        boxprops=dict(edgecolor='black', linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1),
        medianprops=dict(color='black', linewidth=1),
        flierprops=dict(marker='o', markersize=3, linestyle='none', color='black', alpha=0.5)
    )

    ### PANEL A ###
    sns.boxplot(
        ax=axes[0],
        data=inverse_df,
        x="Treatment Type",
        y="Mean Inverse Flow",
        color="white",
        showfliers=False,
        **boxplot_style
    )
    sns.stripplot(
        ax=axes[0],
        data=inverse_df,
        x="Treatment Type",
        y="Mean Inverse Flow",
        hue="Group Simple Global",
        palette=global_color_map,
        dodge=False,
        jitter=True,
        alpha=0.8,
        size=6
    )
    xticks_labels_a = [t.get_text() for t in axes[0].get_xticklabels()]
    for _, row in group_median_mean.iterrows():
        x_pos = xticks_labels_a.index(row["Treatment Type"])
        group_simple = inverse_df[inverse_df["Group ID"] == row["Group ID"]]["Group Simple Global"].iloc[0]
        color = global_color_map[group_simple]
        x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
        axes[0].scatter(
            x_pos + x_jitter, row["Mean Inverse Flow"],
            marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
        )
    axes[0].set_title(f"a) Mean Inverse Flow by Treatment\n({label_for_title})\nSquares = Group medians")
    axes[0].set_xlabel("Treatment Type")
    axes[0].set_ylabel("Mean Inverse Flow [Click Space]")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    ### PANEL B ###
    sns.boxplot(
        ax=axes[1],
        data=inverse_df,
        x="Treatment Type",
        y="StdDev Inverse Flow",
        color="white",
        showfliers=False,
        **boxplot_style
    )
    sns.stripplot(
        ax=axes[1],
        data=inverse_df,
        x="Treatment Type",
        y="StdDev Inverse Flow",
        hue="Group Simple Global",
        palette=global_color_map,
        dodge=False,
        jitter=True,
        alpha=0.8,
        size=6
    )
    xticks_labels_b = [t.get_text() for t in axes[1].get_xticklabels()]
    for _, row in group_median_std.iterrows():
        x_pos = xticks_labels_b.index(row["Treatment Type"])
        group_simple = inverse_df[inverse_df["Group ID"] == row["Group ID"]]["Group Simple Global"].iloc[0]
        color = global_color_map[group_simple]
        x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
        axes[1].scatter(
            x_pos + x_jitter, row["StdDev Inverse Flow"],
            marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
        )
    axes[1].set_title(f"b) StdDev of Inverse Flow by Treatment\n({label_for_title})\nSquares = Group medians")
    axes[1].set_xlabel("Treatment Type")
    axes[1].set_ylabel("StdDev of Inverse Flow [Click Space]")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    ### LEGENDS ###
    if force_paired_legend:
        unique_paired_groups_sorted = sorted(inverse_df["Group ID"].unique())
        paired_group_id_map = {group: f"Group {i+1}" for i, group in enumerate(unique_paired_groups_sorted)}
        legend_order = [paired_group_id_map[group] for group in unique_paired_groups_sorted]
        handles, labels_ = axes[0].get_legend_handles_labels()
        label_to_group_simple = {global_group_id_map[group]: paired_group_id_map[group] for group in unique_paired_groups_sorted}
        new_labels = [label_to_group_simple.get(l, l) for l in labels_]
        new_legend_pairs = sorted(zip(new_labels, handles), key=lambda x: int(x[0].split(" ")[1]))
        new_labels_sorted, new_handles_sorted = zip(*new_legend_pairs)
        axes[0].legend(handles=new_handles_sorted, labels=new_labels_sorted, title="Group", loc='upper right')
        axes[1].legend(handles=new_handles_sorted, labels=new_labels_sorted, title="Group", loc='upper right')
    elif "UNPAIREDGROUPS" in output_prefix:
        axes[0].legend([], [], frameon=False)
        axes[1].legend([], [], frameon=False)
    else:
        axes[0].legend([], [], frameon=False)
        axes[1].legend([], [], frameon=False)

    plt.tight_layout()
    output_path = os.path.join(graph_output_dir, f"boxplot_inverse_flow_sym_asym_{output_prefix}_FINAL_SQUARES.png")
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved {label_for_title} boxplot to: {output_path}")

# === KDE Plot function ===
def plot_kde_sym_asym(inverse_df, label_for_title, output_prefix):
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("Set2", 2)

    for i, ttype in enumerate(["Symmetric", "Asymmetric"]):
        data_subset = inverse_df[inverse_df["Treatment Type"] == ttype]["Mean Inverse Flow"]
        data_subset_log = np.log(data_subset + 1)

        sns.kdeplot(
            data_subset_log,
            label=f"{ttype}",
            linewidth=2,
            fill=True,
            alpha=0.4,
            color=palette[i],
        )
    plt.title(f"Density Plot of log(Inverse Flow + 1)\nby Treatment ({label_for_title})")
    plt.xlabel("log(Inverse Flow + 1)")
    plt.ylabel("Density")
    plt.legend(title="Treatment Type")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = os.path.join(graph_output_dir, f"kde_inverse_flow_sym_asym_{output_prefix}_LOG.png")
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"KDE plot saved to: {output_path}")


# === Run plots ===
plot_boxplot_panels_sym_asym(inverse_flow_data_all, label_for_title="All Groups", output_prefix="ALLGROUPS")
plot_kde_sym_asym(inverse_flow_data_all, label_for_title="All Groups", output_prefix="ALLGROUPS")

plot_boxplot_panels_sym_asym(inverse_flow_data_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS", force_paired_legend=True)
plot_kde_sym_asym(inverse_flow_data_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS")

plot_boxplot_panels_sym_asym(inverse_flow_data_unpaired, label_for_title="Unpaired Groups", output_prefix="UNPAIREDGROUPS")
plot_kde_sym_asym(inverse_flow_data_unpaired, label_for_title="Unpaired Groups", output_prefix="UNPAIREDGROUPS")



#----------------------------------------------
# SYM vs ASYM — Inverse Flow Version — FULL STANDALONE
#----------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Settings ===
graph_output_dir = "Organized Graphs"
os.makedirs(graph_output_dir, exist_ok=True)

# === Load data ===
inverse_flow_df = pd.read_csv("inverse_flow_times_click_space_all_agents.csv")
treatments = pd.read_csv("treatments_summary.csv")

# === Classify treatments: SYM / ASYM ===
def classify_row(row):
    if row["Transparency"] == "No":
        dist = row["Inventory Distribution"].strip()
        if dist == "2, 2, 2, 2, 2":
            return "Symmetric"
        elif dist == "10, 0, 0, 0, 0":
            return "Asymmetric"
    return None

treatments["Treatment Type"] = treatments.apply(classify_row, axis=1)
valid_treatments = treatments.dropna(subset=["Treatment Type"])

# === Merge Treatment Type into inverse_flow_df ===
inverse_flow_df = inverse_flow_df.merge(
    treatments[["Session Code", "Round", "Group", "Treatment Type"]],
    how="left",
    left_on=["Session", "Round", "Group"],
    right_on=["Session Code", "Round", "Group"]
)

# Drop rows with no Treatment Type assigned
inverse_flow_df = inverse_flow_df.dropna(subset=["Treatment Type"])

# === Compute Mean + StdDev of Flux Click Span per Agent ===
records = []
for (session, round_num, group, agent), agent_df in inverse_flow_df.groupby(["Session", "Round", "Group", "Agent"]):
    treatment_type = agent_df["Treatment Type"].iloc[0]

    mean_inverse_flow = agent_df["Flux Click Span"].mean()
    stddev_inverse_flow = agent_df["Flux Click Span"].std()

    records.append({
        "Agent": agent,
        "Group": group,
        "Session": session,
        "Round": round_num,
        "Treatment Type": treatment_type,
        "Mean Inverse Flow": mean_inverse_flow,
        "StdDev Inverse Flow": stddev_inverse_flow,
        "Group ID": f"{session}_{group}"
    })

# === Create dataframe ===
inverse_flow_data = pd.DataFrame(records)

# === Identify Paired Groups ===
pivot = inverse_flow_data.pivot_table(index="Group ID", columns="Treatment Type", values="Mean Inverse Flow", aggfunc='count')
paired_groups = pivot.dropna(subset=["Symmetric", "Asymmetric"]).index.tolist()

# === Split data ===
inverse_flow_data_paired = inverse_flow_data[inverse_flow_data["Group ID"].isin(paired_groups)]
inverse_flow_data_all = inverse_flow_data.copy()
inverse_flow_data_unpaired = inverse_flow_data[~inverse_flow_data["Group ID"].isin(paired_groups)]

# === Global Group mappings — CONSISTENT ===
all_unique_group_ids = sorted(inverse_flow_data["Group ID"].unique())
global_group_id_map = {group: f"Group {i+1}" for i, group in enumerate(all_unique_group_ids)}

# Apply global mapping
inverse_flow_data_all["Group Simple Global"] = inverse_flow_data_all["Group ID"].map(global_group_id_map)
inverse_flow_data_paired["Group Simple Global"] = inverse_flow_data_paired["Group ID"].map(global_group_id_map)
inverse_flow_data_unpaired["Group Simple Global"] = inverse_flow_data_unpaired["Group ID"].map(global_group_id_map)

# === Global color map ===
global_palette_colors = sns.color_palette("tab20", n_colors=len(all_unique_group_ids))
global_color_map = {global_group_id_map[group]: global_palette_colors[i % len(global_palette_colors)]
                    for i, group in enumerate(all_unique_group_ids)}

# === Function to plot boxplots ===
def plot_boxplot_panels_sym_asym(inverse_df, label_for_title, output_prefix, force_paired_legend=False, axes=None):
    create_new_fig = axes is None
    if create_new_fig:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    group_median_mean = inverse_df.groupby(["Group ID", "Treatment Type"])["Mean Inverse Flow"].median().reset_index()
    group_median_std = inverse_df.groupby(["Group ID", "Treatment Type"])["StdDev Inverse Flow"].median().reset_index()

    boxplot_style = dict(
        boxprops=dict(edgecolor='black', linewidth=1),
        whiskerprops=dict(color='black', linewidth=1),
        capprops=dict(color='black', linewidth=1),
        medianprops=dict(color='black', linewidth=1),
        flierprops=dict(marker='o', markersize=3, linestyle='none', color='black', alpha=0.5)
    )

    ### PANEL A ###
    sns.boxplot(
        ax=axes[0],
        data=inverse_df,
        x="Treatment Type",
        y="Mean Inverse Flow",
        color="white",
        showfliers=False,
        **boxplot_style
    )
    sns.stripplot(
        ax=axes[0],
        data=inverse_df,
        x="Treatment Type",
        y="Mean Inverse Flow",
        hue="Group Simple Global",
        palette=global_color_map,
        dodge=False,
        jitter=True,
        alpha=0.8,
        size=6
    )
    xticks_labels_a = [t.get_text() for t in axes[0].get_xticklabels()]
    for _, row in group_median_mean.iterrows():
        x_pos = xticks_labels_a.index(row["Treatment Type"])
        group_simple = inverse_df[inverse_df["Group ID"] == row["Group ID"]]["Group Simple Global"].iloc[0]
        color = global_color_map[group_simple]
        x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
        axes[0].scatter(
            x_pos + x_jitter, row["Mean Inverse Flow"],
            marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
        )
    axes[0].set_title(f"Mean Inverse Flow by Treatment\n({label_for_title})\nSquares = Group medians")
    axes[0].set_xlabel("Treatment Type")
    axes[0].set_ylabel("Mean Inverse Flow [Click Space]")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    ### PANEL B ###
    sns.boxplot(
        ax=axes[1],
        data=inverse_df,
        x="Treatment Type",
        y="StdDev Inverse Flow",
        color="white",
        showfliers=False,
        **boxplot_style
    )
    sns.stripplot(
        ax=axes[1],
        data=inverse_df,
        x="Treatment Type",
        y="StdDev Inverse Flow",
        hue="Group Simple Global",
        palette=global_color_map,
        dodge=False,
        jitter=True,
        alpha=0.8,
        size=6
    )
    xticks_labels_b = [t.get_text() for t in axes[1].get_xticklabels()]
    for _, row in group_median_std.iterrows():
        x_pos = xticks_labels_b.index(row["Treatment Type"])
        group_simple = inverse_df[inverse_df["Group ID"] == row["Group ID"]]["Group Simple Global"].iloc[0]
        color = global_color_map[group_simple]
        x_jitter = 0.01 * (hash(group_simple) % 4 - 2)
        axes[1].scatter(
            x_pos + x_jitter, row["StdDev Inverse Flow"],
            marker='s', s=120, facecolors=color, edgecolors='black', linewidth=1.5, zorder=6
        )
    axes[1].set_title(f"StdDev of Inverse Flow by Treatment\n({label_for_title})\nSquares = Group medians")
    axes[1].set_xlabel("Treatment Type")
    axes[1].set_ylabel("StdDev of Inverse Flow [Click Space]")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    ### LEGENDS ###
    if force_paired_legend:
        unique_paired_groups_sorted = sorted(inverse_df["Group ID"].unique())
        paired_group_id_map = {group: f"Group {i+1}" for i, group in enumerate(unique_paired_groups_sorted)}
        legend_order = [paired_group_id_map[group] for group in unique_paired_groups_sorted]
        handles, labels_ = axes[0].get_legend_handles_labels()
        label_to_group_simple = {global_group_id_map[group]: paired_group_id_map[group] for group in unique_paired_groups_sorted}
        new_labels = [label_to_group_simple.get(l, l) for l in labels_]
        new_legend_pairs = sorted(zip(new_labels, handles), key=lambda x: int(x[0].split(" ")[1]))
        new_labels_sorted, new_handles_sorted = zip(*new_legend_pairs)
        axes[0].legend(handles=new_handles_sorted, labels=new_labels_sorted, title="Group", loc='upper right')
        axes[1].legend(handles=new_handles_sorted, labels=new_labels_sorted, title="Group", loc='upper right')
    elif "UNPAIREDGROUPS" in output_prefix:
        axes[0].legend([], [], frameon=False)
        axes[1].legend([], [], frameon=False)
    else:
        axes[0].legend([], [], frameon=False)
        axes[1].legend([], [], frameon=False)

    if create_new_fig:
        plt.tight_layout()
        output_path = os.path.join(graph_output_dir, f"boxplot_inverse_flow_sym_asym_{output_prefix}_FINAL_SQUARES.png")
        plt.savefig(output_path, dpi=600)
        plt.close()
        print(f"Saved {label_for_title} boxplot to: {output_path}")


def plot_kde_sym_asym(inverse_df, label_for_title, output_prefix, ax=None):
    create_new_fig = ax is None
    if create_new_fig:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

    palette = sns.color_palette("Set2", 2)

    for i, ttype in enumerate(["Symmetric", "Asymmetric"]):
        data_subset = inverse_df[inverse_df["Treatment Type"] == ttype]["Mean Inverse Flow"]
        data_subset_log = np.log(data_subset + 1)

        sns.kdeplot(
            data_subset_log,
            label=f"{ttype}",
            linewidth=2,
            fill=True,
            alpha=0.4,
            color=palette[i],
            ax=ax
        )
    ax.set_title(f"Density Plot of log(Inverse Flow + 1)\n({label_for_title})")
    ax.set_xlabel("log(Inverse Flow + 1)")
    ax.set_ylabel("Density")
    ax.legend(title="Treatment Type")
    ax.grid(True, linestyle="--", alpha=0.4)

    if create_new_fig:
        plt.tight_layout()
        output_path = os.path.join(graph_output_dir, f"kde_inverse_flow_sym_asym_{output_prefix}_LOG.png")
        plt.savefig(output_path, dpi=600)
        plt.close()
        print(f"KDE plot saved to: {output_path}")


def plot_all_panels_grid():
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Row 1: All Groups
    plot_boxplot_panels_sym_asym(inverse_flow_data_all, label_for_title="All Groups", output_prefix="ALLGROUPS", axes=[axes[0,0], axes[0,1]])
    plot_kde_sym_asym(inverse_flow_data_all, label_for_title="All Groups", output_prefix="ALLGROUPS", ax=axes[0,2])

    # Row 2: Paired Groups
    plot_boxplot_panels_sym_asym(inverse_flow_data_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS", force_paired_legend=True, axes=[axes[1,0], axes[1,1]])
    plot_kde_sym_asym(inverse_flow_data_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS", ax=axes[1,2])

    # Row 3: Unpaired Groups
    plot_boxplot_panels_sym_asym(inverse_flow_data_unpaired, label_for_title="Unpaired Groups", output_prefix="UNPAIREDGROUPS", axes=[axes[2,0], axes[2,1]])
    plot_kde_sym_asym(inverse_flow_data_unpaired, label_for_title="Unpaired Groups", output_prefix="UNPAIREDGROUPS", ax=axes[2,2])

    # Add overall panel labels
    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)']
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(panel_labels[i] + " " + ax.get_title(), fontsize=12)

    # Save the full grid
    output_path = os.path.join(graph_output_dir, f"grid_3x3_inverse_flow_FINAL.png")
    plt.savefig(output_path, dpi=600)
    plt.close()
    print(f"Saved full 3x3 grid to: {output_path}")

# === Run all individual plots (optional, can keep for backup) ===
plot_boxplot_panels_sym_asym(inverse_flow_data_all, label_for_title="All Groups", output_prefix="ALLGROUPS")
plot_kde_sym_asym(inverse_flow_data_all, label_for_title="All Groups", output_prefix="ALLGROUPS")

plot_boxplot_panels_sym_asym(inverse_flow_data_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS", force_paired_legend=True)
plot_kde_sym_asym(inverse_flow_data_paired, label_for_title="Paired Groups", output_prefix="PAIREDGROUPS")

plot_boxplot_panels_sym_asym(inverse_flow_data_unpaired, label_for_title="Unpaired Groups", output_prefix="UNPAIREDGROUPS")
plot_kde_sym_asym(inverse_flow_data_unpaired, label_for_title="Unpaired Groups", output_prefix="UNPAIREDGROUPS")

# === Run the full 3x3 grid ===
plot_all_panels_grid()



#----------------------------------------------------------
# SYM vs ASYM — Statistical Tests on Inverse Flow
# Generates full CSV with 9 rows
#----------------------------------------------------------

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, kruskal

# === Helper function ===
def describe_values(series):
    mean_val = series.mean()
    median_val = series.median()
    return f"{mean_val:.3f} ({median_val:.3f})"

# === Prepare tables ===

# Group median click frequency
group_median_mean = inverse_flow_data.groupby(['Group ID', 'Treatment Type'])['Mean Inverse Flow'].median().reset_index()
group_median_std = inverse_flow_data.groupby(['Group ID', 'Treatment Type'])['StdDev Inverse Flow'].median().reset_index()

# Pivot for medians
pivot_median_mean = group_median_mean.pivot(index='Group ID', columns='Treatment Type', values='Mean Inverse Flow')
pivot_median_std = group_median_std.pivot(index='Group ID', columns='Treatment Type', values='StdDev Inverse Flow')

# Identify paired, unpaired
paired_groups_list = pivot_median_mean.dropna(subset=["Symmetric", "Asymmetric"])
paired_group_ids = paired_groups_list.index.tolist()

all_group_ids = pivot_median_mean.index.tolist()
unpaired_group_ids = [g for g in all_group_ids if g not in paired_group_ids]

# Subset unpaired
pivot_median_mean_unpaired = pivot_median_mean.loc[unpaired_group_ids]
pivot_median_std_unpaired = pivot_median_std.loc[unpaired_group_ids]

# === Run tests ===

results = []

# --- Kruskal-Wallis tests on raw Flux Click Span ---

# All groups
flow_sym_all = inverse_flow_df[inverse_flow_df["Treatment Type"] == "Symmetric"]["Flux Click Span"]
flow_asym_all = inverse_flow_df[inverse_flow_df["Treatment Type"] == "Asymmetric"]["Flux Click Span"]

kruskal_all = kruskal(flow_sym_all, flow_asym_all)
results.append({
    "Section": "All groups",
    "Metric": "Raw Flux Click Span",
    "Symmetric (Mean/Median)": describe_values(flow_sym_all),
    "Asymmetric (Mean/Median)": describe_values(flow_asym_all),
    "Test": "Kruskal-Wallis",
    "p-value": f"{kruskal_all.pvalue:.4f}"
})

# Paired groups
flow_sym_paired = inverse_flow_df[(inverse_flow_df["Treatment Type"] == "Symmetric") & (inverse_flow_df["Group ID"].isin(paired_group_ids))]["Flux Click Span"]
flow_asym_paired = inverse_flow_df[(inverse_flow_df["Treatment Type"] == "Asymmetric") & (inverse_flow_df["Group ID"].isin(paired_group_ids))]["Flux Click Span"]

kruskal_paired = kruskal(flow_sym_paired, flow_asym_paired)
results.append({
    "Section": "Paired groups",
    "Metric": "Raw Flux Click Span",
    "Symmetric (Mean/Median)": describe_values(flow_sym_paired),
    "Asymmetric (Mean/Median)": describe_values(flow_asym_paired),
    "Test": "Kruskal-Wallis",
    "p-value": f"{kruskal_paired.pvalue:.4f}"
})

# Unpaired groups
flow_sym_unpaired = inverse_flow_df[(inverse_flow_df["Treatment Type"] == "Symmetric") & (inverse_flow_df["Group ID"].isin(unpaired_group_ids))]["Flux Click Span"]
flow_asym_unpaired = inverse_flow_df[(inverse_flow_df["Treatment Type"] == "Asymmetric") & (inverse_flow_df["Group ID"].isin(unpaired_group_ids))]["Flux Click Span"]

kruskal_unpaired = kruskal(flow_sym_unpaired, flow_asym_unpaired)
results.append({
    "Section": "Unpaired groups",
    "Metric": "Raw Flux Click Span",
    "Symmetric (Mean/Median)": describe_values(flow_sym_unpaired),
    "Asymmetric (Mean/Median)": describe_values(flow_asym_unpaired),
    "Test": "Kruskal-Wallis",
    "p-value": f"{kruskal_unpaired.pvalue:.4f}"
})

# --- Paired groups — Wilcoxon ---

# Group Median of Mean Inverse Flow
wilcoxon_mean_paired = wilcoxon(paired_groups_list["Symmetric"], paired_groups_list["Asymmetric"])
results.append({
    "Section": "Paired groups",
    "Metric": "Group Median of Mean Inverse Flow",
    "Symmetric (Mean/Median)": describe_values(paired_groups_list["Symmetric"]),
    "Asymmetric (Mean/Median)": describe_values(paired_groups_list["Asymmetric"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_mean_paired.pvalue:.4f}"
})

# Group Median of StdDev Inverse Flow
paired_std = pivot_median_std.loc[paired_group_ids]
wilcoxon_std_paired = wilcoxon(paired_std["Symmetric"], paired_std["Asymmetric"])
results.append({
    "Section": "Paired groups",
    "Metric": "Group Median of StdDev Inverse Flow",
    "Symmetric (Mean/Median)": describe_values(paired_std["Symmetric"]),
    "Asymmetric (Mean/Median)": describe_values(paired_std["Asymmetric"]),
    "Test": "Wilcoxon",
    "p-value": f"{wilcoxon_std_paired.pvalue:.4f}"
})

# --- All groups — Mann-Whitney ---

# Group Median of Mean Inverse Flow
mann_mean_all = mannwhitneyu(pivot_median_mean["Symmetric"].dropna(), pivot_median_mean["Asymmetric"].dropna(), alternative='two-sided')
results.append({
    "Section": "All groups",
    "Metric": "Group Median of Mean Inverse Flow",
    "Symmetric (Mean/Median)": describe_values(pivot_median_mean["Symmetric"].dropna()),
    "Asymmetric (Mean/Median)": describe_values(pivot_median_mean["Asymmetric"].dropna()),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_mean_all.pvalue:.4f}"
})

# Group Median of StdDev Inverse Flow
mann_std_all = mannwhitneyu(pivot_median_std["Symmetric"].dropna(), pivot_median_std["Asymmetric"].dropna(), alternative='two-sided')
results.append({
    "Section": "All groups",
    "Metric": "Group Median of StdDev Inverse Flow",
    "Symmetric (Mean/Median)": describe_values(pivot_median_std["Symmetric"].dropna()),
    "Asymmetric (Mean/Median)": describe_values(pivot_median_std["Asymmetric"].dropna()),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_std_all.pvalue:.4f}"
})

# --- Unpaired groups — Mann-Whitney ---

# Group Median of Mean Inverse Flow
mann_mean_unpaired = mannwhitneyu(pivot_median_mean_unpaired["Symmetric"].dropna(), pivot_median_mean_unpaired["Asymmetric"].dropna(), alternative='two-sided')
results.append({
    "Section": "Unpaired groups",
    "Metric": "Group Median of Mean Inverse Flow",
    "Symmetric (Mean/Median)": describe_values(pivot_median_mean_unpaired["Symmetric"].dropna()),
    "Asymmetric (Mean/Median)": describe_values(pivot_median_mean_unpaired["Asymmetric"].dropna()),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_mean_unpaired.pvalue:.4f}"
})

# Group Median of StdDev Inverse Flow
mann_std_unpaired = mannwhitneyu(pivot_median_std_unpaired["Symmetric"].dropna(), pivot_median_std_unpaired["Asymmetric"].dropna(), alternative='two-sided')
results.append({
    "Section": "Unpaired groups",
    "Metric": "Group Median of StdDev Inverse Flow",
    "Symmetric (Mean/Median)": describe_values(pivot_median_std_unpaired["Symmetric"].dropna()),
    "Asymmetric (Mean/Median)": describe_values(pivot_median_std_unpaired["Asymmetric"].dropna()),
    "Test": "Mann-Whitney",
    "p-value": f"{mann_std_unpaired.pvalue:.4f}"
})

# === Save results ===

results_df = pd.DataFrame(results)
print(results_df)

# Save to CSV
results_df.to_csv(os.path.join(graph_output_dir, "inverse_flow_statistics_table_SYM_ASYM_FINAL.csv"), index=False)
print("Statistics table saved to inverse_flow_statistics_table_SYM_ASYM_FINAL.csv.")









#############################################################################
# H8: Learning within rounds
#############################################################################



import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon

# === H7 refined ===
print("Running H7 analysis (10s bins, group-level)...")

csv_files = glob.glob("treatment_outputs/*.csv")

h7_records = []

time_bin_edges = np.arange(0, 190, 10)  # 0 to 180 in 10s bins
time_bin_labels = [f"{time_bin_edges[i]}-{time_bin_edges[i+1]}" for i in range(len(time_bin_edges)-1)]

for file in csv_files:
    df = pd.read_csv(file)
    
    # Extract identifiers from filename
    fname = os.path.basename(file)
    session = fname.split("session")[1].split("_")[0]
    round_ = int(fname.split("round")[1].split("_")[0])
    group = fname.split("group")[1].split("_")[0]
    
    df['Time Bin'] = pd.cut(df['Time Step'], bins=time_bin_edges, labels=range(len(time_bin_labels)), right=False)
    
    # Sum Requested per Time Bin (group level)
    group_clicks = df.groupby('Time Bin')['Requested'].sum().reset_index()
    
    # Store bin-level data for plotting later
    for _, row in group_clicks.iterrows():
        h7_records.append({
            'Session': session,
            'Round': round_,
            'Group': group,
            'Time Bin Index': row['Time Bin'],
            'Clicks': row['Requested']
        })

# Combine data
h7_df = pd.DataFrame(h7_records)

# === Now: compute Spearman rho for each round ===

spearman_results = []

for (session, round_, group), group_df in h7_df.groupby(['Session', 'Round', 'Group']):
    # Prepare series for correlation
    time_bin_idx = group_df['Time Bin Index']
    clicks = group_df['Clicks']
    
    # Only compute if there is variation
    if clicks.nunique() > 1:
        rho, pval = spearmanr(time_bin_idx, clicks)
    else:
        rho, pval = np.nan, np.nan  # no variation
    
    spearman_results.append({
        'Session': session,
        'Round': round_,
        'Group': group,
        'Spearman Rho': rho,
        'Spearman P': pval
    })

# Combine Spearman results
spearman_df = pd.DataFrame(spearman_results)

# Save Spearman results
spearman_df.to_csv("H7_spearman_rho_per_round.csv", index=False)
print("H7 Spearman results saved: H7_spearman_rho_per_round.csv")

# === Plot example rounds ===

# Pick 6 example rounds to visualize (optional: random or handpick)
example_rounds = spearman_df.sort_values('Spearman Rho', ascending=False).dropna().head(3).append(
    spearman_df.sort_values('Spearman Rho', ascending=True).dropna().head(3)
)

plt.figure(figsize=(12,8))
for i, row in enumerate(example_rounds.itertuples()):
    example_df = h7_df[
        (h7_df['Session'] == row.Session) &
        (h7_df['Round'] == row.Round) &
        (h7_df['Group'] == row.Group)
    ]
    plt.subplot(2,3,i+1)
    plt.plot(example_df['Time Bin Index'], example_df['Clicks'], marker='o')
    plt.title(f"Sess {row.Session} Round {row.Round} Group {row.Group}\nRho={row._4:.2f} p={row._5:.3f}")
    plt.xlabel("Time Bin Index")
    plt.ylabel("Clicks")
    plt.grid(True, linestyle='--', alpha=0.5)
    
plt.tight_layout()
plt.savefig("H7_example_rounds_clicks_over_time.png", dpi=300)
plt.close()
print("H7 example rounds plot saved: H7_example_rounds_clicks_over_time.png")

# === Global analysis: is rho > 0 across rounds? ===

# Keep only valid rho values
valid_rho = spearman_df['Spearman Rho'].dropna()

# Plot distribution of rho
plt.figure(figsize=(8,6))
sns.histplot(valid_rho, kde=True, bins=20)
plt.axvline(0, color='red', linestyle='--')
plt.title("H7: Distribution of Spearman Rho across Rounds")
plt.xlabel("Spearman Rho (Clicks vs Time Bin Index)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("H7_rho_distribution.png", dpi=300)
plt.close()
print("H7 Rho distribution plot saved: H7_rho_distribution.png")

# Statistical test: is median rho > 0?
# Wilcoxon signed-rank test vs 0
stat, pval = wilcoxon(valid_rho - 0)

print(f"H7 Wilcoxon test: stat={stat:.3f}, p-value={pval:.4f}")
print(f"Median Spearman Rho across rounds: {np.median(valid_rho):.3f}")

# === Build H7 summary table ===

summary_counts = {
    'Significant positive trend': ((spearman_df['Spearman Rho'] > 0) & (spearman_df['Spearman P'] < 0.05)).sum(),
    'Significant negative trend': ((spearman_df['Spearman Rho'] < 0) & (spearman_df['Spearman P'] < 0.05)).sum(),
    'Non-significant trend': ((spearman_df['Spearman P'] >= 0.05) | (spearman_df['Spearman P'].isna())).sum()
}

# Print summary nicely
print("\n=== H7 Summary Table ===")
for k, v in summary_counts.items():
    print(f"{k}: {v} rounds")

print(f"Total rounds: {sum(summary_counts.values())}")

################################################ FORMATTED PLOTS

# === Plot example rounds ===

example_rounds = spearman_df.sort_values('Spearman Rho', ascending=False).dropna().head(3).append(
    spearman_df.sort_values('Spearman Rho', ascending=True).dropna().head(3)
)

# Marker & color settings
grey = "#444444"
marker = 'o'

plt.figure(figsize=(12,8))
for i, row in enumerate(example_rounds.itertuples()):
    example_df = h7_df[
        (h7_df['Session'] == row.Session) &
        (h7_df['Round'] == row.Round) &
        (h7_df['Group'] == row.Group)
    ]
    plt.subplot(2,3,i+1)
    # Line
    plt.plot(example_df['Time Bin Index'], example_df['Clicks'], color=grey, linewidth=2)
    # Points
    plt.scatter(example_df['Time Bin Index'], example_df['Clicks'],
                facecolors='white', edgecolors=grey, marker=marker, linewidth=1.5, s=20)
    
    plt.title(f"Sess {row.Session} Round {row.Round} Group {row.Group}\nRho={row._4:.2f} p={row._5:.3f}")
    plt.xlabel("Time Bin Index")
    plt.ylabel("Total Clicks (Group)")
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("H7_example_rounds_clicks_over_time.png", dpi=300)
plt.close()
print("H7 example rounds plot saved: H7_example_rounds_clicks_over_time.png")



# === Global analysis: is rho > 0 across rounds? ===

valid_rho = spearman_df['Spearman Rho'].dropna()

# Statistical test
stat, pval = wilcoxon(valid_rho - 0)
median_rho = np.median(valid_rho)

# Summary counts for annotation
summary_counts = {
    'Significant positive trend': ((spearman_df['Spearman Rho'] > 0) & (spearman_df['Spearman P'] < 0.05)).sum(),
    'Significant negative trend': ((spearman_df['Spearman Rho'] < 0) & (spearman_df['Spearman P'] < 0.05)).sum(),
    'Non-significant trend': ((spearman_df['Spearman P'] >= 0.05) | (spearman_df['Spearman P'].isna())).sum()
}

# Plot rho distribution
plt.figure(figsize=(8,6))
grey = "#444444"
sns.histplot(valid_rho, kde=True, bins=20, color=grey, edgecolor='black')
plt.axvline(0, color='black', linestyle='--')
plt.title("Distribution of Spearman Rho across Rounds")
plt.xlabel("Spearman Rho (Clicks vs Time Bin Index)")
plt.ylabel("Count")

# Add annotation box
textstr = '\n'.join((
    f"Significant positive: {summary_counts['Significant positive trend']} rounds",
    f"Non-significant: {summary_counts['Non-significant trend']} rounds",
    f"Significant negative: {summary_counts['Significant negative trend']} rounds",
    f"Median rho = {median_rho:.3f}",
    f"Wilcoxon p = {pval:.3f}"
))
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
plt.gca().text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig("H7_rho_distribution.png", dpi=300)
plt.close()
print("H7 Rho distribution plot saved: H7_rho_distribution.png")


#----------------------------------------------
# Statistics saved
#-----------------------------------------------



# === Save H7 summary table to CSV ===

# Convert summary_counts to DataFrame
summary_df = pd.DataFrame(list(summary_counts.items()), columns=['Trend Type', 'Number of Rounds'])

# Add total row
summary_df.loc[len(summary_df)] = ['Total rounds', sum(summary_counts.values())]

# Save to CSV
summary_df.to_csv("H7_summary_table.csv", index=False)
print("H7 summary table saved: H7_summary_table.csv")



########################################## ADDITIONAL CONTROL



#------------------------------------------------------
# H7 Control Analysis — Final Version with Annotations
# Plots + significance tables by Session and Treatment
#------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
spearman_df = pd.read_csv("H7_spearman_rho_per_round.csv")
treatments_summary = pd.read_csv("treatments_summary.csv")

# === Prepare merge ===
# Make Group columns string
spearman_df['Group'] = spearman_df['Group'].astype(str)
treatments_summary['Group'] = treatments_summary['Group'].astype(str)

# Merge Treatment info
spearman_df = spearman_df.merge(
    treatments_summary[['Session Code', 'Round', 'Group', 'Transparency', 'Inventory Distribution']],
    left_on=['Session', 'Round', 'Group'],
    right_on=['Session Code', 'Round', 'Group'],
    how='left'
)

# Drop missing rows
spearman_df = spearman_df.dropna(subset=['Transparency', 'Inventory Distribution'])

# Create Treatment Label
spearman_df['Treatment Label'] = spearman_df['Transparency'].astype(str) + " | " + spearman_df['Inventory Distribution'].astype(str)

# === Session significance summary ===
session_summary = []
for session_name, group_df in spearman_df.groupby('Session'):
    num_rounds = len(group_df)
    num_pos = ((group_df['Spearman Rho'] > 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_neg = ((group_df['Spearman Rho'] < 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_nonsig = ((group_df['Spearman P'] >= 0.05) | (group_df['Spearman P'].isna())).sum()
    
    session_summary.append({
        'Session': session_name,
        'Num Rounds': num_rounds,
        'Num Significant Positive': num_pos,
        'Num Significant Negative': num_neg,
        'Num Non-significant': num_nonsig
    })

session_summary_df = pd.DataFrame(session_summary)
session_summary_df.to_csv("H7_session_significance_summary.csv", index=False)
print("Saved H7_session_significance_summary.csv")

# === Treatment significance summary ===
treatment_summary = []
for treatment_label, group_df in spearman_df.groupby('Treatment Label'):
    num_rounds = len(group_df)
    num_pos = ((group_df['Spearman Rho'] > 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_neg = ((group_df['Spearman Rho'] < 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_nonsig = ((group_df['Spearman P'] >= 0.05) | (group_df['Spearman P'].isna())).sum()
    
    treatment_summary.append({
        'Treatment Label': treatment_label,
        'Num Rounds': num_rounds,
        'Num Significant Positive': num_pos,
        'Num Significant Negative': num_neg,
        'Num Non-significant': num_nonsig
    })

treatment_summary_df = pd.DataFrame(treatment_summary)
treatment_summary_df.to_csv("H7_treatment_significance_summary.csv", index=False)
print("Saved H7_treatment_significance_summary.csv")

# === Build annotation dicts ===
# For Session
session_annotations = {}
for row in session_summary_df.itertuples():
    text = f"{row._3}/{row._2} pos"
    session_annotations[row.Session] = text

# For Treatment Label
treatment_annotations = {}
for row in treatment_summary_df.itertuples():
    text = f"{row._3}/{row._2} pos"
    treatment_annotations[row._1] = text  # Treatment Label

# === Boxplot: Rho by Session with annotation ===
plt.figure(figsize=(8,6))
ax = sns.boxplot(x='Session', y='Spearman Rho', data=spearman_df, color='#cccccc')
plt.axhline(0, color='black', linestyle='--')
plt.title("Spearman Rho across Sessions")
plt.ylabel("Spearman Rho (Clicks vs Time Bin Index)")
plt.xlabel("Session")

# Update x-tick labels with annotation
xticklabels = [f"{tick.get_text()}\n{session_annotations.get(tick.get_text(), '')}" for tick in ax.get_xticklabels()]
ax.set_xticklabels(xticklabels)

plt.tight_layout()
plt.savefig("H7_rho_by_session_annotated.png", dpi=300)
plt.close()
print("H7 Rho by Session (annotated) plot saved.")

# === Boxplot: Rho by Treatment Label with annotation ===
plt.figure(figsize=(12,6))
ax = sns.boxplot(x='Treatment Label', y='Spearman Rho', data=spearman_df, color='#cccccc')
plt.axhline(0, color='black', linestyle='--')
plt.title("Spearman Rho across Treatment Labels")
plt.ylabel("Spearman Rho (Clicks vs Time Bin Index)")
plt.xlabel("Treatment (Transparency | Inventory Distribution)")

# Update x-tick labels with annotation
xticklabels = [f"{tick.get_text()}\n{treatment_annotations.get(tick.get_text(), '')}" for tick in ax.get_xticklabels()]
ax.set_xticklabels(xticklabels)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("H7_rho_by_treatment_label_annotated.png", dpi=300)
plt.close()
print("H7 Rho by Treatment Label (annotated) plot saved.")

print("\n=== DONE === H7 Control Analysis with Annotations completed ===")






#--------------------------------------------------
# H7 Control Analysis — FINAL CLEANED FULL STANDALONE
# Nice spacing between panels, normal titles, wider example grid
#--------------------------------------------------

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
import matplotlib.gridspec as gridspec

# === Settings ===
print("Running FULL STANDALONE H7 analysis (10s bins, group-level)...")

# === Load data ===
csv_files = glob.glob("treatment_outputs/*.csv")
treatments_summary = pd.read_csv("treatments_summary.csv")

# === Compute h7_df ===
h7_records = []

time_bin_edges = np.arange(0, 190, 10)  # 0 to 180 in 10s bins
time_bin_labels = [f"{time_bin_edges[i]}-{time_bin_edges[i+1]}" for i in range(len(time_bin_edges)-1)]

for file in csv_files:
    df = pd.read_csv(file)
    
    # Extract identifiers from filename
    fname = file.split("/")[-1]
    session = fname.split("session")[1].split("_")[0]
    round_ = int(fname.split("round")[1].split("_")[0])
    group = fname.split("group")[1].split("_")[0]
    
    df['Time Bin'] = pd.cut(df['Time Step'], bins=time_bin_edges, labels=range(len(time_bin_labels)), right=False)
    
    # Sum Requested per Time Bin (group level)
    group_clicks = df.groupby('Time Bin')['Requested'].sum().reset_index()
    
    for _, row in group_clicks.iterrows():
        h7_records.append({
            'Session': session,
            'Round': round_,
            'Group': group,
            'Time Bin Index': row['Time Bin'],
            'Clicks': row['Requested']
        })

h7_df = pd.DataFrame(h7_records)

# === Compute Spearman rho per round ===
spearman_results = []

for (session, round_, group), group_df in h7_df.groupby(['Session', 'Round', 'Group']):
    time_bin_idx = group_df['Time Bin Index']
    clicks = group_df['Clicks']
    
    if clicks.nunique() > 1:
        rho, pval = spearmanr(time_bin_idx, clicks)
    else:
        rho, pval = np.nan, np.nan
    
    spearman_results.append({
        'Session': session,
        'Round': round_,
        'Group': group,
        'Spearman Rho': rho,
        'Spearman P': pval
    })

spearman_df = pd.DataFrame(spearman_results)

# === Merge Treatment info ===
spearman_df['Group'] = spearman_df['Group'].astype(str)
treatments_summary['Group'] = treatments_summary['Group'].astype(str)

spearman_df = spearman_df.merge(
    treatments_summary[['Session Code', 'Round', 'Group', 'Transparency', 'Inventory Distribution']],
    left_on=['Session', 'Round', 'Group'],
    right_on=['Session Code', 'Round', 'Group'],
    how='left'
)

spearman_df = spearman_df.dropna(subset=['Transparency', 'Inventory Distribution'])

spearman_df['Treatment Label'] = spearman_df['Transparency'].astype(str) + " | " + spearman_df['Inventory Distribution'].astype(str)

# === Global analysis ===
valid_rho = spearman_df['Spearman Rho'].dropna()
stat, pval = wilcoxon(valid_rho - 0)
median_rho = np.median(valid_rho)

# === Session significance summary ===
session_summary = []
for session_name, group_df in spearman_df.groupby('Session'):
    num_rounds = len(group_df)
    num_pos = ((group_df['Spearman Rho'] > 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_neg = ((group_df['Spearman Rho'] < 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_nonsig = ((group_df['Spearman P'] >= 0.05) | (group_df['Spearman P'].isna())).sum()
    
    session_summary.append({
        'Session': session_name,
        'Num Rounds': num_rounds,
        'Num Significant Positive': num_pos,
        'Num Significant Negative': num_neg,
        'Num Non-significant': num_nonsig
    })

session_summary_df = pd.DataFrame(session_summary)
session_summary_df.to_csv("H7_session_significance_summary.csv", index=False)
print("Saved H7_session_significance_summary.csv")

# === Treatment significance summary ===
treatment_summary = []
for treatment_label, group_df in spearman_df.groupby('Treatment Label'):
    num_rounds = len(group_df)
    num_pos = ((group_df['Spearman Rho'] > 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_neg = ((group_df['Spearman Rho'] < 0) & (group_df['Spearman P'] < 0.05)).sum()
    num_nonsig = ((group_df['Spearman P'] >= 0.05) | (group_df['Spearman P'].isna())).sum()
    
    treatment_summary.append({
        'Treatment Label': treatment_label,
        'Num Rounds': num_rounds,
        'Num Significant Positive': num_pos,
        'Num Significant Negative': num_neg,
        'Num Non-significant': num_nonsig
    })

treatment_summary_df = pd.DataFrame(treatment_summary)
treatment_summary_df.to_csv("H7_treatment_significance_summary.csv", index=False)
print("Saved H7_treatment_significance_summary.csv")

# === Build annotation dicts ===
session_annotations = {}
for row in session_summary_df.itertuples():
    text = f"{row._3}/{row._2} pos"
    session_annotations[row.Session] = text

treatment_annotations = {}
for row in treatment_summary_df.itertuples():
    text = f"{row._3}/{row._2} pos"
    treatment_annotations[row._1] = text

# === Example rounds ===
example_rounds = spearman_df.sort_values('Spearman Rho', ascending=False).dropna().head(3).append(
    spearman_df.sort_values('Spearman Rho', ascending=True).dropna().head(3)
)

# === Final combined 4-panel grid ===
fig = plt.figure(figsize=(18, 16))  # bigger figure to make everything breathe
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.8, 1.2], width_ratios=[1,1])  # more vertical and horizontal space

# Panel a)
gs_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0, 0], wspace=0.5, hspace=0.28)  # MORE space between 6 example subplots
for i, row in enumerate(example_rounds.itertuples()):
    ax = fig.add_subplot(gs_a[i])
    example_df = h7_df[
        (h7_df['Session'] == row.Session) &
        (h7_df['Round'] == row.Round) &
        (h7_df['Group'] == row.Group)
    ]
    ax.plot(example_df['Time Bin Index'], example_df['Clicks'], color="#444444", linewidth=2)
    ax.scatter(example_df['Time Bin Index'], example_df['Clicks'],
               facecolors='white', edgecolors="#444444", marker='o', linewidth=1.5, s=20)
    ax.set_title(f"Sess {row.Session} Round {row.Round} Group {row.Group}\nRho={row._4:.2f} p={row._5:.3f}", fontsize=9)
    ax.set_xlabel("Time Bin Index", fontsize=9)
    ax.set_ylabel("Total Clicks (Group)", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(False)

fig.text(0.27, 0.945, "a) Example rounds", fontsize=14, fontweight='normal', ha='center')

# Panel b)
ax_b = fig.add_subplot(gs[0,1])
sns.histplot(valid_rho, kde=True, bins=20, color="#444444", edgecolor='black', ax=ax_b)
ax_b.axvline(0, color='black', linestyle='--')
ax_b.set_title("b) Rho distribution (Wilcoxon)", fontsize=14, fontweight='normal')
ax_b.set_xlabel("Spearman Rho (Clicks vs Time Bin Index)")
ax_b.set_ylabel("Count")

textstr = '\n'.join((
    f"Significant positive: {((spearman_df['Spearman Rho'] > 0) & (spearman_df['Spearman P'] < 0.05)).sum()} rounds",
    f"Non-significant: {((spearman_df['Spearman P'] >= 0.05) | (spearman_df['Spearman P'].isna())).sum()} rounds",
    f"Significant negative: {((spearman_df['Spearman Rho'] < 0) & (spearman_df['Spearman P'] < 0.05)).sum()} rounds",
    f"Median rho = {median_rho:.3f}",
    f"Wilcoxon p = {pval:.3f}"
))
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
ax_b.text(0.02, 0.95, textstr, transform=ax_b.transAxes, fontsize=10, verticalalignment='top', bbox=props)

# Panel c)
ax_c = fig.add_subplot(gs[1,0])
sns.boxplot(x='Treatment Label', y='Spearman Rho', data=spearman_df, color='#cccccc', ax=ax_c)
ax_c.axhline(0, color='black', linestyle='--')
ax_c.set_title("c) Rho by Treatment", fontsize=14, fontweight='normal')
ax_c.set_ylabel("Spearman Rho (Clicks vs Time Bin Index)")
ax_c.set_xlabel("Treatment (Transparency | Inventory Distribution)")
xticklabels = [f"{tick.get_text()}\n{treatment_annotations.get(tick.get_text(), '')}" for tick in ax_c.get_xticklabels()]
ax_c.set_xticklabels(xticklabels, rotation=45, ha='right')

# Panel d)
ax_d = fig.add_subplot(gs[1,1])
sns.boxplot(x='Session', y='Spearman Rho', data=spearman_df, color='#cccccc', ax=ax_d)
ax_d.axhline(0, color='black', linestyle='--')
ax_d.set_title("d) Rho by Session", fontsize=14, fontweight='normal')
ax_d.set_ylabel("Spearman Rho (Clicks vs Time Bin Index)")
ax_d.set_xlabel("Session")
xticklabels = [f"{tick.get_text()}\n{session_annotations.get(tick.get_text(), '')}" for tick in ax_d.get_xticklabels()]
ax_d.set_xticklabels(xticklabels)

# Save combined plot
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("H7_final_4panel_grid_examples6.png", dpi=600)
plt.close()

print("\n=== DONE === Saved H7_final_4panel_grid_examples6.png")


######### Significance numbers for report

# === Group-level median Rho (per Group ID across rounds) ===
spearman_df['Group ID'] = spearman_df['Session'].astype(str) + "_" + spearman_df['Group'].astype(str)

group_level_summary = spearman_df.groupby('Group ID')['Spearman Rho'].median().reset_index()
group_level_summary = group_level_summary.rename(columns={'Spearman Rho': 'Median Spearman Rho (Group level)'})

# Save to CSV if you want:
group_level_summary.to_csv("H7_group_level_median_rho.csv", index=False)
print("Saved H7_group_level_median_rho.csv")

# You can also check overall how many Groups have median rho > 0 etc:
num_groups_pos = (group_level_summary['Median Spearman Rho (Group level)'] > 0).sum()
num_groups_neg = (group_level_summary['Median Spearman Rho (Group level)'] < 0).sum()
num_groups_zero = (group_level_summary['Median Spearman Rho (Group level)'] == 0).sum()

print("\n=== H7 Group-level Median Rho Summary ===")
print(f"Groups with positive median rho: {num_groups_pos}")
print(f"Groups with negative median rho: {num_groups_neg}")
print(f"Groups with median rho = 0: {num_groups_zero}")
print(f"Total groups: {group_level_summary.shape[0]}")



# --------------------------------------
# H8 Consistency Test for Group, Treatment, Session
# --------------------------------------

import pandas as pd

# Load your existing H8 rho dataframe
spearman_df = pd.read_csv("H7_spearman_rho_per_round.csv")

# Also merge Treatment info
treatments_summary = pd.read_csv("treatments_summary.csv")
spearman_df['Group'] = spearman_df['Group'].astype(str)
treatments_summary['Group'] = treatments_summary['Group'].astype(str)

spearman_df = spearman_df.merge(
    treatments_summary[['Session Code', 'Round', 'Group', 'Transparency', 'Inventory Distribution']],
    left_on=['Session', 'Round', 'Group'],
    right_on=['Session Code', 'Round', 'Group'],
    how='left'
)

spearman_df = spearman_df.dropna(subset=['Transparency', 'Inventory Distribution'])
spearman_df['Treatment Label'] = spearman_df['Transparency'].astype(str) + " | " + spearman_df['Inventory Distribution'].astype(str)

# --------------------------------------------------------
# Function to compute consistency summary for any grouping
# --------------------------------------------------------

def compute_consistency(groupby_field, field_name_for_print):

    consistency_results = []

    for group_id, group_df in spearman_df.groupby(groupby_field):
        num_rounds = len(group_df)
        num_pos_sig = ((group_df['Spearman Rho'] > 0) & (group_df['Spearman P'] < 0.05)).sum()
        num_neg_sig = ((group_df['Spearman Rho'] < 0) & (group_df['Spearman P'] < 0.05)).sum()

        all_pos_sig = (num_pos_sig == num_rounds) and (num_rounds > 0)
        all_neg_sig = (num_neg_sig == num_rounds) and (num_rounds > 0)

        consistency_results.append({
            field_name_for_print: group_id,
            'Num Rounds': num_rounds,
            'Num Significant Positive': num_pos_sig,
            'Num Significant Negative': num_neg_sig,
            'All Positive': all_pos_sig,
            'All Negative': all_neg_sig
        })

    consistency_df = pd.DataFrame(consistency_results)
    return consistency_df

# --------------------------------
# Run for Group
# --------------------------------

print("\n=== H8 Consistency by Group ===")
group_consistency_df = compute_consistency(groupby_field=['Session', 'Group'], field_name_for_print='Session_Group')
group_consistency_df.to_csv("H8_group_consistency_summary.csv", index=False)

print(group_consistency_df[['Session_Group', 'All Positive', 'All Negative']])
num_all_pos_groups = group_consistency_df['All Positive'].sum()
num_all_neg_groups = group_consistency_df['All Negative'].sum()

print(f"\nGroups with ALL rounds significantly positive: {num_all_pos_groups}")
print(f"Groups with ALL rounds significantly negative: {num_all_neg_groups}")

# --------------------------------
# Run for Treatment
# --------------------------------

print("\n=== H8 Consistency by Treatment ===")
treatment_consistency_df = compute_consistency(groupby_field='Treatment Label', field_name_for_print='Treatment Label')
treatment_consistency_df.to_csv("H8_treatment_consistency_summary.csv", index=False)

print(treatment_consistency_df[['Treatment Label', 'All Positive', 'All Negative']])
num_all_pos_treatments = treatment_consistency_df['All Positive'].sum()
num_all_neg_treatments = treatment_consistency_df['All Negative'].sum()

print(f"\nTreatments with ALL rounds significantly positive: {num_all_pos_treatments}")
print(f"Treatments with ALL rounds significantly negative: {num_all_neg_treatments}")

# --------------------------------
# Run for Session
# --------------------------------

print("\n=== H8 Consistency by Session ===")
session_consistency_df = compute_consistency(groupby_field='Session', field_name_for_print='Session')
session_consistency_df.to_csv("H8_session_consistency_summary.csv", index=False)

print(session_consistency_df[['Session', 'All Positive', 'All Negative']])
num_all_pos_sessions = session_consistency_df['All Positive'].sum()
num_all_neg_sessions = session_consistency_df['All Negative'].sum()

print(f"\nSessions with ALL rounds significantly positive: {num_all_pos_sessions}")
print(f"Sessions with ALL rounds significantly negative: {num_all_neg_sessions}")

# --------------------------------
print("\n=== DONE === H8 Consistency Test completed. CSVs saved.")







###############################################
# H9: LEARNING ACROSS ROUNDS
# Deviation from Treatment Median Total Balance
# Plot 6 example groups (exactly 3 rounds)
###############################################

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import matplotlib.gridspec as gridspec

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})


# === Load treatments summary ===
treatments_summary = pd.read_csv("treatments_summary.csv")

# === Load treatment_outputs ===
csv_files = glob.glob("treatment_outputs/*.csv")
records = []

for file in csv_files:
    df = pd.read_csv(file)

    # Extract identifiers
    fname = os.path.basename(file)
    session = fname.split("session")[1].split("_")[0]
    round_ = int(fname.split("round")[1].split("_")[0])
    group = fname.split("group")[1].split("_")[0]

    # Match treatment info
    treatment_row = treatments_summary[
        (treatments_summary['Session Code'] == session) &
        (treatments_summary['Round'] == round_) &
        (treatments_summary['Group'].astype(str) == group)
    ]
    if treatment_row.empty:
        continue

    transparency = treatment_row['Transparency'].iloc[0]
    inventory_dist = treatment_row['Inventory Distribution'].iloc[0]
    treatment_label = f"{transparency} | {inventory_dist}"

    # Use only end-of-round data
    max_time = df['Time Step'].max()
    end_df = df[df['Time Step'] == max_time]

    if end_df['Agent'].nunique() < 2:
        continue

    total_balance = end_df['Balance'].sum()

    records.append({
        "Session": str(session),
        "Group": str(group),
        "Round": int(round_),
        "Total Balance": float(total_balance),
        "Treatment Label": treatment_label
    })

# === Create dataframe ===
df = pd.DataFrame(records)
df['Group ID'] = df['Session'] + "_" + df['Group']

# === Compute treatment median ===
treatment_median_df = df.groupby('Treatment Label')['Total Balance'].median().reset_index()
treatment_median_df = treatment_median_df.rename(columns={'Total Balance': 'Treatment Median Balance'})

# === Merge treatment median into df ===
df = df.merge(treatment_median_df, on='Treatment Label', how='left')

# === Compute Deviation ===
df['Deviation from Median'] = df['Total Balance'] - df['Treatment Median Balance']

# === Select only groups with exactly 3 rounds ===
group_counts = df.groupby('Group ID')['Round'].nunique().reset_index()
valid_groups = group_counts[group_counts['Round'] == 3]['Group ID']
df_valid = df[df['Group ID'].isin(valid_groups)]

# === Select 6 example groups ===
example_groups = df_valid['Group ID'].drop_duplicates().sample(n=6, random_state=42)

# === Plot in H7 style ===
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.35)

# === Print selected example groups ===
print("\nSelected example groups (Session X Group):")
for gid in example_groups:
    session, group = gid.split("_")
    print(f"Session {session} Group {group}")

# === Plot each group ===
for i, group_id in enumerate(example_groups):
    ax = fig.add_subplot(gs[i])
    session, group = group_id.split("_")
    subset = df_valid[df_valid['Group ID'] == group_id].sort_values(by='Round')

    # Plot line
    ax.plot(subset['Round'], subset['Deviation from Median'], color="#444444", linewidth=2)

    # Plot scatter points
    ax.scatter(subset['Round'], subset['Deviation from Median'],
               facecolors='white', edgecolors="#444444", marker='o', linewidth=1.5, s=50)

    # Subplot title → regular font, size 11
    ax.set_title(f"Session {session} Group {group}", fontsize=11, fontweight='normal')

    # Axis labels → regular font, size 10
    ax.set_xlabel("Round", fontsize=10, fontweight='normal')
    ax.set_ylabel("Normalized End Balance of Group", fontsize=10, fontweight='normal')

    # Ticks → font size 9
    ax.tick_params(axis='both', labelsize=9)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.5)

# === Main figure title → regular font, size 14 ===
fig.text(0.5, 0.94, "a) Example groups: Normalized End Balance of Group by Round",
         fontsize=14, fontweight='normal', ha='center')

# === Save ===
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("H8_example_groups_deviation_from_treatment_median.png", dpi=300)
plt.close()

print("Saved H8_example_groups_deviation_from_treatment_median.png")


#-----------------------------------------------------------
# Spearman Rho of Round vs Deviation from Treatment Median
# Wilcoxon test + Rho distribution plot + summary table
#-----------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon


# === Compute Spearman Rho per group ===
spearman_results = []

for group_id, group_df in df_valid.groupby('Group ID'):
    round_series = group_df['Round']
    deviation_series = group_df['Deviation from Median']
    
    if deviation_series.nunique() > 1:
        rho, pval = spearmanr(round_series, deviation_series)
    else:
        rho, pval = np.nan, np.nan
    
    session, group = group_id.split("_")
    
    spearman_results.append({
        'Session': session,
        'Group': group,
        'Group ID': group_id,
        'Spearman Rho': rho,
        'Spearman P': pval
    })

# === Build dataframe ===
spearman_df = pd.DataFrame(spearman_results)

# === Save Spearman Rho results ===
spearman_df.to_csv("H8_spearman_rho_per_group.csv", index=False)
print("Saved H8_spearman_rho_per_group.csv")

# === Wilcoxon signed-rank test vs 0 ===
valid_rho = spearman_df['Spearman Rho'].dropna()

stat, pval = wilcoxon(valid_rho - 0)
median_rho = np.median(valid_rho)

print(f"\nH8 Wilcoxon test: stat={stat:.3f}, p-value={pval:.4f}")
print(f"Median Spearman Rho across groups: {median_rho:.3f}")

# === Plot Rho distribution ===
plt.figure(figsize=(8,6))
sns.histplot(valid_rho, kde=True, bins=20, color="#444444", edgecolor='black')
plt.axvline(0, color='black', linestyle='--')
plt.title("Rho distribution: Round vs Normalized End Balance of Group")
plt.xlabel("Spearman Rho (Round vs Normalized End Balance of Group)")
plt.ylabel("Count")

# === Box with stats ===
textstr = '\n'.join((
    f"Significant positive: {((spearman_df['Spearman Rho'] > 0) & (spearman_df['Spearman P'] < 0.05)).sum()} groups",
    f"Non-significant: {((spearman_df['Spearman P'] >= 0.05) | (spearman_df['Spearman P'].isna())).sum()} groups",
    f"Significant negative: {((spearman_df['Spearman Rho'] < 0) & (spearman_df['Spearman P'] < 0.05)).sum()} groups",
    f"Median rho = {median_rho:.3f}",
    f"Wilcoxon p = {pval:.3f}"
))
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=props)

# === Save plot ===
plt.tight_layout()
plt.savefig("H8_rho_distribution.png", dpi=300)
plt.close()

print("Saved H8_rho_distribution.png")

# === Build summary table ===
summary_counts = {
    'Significant positive trend': ((spearman_df['Spearman Rho'] > 0) & (spearman_df['Spearman P'] < 0.05)).sum(),
    'Significant negative trend': ((spearman_df['Spearman Rho'] < 0) & (spearman_df['Spearman P'] < 0.05)).sum(),
    'Non-significant trend': ((spearman_df['Spearman P'] >= 0.05) | (spearman_df['Spearman P'].isna())).sum()
}

# === Print summary nicely ===
print("\n=== H8 Summary Table ===")
for k, v in summary_counts.items():
    print(f"{k}: {v} groups")

print(f"Total groups: {sum(summary_counts.values())}")

print("\n=== DONE === H8_analysis completed ===")


# === Build lists of Normalized End Balance in Round 1 and Round 2 ===
list_round1 = []
list_round2 = []

for group_id, group_df in df_valid.groupby('Group ID'):
    group_df_sorted = group_df.sort_values(by='Round')
    
    # Must have at least rounds 1 and 2 present:
    if 1 in group_df_sorted['Round'].values and 2 in group_df_sorted['Round'].values:
        round1_value = group_df_sorted[group_df_sorted['Round'] == 1]['Deviation from Median'].values[0]
        round2_value = group_df_sorted[group_df_sorted['Round'] == 2]['Deviation from Median'].values[0]
        
        list_round1.append(round1_value)
        list_round2.append(round2_value)

# === Perform Wilcoxon signed-rank test ===
stat, pval = wilcoxon(list_round2, list_round1)

# === Print results ===
print("\n=== H8 Round 1 vs Round 2 Wilcoxon test ===")
print(f"Wilcoxon stat={stat:.3f}, p-value={pval:.4f}")
print(f"Median Round 1 Normalized End Balance: {np.median(list_round1):.3f}")
print(f"Median Round 2 Normalized End Balance: {np.median(list_round2):.3f}")
print(f"Number of paired groups: {len(list_round1)}")

# === KDE plot of Normalized End Balance of Group across Rounds — FINAL style ===

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure(figsize=(8,6))
palette = sns.color_palette("Set2", 3)

summary_stats = []

for i, round_num in enumerate([1,2,3]):
    subset = df_valid[df_valid['Round'] == round_num]['Deviation from Median']
    
    sns.kdeplot(
        subset,
        label=f"Round {round_num}",
        linewidth=2,
        fill=True,
        alpha=0.4,
        color=palette[i]
    )
    
    mean_val = subset.mean()
    std_dev = subset.std()
    skew = subset.skew()
    kurt = subset.kurt()
    
    summary_stats.append({
        "Round": round_num,
        "Mean": mean_val,
        "StdDev": std_dev,
        "Skewness": skew,
        "Kurtosis": kurt
    })

# Build box text → Round in bold, rest normal
text_lines = []
for stats in summary_stats:
    text_lines.append(f"$\\bf{{Round {stats['Round']}}}$:")
    text_lines.append(f"  Mean = {stats['Mean']:.1f}")
    text_lines.append(f"  StdDev = {stats['StdDev']:.1f}")
    text_lines.append(f"  Skew = {stats['Skewness']:.2f}")
    text_lines.append(f"  Kurt = {stats['Kurtosis']:.2f}")
    text_lines.append("")

textstr = "\n".join(text_lines)

# Plot styling
plt.title("Distribution of Normalized End Balance of Group across Rounds", fontsize=14)
plt.xlabel("Normalized End Balance of Group", fontsize=12)
plt.ylabel("Density of Group", fontsize=12)
legend = plt.legend(title="Round", loc='upper left')

plt.grid(True, linestyle="--", alpha=0.4)

# Remove scientific notation on y-axis
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

# Add box → with bold Round X
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
plt.text(0.98, 0.98, textstr, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

# Save
plt.tight_layout()
plt.savefig("H8_KDE_with_mean_stddev_moments_FINAL_v3.png", dpi=300)
plt.show()

print("Saved H8_KDE_with_mean_stddev_moments_FINAL_v3.png")


#-----------------------------------------------------------
# H8 Final 3-Panel Grid Plot — improved layout
# a) Example Groups
# b) Rho Distribution (with \n)
# c) KDE (bigger height)
#-----------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from scipy.stats import wilcoxon

# === Matplotlib configuration for scientific style ===
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "text.usetex": False
})

# === Setup Grid ===
fig = plt.figure(figsize=(16, 14))  # more height for panel c
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.4], width_ratios=[2, 1], hspace=0.4, wspace=0.25)

# === Panel a) Example Groups ===
gs_a = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[0,0], wspace=0.6, hspace=0.5)

for i, group_id in enumerate(example_groups):
    ax = fig.add_subplot(gs_a[i])
    session, group = group_id.split("_")
    subset = df_valid[df_valid['Group ID'] == group_id].sort_values(by='Round')

    ax.plot(subset['Round'], subset['Deviation from Median'], color="#444444", linewidth=2)
    ax.scatter(subset['Round'], subset['Deviation from Median'],
               facecolors='white', edgecolors="#444444", marker='o', linewidth=1.5, s=50)

    ax.set_title(f"Session {session} Group {group}", fontsize=11, fontweight='normal')
    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel("Normalized End Balance", fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

# Move the title a) to x=0.27 and a bit lower in y (aligned with b title)
fig.text(0.35, 0.915, "a) Example groups: Normalized End Balance of Group by Round",
         fontsize=14, fontweight='normal', ha='center')

# === Panel b) Rho Distribution ===
ax_b = fig.add_subplot(gs[0,1])

valid_rho = spearman_df['Spearman Rho'].dropna()
median_rho = np.median(valid_rho)
stat, pval = wilcoxon(valid_rho - 0)

sns.histplot(valid_rho, kde=True, bins=20, color="#444444", edgecolor='black', ax=ax_b)
ax_b.axvline(0, color='black', linestyle='--')
ax_b.set_title("b) Rho distribution:\nRound vs Normalized End Balance of Group")
ax_b.set_xlabel("Spearman Rho\n(Round vs Normalized End Balance of Group)")
ax_b.set_ylabel("Count")

# Box with stats
textstr = '\n'.join((
    f"Significant positive: {((spearman_df['Spearman Rho'] > 0) & (spearman_df['Spearman P'] < 0.05)).sum()} groups",
    f"Non-significant: {((spearman_df['Spearman P'] >= 0.05) | (spearman_df['Spearman P'].isna())).sum()} groups",
    f"Significant negative: {((spearman_df['Spearman Rho'] < 0) & (spearman_df['Spearman P'] < 0.05)).sum()} groups",
    f"Median rho = {median_rho:.3f}",
    f"Wilcoxon p = {pval:.3f}"
))
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
ax_b.text(0.02, 0.92, textstr, transform=ax_b.transAxes,
          fontsize=10, verticalalignment='top', bbox=props)

# === Panel c) KDE ===
ax_c = fig.add_subplot(gs[1,:])

palette = sns.color_palette("Set2", 3)
summary_stats = []

for i, round_num in enumerate([1,2,3]):
    subset = df_valid[df_valid['Round'] == round_num]['Deviation from Median']
    
    sns.kdeplot(
        subset,
        label=f"Round {round_num}",
        linewidth=2,
        fill=True,
        alpha=0.4,
        color=palette[i],
        ax=ax_c
    )
    
    mean_val = subset.mean()
    std_dev = subset.std()
    skew = subset.skew()
    kurt = subset.kurt()
    
    summary_stats.append({
        "Round": round_num,
        "Mean": mean_val,
        "StdDev": std_dev,
        "Skewness": skew,
        "Kurtosis": kurt
    })

# Build box text
text_lines = []
for stats in summary_stats:
    text_lines.append(f"$\\bf{{Round {stats['Round']}}}$:")
    text_lines.append(f"  Mean = {stats['Mean']:.1f}")
    text_lines.append(f"  StdDev = {stats['StdDev']:.1f}")
    text_lines.append(f"  Skew = {stats['Skewness']:.2f}")
    text_lines.append(f"  Kurt = {stats['Kurtosis']:.2f}")
    text_lines.append("")

textstr_c = "\n".join(text_lines)

ax_c.set_title("c) Distribution of Normalized End Balance of Group across Rounds")
ax_c.set_xlabel("Normalized End Balance of Group")
ax_c.set_ylabel("Density of Group")

legend = ax_c.legend(title="Round", loc='upper left')
ax_c.grid(True, linestyle="--", alpha=0.4)

# Remove scientific notation on y-axis
ax_c.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0f}'.format(y)))

# Add box
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8)
ax_c.text(0.98, 0.98, textstr_c, transform=ax_c.transAxes,
          fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

# === Save combined grid ===
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("H8_final_3panel_grid_IMPROVED.png", dpi=600)
plt.show()

print("Saved H8_final_3panel_grid_IMPROVED.png")
plt.savefig("H8_final_3panel_grid_IMPROVED.pdf")




#------------------------------------------------------------
# Statistics


# Build list of differences Round2 - Round1
diff_list = []

# For counting positive and negative diffs
count_pos = 0
count_neg = 0
count_zero = 0

print("\n=== Round 1 vs Round 2 Normalized End Balance per Group ===")
print("Group ID\tRound 1\tRound 2\tDifference (R2 - R1)")

for group_id, group_df in df_valid.groupby('Group ID'):
    group_rounds = group_df['Round'].unique()
    if 1 in group_rounds and 2 in group_rounds:
        val_r1 = group_df[group_df['Round'] == 1]['Deviation from Median'].values[0]
        val_r2 = group_df[group_df['Round'] == 2]['Deviation from Median'].values[0]
        
        diff = val_r2 - val_r1
        diff_list.append(diff)
        
        # Count positive / negative / zero
        if diff > 0:
            count_pos += 1
        elif diff < 0:
            count_neg += 1
        else:
            count_zero += 1
        
        print(f"{group_id}\t{val_r1:.1f}\t{val_r2:.1f}\t{diff:.1f}")

# Now run Wilcoxon test on differences
from scipy.stats import wilcoxon

stat, pval = wilcoxon(diff_list)

print("\n=== Round 2 - Round 1 Differences Wilcoxon test ===")
print(f"Wilcoxon stat={stat:.3f}, p-value={pval:.4f}")
print(f"Median difference (Round 2 - Round 1): {np.median(diff_list):.1f}")
print(f"Number of paired groups: {len(diff_list)}")

# Print positive / negative / zero counts
print(f"\nNumber of groups with positive difference: {count_pos}")
print(f"Number of groups with negative difference: {count_neg}")
print(f"Number of groups with zero difference: {count_zero}")

from scipy.stats import kruskal

# Build list of deviation arrays per round
round1_dev = df_valid[df_valid['Round'] == 1]['Deviation from Median'].values
round2_dev = df_valid[df_valid['Round'] == 2]['Deviation from Median'].values
round3_dev = df_valid[df_valid['Round'] == 3]['Deviation from Median'].values

# Kruskal-Wallis test
kruskal_stat, kruskal_pval = kruskal(round1_dev, round2_dev, round3_dev)

# Print result
print("\n=== H9 Kruskal-Wallis test across rounds ===")
print(f"Kruskal-Wallis stat={kruskal_stat:.3f}, p-value={kruskal_pval:.4f}")
