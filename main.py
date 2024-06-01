import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


df_clean = pd.read_csv('finalNum.csv')

# Ensure 'dance_frequency' is numeric
df_clean['dance_frequency'] = pd.to_numeric(df_clean['dance_frequency'], errors='coerce')

# Filter out any rows where 'dance_frequency' or 'ikigai_score' is NaN
df_clean = df_clean.dropna(subset=['dance_frequency', 'ikigai_score', 'life_satisfaction_score'])

# Function to categorize Ikigai scores
df_clean['ikigai_score_category'] = df_clean['ikigai_score'].apply(lambda x: 'Low Ikigai' if x <= 35 else 'High Ikigai')

# Descriptive Statistics
def descriptive_statistics(df):
    stats = df.describe()
    return stats

descriptive_stats = descriptive_statistics(df_clean)
print(descriptive_stats)

# T-Test
def t_test(group1, group2):
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_val

t_stat_ikigai, p_val_ikigai = t_test(df_clean[df_clean['dancer'] == 1]['ikigai_score'], df_clean[df_clean['dancer'] == 2]['ikigai_score'])
t_stat_ls, p_val_ls = t_test(df_clean[df_clean['dancer'] == 1]['life_satisfaction_score'], df_clean[df_clean['dancer'] == 2]['life_satisfaction_score'])

print(f'T-Test Ikigai: t-statistic={t_stat_ikigai}, p-value={p_val_ikigai}')
print(f'T-Test Life Satisfaction: t-statistic={t_stat_ls}, p-value={p_val_ls}')

# Correlation Analysis
def correlation_analysis(df, var1, var2):
    corr_coeff, p_val = stats.pearsonr(df[var1], df[var2])
    return corr_coeff, p_val

corr_ikigai_ls, p_val_ikigai_ls = correlation_analysis(df_clean, 'ikigai_score', 'life_satisfaction_score')
corr_dance_ikigai, p_val_dance_ikigai = correlation_analysis(df_clean, 'dance_frequency', 'ikigai_score')

print(f'Correlation Ikigai and Life Satisfaction: coefficient={corr_ikigai_ls}, p-value={p_val_ikigai_ls}')
print(f'Correlation Dance Frequency and Ikigai: coefficient={corr_dance_ikigai}, p-value={p_val_dance_ikigai}')

# Chi-Square Test for Dancer Status and Ikigai Score Category
contingency_table_dancer_ikigai = pd.crosstab(df_clean['dancer'], df_clean['ikigai_score_category'])
chi2_stat_dancer_ikigai, p_val_dancer_ikigai, dof_dancer_ikigai, expected_dancer_ikigai = stats.chi2_contingency(contingency_table_dancer_ikigai)

print(f'Chi-Square Test Dancer and Ikigai: chi2_stat={chi2_stat_dancer_ikigai}, p-value={p_val_dancer_ikigai}, dof={dof_dancer_ikigai}')
print('Contingency Table:\n', contingency_table_dancer_ikigai)

# Chi-Square Test for Dance Frequency and Ikigai Score Category
df_clean['dance_frequency_category'] = df_clean['dance_frequency'].apply(lambda x: 'Low Frequency' if x == 1 else 'High Frequency')
contingency_table_freq_ikigai = pd.crosstab(df_clean['dance_frequency_category'], df_clean['ikigai_score_category'])
chi2_stat_freq_ikigai, p_val_freq_ikigai, dof_freq_ikigai, expected_freq_ikigai = stats.chi2_contingency(contingency_table_freq_ikigai)

print(f'Chi-Square Test Frequency and Ikigai: chi2_stat={chi2_stat_freq_ikigai}, p-value={p_val_freq_ikigai}, dof={dof_freq_ikigai}')
print('Contingency Table:\n', contingency_table_freq_ikigai)

# Visualizations

# Scatter plot with regression line for Dance Frequency and Ikigai Scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x='dance_frequency', y='ikigai_score', data=df_clean)
sns.regplot(x='dance_frequency', y='ikigai_score', data=df_clean, scatter=False, color='red')
plt.title('Correlation between Dance Frequency and Ikigai Scores')
plt.xlabel('Dance Frequency')
plt.ylabel('Ikigai Score')
plt.text(1, max(df_clean['ikigai_score']) - 2,
         'Pearson Correlation Coefficient: 0.166\nP-Value: 0.070\nConclusion: No significant correlation',
         fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
plt.show()

# Bar plot for the Contingency Table of Dance Frequency and Ikigai Categories
contingency_table_freq_ikigai.plot(kind='bar', stacked=True)
plt.title('Contingency Table of Dance Frequency and Ikigai Categories')
plt.xlabel('Dance Frequency Category')
plt.ylabel('Count')
plt.show()

# Box plots for T-Test results
plt.figure(figsize=(10, 6))
sns.boxplot(x='dancer', y='ikigai_score', data=df_clean, palette='pastel')
plt.title('Ikigai Scores by Dancer Status')
plt.xlabel('Dancer Status')
plt.ylabel('Ikigai Score')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='dancer', y='life_satisfaction_score', data=df_clean, palette='pastel')
plt.title('Life Satisfaction Scores by Dancer Status')
plt.xlabel('Dancer Status')
plt.ylabel('Life Satisfaction Score')
plt.show()

# Correlation between Ikigai and Life Satisfaction Scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x='life_satisfaction_score', y='ikigai_score', data=df_clean)
sns.regplot(x='life_satisfaction_score', y='ikigai_score', data=df_clean, scatter=False, color='red')
plt.title('Correlation between Life Satisfaction and Ikigai Scores')
plt.xlabel('Life Satisfaction Score')
plt.ylabel('Ikigai Score')
plt.show()



# Create a more visually appealing box plot with colors and annotations higher but still below the boxes
plt.figure(figsize=(10, 6))
boxprops = dict(linewidth=2, color='skyblue')
medianprops = dict(linewidth=2, color='darkblue')





# Extract the relevant columns
data_relevant = data[['באיזו תדירות אתה רוקד ריקוד חברתי?', 'ikigai score']]
data_relevant.columns = ['Dance Frequency', 'Ikigai Score']

# Map the frequency of dancing to meaningful labels
frequency_mapping = {0: 'Does not dance', 1: 'Less than once a week', 2: 'Once a week or more'}
data_relevant['Dance Frequency'] = data_relevant['Dance Frequency'].map(frequency_mapping)

# Calculate the average ikigai score for each group
average_ikigai = data_relevant.groupby('Dance Frequency')['Ikigai Score'].mean().reset_index()

# Create the bar plot
plt.figure(figsize=(10, 6))
plt.bar(average_ikigai['Dance Frequency'], average_ikigai['Ikigai Score'], color=['skyblue', 'lightgreen', 'salmon'])
plt.xlabel('Dance Frequency')
plt.ylabel('Average Ikigai Score')
plt.title('Average Ikigai Score by Dance Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


boxplot = data_relevant.boxplot(
    column='Ikigai Score',
    by='Dance Frequency',
    grid=False,
    patch_artist=True,
    boxprops=boxprops,
    medianprops=medianprops,
    return_type='dict'
)

colors = ['skyblue', 'lightgreen', 'salmon']
for patch, color in zip(boxplot['Ikigai Score']['boxes'], colors):
    patch.set_facecolor(color)

plt.xlabel('Dance Frequency')
plt.ylabel('Ikigai Score')
plt.title('Ikigai Score by Dance Frequency')
plt.suptitle('')  # Remove the automatic title
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate the plot with the average and standard deviation values higher but still below the boxes
for i, row in average_ikigai.iterrows():
    plt.text(i+1, data_relevant['Ikigai Score'].min() + 1,
             f'Avg: {row["Average Ikigai Score"]:.2f}\nStd: {row["Std Ikigai Score"]:.2f}',
             horizontalalignment='center', size=12, color='red', weight='semibold', verticalalignment='top')

plt.tight_layout()
plt.show()

# Separate the data for dancers and non-dancers
dancers = df_clean[df_clean['באיזו תדירות אתה רוקד ריקוד חברתי?'] == 2]['ikigai score']
non_dancers = df_clean[df_clean['באיזו תדירות אתה רוקד ריקוד חברתי?'] == 0]['ikigai score']

# Q-Q Plot for dancers
plt.figure(figsize=(10, 6))
stats.probplot(dancers.dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot for Ikigai Scores (Dancers)')
plt.show()

# Q-Q Plot for non-dancers
plt.figure(figsize=(10, 6))
stats.probplot(non_dancers.dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot for Ikigai Scores (Non-Dancers)')
plt.show()