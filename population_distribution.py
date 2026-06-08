import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("population_data.csv")

plt.figure(figsize=(10,6))
sns.barplot(x="Age_Group", y="Population", hue="Gender", data=df)

plt.title("Population Distribution by Age Group and Gender")
plt.xlabel("Age Group")
plt.ylabel("Population")
plt.legend(title="Gender")
plt.tight_layout()

plt.savefig("population_distribution_chart.png")
plt.show()

print("Chart saved as population_distribution_chart.png")
