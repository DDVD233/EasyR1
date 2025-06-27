import os
import time

import plotly.express as px
import plotly.graph_objects as go


# Create images directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# https://github.com/plotly/plotly.py/issues/3469
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("images/random.pdf")
time.sleep(1)

# Record AUC for both models
drpo_acc = {
    "CT RSPECT": 0.5740740740740741,
    "CT INSPECT": 0.6921739130434783,
    "MIMIC-CXR": 0.7499164159144099,
    "Fundus JICHI": 0.625,
    "Fundus APTOS": 0.6816666666666666,
    "CT LNDB": 0.5208333333333334,
    "ISIC 2020": 0.6666666666666666,
    "CBIS-DDSM": 0.7122448979591838,
    "BUSI": 0.7037037037037037,
    "LC25000": 0.7633333333333333,
    "HAM10000": 0.7,
    "VinDr CXR": 0.7512155591572123,
    "CoronaHack": 0.5648148148148149,
    "BCSS": 0.6510416666666666,
    "VinDr Mammo": 0.6766666666666667,
    "COVID-BLUES": 0.5,
    "Brain Tumor": 0.734375,
    "KiTS23": 0.6153846153846154,
    "CheXpert": 0.7260787992495309,
    "PAD-UFES-20": 0.7547169811320755,
    "COVID-19 CXR": 0.6604938271604939,
    "CMMD": 0.8888888888888888,
    "COVID-US": 0.4666666666666666,
    "Messidor-2": 0.7021276595744681,
    "CT Hemorrhage": 0.5,
    "Brain Tumor 2": 0.8780487804878049,
}

grpo_acc = {
    "CT RSPECT": 0.5555555555555555,
    "CT INSPECT": 0.6834782608695652,
    "MIMIC-CXR": 0.7742274442374745,
    "Fundus JICHI": 0.625,
    "Fundus APTOS": 0.68,
    "CT LNDB": 0.5,
    "ISIC 2020": 0.48958333333333337,
    "CBIS-DDSM": 0.6081632653061224,
    "BUSI": 0.5555555555555555,
    "LC25000": 0.8566666666666668,
    "HAM10000": 0.7066666666666667,
    "VinDr CXR": 0.7421663965424093,
    "CoronaHack": 0.5555555555555555,
    "BCSS": 0.6666666666666667,
    "VinDr Mammo": 0.68,
    "COVID-BLUES": 0.5,
    "Brain Tumor": 0.625,
    "KiTS23": 0.38461538461538464,
    "CheXpert": 0.7590994371482176,
    "PAD-UFES-20": 0.739622641509434,
    "COVID-19 CXR": 0.6481481481481481,
    "CMMD": 0.1111111111111111,
    "COVID-US": 0.44,
    "Messidor-2": 0.7021276595744681,
    "CT Hemorrhage": 0.5,
    "Brain Tumor 2": 0.5853658536585366,
}

# Define dataset categories
novel_task_datasets = [
    dataset for dataset in drpo_acc.keys() if "COVID" in dataset.upper() or "CORONA" in dataset.upper()
]

understudied_modality_datasets = [
    "BUSI",
    "COVID-US",
    "COVID-BLUES",
    "KiTS23",
    "CPSC",
    "Chapman",
    "Ga",
    "PTB-XL",
    "CBIS-DDSM",
    "VinDr Mammo",
]

understudied_location_datasets = ["PAD-UFES-20", "COVID-BLUES", "CPSC", "Chapman", "Fundus APTOS"]

# Calculate differences and filter out datasets with missing values
differences = {}
for dataset in drpo_acc:
    if grpo_acc[dataset] is not None:
        diff = drpo_acc[dataset] - grpo_acc[dataset]
        differences[dataset] = diff

# Sort differences in descending order
sorted_differences = dict(sorted(differences.items(), key=lambda x: x[1], reverse=True))

# Filter for positive differences (where multitask performs better)
positive_differences = {k: v for k, v in sorted_differences.items() if k not in ["CMMD"]}

# Define colors
colors = []
for dataset in positive_differences.keys():
    is_novel = dataset in novel_task_datasets
    is_understudied_mod = dataset in understudied_modality_datasets
    is_understudied_loc = dataset in understudied_location_datasets

    category_count = sum([is_novel, is_understudied_mod, is_understudied_loc])

    if category_count >= 2:
        colors.append("#da81c1")  # Green for 2+ categories
    elif is_novel:
        colors.append("#b0d766")  # Pink for novel tasks
    elif is_understudied_mod:
        colors.append("#7dbfa7")  # Teal for understudied modalities
    elif is_understudied_loc:
        colors.append("#8ca0cb")  # Blue for understudied locations
    else:
        colors.append("#CCCCCC")  # Gray for others

# Create bar plot
fig = go.Figure(
    data=[
        go.Bar(
            x=list(positive_differences.keys()),
            y=list(positive_differences.values()),
            text=[f"{v:.4f}" for v in positive_differences.values()],
            textposition="auto",
            marker_color=colors,
        )
    ]
)

# Update layout
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis_title="Dataset",
    yaxis_title="Accuracy Difference (DRPO - GRPO)",
    xaxis_tickangle=-45,
    height=550,
    width=700,
    showlegend=False,
    margin=dict(b=100, t=10, r=10),  # Increase bottom margin for rotated labels
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.2)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(128,128,128,0.2)",
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(128,128,128,0.2)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(128,128,128,0.2)",
    ),
)

# Define legend items with new colors
legend_items = [
    ("Novel Task", "#b0d766"),
    ("Understudied Modality", "#7dbfa7"),
    ("Underrepresented Regions", "#8ca0cb"),
    ("Multiple Categories", "#da81c1"),
    ("Other Datasets", "#CCCCCC"),
]

for i, (name, color) in enumerate(legend_items):
    # Add colored rectangle
    fig.add_shape(
        type="rect",
        x0=0.59,
        x1=0.64,
        y0=0.85 - i * 0.06,
        y1=0.89 - i * 0.06,
        xref="paper",
        yref="paper",
        fillcolor=color,
        line_color="black",
        line_width=1,
    )
    # Add text
    y = 0.905 - i * 0.06
    if i == 4:
        y -= 0.032  # Adjust position for "Other Datasets"
    fig.add_annotation(
        x=0.655, y=y, xref="paper", yref="paper", text=name, showarrow=False, xanchor="left", font=dict(size=18)
    )

fig.update_layout(
    xaxis=dict(
        title_font=dict(size=22),  # Font size for x-axis title
        tickfont=dict(size=18),  # Font size for x-axis tick labels
    ),
    yaxis=dict(
        title_font=dict(size=22),  # Font size for y-axis title
        tickfont=dict(size=18),  # Font size for y-axis tick labels
    ),
    font=dict(
        family="Computer Modern",  # Use any installed font name here
        color="black",
    ),
)

# Save the figure
fig.write_image("images/algo_diff_comparison.png", width=800, height=600, scale=8)

# Print the differences for reference
print("\nAcc Differences (DRPO - GRPO) where Multitask performs better:")
for dataset, diff in positive_differences.items():
    categories = []
    if dataset in novel_task_datasets:
        categories.append("Novel Task")
    if dataset in understudied_modality_datasets:
        categories.append("Understudied Modality")
    if dataset in understudied_location_datasets:
        categories.append("Underrepresented Regions")
    category_str = " & ".join(categories) if categories else "Other"
    print(f"{dataset} ({category_str}): {diff:.4f}")
