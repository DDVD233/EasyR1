import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Define the color palette from your template
colors = ["#da81c1", "#7dbfa7", "#b0d766", "#8ca0cb", "#ee946c", "#da81c1"]

# https://github.com/plotly/plotly.py/issues/3469
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("images/random.pdf")
time.sleep(1)

# Create the data with reversed order
data = pd.DataFrame(
    [
        ["QoQ-Med-32B", 0.732, 0.661, 0.571, 0.765, 0.726, 0.758, 0.609, 0.904, 0.691],
        # ["QvQ-Med-7B", 0.691, 0.707, 0.580, 0.765, 0.707, 0.759, 0.568, 0.806, 0.670],
        ["Med-R1", 0.641, 0.630, 0.530, 0.0, 0.659, 0.596, 0.549, 0.550, 0.671],
        ["LLaVa-Med", 0.088, 0.466, 0.448, 0.0, 0.000, 0.049, 0.448, 0.363, 0.434],
        ["o4-mini", 0.198, 0.467, 0.441, 0.0, 0.514, 0.297, 0.267, 0.725, 0.378],
        ["GPT-4o", 0.261, 0.442, 0.222, 0.0, 0.575, 0.036, 0.244, 0.896, 0.401],
    ],
    columns=["Model", "CXR", "Dermoscopy", "CT Scan", "ECG", "Pathology", "Mammo", "Ultrasound", "MRI", "Fundus"],
)

# Prepare the data for plotting
categories = data.columns[1:].tolist()
categories = categories + [categories[0]]  # Close the loop

# Create traces for each model
fig = go.Figure()

for i, model in enumerate(data["Model"]):
    values = data.loc[data["Model"] == model].iloc[0, 1:].tolist()
    values = values + [values[0]]  # Close the loop

    base_color = colors[i]
    fill_color = f"rgba({int(base_color[1:3], 16)}, {int(base_color[3:5], 16)}, {int(base_color[5:7], 16)}, 0.2)"

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            name=model,
            fill="toself",
            line=dict(color=base_color, width=4),  # Solid color for border, increased width
            fillcolor=fill_color,  # Transparent fill
        )
    )

# Update the layout with white background and light gray grid
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0.0, 0.92],
            showline=False,
            tickfont=dict(size=32),
            dtick=0.2,
            ticktext=["0.4", "0.6", "0.8"],
            tickvals=[0.4, 0.6, 0.8],
            gridcolor="lightgray",
        ),
        angularaxis=dict(tickfont=dict(size=32), gridcolor="lightgray"),
        bgcolor="white",
    ),
    showlegend=True,
    width=900,
    height=800,
    font=dict(size=34, color="black", family="Computer Modern"),
    legend=dict(
        font=dict(size=24, family="Computer Modern", color="black"),
        x=0.8,
        y=1.1,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
    margin=dict(r=10),
)

# Save the plot
fig.write_image("images/vlm_compare.png", width=900, height=800, scale=6)
