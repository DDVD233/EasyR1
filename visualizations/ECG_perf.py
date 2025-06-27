import os
import time

import plotly.express as px
import plotly.graph_objects as go


# https://github.com/plotly/plotly.py/issues/3469
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.write_image("images/random.pdf")
time.sleep(1)

# Data from the image
methods = ["ReMax", "RLOO", "GRPO", "DRPO"]
ecg_perf = [76.67, 74.49, 76.01, 76.8]

# reverse order
methods = methods[::-1]
ecg_perf = ecg_perf[::-1]
font_size = 24

# Create figure
fig = go.Figure()

fig.add_trace(
    go.Bar(
        y=methods,  # <-- was x=methods
        x=ecg_perf,  # <-- was y=avg_tps
        orientation="h",  # NEW
        name="ECG Diagnosis Accuracy (%)",
        marker_color="#da81c1",
        width=0.9,
        hovertemplate="Method: %{y}<br>Avg. TPS: %{x:.1f}s<extra></extra>",
    )
)

# Update layout
fig.update_layout(
    # title={
    #     'text': "Breakdown of Time Spent on Each Each RL Method",
    #     'y': 0.95,
    #     'x': 0.5,
    #     'xanchor': 'center',
    #     'yanchor': 'top'
    # },
    xaxis_title="ECG Diagnosis Accuracy (%)",  # swapped
    yaxis_title="RL Method",  # swapped
    barmode="overlay",
    bargap=0,
    # legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.02,
    #     xanchor="right",
    #     x=1
    # ),
    template="plotly_white",
    width=500,
    height=400,
    font=dict(size=font_size, family="Computer Modern"),
)

# Adjust y-axis to make the scale more appropriate for showing both metrics
# fig.update_yaxes(range=[0, max(avg_tps) * 1.1])
fig.update_xaxes(range=[70, max(ecg_perf) * 1.02])

# Add annotations for the Avg. TPS values at the top of each bar
for i, method in enumerate(methods):
    # Primary bar label
    fig.add_annotation(
        y=method,
        x=ecg_perf[i] + 1.1,  # now offset in x
        text=f"{ecg_perf[i]}",
        showarrow=False,
        font=dict(size=font_size),
    )

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Save the figure as a PNG
fig.write_image("images/ecg_perf.jpg", scale=4)

# Also display the figure (useful for interactive viewing)
fig.show()
