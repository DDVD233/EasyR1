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
avg_tps = [532, 386, 425, 429]
t_on_reward = [8.33, 6.79, 8.28, 9.86]

# reverse order
methods = methods[::-1]
avg_tps = avg_tps[::-1]
t_on_reward = t_on_reward[::-1]
font_size = 24

# Create figure
fig = go.Figure()

fig.add_trace(
    go.Bar(
        y=methods,  # <-- was x=methods
        x=avg_tps,  # <-- was y=avg_tps
        orientation="h",  # NEW
        name="Avg. Time Per Step (s)",
        marker_color="#da81c1",
        width=0.9,
        hovertemplate="Method: %{y}<br>Avg. TPS: %{x:.1f}s<extra></extra>",
    )
)

# ---- trace 2: T on Reward ----
fig.add_trace(
    go.Bar(
        y=methods,
        x=t_on_reward,
        orientation="h",  # NEW
        name="Time Spent on Reward Calculation (s)",
        marker_color="#7dbfa7",
        width=0.9,
        hovertemplate="Method: %{y}<br>T on Reward: %{x:.2f}s<extra></extra>",
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
    xaxis_title="Time (s)",  # swapped
    yaxis_title="RL Method",  # swapped
    barmode="overlay",
    bargap=0,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    width=500,
    height=350,
    font=dict(size=font_size, family="Computer Modern"),
)

# Adjust y-axis to make the scale more appropriate for showing both metrics
# fig.update_yaxes(range=[0, max(avg_tps) * 1.1])
fig.update_xaxes(range=[0, max(avg_tps) * 1.1])

# Add annotations for the Avg. TPS values at the top of each bar
for i, method in enumerate(methods):
    # Primary bar label
    fig.add_annotation(
        y=method,
        x=avg_tps[i] + 40,  # now offset in x
        text=f"{avg_tps[i]}",
        showarrow=False,
        font=dict(size=font_size),
    )

    # Reward bar label + arrow
    fig.add_annotation(
        y=method,
        x=t_on_reward[i] + 20,  # slight offset to the right of the green bar
        text=f"{t_on_reward[i]}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=0,  # arrow pointing left
        font=dict(color="white", size=font_size),
    )

# Create images directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Save the figure as a PNG
fig.write_image("images/method_efficiency.jpg", scale=4)

# Also display the figure (useful for interactive viewing)
fig.show()
