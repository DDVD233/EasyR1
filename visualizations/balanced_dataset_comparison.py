import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Define the modalities from the table
modalities = ["Chest X-Ray", "Mammo.", "Dermoscopy", "CT Scan", "Fundus", "Ultrasound", "MRI", "Pathology", "Overall"]

# Extract the Accuracy and F1 scores for each model and modality
grpo_acc = [0.670, 0.754, 0.694, 0.571, 0.669, 0.636, 0.861, 0.770, 0.703]
drpo_acc = [0.713, 0.761, 0.680, 0.571, 0.689, 0.656, 0.881, 0.823, 0.722]
grpo_f1 = [0.064, 0.222, 0.343, 0.257, 0.083, 0.365, 0.780, 0.449, 0.320]
drpo_f1 = [0.164, 0.228, 0.316, 0.257, 0.146, 0.372, 0.813, 0.598, 0.362]

# Create a subplot layout with 2 rows and 1 column
fig = make_subplots(
    rows=2, cols=1, subplot_titles=("Accuracy Comparison", "F1 Score Comparison"), vertical_spacing=0.18
)
font_name = "Computer Modern"


# Text position function to place text in middle of bars
def get_text_positions(values):
    return ["inside" for v in values]


# Format numbers function - format as .000 (3 decimal places)
def format_numbers(values):
    return [f"{v:.3f}" for v in values]


# Add GRPO accuracy bars (first subplot)
fig.add_trace(
    go.Bar(
        x=modalities,
        y=grpo_acc,
        name="GRPO-Balanced (Acc)",
        marker_color="#7dbfa7",
        text=format_numbers(grpo_acc),
        legendgroup="GRPO",
        textposition="inside",
        textfont=dict(color="white", size=10),
        insidetextanchor="end",
        width=0.4,
    ),
    row=1,
    col=1,
)

# Add DRPO accuracy bars (first subplot)
fig.add_trace(
    go.Bar(
        x=modalities,
        y=drpo_acc,
        name="DRPO-Balanced (Acc)",
        marker_color="#da81c1",
        text=format_numbers(drpo_acc),
        textposition="inside",
        textfont=dict(color="white", size=10),
        legendgroup="DRPO",
        insidetextanchor="end",
        width=0.4,
    ),
    row=1,
    col=1,
)

# Add GRPO F1 bars (second subplot)
fig.add_trace(
    go.Bar(
        x=modalities,
        y=grpo_f1,
        name="GRPO-Balanced (F1)",
        marker_color="#7dbfa7",
        text=format_numbers(grpo_f1),
        textposition=get_text_positions(grpo_f1),
        textfont=dict(color="white", size=10),
        insidetextanchor="end",
        width=0.4,
        legendgroup="GRPO",
        showlegend=False,
    ),
    row=2,
    col=1,
)

# Add DRPO F1 bars (second subplot)
fig.add_trace(
    go.Bar(
        x=modalities,
        y=drpo_f1,
        name="DRPO-Balanced (F1)",
        marker_color="#da81c1",
        text=format_numbers(drpo_f1),
        textposition=get_text_positions(drpo_f1),
        textfont=dict(color="white", size=10),
        insidetextanchor="end",
        width=0.4,
        legendgroup="DRPO",
        showlegend=False,
    ),
    row=2,
    col=1,
)

# Update the layout
fig.update_layout(
    # title='Comparison of DRPO vs GRPO on Balanced Datasets',
    barmode="group",
    legend_title="Model",
    font=dict(family=font_name, size=18),
    height=800,  # Increased height for better visualization
    template="plotly_white",
    plot_bgcolor="white",
)

# Update axes
fig.update_xaxes(title_text="Modality", showgrid=False, row=2, col=1)
fig.update_yaxes(
    title_text="Accuracy", range=[0.5, 1], showgrid=True, gridwidth=0.5, gridcolor="lightgray", row=1, col=1
)
fig.update_yaxes(
    title_text="F1 Score", range=[0, 1], showgrid=True, gridwidth=0.5, gridcolor="lightgray", row=2, col=1
)

# Update legend to make it more concise
fig.update_layout(
    legend=dict(
        title="Model", orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, itemsizing="constant"
    ),
)
# Save the figure as a PDF
fig.write_image("images/balanced_dataset_comparisons.png", scale=2)

# # Show the figure (optional for interactive viewing)
# fig.show()
