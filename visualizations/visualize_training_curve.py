import pandas as pd  # optional, nice for tabular viewing
import wandb


api = wandb.Api()
run = api.run("ddavid233/easy_r1/l3gq150m")  # No KL

records = list(run.scan_history(min_step=75, max_step=76))  # generator → list

df255 = pd.DataFrame(records)  # convenient if you prefer a dataframe
print(df255.T)  # transpose just for prettier console output
