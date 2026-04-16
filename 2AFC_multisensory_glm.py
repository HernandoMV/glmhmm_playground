import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lecilab_behavior_analysis.utils as utils
    import lecilab_behavior_analysis.df_transforms as dft
    import lecilab_behavior_analysis.plots as plots
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from glmhmmt.model import SoftmaxGLMHMM
    from pathlib import Path

    return Path, dft, pd, utils


@app.cell
def _(Path, pd, utils):
    project = "COT_cannula_data"
    animals = ['NUO001']#, 'NUO002', 'NUO003', 'NUO004', 'NUO005', 'NUO006', 'NUO007', 'NUO008', 'NUO009', 'NUO010', 'NUO011', 'NUO012']
    df_list = []
    for mouse in animals:
        local_path = Path(utils.get_outpath()) / Path(project) / Path("sessions") / Path(mouse)
        df_animal = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")
        df_list.append(df_animal)
        print(f"Loaded data for {mouse}.")
    # concatenate the dataframes
    df_raw = pd.concat(df_list, ignore_index=True)
    # clear_output(wait=True)
    # time.sleep(.5)
    print("Data read successfully.")
    return (df_raw,)


@app.cell
def _(df_raw, dft):
    df = dft.analyze_df(df_raw)
    print("Dataframe analyzed.")
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(add_day_column_to_df):
    df = add_day_column_to_df(df)
    return (df,)


@app.cell
def _(df):
    # select the date from 2025-10-10, where it did a few performance oscillations
    df_sel = df[df['year_month_date'] == '2025-10-10'].copy()
    return (df_sel,)


@app.cell
def _(df_sel):
    df_sel.head()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
