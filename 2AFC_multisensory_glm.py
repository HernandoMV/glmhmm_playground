import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import lecilab_behavior_analysis.utils as utils
    import lecilab_behavior_analysis.df_transforms as dft
    import lecilab_behavior_analysis.plots as plots
    import lecilab_behavior_analysis.figure_maker as figure_maker
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from glmhmmt.model import SoftmaxGLMHMM
    from pathlib import Path
    import jax.numpy as jnp

    return Path, SoftmaxGLMHMM, dft, jnp, mo, np, pd, plots, plt, sns, utils


@app.cell
def _(Path, pd, utils):
    project = "COT_cannula_data"
    animals = ["NUO001"]
    df_list = []
    for mouse in animals:
        local_path = Path(utils.get_outpath()) / Path(project) / Path("sessions") / Path(mouse)
        df_animal = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")
        df_list.append(df_animal)
        print(f"Loaded data for {mouse}.")
    # concatenate the dataframes
    df_raw = pd.concat(df_list, ignore_index=True)
    print("Data read successfully.")
    return (df_raw,)


@app.cell
def _(df_raw, dft):
    df = dft.analyze_df(df_raw)
    print("Dataframe analyzed.")
    return (df,)


@app.cell
def _(df, mo):
    unique_dates = sorted(df['year_month_day'].unique())

    from_date_dropdown = mo.ui.dropdown(options=unique_dates, value=unique_dates[0])
    return from_date_dropdown, unique_dates


@app.cell
def _(from_date_dropdown, mo, unique_dates):
    # Ensure to_date_dropdown updates reactively when from_date changes
    _ = from_date_dropdown.value  # Reference to establish dependency
    to_date_dropdown = mo.ui.dropdown(options=unique_dates, value=from_date_dropdown.value)

    mo.hstack([
        mo.md("**From Date:**"),
        from_date_dropdown,
        mo.md("**To Date:**"),
        to_date_dropdown,
    ], gap=1)
    return (to_date_dropdown,)


@app.cell
def _(df, from_date_dropdown, to_date_dropdown):
    # Filter data between selected date range
    from_date = from_date_dropdown.value
    to_date = to_date_dropdown.value

    df_sel = df[(df['year_month_day'] >= from_date) & (df['year_month_day'] <= to_date)].copy()
    print(f"Filtered data from {from_date} to {to_date}: {len(df_sel)} trials")
    return (df_sel,)


@app.cell
def _(dft, plots, plt, utils):
    def create_analysis_plots(df_data, window=25):
        """Create analysis plots for behavioral data.

        Parameters:
        -----------
        df_data : pd.DataFrame
            Filtered dataframe for analysis
        window : int
            Rolling window size for performance calculation

        Returns:
        --------
        matplotlib figure object
        """
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # First row - performance plots (expanded)
        perf_ax = fig.add_subplot(gs[0, 0])  
        roap_ax = fig.add_subplot(gs[0, 1:3]) # spans 2 columns

        # Second row - text and psychometric plots
        text_ax = fig.add_subplot(gs[1, 0])
        visual_psych_by_difficulty_ratio_ax = fig.add_subplot(gs[1, 1])
        auditory_psych_by_difficulty_ratio_ax = fig.add_subplot(gs[1, 2])

        # Summary text
        text_ax = plots.summary_text_plot(df_data, kind="session", ax=text_ax)

        # Performance through trials
        df_perf = dft.get_performance_through_trials(df_data, window=window)
        session_changes = df_perf[df_perf.session != df_perf.session.shift(1)].index

        if df_data.current_training_stage.nunique() > 1:
            perf_hue = "current_training_stage"
        else:
            perf_hue = "stimulus_modality"

        plots.performance_vs_trials_plot(df_perf, perf_ax, legend=True, session_changes=session_changes, hue=perf_hue)

        df_roap = df_perf.copy()
        df_roap["repeat_or_alternate"] = dft.get_repeat_or_alternate_series(df_roap.correct_side)
        df_roap = dft.get_repeat_or_alternate_performance(df_roap, window=window)
        plots.repeat_or_alternate_performance_plot(df_roap, roap_ax, session_changes=session_changes)

        # Psychometric curves for each modality
        df_task = df_data[df_data['task'] != 'Habituation']

        for mod, ax in [('visual', visual_psych_by_difficulty_ratio_ax), 
                         ('auditory', auditory_psych_by_difficulty_ratio_ax)]:
            if len(df_task) == 0:
                ax.text(0.1, 0.5, "Habituation phase", fontsize=10, color='k')
                continue
            if mod not in df_task['stimulus_modality'].unique():
                ax.text(0.1, 0.5, f"No trials in {mod}", fontsize=10, color='k')
                continue

            df_mod = df_task[df_task["stimulus_modality"] == mod].copy()

            if df_mod['difficulty'].nunique() == 1 and df_mod['difficulty'].unique()[0] == 'easy':
                df_mod["side_difficulty"] = df_mod.apply(lambda row: utils.side_and_difficulty_to_numeric(row), axis=1)
                df_mod = dft.add_mouse_first_choice(df_mod)
                df_mod['first_choice_numeric'] = df_mod['first_choice'].apply(utils.transform_side_choice_to_numeric)
                plots.choice_by_difficulty_plot(df_mod, ax=ax)
            else:
                xvar = 'visual_stimulus_ratio' if mod == 'visual' else 'total_evidence_strength'
                value_type = 'discrete' if mod == 'visual' else 'continue'
                psych_df = dft.get_performance_by_difficulty_ratio(df_mod)
                plots.psychometric_plot(psych_df, x=xvar, y='first_choice_numeric', ax=ax, valueType=value_type)

            ax.set_title(f"choices on {mod} trials", fontsize=10)
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        return fig

    return (create_analysis_plots,)


@app.cell
def _(create_analysis_plots, df_sel):
    fig_analysis = create_analysis_plots(df_sel, window=25)
    fig_analysis
    return


@app.cell
def _(df_sel, jnp, np):
    # prepare the data for GLM-HMM fitting to get the probality of going right on each trial
    y = df_sel['first_choice'].map({'left': -1, 'right': 1}).values
    stimulus_evidence = df_sel['correct_side'].map({'left': -1, 'right': 1}).values
    # get the previous choice using y shifted by 1
    previous_choice = np.roll(y, 1)
    bias = np.ones_like(y)

    inputs = np.column_stack([stimulus_evidence, previous_choice, bias])
    inputs_colnames = ['stimulus_evidence', 'previous_choice', 'bias']

    # delete the first row of inputs and y, since they have NaN values due to the shift
    inputs = inputs[1:]
    y = y[1:]

    # convert to jax arrays
    inputs = jnp.array(inputs)
    y = jnp.array(y)
    return inputs, inputs_colnames, y


@app.cell
def _(SoftmaxGLMHMM, inputs_colnames):
    num_states = 2
    num_classes = 2
    emission_input_dim = 3
    transition_input_dim = 0

    model = SoftmaxGLMHMM(
        num_states=num_states,
        num_classes=num_classes,
        emission_input_dim=emission_input_dim,
        transition_input_dim=transition_input_dim,
        emission_feature_names=inputs_colnames,
        # transition_matrix_stickiness=10
    )

    params, props = model.initialize()
    return model, params, props


@app.cell
def _():
    # fit each session separately
    # session_ids = df_sel['session'].values[1:] # get the session ids, also delete the first one to match the inputs and y
    return


@app.cell
def _(inputs, model, params, props, y):
    fitted_params, log_probs = model.fit_em(
        params,
        props,
        y,
        inputs,
        # session_ids=session_ids,
        num_iters=50,
        verbose=True,
    )
    return (fitted_params,)


@app.cell
def _(fitted_params, inputs, model, y):
    states = model.predict_state_probs(fitted_params, y, inputs)
    # get only one column
    state_one = states[:,0]
    return (state_one,)


@app.cell
def _(df_sel, np, pd, plt, sns, state_one):
    # Performance and state-1 probability in one plot
    performance_trace = pd.Series(df_sel["correct"].astype(float).values[1:])
    rolled_performance = performance_trace.rolling(window=25).mean()
    state_one_np = np.asarray(state_one)

    n = min(len(rolled_performance), len(state_one_np))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, rolled_performance.values[:n], color="tab:blue", linewidth=1.2, label="Performance (rolling)")
    ax.plot(x, state_one_np[:n], color="tab:orange", linewidth=1.2, label="State 1 prob")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Value")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=False)
    sns.despine()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
