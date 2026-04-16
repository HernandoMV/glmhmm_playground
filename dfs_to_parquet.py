from pathlib import Path
from lecilab_behavior_analysis import utils
import pandas as pd
import fire

# TODO: finish this script if needed

def main(project, animals):
    df_list = []
    for mouse in animals:
        local_path = Path(utils.get_outpath()) / Path(project) / Path("sessions") / Path(mouse)
        df = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")
        df_list.append(df)
        print(f"Loaded data for {mouse}.")
    # concatenate the dataframes
    df = pd.concat(df_list, ignore_index=True)
    print("Data read successfully.")


if __name__ == "__main__":
    fire.Fire(main)