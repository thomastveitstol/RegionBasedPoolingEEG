"""
Script for saving the channel positions of cleaned Child Mind Institute EEG data
"""
import mne
import pickle

from src.data.datasets.cleaned_child_data import CleanedChildData


def main() -> None:
    # Load example subject
    example_data = mne.io.read_raw_eeglab("/media/thomas/AI-Mind - Anonymised data/child_mind_data_resting_state_"
                                          "preprocessed/NDARAA075AMK.set", preload=True)

    # Extracting channel positions
    ch_positions = example_data.get_montage().get_positions()

    # Save channel positions
    path = CleanedChildData().root_dir
    with open(f"{path}/channel_positions.pkl", "wb") as f:
        pickle.dump(ch_positions, f)


if __name__ == "__main__":
    main()
