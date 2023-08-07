"""
Script for converting the Child Mind Institute EEG data, cleaned by Christoffer Hatlestad-Hall, to numpy arrays
"""
from src.data.datasets.cleaned_child_data import CleanedChildData


def main() -> None:
    # ----------------------
    # Create data object
    # ----------------------
    dataset = CleanedChildData()

    # ----------------------
    # Save data as numpy arrays
    # ----------------------
    dataset.save_eeg_data_as_numpy(from_root_path="/media/thomas/AI-Mind - Anonymised data/"
                                                  "child_mind_data_resting_state_preprocessed",
                                   to_root_path=dataset.root_dir,
                                   num_time_steps=500*30, time_series_start=500*10)


if __name__ == "__main__":
    main()
