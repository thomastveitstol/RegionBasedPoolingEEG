"""
Script for downloading resting state .mat files of Child Mind Institute dataset.

Required changes:
- download_folder: Where to save the resting state mat files

Authors: Mats Tveter and Thomass Tveitstøl
"""
from typing import List

import boto3
from botocore import UNSIGNED
from botocore.client import Config


def main() -> None:
    BUCKET = 'fcp-indi'
    PREFIX = "data/Projects/HBN/EEG"

    s3client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Creating buckets needed for downloading the files
    s3_resource = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    s3_bucket = s3_resource.Bucket(BUCKET)

    # Paginator is need because the amount of files exceeds the boto3.client possible maxkeys
    paginator = s3client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    # All subject names will be appended to the following list
    subject_names: List[str] = list()

    number_of_downloaded_subjects = 0
    overlapping_subjects = 0

    # Change this to where you want to store the data
    download_folder = "/media/thomas/AI-Mind - Anonymised data/child_mind_data_resting_state/"

    # Looping throught the content of the HBN bucket
    for page in pages:
        for obj in page['Contents']:
            temp_key = obj['Key']

            # Download only RestingState.mat files in raw folder
            if "raw" in temp_key and "RestingState.mat" in temp_key:

                # Get subject name from the dictionary key
                name = temp_key.split("/")[4]

                # In case of overlapping, print warning, else download
                if name in subject_names:
                    print("Key already exists! Key: ", name)
                    overlapping_subjects += 1
                else:
                    print(f"Downloaded: {name}")
                    subject_names.append(name)
                    s3_bucket.download_file(temp_key, download_folder + name + ".mat")
                    number_of_downloaded_subjects += 1

    # Print information about the runs
    print(f"Downloaded Subjects: {number_of_downloaded_subjects}")
    print(f"Overlapping Subjects: {number_of_downloaded_subjects-overlapping_subjects}")


if __name__ == "__main__":
    main()
