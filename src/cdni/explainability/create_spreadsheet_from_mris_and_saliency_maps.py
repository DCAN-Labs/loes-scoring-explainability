import logging
import os
import sys

from tqdm import tqdm

from dcan.explainability.saliency_maps_from_spreadsheet import create_dfs, \
    get_saliency_file_name


def compute_saliency_from_spreadsheet(
        input_spreadsheet_file_path, output_folder, output_spreadsheet_file_name):
    df, output_df = create_dfs(input_spreadsheet_file_path)
    if not os.path.exists(output_folder):
        logging.error(f"saliency_maps_folder doesn't exist: {output_folder}")

    for index, row in tqdm(df.iterrows()):
        input_file = row['file']
        get_saliency_file_name(index, input_file, output_df, output_folder)
    output_df = output_df.sort_values(by='loes-score', ascending=False)
    output_df.to_csv(output_spreadsheet_file_name, index=False)


if __name__ == '__main__':
    compute_saliency_from_spreadsheet(
        sys.argv[1], sys.argv[3], sys.argv[4])
