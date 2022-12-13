import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
import sys
torch.manual_seed(42)
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))


def read_config(config_file):
    """Read a config file in YAML.

    Parameters
    ----------
    config_file : str
        Path towards the con fig file in YAML.

    Returns
    -------
    dict
        The parsed config
    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError("Could not find the config to read.")
    with open(config_file, "r") as file:
        dict = yaml.load(file, Loader=yaml.FullLoader)
    return dict


def get_config_file_path(dataset_name, debug):
    """Get the config_file path in real or debug mode.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to get the config from.
    debug : bool
       The mode in which we download the dataset.

    Returns
    -------
    str
        The path towards the config file.
    """
    # assert dataset_name in [
    #     "fed_kits19", "fed_fets2021"
    # ], f"Dataset name {dataset_name} not valid."
    config_file_name = (
        "dataset_location_debug.yaml" if debug else "dataset_location.yaml"
    )
    path_to_config_file_folder = os.getcwd()
    print(path_to_config_file_folder)
    # datasets_dir = str(Path(os.path.realpath(FML_Federated_Medical_Learning.__file__)).parent.resolve())
    # print(datasets_dir)
    # path_to_config_file_folder = os.path.join(
    #     datasets_dir, 'datasets', dataset_name
    # )
    config_file = os.path.join(path_to_config_file_folder, config_file_name)
    return config_file


def create_config(output_folder, debug, dataset_name="fed_camelyon16"):
    """Create or modify config file by writing the absolute path of \
        output_folder in its dataset_path key.

    Parameters
    ----------
    output_folder : str
        The folder where the dataset will be downloaded.
    debug : bool
        Whether or not we are in debug mode.
    dataset_name: str
        The name of the dataset to get the config from.

    Returns
    -------
    Tuple(dict, str)
        The parsed config and the path to the file written on disk.
    Raises
    ------
    ValueError
        If output_folder is not a directory.
    """
    if not (os.path.isdir(output_folder)):
        raise ValueError(f"{output_folder} is not recognized as a folder")

    config_file = get_config_file_path(dataset_name, debug)
    print(config_file)
    if not (os.path.exists(config_file)):
        print(output_folder)
        dataset_path = os.path.realpath(output_folder)
        print(dataset_path)
        dict = {
            "dataset_path": dataset_path,
            "download_complete": False,
            "preprocessing_complete": False,
        }

        with open(config_file, "w") as file:
            yaml.dump(dict, file)
    else:
        dict = read_config(config_file)
    print(dict)
    return dict, config_file


def write_value_in_config(config_file, key, value):
    """Update config_file by modifying one of its key with its new value.

    Parameters
    ----------
    config_file : str
        Path towards a config file
    key : str
        A key belonging to download_complete, preprocessing_complete, dataset_path
    value : Union[bool, str]
        The value to write for the key field.
    Raises
    ------
    ValueError
        If the config file does not exist.
    """
    if not (os.path.exists(config_file)):
        raise FileNotFoundError(
            "The config file doesn't exist. \
            Please create the config file before updating it."
        )
    dict = read_config(config_file)
    dict[key] = value
    with open(config_file, "w") as file:
        yaml.dump(dict, file)


def check_dataset_from_config(dataset_name, debug):
    """Verify that the dataset is ready to be used by reading info from the config
    files.

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to check
    debug : bool
        Whether to use the debug dataset or not.
    Returns
    -------
    dict
        The parsed config.
    Raises
    ------
    ValueError
        The dataset download or preprocessing did not finish.
    """
    try:
        dict = read_config(get_config_file_path(dataset_name, debug))
    except FileNotFoundError:
        if debug:
            raise ValueError(
                f"The dataset was not downloaded, config file "
                "not found for debug mode. Please refer to "
                "the download instructions inside "
                f"FLamby/flamby/datasets/{dataset_name}/README.md"
            )
        else:
            debug = True
            print(
                "WARNING USING DEBUG MODE DATASET EVEN THOUGH DEBUG WAS "
                "SET TO FALSE, COULD NOT FIND NON DEBUG DATASET CONFIG FILE"
            )
            try:
                dict = read_config(get_config_file_path(dataset_name, debug))
            except FileNotFoundError:
                raise ValueError(
                    f"It seems the dataset {dataset_name} was not downloaded as "
                    "the config file is not found for either normal or debug "
                    "mode. Please refer to the download instructions inside "
                    f"FLamby/flamby/datasets/{dataset_name}/README.md"
                )
    if not (dict["download_complete"]):
        raise ValueError(
            f"It seems the dataset {dataset_name} was only partially downloaded"
            "please restart the download script to finish the download."
        )
    if not (dict["preprocessing_complete"]):
        raise ValueError(
            f"It seems the preprocessing for dataset {dataset_name} is not "
            "yet finished please run the appropriate preprocessing scripts "
            "before use"
        )
    return dict
