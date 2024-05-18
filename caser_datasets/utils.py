import os

def get_data_dir(name: str, base_dir: str = None) -> str:
    if base_dir is None:
        base_dir = os.getcwd()
    return os.path.join(base_dir, name)

def apply_action_with_flag(data_dir: str, dataset_name: str, action_method: callable, alternative_method: callable, flag_path: str) -> None:
    flag_file_path = os.path.join(data_dir, flag_path)
    if os.path.isfile(flag_file_path):
        print(f"{action_method.__name__} already applied for {dataset_name}, applying {alternative_method.__name__} instead")
        alternative_method(dataset_name, data_dir)
    else:
        action_method(dataset_name, data_dir)
        with open(flag_file_path, 'w') as flag_file:
            flag_file.write("Processed")
        print(f"{action_method.__name__} applied for {dataset_name}")