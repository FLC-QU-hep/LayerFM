import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from pointcountfm.preprocessing import Identity, Transformation, compose


def load_dataset(
    file: h5py.File,
    key: str,
    start: int = 0,
    end: int | None = None,
) -> Tensor:
    if key not in file:
        raise KeyError(f"Key '{key}' not found in HDF5 file.")
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"Key '{key}' is not a dataset in HDF5 file.")
    data = dataset[start:end]
    is_integer = issubclass(data.dtype.type, (np.integer, np.bool_))
    tensor_dtype = torch.int64 if is_integer else torch.get_default_dtype()
    return torch.from_numpy(data).to(tensor_dtype, copy=False)


def load_data_file(
    data_file: str,
    start: int = 0,
    end: int | None = None,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    with h5py.File(data_file, "r") as file:
        energy_key = "energy" if "energy" in file else "energies"
        e_incident = load_dataset(file, energy_key, start, end)
        num_points = load_dataset(file, "num_points", start, end)
        mint = (
            load_dataset(file, "min_time_per_layer", start, end)
            if "min_time_per_layer" in file
            else None
        )
        en_perl = (
            load_dataset(file, "energy_per_layer", start, end)
            if "energy_per_layer" in file
            else None
        )
        noise = load_dataset(file, "noise", start, end) if load_noise else None
        labels = load_dataset(file, "labels", start, end) if "labels" in file else None
        directions = (
            load_dataset(file, "directions", start, end)
            if "directions" in file
            else None
        )

    if labels is not None:
        if not num_classes:
            num_classes = int(torch.max(labels).item()) + 1
        labels = F.one_hot(labels, num_classes=num_classes).to(
            torch.get_default_dtype()
        )

    return (
        e_incident,
        num_points,
        mint,
        en_perl,
        noise,
        labels,
        directions,
    )


def load_data_directory(
    data_directory: str,
    start: int = 0,
    end: int | None = None,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
    e_incident_list = []
    num_points_list = []
    noise_list = []
    labels_list = []
    directions_list = []
    mint_list = []
    en_perl_list = []

    for i, file_name in enumerate(sorted(os.listdir(data_directory))):
        if not file_name.endswith(".h5"):
            continue
        file_path = os.path.join(data_directory, file_name)
        e_incident, num_points, mint, en_perl, noise, labels, directions = (
            load_data_file(file_path, start=start, end=end, load_noise=load_noise)
        )
        if labels is not None and not torch.all(torch.argmax(labels, dim=1) == i):
            raise ValueError(
                f"Labels in file {file_name} do not match expected label {i}."
            )
        e_incident_list.append(e_incident)
        num_points_list.append(num_points)
        noise_list.append(noise)
        if mint is not None:
            mint_list.append(mint)
        if en_perl is not None:
            en_perl_list.append(en_perl)
        labels_list.append(torch.full((e_incident.shape[0],), i, dtype=torch.int64))
        if directions is not None:
            directions_list.append(directions)

    e_incident = torch.cat(e_incident_list, dim=0)
    num_points = torch.cat(num_points_list, dim=0)
    mint = torch.cat(mint_list, dim=0) if mint_list else None
    en_perl = torch.cat(en_perl, dim=0) if en_perl_list else None
    noise = torch.cat(noise_list, dim=0) if load_noise else None
    labels = torch.cat(labels_list, dim=0)
    directions = torch.cat(directions_list, dim=0) if directions_list else None

    permutation = torch.randperm(e_incident.shape[0])
    e_incident = e_incident[permutation]
    num_points = num_points[permutation]
    if mint is not None:
        mint = mint[permutation]
    if en_perl is not None:
        en_perl = en_perl[permutation]
    if noise is not None:
        noise = noise[permutation]
    labels = labels[permutation]
    if not num_classes:
        num_classes = int(torch.max(labels).item()) + 1
    labels = F.one_hot(labels, num_classes=num_classes).to(torch.get_default_dtype())
    if directions is not None:
        directions = directions[permutation]

    return (
        e_incident,
        num_points,
        mint,
        en_perl,
        noise,
        labels,
        directions,
    )


def load_data(
    data_file: str,
    start: int = 0,
    end: int | None = None,
    load_noise: bool = False,
    num_classes: int = 0,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    if os.path.isdir(data_file):
        return load_data_directory(
            data_file, start, end, load_noise, num_classes=num_classes
        )
    elif os.path.isfile(data_file):
        return load_data_file(
            data_file, start, end, load_noise, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Data file {data_file} does not exist or is not a file or directory."
        )


class DataLoader:
    def __init__(
        self,
        data_file: str,
        transform_inc: Transformation | list | None = None,
        transform_num_points: Transformation | list | None = None,
        transform_min_time: Transformation | list | None = None,
        transform_en_perl: Transformation | list | None = None,
        batch_size: int = 1,
        shuffle: bool = False,
        start: int = 0,
        end: int | None = None,
        fit_transform: bool = False,
        device: torch.device | str = "cpu",
        load_noise: bool = False,
        num_classes: int = 0,
        num_layers: int = 78,
        use_directions: bool = True,
        use_labels: bool = True,
    ) -> None:
        self.data_file = data_file
        self.transform_inc = self.__compose_trafo(transform_inc)
        self.transform_num_points = self.__compose_trafo(transform_num_points)
        self.transform_min_time = self.__compose_trafo(transform_min_time)
        self.transform_en_perl = self.__compose_trafo(transform_en_perl)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_layer = num_layers

        e_incident, num_points, mint, en_perl, noise, labels, directions = load_data(
            data_file,
            start=start,
            end=end,
            load_noise=load_noise,
            num_classes=num_classes,
        )
        self.num_samples = len(e_incident)
        if labels is not None:
            self.num_classes = labels.shape[1]
        else:
            self.num_classes = 0
        self.energy_min = e_incident.min().item()
        self.energy_max = e_incident.max().item()

        e_incident = e_incident.to(device)
        num_points = num_points.to(device)
        if mint is not None:
            mint = mint.to(device)
        if en_perl is not None:
            en_perl = en_perl.to(device)
        if noise is not None:
            noise = noise.to(device)
        if labels is not None:
            labels = labels.to(device)
        if directions is not None:
            directions = directions.to(device)

        if fit_transform:
            e_incident = self.transform_inc.fit(e_incident)
            active_mask = (
                num_points > 0
            )  # layers with hits, used to mask per-layer transforms
            num_points = self.transform_num_points.fit(num_points)
            if mint is not None:
                mint = self.transform_min_time.fit(mint, active_mask)
            if en_perl is not None:
                en_perl = self.transform_en_perl.fit(en_perl, active_mask)
        else:
            e_incident = self.transform_inc(e_incident)
            num_points = self.transform_num_points(num_points)
            if mint is not None:
                mint = self.transform_min_time(mint)
            if en_perl is not None:
                en_perl = self.transform_en_perl(en_perl)

        if mint is not None:
            self.data = torch.cat([num_points, mint], dim=-1)
        else:
            self.data = num_points
        if en_perl is not None:
            self.data = torch.cat([self.data, en_perl], dim=-1)

        parts = [e_incident]
        if labels is not None and use_labels:
            parts.append(labels)
        if directions is not None and use_directions:
            parts.append(directions)
        self.condition = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        self.has_directions = directions is not None and use_directions
        self.noise = noise

    @staticmethod
    def __compose_trafo(transformation: Transformation | list | None) -> Transformation:
        if transformation is None:
            return Identity()
        if isinstance(transformation, list):
            return compose(transformation)
        return transformation

    def __len__(self) -> int:
        return self.num_samples // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples)
        else:
            indices = torch.arange(self.num_samples)
        for i in range(len(self)):
            idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            batch = {"data": self.data[idx], "condition": self.condition[idx]}
            if self.noise is not None:
                batch["noise"] = self.noise[idx]
            else:
                batch["noise"] = None
            yield batch

    def to(self, device_dtype: torch.device | torch.dtype | str) -> None:
        self.data = self.data.to(device_dtype)
        self.condition = self.condition.to(device_dtype)
        self.transform_inc.to(device_dtype)
        self.transform_num_points.to(device_dtype)
        self.transform_min_time.to(device_dtype)
        self.transform_en_perl.to(device_dtype)
        if self.noise is not None:
            self.noise = self.noise.to(device_dtype)


def get_loaders(
    data_file: str,
    transform_inc: Transformation | list | None = None,
    transform_num_points: Transformation | list | None = None,
    transform_min_time: Transformation | list | None = None,
    transform_en_perl: Transformation | list | None = None,
    batch_size: int = 128,
    batch_size_val: int | None = None,
    device: torch.device | str = "cpu",
    num_train: int | None = None,
    num_val: int = 10_000,
    load_noise: bool = False,
    num_classes: int = 0,
    num_layers: int = 78,
    use_directions: bool = True,
    use_labels: bool = True,
) -> tuple[DataLoader, DataLoader]:
    if num_train is None:
        num_train = -num_val
    if batch_size_val is None:
        batch_size_val = batch_size
    train_loader = DataLoader(
        data_file,
        transform_inc=transform_inc,
        transform_num_points=transform_num_points,
        transform_min_time=transform_min_time,
        transform_en_perl=transform_en_perl,
        batch_size=batch_size,
        shuffle=True,
        start=0,
        end=num_train,
        fit_transform=True,
        device=device,
        load_noise=load_noise,
        num_classes=num_classes,
        num_layers=num_layers,
        use_directions=use_directions,
        use_labels=use_labels,
    )
    test_loader = DataLoader(
        data_file,
        transform_inc=train_loader.transform_inc,
        transform_num_points=train_loader.transform_num_points,
        transform_min_time=train_loader.transform_min_time,
        transform_en_perl=train_loader.transform_en_perl,
        batch_size=batch_size_val,
        shuffle=False,
        start=-num_val,
        end=None,
        fit_transform=False,
        device=device,
        load_noise=load_noise,
        num_classes=num_classes,
        num_layers=num_layers,
        use_directions=use_directions,
        use_labels=use_labels,
    )
    return train_loader, test_loader
