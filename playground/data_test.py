# Set the python path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dataset import (  # noqa
    PalayKeyComponent,
    AFFGDPComponent,
    ElNinoSSTComponent,
    AreaHarvestedComponent,
    PalayYieldComponent,
    PalayDataset,
)

import torch.nn as nn
import torch.optim as optim
from models import RNNModel

GENERATE_COMPONENTS = True

if GENERATE_COMPONENTS:
    component_key = PalayKeyComponent(
        "/scratch/blherre4/datasets/palay/source/palay_yield_new.csv"
    )
    component_key.save_component()
    aff_gdp_component = AFFGDPComponent(
        "/scratch/blherre4/datasets/palay/source/quarterly_gdp.csv",
        key=component_key.dataframe,
        history_spread=4,
        include_unknown_period=True,
    )
    aff_gdp_component.save_component()
    el_nino_sst_component = ElNinoSSTComponent(
        "/scratch/blherre4/datasets/palay/source/el_nino_sst.csv",
        key=component_key.dataframe,
        history_spread=4,
        include_unknown_period=True,
    )
    el_nino_sst_component.save_component()
    area_harvested_component = AreaHarvestedComponent(
        "/scratch/blherre4/datasets/palay/source/area_harvested_new.csv",
        key=component_key.dataframe,
        history_spread=4,
        include_unknown_period=True,
    )
    area_harvested_component.save_component()
    palay_yield_component = PalayYieldComponent(
        "/scratch/blherre4/datasets/palay/source/palay_yield_new.csv",
        key=component_key.dataframe,
        history_spread=0,
    )
    palay_yield_component.save_component()

else:
    component_key = PalayKeyComponent(save_path=True)
    aff_gdp_component = AFFGDPComponent(save_path=True)
    el_nino_sst_component = ElNinoSSTComponent(save_path=True)
    area_harvested_component = AreaHarvestedComponent(save_path=True)
    palay_yield_component = PalayYieldComponent(save_path=True)


# >  _____ ___ ___ _____ ___ _  _  ___    ___ ___  ___  _   _ _  _ ___  ___
# > |_   _| __/ __|_   _|_ _| \| |/ __|  / __| _ \/ _ \| | | | \| |   \/ __|
# >   | | | _|\__ \ | |  | || .` | (_ |   (_ |   / (_) | |_| | .` | |) \__ \
# >   |_| |___|___/ |_| |___|_|\_|\___|  \___|_|_\\___/ \___/|_|\_|___/|___/
# >

data = PalayDataset(
    key_component=component_key,
    feature_components=[
        aff_gdp_component,
        el_nino_sst_component,
        area_harvested_component,
    ],
    label_components=[palay_yield_component],
    categorical_encoder="label",
    state_version=True,
)
print(len(data), "\n")
print(data[0])


exit()

model = RNNModel(data[0][0].shape[1], 10, data[0][1].numel(), 2, 0.05)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (one epoch)
for i, (seq, target) in enumerate(data):
    model.train()  # Set the model to training mode

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = model(seq)
    target = target.unsqueeze(0)

    # Calculate loss
    loss = criterion(output, target)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print loss for this batch
    print(f"Batch {i+1}, Loss: {loss.item():.4f}")

# train, valid, test = data.split_dataset(0.6, 0.2, True)


# def get_year_range(subset, original_data):
#     indices = subset.indices
#     subset_data = original_data.iloc[indices]
#     min_year = subset_data["Year"].min()
#     max_year = subset_data["Year"].max()
#     return min_year, max_year


# train_min_year, train_max_year = get_year_range(train, data.original_data)
# valid_min_year, valid_max_year = get_year_range(valid, data.original_data)
# test_min_year, test_max_year = (
#     get_year_range(test, data.original_data) if test else (None, None)
# )


# print(f"Train set year range: {train_min_year} - {train_max_year}")
# print(f"Validation set year range: {valid_min_year} - {valid_max_year}")
# print(f"Test set year range: {test_min_year} - {test_max_year}")
