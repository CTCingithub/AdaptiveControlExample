import yaml
import torch
import warnings
import torch.nn as nn
from torch.linalg import pinv
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


def GET_DEVICE(DEVICE=0):
    # Selecting training device, -1 for cpu
    if DEVICE == -1:
        return "cpu"
    elif torch.cuda.is_available():
        device_name = f"cuda:{DEVICE}"
        if torch.cuda.device_count() > DEVICE:
            return device_name
        print(f"No such cuda device: {DEVICE}")
        return "cpu"


def SETUP_SEED(SEED=42):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def INIT_WEIGHTS(MODEL, CONFIG):
    if CONFIG["INIT"] == "False":
        pass
    elif CONFIG["INIT"] == "Zero":
        for param in MODEL.parameters():
            nn.init.zeros_(param)
    elif CONFIG["INIT"] == "Xavier":
        for param in MODEL.parameters():
            if type(MODEL) == nn.Linear:
                nn.init.xavier_uniform_(param)
    elif CONFIG["INIT"] == "Kaiming":
        for param in MODEL.parameters():
            if type(MODEL) == nn.Linear:
                nn.init.kaiming_uniform_(param)
    else:
        warnings.warn("Unknown initial weights method, use default Kaiming")
        for param in MODEL.parameters():
            if type(MODEL) == nn.Linear:
                nn.init.kaiming_uniform_(param)


def Split2Loaders(INPUT, OUTPUT, CONFIG):
    # Load config
    config = CONFIG["DATA_LOADER"]
    RATIO = config["SPLIT_RATIO"]
    BATCHSIZE = config["BATCH_SIZE"]
    SHUFFLE = config["SHUFFLE"]
    PIN_MEMORY = config["PIN_MEMORY"]

    if len(RATIO) == 2:
        train_ratio = RATIO[0] / (RATIO[0] + RATIO[1])
        training_size = int(train_ratio * INPUT.shape[0])
        validation_size = INPUT.shape[0] - training_size
        training_dataset, validation_dataset = random_split(
            TensorDataset(INPUT, OUTPUT), [training_size, validation_size]
        )
        return DataLoader(
            training_dataset,
            batch_size=BATCHSIZE,
            shuffle=SHUFFLE,
            pin_memory=PIN_MEMORY,
        ), DataLoader(
            validation_dataset,
            batch_size=BATCHSIZE,
            shuffle=SHUFFLE,
            pin_memory=PIN_MEMORY,
        )
    elif len(RATIO) == 3:
        training_ratio = RATIO[0] / (RATIO[0] + RATIO[1] + RATIO[2])
        validation_ratio = RATIO[1] / (RATIO[0] + RATIO[1] + RATIO[2])
        training_size = int(training_ratio * INPUT.shape[0])
        validation_size = int(validation_ratio * INPUT.shape[0])
        test_size = INPUT.shape[0] - training_size - validation_size
        training_dataset, validation_dataset, test_dataset = random_split(
            TensorDataset(INPUT, OUTPUT), [training_size, validation_size, test_size]
        )
        return (
            DataLoader(
                training_dataset,
                batch_size=BATCHSIZE,
                shuffle=SHUFFLE,
                pin_memory=PIN_MEMORY,
            ),
            DataLoader(
                validation_dataset,
                batch_size=BATCHSIZE,
                shuffle=SHUFFLE,
                pin_memory=PIN_MEMORY,
            ),
            DataLoader(
                test_dataset,
                batch_size=BATCHSIZE,
                shuffle=SHUFFLE,
                pin_memory=PIN_MEMORY,
            ),
        )
    else:
        raise ValueError("RATIO must be a list of length 2 or 3")


def freeze_param(Layer, Filter_Weight, Tol_Ratio=None):
    for param in Layer.parameters():
        param.requires_grad = False
    if Filter_Weight:
        for name, param in Layer.named_parameters():
            if "weight" in name:
                param.data = torch.where(
                    torch.abs(param.data)
                    <= Tol_Ratio * torch.abs(Layer.weight.detach()).max(),
                    torch.zeros_like(param.data),
                    param.data,
                )


def TRAIN_WITH_PROGRESS_BAR_TWO_LOSS(
    MODEL,
    CONFIG,
    OPTIMIZER,
    TRAIN_LOADER,
    VALIDATION_LOADER,
    LOSS_TUPLE=None,
    FREEZE_LAYER=None,
):
    # Load config
    config = CONFIG["TRAIN"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    DEVICE = GET_DEVICE(config["DEVICE"])
    LOSS_WEIGHTS = config["LOSS_WEIGHTS"]
    GRAD_MAX = config["GRAD_MAX"]
    LOSS_SWITCH_VALUE = config["LOSS_SWITCH_VALUE"]
    FREEZE_EPOCH = config["FREEZE_EPOCH"]
    TOL_RATIO = config["TOL_RATIO"]
    FILTER_WEIGHTS = config["FILTER_WEIGHTS"]

    if LOSS_TUPLE is None:
        LOSS_TUPLE = [nn.MSELoss(), nn.MSELoss()]

    print("PyTorch Version:", torch.__version__)
    INIT_WEIGHTS(MODEL, config)
    print("Weight initialized with", config["INIT"], "Initialization")
    print("Training on", DEVICE)
    print(
        "====================================Start training===================================="
    )
    # Transfer model to selected device
    MODEL.to(DEVICE)

    # Loss for training and validation
    LOSS_1, LOSS_2 = LOSS_TUPLE

    def compute_loss_no_grad(DATA_LOADER):
        MODEL.eval()
        total_loss_1 = 0.0
        total_loss_2 = 0.0
        with torch.no_grad():
            for x, y in DATA_LOADER:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                output_pred = MODEL(x)
                loss_1 = LOSS_WEIGHTS[0] * LOSS_1(output_pred, y) + LOSS_WEIGHTS[
                    1
                ] * LOSS_2(output_pred, y)
                loss_2 = LOSS_1(output_pred, y)
                total_loss_1 += loss_1.item()
                total_loss_2 += loss_2.item()
        return total_loss_1 / len(DATA_LOADER), total_loss_2 / len(DATA_LOADER)

    # Loss when not trained
    LOSS_TRAIN_AVERAGE_1, LOSS_TRAIN_AVERAGE_2 = compute_loss_no_grad(TRAIN_LOADER)
    LOSS_VALIDATION_AVERAGE_1, LOSS_VALIDATION_AVERAGE_2 = compute_loss_no_grad(
        VALIDATION_LOADER
    )

    # Loss recorders
    training_losses_1 = [
        LOSS_TRAIN_AVERAGE_1,
    ]
    training_losses_2 = [
        LOSS_TRAIN_AVERAGE_2,
    ]
    validation_losses_1 = [
        LOSS_VALIDATION_AVERAGE_1,
    ]
    validation_losses_2 = [
        LOSS_VALIDATION_AVERAGE_2,
    ]

    # Start training
    for epoch in range(NUM_EPOCHS):
        # Switch to train mode
        MODEL.train()

        # Record loss sum before gradient descent
        LOSS_TRAIN_1 = torch.tensor(0.0)
        LOSS_TRAIN_2 = torch.tensor(0.0)

        # Gradient descent
        with tqdm(
            TRAIN_LOADER, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"
        ) as t:
            for x, y in t:
                # Forward propagation
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = MODEL(x)
                loss_1 = LOSS_WEIGHTS[0] * LOSS_1(output, y) + LOSS_WEIGHTS[1] * LOSS_2(
                    x, y
                )
                loss_2 = LOSS_1(output, y)

                # Backward propagation
                OPTIMIZER.zero_grad()
                if LOSS_TRAIN_AVERAGE_2 < LOSS_SWITCH_VALUE:
                    # Train with Loss 2 when MODEL is correct enough
                    loss_2.backward()
                    # Fix parameters of specified layer
                    if len(FREEZE_EPOCH) == 1:
                        if epoch >= FREEZE_EPOCH[0] and FREEZE_LAYER is not None:
                            for layer in FREEZE_LAYER:
                                freeze_param(
                                    Layer=layer,
                                    Filter_Weight=FILTER_WEIGHTS[0],
                                    Tol_Ratio=TOL_RATIO[0],
                                )

                    else:
                        for i in range(len(FREEZE_EPOCH)):
                            if epoch >= FREEZE_EPOCH[i]:
                                freeze_param(
                                    Layer=FREEZE_LAYER[i],
                                    Filter_Weight=FILTER_WEIGHTS[i],
                                    Tol_Ratio=TOL_RATIO[i],
                                )
                else:
                    # Train with Loss 1 when MODEL is not correct enough
                    loss_1.backward()

                # Gradient clipping
                clip_grad_norm_(MODEL.parameters(), GRAD_MAX)

                OPTIMIZER.step()
                LOSS_TRAIN_1 += loss_1.item()
                LOSS_TRAIN_2 += loss_2.item()
                t.set_postfix(loss_1=loss_1.item(), loss_2=loss_2.item())

        LOSS_TRAIN_AVERAGE_1 = LOSS_TRAIN_1 / len(TRAIN_LOADER)
        LOSS_TRAIN_AVERAGE_2 = LOSS_TRAIN_2 / len(TRAIN_LOADER)
        training_losses_1.append(LOSS_TRAIN_AVERAGE_1)
        training_losses_2.append(LOSS_TRAIN_AVERAGE_2)

        # Model evaluation
        LOSS_VALIDATION_AVERAGE_1, LOSS_VALIDATION_AVERAGE_2 = compute_loss_no_grad(
            VALIDATION_LOADER
        )
        validation_losses_1.append(LOSS_VALIDATION_AVERAGE_1)
        validation_losses_2.append(LOSS_VALIDATION_AVERAGE_2)

    print(
        "====================================Finish training====================================\n"
    )

    return (
        training_losses_1,
        validation_losses_1,
        training_losses_2,
        validation_losses_2,
    )


def TRAIN_WITH_PROGRESS_BAR_ONE_LOSS(
    MODEL,
    CONFIG,
    OPTIMIZER,
    TRAIN_LOADER,
    VALIDATION_LOADER,
    LOSS_TYPE=nn.MSELoss(),
):
    # Load config
    config = CONFIG["TRAIN"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    DEVICE = GET_DEVICE(config["DEVICE"])
    GRAD_MAX = config["GRAD_MAX"]

    print("PyTorch Version:", torch.__version__)
    INIT_WEIGHTS(MODEL, config)
    print("Weight initialized with", config["INIT"], "Initialization")
    print("Training on", DEVICE)
    print(
        "====================================Start training===================================="
    )
    # Transfer model to selected device
    MODEL.to(DEVICE)

    def compute_loss_no_grad(DATA_LOADER):
        MODEL.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in DATA_LOADER:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                output = MODEL(x)
                loss = LOSS_TYPE(output, y)
                total_loss += loss.item()
        return total_loss / len(DATA_LOADER)

    # Loss when not trained
    LOSS_TRAIN_AVERAGE = compute_loss_no_grad(TRAIN_LOADER)
    LOSS_VALIDATION_AVERAGE = compute_loss_no_grad(VALIDATION_LOADER)

    # loss recorders
    train_losses = [
        LOSS_TRAIN_AVERAGE,
    ]
    validation_losses = [
        LOSS_VALIDATION_AVERAGE,
    ]

    for epoch in range(NUM_EPOCHS):
        # Switch to train mode
        MODEL.train()

        # Record loss sum in 1 epoch
        LOSS_TRAIN = torch.tensor(0.0)
        LOSS_VALIDATION = torch.tensor(0.0)

        # Gradient descent
        with tqdm(
            TRAIN_LOADER, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"
        ) as t:
            for x, y in t:
                # Forward propagation
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = MODEL(x)
                loss = LOSS_TYPE(output, y)

                # Backward propagation
                OPTIMIZER.zero_grad()
                loss.backward()

                # Gradient clipping
                if GRAD_MAX is not None:
                    clip_grad_norm_(MODEL.parameters(), GRAD_MAX)

                OPTIMIZER.step()
                t.set_postfix(loss=loss.item())
                LOSS_TRAIN += loss.item()

        LOSS_TRAIN_AVERAGE = LOSS_TRAIN / len(TRAIN_LOADER)
        train_losses.append(LOSS_TRAIN_AVERAGE)

        # Model evaluation
        LOSS_VALIDATION_AVERAGE = compute_loss_no_grad(VALIDATION_LOADER)
        validation_losses.append(LOSS_VALIDATION_AVERAGE)

    print(
        "====================================Finish training====================================\n"
    )

    return train_losses, validation_losses
