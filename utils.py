import torch
import torch.nn as nn
from torch.linalg import pinv
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm


def GET_DEVICE(DEVICE=0):
    # Selecting training device
    if torch.cuda.is_available():
        device_name = f"cuda:{DEVICE}"
        if torch.cuda.device_count() > DEVICE:
            return device_name
        else:
            print(f"No such cuda device: {DEVICE}")
    return "cpu"


def INIT_WEIGHTS_XAVIER(MODEL):
    # Xavier initial weights
    for param in MODEL.parameters():
        if type(MODEL) == nn.Linear:
            nn.init.xavier_uniform_(param)
            MODEL.bias.data.fill_(0.01)


def INIT_WEIGHTS_ZERO(MODEL):
    # Zero initial weights
    for param in MODEL.parameters():
        nn.init.zeros_(param)


def Split2Loaders(INPUT, OUTPUT, BATCHSIZE, RATIO=0.8, SHUFFLE=True):
    training_size = int(RATIO * INPUT.shape[0])
    validation_size = INPUT.shape[0] - training_size
    training_dataset, validation_dataset = random_split(
        TensorDataset(INPUT, OUTPUT), [training_size, validation_size]
    )
    return DataLoader(training_dataset, BATCHSIZE, SHUFFLE), DataLoader(
        validation_dataset, BATCHSIZE, SHUFFLE
    )


def TRAIN_WITH_PROGRESS_BAR(
    MODEL,
    NUM_EPOCHS,
    OPTIMIZER,
    TRAIN_LOADER,
    VALIDATION_LOADER,
    LOSS_TUPLE=None,
    LOSS_WEIGHTS=None,
    DEVICE=0,
    GRAD_MAX=5,
    LOSS_SWITCH_VALUE=5,
):
    if LOSS_WEIGHTS is None:
        LOSS_WEIGHTS = [1, 1e-4]
    if LOSS_TUPLE is None:
        LOSS_TUPLE = [nn.MSELoss(), nn.MSELoss()]
    print("PyTorch Version:", torch.__version__)
    device = GET_DEVICE(DEVICE)
    print("Training on", device)
    print(
        "====================================Start training===================================="
    )
    # Transfer model to selected device
    MODEL.to(device)

    # loss for training and validation
    LOSS_1, LOSS_2 = LOSS_TUPLE

    # loss recorders
    training_losses_1 = []
    training_losses_2 = []
    validation_losses_1 = []
    validation_losses_2 = []

    for epoch in range(NUM_EPOCHS):
        # Switch to train mode
        MODEL.train()

        # Record loss sum in 1 epoch
        LOSS_TRAIN_1 = torch.tensor(0.0)
        LOSS_TRAIN_2 = torch.tensor(0.0)
        LOSS_VALIDATION_1 = torch.tensor(0.0)
        LOSS_VALIDATION_2 = torch.tensor(0.0)
        LOSS_TRAIN_AVERAGE_1 = torch.tensor(0.0)
        LOSS_TRAIN_AVERAGE_2 = torch.tensor(0.0)

        # Gradient descent
        with tqdm(
            TRAIN_LOADER, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"
        ) as t:
            for x, y in t:
                # Forward propagation
                x, y = x.to(device), y.to(device)
                output = MODEL(x)
                loss_1 = LOSS_WEIGHTS[0] * LOSS_1(output, y) + LOSS_WEIGHTS[1] * LOSS_2(
                    x, y
                )
                loss_2 = LOSS_1(output, y)

                # Backward propagation
                OPTIMIZER.zero_grad()
                if LOSS_TRAIN_AVERAGE_2 < LOSS_SWITCH_VALUE:
                    loss_2.backward()
                else:
                    loss_1.backward()

                # Gradient clipping
                clip_grad_norm_(MODEL.parameters(), GRAD_MAX)

                OPTIMIZER.step()
                t.set_postfix(loss_1=loss_1.item(), loss_2=loss_2.item())
                LOSS_TRAIN_1 += loss_1.item()
                LOSS_TRAIN_2 += loss_2.item()

        LOSS_TRAIN_AVERAGE_1 = LOSS_TRAIN_1 / len(TRAIN_LOADER)
        LOSS_TRAIN_AVERAGE_2 = LOSS_TRAIN_2 / len(TRAIN_LOADER)
        training_losses_1.append(LOSS_TRAIN_AVERAGE_1)
        training_losses_2.append(LOSS_TRAIN_AVERAGE_2)

        # Model evaluation
        MODEL.eval()
        with torch.no_grad():
            for x, y in VALIDATION_LOADER:
                x, y = x.to(device), y.to(device)
                output = MODEL(x)
                loss_1 = LOSS_WEIGHTS[0] * LOSS_1(output, y) + LOSS_WEIGHTS[1] * LOSS_2(
                    x, y
                )
                loss_2 = LOSS_1(output, y)
                LOSS_VALIDATION_1 += loss_1.item()
                LOSS_VALIDATION_2 += loss_2.item()

        LOSS_VALIDATION_AVERAGE_1 = LOSS_VALIDATION_1 / len(VALIDATION_LOADER)
        LOSS_VALIDATION_AVERAGE_2 = LOSS_VALIDATION_2 / len(VALIDATION_LOADER)
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


def TRAIN_NO_PROGRESS_BAR(
    MODEL,
    NUM_EPOCHS,
    OPTIMIZER,
    TRAIN_LOADER,
    VALIDATION_LOADER,
    LOSS=nn.MSELoss(),
    DEVICE=0,
    GRAD_MAX=5,
):
    # Transfer model to selected device
    device = GET_DEVICE(DEVICE)
    MODEL.to(device)

    # loss recorders
    training_losses = []
    validation_losses = []

    for _ in range(NUM_EPOCHS):
        # Switch to train mode
        MODEL.train()

        # Record loss sum in 1 epoch
        LOSS_TRAIN = torch.tensor(0.0)
        LOSS_VALIDATION = torch.tensor(0.0)

        # Gradient descent
        for x, y in TRAIN_LOADER:
            # Forward propagation
            x, y = x.to(device), y.to(device)
            output = MODEL(x)
            loss = LOSS(output, y)

            # Backward propagation
            OPTIMIZER.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(MODEL.parameters(), GRAD_MAX)

            OPTIMIZER.step()
            LOSS_TRAIN += loss.item()

        LOSS_TRAIN_AVERAGE = LOSS_TRAIN / len(TRAIN_LOADER)
        training_losses.append(LOSS_TRAIN_AVERAGE)

        # Model evaluation
        MODEL.eval()
        with torch.no_grad():
            for x, y in VALIDATION_LOADER:
                x, y = x.to(device), y.to(device)
                output = MODEL(x)
                loss = LOSS(output, y)
                LOSS_VALIDATION += loss.item()

        LOSS_VALIDATION_AVERAGE = LOSS_VALIDATION / len(VALIDATION_LOADER)
        validation_losses.append(LOSS_VALIDATION_AVERAGE)

    return training_losses, validation_losses
