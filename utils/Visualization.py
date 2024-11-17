import matplotlib.pyplot as plt
import numpy as np


def LOSS_EPOCH_DIAGRAM_TWO_LOSS(LOSS_HISTORY_TUPLE, CONFIG, LAYER_NAME=None):
    fig, ax = plt.subplots(2, 1)

    # Load loss history
    (
        Loss_1_Training_History,
        Loss_1_Validation_History,
        Loss_2_Training_History,
        Loss_2_Validation_History,
    ) = LOSS_HISTORY_TUPLE

    # Load config
    NUM_EPOCHS = CONFIG["TRAIN"]["NUM_EPOCHS"]
    FREEZE_EPOCH = CONFIG["TRAIN"]["FREEZE_EPOCH"]
    TITLE_FONTSIZE = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["TITLE_FONTSIZE"]
    LABEL_FONTSIZE = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["LABEL_FONTSIZE"]
    LOSS_EPOCH_YSCALE = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["Y_SCALE"]
    LOSS_EPOCH_SEGMENT = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["X_SEGMENT"]
    LINE_WIDTH = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["LINE_WIDTH"]
    Epoch_History = np.arange(0, NUM_EPOCHS + 1)

    # Color map for axvlines
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    def AXVLINE(ax, Epoch, Color, Layer=None):
        if Epoch != NUM_EPOCHS:
            if Layer != None:
                ax.axvline(
                    Epoch,
                    linestyle="--",
                    linewidth=LINE_WIDTH,
                    color=Color,
                    label=f"Freeze {Layer}",
                )
            else:
                ax.axvline(
                    Epoch,
                    linestyle="--",
                    linewidth=LINE_WIDTH,
                    color=Color,
                    label=f"Freeze at {Epoch}th Epoch",
                )

    def AX_LOSS_HISTORY(ax, EPOCH_HISTORY, LOSS_HISTORY, TITLE):
        ax.plot(
            EPOCH_HISTORY, LOSS_HISTORY[0], linewidth=LINE_WIDTH, label="Training Loss"
        )
        ax.plot(
            EPOCH_HISTORY,
            LOSS_HISTORY[1],
            linewidth=LINE_WIDTH,
            label="Validation Loss",
        )
        ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
        ax.set_xticks(
            np.linspace(
                0,
                NUM_EPOCHS,
                LOSS_EPOCH_SEGMENT + 1,
                dtype=int,
            )
        )
        ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Loss", fontsize=LABEL_FONTSIZE)
        ax.set_yscale(LOSS_EPOCH_YSCALE)
        ax.grid()

    AX_LOSS_HISTORY(
        ax[0],
        Epoch_History,
        (Loss_1_Training_History, Loss_1_Validation_History),
        "Loss with Sparity",
    )
    AX_LOSS_HISTORY(
        ax[1],
        Epoch_History,
        (Loss_2_Training_History, Loss_2_Validation_History),
        "Loss without Sparity",
    )

    if len(FREEZE_EPOCH) == 1:
        AXVLINE(ax=ax[0], Epoch=FREEZE_EPOCH[0], Color=colors[2])
        AXVLINE(ax=ax[1], Epoch=FREEZE_EPOCH[0], Color=colors[2])
    else:
        for i in range(len(FREEZE_EPOCH)):
            AXVLINE(
                ax=ax[0],
                Epoch=FREEZE_EPOCH[i],
                Color=colors[i + 2],
                Layer=LAYER_NAME[i],
            )
            AXVLINE(
                ax=ax[1],
                Epoch=FREEZE_EPOCH[i],
                Color=colors[i + 2],
                Layer=LAYER_NAME[i],
            )

    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    return fig


def LOSS_EPOCH_DIAGRAM_ONE_LOSS(LOSS_HISTORY_TUPLE, CONFIG):
    fig, ax = plt.subplots()

    # Load config
    NUM_EPOCHS = CONFIG["TRAIN"]["NUM_EPOCHS"]
    TITLE_FONTSIZE = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["TITLE_FONTSIZE"]
    LABEL_FONTSIZE = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["LABEL_FONTSIZE"]
    LOSS_EPOCH_YSCALE = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["Y_SCALE"]
    LOSS_EPOCH_SEGMENT = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["X_SEGMENT"]
    LINE_WIDTH = CONFIG["VISUALIZATION"]["LOSS_EPOCH_DIAGRAM"]["LINE_WIDTH"]
    Epoch_History = np.arange(0, NUM_EPOCHS + 1)

    def AX_LOSS_HISTORY(Ax, EPOCH_HISTORY, LOSS_HISTORY, TITLE):
        Ax.plot(
            EPOCH_HISTORY, LOSS_HISTORY[0], linewidth=LINE_WIDTH, label="Training Loss"
        )
        Ax.plot(
            EPOCH_HISTORY,
            LOSS_HISTORY[1],
            linewidth=LINE_WIDTH,
            label="Validation Loss",
        )
        Ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
        Ax.set_xticks(
            np.linspace(
                0,
                NUM_EPOCHS,
                LOSS_EPOCH_SEGMENT + 1,
                dtype=int,
            )
        )
        Ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
        Ax.set_ylabel("Loss", fontsize=LABEL_FONTSIZE)
        Ax.set_yscale(LOSS_EPOCH_YSCALE)
        Ax.grid()

    AX_LOSS_HISTORY(
        Ax=ax,
        EPOCH_HISTORY=Epoch_History,
        LOSS_HISTORY=LOSS_HISTORY_TUPLE,
        TITLE="Loss - Epoch Diagram",
    )
    ax.legend()
    plt.tight_layout()
    return fig


def VISUALIZE_MATRIX(MATRIX, CONFIG, TITLE):
    COLOR_MAP = CONFIG["COLOR_MAP"]
    TITLE_FONTSIZE = CONFIG["TITLE_FONTSIZE"]
    Fig, Ax = plt.subplots()
    CurrentPlot = Ax.pcolor(
        MATRIX, cmap=COLOR_MAP, vmin=-np.abs(MATRIX).max(), vmax=np.abs(MATRIX).max()
    )
    Ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
    Ax.invert_yaxis()
    plt.colorbar(CurrentPlot, ax=Ax)
    return Fig


def VISUALIZE_MATRIX_AX(Ax, MATRIX, CONFIG, TITLE):
    COLOR_MAP = CONFIG["COLOR_MAP"]
    TITLE_FONTSIZE = CONFIG["TITLE_FONTSIZE"]
    CurrentPlot = Ax.pcolor(
        MATRIX, cmap=COLOR_MAP, vmin=-np.abs(MATRIX).max(), vmax=np.abs(MATRIX).max()
    )
    Ax.set_title(TITLE, fontsize=TITLE_FONTSIZE)
    Ax.invert_yaxis()
    plt.colorbar(CurrentPlot, ax=Ax)


def VISUALIZE_NN(NN, CONFIG):
    NUM_LAYERS = len(NN)
    if CONFIG["NUM_ROW"] == -1 or CONFIG["NUM_COL"] == -1:
        NUM_ROW = int(np.floor(np.sqrt(NUM_LAYERS)))
        NUM_COL = int(np.ceil(np.sqrt(NUM_LAYERS)))
    else:
        NUM_ROW = CONFIG["NUM_ROW"]
        NUM_COL = CONFIG["NUM_COL"]
    INPUT_SHAPE = NN[0].weight.shape[1]
    FIG_SIZE = CONFIG["FIG_SIZE"]
    if FIG_SIZE[0] == -1 or FIG_SIZE[1] == -1:
        fig, ax = plt.subplots(NUM_ROW, NUM_COL, dpi=CONFIG["DPI"])
    else:
        fig, ax = plt.subplots(NUM_ROW, NUM_COL, figsize=FIG_SIZE, dpi=CONFIG["DPI"])
    if NUM_ROW == 1 or NUM_COL == 1:
        for i in range(NUM_LAYERS):
            data_temp = NN[i].weight.detach().to("cpu").numpy()
            VISUALIZE_MATRIX_AX(ax[i], data_temp, CONFIG, f"Layer {i+1}")
            if i == 0:
                ax[i].set_xticks(int(INPUT_SHAPE / 4) * np.arange(1, 5))
                ax[i].set_xticklabels(
                    [r"$r_i$", r"$v_i$", r"$\phi_{i}$", r"$\psi_{i}$"]
                )
    else:
        for i in range(NUM_LAYERS):
            data_temp = NN[i].weight.detach().to("cpu").numpy()
            VISUALIZE_MATRIX_AX(
                ax[i // NUM_COL, i % NUM_COL], data_temp, CONFIG, f"Layer {i+1}"
            )
            if i == 0:
                ax[i // NUM_COL, i % NUM_COL].set_xticks(
                    int(INPUT_SHAPE / 4) * np.arange(1, 5)
                )
                ax[i // NUM_COL, i % NUM_COL].set_xticklabels(
                    [r"$r_i$", r"$v_i$", r"$\phi_{i}$", r"$\psi_{i}$"]
                )
    plt.tight_layout()
    return fig
