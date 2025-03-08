from naml.modules import Generator, Tuple, List
from naml.modules import torch, plt, tqdm, np


def simple(m: torch.tensor, title: str = ""):
    plt.plot(m.detach().cpu().numpy())
    plt.title(title)
    plt.grid()


def simple_animated(
    m: Generator[Tuple[float], None, None],
    n_dim: int = 1,
    x_lim_min: float = 1,
    label_x: str = "epoch",
    labels_y: List[str] = ["loss"],
    title: str = "",
):
    from IPython import display

    assert n_dim == len(labels_y)
    fig, ax = plt.subplots(1, 1)
    ax.grid()
    ax.set_title(title)
    lines = [ax.add_line(plt.Line2D([], [])) for _ in range(n_dim)]
    px, pys = np.array([]), [np.array([]) for _ in range(n_dim)]
    ax.set_xlabel(label_x)
    for i, label_y in enumerate(labels_y):
        lines[i].set_label(label_y)
        lines[i].set_color(f"C{i}")
    disp = display.display(plt.gcf(), display_id=True)
    y_min, y_max = float("inf"), float("-inf")
    fig.legend()
    for smp in m:
        assert len(smp) == n_dim
        px = np.append(px, len(px))
        ax.set_xlim(0, max(x_lim_min, len(px)))
        for i, y in enumerate(smp):
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            line = lines[i]
            pys[i] = np.append(pys[i], y)
            line.set_data(px, pys[i])
        if y_min != y_max:
            ax.set_ylim(y_min, y_max)
        disp.update(fig)
    fig.clear()


def heatmap(
    m: torch.Tensor,
    cmap: str = "Reds",
    label_x: str = "",
    label_y: str = "",
    title: str = "",
):
    if m.ndim == 4:
        nrow, ncol = m.size()[:2]
    else:
        assert m.ndim == 2
        nrow, ncol = 1, 1
    fig, axes = plt.subplots(nrow, ncol, squeeze=False)
    for i in range(nrow):
        for j in range(ncol):
            ax = axes[i, j]
            if m.ndim == 4:
                ax.imshow(m[i, j].detach().cpu().numpy(), cmap=cmap)
            else:
                ax.imshow(m.detach().cpu().numpy(), cmap=cmap)
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            ax.set_title(title)
    fig.colorbar(ax.get_images()[0], ax=axes, orientation="vertical")


def kernel_regression(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    x_train: torch.Tensor,
):
    plt.plot(x_test, y_true, label="True")
    plt.plot(x_test, y_hat, label="Predict", color="blue")
    plt.scatter(x_train, y_train, label="Train", color="red")
    plt.legend()
    plt.grid()
