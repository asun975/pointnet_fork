import matplotlib.pyplot as plt


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results model checkpoints.

    Args:
        results (list): a list of dictionaries containing,
            {"train_loss": [...],
             "test_acc": [...],}
        for each saved model checkpoint.
    """

    # Plot training loss
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)

    for index, model in enumerate(results):
        chkpoint_label = f"Model {index}"
        loss = model["train_loss"]
        batches = range(len(model["train_loss"]))
        plt.plot(batches, loss, label=chkpoint_label)

    plt.title("train Loss")
    plt.xlabel("Batches")
    plt.legend()
    #plt.show()

    # Plot test accuracy
    acc = []
    batches = range(len(results))
    '''for model in results:
        print(model)'''
    for model in results:
        print(model)
        acc.append(model["test_acc"])
    plt.subplot(1, 2, 2)
    plt.plot(batches, acc, label="Saved models")
    plt.legend()
    plt.show()
    '''labels = [f"Model{index}" for index in results]
    acc = [chkpoint["test_acc"] for chkpoint in results]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, acc, width=0.4)
    plt.xlabel("Model checkpoints")
    plt.ylabel("Accuracy %")
    plt.title("Test Accuracy of model checkpoints")
    plt.legend()
    plt.show()'''
