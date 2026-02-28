import os

# Force headless Matplotlib backend for ALL tests to avoid GUI windows / hangs on Windows.
# Must be set before importing matplotlib.pyplot anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")


def pytest_configure(config):
    # Import here to ensure env var is set first.
    import matplotlib

    matplotlib.use("Agg", force=True)

    # Make sure no test blocks on plt.show()
    import matplotlib.pyplot as plt

    plt.show = lambda *args, **kwargs: None
