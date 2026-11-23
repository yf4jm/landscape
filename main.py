# Example we should be working on

import numpy as np
from landscape.models.linear_regression import LinearRegression
import matplotlib.pyplot as plt

# ---------------- data ----------------
x = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 2.0, 4.0])

def plot_loss_log(model, exact=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(model.loss_history, "o-", ms=2.5, lw=1.2, label="train loss")
    if exact is not None:
        ax.axhline(exact, ls="--", color="red", lw=1.2,
                   label=f"floor = {exact:.5f}")
    ax.set(xlabel="iteration", ylabel="loss (log scale)",
           title="Loss vs. iteration (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.show()

model = LinearRegression(debug=True)
model.fit(x,y)
test_data = np.array([2])
plot_loss_log(model)
print(model.predict(test_data))

