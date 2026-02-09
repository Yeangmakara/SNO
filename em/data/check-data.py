import numpy as np
import matplotlib.pyplot as plt

data = np.load("simple-wave-c1-dx0.010101010101010102-dt0.001001001001001001-I01.npz")
inputs = data["inputs"]    # shape (10, 200)
targets = data["targets"]  # shape (10, 2, 100)

print(inputs.shape)    # (num_samples, 2*NX)
print(targets.shape)   # (num_samples, 2, NX)

NX = inputs.shape[1] // 2
x = np.linspace(0, 1, NX)

sample = 0
f_init = inputs[sample][:NX]
g_init = inputs[sample][NX:]

f_new = targets[sample][0]   # corrected here!
g_new = targets[sample][1]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, f_init, label="f(x), t=0")
plt.plot(x, f_new, label="f_new(x), t=Δt")
plt.xlabel("x")
plt.ylabel("Displacement")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, g_init, label="g(x), t=0")
plt.plot(x, g_new, label="g_new(x), t=Δt")
plt.xlabel("x")
plt.ylabel("Velocity")
plt.legend()

plt.tight_layout()
plt.show()
