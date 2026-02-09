import numpy as np
#Interval: x \in I
I=[0,1]
c=1 #wave speed
NX = 100*(I[1]-I[0])
x = np.linspace(I[0],I[1],NX)
dx = x[1]-x[0]

TI = [0,1] #Time Interval
NT = 1000*(TI[1]-TI[0])
t = np.linspace(TI[0],TI[1],int(NT))
dt = t[1]-t[0]

MAX_CHEBYSHEV_DEGREE=100
NUMBER_OF_RANDOM_FUNCTION=100000

#Generate Random function 
#by using Chebyshev polynomial
# with Dirichlet Boundary condition
def random_func_T(x):
    x_mapped = (2 * (x - I[0]) / (I[1] - I[0])) - 1
    cheb_coeffs = np.random.uniform(-1, 1, size=MAX_CHEBYSHEV_DEGREE+1)
    cheb_polys = [np.polynomial.chebyshev.Chebyshev.basis(i)(x_mapped) for i in range(len(cheb_coeffs))]
    func = sum(c * p*(x-I[0])*(I[1]-x) for c, p in zip(cheb_coeffs, cheb_polys))
    # Normalize the function to the range [-1, 1]
    max_abs_value = np.max(np.abs(func))
    if max_abs_value > 1:
        func /= max_abs_value

    return func


from scipy.integrate import solve_ivp
# Solve Maxwell's equations using finite difference method
def solve_ivp_fd(E_init, B_init, dt=dt):
    NX = len(E_init)

    def maxwell_eq(t, y):
        E = y[:NX]
        B = y[NX:]
        dEdt = np.zeros(NX)
        dBdt = np.zeros(NX)
        # Finite difference for spatial derivatives (central difference)
        dEdt[1:-1] = (c / dx) * (B[2:] - B[:-2]) / 2  # ∂B/∂x
        dBdt[1:-1] =(c / dx) * (E[2:] - E[:-2]) / 2  # ∂E/∂x
        
        return np.concatenate([dEdt, dBdt])

    # Initial conditions
    y0 = np.concatenate([E_init, B_init])
    
    # Solve using scipy's solve_ivp with RK45 method
    sol = solve_ivp(maxwell_eq, [0, dt], y0, method='RK45', rtol=1e-8, atol=1e-8)
    
    E_new = sol.y[:NX, -1].T
    B_new = sol.y[NX:, -1].T
    
    return E_new, B_new

if False:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    # Define initial functions
    # Precompute the random function values
    f = random_func_T(x)
    g = random_func_T(x)
    f0 = f.copy()
    g0 = g.copy()
    f1 = f.copy()
    g1 = g.copy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    line_f_ivpfd, = ax.plot(x, f, color='blue', label="Displacement (u_f)ivpfd")
    line_g_ivpfd, = ax.plot(x, g, color='red', label="Velocity (u_g)ivpfd")
    line_f_sefd, = ax.plot(x, f1, color='black', label="Displacement (u_f)sefd")
    line_g_sefd, = ax.plot(x, g1, color='orange', label="Velocity (u_g)sefd")
    line_f_init, = ax.plot(x, f0, '--', color='blue', alpha=0.5, label="Initial f(x)")
    line_g_init, = ax.plot(x, g0, '--', color='red', alpha=0.5, label="Initial g(x)")
    ax.set_xlim(I[0], I[1])
    ax.set_ylim(-max(f.max(), g.max(),) - 1, max(f.max(), g.max()) + 1)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    ax.set_title("Wave Equation Animation")

    def update(frame):
        global f,g
        f,g=solve_ivp_fd(f,g)
        line_f_ivpfd.set_ydata(f)
        line_g_ivpfd.set_ydata(g)
        # Initial f and g remain constant
        line_f_init.set_ydata(f0)
        line_g_init.set_ydata(g0)
        ax.set_title(f"Wave Equation Animation (t = {t[frame]:.2f})")
        return line_f_ivpfd, line_g_ivpfd, line_f_init, line_g_init

    ani = FuncAnimation(fig, update, frames=range(0, 100), interval=5, blit=True)
    plt.show()

#Purpose: generating data set and save it in simple-wave-data.npz
        #data format: input  = [f_i(x_0),...,f_i(x_d),g_i(x_0),...g_i(x_d)],
        #             target = [fnew_i(x_0),...,fnew_i(x_d),gnew_i(x_0),...gnew_i(x_d)]]
#where f and g are initial condition of wave equation u_{tt}=c^2*u_{xx}, and we use Dirichlet boundary condition.

# Generate Dataset
def generate_dataset(num_samples, x=x, dt=dt):
    dataset = []
    for sam in range(num_samples):
        if sam%1000==999:
            print(sam+1)
        f = random_func_T(x)  # Random initial displacement
        g = random_func_T(x)  # Random initial velocity

        f_new, g_new = solve_ivp_fd(f, g)
        input = np.concatenate([f, g])  # current state
        target = [f_new, g_new]  # Next state at x[j]
        dataset.append((input, target))
    return dataset

def save_dataset(dataset, filename="data.npz"):
    inputs = np.array([data[0] for data in dataset])
    targets = np.array([data[1] for data in dataset])

    np.savez(filename, inputs=inputs, targets=targets)



if __name__=="__main__":
    # Generate and save data
    num_samples = NUMBER_OF_RANDOM_FUNCTION
    delta_t=dt
    file_name=f"data.npz"
    dataset = generate_dataset(num_samples)
    print("ok")
    save_dataset(dataset,file_name)
    print(f"Dataset saved to {file_name} with {num_samples} samples.")

