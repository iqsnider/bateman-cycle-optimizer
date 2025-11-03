import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


class DecaySystem:
    def __init__(self, parent_half_life, daughter_half_life, implantation_rate):
        """
        Initialize the decay system

        Parameters:
        parent_half_life: half-life of parent nucleus in seconds
        daughter_half_life: half-life of daughter nucleus in seconds  
        implantation_rate: implantation rate R (nuclei/second)
        """
        self.lambda_p = np.log(2) / parent_half_life
        self.lambda_d = np.log(2) / daughter_half_life
        self.R = implantation_rate

    def decay_equations(self, t, y):
        """
        System of differential equations for parent-daughter decay
        dy/dt = [dNp/dt, dNd/dt]
        """
        N_p, N_d = y
        dNp_dt = self.R - self.lambda_p * N_p
        dNd_dt = self.lambda_p * N_p - self.lambda_d * N_d
        return [dNp_dt, dNd_dt]

    def solve_system(self, t_cycle, t_eval=None):
        """
        Solve the system of equations for a given cycle time
        """
        t_span = (0, t_cycle)
        y0 = [0, 0]  # Initial conditions: N_p(0) = 0, N_d(0) = 0

        if t_eval is None:
            t_eval = np.linspace(0, t_cycle, 1000)

        solution = solve_ivp(
            self.decay_equations,
            t_span,
            y0,
            t_eval=t_eval,
            method='RK45'
        )

        return solution

    def calculate_activities(self, solution):
        """
        Calculate activities from the solution
        A_p = lambda_p * N_p, A_d = lambda_d * N_d
        """
        t = solution.t
        N_p, N_d = solution.y
        A_p = self.lambda_p * N_p
        A_d = self.lambda_d * N_d
        return t, A_p, A_d

    def calculate_snr(self, t_cycle):
        """
        Calculate Signal-to-Noise Ratio for a given cycle time
        SNR = (integral of daughter activity) / (integral of parent activity)
        """
        solution = self.solve_system(t_cycle)
        t, A_p, A_d = self.calculate_activities(solution)

        # Calculate integrals using trapezoidal rule
        integral_A_p = np.trapz(A_p, t)
        integral_A_d = np.trapz(A_d, t)

        # Avoid division by zero
        if integral_A_p == 0:
            return 0

        snr = integral_A_d / integral_A_p
        return snr

    def optimize_cycle_time(self, t_bounds=(0.1, 100), method='bounded'):
        """
        Find the cycle time that maximizes SNR

        Parameters:
        t_bounds: tuple of (min_time, max_time) in seconds
        method: optimization method ('bounded' or 'minimize')
        """
        if method == 'bounded':
            result = minimize_scalar(
                lambda t: -self.calculate_snr(t),  # Negative for maximization
                bounds=t_bounds,
                method='bounded'
            )
        else:
            result = minimize_scalar(
                lambda t: -self.calculate_snr(t),
                bounds=t_bounds,
                method='golden'
            )

        optimal_t_cycle = result.x
        max_snr = -result.fun

        return optimal_t_cycle, max_snr

    def plot_system(self, t_cycle):
        """
        Plot the activities and populations for a given cycle time
        """
        solution = self.solve_system(t_cycle)
        t, A_p, A_d = self.calculate_activities(solution)
        N_p, N_d = solution.y

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot activities
        ax1.plot(t, A_p, 'b-', label='Parent Activity (A_p)', linewidth=2)
        ax1.plot(t, A_d, 'r-', label='Daughter Activity (A_d)', linewidth=2)
        ax1.set_ylabel('Activity (decays/s)')
        ax1.set_title(f'Activities for t_cycle = {t_cycle:.2f} s')
        ax1.legend()
        ax1.grid(True)

        # Plot populations
        ax2.plot(t, N_p, 'b--', label='Parent Population (N_p)', linewidth=2)
        ax2.plot(t, N_d, 'r--', label='Daughter Population (N_d)', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Population')
        ax2.set_title('Nuclear Populations')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        return fig

# Example usage for the specific isotopes you mentioned


def analyze_systems():
    """
    Analyze the specific Ba-La decay systems
    """
    # Define the systems
    systems = {
        '147Ba-147La': DecaySystem(
            parent_half_life=0.894,    # 147Ba
            daughter_half_life=4.06,   # 147La
            implantation_rate=1000     # nuclei/s (you can adjust this)
        ),
        '148Ba-148La': DecaySystem(
            parent_half_life=0.619,    # 148Ba
            daughter_half_life=1.411,  # 148La
            implantation_rate=1000     # nuclei/s
        )
    }

    results = {}

    for name, system in systems.items():
        print(f"\nAnalyzing {name} system:")
        print(f"Parent λ = {system.lambda_p:.4f} s⁻¹")
        print(f"Daughter λ = {system.lambda_d:.4f} s⁻¹")

        # Find optimal cycle time
        optimal_t, max_snr = system.optimize_cycle_time(t_bounds=(0.1, 20))

        print(f"Optimal cycle time: {optimal_t:.3f} s")
        print(f"Maximum SNR: {max_snr:.4f}")

        # Calculate some additional metrics
        solution = system.solve_system(optimal_t)
        t, A_p, A_d = system.calculate_activities(solution)

        max_A_p = np.max(A_p)
        max_A_d = np.max(A_d)
        contamination_ratio = np.trapz(
            A_p, t) / np.trapz(A_d, t) if np.trapz(A_d, t) > 0 else np.inf

        print(f"Max parent activity: {max_A_p:.2f} decays/s")
        print(f"Max daughter activity: {max_A_d:.2f} decays/s")
        print(
            f"Contamination ratio (parent/daughter): {contamination_ratio:.4f}")

        results[name] = {
            'system': system,
            'optimal_t_cycle': optimal_t,
            'max_snr': max_snr,
            'contamination_ratio': contamination_ratio
        }

        # Plot the optimal system
        system.plot_system(optimal_t)
        plt.suptitle(f'{name} - Optimal t_cycle = {optimal_t:.3f} s')
        plt.show()

    return results


def snr_vs_cycle_time_analysis(system, t_range=np.linspace(0.1, 20, 100)):
    """
    Analyze how SNR varies with cycle time
    """
    snr_values = []
    for t in t_range:
        snr = system.calculate_snr(t)
        snr_values.append(snr)

    plt.figure(figsize=(10, 6))
    plt.plot(t_range, snr_values, 'b-', linewidth=2)
    plt.xlabel('Cycle Time (s)')
    plt.ylabel('SNR (A_d integral / A_p integral)')
    plt.title('SNR vs Cycle Time')
    plt.grid(True)

    # Mark the maximum
    max_idx = np.argmax(snr_values)
    plt.plot(t_range[max_idx], snr_values[max_idx], 'ro', markersize=8,
             label=f'Max SNR = {snr_values[max_idx]:.3f} at t = {t_range[max_idx]:.2f} s')
    plt.legend()

    return t_range, snr_values


# Run the analysis
if __name__ == "__main__":
    # Example with one system
    system = DecaySystem(
        parent_half_life=0.894,    # 147Ba
        daughter_half_life=4.06,   # 147La
        implantation_rate=1000     # Adjust based on your beam current
    )

    # Find optimal cycle time
    optimal_t, max_snr = system.optimize_cycle_time()
    print(f"Optimal cycle time: {optimal_t:.3f} seconds")
    print(f"Maximum SNR: {max_snr:.4f}")

    # Plot SNR vs cycle time
    t_range, snr_values = snr_vs_cycle_time_analysis(system)

    # Plot the optimal system behavior
    system.plot_system(optimal_t)
    plt.show()

    # Analyze all systems
    results = analyze_systems()
