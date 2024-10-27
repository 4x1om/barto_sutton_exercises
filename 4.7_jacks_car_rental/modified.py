import random
from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

# POISSON_BOUNDS = [0, 0, 7, 9, 11]
POISSON_BOUNDS = [0, 0, 9, 11, 13]

GAMMA = 0.9
THETA = 0.1


def main():

    poissons = {}
    for x in range(max(POISSON_BOUNDS)):
        poissons[(x, 2)] = poisson.pmf(x, 2)
        poissons[(x, 3)] = poisson.pmf(x, 3)
        poissons[(x, 4)] = poisson.pmf(x, 4)

    states = {(c1, c2) for c1 in range(21) for c2 in range(21)}
    rewards = {}
    p1, p2 = {}, {}

    for c1, c2 in states:
        for a in range(-5, 7):
            m1, m2 = c1 - a, c2 + a
            if not (0 <= m1 <= 20 and 0 <= m2 <= 20):
                continue
            rewards[(c1, c2), a] = 2 - 2 * a if a > 0 else 2 * a
            rewards[(c1, c2), a] -= 4 * (m1 > 10)
            rewards[(c1, c2), a] -= 4 * (m2 > 10)
            for q1 in range(POISSON_BOUNDS[3]):
                for q2 in range(POISSON_BOUNDS[4]):
                    q_prob = poissons[q1, 3] * poissons[q2, 4]
                    r = 10 * (min(m1, q1) + min(m2, q2))
                    rewards[(c1, c2), a] += q_prob * r

    for c, c_prime in states:
        p1[c, c_prime] = p2[c, c_prime] = 0

    # m1 refers to the number of cars in the morning after moving cars over the night
    for m1 in range(21):
        for q1 in range(POISSON_BOUNDS[3]):
            for t1 in range(POISSON_BOUNDS[3]):
                prob = poissons[q1, 3] * poissons[t1, 3]
                c1_prime = min(max(0, m1 - q1) + t1, 20)
                p1[m1, c1_prime] += prob

    for m2 in range(21):
        for q2 in range(POISSON_BOUNDS[4]):
            for t2 in range(POISSON_BOUNDS[2]):
                prob = poissons[q2, 4] * poissons[t2, 2]
                c2_prime = min(max(0, m2 - q2) + t2, 20)
                p2[m2, c2_prime] += prob

    print(f"Precomputated the dynamics of the environment")

    v = np.zeros((21, 21))
    pi = np.zeros((21, 21), dtype=int)

    def eval():
        delta = float("inf")
        while delta > THETA:
            delta = 0
            for c1, c2 in states:
                v_old = v[c1, c2]
                a = pi[c1, c2]
                m1, m2 = c1 - a, c2 + a
                v_new = rewards[(c1, c2), a]
                for c1_prime, c2_prime in states:
                    prob = p1[m1, c1_prime] * p2[m2, c2_prime]
                    v_new += GAMMA * prob * v[c1_prime, c2_prime]
                v[c1, c2] = v_new
                delta = max(delta, abs(v_new - v_old))

    def improve():
        is_stable = True
        for c1, c2 in states:
            old_action = pi[c1, c2]
            action_values = []
            for a in range(-5, 6):
                m1, m2 = c1 - a, c2 + a
                if not (0 <= m1 <= 20 and 0 <= m2 <= 20):
                    continue
                q = rewards[(c1, c2), a]
                for c1_prime, c2_prime in states:
                    prob = p1[m1, c1_prime] * p2[m2, c2_prime]
                    q += GAMMA * prob * v[c1_prime, c2_prime]
                action_values.append((a, q))
            pi[c1, c2] = max(action_values, key=lambda t: t[1])[0]
            if old_action != pi[c1, c2]:
                is_stable = False
        return is_stable

    def visualize(iteration, save=False):
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        im = ax1.imshow(pi, cmap="viridis", interpolation="nearest", origin="lower")
        fig.colorbar(im, ax=ax1)
        ax1.set_xlabel("Cars at the second location")
        ax1.set_ylabel("Cars at the first location")
        ax1.set_title(f"π_{iteration}")

        X = np.arange(0, 21)
        Y = np.arange(0, 21)
        X, Y = np.meshgrid(X, Y)
        surf = ax2.plot_surface(X, Y, v, cmap="viridis")
        # fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
        ax2.set_xlabel("Cars at the second location")
        ax2.set_ylabel("Cars at the first location")
        ax2.set_title(f"v_π_{iteration}")

        plt.tight_layout()
        if save:
            plt.savefig("optimal_policy.png", format="png")
        else:
            plt.show()

    iteration = 0
    visualize(0)
    is_stable = False
    while not is_stable:
        eval()
        is_stable = improve()
        iteration += 1
        visualize(iteration, is_stable)
    print(f"π_{iteration} is the optimal policy.")


if __name__ == "__main__":
    main()
