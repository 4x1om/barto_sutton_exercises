from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.9
THETA = 1


def clamp_tuple(t):
    return tuple(max(0, min(x, 20)) for x in t)


def main():
    p = poisson.pmf
    p_table = {}
    for q1 in range(9):
        for q2 in range(11):
            for t1 in range(9):
                for t2 in range(7):
                    prob = p(q1, 3) * p(q2, 4) * p(t1, 3) * p(t2, 2)
                    if prob >= 2e-5:
                        p_table[q1, q2, t1, t2] = prob
    print(f"Precomputation complete: memoized {len(p_table)} probabilities")
    # for key, value in sorted(p_table.items(), key=lambda item: item[1], reverse=True):
    #     print(f"{key=}, {p_table[key]=}")
    # print(len(p_table))
    # return

    v = np.zeros((21, 21))
    pi = np.zeros((21, 21), dtype=int)
    for c1 in range(21):
        for c2 in range(21):
            v[c1, c2] = np.random.normal(500, 50)
            pi[c1, c2] = 0
            # np.random.uniform(-5, 6)

    def visualize(iteration):
        plt.imshow(pi, cmap="viridis", interpolation="nearest", origin="lower")
        plt.colorbar()
        plt.xlabel("Cars at the second location")
        plt.ylabel("Cars at the first location")
        plt.title(f"π{iteration}")
        plt.show()

    def eval():
        delta = float("inf")
        while delta > THETA:
            delta = 0
            for c1 in range(21):
                for c2 in range(21):
                    v_old = v[c1, c2]
                    a = pi[c1, c2]
                    v[c1, c2] = 0
                    for (q1, q2, t1, t2), p in p_table.items():
                        r = 10 * (min(c1 - a, q1) + min(c2 + a, q2)) - 2 * abs(a)
                        c1_prime = min(max(0, c1 - a - q1) + t1, 20)
                        c2_prime = min(max(0, c2 + a - q2) + t2, 20)
                        v[c1, c2] += p * (r + GAMMA * v[c1_prime, c2_prime])
                    delta = max(delta, abs(v[c1, c2] - v_old))
                    # print(f"updated v[{c1}, {c2}] from {v_old} -> {v[c1, c2]}")

    def improve():
        is_stable = True
        for c1 in range(21):
            for c2 in range(21):
                old_action = pi[c1, c2]
                action_values = []
                for a in range(max(-5, -c2), min(c1, 5) + 1):
                    action_value = 0
                    for (q1, q2, t1, t2), p in p_table.items():
                        r = 10 * (min(c1 - a, q1) + min(c2 + a, q2)) - 2 * abs(a)
                        c1_prime = min(max(0, c1 - a - q1) + t1, 20)
                        c2_prime = min(max(0, c2 + a - q2) + t2, 20)
                        action_value += p * (r + GAMMA * v[c1_prime, c2_prime])
                    action_values.append((a, action_value))
                pi[c1, c2] = max(action_values, key=lambda t: t[1])[0]
                if old_action != pi[c1, c2]:
                    is_stable = False
        return is_stable

    iteration = 0
    is_stable = False
    while not is_stable:
        eval()
        is_stable = improve()
        iteration += 1
        visualize(iteration)


if __name__ == "__main__":
    main()
