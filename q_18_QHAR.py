"""
QHAR - Quantum Harmonic Analysis Regression
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/Users/4c/Desktop/GHQ/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
NUM_HARMONICS = 6
LAMBDA_REG = 0.01


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def value_to_features(v):
    theta = v * np.pi / 31.0
    return np.array([theta * (k + 1) for k in range(NUM_QUBITS)])


def harmonic_circuit(x, harmonic_k):
    qc = QuantumCircuit(NUM_QUBITS)

    freq = (harmonic_k + 1)
    for i in range(NUM_QUBITS):
        qc.ry(x[i] * freq, i)

    for i in range(NUM_QUBITS - 1):
        qc.cx(i, i + 1)

    for i in range(NUM_QUBITS):
        qc.rz(x[i] * freq * 0.5, i)

    for rep in range(harmonic_k % 3):
        for i in range(NUM_QUBITS):
            qc.ry(x[i] * freq * (rep + 2) * 0.3, i)
        for i in range(NUM_QUBITS - 1):
            qc.cx(i, i + 1)

    return qc


def compute_harmonic_features(X_feats):
    n = len(X_feats)
    all_features = []

    for k in range(NUM_HARMONICS):
        feats_k = []
        for feat in X_feats:
            circ = harmonic_circuit(feat, k)
            sv = Statevector.from_instruction(circ)
            probs = sv.probabilities()
            feats_k.append(probs)
        all_features.append(np.array(feats_k))

    return np.hstack(all_features)


def ridge_fit_predict(X, y, lam=LAMBDA_REG):
    alpha = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return X @ alpha


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    n_states = 1 << NUM_QUBITS
    X_feats = np.array([value_to_features(v) for v in range(n_states)])

    print(f"\n--- Quantum Harmonic Analysis ({NUM_QUBITS}q, "
          f"{NUM_HARMONICS} harmonika) ---")
    print(f"  Generisanje harmonickih feature-a...", end=" ", flush=True)
    feat_matrix = compute_harmonic_features(X_feats)
    print(f"{feat_matrix.shape[0]}x{feat_matrix.shape[1]}")

    print(f"\n--- QHAR regresija po pozicijama ---")
    dists = []
    for pos in range(7):
        y = build_empirical(draws, pos)
        pred = ridge_fit_predict(feat_matrix, y)
        pred = pred - pred.min()
        if pred.sum() > 0:
            pred /= pred.sum()
        dists.append(pred)

        top_idx = np.argsort(pred)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{pred[i]:.3f}" for i in top_idx)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QHAR, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Quantum Harmonic Analysis (5q, 6 harmonika) ---
  Generisanje harmonickih feature-a... 32x192

--- QHAR regresija po pozicijama ---
  Poz 1 [1-33]: 1:0.169 | 2:0.147 | 3:0.130
  Poz 2 [2-34]: 8:0.085 | 5:0.076 | 9:0.076
  Poz 3 [3-35]: 13:0.064 | 12:0.062 | 14:0.062
  Poz 4 [4-36]: 23:0.063 | 21:0.063 | 18:0.062
  Poz 5 [5-37]: 29:0.065 | 26:0.064 | 27:0.063
  Poz 6 [6-38]: 33:0.083 | 32:0.081 | 35:0.080
  Poz 7 [7-39]: 7:0.182 | 38:0.152 | 37:0.132

==================================================
Predikcija (QHAR, deterministicki, seed=39):
[1, 8, 13, 23, 29, 33, 38]
==================================================
"""



"""
QHAR - Quantum Harmonic Analysis Regression

6 kvantnih harmonika: svaka enkodira ulaz sa razlicitom frekvencijom (1x, 2x, ... 6x)
Visi harmonici imaju vise slojeva (dodatni Ry+CX repovi) - rastuci kapacitet
Feature vektor: Born verovatnoce iz svih 6 harmonika spojene = 6 x 32 = 192 dimenzije
Inspirisano Furieovom analizom: razliciti harmonici hvataju razlicite frekvencijske komponente signala
Ridge regresija nad bogatim harmonickim prostorom
Deterministicki, bez treniranja kola
"""
