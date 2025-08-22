import numpy as np
from typing import Tuple, List, Dict, Iterable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt


# =========================
# HMM with two training modes
# =========================
class HiddenMarkovModel:
    def __init__(self, N: int, T: int, init: str = "uniform", seed: int = 42):
        """
        N: number of hidden states
        T: number of observation symbols (vocab size)
        init: "uniform" or "dirichlet" (random)
        """
        self.N = int(N)
        self.T = int(T)
        rng = np.random.default_rng(seed)

        if init == "dirichlet":
            self.pi = rng.dirichlet(np.ones(self.N)).astype(np.float32)
            self.A  = rng.dirichlet(np.ones(self.N), size=self.N).astype(np.float32)
            self.B  = rng.dirichlet(np.ones(self.T), size=self.N).astype(np.float32)
        else:
            self.pi = np.ones(self.N, dtype=np.float32) / self.N
            self.A  = np.ones((self.N, self.N), dtype=np.float32) / self.N
            self.B  = np.ones((self.N, self.T), dtype=np.float32) / self.T

    def save(self, filepath: str) -> None:
        """
        Save the model parameters to a file.
        """
        np.savez_compressed(filepath, pi=self.pi, A=self.A, B=self.B)

    @classmethod
    def load(cls, filepath: str) -> "HiddenMarkovModel":
        """
        Load the model parameters from a file.
        """
        data = np.load(filepath)
        model = cls(N=data["A"].shape[0], T=data["B"].shape[1])
        model.pi = data["pi"]
        model.A = data["A"]
        model.B = data["B"]
        return model

    # --------- Forward / Backward / Viterbi ----------
    def forward(self, O: np.ndarray) -> Tuple[np.ndarray, float]:
        T_len = len(O)
        fwd = np.zeros((T_len, self.N), dtype=np.float32)
        fwd[0] = self.pi * self.B[:, O[0]]
        for t in range(1, T_len):
            # fwd[t][s] = sum_k fwd[t-1][k] * A[k,s] * B[s, O[t]]
            fwd[t] = (fwd[t-1] @ self.A) * self.B[:, O[t]]
        return fwd, float(np.sum(fwd[-1]))

    def backward(self, O: np.ndarray) -> Tuple[np.ndarray, float]:
        T_len = len(O)
        back = np.zeros((T_len, self.N), dtype=np.float32)
        back[-1] = 1.0
        for t in range(T_len - 2, -1, -1):
            # back[t][s] = sum_j A[s,j] * B[j,O[t+1]] * back[t+1][j]
            back[t] = self.A @ (self.B[:, O[t+1]] * back[t+1])
        prob = float(np.sum(self.pi * self.B[:, O[0]] * back[0]))
        return back, prob

    def viterbi(self, O: np.ndarray) -> Tuple[np.ndarray, float]:
        T_len = len(O)
        v = np.zeros((T_len, self.N), dtype=np.float32)
        bptr = np.zeros((T_len, self.N), dtype=int)

        v[0] = self.pi * self.B[:, O[0]]
        bptr[0] = -1
        for t in range(1, T_len):
            # For each state s, choose best previous k
            seq_probs = v[t-1][:, None] * self.A  # shape: (N, N) -> k->s
            bptr[t] = np.argmax(seq_probs, axis=0)
            v[t]    = seq_probs[bptr[t], np.arange(self.N)] * self.B[:, O[t]]

        max_prob = float(np.max(v[-1]))
        last = int(np.argmax(v[-1]))

        path = np.zeros(T_len, dtype=int)
        for t in range(T_len - 1, -1, -1):
            path[t] = last
            last = bptr[t, last]
        return path, max_prob

    # --------- Supervised training ----------
    def fit_supervised(self, X: List[np.ndarray], Y: List[np.ndarray]) -> None:
        """
        X: list of observation sequences (word indices)
        Y: list of gold state sequences (tag indices), same lengths as X
        Estimates MLE for pi, A, B.
        """
        pi = np.zeros(self.N, dtype=np.float32)
        A  = np.zeros((self.N, self.N), dtype=np.float32)
        B  = np.zeros((self.N, self.T), dtype=np.float32)

        for x_seq, y_seq in zip(X, Y):
            if len(y_seq) == 0: 
                continue
            pi[y_seq[0]] += 1.0
            for t in range(len(y_seq) - 1):
                A[y_seq[t], y_seq[t+1]] += 1.0
            for t in range(len(y_seq)):
                B[y_seq[t], x_seq[t]] += 1.0

        # Normalize with smoothing to avoid zero rows
        self.pi = self._safe_row_norm(pi)
        self.A  = self._safe_row_norm(A)
        self.B  = self._safe_row_norm(B)

    # --------- Unsupervised training (Baum–Welch, memory-efficient) ----------
    def fit_unsupervised(self, X: List[np.ndarray], n_iters: int = 10) -> None:
        for _ in tqdm(range(n_iters)):
            # Accumulators across all sentences
            pi_acc  = np.zeros(self.N, dtype=np.float32)
            A_num   = np.zeros((self.N, self.N), dtype=np.float32)
            A_den   = np.zeros(self.N, dtype=np.float32)
            B_num   = np.zeros((self.N, self.T), dtype=np.float32)
            B_den   = np.zeros(self.N, dtype=np.float32)

            for O in X:
                if len(O) == 0:
                    continue
                alpha, _ = self.forward(O)
                beta,  _ = self.backward(O)
                # gamma[t, s] ∝ alpha[t, s] * beta[t, s]
                gamma = alpha * beta
                gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-12

                # Accumulate pi, B
                pi_acc += gamma[0]
                B_den  += np.sum(gamma, axis=0)
                # For each timestep, add gamma[t] into column of the observed symbol
                for t, ot in enumerate(O):
                    B_num[:, ot] += gamma[t]

                # A via on-the-fly xi accumulation, no full epsilon tensor
                for t in range(len(O) - 1):
                    # xi[i, j] ∝ alpha[t,i] * A[i,j] * B[j, O[t+1]] * beta[t+1, j]
                    xi = (alpha[t][:, None] * self.A) * (self.B[:, O[t+1]] * beta[t+1])[None, :]
                    xi_sum = np.sum(xi)
                    if xi_sum > 0:
                        xi /= xi_sum
                    A_num += xi
                    A_den += np.sum(xi, axis=1)

            # Re-estimate
            self.pi = self._safe_row_norm(pi_acc)
            self.A  = A_num / (A_den[:, None] + 1e-12)
            self.B  = B_num / (B_den[:, None] + 1e-12)

            # Row-normalize just in case of numerical drift
            self.A  = self._safe_row_norm(self.A)
            self.B  = self._safe_row_norm(self.B)

    # --------- Utility ----------
    def predict(self, O: np.ndarray) -> Tuple[np.ndarray, float]:
        return self.viterbi(O)

    @staticmethod
    def _safe_row_norm(M: np.ndarray) -> np.ndarray:
        M = M.astype(np.float32, copy=False)
        if M.ndim == 1:
            s = float(np.sum(M))
            if s == 0:
                return np.ones_like(M, dtype=np.float32) / len(M)
            return (M / s).astype(np.float32)
        s = np.sum(M, axis=1, keepdims=True)
        M = M / (s + 1e-12)
        # replace NaNs if any row was all zeros
        nan_rows = ~np.isfinite(M).all(axis=1)
        M[nan_rows] = 1.0 / M.shape[1]
        return M.astype(np.float32)