#!/usr/bin/env python3
import numpy as np
import warnings
warnings.filterwarnings('ignore')

PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI

class PisotSubstitution:
    def __init__(self):
        self.rules = {'1': '12', '2': '13', '3': '1'}
        
    def generate_sequence(self, n_steps: int, start: str = '1') -> str:
        seq = start
        while len(seq) < n_steps:
            seq = ''.join([self.rules.get(ch, ch) for ch in seq])
        return seq[:n_steps]
    
    def sequence_to_gamma(self, seq: str, base_gamma: float = 0.1) -> np.ndarray:
        gamma_map = {'1': base_gamma, '2': base_gamma * PHI_INV, '3': base_gamma * PHI_INV**2}
        return np.array([gamma_map.get(ch, base_gamma) for ch in seq])

class RiemannianHypersphereOptimizer:
    def __init__(self, N: int = 60, alpha: float = 0.005, gamma: float = 0.1):
        self.N = N
        self.alpha = alpha
        self.gamma = gamma
        self.reset()
        
    def reset(self):
        np.random.seed(42)
        self.x = np.random.randn(self.N)
        self.x /= np.linalg.norm(self.x)
        self.history = {'x': []}
        
    def _circulant(self, n):
        c = np.zeros(n)
        c[1] = 0.5
        c[-1] = 0.5
        return np.fromiter((c[(i - j) % n] for i in range(n) for j in range(n)), dtype=float).reshape(n, n)
    
    def gradient(self, x):
        A = np.eye(self.N) + self.gamma * self._circulant(self.N)
        return -A @ x
        
    def step(self):
        grad = self.gradient(self.x)
        grad_R = grad - np.dot(self.x, grad) * self.x
        self.x = self.x - self.alpha * grad_R
        self.x /= np.linalg.norm(self.x)
        self.history['x'].append(self.x.copy())
    
    def run(self, n_steps: int = 4000):
        for _ in range(n_steps):
            self.step()
        return self.x
    
    def compute_ratio(self):
        if len(self.history['x']) < 2:
            return 1.0
        final_state = self.history['x'][-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.abs(final_state[1:] / final_state[:-1])
            ratios = ratios[np.isfinite(ratios)]
        return np.nanmean(ratios) if len(ratios) > 0 else 1.0

class PisotRiemannianOptimizer(RiemannianHypersphereOptimizer):
    def __init__(self, N: int = 60, alpha: float = 0.005, base_gamma: float = 0.1):
        super().__init__(N, alpha, base_gamma)
        self.pisot = PisotSubstitution()
        self.base_gamma = base_gamma
        self.gamma_seq = self.pisot.sequence_to_gamma(self.pisot.generate_sequence(5000), self.base_gamma)
        
    def gradient(self, x, step):
        gamma_t = self.gamma_seq[step % len(self.gamma_seq)]
        A = np.eye(self.N) + gamma_t * self._circulant(self.N)
        return -A @ x
        
    def step_modulated(self, step):
        grad = self.gradient(self.x, step)
        grad_R = grad - np.dot(self.x, grad) * self.x
        self.x = self.x - self.alpha * grad_R
        self.x /= np.linalg.norm(self.x)
        self.history['x'].append(self.x.copy())
        
    def run(self, n_steps: int = 4000):
        for step in range(n_steps):
            self.step_modulated(step)
        return self.x

def run_comparison():
    print("\n====================================================")
    print("🔱 ESQET PISOT-RIEMANNIAN MODULATION ANALYSIS")
    print("====================================================")
    
    gammas = [0.05, 0.10, 0.15, 0.20]
    for g in gammas:
        c_opt = RiemannianHypersphereOptimizer(gamma=g)
        c_opt.run()
        r_const = c_opt.compute_ratio()
        
        p_opt = PisotRiemannianOptimizer(base_gamma=g)
        p_opt.run()
        r_pisot = p_opt.compute_ratio()
        
        print(f"γ Base: {g:.2f} | Constant Ratio: {r_const:.4f} | Pisot-Modulated Ratio: {r_pisot:.4f}")
    print("====================================================")

if __name__ == "__main__":
    run_comparison()
