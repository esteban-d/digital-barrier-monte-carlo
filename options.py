import numpy as np

class CashOrNothing_DownAndOut_Put:
    def __init__(self, B, K, x, T):
        self.B = B
        self.K = K
        self.x = x
        self.T = T
        self.crossed_barrier = False

    def payoff(self, S):
        if not self.crossed_barrier and S < self.K:
            return self.x
        else:
            return 0

    def check_barrier(self, S):
        if S <= self.B:
            self.crossed_barrier = True

        return self.crossed_barrier

    def check_bridge(self, S_prev, S_next, delta_t, sigma):

        if self.check_barrier(S_next): 
            return True 

        p_n = np.exp(-2 * (S_prev - self.B) * (S_next - self.B) / (sigma**2 * S_prev**2 * delta_t))


        if np.random.uniform() < p_n:
            self.crossed_barrier = True

        return self.crossed_barrier
        
        

    def reset(self):
        self.crossed_barrier = False