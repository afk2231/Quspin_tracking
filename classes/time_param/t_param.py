import numpy as np

class time_evolution_params:
    def __init__(self, perimeter_params, cycles, nsteps, start=0, plotting=0):
        self.cycles = cycles  # time in cycles of field frequency
        self.n_steps = nsteps
        self.start = start
        if plotting:
            self.stop = self.cycles
        else:
            self.stop = self.cycles / perimeter_params.freq
        self.times, self.delta = np.linspace(self.start, self.stop, num=self.n_steps, endpoint=True, retstep=True)