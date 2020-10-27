
class unscaledparam:
    def __init__(self, L, t0, U, pbc, field, F0, a, a_scale=1, J_scale=1, tracking=0):
        """Hubbard model Parameters"""
        self.L = L  # system size
        self.N_up = self.L // 2 + self.L % 2  # number of fermions with spin up
        self.N_down = self.L // 2  # number of fermions with spin down
        self.N = self.N_up + self.N_down  # number of particles
        self.t0 = t0  # hopping strength
        # U = 0*t0  # interaction strength
        self.U = U * t0  # interaction strength
        self.pbc = pbc

        """Laser pulse parameters"""
        self.field = field  # field angular frequency THz
        self.F0 = F0  # Field amplitude MV/cm
        self.a = a  # Lattice constant Angstroms
        if tracking:
            """ Tracking Hubbard model Parameters"""
            self.a_scale = a_scale  # scaling lattic parameter for tracking
            self.J_scale = J_scale  # scaling J for tracking.
