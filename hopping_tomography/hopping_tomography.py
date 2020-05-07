import numpy as np
import cvxpy as cvx

class hopping_tomography:

    def __init__( self, input_data, input_times, PNP_H = None, load = None, save = None ):
        """
        data_input: list of diagonal input correlations
        input_times: times at which input correlations are taken
        """
        self.input_data = input_data
        self.input_times = input_times
        self.nmb_times = len( input_times )
        if PNP_H is None:
            self.LL = input_data.shape(1)
            self.H = nearest_neighbor_hopping( self.L )
        else:
            self.H = PNP_H
        self.LL = self.H.L
    

    def recover_SDP(self, SDP_constraints = True, verbose = True, eps = 1e-5, max_iters = 2500 ):

        A = []
        G_t = []
        b_t = []
        for t_ind in range( self.nmb_times ):
            G_i = self.H.G( self.input_times[ t_ind ] )
            G_t.append(G_i)
            b_t.append(self.input_data[ t_ind ])

        Gamma = cvx.Variable( (self.LL, self.LL), complex = True)
        
        objective = cvx.Minimize( cvx.sum([ 
                    cvx.norm( (G_t[t] @ Gamma @ G_t[t].conj().T)[range(self.LL),range(self.LL)]
                                             - b_t[t] , 2 )  for t in range(self.nmb_times) ] ))

        Id = np.eye( self.LL ) 
        if SDP_constraints == True:
            constraints = [ Gamma >> 0, Id-Gamma >>0 ] 
            #constraints = constraints + [ Gamma[ self.ini_matrix_select_ind( self.LL ) ] == 0 ] 
        else:
            constraints = []

        self.prob = cvx.Problem(objective, constraints)

        result = self.prob.solve(solver=cvx.SCS, eps=eps, verbose=verbose, max_iters = max_iters)

        self.ReGamma = Gamma.value.real
        self.ImGamma = Gamma.value.imag
        self.Gamma = Gamma.value 
        self.chi2 = self.prob.value
        
        if verbose == True:
            print("Final residue:", self.prob.value)
            print("Normalized residue /size/times ", self.prob.value/(self.nmb_times) )
            print("Reconstructed matrix\n", self.Gamma )
        
    def ini_matrix_select_ind( self, L ):
        x_ind = []
        y_ind = []
        for x in range( L ):
            for y in range( L ):
                if x % 2 == 0 and y % 2 == 0:
                    x_ind.append( x )
                    y_ind.append( y )
        return ( x_ind, y_ind )
        
