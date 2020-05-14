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

    def show_results( self, true_cov, save_path = None, fontsize = 40 ):
       
        LL = self.LL
        N = len(self.input_data)
        T = self.input_times[-1]
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['axes.titlesize'] = fontsize
        plt.rcParams['font.size'] = fontsize
        #set_matplotlib_formats('pdf', 'png')
        plt.rcParams['savefig.dpi'] = 75
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['legend.labelspacing'] = .3
        plt.rcParams['legend.columnspacing']= .3
        plt.rcParams['legend.handletextpad']= .1
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = "serif"
        plt.rcParams['font.serif'] = "cm"
        #Main plot
        fig = plt.figure( figsize = ( 20,20 ) )


        label_a_x = -2.7
        label_a_y = 1.05*N
        label_b_x = -5
        label_b_y = LL+1   
        label_c_x = -1.5
        label_c_y = label_b_y

        ## a)
        #Input data
        ax_input_data = plt.subplot(1,3,1)#add_axes( [ data_pos_x, data_pos_y, data_width, data_height ] )   

        im_input_data = ax_input_data.imshow( self.input_data, cmap='Blues', interpolation = None, 
                                             extent= [0, LL, N, 0])
        #plt.text( label_a_x,label_a_y, 'a)')

        title = r'Input data'
        plt.title( title )
        plt.xlabel( 'Lattice site $x$' )
        plt.ylabel( 'Time $t_i$' )
        plt.clabel = r'$N_x(t)$'

        plt.grid() 

        plt.xlim((0,LL))
        plt.ylim((0,N))

        xlabels = ['']*(LL+1)
        xlabels[0]='0'
        xlabels[-1]=(LL)
        xlabels[int(LL/2)] =(LL/2)
        ylabels = ['']*(N+1)
        ylabels[1]=T/N
        ylabels[-1]=T
        ylabels[int(N/2)]= ( (T/2))
        #ylabels[5]= ( (T/2))

        plt.xticks( range(0,LL+1),xlabels )
        plt.yticks( range(0,N+1), ylabels)

        ax_input_data.tick_params(direction='in', length=2, width=2, colors='k')#,    grid_color='k', grid_alpha=0.5)

        divider = make_axes_locatable(ax_input_data)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)        
        cbar = plt.colorbar(im_input_data, cax=cax)
        cbar.set_clim( 0, 1 )
        ax_input_data.set_aspect(1)

        ## b)
        #Plot cov_ini
        #fig = plt.figure( figsize = ( 20,20 ) )

        ax_cov_ini = plt.subplot(1,3,2) #add_axes( [ ini_pos_x, ini_pos_y, ini_width, ini_height ] )

        im_cov_ini = ax_cov_ini.imshow( abs(true_cov), cmap='RdBu', aspect='equal', interpolation = None, extent = [ 1, LL, LL, 1])
        #plt.text( label_b_x,label_b_y, 'b)')

        title = r'True $\Gamma$'
        plt.title( title )
        plt.xlim((1,LL))
        plt.ylim((1,LL))
        plt.xlabel( 'Lattice site $x$' )
        plt.ylabel( 'Lattice site $y$' )
        ax_cov_ini.tick_params(direction='in', length=2, width=2, colors='k')
        divider = make_axes_locatable(ax_cov_ini)
        cax = divider.append_axes("right", size="2.5%", pad=0.05)

        cbar = plt.colorbar(im_cov_ini, cax=cax)
        range_plot = np.max( np.abs( true_cov ) )
        cbar.set_clim( -range_plot, range_plot )

        ## c)
        #Plot cov reconstructed + inset deviation

        ax_cov_rec = plt.subplot(1,3,3)#add_axes( [ rec_pos_x, rec_pos_y, rec_width, rec_height ] )
        cov_rec = abs(self.Gamma)
        inset_cov = abs( true_cov - self.Gamma )
        im_cov_rec = ax_cov_rec.imshow( cov_rec, cmap='RdBu', aspect='equal', interpolation = None, extent = [ 1, LL, LL, 1])
        #plt.text( label_c_x,label_c_y, 'c)')

        title = r'Reconstruction $\Gamma^{\rm(Rec)}$'
        plt.title( title )
        plt.xlim((1,LL))
        plt.ylim((1,LL))
        plt.xlabel( 'Lattice site $x$' )
        ylabels = ['']*5
        plt.yticks(range(10,LL,10), ylabels)
        ax_cov_rec.tick_params(direction='in', length=2, width=2, colors='k')

        divider = make_axes_locatable( ax_cov_rec )
        cax_cov_rec = divider.append_axes("right", size="2.5%", pad=0.05)

        cbar = plt.colorbar( im_cov_rec, cax = cax_cov_rec )
        range_plot = np.max( np.abs( cov_rec ) )
        cbar.set_clim( -range_plot, range_plot )
        fig.tight_layout()

        if save_path is None:
            save_path = "tomography_run"
        plt.savefig( save_path, format='pdf')
        plt.show()
                
