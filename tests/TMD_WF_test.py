import gpt as g
from gpt_qTMD_utils import TMD_WF_measurement

# momenta setup
parameters = {
    "eta"   : [4],
    "pzmin" : 1,
    "pzmax" : 2,
    "b_z"   : 1,
    "b_T"   : 1,
    "zmax"  : 0,
    "width" : 2.0,
    "pos_boost" : [0,0,2],
    "neg_boost" : [0,0,-2],
    "save_propagators" : False
}

jobs = {
    "test_exact_0": {
        "exact": 1,
        "sloppy": 1,
        "low": 0,
    },  
}


##### small dummy used for testing
grid = g.grid([8,8,8,8], g.double)
rng = g.random("seed text")
U = g.qcd.gauge.random(grid, rng)

g.message("finished creating gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=500)
del U_prime
L = U[0].grid.fdimensions

Measurement = TMD_WF_measurement(parameters)
prop_exact, prop_sloppy = Measurement.make_debugging_inverter(U)


g.mem_report(details=False)
g.message(
"""
================================================================================
       2pt test run:
================================================================================
"""
)

src_origin = [0,0,0,0] 

source_positions_exact = [src_origin]
source_positions_sloppy = [
    [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
    for j in range(jobs["test_exact_0"]["sloppy"])
]

Measurement.set_output_facilities("./TMD_corr_exact","./propagators")


#testing correctness of staple shaped wilson lines
staples = Measurement.create_TMD_WL(U)

# # W = []


# prv_link = g.qcd.gauge.unit(U[2].grid)[0]
# current_link = prv_link
# for dz in range(0,2):
#     g.message(f"stepping {dz} from origin")
#     current_link=g.eval(prv_link * g.cshift(U[2],2,dz))
#     prv_link = current_link

# test_link = current_link
# prv_link = test_link
# for dz in reversed(range(1,2)):
#     test_link=g.eval(prv_link * g.adj(g.cshift(U[2],2,dz)))
#     prv_link = test_link

# g.message(g.eval((test_link-g.cshift(U[2],2,0))))

td_offset = parameters["b_T"]*parameters["b_z"]*len(parameters["eta"])
eta_offset = parameters["b_T"]*parameters["b_z"]
bz_offset = parameters["b_T"]

for transverse_direction in [0,1]:
    for eta_idx,current_eta in enumerate(parameters["eta"]):
        for current_bz in range(0, parameters["b_z"]):
            for current_b_T in range (0, parameters["b_T"]):

                WL_index = current_b_T + bz_offset*current_bz + eta_offset*eta_idx + td_offset*transverse_direction

                #unwind wilson loop except last step:

                #start from end
                prv_link = staples[WL_index]
                
                current_link = prv_link

                #reverse order and take adjoint
                for dz in range(0, current_eta-current_bz):
                    current_link=g.eval(prv_link * g.cshift(g.cshift(g.cshift(U[2],2,current_eta+current_bz-1),transverse_direction, current_b_T-1),2,dz))
                    prv_link=current_link

                            
                for dx in reversed(range(0, current_b_T)):
                    current_link=g.eval(prv_link * g.adj(g.cshift(g.cshift(U[transverse_direction],2,current_eta+current_bz-1),transverse_direction, dx)))
                    prv_link=current_link


                for dz in reversed(range(1, current_eta+current_bz)):
                    current_link=g.eval(prv_link * g.adj(g.cshift(U[2],2, dz)))
                    prv_link=current_link                        

                
                
                g.message("Testing: Current link - first link = ")
                g.message(g.eval(g.sum(current_link - g.cshift(U[2],2,0))))



