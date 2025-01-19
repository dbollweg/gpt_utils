import gpt as g 
import os
import sys
import numpy as np
import math
from qTMD.proton_qTMD_draft import proton_TMD


from tools import *
from io_corr import *

root_output = "."
src_shift = np.array([0,0,0,0]) + np.array([1,3,5,7])
data_dir = "/lustre/orion/nph159/proj-shared/xgao/prod/64I/qTMD_proton/p0"

#smearing
#smear_list = [['flow', '05eps01', 5], ['flow', '10eps01', 10], ['flow', '20eps01', 20], ['flow', '40eps01', 40]]
smear_list = [['flow', '10eps01', 10], ['flow', '20eps01', 20]]

# tags
sm_tag = "GSRC_W80_k0"
lat_tag = "64I"

#(8*8+12*12+16*16+20*20) * 4 * 4
#b_z < 1 fm
#b_T < 1 fm
# CG: 20*20 * 4 = 1600
# (15s - 2s) / 5
parameters = {
    
    # NOTE: eta > 12 will only run bz=0: check qTMD.proton_qTMD_draft.create_TMD_Wilsonline_index_list
    "eta": [12,13,14,15,16],
    "b_z": 10,
    "b_T": 10,

    #"qext": [-2, -1, 0, 1, 2],
    "qext": [0], # momentum transfer for TMD
    "qext_PDF": [-2, -1, 0], # momentum transfer for PDF
    #"pf": [0,0,6,0],
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [-2, -1, 0] for y in [-2, -1, 0] for z in [-2, -1, 0]], # 2pt momentum

    #"boost_in": [0,0,3],
    #"boost_out": [0,0,3],
    #"width" : 8.0,
    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "width" : 8.0,

    "pol": ["PpSzp", "PpSzm", "PpSxp"],
    "t_insert": 6, #starting with 10 source sink separation

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2]) + "dt" + str(parameters["t_insert"])

#ADD: Add coulomb gauge version: set longitudinal links to one
config_number = g.default.get_int("--config_num", 0)

groups = {
    "Frontier_batch_0": {
        "confs": [
            str(config_number),
        ],
        "evec_fmt": "/lustre/orion/proj-shared/nph159/xgao/DWF/64I/%s.evecs/lanczos.output",
        "conf_fmt": "/lustre/orion/proj-shared/nph159/xgao/DWF/64I/Coulomb/ckpoint_lat.Coulomb.%s",
        #"evec_fmt": "/lustre/orion/proj-shared/nph159/data/64I/job-0%s/lanczos.output",
        #"conf_fmt": "/lustre/orion/proj-shared/nph159/data/64I/ckpoint_lat.Coulomb.%s",
    },
}

# NOTE, now it is the location instead of number of srcs
ex_src = g.default.get_int("--ex_src", -1)
sl_src = g.default.get_int("--sl_src", -1)
jobs = {
    "proton_TMD_AMA": {
        "exact": int(ex_src),
        "sloppy": int(sl_src),
        "low": 0,
    },
}


jobs_per_run = g.default.get_int("--gpt_jobs", 1)

# find jobs for this run
def get_job(only_on_conf=None):
    # statistics
    n = 0
    for group in groups:
        for job in jobs:
            for conf in groups[group]["confs"]:
                n += 1

    jid = -1
    for group in groups:
        for conf in groups[group]["confs"]:
            for job in jobs:
                jid += 1
                if only_on_conf is not None and only_on_conf != conf:
                    continue
                return group, job, conf, jid, n

    return None

if g.rank() == 0:
    first_job = get_job()
    run_jobs = str(
        list(
            filter(
                lambda x: x is not None,
                [first_job] + [get_job(first_job[2]) for i in range(1, jobs_per_run)],
            )
        )
    ).encode("utf-8")
else:
    run_jobs = bytes()
run_jobs = eval(g.broadcast(0, run_jobs).decode("utf-8"))

"""
================================================================================
            Every node now knows what to do -> Now initialization
================================================================================
"""

# configuration needs to be the same for all jobs, so load eigenvectors and configuration
conf = run_jobs[0][2]
group = run_jobs[0][0]


# loading gauge configuration
g.message("Loading ")
print(groups[group]["conf_fmt"] % conf)
U = g.load(groups[group]["conf_fmt"] % conf)
U_smear = g.copy(U)
g.message("finished loading gauge config")

# do gauge fixing
U_prime, trafo = g.gauge_fix(U, maxiter=5000)
del U_prime
L = U[0].grid.fdimensions


Measurement = proton_TMD(parameters)
prop_exact, prop_sloppy, pin = Measurement.make_64I_inverter(U, groups[group]["evec_fmt"] % conf)

W_index_list = Measurement.create_TMD_Wilsonline_index_list()
W_index_list_CG = Measurement.create_TMD_Wilsonline_index_list_CG(U[0].grid)
W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list(U[0].grid)
g.message("W_index_list:", np.shape(W_index_list))
g.message(W_index_list)
g.message("W_index_list_CG:", np.shape(W_index_list_CG))
g.message(W_index_list_CG)
g.message("W_index_list_PDF:", np.shape(W_index_list_PDF))
g.message(W_index_list_PDF)

# show available memory
g.mem_report(details=False)
g.message(
"""
================================================================================
       Propagator generation run:
================================================================================
"""
)

for group, job, conf, jid, n in run_jobs:
    
    g.message(f"""Job {jid} / {n} :  configuration {conf}, job tag {job}""")
    
    src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
    source_positions = srcLoc_distri_eq(L, src_origin)

    source_positions_sloppy, source_positions_exact = [], []
    if jobs[job]["sloppy"] >= 0:
        source_positions_sloppy += [source_positions[jobs[job]["sloppy"]]]
    if jobs[job]["exact"] >= 0:
        source_positions_exact += [source_positions[jobs[job]["exact"]]]
    #source_positions_sloppy = source_positions[:jobs[job]["sloppy"]]
    #source_positions_exact = source_positions[:jobs[job]["exact"]]

    g.message(jobs[job]["sloppy"], f" positions_sloppy = {source_positions_sloppy}")
    g.message(jobs[job]["exact"], f" positions_exact = {source_positions_exact}")
    
    sample_log_file = data_dir + "/sample_log_qtmd/" + conf + '_' + sm_tag
    if g.rank() == 0:
        f = open(sample_log_file, "a+")
        f.close()

    # exact positions
    corr_dir = data_dir + "corr_ex/" + conf
    for pos in source_positions_exact:
        
        sample_log_tag = get_sample_log_tag("ex"+str(jobs[job]["exact"]), pos, sm_tag + "_" + pf_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file, "a+") as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop exact")
        prop_exact_f = g.eval(prop_exact * srcDp)
        g.message("forward prop done")

        g.message("Contraction: Starting 2pt (includes sink smearing)")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
        phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
        Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag)
        g.message("Contraction: Done 2pt (includes sink smearing)")

        sequential_bw_prop_down = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 2, pos)
        sequential_bw_prop_up = Measurement.create_bw_seq(prop_exact, prop_exact_f, trafo, 1, pos)
        g.message("backward prop done")

        phases_3pt = Measurement.make_mom_phases_3pt(U[0].grid, pos)
        phases_PDF = Measurement.make_mom_phases_PDF(U[0].grid, pos)
        for ism, smear in enumerate(smear_list):
           
            U_smear = g.copy(U)
            contract_tag, n_sm = smear[0]+smear[1], smear[2]
            if smear[0] == 'hyp':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.hyp(U_smear, alpha = np.array([0.75, 0.6, 0.3]))
            if smear[0] == 'flow':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.wilson_flow(U_smear, epsilon=0.1)
            g.message("Gauge: Smearing/Flow finished")

            g.message("\ncontract_PDF: GI")
            # gauge invariant/traditional straight gauge link
            for iW, WL_indices in enumerate(W_index_list_PDF):
                W = Measurement.create_PDF_Wilsonline(U_smear, WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_exact_f, [W], [WL_indices])
                g.message("TMD forward prop done")

                qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI_PDF.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI_PDF.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_down,phases_PDF, WL_indices, qtmd_tag_exact_D, iW)
                proton_TMDs_up = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_up,phases_PDF, WL_indices, qtmd_tag_exact_U, iW)

                del tmd_forward_prop

            g.message("\ncontract_TMD: GI")
            # gauge invariant/traditional stape gauge link
            for iW, WL_indices in enumerate(W_index_list):
                W = Measurement.create_TMD_Wilsonline(U_smear, WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, [W], [WL_indices])
                g.message("TMD forward prop done")

                qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)

                del tmd_forward_prop

            # Only do once for CG
            if ism == 0:

                g.message("\ncontract_PDF: CG")
                # gauge invariant/traditional straight gauge link
                for iW, WL_indices in enumerate(W_index_list_PDF):
                    W = Measurement.create_TMD_Wilsonline_CG(U_smear, WL_indices)

                    tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_exact_f, [W], [WL_indices])
                    g.message("TMD forward prop done")

                    qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_PDF.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_PDF.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    g.message("Starting TMD contractions")
                    proton_TMDs_down = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_down,phases_PDF, WL_indices, qtmd_tag_exact_D, iW)
                    proton_TMDs_up = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_up,phases_PDF, WL_indices, qtmd_tag_exact_U, iW)

                    del tmd_forward_prop

                g.message("\ncontract_TMD: CG Tlink")
                # CG gauge links with Transverse link only
                for iW, WL_indices in enumerate(W_index_list):
                    W = Measurement.create_TMD_Wilsonline_CG_Tlink(U_smear,WL_indices)

                    tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, [W], [WL_indices])
                    g.message("TMD forward prop done")

                    qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                    g.message("Starting TMD contractions")
                    proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                    proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)

                    del tmd_forward_prop

                g.message("\ncontract_TMD: CG no links")
                # CG without links
                for iW, WL_indices in enumerate(W_index_list_CG):
                    W = Measurement.create_TMD_Wilsonline_CG(U_smear,WL_indices)

                    tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_exact_f, [W], [WL_indices])
                    g.message("TMD forward prop done")

                    qtmd_tag_exact_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.D.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    qtmd_tag_exact_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.U.ex", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                    g.message("Starting TMD contractions")
                    proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_exact_D, iW)
                    proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_exact_U, iW)

                    del tmd_forward_prop

        with open(sample_log_file, "a+") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_exact_f

    del prop_exact


    # sloppy positions
    corr_dir = data_dir + "corr_sl/" + conf
    for pos in source_positions_sloppy:
        
        sample_log_tag = get_sample_log_tag("sl"+str(jobs[job]["sloppy"]), pos, sm_tag + "_" + pf_tag)
        g.message(f"START: {sample_log_tag}")
        with open(sample_log_file, "a+") as f:
            if sample_log_tag in f.read():
                g.message("SKIP: " + sample_log_tag)
                continue

        g.message("Generatring boosted src's")
        srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)

        g.message("Starting prop sloppy")
        prop_sloppy_f = g.eval(prop_sloppy * srcDp)
        g.message("forward prop done")

        g.message("Contraction: Starting 2pt (includes sink smearing)")
        tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "sl", pos, sm_tag)
        phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
        Measurement.contract_2pt_TMD(prop_sloppy_f, phases_2pt, trafo, tag)
        g.message("Contraction: Done 2pt (includes sink smearing)")

        sequential_bw_prop_down = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_f, trafo, 2, pos)
        sequential_bw_prop_up = Measurement.create_bw_seq(prop_sloppy, prop_sloppy_f, trafo, 1, pos)
        g.message("backward prop done")

        phases_3pt = Measurement.make_mom_phases_3pt(U[0].grid, pos)
        phases_PDF = Measurement.make_mom_phases_PDF(U[0].grid, pos)
        for ism, smear in enumerate(smear_list):
            
            U_smear = g.copy(U)
            contract_tag, n_sm = smear[0]+smear[1], smear[2]
            if smear[0] == 'hyp':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.hyp(U_smear, alpha = np.array([0.75, 0.6, 0.3]))
            if smear[0] == 'flow':
                for i in range(n_sm):
                    U_smear = g.qcd.gauge.smear.wilson_flow(U_smear, epsilon=0.1)
            g.message("Gauge: Smearing/Flow finished")

            g.message("\ncontract_PDF: GI")
            # gauge invariant/traditional straight gauge link
            for iW, WL_indices in enumerate(W_index_list_PDF):
                W = Measurement.create_PDF_Wilsonline(U_smear, WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_sloppy_f, [W], [WL_indices])
                g.message("TMD forward prop done")

                qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI_PDF.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI_PDF.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_down,phases_PDF, WL_indices, qtmd_tag_sloppy_D, iW)
                proton_TMDs_up = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_up,phases_PDF, WL_indices, qtmd_tag_sloppy_U, iW)

                del tmd_forward_prop

            g.message("\ncontract_TMD: GI")
            # gauge invariant/traditional stape gauge link
            for iW, WL_indices in enumerate(W_index_list):
                W = Measurement.create_TMD_Wilsonline(U_smear,WL_indices)

                tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_sloppy_f, [W], [WL_indices])
                g.message("TMD forward prop done")

                qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"GI.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                g.message("Starting TMD contractions")
                proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_sloppy_D, iW)
                proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_sloppy_U, iW)

                del tmd_forward_prop
            
            g.message("\ncontract_TMD: CG Tlink")
            # Only do once for CG
            if ism == 0:

                g.message("\ncontract_PDF: CG")
                # gauge invariant/traditional straight gauge link
                for iW, WL_indices in enumerate(W_index_list_PDF):
                    W = Measurement.create_TMD_Wilsonline_CG(U_smear, WL_indices)

                    tmd_forward_prop = Measurement.create_fw_prop_PDF(prop_sloppy_f, [W], [WL_indices])
                    g.message("TMD forward prop done")

                    qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_PDF.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_PDF.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    g.message("Starting TMD contractions")
                    proton_TMDs_down = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_down,phases_PDF, WL_indices, qtmd_tag_sloppy_D, iW)
                    proton_TMDs_up = Measurement.contract_PDF(tmd_forward_prop, sequential_bw_prop_up,phases_PDF, WL_indices, qtmd_tag_sloppy_U, iW)

                    del tmd_forward_prop

                # CG gauge links with Transverse link only
                for iW, WL_indices in enumerate(W_index_list):
                    W = Measurement.create_TMD_Wilsonline_CG_Tlink(U_smear,WL_indices)

                    tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_sloppy_f, [W], [WL_indices])
                    g.message("TMD forward prop done")

                    qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG_T.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                    g.message("Starting TMD contractions")
                    proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_sloppy_D, iW)
                    proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_sloppy_U, iW)

                    del tmd_forward_prop

                g.message("\ncontract_TMD: CG no links")
                # CG without links; Only do once
                for iW, WL_indices in enumerate(W_index_list_CG):
                    W = Measurement.create_TMD_Wilsonline_CG(U_smear,WL_indices)

                    tmd_forward_prop = Measurement.create_fw_prop_TMD(prop_sloppy_f, [W], [WL_indices])
                    g.message("TMD forward prop done")

                    qtmd_tag_sloppy_D = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.D.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)
                    qtmd_tag_sloppy_U = get_qTMD_file_tag(data_dir,lat_tag,conf,"CG.U.sl", pos, sm_tag+'.'+pf_tag+"."+contract_tag)

                    g.message("Starting TMD contractions")
                    proton_TMDs_down = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_down,phases_3pt, WL_indices, qtmd_tag_sloppy_D, iW)
                    proton_TMDs_up = Measurement.contract_TMD(tmd_forward_prop, sequential_bw_prop_up,phases_3pt, WL_indices, qtmd_tag_sloppy_U, iW)

                    del tmd_forward_prop
                
        with open(sample_log_file, "a+") as f:
            if g.rank() == 0:
                f.write(sample_log_tag+"\n")
        g.message("DONE: " + sample_log_tag)

        del prop_sloppy_f
        
    del prop_sloppy

del pin
