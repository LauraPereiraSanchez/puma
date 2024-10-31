# Script for vertex fits in Paper
from ftag.hdf5 import H5Reader # I use ftag tools to read the file
#Plotting
from ftag import Cuts, Flavour, Flavours
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot

import numpy as np
from scipy.stats import iqr
import sys
import os

# starting parameters

inclusive_vertex = False
plot_Lxy = False     # You need to install JAX to be able to run these.                                                                                                                                    
plot_mass = True

if (plot_Lxy == False) and (plot_mass == False):
    print("ERROR: Set plot_Lxy or plot_Mass to TRUE")
    sys.exit()
    
n_jets = 150000
n_tracks = 40

sample = "ttbar"
mc = "MC23a"
directory = "/fs/ddn/sdf/group/atlas/d/lapereir/GN2/OpenDataset_final_v2/"
path=directory+"/"+mc+"_"+"new-"+sample+".h5"

output = "GN2_PaperPlots/"
os.makedirs(output, exist_ok=True)

if sample == "ttbar":
    sample_str = "$t\\overline{t}$"
    cuts = [
        ("pt", ">=", 20000),
        ("pt", "<=", 250000),
        ("eta", ">", -2.5),
        ("eta", "<", 2.5),
        ("HadronConeExclTruthLabelID", ">=", 4),
        ("HadronConeExclTruthLabelID", "<=", 5),
        ("n_truth_promptLepton", "==", 0)
    ]
    cut_str = "20 < $p_T$ < 250 GeV, $|\\eta| < 2.5$"

elif sample == "zprime":
    sample_str = "Z'"
    cuts = [
        ("pt", ">=", 250000),
        ("pt", "<=", 6000000),
        ("eta", ">", -2.5),
        ("eta", "<", 2.5),
        ("HadronConeExclTruthLabelID", ">=", 4),
        ("HadronConeExclTruthLabelID", "<=", 5),
        ("n_truth_promptLepton", "==", 0)
    ]
    cut_str = "250 < $p_T$ < 6000 GeV, $|\\eta| < 2.5$"


######## Code starts here #########

basic_extra_string = "_InclusiveVertex"
extra = " Inclusive Vertex"
max_sv = 1
if inclusive_vertex == False:
    basic_extra_string = "_NonInclusiveVertex"
    extra = " Exclusive Vertex"
    max_sv = 3

    
com = "13" if "MC20" in mc else "13.6"



def jet_flavour(jet, f=""):
    if f == "b": return jet["HadronConeExclTruthLabelID"] == 5
    elif f == "c": return jet["HadronConeExclTruthLabelID"] == 4
    elif f == "light": return jet["HadronConeExclTruthLabelID"] == 0
    else:  return jet ["HadronConeExclTruthLabelID"] >= 0

def LoadDataset(file_path,kinematic_cuts,  n_jets=-1, n_tracks=40):

    track_var = ["dphi", "d0Uncertainty", "z0SinThetaUncertainty", "phiUncertainty", "thetaUncertainty", "qOverPUncertainty", "qOverP", "deta", "theta", "dphi"] # for vertex fit                                                                                                                                                              
    track_var += ["d0RelativeToBeamspot", "d0RelativeToBeamspotUncertainty","z0RelativeToBeamspot", "z0RelativeToBeamspotUncertainty",  "ftagTruthOriginLabel",  "GN2v01_aux_TrackOrigin", "GN2v01_aux_VertexIndex",  "ftagTruthVertexIndex", "ftagTruthParentBarcode"]
    track_var += ["JFVertexIndex", "pt", "SV1VertexIndex", "valid"]

    jet_var = ["eventNumber","GN2v01_pb", "GN2v01_pc", "GN2v01_pu", "phi", "eta", "HadronConeExclTruthLabelID", "HadronConeExclExtendedTruthLabelID"] # \phi is needed for vertex fit if track phi is not available # v00 instead of v01                                                                                                   

    truth_hadrons = ['pt', 'mass', 'eta', 'phi', 'displacementZ','Lxy', 'pdgId', 'barcode', 'ftagTruthParentBarcode', 'valid']

    event_var = ["primaryVertexToBeamDisplacementX", "primaryVertexToBeamDisplacementY", "primaryVertexToBeamDisplacementZ", "primaryVertexToTruthVertexDisplacementX", "primaryVertexToTruthVertexDisplacementY", "primaryVertexToTruthVertexDisplacementZ"]

    ## read it!                                                                                                                                                                   
    my_reader = H5Reader(file_path, precision="full", shuffle=False, batch_size=100)

    if n_jets == -1:
        my_data = my_reader.load({"jets": jet_var, "tracks" : track_var, "truth_hadrons" : truth_hadrons, "eventwise": event_var},  cuts=kinematic_cuts)
    else:
        my_data = my_reader.load({"jets": jet_var, "tracks" : track_var, "truth_hadrons" : truth_hadrons, "eventwise": event_var}, num_jets=n_jets, cuts=kinematic_cuts)

    return my_data



def SV_Finding(vertex_index, track_origin, inclusive=False, remove_non_HF_tracks=True, min2tracks=True):    
    
    if remove_non_HF_tracks:
        raw_vertex_index = vertex_index.copy()
        # remove tracks not from HF 
        #mask = np.where(track_origin > 2  , 1, 0)
        mask = np.where((track_origin > 2) & (track_origin < 6), 1, 0)
        vertex_index = np.where(mask, raw_vertex_index, -1)
    
    track_weights = np.array([[[1 if r == i else 0 for r in row ] for i in set(row[row >=0])] for row in vertex_index], dtype="object")
    
    max_jets = track_weights.shape[0]

    max_n_hadrons = max(len(inner_list) for inner_list in track_weights)     
    padded_track_weights = np.zeros((max_jets, max_n_hadrons, n_tracks))
    
    for i, outer in enumerate(track_weights):
        for j, inner in enumerate(outer):
            padded_track_weights[i, j, :len(inner)] = inner  # Only copy the actual values, pad with zeros

    if inclusive:
        padded_track_weights = np.sum(padded_track_weights, axis=1)

        if min2tracks:
            # if only 1 track in vertex, set it off
            mask = (np.sum(padded_track_weights, axis=1) == 1)
            padded_track_weights[mask] = np.zeros(padded_track_weights.shape[1])
    
        return padded_track_weights
    
    else:
        if min2tracks:
            # if only 1 track in vertex, set it off
            mask = (np.sum(padded_track_weights, axis=2) == 1)
            padded_track_weights[mask] = np.zeros(padded_track_weights.shape[2])    

        # Now remove empty vertices and repeat the padding        
        track_weights = np.array([[arr for arr in jet if np.sum(np.array(arr)) > 0] for jet in padded_track_weights], dtype="object") # > 1 drops 1 track vertices
 
        len_vertices = [len(inner_list) for inner_list in track_weights]
        max_n_hadrons = max(len_vertices)     
        padded_track_weights = np.zeros((max_jets, max_n_hadrons, n_tracks))

        for i, outer in enumerate(track_weights):
            for j, inner in enumerate(outer):
                padded_track_weights[i, j, :len(inner)] = inner  # Only copy the actual values, pad with zeros

        return padded_track_weights


    
dataset = LoadDataset(path, Cuts.from_list(cuts), n_jets=n_jets)

good_jets = np.where(dataset["jets"]["HadronConeExclExtendedTruthLabelID"] < 6, 1, 0).astype(bool) # remove double b-jets

# select only jets with single b-tag label

jets = dataset["jets"][good_jets]
tracks = dataset["tracks"][good_jets]
events = dataset["eventwise"][good_jets]
hadrons = dataset["truth_hadrons"][good_jets]

n_tracks = tracks.shape[1]
n_hadrons = hadrons.shape[1]

if plot_Lxy:
    # You need to install JAX to be able to run these.                                                                                                                                                     
    from puma.utils.billoir_vertex_fit import billoir_vertex_fit, billoir_forward
    import jax.numpy as jnp
    from puma.utils.billoir_preprocessing import ListVariables, TransformData, GetTrackWeights

    # Process variables required for Vertex fit
    # Get variables for tracks with perigree representation
    x, track_data, mask_tracks = TransformData(dataset, good_jets, n_tracks=n_tracks, drop_unrelated_hadrons=True)
    seed = jnp.zeros((x.shape[0], 3))
    
    PVtoBeam_X = events['primaryVertexToBeamDisplacementX'] # Beam - PV coordinate
    PVtoBeam_Y = events['primaryVertexToBeamDisplacementY']
    PVtoBeam_Z = events['primaryVertexToBeamDisplacementZ']
    

############################
##      Reco (GN2)      ##
############################

# Get Vertex Index, where are the secondary vertices? 

#reco_track_weights, reco_vertex_index = GetTrackWeights(track_data, incl_vertexing=inclusive_vertex, truth=False, max_sv=max_sv) # if not inclusive you can choose to store more than one vertex i.e. max_sv != 1

## process SV finding ###

vertex_index = tracks["GN2v01_aux_VertexIndex"]
track_origin =  tracks["GN2v01_aux_TrackOrigin"]

tmp_track_weights = SV_Finding(vertex_index, track_origin, inclusive=False,  remove_non_HF_tracks=False, min2tracks=False)

# find vertex with highest number of tracks from PV (with origin = 2)

origin_vertex_candidates = np.array([np.where(row, track_origin[r], 0) for r, row in enumerate(tmp_track_weights)])
pv_candidates = np.where(origin_vertex_candidates == 2, 1, 0) # first check the amount of tracks with origin = 2 for each vertex candidate
mask_tracks_pv = np.array([1 if np.sum(row) > 0 else 0 for row in np.where(np.sum(pv_candidates, axis=2) > 0, 1, 0)]) # make a mask to only modify vertices with at least 1 track from the PV
vertex_most_pv = np.argmax(np.sum(pv_candidates, axis=2), axis=1) # find which vertex has most tracks from origin 2
tracks_from_pv_candidate = origin_vertex_candidates[np.arange(pv_candidates.shape[0]), vertex_most_pv]
tracks_from_pv_candidate = np.where(mask_tracks_pv[:, np.newaxis], tracks_from_pv_candidate, 0) 

# drop vertex with highest number of tracks from PV
clean_vertex_index = np.where(tracks_from_pv_candidate, -1, vertex_index)

reco_track_weights = SV_Finding(clean_vertex_index, track_origin, inclusive=inclusive_vertex, remove_non_HF_tracks=True, min2tracks=True)


if plot_Lxy:

    
    if inclusive_vertex == False: 
        index_reco_tracks = np.argmax(np.sum(reco_track_weights, axis=2), axis=1)
        vertex_fit, vertex_covariance_fit, vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(reco_track_weights[np.arange(reco_track_weights.shape[0]), index_reco_tracks]), jnp.array(seed[:]))

    else:
        vertex_fit, vertex_covariance_fit, vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(reco_track_weights), jnp.array(seed[:]))

    # change dummy values (0 by default) to np.nan
    vertex_fit = np.where(vertex_fit == 0, np.nan, vertex_fit)

    Displacement_x = np.array(vertex_fit[:, 0] + PVtoBeam_X) # the - will have to be a + when we use a newer sample vesion bc Dan changed the definition
    Displacement_y = np.array(vertex_fit[:, 1] + PVtoBeam_Y)
    #Displacement_z = np.array(vertex_fit[:, 2] + PVtoBeam_Z)

    Lxy = np.sqrt( Displacement_x**2 + Displacement_y**2)

    # This is not the error on the displacement but the error on the X, Y, Z position of the SV
    Displacement_x_error = np.sqrt(vertex_covariance_fit[:, 0, 0])
    Displacementx_y_error = np.sqrt(vertex_covariance_fit[:, 1, 1])
    #Displacement_z_error = np.sqrt(vertex_covariance_fit[:, 2, 2])





############################
##      TRUTH (FTAG)      ##
############################

#truth_track_weights, truth_vertex_index = GetTrackWeights(track_data, incl_vertexing=inclusive_vertex, truth=True, max_sv=max_sv) 

vertex_index = tracks["ftagTruthVertexIndex"]
track_origin = tracks["ftagTruthOriginLabel"]

truth_track_weights = SV_Finding(vertex_index, track_origin, inclusive=inclusive_vertex, remove_non_HF_tracks=True, min2tracks=True)

if plot_Lxy:

    if inclusive_vertex == False: 
        index_truth_tracks = np.argmax(np.sum(truth_track_weights, axis=2), axis=1)
        truth_vertex_fit, truth_vertex_covariance_fit, truth_vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(truth_track_weights[np.arange(truth_track_weights.shape[0]), index_truth_tracks]), jnp.array(seed[:])) # the y axis for reco_track_weights indicates which vertex is fitted 0 = leading vertex

    else:
        index_truth_tracks = 0 
        truth_vertex_fit, truth_vertex_covariance_fit, truth_vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(truth_track_weights), jnp.array(seed[:])) # the y axis for reco_track_weights indicates which vertex is fitted 0 = leading vertex


    truth_vertex_fit = np.where(truth_vertex_fit == 0, np.nan, truth_vertex_fit)

    Displacement_x_truth_tracks = np.array(truth_vertex_fit[:, 0] + PVtoBeam_X) # the - will have to be a + when we use a newer sample vesion bc Dan changed the definition
    Displacement_y_truth_tracks = np.array(truth_vertex_fit[:, 1] + PVtoBeam_Y) 
    #Displacement_z_truth_tracks = np.array(truth_vertex_fit[:, 2] + PVtoBeam_Z) 

    Lxy_truth_tracks = np.sqrt( Displacement_x_truth_tracks**2 + Displacement_y_truth_tracks**2)


############################
##   Reco (JetFitter)     ##
############################

JF_track_weights = np.array([[[1 if r == i else 0 for r in row ] for i in range(0, 5)] for row in tracks["JFVertexIndex"]])

if inclusive_vertex == False:
    index_JF_tracks = np.argmax(np.sum(JF_track_weights, axis=2), axis=1)
else:
    JF_track_weights = np.sum(JF_track_weights, axis=1)
    mask = (np.sum(JF_track_weights, axis=1) == 1)
    JF_track_weights[mask] = np.zeros(JF_track_weights.shape[1])
    index_JF_tracks = 0
    JF_vertex_fit, JF_vertex_covariance_fit, JF_vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(JF_track_weights), jnp.array(seed[:]))


if inclusive_vertex == False:
    mask = (np.sum(JF_track_weights, axis=2) == 1)
    JF_track_weights[mask] = np.zeros(JF_track_weights.shape[2])

    if plot_Lxy:
        JF_vertex_fit, JF_vertex_covariance_fit, JF_vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(JF_track_weights[np.arange(JF_track_weights.shape[0]), index_JF_tracks]), jnp.array(seed[:]))

if plot_Lxy:

    # change dummy values (0 by default) to np.nan
    JF_vertex_fit = np.where(JF_vertex_fit == 0, np.nan, JF_vertex_fit)

    Displacement_x_JF = np.array(JF_vertex_fit[:, 0] + PVtoBeam_X) # the - will have to be a + when we use a newer sample vesion bc Dan changed the definition
    Displacement_y_JF = np.array(JF_vertex_fit[:, 1] + PVtoBeam_Y) 

    Lxy_JF = np.sqrt( Displacement_x_JF**2 + Displacement_y_JF**2)

############################
##   Reco (SV1)     ##
############################

if inclusive_vertex == True:

    SV1_track_weights = np.where(tracks["SV1VertexIndex"] >= 0, 1, 0)
    mask = (np.sum(SV1_track_weights, axis=1) == 1)
    SV1_track_weights[mask] = np.zeros(SV1_track_weights.shape[1])

    if plot_Lxy:

        SV1_vertex_fit, SV1_vertex_covariance_fit, SV1_vertex_fit_chi2  = billoir_vertex_fit(jnp.array(x[:]), jnp.array(SV1_track_weights), jnp.array(seed[:]))
        
        SV1_vertex_fit = np.where(SV1_vertex_fit == 0, np.nan, SV1_vertex_fit)

        Displacement_x_SV1 = np.array(SV1_vertex_fit[:, 0] + PVtoBeam_X) # the - will have to be a + when we use a newer sample vesion bc Dan changed the definition
        Displacement_y_SV1 = np.array(SV1_vertex_fit[:, 1] + PVtoBeam_Y) 

        Lxy_SV1 = np.sqrt( Displacement_x_SV1**2 + Displacement_y_SV1**2)


#####################################                                                                                                                                  
##   Now Plot Lxy and Residual     ##                                                                                                                                        
#####################################                                                                                                                                  

if plot_Lxy:

    if inclusive_vertex == True: 
        vertexing = "Inclusive vertexing, "
    else:
        vertexing = "Exclusive vertexing, "

    Truth = Lxy_truth_tracks

    GN2_Residual =  Lxy - Truth
    if inclusive_vertex: SV1_Residual =  Lxy_SV1 - Truth
    JF_Residual = Lxy_JF - Truth 


    for flavour in ["bjets", "cjets"]:
        
        normalise = True
        extra_string = basic_extra_string
        y_axis = "Number of vertices"
        if normalise: 
            extra_string = basic_extra_string+"_norm"
            y_axis = "Normalised number of jets"

    
        # Choose selection
        if flavour == "all":
            f = ""
            flav_str = "(all flavours)"
    
        if flavour == "bjets":
            f = "b"
            flav_str = "$b$-jets"
    
        if flavour == "cjets":
            f = "c"
            flav_str = "$c$-jets"
    
        if flavour == "light":
            f = "light"
            flav_str = "(light jets)"

        selection = (jet_flavour(jets, f))


        w  = 1
        if normalise: 
            n_jets = np.sum(selection)
            w = 1/n_jets

    
        truth_hist = Histogram(Truth[selection], label="MC Truth", histtype="step", alpha=1, colour="black", weights=np.where(~np.isnan(Truth[selection]), w, 0))
        gn2_hist =  Histogram(Lxy[selection], label="GN2", histtype="step", alpha=1, colour="deepskyblue", weights=np.where(~np.isnan(Lxy[selection]), w, 0))
        if inclusive_vertex: sv1_hist = Histogram(Lxy_SV1[selection], label="SV1", histtype="step", alpha=1, colour="pink", weights=np.where(~np.isnan(Lxy_SV1[selection]), w, 0))
        JetFitter_hist = Histogram(Lxy_JF[selection],label="JetFitter", histtype="step", alpha=1, colour="green", weights=np.where(~np.isnan(Lxy_JF[selection]), w, 0))
        
        
        # Initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel=y_axis,
            xlabel="L$_{\mathrm{xy}}$ [mm]",
            logy=True,
            # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
            bins=30,  # you can also define an integer number for the number of bins
            bins_range=(0,30),  # only considered if bins is an integer
            norm=False,
            atlas_first_tag="Simulation Internal",
            atlas_second_tag="$\\sqrt{s}=" + com + "$ TeV, " + mc + "\n" + sample_str + ", " + cut_str+" \n"+vertexing+flav_str,
            figsize=(6, 5),
            y_scale=1.7,
            n_ratio_panels=1,
        )
    
    
        plot_histo.add(truth_hist, reference=True)
        if inclusive_vertex: plot_histo.add(sv1_hist, reference=False)
        plot_histo.add(JetFitter_hist, reference=False)
        plot_histo.add(gn2_hist, reference=False)


        plot_histo.draw()
        
        print(output+"Paper_Histogram_Lxy_"+flavour+extra_string+".png")
        plot_histo.savefig(output+"Paper_Histogram_Lxy_"+flavour+extra_string+".png", transparent=False)


        log_scale = False
        normalise = True
        extra_string = basic_extra_string
        y_axis = "Number of jets"
        if normalise: 
            extra_string = basic_extra_string+"_norm"
            y_axis = "Normalised number of jets"

        if log_scale:
            extra_string = extra_string+"_log"    

        gn2_hist =  Histogram(GN2_Residual[selection], label="GN2", histtype="step", alpha=1, colour="deepskyblue", weights=np.where(~np.isnan(GN2_Residual[selection]), w, 0))
        if inclusive_vertex: sv1_hist = Histogram(SV1_Residual[selection], label="SV1", histtype="step", alpha=1, colour="pink", weights=np.where(~np.isnan(SV1_Residual[selection]), w, 0))
        JetFitter_hist = Histogram(JF_Residual[selection],label="JetFitter", histtype="step", alpha=1, colour="green", weights=np.where(~np.isnan(JF_Residual[selection]), w, 0))

        gn2_mu = np.nanmean(GN2_Residual[selection])
        gn2_std = np.nanstd(GN2_Residual[selection])
        gn2_iqr = iqr(GN2_Residual[selection], nan_policy='omit')

        if inclusive_vertex:
            sv1_mean = np.nanmean(SV1_Residual[selection])
            sv1_std = np.nanstd(SV1_Residual[selection])
            sv1_iqr = iqr(SV1_Residual[selection], nan_policy='omit')
    
        jetfitter_mean = np.nanmean(JF_Residual[selection])
        jetfitter_std = np.nanstd(JF_Residual[selection])
        jetfitter_iqr = iqr(JF_Residual[selection], nan_policy='omit')

        
        # Initialise histogram plot
        plot_histo = HistogramPlot(
            ylabel=y_axis,
            xlabel="L$_{\mathrm{xy}}$ Residual [mm] (reco - truth)",
            logy=log_scale,
            # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
            bins=19,  # you can also define an integer number for the number of bins
            bins_range=(-4.25,4.25),  # only considered if bins is an integer
            norm=False,
            atlas_first_tag="Simulation Internal",
            atlas_second_tag="$\\sqrt{s}=" + com + "$ TeV, " + mc + "\n" + sample_str + ", " + cut_str+" \n"+vertexing+flav_str,
            figsize=(6, 5),
            n_ratio_panels=0,
        )

        ax = plot_histo.axis_top
        text_size = 11
        y0 = 0.65
        x0=0.56
        dy = 0.05
        idx = 0
        xs = 1


        if inclusive_vertex == True: 
            plot_histo.add(sv1_hist, reference=False)
            ax.text(x0, y0-idx*dy, f'$\mu = ${sv1_mean:.2f}, $\sigma = ${sv1_std:.2f}, IQR = {sv1_iqr:.2f}', transform=plot_histo.axis_top.transAxes, fontsize=text_size, color='pink')
            idx +=1
        
        plot_histo.add(JetFitter_hist, reference=False)    
        ax.text(x0, y0-idx*dy, f'$\mu = ${jetfitter_mean:.2f}, $\sigma = ${jetfitter_std:.2f}, IQR = {jetfitter_iqr:.2f}',  transform=plot_histo.axis_top.transAxes,  fontsize=text_size, color='green')
        idx +=1
        plot_histo.add(gn2_hist, reference=True)
        ax.text(x0, y0-idx*dy, f'$\mu = ${gn2_mu:.2f}, $\sigma = ${gn2_std:.2f}, IQR = {gn2_iqr:.2f}',  transform=plot_histo.axis_top.transAxes,  fontsize=text_size, color='deepskyblue')
    
        plot_histo.draw()
        
        plot_histo.savefig(output+"Paper_Histogram_Lxy_Residual_"+flavour+extra_string+".png", transparent=False)


    
def WeightsToIndex(track_weights):
    track_weights_tmp = np.array([[v*(j+1) for j, v in enumerate(r)] for r in track_weights])
    track_weights_tmp = np.sum(track_weights_tmp, axis=1)
    return np.where(track_weights_tmp == 0, -1, track_weights_tmp)
    

if plot_mass:

    from puma.utils.mass import calculate_vertex_mass
    # track Eta
    jet_eta = np.repeat(jets["eta"], n_tracks).reshape(jets.shape[0], n_tracks) # This is needed because jets have a different format than tracks
    track_eta =  tracks["deta"] + jet_eta 

    # track phi
    jet_phi = np.repeat(jets["phi"], n_tracks).reshape(jets.shape[0], n_tracks) 
    track_phi = jet_phi + tracks["dphi"]
    track_phi = np.where(track_phi < -np.pi, 2*np.pi + (jet_phi + tracks["dphi"]), track_phi)
    track_phi = np.where(track_phi > np.pi,  -2*np.pi + (jet_phi + tracks["dphi"]), track_phi)

    if inclusive_vertex:
        reco_track_weights_tmp = reco_track_weights
        truth_track_weights_tmp = truth_track_weights
        JF_track_weights_tmp = JF_track_weights
        SV1_track_weights_tmp = SV1_track_weights

    else:

        truth_track_weights_tmp = WeightsToIndex(truth_track_weights)
        #truth_track_weights_tmp = truth_track_weights[np.arange(truth_track_weights.shape[0]), index_truth_tracks]
        reco_track_weights_tmp = WeightsToIndex(reco_track_weights)
        #reco_track_weights_tmp = reco_track_weights[np.arange(reco_track_weights.shape[0]), index_reco_tracks]    
        JF_track_weights_tmp = WeightsToIndex(JF_track_weights)    # if keeping all 
        #JF_track_weights_tmp = JF_track_weights[np.arange(JF_track_weights.shape[0]), index_JF_tracks]


    GN2_mass = calculate_vertex_mass(tracks["pt"]/1000, track_eta, tracks["dphi"], np.where(reco_track_weights_tmp == 0, -1, reco_track_weights_tmp), particle_mass=0.13957)
    truth_mass = calculate_vertex_mass(tracks["pt"]/1000, track_eta,  tracks["dphi"], np.where(truth_track_weights_tmp == 0, -1, truth_track_weights_tmp), particle_mass=0.13957)
    JF_mass = calculate_vertex_mass(tracks["pt"]/1000, track_eta, tracks["dphi"], np.where(JF_track_weights_tmp == 0, -1, JF_track_weights_tmp), particle_mass=0.13957)

    if inclusive_vertex == True: SV1_mass = calculate_vertex_mass(tracks["pt"]/1000, track_eta, tracks["dphi"], np.where(SV1_track_weights_tmp ==0, -1, SV1_track_weights_tmp), particle_mass=0.13957)

    if inclusive_vertex:
        GN2_mass = np.max(GN2_mass, axis=1)
        if inclusive_vertex: SV1_mass = np.max(SV1_mass, axis=1)
        JF_mass = np.max(JF_mass, axis=1)
        truth_mass = np.max(truth_mass, axis=1)
    else:
        GN2_mass =np.unique(GN2_mass, axis=1).flatten()
        JF_mass = np.unique(JF_mass, axis=1).flatten()
        truth_mass = np.unique(truth_mass, axis=1).flatten()



if inclusive_vertex == True: 
    vertexing = "Inclusive vertexing, "
else:
    vertexing = "Exclusive vertexing, "

normalise = True
extra_string = basic_extra_string
y_axis = "Number of jets"
if normalise: 
    extra_string = basic_extra_string+"_norm_jets"
    y_axis = "Normalised number of jets"


for flavour in ["bjets", "cjets"]:

    # Choose selection
    if flavour == "all":
        f = ""
        flav_str = "(all flavours)"
    
    if flavour == "bjets":
        f = "b"
        flav_str = "$b$-jets"
    
    if flavour == "cjets":
        f = "c"
        flav_str = "$c$-jets"
    
    if flavour == "light":
        f = "light"
        flav_str = "(light jets)"

    selection = jet_flavour(jets, f)

    w  = 1
    if normalise: 
        n_jets = np.sum(selection)
        w = 1/n_jets

    
    truth_hist = Histogram(truth_mass[(selection & (truth_mass > 0.14))], label="MC Truth", histtype="step", alpha=1, colour="black", weights=np.where(~np.isnan(truth_mass[(selection & (truth_mass > 0.14))]), w, 0))
    gn2_hist =  Histogram(GN2_mass[(selection & (GN2_mass > 0.14))], label="GN2", histtype="step", alpha=1, colour="deepskyblue", weights=np.where(~np.isnan(GN2_mass[(selection & (GN2_mass > 0.14))]), w, 0))
    if inclusive_vertex == True: sv1_hist = Histogram(SV1_mass[(selection & (SV1_mass > 0.14))], label="SV1", histtype="step", alpha=1, colour="pink", weights=np.where(~np.isnan(SV1_mass[(selection & (SV1_mass > 0.14))]), w, 0))
    JetFitter_hist = Histogram(JF_mass[(selection & (JF_mass > 0.14))],label="JetFitter", histtype="step", alpha=1, colour="green", weights=np.where(~np.isnan(JF_mass[(selection & (JF_mass > 0.14))]), w, 0))

    
    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel=y_axis,
        xlabel="m$_{SV}$ [GeV]",
        logy=False,
        bins_range=(0,5),  # only considered if bins is an integer
        norm=False,
        atlas_first_tag="Simulation Internal",
        atlas_second_tag="$\\sqrt{s}=" + com + "$ TeV, " + mc + "\n" + sample_str + ", " + cut_str+" \n"+vertexing+flav_str,
        figsize=(6, 5),
        y_scale=1.4, #inclusive

        #y_scale=1.7,
        n_ratio_panels=1,
    )

    plot_histo.add(truth_hist, reference=True)
    if inclusive_vertex: plot_histo.add(sv1_hist, reference=False)
    plot_histo.add(JetFitter_hist, reference=False)
    plot_histo.add(gn2_hist, reference=False)
    

    plot_histo.draw()

    plot_histo.savefig(output+"Paper_Histogram_Mass_"+flavour+extra_string+"_v_cross_check_dphi.png", transparent=False)


