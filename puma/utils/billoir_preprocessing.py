import numpy as np
import jax.numpy as jnp
import h5py
from ftag.hdf5 import H5Reader # I use ftag tools to read the file

from puma.utils.vertexing import  build_vertices, clean_reco_vertices, clean_truth_vertices


def ListVariables(file_path):
    with h5py.File(file_path, "r") as f:
        print(f.keys())
        for k in list(f.keys()):
            print(k)
            print(f[k].dtype.fields.keys())

    return

    
def TransformData(my_data, good_jets, n_tracks=40, drop_unrelated_hadrons = True):

    # Function to calculate the track parameters in the perigree representation.
    # Returns data x with the format n_jets x n_tracks x n_parameters
    # The n_parameters will first have the variables needed for the billoir fit, some will have to be build by hand because not everything is available

    n_jets, max_tracks = my_data["tracks"].shape

    track = my_data["tracks"][:, 0:n_tracks]
    jet = my_data["jets"][:] # Only needed if you need to calculate the track phi from dphi.

    # Start by getting a mask of the real tracks

    # Get real tracks
    track_mask  = np.where(my_data["tracks"]["valid"], 1, 0)

    # Compute Input Variables for Billoir Vertex Fit
    ### set parameters for dummy tracks to 1. They will be masked out by the track weight and if you choose a very low value the fit will not work well.

    d0 = jnp.where(track_mask == 0, 1, -track["d0RelativeToBeamspot"])  # d0RelativeToBeamspot # NEGATIVE for Billoir fit (different definitions between ATLAS and the billoir paper)
    z0 = jnp.where(track_mask == 0, 1, track["z0RelativeToBeamspot"]) 
    
    jet_phi = jnp.repeat(jet["phi"], n_tracks).reshape(n_jets, n_tracks) # This is needed because jets have a different format than tracks
    #phi = track["phi"] # take track phi directly
    # if you calculate track phi from dphi you need the following 3 lines
    phi = jet_phi + my_data["tracks"]["dphi"]
    phi = np.where(phi < -np.pi, 2*np.pi + (jet_phi + my_data["tracks"]["dphi"]), phi)
    phi = np.where(phi > np.pi,  -2*np.pi + (jet_phi + my_data["tracks"]["dphi"]), phi)

    phi = jnp.where(track_mask == 0, 1, phi)

    theta  = jnp.where(track_mask == 0, 1, track["theta"])
    rho    = jnp.where(track_mask == 0, 1, -track["qOverP"]*2*0.2299792*0.001/jnp.sin(track["theta"])) #NEGATIVE for Billoir fit (different definitions between ATLAS and the billoir paper)

    d0_error     = jnp.where(track_mask == 0, 1, track["d0RelativeToBeamspotUncertainty"])
    z0_error     = jnp.where(track_mask == 0, 1, track["z0RelativeToBeamspotUncertainty"])

    phi_error    = jnp.where(track_mask == 0, 1, track["phiUncertainty"])
    theta_error  = jnp.where(track_mask == 0, 1, track["thetaUncertainty"])

    rho_error    = jnp.where(track_mask == 0, 1, jnp.sqrt((2*0.2299792*0.001/jnp.sin(track["theta"]) * track["qOverPUncertainty"])**2  + (track["qOverP"]*2*0.2299792*0.001/(jnp.sin(track["theta"])**2 * jnp.cos(track["theta"])) * track["thetaUncertainty"] )**2) )

    track_origin = jnp.where(track_mask == 0, 1, track["GN2v01_aux_TrackOrigin"])
    track_vertex = jnp.where(track_mask == 0, 1, track["GN2v01_aux_VertexIndex"])
     
    x = jnp.stack([d0, z0, phi, theta, rho, d0_error, z0_error, phi_error, theta_error, rho_error, track_origin, track_vertex], axis = 2)

    if drop_unrelated_hadrons == True:
        x = x[good_jets]
        track = track[good_jets]
        track_mask = track_mask[good_jets]

    return x, track, track_mask



# Get the vertex indices and track weights! Which tracks belong to which vertex according? The track origin is used for the cleaning
## This is slower and uses the already implemented SV finding functions 
def GetTrackWeights(track_data, incl_vertexing=False, truth=False, max_sv=1):

    if truth:
        raw_vertex_index = track_data["ftagTruthVertexIndex"] # your raw vertex
        track_origin = track_data["ftagTruthOriginLabel"]

    else:
        # Reco Level
        raw_vertex_index = track_data["GN2v01_aux_VertexIndex"] # your raw vertex
        track_origin = track_data["GN2v01_aux_TrackOrigin"]

    # Now clean vertices
    vertex_index  = raw_vertex_index.copy()

    # Prepare mask for filling up
    #dummy_track_weights = jnp.zeros((vertex_index.shape[0], max_sv, n_tracks))
    track_weights = jnp.zeros((vertex_index.shape[0], max_sv, track_data.shape[1]))

    #track_weights = jnp.where(dummy_track_weights == 0, 0, dummy_track_weights) #why np.nan?
    
    for i in range(track_data["GN2v01_aux_VertexIndex"].shape[0]):

        if truth:
            vertex_index[i] = clean_truth_vertices(
                vertex_index[i], track_origin[i], incl_vertexing=incl_vertexing
            )
        
        else:
            vertex_index[i] = clean_reco_vertices(
                vertex_index[i], track_origin[i], incl_vertexing=incl_vertexing
            )

        vertices = build_vertices(vertex_index[i]) # Convert indices to true/false

        for j in range(0, max_sv):
            try:
                track_weights = track_weights.at[i, j].set(vertices[j])
            except:
                continue


    return track_weights, vertex_index


