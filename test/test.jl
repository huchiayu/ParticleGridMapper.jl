#push!(LOAD_PATH, pwd())

using ParticleGridMapper

using HDF5
using StaticArrays
using Statistics
using PyPlot
using LinearAlgebra
using .Threads


function read_snap(filename)

    T=Float64
    header::Dict = h5readattr(filename, "/Header")
    boxsize::T = header["BoxSize"]
    time::T    = header["Time"]

    N_gas::Int64 = header["NumPart_ThisFile"][1]

    pos_gas::Matrix{T} = h5read(filename, "PartType0/Coordinates");
    vel_gas::Matrix{T} = h5read(filename, "PartType0/Velocities");
    rho::Vector{T}     = h5read(filename, "PartType0/Density");
    u::Vector{T}       = h5read(filename, "PartType0/InternalEnergy");
    m_gas::Vector{T}   = h5read(filename, "PartType0/Masses");
    hsml::Vector{T}    = h5read(filename, "PartType0/SmoothingLength");
    #scal::Vector{T}       = h5read(filename, "PartType0/PassiveScalarField");

    id_gas::Vector{Int64} = h5read(filename, "PartType0/ParticleIDs");
    #=
    abund::Matrix{T} = h5read(filename, "PartType0/ChemicalAbundancesSG");
    fH2ss::Vector{T}    = h5read(filename, "PartType0/ShieldingFactorH2");
    fH2dust::Vector{T}    = h5read(filename, "PartType0/ShieldingFactorDust");

    col_tot::Matrix{T}    = h5read(filename, "PartType0/TreecolColumnDensitiesAll");
    col_H2::Matrix{T}    = h5read(filename, "PartType0/TreecolColumnDensitiesH2");
    col_CO::Matrix{T}    = h5read(filename, "PartType0/TreecolColumnDensitiesCO");

    Tdust::Vector{T}    = h5read(filename, "PartType0/DustTemperature");
    =#
    return N_gas, pos_gas, vel_gas, rho, u, m_gas, hsml, id_gas,
        #abund, fH2ss, fH2dust, col_tot, col_H2, col_CO, Tdust,
        boxsize, time
end




function vec2svec(vec::Matrix{T}) where {T}
    svec = [SVector{3,T}(vec[:,i]) for i in 1:size(vec,2)]
end
function mat2smat(mat::Array{T,3}) where {T}
    smat = [SMatrix{3,3,T}(mat[:,:,i]) for i in 1:size(mat,3)]
end
T = Float64
const N = 3

const XH = 0.71
const BOLTZMANN=1.3806e-16
const PROTONMASS=1.6726e-24
const GRAVCON=6.67e-8
const UnitMass_in_g = 1.989e43
const UnitLength_in_cm    = 3.085678e21
const UnitTime_in_s = 3.08568e+16
const UnitDensity_in_cgs = UnitMass_in_g / UnitLength_in_cm^3
const UnitDensity_in_pccm = UnitDensity_in_cgs/PROTONMASS
const Year_in_s = 31556926.
const fac_col = (UnitMass_in_g/UnitLength_in_cm^2)*(XH/PROTONMASS)

#function mapping()
#for i in 13:13
#for i in 300:300
i=690
@show i
snap = ""
if i < 10
    snap = "00" * string(i)
elseif i < 100
    snap = "0" * string(i)
else
    snap = string(i)
end

#file_path = "./nH10_box150pc_S4_N1e6_myIC"
#file_path = "/Users/chu/code/snapshots/SNbox_CO/nH1_box250pc_S2_N1e6_myIC_SF_G0CRwSFR_turbIC"

#file_path = "/Users/chu/simulations/SNbox_CO/nH1_box100pc_S2_N1e5_myIC"
file_path = "/Users/chu/simulations/turbbox/"
#file_path = "/Users/chu/simulations/tallbox/SFSNPI_N1e6_gS10H250dS40H250_soft4_SFLJ4_eff0p5_stoIMFfix_rngSF_convSF/"
#file_path = "./"
filename = file_path * "/snap_" * snap * ".hdf5"

N_gas, pos, vel, rho, u, mass, hsml, id_gas,
#abund, fH2ss, fH2dust, col_tot, col_H2, col_CO, Tdust,
boxsize, time = read_snap(filename);

#pos[3,:] .+= 5.000
#pos .+= 0.0001

X = vec2svec(pos);
#Vel = vec2svec(vel);

const Ngrid = 200
const Ngrid_x = Ngrid
#const Ngrid_x = 1
#const Ngrid_y = div(Ngrid,100)
const Ngrid_y = Ngrid
const Ngrid_z = Ngrid
#const BOXSIZE_X = 1.
#const BOXSIZE_Y = BOXSIZE_X
#const BOXSIZE_Z = BOXSIZE_X

ngrids = (Ngrid_x, Ngrid_y, Ngrid_z)
boxsizes = (boxsize, boxsize, boxsize)
#xmin = (0., 0.5, 0.)
#xmax = (1., 0.5, 1.)
xmin = (0., 0., 0.)
xmax = (1., 1., 1.)
#xmin = (0.5, 0., -2.)
#xmax = (0.5, 1., 2.)
#xmin = (0., 0., -1.)
#xmax = (1., 1., 1.)

ix,iy = 1,3

cm = "viridis";
fig, ax1 = PyPlot.subplots(1,1, figsize=(12,12))
#fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(2,3,figsize=(12,8))

icen=div(Ngrid,2)


rho_grid_NGP = map_particle_to_3Dgrid_NGP(rho, mass, X, ngrids, xmin, xmax);
rho_grid_NGP_thread = map_particle_to_3Dgrid_NGP_thread(rho, mass, X, ngrids, xmin, xmax);
rho_grid = map_particle_to_2Dgrid_loopP(rho, mass./rho, X, hsml, xmin, xmax,
    ngrids=ngrids,
    xaxis=ix, yaxis=iy,
    #column=false,
    pbc = (true, true, false)
    );
rho_grid_thread = map_particle_to_2Dgrid_loopP_thread(rho, mass./rho, X, hsml, xmin, xmax,
    ngrids=ngrids, xaxis=ix, yaxis=iy,
    #column=false,
    pbc = (true, true, false)
    );

#rho_grid_noCar = map_particle_to_2Dgrid_loopP_noCar(rho, mass./rho, X, hsml, xmin, xmax,
    #ngrids=ngrids, xaxis=ix, yaxis=iy,
    #column=false,
    #pbc = (true, true, false)
    #);

#rho_3Dgrid_thread = map_particle_to_3Dgrid_loopP_thread(rho, mass./rho, X, hsml, xmin, xmax,
#    ngrids=ngrids);
#rho_3Dgrid_thread = dropdims(sum(rho_3Dgrid_thread, dims=3),dims=3);
#@show all(rho_grid_thread .≈ rho_grid)
#@show all(rho_grid_thread .≈ rho_grid_noCar)
#@show all(rho_3Dgrid_thread .≈ rho_grid_thread)
#rho_grid = map_particle_to_grid_loopP_noCar(rho, mass, rho, X, hsml, boxsizes);
#tree = buildtree(X, hsml, mass, mass_H2, mass_CO, boxsizes)
#rho_grid = map_particle_to_grid(rho, mass, rho, X, hsml, boxsizes, tree);

Nmin, Nmax = 20, 26
nmin, nmax = -3, 3
ii=2
#aa=dropdims(sum(rho_grid_NGP, dims=ii),dims=ii);
aa=rho_grid_thread
#aa .*= fac_col
im1 = ax1.imshow(log10.(aa'), cmap=cm, aspect="equal", origin="lower",
    #vmin=Nmin, vmax=Nmax,
    extent=(xmin[ix],xmax[ix],xmin[iy],xmax[iy]), interpolation="none")
cb = colorbar(im1, ax=ax1, orientation="horizontal", aspect=30, fraction=0.025, pad=0.05)
cb.set_label("log10 NH (cm^-2)")
ax1.xaxis.set_visible(false)
ax1.yaxis.set_visible(false)

#=
rho_slice = dropdims(mean(rho_grid[:,:,icen-5:icen+5], dims=3),dims=3)
rho_slice .*= 287
im3 = ax3.imshow(log10.(rho_slice' .+ 1e-2), cmap=cm, aspect="equal", origin="lower",
    vmin=nmin, vmax=nmax,
    extent=(0,BOXSIZE_X,0,BOXSIZE_X), interpolation="none")
cb = colorbar(im3, ax=ax3, orientation="horizontal", aspect=30, fraction=0.025, pad=0.05)
cb.set_label("log10 nH [cm^-3])")
ax3.xaxis.set_visible(false)
ax3.yaxis.set_visible(false)


xH2 = abund[1,:]
rhoH2_grid = map_particle_to_grid_N(rho.*xH2, mass, rho, X, hsml, boxsizes);

ii=3
aa=dropdims(sum(rhoH2_grid, dims=ii),dims=ii);
aa .*= fac_col
im2 = ax2.imshow(log10.(aa' .+ 1e10), cmap=cm, aspect="equal", origin="lower",
    vmin=Nmin, vmax=Nmax,
    extent=(0,BOXSIZE_X,0,BOXSIZE_X), interpolation="none")
cb = colorbar(im2, ax=ax2, orientation="horizontal", aspect=30, fraction=0.025, pad=0.05)
cb.set_label("log10 NH2 (cm^-2)")
#ax1 = gca()
ax2.xaxis.set_visible(false)
ax2.yaxis.set_visible(false)
#tight_layout(w_pad=-1, h_pad=-1, pad=-0.5)
#colorbar(im, orientation="vertical", fraction=0.15)
#cb = colorbar(im, orientation="vertical", aspect=30, pad=-0.1)

rhoH2_slice = dropdims(mean(rhoH2_grid[:,:,icen-5:icen+5], dims=3),dims=3)
rhoH2_slice .*= 287
im4 = ax4.imshow(log10.(rhoH2_slice' .+ 1e-2), cmap=cm, aspect="equal", origin="lower",
    vmin=nmin, vmax=nmax,
    extent=(0,BOXSIZE_X,0,BOXSIZE_X), interpolation="none")
cb = colorbar(im4, ax=ax4, orientation="horizontal", aspect=30, fraction=0.025, pad=0.05)
cb.set_label("log10 nH2 [cm^-3])")
ax4.xaxis.set_visible(false)
ax4.yaxis.set_visible(false)

cm="cool"
fH2_grid = map_particle_to_grid_N(fH2ss.*fH2dust, mass, rho, X, hsml, boxsizes);
icen=div(Ngrid,2)
fH2_slice = dropdims(mean(fH2_grid[:,:,icen-5:icen+5], dims=3),dims=3)
im5 = ax5.imshow(log10.(fH2_slice' .+ 1e-8), cmap=cm, aspect="equal", origin="lower",
    vmin=-8, vmax=0,
    extent=(0,BOXSIZE_X,0,BOXSIZE_X), interpolation="none")
cb = colorbar(im5, ax=ax5, orientation="horizontal", aspect=30, fraction=0.025, pad=0.05)
cb.set_label("log10 f_shield (H2)")
ax5.xaxis.set_visible(false)
ax5.yaxis.set_visible(false)

cm="plasma"
temp = 80 .* u
temp_grid = map_particle_to_grid_N(temp, mass, rho, X, hsml, boxsizes);
icen=div(Ngrid,2)
temp_slice = dropdims(mean(temp_grid[:,:,icen-5:icen+5], dims=3),dims=3)
im6 = ax6.imshow(log10.(temp_slice' .+ 1), cmap=cm, aspect="equal", origin="lower",
    vmin=1, vmax=6,
    extent=(0,BOXSIZE_X,0,BOXSIZE_X), interpolation="none")
cb = colorbar(im6, ax=ax6, orientation="horizontal", aspect=30, fraction=0.025, pad=0.05)
cb.set_label("log10 Temperature [K])")
ax6.xaxis.set_visible(false)
ax6.yaxis.set_visible(false)

if i < 1000
    snap = "0" * snap
end
tight_layout()
#savefig(file_path*"/box_"*snap*".png")
#clf()
#end

#return rho_grid
#end
#rho_grid_N = mapping();
=#
