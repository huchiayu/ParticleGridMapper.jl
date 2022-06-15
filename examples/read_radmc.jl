using Mmap
using NumericalIntegration
using PyPlot
using HDF5

file_path_radmc = "./"

#i=620
#nx=512

fname_image = joinpath(file_path_radmc,"image.bout")
fname_level = joinpath(file_path_radmc, "levelpop_co.bdat")

const BOLTZMANN = 1.3807e-16 #erg/K
const LIGHTSPEED = 29979245800. #in cm/s
const wavelength_0 = 2600.75763346 #in micron, CO J=1-0
const PLANCK = 6.6262e-27  #erg s
const wavelength_0_cm = wavelength_0 .* 1e-4 #in cm
const freq0 = LIGHTSPEED ./ wavelength_0_cm #Hz

const XH = 0.71
const PROTONMASS=1.6726e-24
const UnitMass_in_g = 1.989e43
const UnitLength_in_cm    = 3.085678e21
const UnitTime_in_s = 3.08568e+16
const UnitDensity_in_cgs = UnitMass_in_g / UnitLength_in_cm^3
const UnitDensity_in_pccm = UnitDensity_in_cgs/PROTONMASS
const fac_col = (UnitMass_in_g/UnitLength_in_cm^2)*(XH/PROTONMASS)

##################
get_Tex(f1,f2) = PLANCK * freq0 / BOLTZMANN / (log(f1*3/f2))

function read_level(fname_level)

  raw = Mmap.mmap(fname_level)
  ndata = div(length(raw),8)
  raw_r = reshape(raw, 8, ndata)

  iformat, precision, ncell, nrlevels_subset, level1, level2 = reinterpret(Int64, raw_r[:,1:6])
  data = reinterpret(Float64, raw_r[:,7:end])[:]
  nCO_1 = data[1:2:end]
  nCO_2 = data[2:2:end]

  Tex = get_Tex.(nCO_1, nCO_2)

  return Tex, nCO_1, nCO_2
end
##################

function read_2D_cartesian(fname)
    fid=h5open(fname,"r")
    NHtot = h5read(fname, "gas/map_NHtot_time")
    NH2 = h5read(fname, "gas/map_NH2_time")
    NCI = h5read(fname, "gas/map_NCI_time")
    NCO = h5read(fname, "gas/map_NCO_time")
    #NCp = h5read(fname, "gas/map_NCp_time")
    close(fid)
    return NHtot[:,:,1], NH2[:,:,1], NCO[:,:,1]
end

function read_2D(fname)
    fid=h5open(fname,"r")
    NHtot = h5read(fname, "AMRmaps2D/NHtot")
    NH2   = h5read(fname, "AMRmaps2D/NH2")
    NHI   = h5read(fname, "AMRmaps2D/NHI")
    NCO   = h5read(fname, "AMRmaps2D/NCO")
    NCI   = h5read(fname, "AMRmaps2D/NCI")
    NCp   = h5read(fname, "AMRmaps2D/NCp")
    close(fid)
    return NHtot, NH2, NHI, NCO, NCI, NCp
end

function read_3D(fname)
    fid=h5open(fname,"r")
    nHtot = h5read(fname, "AMRmaps3D/nHtot_AMR_MFM")
    nH2   = h5read(fname, "AMRmaps3D/nH2_AMR_MFM")
    nHI   = h5read(fname, "AMRmaps3D/nHI_AMR_MFM")
    nCO   = h5read(fname, "AMRmaps3D/nCO_AMR_MFM")
    nCI   = h5read(fname, "AMRmaps3D/nCI_AMR_MFM")
    nCp   = h5read(fname, "AMRmaps3D/nCp_AMR_MFM")
    temp  = h5read(fname, "AMRmaps3D/temp_AMR_MFM")
    vx    = h5read(fname, "AMRmaps3D/vx_AMR_MFM")
    vy    = h5read(fname, "AMRmaps3D/vy_AMR_MFM")
    vz    = h5read(fname, "AMRmaps3D/vz_AMR_MFM")
    close(fid)
    return nHtot, nH2, nHI, nCO, nCI, nCp, temp, vx, vy, vz
end

get_TB_true(Imu, mu) = PLANCK * mu / BOLTZMANN / (log( (2.0 * PLANCK * mu^3 / LIGHTSPEED^2 / Imu) + 1.0 ) )
get_TB(Imu)          = PLANCK * freq0 / BOLTZMANN / (log( (2.0 * PLANCK * freq0^3 / LIGHTSPEED^2 / Imu) + 1.0 ) )

function read_COimage_spectrum(fname_image)
    raw = Mmap.mmap(fname_image)
    ndata = div(length(raw),8)
    raw_r = reshape(raw, 8, ndata)
    nx, ny, nlam = reinterpret(Int64, raw_r[:,2:4])

    #npix_x = reinterpret(Int64, raw_r[:,5]) #doesn't work...
    #npix_y = reinterpret(Int64, raw_r[:,6])
    npix_x = npix_y = Int( sqrt( (ndata -1-2-1-2 - nlam) / nlam ) )

    wavelength = reinterpret(Float64, raw_r[:,7:7+nlam-1])[:]

    #npix_x by npix_y spectrum of CO
    image = reinterpret(Float64, raw_r[:,7+nlam:end])[:]
    image = reshape(image, npix_x, npix_y, nlam)

    wavelength_shift = wavelength .- wavelength_0 #in micron
    vel_shift = @. wavelength_shift / wavelength_0 * LIGHTSPEED #in cm/s
    vel_shift_kms = vel_shift .* 1e-5 #in km/s

    wavelength_cm = wavelength .* 1e-4 #in cm
    freq = LIGHTSPEED ./ wavelength_cm

    WCO = [integrate(vel_shift_kms, image[i,j,:], SimpsonEven()) for i in 1:nx, j in 1:ny] .* (0.5 * wavelength_0_cm^2 / BOLTZMANN)

    TR = (0.5 * wavelength_0_cm^2 / BOLTZMANN) .* image
    #TB_true = get_TB_true.(image,freq) #unnecessary (almost the same as TB)
    TB = get_TB.(image)

    #TB_c = TB[1,1,div(nlam-1,2)+1]
    #TR_c = TR[1,1,div(nlam-1,2)+1]
    #@show TB_c, TR_c

    return WCO, TR, TB, vel_shift_kms, nx, ny, nlam
end

#read ascii outputs
#data = readdlm("/Users/chu/simulations/tallbox/SFSNPI_N1e7_gS10H250dS40H250_soft0p2_from100Myr_Z1/Npix256_uniform_vturb3/levelpop_co.dat")

WCO, TR, TB, vel_shift_kms, nx, ny, nlam = read_COimage_spectrum(fname_image)

Tex, nCO_1, nCO_2 = read_level(fname_level)
