#! /usr/bin/env python
import xarray as xr
import pandas as pd

#
# Functions for working with dictionaries describing each set of results (lw_info and sw_info below)
#
def expand_info(info):
    '''Expand a dictionary with multiple values of forcing and/or physics into a list of dictionaries'''
    out = [{k:info[k] for k in ["name", "version", "institution", "location"]} for i in range(len(info["forcing"]))]
    [out[i].update({"physics":info["physics"][i],
                    "forcing":info["forcing"][i]}) for i in range(len(info["forcing"]))]
    return(out)

def construct_realization(info):
    ''' Construct a string describing model/source, physics variant, forcing variant'''
    return("{}-p{}-f{}".format(info["name"], info["physics"], info["forcing"]))

def open_one_file_set(info, vars):
    '''Open one realization's worth of files'''
    name = info["name"]
    member = 'r1i1p{}f{}'.format(info["physics"], info["forcing"])
    if "http" in info["location"]:
        loc_tmpl   = '{}/RFMIP/{}/{}/rad-irf/{}/Efx/'.format(info["location"], info["institution"], name, member) + \
                     '{}' + '/gn/{}/'.format(info["version"])
    else:
        loc_tmpl   = info["location"] + "/"
    fname_tmpl = '{}' + '_Efx_{}_rad-irf_{}_gn.nc'.format(name, member)
    print("Opening " + name)
    x = xr.open_mfdataset([loc_tmpl.format(v) + fname_tmpl.format(v) for v in vars], combine='by_coords')
    return(x)
#
# Using intake would also have been sensible. 
#
def construct_ds_from_dict(info, expt_labels, vars = ["rld", "rlu"]):
    ''' Contruct an xarray Dataset with global-mean up- and down fluxes for each realization and each experiement.
    Mean fluxes are supplemented with net fluxes, absorption, and forcing

    The location of the data is constructed from the keys in argument info, depending on whether the data is remote
    (assumed to be the Earth System) or local

    Parameters:
        info (dict): a dictionary with keys name, location, institution, physics, forcing, and realization
        expt_labels (array of strings): Names of the experiments, length needs to match size of "experiment" dimension in files
        vars = ["rld", "rlu"]: variables to be read
    '''
    #
    # info is a list of dictionaries
    #
    out = xr.concat([open_one_file_set(i, vars) for i in info],
                    dim=pd.Index([i["realization"] for i in info], name="realization"))

    out["forcing_index"] = xr.Variable(dims="realization", data=[i["forcing"] for i in info])
    out["physics_index"] = xr.Variable(dims="realization", data=[i["physics"] for i in info])
    out = out.assign_coords(expt=expt_labels)
    #
    # Weighted mean across profiles - profiles_weights should be the same across all realizations
    #
    x = (out * out.profile_weight/out.profile_weight.sum(dim='site')).sum(dim='site')
    # Profile weight depend on site but we've averaged over all those
    x = x.drop("profile_weight")
    # Variable attributes get lost in that reduction
    for v in x.variables: x[v].attrs = out[v].attrs
    out = x

    toa = out.isel(expt=0).plev.argmin().values
    sfc = out.isel(expt=0).plev.argmax().values

    if "rld" in out:
        band = "l"
        #
        # Net flux; atmospheric absorption
        #
        net = out["r" + band + "d"] - out["r" + band + "u"]
        net.attrs = {"standard_name":"net_downward_longwave_flux_in_air",
                     "variable_name":"rln",
                     "units":out["r" + band + "u"].attrs["units"],
                     "cell_methods":out["r" + band + "u"].attrs["cell_methods"]}
        out["r" + band + "n"] = net
        out["r" + band + "a"] = net.sel(level=toa) - net.sel(level=sfc)
        out["r" + band + "a"].attrs = \
                     {"standard_name":"atmosphere_net_rate_of_absorption_of_longwave_energy",
                     "variable_name":"rla",
                     "units":out["r" + band + "u"].attrs["units"],
                     "cell_methods":out["r" + band + "u"].attrs["cell_methods"]}
        out = compute_forcing(out, band)
    if "rsd" in out:
        band = "s"
        #
        # Net flux; atmospheric absorption
        #
        net = out["r" + band + "d"] - out["r" + band + "u"]
        net.attrs = {"standard_name":"net_downward_shortwave_flux_in_air",
                     "variable_name":"rln",
                     "units":out["r" + band + "u"].attrs["units"],
                     "cell_methods":out["r" + band + "u"].attrs["cell_methods"]}
        out["r" + band + "n"] = net
        out["r" + band + "a"] = net.sel(level=toa) - net.sel(level=sfc)
        out["r" + band + "a"].attrs = \
                     {"standard_name":"atmosphere_net_rate_of_absorption_of_shortwave_energy",
                     "variable_name":"rla",
                     "units":out["r" + band + "u"].attrs["units"],
                     "cell_methods":out["r" + band + "u"].attrs["cell_methods"]}
        out = compute_forcing(out, band)
    return(out)
################################################################
def compute_forcing(da, band_label):
    '''
    Compute forcing - up, down, net, absorption
    Forcing calculated as
     -(X-PD) for experiments with "PI" in the title,
       X-PD for +4K experiments,
       X-PI_CO2 for the CO2 experiments
       X-PI for all other experiments
    Parameters:
        da (xarray Dataset): variables rxd, rxu, rxn, and rxa for up, down, net, and column absorption where x is band_label
        band_label (char): 'l' for longwave, 's' for shortwave
    '''
    piExpts   = [f for f in expt_labels if "PI"  in f]
    tempExpts = [f for f in expt_labels if "+4K" in f]
    co2Expts  = [f for f in expt_labels if "xCO2" in f]
    for v in ["u", "d", "n", "a"]:
        fVar = "f"+band_label+v
        rVar = "r"+band_label+v
        da[fVar] = da[rVar] - da[rVar].sel(expt="PI")
        da[fVar].load()
        da[fVar].loc[dict(expt=piExpts)  ] = -(da[rVar].loc[dict(expt=piExpts)  ] - da[rVar].sel(expt="PD"))
        da[fVar].loc[dict(expt=co2Expts) ] =   da[rVar].loc[dict(expt=co2Expts) ] - da[rVar].sel(expt="PI CO2")
        da[fVar].loc[dict(expt=tempExpts)] =   da[rVar].loc[dict(expt=tempExpts)] - da[rVar].sel(expt="PD")
        da[fVar].attrs = {"description":"instantaneous radiative forcing with respect to present-day conditions for " + rVar,
                         "units":da[rVar].attrs["units"],
                         "cell_methods":da[rVar].attrs["cell_methods"]}
    return(da)


########################################################################
if __name__ == '__main__':
    #
    # Information about the available calculations.
    #   Maybe the intake python module would be a good alternative?
    #
    lw_info      = expand_info({'name':'ARTS-2-3',
                           'version':'v20190620',
                           'institution':'UHH',
                           'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                           'physics':[1,1,1,2],
                           'forcing':[1,2,3,1]})
    lw_info.extend(expand_info({
                           'name':'GFDL-GRTCODE',
                           'version':'v20180701',
                           'institution':'NOAA-GFDL',
                           'location':'https://esgdata.gfdl.noaa.gov/thredds/dodsC/gfdl_dataroot4',
                           'physics':[1,1,1],
                           'forcing':[1,2,3]}))
    lw_info.extend([{'name':'GFDL-RFM-DISORT',
                     'version':'v20180701',
                     'institution':'NOAA-GFDL',
                     'location':'https://esgdata.gfdl.noaa.gov/thredds/dodsC/gfdl_dataroot4',
                     'physics':1,
                     'forcing':2}])
    lw_info.extend([{'name':'HadGEM3-GC31-LL',
                    'version':'v20190605',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':2,
                    'forcing':2},
                    {'name':'HadGEM3-GC31-LL',
                    'version':'v20190417',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':2,
                    'forcing':3},
                    {'name':'HadGEM3-GC31-LL',
                    'version':'v20191031',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':3,
                    'forcing':2},
                    {'name':'HadGEM3-GC31-LL',
                    'version':'v20191030',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':3,
                    'forcing':3}])
    lw_info.extend([{'name':'LBLRTM-12-8',
                     'version':'v20190514',
                     'institution':'AER',
                     'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                     'physics':1,
                     'forcing':1},
                    {'name':'LBLRTM-12-8',
                     'version':'v20190517',
                     'institution':'AER',
                     'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                     'physics':1,
                     'forcing':2},
                    {'name':'LBLRTM-12-8',
                     'version':'v20190516',
                     'institution':'AER',
                     'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                     'physics':1,
                     'forcing':3}])
    lw_info.extend(expand_info({
                           'name':'4AOP-v1-5',
                           # 'version':'v20200611', # OpenDAP access to 4AOP isn't working
                           'version':'v20200402',
                           'institution':'IPSL',
                           # 'location':'https://vesg.ipsl.upmc.fr/thredds/dodsC/cmip6',
                           'location':'/Users/robert/Dropbox/Scientific/Projects/RFMIP/RFMIP-IRF/benchmarks/4AOP-v1-5',
                           'physics':[1,1,1],
                           'forcing':[1,2,3]}))
    for x in lw_info: x.update({"realization":construct_realization(x)})

    #
    # Could possibly extract the SW from the LW by choosing a subset of the dictionaries.
    #
    sw_info = expand_info({
                           'name':'GFDL-GRTCODE',
                           'version':'v20180701',
                           'institution':'NOAA-GFDL',
                           'location':'https://esgdata.gfdl.noaa.gov/thredds/dodsC/gfdl_dataroot4',
                           'physics':[1,1,1],
                           'forcing':[1,2,3]})
    sw_info.extend([{'name':'GFDL-RFM-DISORT',
                     'version':'v20180701',
                     'institution':'NOAA-GFDL',
                     'location':'https://esgdata.gfdl.noaa.gov/thredds/dodsC/gfdl_dataroot4',
                     'physics':1,
                     'forcing':2}])
    sw_info.extend([{'name':'HadGEM3-GC31-LL',
                    'version':'v20190605',
                    'institution':'MOHC',
                    #'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'location':'/Users/robert/Dropbox/Scientific/Projects/RFMIP/RFMIP-IRF/benchmarks/UKESM',
                    'physics':2,
                    'forcing':2},
                    {'name':'HadGEM3-GC31-LL',
                    'version':'v20190417',
                    'institution':'MOHC',
                    #'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'location':'/Users/robert/Dropbox/Scientific/Projects/RFMIP/RFMIP-IRF/benchmarks/UKESM',
                    'physics':2,
                    'forcing':3},
                    {'name':'HadGEM3-GC31-LL',
                    'version':'v20191031',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':3,
                    'forcing':2},
                    {'name':'HadGEM3-GC31-LL',
                    'version':'v20191030',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':3,
                    'forcing':3}])
    sw_info.extend([{'name':'LBLRTM-12-8',
                     'version':'v20190514',
                     'institution':'AER',
                     'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                     'physics':1,
                     'forcing':1}])
    for x in sw_info: x.update({"realization":construct_realization(x)})
    #
    # GCM contributions
    gcm_info = [{'name':'GFDL-CM4',
                 'version':'v20180701',
                 'institution':'NOAA-GFDL',
                 'location':'https://esgdata.gfdl.noaa.gov/thredds/dodsC/gfdl_dataroot4',
                 'physics':1,
                 'forcing':2}]
    if True: gcm_info = [] # Because AM4 is missing for now.
    gcm_info.extend([{'name':'HadGEM3-GC31-LL',
                    'version':'v20190605',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':1,
                    'forcing':2}])
    gcm_info.extend([{'name':'HadGEM3-GC31-LL',
                    'version':'v20190417',
                    'institution':'MOHC',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':1,
                    'forcing':3}])
    gcm_info.extend([{'name':'RTE-RRTMGP-181204',
                    'version':'v20191007',
                    'institution':'RTE-RRTMGP-Consortium',
                    'location':'https://esgf3.dkrz.de/thredds/dodsC/cmip6',
                    'physics':1,
                    'forcing':1}])
    gcm_info.extend([{'name':'CanESM5',
                    'version':'v20200402', # Needs updating after publishing on the ESGF
                    'institution':'CCCma',
                    'location':'/Users/robert/Dropbox/Scientific/Projects/RFMIP/RFMIP-IRF/CanESM5',
                    'physics':1,
                    'forcing':1}])
    gcm_info.extend([{'name':'MIROC6',
                    'version':'v20200430',
                    'institution':'MIROC',
                    'location':'https://esgf-data2.diasjp.net/thredds/dodsC/esg_dataroot/CMIP6',
                    'physics':1,
                    'forcing':1}])

    for x in gcm_info: x.update({"realization":construct_realization(x)})
    ########################################################################
    #
    # Experiment labels come from the file describing atmospheric conditions used by
    #   all the calculations.
    #
    profiles = "http://aims3.llnl.gov/thredds/dodsC/user_pub_work/input4MIPs/CMIP6/RFMIP/" + \
               "UColorado/UColorado-RFMIP-1-2/atmos/fx/multiple/none/v20190401/multiple_input4MIPs_radiation_RFMIP_UColorado-RFMIP-1-2_none.nc"
    inputs = xr.open_dataset(profiles)
    expt_labels = [e.decode('utf-8') for e in inputs.expt_label.values]
    inputs.close()
    print("Read experiment labels")
    #
    # Simplify some of those labels
    #
    expt_labels = [s.replace("Present day (PD)", "PD") for s in expt_labels]
    expt_labels = [s.replace("Pre-industrial (PI) greenhouse gas concentrations", "PI") for s in expt_labels]
    expt_labels = [s.replace('"future"', "future") for s in expt_labels]
    ########################################################################
    if False:
        lw = construct_ds_from_dict(lw_info, expt_labels, vars=["rld", "rlu"])
        print("Longwave realizations")
        for r in lw.realization.values: print("  " + r)
        lw.to_netcdf("lw-lbl-models.nc", format="NETCDF4_CLASSIC")
        lw.close()
        print("Wrote longwave summary")

    if False:
        sw = construct_ds_from_dict(sw_info, expt_labels, vars=["rsd", "rsu"])
        print("Shortwave realizations")
        for r in sw.realization.values: print("  " + r)
        sw.to_netcdf("sw-lbl-models.nc", format="NETCDF4_CLASSIC")
        sw.close()
        print("Wrote shortwave summary")

    gcm = construct_ds_from_dict(gcm_info, expt_labels, vars=["rld", "rlu", "rsd", "rsu"])
    print("GCM realizations")
    for r in gcm.realization.values: print("  " + r)
    gcm.to_netcdf("gcm-models.nc", format="NETCDF4_CLASSIC")
    gcm.close()
    print("Wrote GCM summary")
