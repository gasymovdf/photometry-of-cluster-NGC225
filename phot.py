import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm
import os
import glob
import time
from astropy import units as u
from lmfit import Model
from astropy import wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import ascii
from astropy.table import Column
from astropy.nddata.utils import Cutout2D
from photutils import aperture_photometry, CircularAperture
from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from photutils.psf import IterativelySubtractedPSFPhotometry
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
# plt.rc('font', family='sans-serif')
# plt.rc('font', family='cursive')
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'normal'
from astropy.utils import iers
import warnings
from astropy.utils.exceptions import AstropyWarning
iers.conf.auto_download = False
warnings.simplefilter('ignore', AstropyWarning)


def bkg(data, SNR=5, box_size=30, filter_size=3):
    '''
    Применение медианного фильтра к изображению.
    '''
    sigma_clip = SigmaClip(sigma=SNR)
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (box_size, box_size), filter_size=(filter_size, filter_size),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    #hdu = fits.PrimaryHDU()
    mean_bkg = np.mean(bkg.background)
    fits.writeto('background.fits', bkg.background, overwrite=True)
    return data - bkg.background, mean_bkg


def COORD_XY(OBJ, DATE, data_path='./data', astrm=False, dx=0, dy=0):
    if astrm:
        files = glob.glob(f"{data_path}/{DATE}/{OBJ}/*.fit")
        hdu = fits.open(files[0])
        data, head = hdu[0].data.astype(np.float32), hdu[0].header
        os.system(f'/usr/local/astrometry/bin/solve-field  --downsample 2 --resort --no-verify -O --ra ' +
                   head['OBJCTRA'].replace(' ', ':')+' --dec '+head['OBJCTDEC'].replace(' ', ':') +
                   ' --radius 0.1 -L 0.65 -H 0.70 -u app ' + files[0])

    coord_table = ascii.read(f'{data_path}/{OBJ}.csv', format='csv', fast_reader=False)
    wcs_file = glob.glob(f"{data_path}/{DATE}/{OBJ}/*.wcs")[0]
    w = wcs.WCS(wcs_file)
    coord_table.add_column(Column([np.nan]*len(coord_table)), name='X')
    coord_table.add_column(Column([np.nan]*len(coord_table)), name='Y')
    stars = SkyCoord(ra=coord_table['RA'], dec=coord_table['Dec'], frame='icrs', unit=(
        u.hourangle, u.deg))
    coord_table['X'], coord_table['Y'] = w.wcs_world2pix(
        stars.ra.deg, stars.dec.deg, 0)
    coord_table['X'] = coord_table['X'] + dx
    coord_table['Y'] = coord_table['Y'] + dy
    return coord_table

def apert_photometry(data, coord_table, rad_aperture=15):
    filtr = [coord_table['good_star']]
    positions = np.transpose((coord_table['X'][filtr], coord_table['Y'][filtr]))
    apertures = CircularAperture(positions, r=rad_aperture)
    phot_table = aperture_photometry(data, apertures)
    return phot_table['aperture_sum']


def PSF_photometry(data, coord_table, sigma_psf=10, scale=0.67, step=0.5):
    FLUX = []
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(data)
    iraffind = IRAFStarFinder(threshold=3.5*std,
                              fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                              minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                              sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    # psf_model.x_0.fixed = True
    # psf_model.y_0.fixed = True
    
    pos = Table(names=['x_0', 'y_0'], data=[coord_table['X'],
                                            coord_table['Y']])[coord_table['good_star']]

    photometry = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                    group_maker=daogroup,
                                                    bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model,
                                                    fitter=LevMarLSQFitter(),
                                                    niters=1, fitshape=(41,41))
    result_tab = photometry(image=data, init_guesses=pos)
    return result_tab['flux_fit']


def STAR_FLUX(coord_table, OBJ, f, numb=[1, 2, 3], rad_aperture=10, standart=False,\
                main_obj=False, apert=False, PSF=False, PLOT_data=False, flux_10_m=None, err_flux_10_m=None):
    file = f'./data/2020-08-19/{OBJ}/{OBJ}-1MHz-76mcs-PreampX4-000' + '{}' + f'{f}.fit'
    file_0 = file.format(numb[0])
    hdu_0 = fits.open(file_0)
    data, head = hdu_0[0].data.astype(np.float32), hdu_0[0].header
    for i in numb[1:]:
        data += fits.open(file.format(i))[0].data.astype(np.float32)
    data = data/(3*head['EXPTIME'])
    int_max = 0.9*2**16/(head['EXPTIME'])
    data_clear, backgr_mean = bkg(data, SNR=5, box_size=30, filter_size=10)

    good_coord = (coord_table['X']>=0)&(coord_table['X']<len(data))&\
        (coord_table['Y']>=0)&(coord_table['Y']<len(data))
    for i, j, good, num in zip(coord_table['X'], coord_table['Y'], good_coord, range(len(coord_table))):
        if not good:
            continue
        cutout = Cutout2D(data, (i, j), (2*rad_aperture+1, 2*rad_aperture+1)).data
        # print(i, j)
        # plt.imshow(cutout)
        if np.max(cutout) > int_max:
            good_coord[num] = False
        if np.mean(np.sort(np.ravel(cutout))[-2:]) < np.median(cutout) + 3*np.std(cutout):
            good_coord[num] = False
        # plt.colorbar()
        # plt.show()
        # plt.close()

    if "good_star" not in coord_table.columns:
        coord_table.add_column(
                Column([True]*len(coord_table)), name="good_star")
        coord_table['good_star'] = good_coord
    coord_table['good_star'] = coord_table['good_star']*good_coord


    if apert:
        FLUX = apert_photometry(data_clear, coord_table, rad_aperture=rad_aperture)
    if PSF:
        FLUX = PSF_photometry(data, coord_table, sigma_psf=rad_aperture)

    if PLOT_data:
        plt.figure(figsize=(13, 13))
        plt.imshow(data, norm=LogNorm(), cmap='gray')
        apertures = CircularAperture(np.transpose((coord_table['X'], coord_table['Y'])),\
                                        r=rad_aperture)
        apertures.plot(color='blue', lw=1., alpha=0.5)
        plt.colorbar()
        plt.show()

    if standart:
        filtr = [(coord_table[f] != -100)&(coord_table['good_star'])]
        col_name = f"FLUX_{f}"
        coord_table.add_column(
            Column([np.nan]*len(coord_table)), name=col_name)
        coord_table[col_name][filtr] = FLUX[filtr]
        coord_table.add_column(Column([np.nan]*len(coord_table)), name=f'10m_FLUX_{f}')
        coord_table[f'10m_FLUX_{f}'][filtr] = coord_table[col_name][filtr] * 10**((coord_table[f][filtr] - 10)/2.5)

    if main_obj:
        col_name = f"FLUX_{f}"
        filtr = [coord_table['good_star']]
        coord_table.add_column(
            Column([np.nan]*len(coord_table)), name=col_name)
        coord_table.add_column(Column([np.nan]*len(coord_table)), name=f'MAG_{f}')
        coord_table.add_column(Column([np.nan]*len(coord_table)), name=f'E_MAG_{f}')
        coord_table[col_name][filtr] = FLUX
        coord_table[f'MAG_{f}'][filtr] = 10 - 2.5 * np.log10(coord_table[col_name][filtr]/flux_10_m[f])
        coord_table[f'E_MAG_{f}'][filtr] = - 2.5 * np.log10(1 - err_flux_10_m[f]/flux_10_m[f])
    return coord_table


def PLOT(coord_table, model_BV, model_UB, xlim=[-0.5, 2.], ylim=[2.0, -1.2], OBJ_STND=None):
    plt.figure(figsize=(8, 13))
    filtr = [coord_table['good_star']]
        
    plt.errorbar(coord_table['MAG_B'][filtr] - coord_table['MAG_V'][filtr],
                coord_table['MAG_U'][filtr] - coord_table['MAG_B'][filtr],\
                xerr=coord_table['E_MAG_B'][filtr] + coord_table['E_MAG_V'][filtr],\
                yerr=coord_table['E_MAG_U'][filtr] + coord_table['E_MAG_B'][filtr], color='red', fmt='*',\
                capthick=1, linestyle='None', markersize=7, ecolor='gray', capsize=4) 
    plt.plot(model_BV, model_UB, 'b', lw=1)
    plt.title('Diagram (U-B)(B-V)')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("B-V")
    plt.ylabel("U-B")
    plt.savefig(f'color-color-{OBJ_STND}.png')


def process(DATE, OBJ_STND):
    flux_10_m, err_flux_10_m = {}, {}
    coord_table_stnd = COORD_XY(OBJ_STND, DATE, astrm=False)
    for f in filt:
        coord_table_stnd = STAR_FLUX(coord_table_stnd, OBJ_STND, f, numb=\
                                        [1, 2, 3], rad_aperture=5, standart=True, apert=False, PSF=True)
        flux_10_m[f] = np.mean(coord_table_stnd[f'10m_FLUX_{f}'][~np.isnan(coord_table_stnd[f'10m_FLUX_{f}'])])
        err_flux_10_m[f] = np.std(coord_table_stnd[f'10m_FLUX_{f}'][~np.isnan(coord_table_stnd[f'10m_FLUX_{f}'])])
    
    print(OBJ_STND)
    for i in flux_10_m.keys():
        print(f"10m point in {i} filter: {flux_10_m[i]:.2f} +/- {err_flux_10_m[i]:.2f} ({err_flux_10_m[i]/flux_10_m[i]*100:.2f} %)")

    coord_table_main = COORD_XY(main_obj, DATE, astrm=False)
    for f in filt:
        coord_table_main = STAR_FLUX(coord_table_main, main_obj, f, numb=\
                                        [4, 5, 6], rad_aperture=5, main_obj=True, apert=False, PSF=True, flux_10_m=flux_10_m, err_flux_10_m=err_flux_10_m)

    Model_hdu = fits.open('model.fits')
    Model_Table_cold = Model_hdu[2].data
    Model_Table_hot = Model_hdu[3].data
    model_BV = np.concatenate(
        (Model_Table_cold['B-V'][-22:], Model_Table_hot['B-V'][-54:]))
    model_UB = np.concatenate(
        (Model_Table_cold['U-B'][-22:], Model_Table_hot['U-B'][-54:]))

    ascii.write(coord_table_main, f'photometry_NGC_225_{DATE}.csv', format='csv', fast_writer=False, overwrite=True)
    ascii.write(coord_table_main, f'photometry_{OBJ_STND}_{DATE}.csv', format='csv', fast_writer=False, overwrite=True)
    PLOT(coord_table_main, model_BV, model_UB,
         xlim=[-0.5, 2.], ylim=[2.0, -1.2], OBJ_STND=OBJ_STND)


if __name__ == "__main__":
    dates = ['2020-08-17', '2020-08-18', '2020-08-19', '2020-09-15']
    standart_objs = ['GD246', 'NGC7790']
    main_obj = 'NGC225'
    filt = {'U', 'B', 'V', 'Rc', 'Ic'}
    DATE = dates[2]
    OBJ_STND = standart_objs[0]
    process(DATE, OBJ_STND)
