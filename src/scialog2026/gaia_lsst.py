import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u

# NOTE: r=16-20 is overlap region between Gaia and LSST (Eyer+ 2012)


def pm_unc(plx_unc, dr=4):
    """
    For a given parallax uncertainty (in uas), return the proper motion
    uncertainty (in uas/yr) for DR4 or DR5.

    Based on expected performance tabulated in
    https://www.cosmos.esa.int/web/gaia/science-performance#astrometric%20performance.
    """
    if dr == 4:
        return 0.54 * plx_unc
    elif dr == 5:
        return 0.27 * plx_unc


def lsst_position_uncertainty(snr, a=1.0):
    """
    Convert LSST SNR to astrometric precision. Astrometric precision
    scales as a/SNR, where a is the airy disk sigma (in arcsec)
    """
    return a / snr


gaia_sensitivity = pd.read_csv(
    "data/gaia_sensitivity.csv", comment="#", names=["gmag", "dr4", "dr5"], skiprows=1
)


lsst_sensitivity = pd.read_csv(
    "data/lsst_sensitivity.csv", comment="#", names=["mag", "log_inv_snr"], skiprows=1
)

alphas = [1, 0.6, 0.3]

plt.figure(figsize=(5, 5))

for i, m_r in enumerate(
    [
        16,
        18,
        20,
    ]
):  # NOTE: I'm assume g-band and r-band magnitudes are approx the same
    # for given r magnitude, pick characteristic distance (assume solar primary)
    solar_absmag_rband = 4.5
    dist_pc = 10 ** ((-solar_absmag_rband + 5 + m_r) / 5)

    # baselines
    delta_t_gaia_dr4 = 5 * u.yr  # 2014-2019
    delta_t_gaia_dr4_lsst0 = 13 * u.yr  # 2014-2027
    delta_t_gaia_lsst_fullsurvey = 23 * u.yr  # 2014-2037

    plx_unc_gaia_dr4 = gaia_sensitivity[gaia_sensitivity.gmag == m_r].dr4.values[0]
    pm_unc_gaia_dr4 = pm_unc(plx_unc_gaia_dr4, dr=4) * u.uas / u.yr

    # NOTE: I'm assuming we don't actually get proper motions from LSST early data; we just get positions.
    # Therefore the proper motion we compute is (LSST position - average Gaia position)/baseline, and the uncertainty on
    # this number is sqrt(LSST position error**2 + Gaia position error**2)/baseline.
    # I assume the quadratic error is just LSST position error for simplicity.
    snr_lsst = 10 ** (
        1 / lsst_sensitivity[lsst_sensitivity.mag == m_r].log_inv_snr.values[0]
    )

    pos_unc_lsst = lsst_position_uncertainty(snr_lsst)
    pm_unc_gaia_dr4_lsst = pos_unc_lsst * u.mas / delta_t_gaia_dr4_lsst0

    # compute acceleration sensitivity of Gaia DR4 and Gaia-LSST
    gaia_acceleration_floor = pm_unc_gaia_dr4 / delta_t_gaia_dr4
    gaia_dr4_lsst_acceleration_floor = (
        pm_unc_gaia_dr4_lsst / delta_t_gaia_lsst_fullsurvey
    )

    # compute detectability limit in mass/orbital separation
    separation_array = np.logspace(-1, np.log10(400), int(1e3))  # [au]

    min_detectable_mass_gaia_mjup = (
        (10 / separation_array) ** 2
        * (40 / dist_pc)
        * (0.01 * u.mas / u.yr**2 / gaia_acceleration_floor)
    ) ** (-1)

    min_detectable_mass_gaia_lsst_mjup = (
        (10 / separation_array) ** 2
        * (40 / dist_pc)
        * (0.01 * u.mas / u.yr**2 / gaia_dr4_lsst_acceleration_floor)
    ) ** (-1)

    plt.plot(
        separation_array,
        (min_detectable_mass_gaia_mjup * u.M_jup).to(u.M_sun),
        label=f"r mag = {m_r}",
        alpha=alphas[i],
        color="rebeccapurple",
        ls="--",
    )
    plt.plot(
        separation_array,
        (min_detectable_mass_gaia_lsst_mjup * u.M_jup).to(u.M_sun),
        # label="Gaia DR4 + LSST",
        alpha=alphas[i],
        color="rebeccapurple",
    )

plt.legend()
ax = plt.gca()
ax.minorticks_on()
ax.tick_params(which="both", top=True, right=True, direction="in")
plt.ylim(1, 100)
plt.xlim(0, 350)
# plt.xscale("log")
plt.yscale("log")
plt.xlabel("separation [au]")
plt.ylabel("mass [Msol]")
plt.savefig("oom.png", dpi=250)
