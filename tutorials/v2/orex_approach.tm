KPL/MK

   Metakernel for the "Navigating OSIRIS-REx" tutorial series:
   OSIRIS-REx on approach to Bennu, June 2018.

   Paths are relative to the tutorial folder (tutorials/v2/), which is the
   working directory when the notebooks run. `python get_data.py` downloads
   these kernels (about 41 MB). Chapter 9 also loads real_data.tm.

\begindata

PATH_VALUES  = ( '../supplementary/supp_data/kernels' )
PATH_SYMBOLS = ( 'KER' )

KERNELS_TO_LOAD = (
    '$KER/locked/lsk/naif0012.tls',
    '$KER/locked/spk/de424.bsp',
    '$KER/scenario/earth_000101_250127_241031.bpc',
    '$KER/locked/spk/earthstns_itrf93_201023.bsp',
    '$KER/scenario/orx_180301_181201_180921_od044_v1.bsp'
)

\begintext
