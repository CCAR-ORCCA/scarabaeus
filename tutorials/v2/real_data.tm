KPL/MK

   Additional kernels for chapter 9 (real DSN data), loaded on top of
   orex_approach.tm.

   - earth_topo_201023.tf: topocentric frames of the DSN stations, which
     the real measurement models query.
   - jup310.bsp, sat360.bsp: the real measurement models evaluate the
     ET - TAI time transformation with the positions of the Jupiter and
     Saturn centers, which only the satellite ephemerides provide.
     They are large (about 1.7 GB together):
     `python get_data.py --real-data` downloads them.

\begindata

PATH_VALUES  = ( '../supplementary/supp_data/kernels' )
PATH_SYMBOLS = ( 'KER' )

KERNELS_TO_LOAD = (
    '$KER/locked/spk/earth_topo_201023.tf',
    '$KER/scenario/jup310.bsp',
    '$KER/scenario/sat360.bsp'
)

\begintext
