KPL/MK

   Metakernel for the "Navigating OSIRIS-REx" tutorial series:
   OSIRIS-REx on approach to Bennu, June 2018.

   Paths are relative to the tutorial folder (tutorials/v2/), which is the
   working directory when the notebooks run.

\begindata

PATH_VALUES  = ( '../supplementary/supp_data/kernels' )
PATH_SYMBOLS = ( 'KER' )

KERNELS_TO_LOAD = (
    '$KER/locked/lsk/naif0012.tls',
    '$KER/locked/pck/pck00010.tpc',
    '$KER/locked/pck/gm_de431.tpc',
    '$KER/locked/spk/de424.bsp',
    '$KER/scenario/earth_000101_250127_241031.bpc',
    '$KER/locked/spk/earthstns_itrf93_201023.bsp',
    '$KER/locked/spk/earth_topo_201023.tf',
    '$KER/validation/earth_assoc_itrf93.tf',
    '$KER/locked/spk/mar099s.bsp',
    '$KER/scenario/jup310.bsp',
    '$KER/scenario/sat360.bsp',
    '$KER/scenario/bennu_v17.tpc',
    '$KER/scenario/orx_v14.tf',
    '$KER/scenario/orx_180301_181201_180921_od044_v1.bsp'
)

\begintext
