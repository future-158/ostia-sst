import bz2
import io
import glob
from ftplib import FTP
from pathlib import Path

from omegaconf import OmegaConf

cfg = OmegaConf.load('conf/config.yml')

years = cfg.years


OUT_DIR = Path(cfg.catalogue.raw)
ftp = FTP('ftp.nodc.noaa.gov')
ftp = FTP('ftp-oceans.ncei.noaa.gov')
ftp.login()

prefix  = '/pub/data.nodc/ghrsst/L4/GLOB/UKMO/OSTIA'
for year in years:
    day_folders = ftp.nlst( prefix + '/' + str(year))
    for folder in day_folders:
        files = ftp.nlst(folder)

        assert len(files) == 1
        file = files[0]

        year = Path(file).parent.name
        index = Path(file).name
        dest = OUT_DIR / f"{year}_{index}.nc"
        if dest.exists():
            continue
        
        bio = io.BytesIO()
        with open(dest, 'wb') as f_out:
            ftp.retrbinary( 'RETR {}'.format(file), bio.write)
        
        bio.seek(0)
        with bz2.BZ2File(bio.read()) as f_in, open(dest, 'wb') as f_out:
            for data in iter(lambda: f_in.read(10000000 * 1024), b''):
                f_out.write(data)

            

    




