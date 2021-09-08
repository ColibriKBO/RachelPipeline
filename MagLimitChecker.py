import matplotlib.pyplot as plt
import sys
import pandas as pd

# filename = sys.argv[1]
filename = "./20210804-Field1.cat"
gfilename = "./gaia_edr3_Field1.psv"

SR = 0.015 # Search distance in degrees

data = pd.read_csv(filename, delim_whitespace=True, skiprows=14)
data.columns = ['Number','XWIN','YWIN','XWORLD','YWORLD','MAG','MAGERROR','FLUX','FLUXERROR','FLUXRADIUS','FWHM','BACKGROUND','RA','DEC']

gdata = pd.read_csv(gfilename, delimiter="|", skiprows=4, low_memory=False)
gdata.columns = ["ID","gRA","gDEC","PAR","gMAG","bMAG","rMAG"] # Make sure that these columns are the ones downloaded from the data server

gdata = gdata.drop('ID', axis=1) # May as well get rid of the ID column

data['BMAG'] = 99
data['GMAG'] = 99
data['RMAG'] = 99

for i in range(len(data.index)):
    RA = data.loc[i,'RA']
    DEC = data.loc[i,'DEC']

    df = gdata[(gdata.gRA < (RA+SR))]
    df = df[(df.gRA > (RA-SR))]
    df = df[(df.gDEC < (DEC+SR))]
    df = df[(df.gDEC > (DEC-SR))]
    df.sort_values(by=["gMAG"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df.index)>1:
        data.loc[i,'BMAG'] = df.loc[0]['bMAG']
        data.loc[i,'GMAG'] = df.loc[0]['gMAG']
        data.loc[i,'RMAG'] = df.loc[0]['rMAG']

print('Pickling...')
data.to_pickle('out.pkl')
print('Pickled...')

print('Plotting figure...')

plt.figure(figsize=(14,10))

plt.scatter(data['GMAG'],data['MAG'], color='green')

plt.xlabel('Gaia DR2 G-band Magnitude', size=15)
plt.ylabel('Instrumental Magnitude', size=15)
plt.xlim(4,13)
plt.ylim(-15,-2)
plt.tight_layout()
plt.savefig('Imag-vs-Gmag.png', dpi=300)

plt.show()


