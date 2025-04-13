import logging,warnings
import lightkurve as lk
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from tesscentroidvetting import centroid_vetting, show_transit_margins
from version import __version__


def set_css():
	st.markdown("""
		<style>
			.js-plotly-plot .plotly, .js-plotly-plot .plotly div {
				margin-bottom: 0px;
			}
			.spc{
				font-size: 13px;
				padding-bottom: 0.2rem;
			}
			.stFormSubmitButton button{
				float:right;
				color: navy;
			}
			[data-testid="stSidebar"][aria-expanded="true"]{
				min-width: 300px;
				max-width: 300px;
			}
			[data-testid="stSidebarContent"] {
				background-color: whitesmoke;
			}
			div[data-testid="stSidebarUserContent"] h1{
				font-size: 2.7rem;
				text-align: center;
			}
			.credits, a, a:visited {
				font-size: 0.8rem;
				text-align: center;
			}
			a, a:visited {
				color: navy;
				font-weight:bold;
			}

			div[data-testid="stMainBlockContainer"] {
				max-width: 90rem;
				padding: 2.2rem 2.5rem 0 2.5rem;
			}
			div[data-testid="stMarkdownContainer"] p{
				font-size: 1.1rem;
			}
			div[data-testid="stVerticalBlock"]{
				gap: 0rem;
			}
			h1{
				padding-bottom: 0.3rem;
			}
		</style>
		""", unsafe_allow_html=True)

@st.cache_resource(ttl='1d')
def load_mdumps():
	if 'mdumps' not in st.session_state:
		try:
			df = pd.read_csv('https://tess.mit.edu/public/files/Table_of_momentum_dumps.csv', comment='#')
			last_mdump = df.iloc[-1,0][:16]
			mdumps = df.iloc[:,1].values
			np.save('data/mom_dumps.npy', mdumps)
		except:
			mdumps = np.load('data/mom_dumps.npy')
		if len(mdumps):
			st.session_state.mdumps = mdumps
	return mdumps

@st.fragment
def plot():
	tit = st.session_state.ss_tit
	df = st.session_state.ss_df
	dumps = st.session_state.ss_dumps
	t0 = 0
	dur = 0
	fig = px.scatter(df, x='time', y='flux', color_discrete_sequence=['navy']) #darkslategrey'])
	fig.update_traces(marker={'size': 3.3})
	minx = min(df['time']) - 0.7
	maxx = max(df['time']) + 0.7
	fig.update_xaxes(range=[minx-0.05, maxx+0.05])
	fig.update_xaxes(showline=True,
		 linewidth=0.5,
		 linecolor='black',
		 mirror=True)

	fig.update_yaxes(showline=True,
		 linewidth=0.5,
		 linecolor='black',
		 mirror=True)

	fig.update_layout(title=dict(text=tit, y=0.96, font=dict(size=20, color='grey')))
	fig.update_layout(yaxis = dict(tickfont = dict(size=13), tickformat = '.4f'), xaxis = dict(tickfont = dict(size=13)))
	fig.update_xaxes(showgrid=True, gridcolor='#ddd',title_standoff = 1, minor=dict(ticklen=4, tickcolor="grey", nticks=5))
	fig.update_yaxes(showgrid=True, gridcolor='#ddd')
	fig.update_layout(height=285)
	fig.update_layout(plot_bgcolor='whitesmoke')
	fig.update_layout(margin=dict(l=60, r=10, t=50, b=35),)
	miny = min(df['flux'])
	maxy = max(df['flux'])
	min2y = miny-(maxy-miny)/20
	max2y = maxy+(maxy-miny)/20
	y0 = min2y
	y1 = (max2y-min2y)/3+min2y
	for x in dumps:
		fig.add_shape(type="line",
			x0=x, y0=y0, x1=x, y1=y1,
			line=dict(color="pink",width=2) #, dash="dashdot")
		)

	event = st.plotly_chart(fig, use_container_width=True, on_select="rerun") #, theme=None)
	if event["selection"]["box"] != []:
		x1 = event["selection"]["box"][0]['x'][0]
		x2 = event["selection"]["box"][0]['x'][1]
		t0 = round((x1+x2)/2, 3)
		dur = round(abs(x2-x1),3)
	ph = st.empty()
	if t0 != 0:
		ph.empty()
		spaces = '&nbsp;&nbsp;&nbsp;&nbsp;'
		st.write('**Selected Transit**: '+ spaces + spaces + 'Epoch = :red[**' +str(t0) + '**]'+ spaces + 'Duration = :red[**'+str(dur) + '**] days.')
		display_button(tit, t0, dur)
	else:
		with ph:
			st.html('<div class="spc" style="color:dimgrey;">(Select a transit or press [<b>HELP</b>] button for instructions)</div>')
	return t0, dur

def get_tpf():
	if 'ss_tpf' not in st.session_state:
		tit = st.session_state.ss_tit
		ticid = st.session_state.ss_tic
		sector = st.session_state.ss_sec
		TICstr = 'TIC ' + str(ticid)
		aut = ''
		if '(SPOC)' in tit:
			aut = 'SPOC'
		elif '(TESS-SPOC)' in tit:
			aut = 'TESS-SPOC'
		if len(aut):
			try:
				tpf = lk.search_targetpixelfile(TICstr, sector=sector, mission='TESS', author=aut).download()
			except:
				return False
		else:
			try:
				sres = lk.search_tesscut(TICstr, sector=sector)
			except:
				return False
			if len(sres):
				try:
					tesscut = sres[0].download(cutout_size=11, quality_bitmask=1073749231)
					tpf = get_corrected_tpf(tesscut)
				except:
					return False
			else:
				return False
		st.session_state.ss_tpf = tpf
	return True


@st.fragment
def display_button(tit, t0, dur):
	ph2 = st.empty()
	with ph2:
		buttonp = st.button("Get Centroid Vetting", type="primary")

	if buttonp:
		with ph2:
			st.html('<div class="spc"><i>Getting TargetPixelFile...</i></div>')

		if not get_tpf():
			with ph2:
				st.write('***Error getting taregetpixelfile. Press [PLOT] button to try again...***')
			st.stop()

		with ph2:
			st.html('<div class="spc"><i>Calculating centroid, please wait...</i></div>')

		fig1, fig2, masked_pixels = get_centroids(t0, dur)

		with ph2.container():
			if fig1 == None:
				st.write('***Error getting centroid. Press [PLOT] button to try again...***')
			else:
				npix = len(masked_pixels)
				if npix == 0:
					st.html('<div class="spc">&nbsp;<div>')
				else:
					st.write('***:red[Warning]: '+ str(npix) + ' brighter pixels were masked around the edges.***')
				st.pyplot(fig1)
				with st.columns([2,3.2,2])[1]:
					st.html('<div class="spc">&nbsp;</div>')
					st.html('<div style="text-align:center;color:navy;">In-Transit and Out-of-Transit used cadences:</div>')
					st.pyplot(fig2)

def aperture_phot(image,aperture):
	flux = np.sum(image[aperture == 1])
	return flux

def get_corrected_tpf(tpf):
	# based on:
	# https://spacetelescope.github.io/notebooks/notebooks/MAST/TESS/interm_tesscut_astroquery/interm_tesscut_astroquery.html
	bg_mask = ~tpf.create_threshold_mask(threshold=0.001, reference_pixel=None)
	tot_bg_pixels = bg_mask.sum()
	bg_flux = np.array(list(map(lambda x: aperture_phot(x, bg_mask), tpf.hdu[1].data['FLUX']))) / tot_bg_pixels
	bckg = np.repeat(bg_flux[:, np.newaxis, np.newaxis], 11, axis=1)
	bckg = np.repeat(bckg, 11, axis=2)
	tpf.hdu[1].data['FLUX_BKG'] = bckg
	tpf.hdu[1].data['FLUX'] -= bckg
	mask = get_target_mask(tpf)
	tpf.hdu[2].data[mask] = 2
	return tpf

def get_target_mask(tpf):
	# get a 3x3 mask around the target
	pix_x, pix_y = tpf.wcs.all_world2pix([(tpf.ra, tpf.dec)], 0)[0]
	xx, yy = int(pix_x + 0.5), int(pix_y + 0.5)
	data = tpf.flux[0]
	mask = np.full(data.shape, False)
	if yy < 1:
		yy = 1
	if xx < 1:
		xx = 1
	if yy > (data.shape[0] - 2):
		yy = data.shape[0] - 2
	if xx > (data.shape[1] - 2):
		xx = data.shape[1] - 2
	mask[yy - 1 : yy + 2, xx - 1 : xx + 2] = True
	return mask

def get_centroids(t0, dur):
	tpf = st.session_state.ss_tpf
	ticid = st.session_state.ss_tic
	try:
		res = centroid_vetting(tpf, [t0], dur, mask_edges=True, ticid=ticid)
	except:
		return None, None, []

	fig1 = res["fig"]
	masked_pixels = res["masked_pixels"]
	fig2 = show_transit_margins(tpf, [t0], dur)
	del res, tpf
	return fig1, fig2, masked_pixels

@st.fragment()
def help_button():
	with st.columns([1,1,1])[1]:
		st.html('<br>')
		if st.button('HELP'):
			help()

@st.dialog('Tess Transit Vetting instructions', width='large')
def help():
	st.image('data/prt_scrtv.png')
	st.stop()


if __name__ == '__main__':
	st.set_page_config(page_title="TESS Transit Vetting", layout="wide")

	set_css()
	try:
		xticid = st.query_params.tic #  ?tic=165795955
		try:
			xsector = st.query_params.sec
		except:
			xsector = ''
	except:
		xticid = ''
		xsector = ''

	mdumps = load_mdumps()

	ticid = 0
	sector = 0
	with st.sidebar:
		st.title('**TESS Transit Vetting**')
		st.html('&nbsp;')
		with st.form("my_form"):
			tic = st.text_input('**TIC number:**', value=xticid, placeholder='', max_chars=10)
			if tic != '':
				try:
					ticid = int(tic)
				except ValueError:
					ticid=0
			ssector = st.text_input('**Sector No.:**', value=xsector, placeholder='', max_chars=3)
			if ssector != '':
				try:
					sector = int(ssector)
				except ValueError:
					sector = 0
			st.html('<div class="credits">&nbsp;</div>')
			st.form_submit_button('**PLOT**')

		st.html('<div align="right">v'+__version__+'</div>')
		st.markdown('<div class="credits">Github source code: <a href="https://github.com/exo-pt/Transit-vetting-app">Tess transit vetting</a><br/>Using ' +\
				'<a href="https://github.com/exo-pt/TESS-Centroid_vetting">Tess-Centroid Vetting</a>' +\
				'<br/><a href="https://github.com/lightkurve/lightkurve">Lightkurve</a>' +\
				' and <a href="https://github.com/plotly/plotly.py">Plotly</a> Python packages.</div>', unsafe_allow_html=True)
		help_button()

	splash = st.empty()
	if (ticid == 0) or (sector == 0):
		with splash.container():
			st.html('<h2>Tess Transit Vetting instructions</h2>') #, anchor=False)
			with st.columns([2,30,2])[1]:
				st.html('Choose a TIC number and a sector in the Sidebar and press PLOT')
				st.image('data/prt_scrtv.png')
				st.html('<i>(These instructions are also available in the Help button in the sidebar)</i>')
	else:
		if 'ss_tic' not in st.session_state:  #first time only
			st.session_state.ss_tic = ticid
			st.session_state.ss_sec = sector
		else:
			if ticid != st.session_state.ss_tic or sector != st.session_state.ss_sec:
				st.session_state.ss_tic = ticid
				st.session_state.ss_sec = sector
				if 'ss_tpf' in st.session_state:
					del st.session_state.ss_tpf
				if 'ss_tit' in st.session_state:
					del st.session_state.ss_tit
					del st.session_state.ss_df
					del st.session_state.ss_dumps

		splash.empty()
		TICstr = 'TIC '+ str(ticid)
		st.header(TICstr)
		placeholder = st.empty()

		if not 'ss_tit' in st.session_state:
			with placeholder:
				st.html('<div class="spc"><i>Searching sector '+str(sector)+' lightcurve...</i></div>')
			try:
				sres=lk.search_lightcurve(TICstr, mission='TESS', sector=sector)
			except:
				sres = ''
				st.error('Error in lk.search_lightcurve... Try again.')
				st.stop()
			if len(sres) == 0:
				with placeholder:
					st.html('<div class="spc"><i>Searching sector '+str(sector)+' (TESScut)...</i></div>')
				try:
					sres=lk.search_tesscut(TICstr, sector=sector)
				except:
					sres=""
				if len(sres) == 0:
					if sres=='':
						st.error('Error in lk.search_tesscut... Try again.')
					else:
						st.error('No available lightcurve data from SPOC, TESS_SPOC, QLP, ELEANOR or TESScut')
					st.stop()
				author = 'TESScut'
				index = 0
			else:
				df = sres.table.to_pandas()
				authors = ['SPOC', 'TESS-SPOC', 'QLP', 'GSFC-ELEANOR-LITE']
				author = 0
				index = 0
				for auth in range(4):
					idx = df.index[(df['sequence_number'] == sector) & (df['author']==authors[auth]) & (df['exptime'] > 100)]
					if len(idx):
						author = authors[auth]
						index = idx[0]
						break

			st.html('&nbsp;')

			warnings.simplefilter("ignore")
			logging.getLogger("lightkurve").setLevel(logging.ERROR)

			tit = 'Sector ' + str(sector) + ' (' + author +')'
			with placeholder:
				st.html('<div class="spc"><i>Downloading ' + tit + '...</i></div>')
			#try:
			match author:
				case 'TESScut':
					try:
					   tesscut = sres[0].download(cutout_size=11, quality_bitmask=1073749231)
					except:
						st.error('Error downloading Tesscut. Try again...')
						st.stop()
					tpf = get_corrected_tpf(tesscut)
					lc0 = tpf.to_lightcurve()
					lc0 = lc0.remove_outliers(sigma_lower=20, sigma_upper=3).normalize().remove_nans()
					st.session_state.ss_tpf = tpf
					del tpf, tesscut
				case _:
					try:
						lc0 = sres[index].download(quality_bitmask=1073749231).remove_outliers(sigma_lower=20, sigma_upper=3).normalize().remove_nans()
					except:
						st.error('Error downloading lightcurve. Try again...')
						st.stop()

			st.html('&nbsp;<br><br>')
			df = lc0.to_pandas().reset_index()
			df = df[['time', 'flux']]

			ini = min(df['time'])
			fim = max(df['time'])
			x = np.where(np.logical_and(mdumps>ini, mdumps<fim))
			dumps= mdumps[x].tolist()

			st.session_state.ss_tit = tit
			st.session_state.ss_df = df
			st.session_state.ss_dumps = dumps

		with placeholder.container():
			t0, dur = plot()

