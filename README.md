# Transit vetting app

Transit vetting app is a small webapp that allows the use of TESS-Centroid_vetting to analyze one single transit in a TESS lightcurve<br/>
The user should provide a TIC number and a sector, any sector observed by TESS for the target.
The corresponding lightcurve is displayed and the user can interact with it (thx to [Plotly](https://github.com/plotly/plotly.py), zooming/panning/selecting any transit that will be analyzed.<br/>
The transit being selected, TESS-Centroid-vetting is called and displayed, showing the PRF Centroid of the difference image, and the distance of the nearest stars to it.

This webapp is currently hosted in Streamlit Cloud. You can access it at<br/>
https://transit-vetting.streamlit.app
<br/>
or start the webapp locally in a Python environmente with:
` streamlit run transit-vetting.py `
<br/>
The app will open in the default browser

If  more than one author is available for the sector, the displayed lightcurve, in availability order, is:
&nbsp;&nbsp; SPOC (2 min) -> TESS-SPOC -> QLP -> ELEANOR.<br/>
If the sector is only available in FFI data, tesscut is used to get a targetpixelfile that, after background subtraction, is used to generate the lightcurve<br/>

![Image](https://github.com/exo-pt/Tess-Lightcurves-app/blob/main/Tess-Lightcurves-app.png?raw=true)

## Dependencies:
- numpy
- pandas
- lightkurve
- astroquery
- plotly
- tesscentroidvetting >= 1.3.4
- streamlit
