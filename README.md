# Transit vetting app

**Transit-vetting** is a small webapp that allows the use of [TESS-Centroid_vetting](https://github.com/exo-pt/TESS-Centroid_vetting) to analyze a single transit in a TESS lightcurve.

The user should provide a TIC and a sector number (any sector observed by TESS for the target).

The corresponding lightcurve is displayed and the user can interact with it (thx to [Plotly](https://github.com/plotly/plotly.py)), zooming/panning/selecting any transit that will be analyzed.<br/>
The transit being selected, **TESS-Centroid-vetting** is called and displayed, showing the **PRF Centroid of the difference image**, and the distance to the nearest stars to it.


You can access the webapp locally in a Python environmente with the command:<br>

>` streamlit run transit-vetting.py `

and the app will open in the default browser.

<!---
This webapp is also currently hosted in Streamlit Cloud. You can access it at<br/>
> https://transit-vetting.streamlit.app
-->
If  more than one author is available for the sector, the displayed lightcurve, in
availability order, is:
&nbsp;&nbsp; SPOC (2 min) -> TESS-SPOC -> QLP -> ELEANOR.

If the sector is only available in FFI data, an FFI cutout is made to get a targetpixelfile that, after background subtraction, is used to generate the lightcurve<br/>

![Image](https://github.com/exo-pt/Transit-vetting-app/blob/main/Transit-vetting-app.png?raw=true)

## Dependencies:
- numpy
- pandas
- lightkurve
- astroquery
- plotly
- tesscentroidvetting >= 1.3.4
- streamlit
