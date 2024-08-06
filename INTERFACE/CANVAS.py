import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

canvas_result = st_canvas(stroke_width=15,
						  stroke_color='rgb(255, 255, 255)',
						  background_color='rgb(0, 0, 0)',
						  height=150,
						  width=150,
						  key="canvas")

if canvas_result.image_data is not None:
  img = canvas_result.image_data
  st.write(img.shape)