//Create anaconda environment and install dependencies:
conda create --name pygame_app python=3.7
conda activate pygame_app
pip install pygame, numpy

//Run with:
>python AppMP.py


//To run with OpenCL, install PyOpenCL
// Windows:
// ...download the appropriate wheel for your Python version from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl
>pip install <name of wheel>
// Linux/MacOS:
>pip install pyopencl

//Run with:
>python AppCL.py