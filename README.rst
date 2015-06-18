Image Stitching
========================================

OpenCV 2.4 sample
------------------------------

.. code-block:: sh

    $ make opencv_sample_2_4
    $ bin/stitching_2_4
    Rotation model images stitcher.

    stitching img1 img2 [...imgN]

    Flags:
      --try_use_gpu (yes|no)
          Try to use GPU. The default value is 'no'. All default values
          are for CPU mode.
      --output <result_img>
          The default is 'result.jpg'.
    $ bin/stitching_2_4 images/A001.jpg images/A002.jpg images/A003.jpg

Resource
------------------------------

* `Wikipedia - Image stitching <http://en.wikipedia.org/wiki/Image_stitching>`_
* `OpenCV - Stitching Pipeline <http://docs.opencv.org/modules/stitching/doc/introduction.html>`_
* `OpenCV Tutorials <http://docs.opencv.org/doc/tutorials/tutorials.html>`_
* `OpenCV-Python Tutorials <http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_tutorials.html>`_
