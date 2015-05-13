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
