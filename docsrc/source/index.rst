GOBRec: GPU Optimized Bandits Recommender
=========================================

GOBRec is a Python library with an optimized implementation of contextual multi-armed bandits (CMABs) for recommender systems. The library has a simple API that allows you to use the CMAB algorithms to generate item (arms) expectations, using it for tasks other than recommendations. You can also use any of the implemented CMABs inside the Recommender to efficiently generate top-K recommendations.

The main contribution of GOBRec is its efficient implementation. With the vectorized code, using only CPU, our implementation was up to X times faster than other libraries. Using GPU optimization, our library achieved a speed gain of Y times. More details about these comparisons can be found in the `benchmark page <benchmark.html>`_ or in our `paper <>`_.

The GOBRec source code is available at: `https://github.com/RecSys-UFSCar/gobrec <https://github.com/RecSys-UFSCar/gobrec>`_

Documentation pages
===================

.. toctree::
   :maxdepth: 1

   installation
   quick start
   benchmark
   public API
   contributing
   resources
