crowdcount.transforms
=======================

.. currentmodule:: crowdcount.transforms

Transforms are common image and density map transformations. They can be chained
together using SingleCompose (for single image or density map) ComplexCompose
(for both image and density map),

Compose functions
----------------

.. autoclass:: SingleCompose

.. autoclass:: ComplexCompose

SingleCompose
-------------

.. autoclass:: ResizeShrink
  :members: __call__
  :special-members:

.. autoclass:: LabelEnlarge
  :members: __call__
  :special-members:

ComplexCompose
--------------

.. autoclass:: TransposeFlip
  :members: __call__
  :special-members:

.. autoclass:: RandomCrop
  :members: __call__
  :special-members:

.. autoclass:: Scale
  :members: __call__
  :special-members:
