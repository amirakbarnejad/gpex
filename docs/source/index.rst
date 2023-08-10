Welcome to GPEX's documentation!
===================================


.. contents:: Table of Contents
   :depth: 2
   :local:


Base Modules
------------------------

.. autoclass:: BaseInfluenceModule
   :members:

.. autoclass:: BaseObjective
   :members:


Influence Modules
------------------------

torch-influence provides three subclasses of :class:`BaseInfluenceModule` out-of-the-box.
Each subclass differs only in how the abstract function :meth:`BaseInfluenceModule.inverse_hvp()`
is implemented. We refer readers to the original influence function
paper_ (Koh & Liang, 2017) for further details.
