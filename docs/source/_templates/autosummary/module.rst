{{ fullname | escape | underline }}

.. rubric:: Description

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
    :toctree: .
    :nosignatures:
    {% for class in classes %}
    {{ class.split('.')[-1] }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: .
    :nosignatures:
    {% for function in functions %}
    {{ function.split('.')[-1] }}
    {% endfor %}

{% endif %}
