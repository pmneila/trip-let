# -*- encoding: utf-8 -*-

from setuptools import setup

setup(
    name="trip-let",
    version="0.0.1",
    description="A simple application to build Trip-lets",
    author="Pablo Márquez Neila",
    author_email="pablo.marquez@artorg.unibe.ch",
    # url="https://github.com/pmneila/PyMCubes",
    license="BSD 3-clause",
    long_description="""
    A simple application to build Trip-lets.

    Trip-lets are described in Hofstadter's book "Gödel, Escher, Bach: An
    Eternal Golden Braid".

    A trip-let is a three-dimensional solid that is shaped in such a way that
    its projections along three mutually perpendicular axes are three different
    letters of the alphabet.

    Of course, this application can also be used to build similar shapes such
    as the cork plug.

    https://mathworld.wolfram.com/Trip-Let.html
    https://mathworld.wolfram.com/CorkPlug.html
    """,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Multimedia :: Graphics :: 3D Modeling"
    ],
    packages=["trip_let"],
    requires=['numpy', 'PyMCubes', 'imageio'],
    entry_points={
        'console_scripts': ['trip-let=trip_let.trip_let:main']
    }
)
