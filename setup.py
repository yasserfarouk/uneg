#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'scikit-learn', 'numpy', 'pandas', 'scipy', 'pytest-runner', 'negmas']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Yasser Mohammad",
    author_email='yasserfarouk@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Defaults to README",
    entry_points={
        'console_scripts': [
            'uneg=uneg.scripts.app:cli',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='uneg',
    name='uneg',
    packages=find_packages(include=['uneg']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yasserfarouk/uneg',
    version='0.1.0',
    zip_safe=False,
)
