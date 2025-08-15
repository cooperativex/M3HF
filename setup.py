#!/usr/bin/env python
"""Setup script for gym-macro-overcooked.

This package provides a multi-agent Overcooked environment with macro actions
for reinforcement learning research.
"""

from setuptools import setup, find_packages
import os

# Read long description from README if available
long_description = ""
try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Overcooked gym environment with macro actions for multi-agent reinforcement learning"

setup(
    name='gym_macro_overcooked',
    version='0.1.0',
    description='Multi-agent Overcooked environment with macro actions and language feedback',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Anonymous',
    author_email='anonymous@example.com',
    url='https://github.com/anonymous/gym-macro-overcooked',
    packages=find_packages(),
    package_dir={},
    package_data={
        'gym_macro_overcooked': ['render/graphics/*.png'],
    },
    include_package_data=True,
    python_requires='>=3.7',

    install_requires=[
        'gym>=0.17.0',
        'gymnasium>=0.26.0',  # Support for both gym versions
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'Pillow>=8.1.1',
        'pygame>=1.9.6',
        'tqdm>=4.50.0',
        'ray[rllib]>=2.0.0',  # For multi-agent training
        'openai>=1.0.0',  # For language model integration
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.0.0',
        ],
        'wandb': [
            'wandb>=0.12.0',
        ],
    },

    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='reinforcement learning, multi-agent, overcooked, gym, environment',
)