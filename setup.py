try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'qanda',
    'author': 'Sami Zemedkun',
    'url': ' ',
    'download_url': ' ',
    'author_email': 'szemedkun@gmail.com',
    'version': '1.01',
    'install_requires': ['nose'],
    'packages': ['qanda'],
    'scripts': [],
    'name': 'qanda'
}

setup(**config)
