from setuptools import setup, find_packages, setup

setup(
    name='mlproject',
    version='0.0.1',
    author='Siwakhile Masilela',
    author_email='wakhiwakhi1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)

