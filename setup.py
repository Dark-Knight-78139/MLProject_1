from setuptools import setup, find_packages
from typing import List

HYPEN = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
    
    # Remove any leading/trailing whitespace and empty lines
    requirements = [req.strip() for req in requirements if req.strip()]
    if HYPEN in requirements:
        requirements.remove(HYPEN)
    
    return requirements

setup(
    name = 'mlp_1',
    version = '0.1.0',
    author = 'Akhil Nekkanti',
    author_email = 'sujayakhil@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt'),
)