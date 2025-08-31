from setuptools import find_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:

    hypen_e_dot = "-e ."
    req = []
    with open(file_path) as file:
        req = file.readlines()
        req = [r.replace("\n","")for r in req]

        if hypen_e_dot in req:
            req.remove(hypen_e_dot)

    return req        


setup(

    name="ML_project",
    version="0.1.0",
    author="ABHIMANYU",
    author_email="abhimanyustomar24@gmail.com",
    description="A machine learning project.",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")


)