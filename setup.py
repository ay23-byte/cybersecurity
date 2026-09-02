from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path:str)->List[str]:
    """This function will return the list of requirements"""
    requirement_lst:List[str]=[]
    try:
        with open(file_path) as f:
            ## Read lines from the file
            lines=f.readlines()
            ## Process each line
            for line in lines:
                requirements=line.strip()
                ## ignore empty lines and -e.
                if requirements and  requirements!="-e .":
                    requirement_lst.append(requirements)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Ayush kumar Prajapati",
    author_email="ayush9838502403@gmial.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)  