import os
from typing import Generator
from doit.tools import create_folder


def get_file_refs(file_path: str) -> Generator[str, None, None]:
    files = [f for f in os.listdir(".") if os.path.isfile(f)]
    with open(file_path, "r") as f:
        file_content = f.read()
        for file in files:
            ext = file.split(".")[-1]
            file_no_ext = file.replace("."+ext, "")
            if (
                file in file_content
                or "{" + file_no_ext in file_content
                or f"import {file_no_ext}" in file_content
                or f"from {file_no_ext}" in file_content
            ):
                yield file


def task_temp_folder():
    return {'actions': [(create_folder, ['temp'])], 'targets': ['temp']}


def task_compile_pdf():
    report_name = "autoreproducelitreview"
    return {
        "actions": [
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-output-directory=temp",
                f"{report_name}.tex",
            ]
        ],
        "file_dep": [f"{report_name}.tex"] + list(get_file_refs(f"{report_name}.tex")),
        "targets": [f"temp/{report_name}.pdf"],
    }


def task_check_python_dependencies():
    files = [f for f in os.listdir(".") if os.path.isfile(f)]
    for file in files:
        if file.split('.')[-1] != 'py':
            continue
        dependencies = list(get_file_refs(file))
        if len(dependencies):
            yield {'name': file, 'actions': [], 'file_dep': dependencies, 'targets': [file]}