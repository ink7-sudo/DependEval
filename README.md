# <h1><img src="https://github.com/ink7-sudo/DependEval/blob/finalversion/assets/logo.png?raw=true" width="32" style="vertical-align:middle; margin-right:8px;">DependEval: Benchmarking LLMs for Repository Dependency Understanding</h1>

The repository contains the data and evaluation code for ACL 2025 Findings paper ["DependEval: Benchmarking LLMs for Repository Dependency Understanding"](https://arxiv.org/pdf/2503.06689)

## Introduction

We introduce **DependEval**, a hierarchical benchmark for evaluating LLMs on repository-level code understanding across 8 programming languages. 

DependEval comprises 2,683 curated repositories across 8 programming languages, and evaluates models on three hierarchical tasks: Dependency Recognition, Repository Construction, and Multi-file Editing. These tasks simulate core challenges faced in large-scale software development, such as resolving inter-file dependencies, generating structured project layouts, and maintaining cross-file consistency during edits.


<img width="1432" alt="abs" src="https://github.com/ink7-sudo/DependEval/blob/finalversion/assets/taskcase.png?raw=true">


DependEval provides fine-grained metrics and analysis tools to advance research on LLM-based software engineering. Our findings highlight key challenges in applying LLMs to large-scale development, and lay the groundwork for future improvements in repository-level understanding.

<img width="1432" alt="abs" src="https://github.com/ink7-sudo/DependEval/blob/finalversion/assets/radar.png?raw=true">


