# FFCG: Effective and Fast Family Column Generation for Solving Large-Scale Linear Program

## How to run the code

1. create a virtual environment 
```bash
conda create -n ffcg python=3.10
conda activate ffcg
pip install -r requirements.txt
```
2. edit `DATA_PATH`, `MODEL_PATH`, `RESULT_PATH` in `src/Parameters.py` to the correct path

3. Run the code
```bash
python src/main.py
```

## Acknowledgments
The research is partially supported by National Key R&D
Program of China under Grant 2021ZD0110400, Anhui Provincial Natural Science Foundation under Grant
2208085MF172, Innovation Program for Quantum Science
and Technology 2021ZD0302900 and China National Natural Science Foundation with No. 62132018, 62231015, “Pioneer” and “Leading Goose” R&D Program of Zhejiang,
2023C01029, and 2023C01143. We also thank Sijia Zhang,
Shuli Zeng, and the anonymous reviewers for their comments and helpful feedback. We also thanks to Chi et al. as FFCG is build  based on [RLCG](https://github.com/khalil-research/RLCG) framework.

## How to cite

```
@inproceedings{hu2025ffcg,
  author     = {Yi-Xiang Hu and Feng Wu and Shaoang Li and Yifang Zhao and Xiang-Yang Li},
  title      = {FFCG: Effective and Fast Family Column Generation for Solving Large-Scale Linear Program},
  booktitle  = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, {AAAI} 2025, February 27-March 2, 2025, Philadelphia, USA},
  publisher  = {{AAAI} Press},
  year       = {2025},
}
```