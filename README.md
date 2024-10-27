# Can Language Models Reason about Individualistic Human Values and Preferences?

**Authors:**
[Liwei Jiang](https://liweijiang.me),
[Taylor Sorensen](https://tsor13.github.io),
[Sydney Levine](https://sites.google.com/site/sydneymlevine/)
[Yejin Choi](https://homes.cs.washington.edu/~yejin/),


This is the codebase for the [paper](https://arxiv.org/abs/2410.03868).

## Data

We transformed the [World Values Survey](https://www.worldvaluessurvey.org/wvs.jsp) into natural languagestatements that describes people's values and preferences.



### Meta Data

- `data/demographics_meta_data.csv` contains the meta data of the demographics questions.

- `data/refined_statements_meta_data.csv` contains the meta data of the *refined* statements.

- `data/statements_meta_data.csv` contains the meta data of the *polarity-grouped* statements.

### Human Label Data

Please find the converted human label data [here](https://drive.google.com/drive/folders/1ebLGZj7YDzSRFViCTYKpoPNYU0CkiRt0?usp=share_link).

- `demographics_in_nl_refined_statements_combined_full_set.csv` contains converted human label data with *refined* statements.

- `demographics_in_nl_statements_combined_full_set.csv` contains converted human label data with *polarity-grouped* statements.


## Citation

If you find our work helpful, please feel free to cite our work!
```
@misc{jiang2024indievalue,
      title={Can Language Models Reason about Individualistic Human Values and Preferences?}, 
      author={Liwei Jiang and Taylor Sorensen and Sydney Levine and Yejin Choi},
      year={2024},
      eprint={2410.03868},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.03868}, 
}
```

